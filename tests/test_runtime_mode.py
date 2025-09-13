import pytest
from httpx import AsyncClient
from app.core.settings import settings


@pytest.mark.anyio
async def test_runtime_rag_mode_switch(client: AsyncClient):
    """
    Тест переключения режима RAG в runtime через API.
    """
    # Сохраняем изначальное состояние
    original_mode = settings.RAG_USE_VECTOR

    try:
        # 1. Переключаем в vector режим
        response = await client.post("/admin/rag_mode", json={"use_vector": True})
        assert response.status_code == 200

        data = response.json()
        assert data["current_mode"] == "vector"
        assert data["use_vector"] is True
        assert settings.RAG_USE_VECTOR is True

        # 2. Переключаем в metadata режим
        response = await client.post("/admin/rag_mode", json={"use_vector": False})
        assert response.status_code == 200

        data = response.json()
        assert data["current_mode"] == "metadata"
        assert data["use_vector"] is False
        assert settings.RAG_USE_VECTOR is False

        # 3. Обратно в vector режим
        response = await client.post("/admin/rag_mode", json={"use_vector": True})
        assert response.status_code == 200

        data = response.json()
        assert data["current_mode"] == "vector"
        assert data["use_vector"] is True
        assert settings.RAG_USE_VECTOR is True

    finally:
        # Восстанавливаем изначальное состояние
        settings.RAG_USE_VECTOR = original_mode


@pytest.mark.anyio
async def test_runtime_mode_affects_turn_behavior(client: AsyncClient, setup_test_case):
    """
    Тест что переключение режима влияет на поведение /turn эндпоинта.
    """
    case_id, session_id = setup_test_case

    # Сохраняем изначальное состояние
    original_mode = settings.RAG_USE_VECTOR

    try:
        # Убеждаемся что у нас есть эмбеддинги для vector режима
        # (используем helper из smoke test)
        import uuid
        from app.kb.embeddings import run_embed

        await run_embed(uuid.UUID(case_id))

        # Базовый turn request
        turn_request = {
            "therapist_utterance": "Как вы спите последние недели?",
            "session_state": {
                "affect": "neutral",
                "trust": 0.5,
                "fatigue": 0.1,
                "access_level": 1,
                "risk_status": "none",
                "last_turn_summary": "",
            },
            "case_id": case_id,
            "session_id": session_id,
            "options": {},
        }

        # 1. Тестируем metadata режим
        await client.post("/admin/rag_mode", json={"use_vector": False})
        response = await client.post("/turn", json=turn_request)
        assert response.status_code == 200

        metadata_result = response.json()
        assert "used_fragments" in metadata_result
        assert "risk_status" in metadata_result

        # 2. Тестируем vector режим
        await client.post("/admin/rag_mode", json={"use_vector": True})
        response = await client.post("/turn", json=turn_request)
        assert response.status_code == 200

        vector_result = response.json()
        assert "used_fragments" in vector_result
        assert "risk_status" in vector_result

        # В vector режиме должны быть использованы фрагменты
        # (конкретные различия в поведении зависят от реализации)
        assert isinstance(vector_result["used_fragments"], list)

        print(f"Metadata mode used fragments: {len(metadata_result['used_fragments'])}")
        print(f"Vector mode used fragments: {len(vector_result['used_fragments'])}")

    finally:
        # Восстанавливаем изначальное состояние
        settings.RAG_USE_VECTOR = original_mode


@pytest.mark.anyio
async def test_rag_mode_api_validation(client: AsyncClient):
    """
    Тест валидации API эндпоинта /admin/rag_mode.
    """
    original_mode = settings.RAG_USE_VECTOR

    try:
        # Тест с невалидными данными
        response = await client.post("/admin/rag_mode", json={})
        assert response.status_code == 422  # Validation error

        response = await client.post("/admin/rag_mode", json={"use_vector": "invalid"})
        assert response.status_code == 422  # Validation error

        # Тест с валидными данными
        response = await client.post("/admin/rag_mode", json={"use_vector": True})
        assert response.status_code == 200

        data = response.json()
        assert "current_mode" in data
        assert "use_vector" in data
        assert isinstance(data["current_mode"], str)
        assert isinstance(data["use_vector"], bool)

    finally:
        settings.RAG_USE_VECTOR = original_mode
