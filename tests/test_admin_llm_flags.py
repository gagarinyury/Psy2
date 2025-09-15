import pytest

from app.core.settings import settings


@pytest.mark.anyio
async def test_admin_llm_flags_get_current(client):
    """
    Тест получения текущих значений флагов LLM.
    """
    # Запрос текущих значений без изменений
    response = await client.post("/admin/llm_flags", json={})

    assert response.status_code == 200

    result = response.json()
    assert "use_reason" in result
    assert "use_gen" in result
    assert isinstance(result["use_reason"], bool)
    assert isinstance(result["use_gen"], bool)


@pytest.mark.anyio
async def test_admin_llm_flags_set_reason_only(client):
    """
    Тест установки только reasoning флага.
    """
    # Запоминаем исходные значения
    original_reason = settings.USE_DEEPSEEK_REASON
    original_gen = settings.USE_DEEPSEEK_GEN

    try:
        # Устанавливаем только reasoning флаг
        response = await client.post("/admin/llm_flags", json={"use_reason": True})

        assert response.status_code == 200

        result = response.json()
        assert result["use_reason"] is True
        assert result["use_gen"] == original_gen  # Не изменился

        # Проверяем что настройки действительно изменились
        assert settings.USE_DEEPSEEK_REASON is True
        assert settings.USE_DEEPSEEK_GEN == original_gen

    finally:
        # Восстанавливаем исходные значения
        settings.USE_DEEPSEEK_REASON = original_reason
        settings.USE_DEEPSEEK_GEN = original_gen


@pytest.mark.anyio
async def test_admin_llm_flags_set_generation_only(client):
    """
    Тест установки только generation флага.
    """
    # Запоминаем исходные значения
    original_reason = settings.USE_DEEPSEEK_REASON
    original_gen = settings.USE_DEEPSEEK_GEN

    try:
        # Устанавливаем только generation флаг
        response = await client.post("/admin/llm_flags", json={"use_gen": True})

        assert response.status_code == 200

        result = response.json()
        assert result["use_reason"] == original_reason  # Не изменился
        assert result["use_gen"] is True

        # Проверяем что настройки действительно изменились
        assert settings.USE_DEEPSEEK_REASON == original_reason
        assert settings.USE_DEEPSEEK_GEN is True

    finally:
        # Восстанавливаем исходные значения
        settings.USE_DEEPSEEK_REASON = original_reason
        settings.USE_DEEPSEEK_GEN = original_gen


@pytest.mark.anyio
async def test_admin_llm_flags_set_both(client):
    """
    Тест установки обоих флагов.
    """
    # Запоминаем исходные значения
    original_reason = settings.USE_DEEPSEEK_REASON
    original_gen = settings.USE_DEEPSEEK_GEN

    try:
        # Устанавливаем оба флага
        response = await client.post(
            "/admin/llm_flags", json={"use_reason": True, "use_gen": False}
        )

        assert response.status_code == 200

        result = response.json()
        assert result["use_reason"] is True
        assert result["use_gen"] is False

        # Проверяем что настройки действительно изменились
        assert settings.USE_DEEPSEEK_REASON is True
        assert settings.USE_DEEPSEEK_GEN is False

    finally:
        # Восстанавливаем исходные значения
        settings.USE_DEEPSEEK_REASON = original_reason
        settings.USE_DEEPSEEK_GEN = original_gen


@pytest.mark.anyio
async def test_admin_llm_flags_disable_both(client):
    """
    Тест отключения обоих флагов.
    """
    # Запоминаем исходные значения
    original_reason = settings.USE_DEEPSEEK_REASON
    original_gen = settings.USE_DEEPSEEK_GEN

    try:
        # Сначала включим флаги
        settings.USE_DEEPSEEK_REASON = True
        settings.USE_DEEPSEEK_GEN = True

        # Затем отключим их через API
        response = await client.post(
            "/admin/llm_flags", json={"use_reason": False, "use_gen": False}
        )

        assert response.status_code == 200

        result = response.json()
        assert result["use_reason"] is False
        assert result["use_gen"] is False

        # Проверяем что настройки действительно изменились
        assert settings.USE_DEEPSEEK_REASON is False
        assert settings.USE_DEEPSEEK_GEN is False

    finally:
        # Восстанавливаем исходные значения
        settings.USE_DEEPSEEK_REASON = original_reason
        settings.USE_DEEPSEEK_GEN = original_gen


@pytest.mark.anyio
async def test_admin_llm_flags_none_values(client):
    """
    Тест что None значения не изменяют настройки.
    """
    # Запоминаем исходные значения
    original_reason = settings.USE_DEEPSEEK_REASON
    original_gen = settings.USE_DEEPSEEK_GEN

    try:
        # Устанавливаем известные значения
        settings.USE_DEEPSEEK_REASON = True
        settings.USE_DEEPSEEK_GEN = False

        # Передаем None значения - они не должны изменить настройки
        response = await client.post("/admin/llm_flags", json={"use_reason": None, "use_gen": None})

        assert response.status_code == 200

        result = response.json()
        assert result["use_reason"] is True  # Не изменился
        assert result["use_gen"] is False  # Не изменился

        # Проверяем что настройки действительно не изменились
        assert settings.USE_DEEPSEEK_REASON is True
        assert settings.USE_DEEPSEEK_GEN is False

    finally:
        # Восстанавливаем исходные значения
        settings.USE_DEEPSEEK_REASON = original_reason
        settings.USE_DEEPSEEK_GEN = original_gen


@pytest.mark.anyio
async def test_admin_llm_flags_invalid_json(client):
    """
    Тест обработки неверного JSON.
    """
    response = await client.post(
        "/admin/llm_flags",
        data="invalid json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422  # Validation error


@pytest.mark.anyio
async def test_admin_llm_flags_partial_update(client):
    """
    Тест частичного обновления - только один флаг из двух.
    """
    # Запоминаем исходные значения
    original_reason = settings.USE_DEEPSEEK_REASON
    original_gen = settings.USE_DEEPSEEK_GEN

    try:
        # Устанавливаем начальные значения
        settings.USE_DEEPSEEK_REASON = False
        settings.USE_DEEPSEEK_GEN = False

        # Изменяем только один флаг
        response = await client.post(
            "/admin/llm_flags",
            json={
                "use_reason": True
                # use_gen не передаем
            },
        )

        assert response.status_code == 200

        result = response.json()
        assert result["use_reason"] is True  # Изменился
        assert result["use_gen"] is False  # Остался прежним

        # Проверяем настройки
        assert settings.USE_DEEPSEEK_REASON is True
        assert settings.USE_DEEPSEEK_GEN is False

    finally:
        # Восстанавливаем исходные значения
        settings.USE_DEEPSEEK_REASON = original_reason
        settings.USE_DEEPSEEK_GEN = original_gen


@pytest.mark.anyio
async def test_admin_llm_flags_response_structure(client):
    """
    Тест структуры ответа API.
    """
    response = await client.post("/admin/llm_flags", json={"use_reason": True, "use_gen": True})

    assert response.status_code == 200

    result = response.json()

    # Проверяем структуру ответа
    assert isinstance(result, dict)
    assert len(result) == 2
    assert "use_reason" in result
    assert "use_gen" in result
    assert isinstance(result["use_reason"], bool)
    assert isinstance(result["use_gen"], bool)
