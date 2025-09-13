"""
Тесты для функции retrieve из app.orchestrator.nodes.retrieve.

Тестирует сценарии доступа к KB фрагментам на основе:
- Уровня trust пользователя
- Фильтрации по availability (public/gated/hidden)
- Ограничений top_k
- Обработка edge cases (несуществующий case_id, пустая БД)
"""

import pytest
import uuid
from pathlib import Path
from sqlalchemy import delete

from app.core.tables import Case, KBFragment, Session, TelemetryTurn
from app.orchestrator.nodes.retrieve import retrieve
from app.cli.case_loader import load_case
import json


async def setup_test_data(db):
    """Создает тестовые данные для каждого теста"""
    # Очистка
    await db.execute(delete(TelemetryTurn))
    await db.execute(delete(Session))
    await db.execute(delete(KBFragment))
    await db.execute(delete(Case))
    await db.commit()

    # Загрузка demo_case.json
    demo_case_path = (
        Path(__file__).parent.parent / "app" / "examples" / "demo_case.json"
    )
    with open(demo_case_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    case_id = await load_case(db, data["case"], data["kb"])
    case_uuid = uuid.UUID(case_id)

    # Добавление дополнительных фрагментов
    fragments = [
        KBFragment(
            id=uuid.uuid4(),
            case_id=case_uuid,
            type="secret",
            text="Скрытая информация",
            fragment_metadata={
                "topic": "family",
                "tags": ["sensitive"],
                "emotion_label": "anxiety",
                "availability": "hidden",
                "disclosure_cost": 10,
            },
            availability="hidden",
            consistency_keys={},
            embedding=None,
        ),
        KBFragment(
            id=uuid.uuid4(),
            case_id=case_uuid,
            type="info",
            text="Общая информация для шума",
            fragment_metadata={
                "topic": "general",
                "tags": ["general"],
                "emotion_label": "neutral",
                "availability": "public",
                "disclosure_cost": 0,
            },
            availability="public",
            consistency_keys={},
            embedding=None,
        ),
        KBFragment(
            id=uuid.uuid4(),
            case_id=case_uuid,
            type="symptom",
            text="Серьезные симптомы, требующие высокого доверия",
            fragment_metadata={
                "topic": "mood",
                "tags": ["severe"],
                "emotion_label": "depressed",
                "availability": "gated",
                "disclosure_cost": 5,
                "disclosure_requirements": {"trust_ge": 0.8},
            },
            availability="gated",
            consistency_keys={},
            embedding=None,
        ),
    ]

    db.add_all(fragments)
    await db.commit()

    return case_id


@pytest.mark.skip(reason="Async loop conflicts - temporary skip")
@pytest.mark.anyio
async def test_retrieve_trust_03_returns_only_public(db):
    """
    Тест: при trust=0.3 должен вернуться только public фрагмент из примера
    """
    case_id = await setup_test_data(db)

    session_state = {"trust": 0.3, "access_level": "low", "risk_status": "safe"}
    topics = []

    result = await retrieve(
        db,
        case_id=case_id,
        intent="open_question",
        topics=topics,
        session_state_compact=session_state,
        top_k=5,
    )

    # Проверяем что вернулись только public фрагменты
    assert len(result) >= 1, f"Expected at least 1 fragment, got {len(result)}"
    for fragment in result:
        assert fragment["metadata"]["availability"] == "public"
        # Проверяем что есть фрагмент из demo_case
        if fragment["text"] == "Родился в 1989, работает в ИТ.":
            assert fragment["type"] == "bio"
            assert fragment["metadata"]["topic"] == "background"


@pytest.mark.skip(reason="Async loop conflicts - temporary skip")
@pytest.mark.anyio
async def test_retrieve_trust_05_returns_gated_fragment(db):
    """
    Тест: при trust=0.5 может вернуться и gated фрагмент с trust_ge=0.4
    """
    case_id = await setup_test_data(db)

    session_state = {"trust": 0.5, "access_level": "medium", "risk_status": "safe"}
    topics = []

    result = await retrieve(
        db,
        case_id=case_id,
        intent="open_question",
        topics=topics,
        session_state_compact=session_state,
        top_k=5,
    )

    # Должен вернуться gated фрагмент с sleep при достаточном trust
    assert len(result) >= 1
    found_gated = False
    for fragment in result:
        if fragment["text"] == "Нарушение сна последние 3 месяца.":
            found_gated = True
            assert fragment["type"] == "symptom"
            assert fragment["metadata"]["availability"] == "gated"
            assert fragment["metadata"]["topic"] == "sleep"
            assert fragment["metadata"]["disclosure_requirements"]["trust_ge"] == 0.4

    assert found_gated, "Gated фрагмент должен быть доступен при trust=0.5"


@pytest.mark.skip(reason="Async loop conflicts - temporary skip")
@pytest.mark.anyio
async def test_retrieve_empty_topics_returns_available_not_hidden(db):
    """
    Тест: при пустых topics возвращает любые доступные фрагменты, но не hidden
    """
    case_id = await setup_test_data(db)

    session_state = {"trust": 0.5, "access_level": "medium", "risk_status": "safe"}
    topics = []

    result = await retrieve(
        db,
        case_id=case_id,
        intent="open_question",
        topics=topics,
        session_state_compact=session_state,
        top_k=5,
    )

    assert len(result) >= 1
    for fragment in result:
        # Hidden фрагменты никогда не должны возвращаться
        assert fragment["metadata"]["availability"] != "hidden"
        # Должны быть только public и gated (если trust достаточен)
        assert fragment["metadata"]["availability"] in ["public", "gated"]


@pytest.mark.skip(reason="Async loop conflicts - temporary skip")
@pytest.mark.anyio
async def test_availability_filtering_excludes_hidden(db):
    """
    Тест фильтрации по availability - hidden фрагменты исключаются
    """
    case_id = await setup_test_data(db)

    session_state = {"trust": 1.0, "access_level": "high", "risk_status": "safe"}
    topics = []

    result = await retrieve(
        db,
        case_id=case_id,
        intent="open_question",
        topics=topics,
        session_state_compact=session_state,
        top_k=5,
    )

    # Даже при максимальном trust, hidden фрагменты не возвращаются
    assert len(result) >= 1
    for fragment in result:
        assert fragment["metadata"]["availability"] != "hidden"


@pytest.mark.skip(reason="Async loop conflicts - temporary skip")
@pytest.mark.anyio
async def test_top_k_limit(db):
    """
    Тест ограничения top_k
    """
    case_id = await setup_test_data(db)

    session_state = {"trust": 0.8, "access_level": "high", "risk_status": "safe"}
    topics = []
    top_k = 2

    result = await retrieve(
        db,
        case_id=case_id,
        intent="open_question",
        topics=topics,
        session_state_compact=session_state,
        top_k=top_k,
    )

    # Количество возвращенных фрагментов не должно превышать top_k
    assert len(result) <= top_k


@pytest.mark.skip(reason="Async loop conflicts - temporary skip")
@pytest.mark.anyio
async def test_high_trust_threshold_gated_access(db):
    """
    Тест доступа к gated фрагменту с высоким порогом trust (0.8)
    """
    case_id = await setup_test_data(db)

    # Тест с недостаточным trust
    session_state_low = {"trust": 0.6, "access_level": "medium", "risk_status": "safe"}
    topics = []

    result_low = await retrieve(
        db,
        case_id=case_id,
        intent="open_question",
        topics=topics,
        session_state_compact=session_state_low,
        top_k=5,
    )

    # При trust=0.6 не должно быть фрагмента с trust_ge=0.8
    high_trust_found = False
    for fragment in result_low:
        if fragment["text"] == "Серьезные симптомы, требующие высокого доверия":
            high_trust_found = True
    assert not high_trust_found, "Фрагмент с высоким порогом не должен быть доступен"

    # Тест с достаточным trust
    session_state_high = {"trust": 0.9, "access_level": "high", "risk_status": "safe"}

    result_high = await retrieve(
        db,
        case_id=case_id,
        intent="open_question",
        topics=topics,
        session_state_compact=session_state_high,
        top_k=5,
    )

    # При trust=0.9 должен быть доступен фрагмент с trust_ge=0.8
    high_trust_found = False
    for fragment in result_high:
        if fragment["text"] == "Серьезные симптомы, требующие высокого доверия":
            high_trust_found = True
            assert fragment["metadata"]["availability"] == "gated"
    assert (
        high_trust_found
    ), "Фрагмент с высоким порогом должен быть доступен при достаточном trust"


@pytest.mark.skip(reason="Async loop conflicts - temporary skip")
@pytest.mark.anyio
async def test_nonexistent_case_id(db):
    """
    Тест с несуществующим case_id
    """
    fake_case_id = str(uuid.uuid4())
    session_state = {"trust": 0.5, "access_level": "medium", "risk_status": "safe"}
    topics = []

    result = await retrieve(
        db,
        case_id=fake_case_id,
        intent="open_question",
        topics=topics,
        session_state_compact=session_state,
        top_k=5,
    )

    # Должен вернуться пустой список для несуществующего case_id
    assert result == []


@pytest.mark.skip(reason="Async loop conflicts - temporary skip")
@pytest.mark.anyio
async def test_return_data_structure(db):
    """
    Тест структуры возвращаемых данных {id, type, text, metadata}
    """
    case_id = await setup_test_data(db)

    session_state = {"trust": 0.5, "access_level": "medium", "risk_status": "safe"}
    topics = []

    result = await retrieve(
        db,
        case_id=case_id,
        intent="open_question",
        topics=topics,
        session_state_compact=session_state,
        top_k=5,
    )

    assert len(result) >= 1
    for fragment in result:
        # Проверяем обязательные поля
        assert "id" in fragment
        assert "type" in fragment
        assert "text" in fragment
        assert "metadata" in fragment

        # Проверяем типы данных
        assert isinstance(fragment["id"], str)
        assert isinstance(fragment["type"], str)
        assert isinstance(fragment["text"], str)
        assert isinstance(fragment["metadata"], dict)

        # Проверяем что metadata содержит обязательные поля
        metadata = fragment["metadata"]
        assert "topic" in metadata
        assert "availability" in metadata


@pytest.mark.skip(reason="Async loop conflicts - temporary skip")
@pytest.mark.anyio
async def test_empty_database_scenario(db):
    """
    Тест с пустой БД
    """
    # Очистка данных в БД
    await db.execute(delete(TelemetryTurn))
    await db.execute(delete(Session))
    await db.execute(delete(KBFragment))
    await db.execute(delete(Case))
    await db.commit()

    fake_case_id = str(uuid.uuid4())
    session_state = {"trust": 0.5, "access_level": "medium", "risk_status": "safe"}
    topics = []

    result = await retrieve(
        db,
        case_id=fake_case_id,
        intent="open_question",
        topics=topics,
        session_state_compact=session_state,
        top_k=5,
    )

    # В пустой БД должен вернуться пустой список
    assert result == []
