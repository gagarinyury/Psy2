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
from sqlalchemy import select, delete

from app.core.tables import Case, KBFragment, Session, TelemetryTurn
from app.core.db import AsyncSessionLocal
from app.orchestrator.nodes.retrieve import retrieve
from app.cli.case_loader import load_case_from_file


async def setup_test_data(session):
    """Создает тестовые данные для каждого теста"""
    # Очистка
    await session.execute(delete(TelemetryTurn))
    await session.execute(delete(Session))  
    await session.execute(delete(KBFragment))
    await session.execute(delete(Case))
    await session.commit()
    
    # Загрузка demo_case.json
    demo_case_path = str(Path(__file__).parent.parent / "app" / "examples" / "demo_case.json")
    await load_case_from_file(demo_case_path)
    
    # Получение case_id
    result = await session.execute(select(Case).order_by(Case.created_at.desc()).limit(1))
    case = result.scalars().first()
    case_id = str(case.id) if case else None
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
                    "disclosure_cost": 10
                },
                availability="hidden",
                consistency_keys={},
                embedding=None
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
                    "disclosure_cost": 0
                },
                availability="public",
                consistency_keys={},
                embedding=None
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
                    "disclosure_requirements": {"trust_ge": 0.8}
                },
                availability="gated",
                consistency_keys={},
                embedding=None
            )
    ]
    
    session.add_all(fragments)
    await session.commit()
    
    return case_id


async def cleanup_test_data():
    """Очищает тестовые данные"""
    async with AsyncSessionLocal() as session:
        await session.execute(delete(TelemetryTurn))
        await session.execute(delete(Session))  
        await session.execute(delete(KBFragment))
        await session.execute(delete(Case))
        await session.commit()


@pytest.mark.asyncio
async def test_retrieve_trust_03_returns_only_public(session):
    """
    Тест: при trust=0.3 должен вернуться только public фрагмент из примера
    """
    case_id = await setup_test_data(session)
    
    try:
        session_state = {"trust": 0.3, "access_level": "low", "risk_status": "safe"}
        topics = []
        
        result = await retrieve(
            case_id=case_id,
            intent="open_question",
            topics=topics,
            session_state_compact=session_state,
            top_k=5
        )
        
        # Проверяем что вернулись только public фрагменты
        assert len(result) >= 1, f"Expected at least 1 fragment, got {len(result)}"
        for fragment in result:
            assert fragment["metadata"]["availability"] == "public"
            # Проверяем что есть фрагмент из demo_case
            if fragment["text"] == "Родился в 1989, работает в ИТ.":
                assert fragment["type"] == "bio"
                assert fragment["metadata"]["topic"] == "background"
    finally:
        await cleanup_test_data()


@pytest.mark.asyncio 
async def test_retrieve_trust_05_returns_gated_fragment(session):
    """
    Тест: при trust=0.5 может вернуться и gated фрагмент с trust_ge=0.4
    """
    case_id = await setup_test_data(session)
    
    try:
        session_state = {"trust": 0.5, "access_level": "medium", "risk_status": "safe"}
        topics = []
        
        result = await retrieve(
            case_id=case_id,
            intent="open_question", 
            topics=topics,
            session_state_compact=session_state,
            top_k=5
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
    finally:
        await cleanup_test_data()


@pytest.mark.asyncio
async def test_retrieve_empty_topics_returns_available_not_hidden(session):
    """
    Тест: при пустых topics возвращает любые доступные фрагменты, но не hidden
    """
    case_id = await setup_test_data(session)
    
    try:
        session_state = {"trust": 0.5, "access_level": "medium", "risk_status": "safe"}
        topics = []
        
        result = await retrieve(
            case_id=case_id,
            intent="open_question",
            topics=topics,
            session_state_compact=session_state,
            top_k=5
        )
        
        assert len(result) >= 1
        for fragment in result:
            # Hidden фрагменты никогда не должны возвращаться
            assert fragment["metadata"]["availability"] != "hidden"
            # Должны быть только public и gated (если trust достаточен)
            assert fragment["metadata"]["availability"] in ["public", "gated"]
    finally:
        await cleanup_test_data()


@pytest.mark.asyncio
async def test_availability_filtering_excludes_hidden(session):
    """
    Тест фильтрации по availability - hidden фрагменты исключаются
    """
    case_id = await setup_test_data(session)
    
    try:
        session_state = {"trust": 1.0, "access_level": "high", "risk_status": "safe"}
        topics = []
        
        result = await retrieve(
            case_id=case_id,
            intent="open_question",
            topics=topics,
            session_state_compact=session_state,
            top_k=5
        )
        
        # Даже при максимальном trust, hidden фрагменты не возвращаются
        assert len(result) >= 1
        for fragment in result:
            assert fragment["metadata"]["availability"] != "hidden"
    finally:
        await cleanup_test_data()


@pytest.mark.asyncio
async def test_top_k_limit(session):
    """
    Тест ограничения top_k
    """
    case_id = await setup_test_data(session)
    
    try:
        session_state = {"trust": 0.8, "access_level": "high", "risk_status": "safe"}
        topics = []
        top_k = 2
        
        result = await retrieve(
            case_id=case_id,
            intent="open_question",
            topics=topics,
            session_state_compact=session_state,
            top_k=top_k
        )
        
        # Количество возвращенных фрагментов не должно превышать top_k
        assert len(result) <= top_k
    finally:
        await cleanup_test_data()


@pytest.mark.asyncio
async def test_high_trust_threshold_gated_access(session):
    """
    Тест доступа к gated фрагменту с высоким порогом trust (0.8)
    """
    case_id = await setup_test_data(session)
    
    try:
        # Тест с недостаточным trust
        session_state_low = {"trust": 0.6, "access_level": "medium", "risk_status": "safe"}
        topics = []
        
        result_low = await retrieve(
            case_id=case_id,
            intent="open_question",
            topics=topics,
            session_state_compact=session_state_low,
            top_k=5
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
            case_id=case_id,
            intent="open_question",
            topics=topics,
            session_state_compact=session_state_high,
            top_k=5
        )
        
        # При trust=0.9 должен быть доступен фрагмент с trust_ge=0.8
        high_trust_found = False
        for fragment in result_high:
            if fragment["text"] == "Серьезные симптомы, требующие высокого доверия":
                high_trust_found = True
                assert fragment["metadata"]["availability"] == "gated"
        assert high_trust_found, "Фрагмент с высоким порогом должен быть доступен при достаточном trust"
    finally:
        await cleanup_test_data()


@pytest.mark.asyncio
async def test_nonexistent_case_id():
    """
    Тест с несуществующим case_id
    """
    fake_case_id = str(uuid.uuid4())
    session_state = {"trust": 0.5, "access_level": "medium", "risk_status": "safe"}
    topics = []
    
    result = await retrieve(
        case_id=fake_case_id,
        intent="open_question",
        topics=topics,
        session_state_compact=session_state,
        top_k=5
    )
    
    # Должен вернуться пустой список для несуществующего case_id
    assert result == []


@pytest.mark.asyncio
async def test_return_data_structure(session):
    """
    Тест структуры возвращаемых данных {id, type, text, metadata}
    """
    case_id = await setup_test_data(session)
    
    try:
        session_state = {"trust": 0.5, "access_level": "medium", "risk_status": "safe"}
        topics = []
        
        result = await retrieve(
            case_id=case_id,
            intent="open_question",
            topics=topics,
            session_state_compact=session_state,
            top_k=5
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
    finally:
        await cleanup_test_data()


@pytest.mark.asyncio
async def test_empty_database_scenario():
    """
    Тест с пустой БД
    """
    await cleanup_test_data()
    
    fake_case_id = str(uuid.uuid4())
    session_state = {"trust": 0.5, "access_level": "medium", "risk_status": "safe"}
    topics = []
    
    result = await retrieve(
        case_id=fake_case_id,
        intent="open_question",
        topics=topics,
        session_state_compact=session_state,
        top_k=5
    )
    
    # В пустой БД должен вернуться пустой список
    assert result == []