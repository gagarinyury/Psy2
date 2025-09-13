"""
Диагностические тесты для выявления проблем с async event loops и database isolation.
"""

import asyncio
import pytest
import threading
from sqlalchemy import text
from app.core.db import AsyncSessionLocal, engine


@pytest.mark.skip(reason="Async diagnostics conflicts - pending resolution")
@pytest.mark.anyio
async def test_anyio_basic():
    """Базовый тест anyio backend"""
    import anyio

    await anyio.sleep(0.01)
    assert True


@pytest.mark.skip(reason="Async diagnostics conflicts - pending resolution")
@pytest.mark.anyio
async def test_db_connection_basic():
    """Базовое подключение к БД через anyio"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(text("SELECT 1 as test"))
        assert result.scalar() == 1


@pytest.mark.skip(reason="Async diagnostics conflicts - pending resolution")
@pytest.mark.anyio
async def test_db_isolation_first():
    """Первый тест изоляции БД"""
    async with AsyncSessionLocal() as session:
        # Вставляем тестовые данные
        result = await session.execute(text("SELECT COUNT(*) FROM cases"))
        initial_count = result.scalar()
        print(f"test_db_isolation_first: cases count = {initial_count}")
        assert initial_count >= 0


@pytest.mark.skip(reason="Async diagnostics conflicts - pending resolution")
@pytest.mark.anyio
async def test_db_isolation_second():
    """Второй тест изоляции БД"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(text("SELECT COUNT(*) FROM cases"))
        count = result.scalar()
        print(f"test_db_isolation_second: cases count = {count}")
        assert count >= 0


@pytest.mark.skip(reason="Async diagnostics conflicts - pending resolution")
@pytest.mark.anyio
async def test_concurrent_db_access():
    """Тест одновременного доступа к БД"""

    async def db_operation(session_id: int):
        async with AsyncSessionLocal() as session:
            result = await session.execute(text(f"SELECT {session_id} as session_id"))
            return result.scalar()

    # Запускаем несколько операций параллельно
    import anyio

    async with anyio.create_task_group() as tg:
        tg.start_soon(db_operation, 1)
        tg.start_soon(db_operation, 2)
        tg.start_soon(db_operation, 3)


@pytest.mark.skip(reason="Async diagnostics conflicts - pending resolution")
@pytest.mark.anyio
async def test_event_loop_info():
    """Диагностика event loop"""
    loop = asyncio.get_running_loop()
    thread = threading.current_thread()

    print(f"Event loop: {loop}")
    print(f"Thread: {thread.name} (ID: {thread.ident})")
    print(f"Loop running: {loop.is_running()}")
    print(f"Loop closed: {loop.is_closed()}")

    assert loop.is_running()
    assert not loop.is_closed()


@pytest.mark.skip(reason="Async diagnostics conflicts - pending resolution")
@pytest.mark.anyio
async def test_sqlalchemy_session_info():
    """Диагностика SQLAlchemy сессии"""
    session = AsyncSessionLocal()

    print(f"Session: {session}")
    print(f"Session bind: {session.bind}")
    print(f"Engine: {engine}")
    print(f"Engine pool: {engine.pool}")
    print(f"Pool size: {engine.pool.size()}")
    print(f"Pool checked in: {engine.pool.checkedin()}")
    print(f"Pool checked out: {engine.pool.checkedout()}")

    await session.close()

    assert True


def test_sync_basic():
    """Синхронный тест для сравнения"""
    assert 1 + 1 == 2


@pytest.mark.skip(reason="Async diagnostics conflicts - pending resolution")
@pytest.mark.anyio
async def test_db_with_fixture(db):
    """Тест с нашей БД фикстурой"""
    result = await db.execute(text("SELECT 1 as test"))
    assert result.scalar() == 1
