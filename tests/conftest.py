import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.core.db import AsyncSessionLocal
from app.main import app as fastapi_app


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="function")
def app() -> FastAPI:
    return fastapi_app


@pytest.fixture(scope="function")
async def client(app: FastAPI):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture(scope="function")
async def db_session():
    """Создает изолированную DB сессию для тестов"""
    session = AsyncSessionLocal()
    try:
        yield session
    finally:
        if session.in_transaction():
            await session.rollback()
        await session.close()
