import os
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.core.settings import settings

# Database URL for tests - use existing settings or TEST_DATABASE_URL env var
DB_URL = os.environ.get("TEST_DATABASE_URL", settings.database_url)


@pytest.fixture
async def engine():
    """Create async engine with NullPool for each test"""
    eng = create_async_engine(
        DB_URL, 
        pool_pre_ping=True, 
        poolclass=NullPool,
        echo=False
    )
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest.fixture
async def session(engine):
    """Create async session for each test"""
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as s:
        yield s


@pytest.fixture
async def client():
    """Create FastAPI test client with proper ASGI transport"""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac