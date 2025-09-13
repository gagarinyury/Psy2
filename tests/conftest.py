import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool
from app.core.settings import settings


@pytest_asyncio.fixture
async def session():
    """Test database session fixture with proper isolation"""
    engine = create_async_engine(
        settings.database_url, 
        pool_pre_ping=True, 
        poolclass=NullPool
    )
    Session = async_sessionmaker(engine, expire_on_commit=False)
    
    try:
        async with Session() as s:
            yield s
    finally:
        await engine.dispose()