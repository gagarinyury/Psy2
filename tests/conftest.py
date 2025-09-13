import asyncio
import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport
from app.main import app as fastapi_app
from app.core.db import engine


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def app() -> FastAPI:
    return fastapi_app


@pytest.fixture(scope="function")
async def client(app: FastAPI):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    
    # Clean up any lingering database connections after each test
    await engine.dispose()