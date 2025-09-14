import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.main import app as fastapi_app


@pytest.fixture(scope="function")
def app() -> FastAPI:
    return fastapi_app

@pytest.fixture(scope="function")
async def client(app: FastAPI):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac