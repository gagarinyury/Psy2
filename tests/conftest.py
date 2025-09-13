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


@pytest.fixture
async def setup_test_case(client: AsyncClient):
    """Создает тестовый case и session для тестов."""
    # Создаем case
    case_request = {
        "case_truth": {
            "dx_target": ["MDD"],
            "ddx": {"MDD": 0.6, "GAD": 0.3},
            "hidden_facts": ["test fact"],
            "red_flags": ["test flag"],
            "trajectories": ["test trajectory"],
        },
        "policies": {
            "disclosure_rules": {
                "full_on_valid_question": True,
                "partial_if_low_trust": False,
                "min_trust_for_gated": 0.4,
            },
            "distortion_rules": {"enabled": True, "by_defense": {}},
            "risk_protocol": {
                "trigger_keywords": ["суицид", "убить себя"],
                "response_style": "stable",
                "lock_topics": [],
            },
            "style_profile": {
                "register": "colloquial",
                "tempo": "medium",
                "length": "short",
            },
        },
    }

    response = await client.post("/case", json=case_request)
    assert response.status_code == 200
    case_id = response.json()["case_id"]

    # Создаем session
    session_request = {"case_id": case_id}
    response = await client.post("/session", json=session_request)
    assert response.status_code == 200
    session_id = response.json()["session_id"]

    return case_id, session_id
