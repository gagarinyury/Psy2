import pytest
from fakeredis.aioredis import FakeRedis

from app.core.settings import settings


@pytest.mark.anyio
async def test_session_blocks_without_ip_check(client, monkeypatch, app):
    # один общий fake redis
    r = FakeRedis()

    async def fake_get_redis():
        return r

    monkeypatch.setattr("app.infra.redis.get_redis", fake_get_redis)

    # Patch app.state.redis directly since tests don't go through lifespan
    app.state.redis = r

    # форсируем лимит сессии: capacity=1, выжать 1 токен и второй запрос получить 429 session
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_SESSION_PER_MIN", 1)
    monkeypatch.setattr(settings, "RATE_LIMIT_IP_PER_MIN", 9999)

    headers = {"X-Session-ID": "sess-1", "content-type": "application/json"}
    payload = {
        "therapist_utterance": "hi",
        "session_state": {
            "affect": "n",
            "trust": 0.5,
            "fatigue": 0,
            "access_level": 1,
            "risk_status": "none",
            "last_turn_summary": "",
        },
        "case_id": "00000000-0000-0000-0000-000000000000",
        "options": {},
    }

    # 1-й — проходит
    resp1 = await client.post("/turn", headers=headers, json=payload)
    assert resp1.status_code in (200, 400, 422)  # зависит от валидатора case_id; важно, что не 429

    # 2-й — должен быть 429 по scope=session
    resp2 = await client.post("/turn", headers=headers, json=payload)
    assert resp2.status_code == 429
    assert resp2.json().get("scope") == "session"


@pytest.mark.anyio
async def test_fail_closed_on_redis_error(client, monkeypatch, app):
    class Boom:
        async def eval(self, *a, **k):
            raise RuntimeError("redis down")

    async def fake_get_redis():
        return Boom()

    monkeypatch.setattr("app.infra.redis.get_redis", fake_get_redis)

    # Patch app.state.redis directly since tests don't go through lifespan
    app.state.redis = Boom()

    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_FAIL_OPEN", False)

    resp = await client.post(
        "/turn",
        headers={"content-type": "application/json"},
        json={
            "therapist_utterance": "x",
            "session_state": {
                "affect": "n",
                "trust": 0.5,
                "fatigue": 0,
                "access_level": 1,
                "risk_status": "none",
                "last_turn_summary": "",
            },
            "case_id": "00000000-0000-0000-0000-000000000000",
            "options": {},
        },
    )
    assert resp.status_code == 503
    assert resp.json()["detail"] == "rate_limiter_unavailable"


@pytest.mark.anyio
async def test_fail_open_allows_when_enabled(client, monkeypatch, app):
    class Boom:
        async def eval(self, *a, **k):
            raise RuntimeError("redis down")

    async def fake_get_redis():
        return Boom()

    monkeypatch.setattr("app.infra.redis.get_redis", fake_get_redis)

    # Patch app.state.redis directly since tests don't go through lifespan
    app.state.redis = Boom()

    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_FAIL_OPEN", True)

    resp = await client.post(
        "/turn",
        headers={"content-type": "application/json"},
        json={
            "therapist_utterance": "x",
            "session_state": {
                "affect": "n",
                "trust": 0.5,
                "fatigue": 0,
                "access_level": 1,
                "risk_status": "none",
                "last_turn_summary": "",
            },
            "case_id": "00000000-0000-0000-0000-000000000000",
            "options": {},
        },
    )
    assert resp.status_code != 503  # пропущен


def test_lua_uses_hset():
    from app.infra.rate_limit import LUA_TOKEN_BUCKET

    assert "HSET" in LUA_TOKEN_BUCKET
    assert "HMSET" not in LUA_TOKEN_BUCKET
