import pytest
from fakeredis.aioredis import FakeRedis

from app.core.settings import settings


@pytest.mark.anyio
async def test_session_skips_ip_even_if_ip_limit_zero(client, monkeypatch, app):
    # Один общий FakeRedis
    r = FakeRedis()

    async def fake_get_redis():
        return r

    monkeypatch.setattr("app.infra.redis.get_redis", fake_get_redis)
    app.state.redis = r

    # Включаем лимитер. IP-лимит = 0 (запрет), но есть session_id → не должен проверяться
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_IP_PER_MIN", 0)  # ПОЛНЫЙ ЗАПРЕТ IP
    monkeypatch.setattr(settings, "RATE_LIMIT_SESSION_PER_MIN", 5)

    headers = {"X-Session-ID": "sess-A", "content-type": "application/json"}
    body = {
        "therapist_utterance": "ok",
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
    # Если IP проверился бы, получили бы 429. Ожидаем не 429.
    resp = await client.post("/turn", headers=headers, json=body)
    assert resp.status_code != 429, (
        f"Expected non-429 status but got {resp.status_code}: {resp.text}"
    )


@pytest.mark.anyio
async def test_no_session_uses_ip_limit(client, monkeypatch, app):
    r = FakeRedis()

    async def fake_get_redis():
        return r

    monkeypatch.setattr("app.infra.redis.get_redis", fake_get_redis)
    app.state.redis = r

    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_IP_PER_MIN", 1)  # Только 1 запрос по IP
    monkeypatch.setattr(settings, "RATE_LIMIT_SESSION_PER_MIN", 9999)

    headers = {"content-type": "application/json"}  # НЕТ X-Session-ID!
    body = {
        "therapist_utterance": "ok",
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
    r1 = await client.post("/turn", headers=headers, json=body)
    r2 = await client.post("/turn", headers=headers, json=body)

    assert r1.status_code != 429, f"First request should pass, got: {r1.status_code} {r1.text}"
    assert r2.status_code == 429, (
        f"Second request should be rate limited, got: {r2.status_code} {r2.text}"
    )
    assert r2.json().get("scope") == "ip", f"Should be IP scope, got: {r2.json()}"


@pytest.mark.anyio
async def test_session_blocks_without_ip_check(client, monkeypatch, app):
    r = FakeRedis()

    async def fake_get_redis():
        return r

    monkeypatch.setattr("app.infra.redis.get_redis", fake_get_redis)
    app.state.redis = r

    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_SESSION_PER_MIN", 1)  # Только 1 запрос по session
    monkeypatch.setattr(settings, "RATE_LIMIT_IP_PER_MIN", 9999)

    headers = {"X-Session-ID": "sess-1", "content-type": "application/json"}
    body = {
        "therapist_utterance": "ok",
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
    ok1 = await client.post("/turn", headers=headers, json=body)
    blk = await client.post("/turn", headers=headers, json=body)

    assert ok1.status_code != 429, f"First request should pass, got: {ok1.status_code} {ok1.text}"
    assert blk.status_code == 429, (
        f"Second request should be blocked, got: {blk.status_code} {blk.text}"
    )
    assert blk.json().get("scope") == "session", f"Should be session scope, got: {blk.json()}"
