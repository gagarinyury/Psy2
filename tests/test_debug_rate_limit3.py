import pytest
from fakeredis.aioredis import FakeRedis

from app.core.settings import settings


@pytest.mark.anyio
async def test_session_blocked_no_ip_check(client, monkeypatch, app):
    """ДЕТАЛЬНЫЙ тест: session блокируется, IP НЕ вызывается"""

    r = FakeRedis()
    app.state.redis = r

    monkeypatch.setattr(settings, 'RATE_LIMIT_ENABLED', True)
    monkeypatch.setattr(settings, 'RATE_LIMIT_IP_PER_MIN', 999)  # Высокий IP лимит
    monkeypatch.setattr(settings, 'RATE_LIMIT_SESSION_PER_MIN', 1)  # Session лимит = 1

    calls = []

    async def mock_allow(redis_client, key: str, capacity: int):
        calls.append(f"allow called with key={key}, capacity={capacity}")
        print(f"MOCK ALLOW CALLED: key={key}, capacity={capacity}")

        if key.startswith("rl:session:"):
            session_call_count = len([c for c in calls if "rl:session:" in c])
            if session_call_count <= 1:
                print(f"SESSION CHECK: {key} -> TRUE (1st call allowed)")
                return True
            else:
                print(f"SESSION CHECK: {key} -> FALSE (2nd call blocked)")
                return False

        if key.startswith("rl:ip:"):
            print(f"IP CHECK: {key} -> TRUE (should NOT be called!)")
            return True

        return True

    monkeypatch.setattr("app.infra.rate_limit.allow", mock_allow)

    headers = {"X-Session-ID": "sess-limit", "content-type": "application/json"}
    body = {
        "therapist_utterance": "ok",
        "session_state": {"affect": "n", "trust": 0.5, "fatigue": 0, "access_level": 1, "risk_status": "none", "last_turn_summary": ""},
        "case_id": "00000000-0000-0000-0000-000000000000",
        "options": {}
    }

    # 1-й запрос - должен пройти
    print("\n=== ПЕРВЫЙ ЗАПРОС (должен пройти) ===")
    resp1 = await client.post("/turn", headers=headers, json=body)

    # 2-й запрос - должен блокироваться по session
    print("\n=== ВТОРОЙ ЗАПРОС (должен блокироваться по session) ===")
    resp2 = await client.post("/turn", headers=headers, json=body)

    print("\n=== АНАЛИЗ ВЫЗОВОВ ===")
    print(f"Всего вызовов allow(): {len(calls)}")
    for call in calls:
        print(f"  - {call}")

    ip_calls = [call for call in calls if "rl:ip:" in call]
    session_calls = [call for call in calls if "rl:session:" in call]

    print(f"\nIP вызовы: {len(ip_calls)}")
    print(f"Session вызовы: {len(session_calls)}")

    # КРИТИЧЕСКИЕ ПРОВЕРКИ
    assert len(session_calls) == 2, f"Должно быть 2 session вызова, получили: {session_calls}"
    assert len(ip_calls) == 0, f"IP НЕ должен вызываться вообще! Получили IP вызовы: {ip_calls}"

    # Проверяем ответы
    assert resp1.status_code != 429, f"1-й запрос должен пройти, получили: {resp1.status_code}"
    assert resp2.status_code == 429, f"2-й запрос должен быть заблокирован по session, получили: {resp2.status_code}"

    if resp2.status_code == 429:
        json_resp = resp2.json()
        assert json_resp.get("scope") == "session", f"Scope должен быть 'session', получили: {json_resp}"

    print("✅ ТЕСТ ПРОЙДЕН: При session блокировке IP вообще НЕ проверяется")