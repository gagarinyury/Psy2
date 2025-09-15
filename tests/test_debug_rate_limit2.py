import pytest
from fakeredis.aioredis import FakeRedis

from app.core.settings import settings


@pytest.mark.anyio
async def test_no_session_truly_uses_ip_check(client, monkeypatch, app):
    """ДЕТАЛЬНЫЙ тест: проверяем что без session_id вызывается только IP"""

    r = FakeRedis()
    app.state.redis = r

    monkeypatch.setattr(settings, 'RATE_LIMIT_ENABLED', True)
    monkeypatch.setattr(settings, 'RATE_LIMIT_IP_PER_MIN', 1)  # Лимит IP = 1
    monkeypatch.setattr(settings, 'RATE_LIMIT_SESSION_PER_MIN', 5)

    # Мокаем функцию allow() чтобы отследить её вызовы
    calls = []

    async def mock_allow(redis_client, key: str, capacity: int):
        calls.append(f"allow called with key={key}, capacity={capacity}")
        print(f"MOCK ALLOW CALLED: key={key}, capacity={capacity}")

        # Для IP лимита: 1-й вызов OK, 2-й блок
        if key.startswith("rl:ip:"):
            ip_call_count = len([c for c in calls if "rl:ip:" in c])
            if ip_call_count <= 1:
                print(f"IP CHECK: {key} -> TRUE (allowed, call #{ip_call_count})")
                return True
            else:
                print(f"IP CHECK: {key} -> FALSE (blocked, call #{ip_call_count})")
                return False

        # Session не должен вызываться
        if key.startswith("rl:session:"):
            print(f"SESSION CHECK: {key} -> TRUE")
            return True
        return True

    monkeypatch.setattr("app.infra.rate_limit.allow", mock_allow)

    headers = {"content-type": "application/json"}  # БЕЗ X-Session-ID!
    body = {
        "therapist_utterance": "ok",
        "session_state": {"affect": "n", "trust": 0.5, "fatigue": 0, "access_level": 1, "risk_status": "none", "last_turn_summary": ""},
        "case_id": "00000000-0000-0000-0000-000000000000",
        "options": {}
    }

    # 1-й запрос
    print("\n=== ПЕРВЫЙ ЗАПРОС ===")
    resp1 = await client.post("/turn", headers=headers, json=body)

    # 2-й запрос
    print("\n=== ВТОРОЙ ЗАПРОС ===")
    resp2 = await client.post("/turn", headers=headers, json=body)

    # Анализ вызовов
    print("\n=== АНАЛИЗ ВЫЗОВОВ ===")
    print(f"Всего вызовов allow(): {len(calls)}")
    for call in calls:
        print(f"  - {call}")

    ip_calls = [call for call in calls if "rl:ip:" in call]
    session_calls = [call for call in calls if "rl:session:" in call]

    print(f"\nIP вызовы: {len(ip_calls)}")
    print(f"Session вызовы: {len(session_calls)}")

    # КРИТИЧЕСКИЕ ПРОВЕРКИ
    assert len(session_calls) == 0, f"Session НЕ должен вызываться без session_id! Получили: {session_calls}"
    assert len(ip_calls) == 2, f"Должно быть 2 IP вызова (по одному на запрос), получили: {ip_calls}"

    # Проверяем ответы
    assert resp1.status_code != 429, f"1-й запрос должен пройти, получили: {resp1.status_code}"
    assert resp2.status_code == 429, f"2-й запрос должен быть заблокирован, получили: {resp2.status_code}"

    if resp2.status_code == 429:
        json_resp = resp2.json()
        assert json_resp.get("scope") == "ip", f"Scope должен быть 'ip', получили: {json_resp}"

    print("✅ ТЕСТ ПРОЙДЕН: Без session_id используется только IP лимит")