import logging

import pytest
from fakeredis.aioredis import FakeRedis

from app.core.settings import settings

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)


@pytest.mark.anyio
async def test_session_truly_skips_ip_check(client, monkeypatch, app, caplog):
    """ДЕТАЛЬНЫЙ тест: проверяем что IP вообще не вызывается при session_id"""

    r = FakeRedis()
    app.state.redis = r

    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_IP_PER_MIN", 0)  # ПОЛНЫЙ ЗАПРЕТ IP
    monkeypatch.setattr(settings, "RATE_LIMIT_SESSION_PER_MIN", 5)

    # Мокаем функцию allow() чтобы отследить её вызовы
    original_allow = None
    calls = []

    async def mock_allow(redis_client, key: str, capacity: int):
        calls.append(f"allow called with key={key}, capacity={capacity}")
        print(f"MOCK ALLOW CALLED: key={key}, capacity={capacity}")

        # Для IP лимита всегда возвращаем False (блок)
        if key.startswith("rl:ip:"):
            print(f"IP CHECK: {key} -> FALSE (blocked)")
            return False
        # Для session лимита возвращаем True (разрешен)
        if key.startswith("rl:session:"):
            print(f"SESSION CHECK: {key} -> TRUE (allowed)")
            return True
        return True

    # Патчим функцию allow в модуле rate_limit
    monkeypatch.setattr("app.infra.rate_limit.allow", mock_allow)

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

    with caplog.at_level(logging.DEBUG):
        resp = await client.post("/turn", headers=headers, json=body)

    # Анализ вызовов
    print("\n=== АНАЛИЗ ВЫЗОВОВ ===")
    print(f"Всего вызовов allow(): {len(calls)}")
    for call in calls:
        print(f"  - {call}")

    # Проверяем что IP НЕ вызывался
    ip_calls = [call for call in calls if "rl:ip:" in call]
    session_calls = [call for call in calls if "rl:session:" in call]

    print(f"\nIP вызовы: {len(ip_calls)}")
    print(f"Session вызовы: {len(session_calls)}")

    # КРИТИЧЕСКИЕ ПРОВЕРКИ
    assert len(session_calls) == 1, f"Должен быть 1 session вызов, получили: {session_calls}"
    assert len(ip_calls) == 0, (
        f"IP НЕ должен вызываться при наличии session_id! Получили IP вызовы: {ip_calls}"
    )

    # Ответ должен быть не 429 (так как session разрешен)
    assert resp.status_code != 429, (
        f"Response should not be 429, got {resp.status_code}: {resp.text}"
    )

    print("✅ ТЕСТ ПРОЙДЕН: IP действительно НЕ проверяется при наличии session_id")
