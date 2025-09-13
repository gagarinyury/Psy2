import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_custom_trigger_keywords_guarantee_risk(client: AsyncClient):
    """
    Тест жёсткой гарантии риска - кастомные trigger_keywords должны точно срабатывать.
    """
    # Создаем case с кастомными trigger_keywords
    case_request = {
        "case_truth": {
            "dx_target": ["MDD"],
            "ddx": {"MDD": 0.8, "GAD": 0.2},
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
                "trigger_keywords": [
                    "опасное_слово",
                    "кастомный_триггер",
                    "тест_риска",
                ],
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

    # Базовый turn request template
    turn_template = {
        "session_state": {
            "affect": "neutral",
            "trust": 0.5,
            "fatigue": 0.1,
            "access_level": 1,
            "risk_status": "none",
            "last_turn_summary": "",
        },
        "case_id": case_id,
        "session_id": session_id,
        "options": {},
    }

    # Тест 1: Фраза без триггеров - должна быть risk="none"
    turn_request = turn_template.copy()
    turn_request["therapist_utterance"] = "Как дела? Что происходит?"

    response = await client.post("/turn", json=turn_request)
    assert response.status_code == 200

    result = response.json()
    assert result["risk_status"] == "none"
    assert result["eval_markers"]["intent"] == "clarify"  # "Как" -> clarify

    # Тест 2: Фраза с кастомным триггером - должна быть risk="acute"
    turn_request = turn_template.copy()
    turn_request["therapist_utterance"] = "Есть ли у вас мысли об опасное_слово?"

    response = await client.post("/turn", json=turn_request)
    assert response.status_code == 200

    result = response.json()
    assert result["risk_status"] == "acute"
    assert result["eval_markers"]["intent"] == "risk_check"  # триггер -> risk_check

    # Тест 3: Фраза с другим кастомным триггером
    turn_request = turn_template.copy()
    turn_request["therapist_utterance"] = (
        "Подскажите про кастомный_триггер и ваше состояние"
    )

    response = await client.post("/turn", json=turn_request)
    assert response.status_code == 200

    result = response.json()
    assert result["risk_status"] == "acute"
    assert result["eval_markers"]["intent"] == "risk_check"

    # Тест 4: Проверяем что стандартные триггеры НЕ работают
    turn_request = turn_template.copy()
    turn_request["therapist_utterance"] = "Есть ли мысли о суициде?"

    response = await client.post("/turn", json=turn_request)
    assert response.status_code == 200

    result = response.json()
    # "суицид" НЕ в кастомном списке, поэтому риска нет
    assert result["risk_status"] == "none"
    # Без триггеров -> дефолтный intent (не risk_check)
    assert result["eval_markers"]["intent"] != "risk_check"


@pytest.mark.anyio
async def test_default_trigger_keywords_still_work(client: AsyncClient):
    """
    Тест что дефолтные trigger_keywords все ещё работают если не заданы кастомные.
    """
    # Создаем case БЕЗ кастомных trigger_keywords (используем дефолтные)
    case_request = {
        "case_truth": {
            "dx_target": ["MDD"],
            "ddx": {"MDD": 0.8},
            "hidden_facts": ["test"],
            "red_flags": ["test"],
            "trajectories": ["test"],
        },
        "policies": {
            "disclosure_rules": {
                "full_on_valid_question": True,
                "partial_if_low_trust": False,
                "min_trust_for_gated": 0.4,
            },
            "distortion_rules": {"enabled": True, "by_defense": {}},
            "risk_protocol": {
                # Используем дефолтные trigger_keywords из RiskProtocol
                "trigger_keywords": ["суицид", "убить себя", "не хочу жить"],
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

    # Turn request с дефолтным триггером
    turn_request = {
        "therapist_utterance": "Бывают ли мысли о суициде?",
        "session_state": {
            "affect": "neutral",
            "trust": 0.5,
            "fatigue": 0.1,
            "access_level": 1,
            "risk_status": "none",
            "last_turn_summary": "",
        },
        "case_id": case_id,
        "session_id": session_id,
        "options": {},
    }

    response = await client.post("/turn", json=turn_request)
    assert response.status_code == 200

    result = response.json()
    assert result["risk_status"] == "acute"  # Триггер сработал
    assert result["eval_markers"]["intent"] == "risk_check"


@pytest.mark.anyio
async def test_case_insensitive_trigger_matching(client: AsyncClient):
    """
    Тест что триггеры работают независимо от регистра.
    """
    # Case с триггером в нижнем регистре
    case_request = {
        "case_truth": {
            "dx_target": ["MDD"],
            "ddx": {"MDD": 1.0},
            "hidden_facts": ["test"],
            "red_flags": ["test"],
            "trajectories": ["test"],
        },
        "policies": {
            "disclosure_rules": {
                "full_on_valid_question": True,
                "partial_if_low_trust": False,
                "min_trust_for_gated": 0.4,
            },
            "distortion_rules": {"enabled": True, "by_defense": {}},
            "risk_protocol": {
                "trigger_keywords": ["опасность", "Угроза", "РИСК"],  # Разный регистр
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

    session_request = {"case_id": case_id}
    response = await client.post("/session", json=session_request)
    assert response.status_code == 200
    session_id = response.json()["session_id"]

    turn_template = {
        "session_state": {
            "affect": "neutral",
            "trust": 0.5,
            "fatigue": 0.1,
            "access_level": 1,
            "risk_status": "none",
            "last_turn_summary": "",
        },
        "case_id": case_id,
        "session_id": session_id,
        "options": {},
    }

    # Тест разных регистров
    test_cases = [
        ("Есть ли ОПАСНОСТЬ?", "acute"),  # uppercase -> trigger
        ("Какая угроза?", "acute"),  # lowercase -> trigger
        ("Большой риск", "acute"),  # uppercase trigger -> acute
        ("Никаких проблем", "none"),  # no trigger -> none
    ]

    for utterance, expected_risk in test_cases:
        turn_request = turn_template.copy()
        turn_request["therapist_utterance"] = utterance

        response = await client.post("/turn", json=turn_request)
        assert response.status_code == 200

        result = response.json()
        assert (
            result["risk_status"] == expected_risk
        ), f"Failed for '{utterance}': expected {expected_risk}, got {result['risk_status']}"
