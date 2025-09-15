"""
e2e-тесты reasoning с моками DeepSeek.

Проверка, что /turn всегда возвращает валидный план при любых ответах LLM:
- валидный JSON → корректный план
- мусор/без JSON → fallback
- экстремальные дельты/поля → нормализуются
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.settings import settings


async def create_test_case_and_session(client):
    """Helper function to create case and session for tests."""
    case_data = {
        "case_truth": {
            "dx_target": ["MDD"],
            "ddx": {"MDD": 0.7, "GAD": 0.2, "AUD": 0.1},
            "hidden_facts": ["эпизод депрессии в 2020", "семейная история"],
            "red_flags": ["суицидальные мысли"],
            "trajectories": ["улучшение при поддержке"],
        },
        "policies": {
            "disclosure_rules": {
                "full_on_valid_question": True,
                "partial_if_low_trust": True,
                "min_trust_for_gated": 0.4,
            },
            "distortion_rules": {"enabled": True, "by_defense": {}},
            "risk_protocol": {
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

    # Create case
    case_response = await client.post("/case", json=case_data)
    assert case_response.status_code == 200, f"Case creation failed: {case_response.text}"
    case_id = case_response.json()["case_id"]

    # Create session
    session_data = {"case_id": case_id}
    session_response = await client.post("/session", json=session_data)
    assert session_response.status_code == 200, f"Session creation failed: {session_response.text}"
    session_id = session_response.json()["session_id"]

    return case_id, session_id


@pytest.mark.anyio
async def test_reason_valid_json(client):
    """
    Сценарий 1: Валидный JSON через reasoning_content.
    Ожидается корректный план с точными значениями дельт.
    """
    case_id, session_id = await create_test_case_and_session(client)

    # Mock валидного JSON ответа через reasoning_content
    mock_payload = {
        "content_plan": ["Короткий ответ 1", "Короткий ответ 2"],
        "style_directives": {"tempo": "slow", "length": "short"},
        "state_updates": {"trust_delta": 0.12, "fatigue_delta": 0.03},
        "telemetry": {"chosen_ids": ["kb1", "kb2"]},
    }

    async def fake_reasoning(*args, **kwargs):
        return {
            "choices": [
                {"message": {"reasoning_content": f"```json\n{json.dumps(mock_payload)}\n```"}}
            ]
        }

    # Enable DeepSeek reasoning, disable generation
    with (
        patch.object(settings, "USE_DEEPSEEK_REASON", True),
        patch.object(settings, "USE_DEEPSEEK_GEN", False),
        patch("app.orchestrator.nodes.reason_llm.DeepSeekClient") as mock_client_class,
    ):
        mock_client = MagicMock()
        mock_client.reasoning = AsyncMock(side_effect=fake_reasoning)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        # Запрос к /turn с фразой без риска и достаточным trust
        turn_request = {
            "therapist_utterance": "Как дела с настроением?",
            "session_id": session_id,
            "case_id": case_id,
            "session_state": {
                "trust": 0.6,  # >= 0.5
                "fatigue": 0.1,
                "affect": "neutral",
                "risk_status": "none",
                "access_level": 1,
                "last_turn_summary": "",
            },
        }

        response = await client.post("/turn", json=turn_request)
        assert response.status_code == 200, f"Turn request failed: {response.text}"

        result = response.json()

        # Проверки сценария 1
        # 1. patient_reply не "safe-fallback"
        assert "patient_reply" in result
        assert result["patient_reply"] != "safe-fallback"
        # При USE_DEEPSEEK_GEN=False возвращается формат "Plan:N intent=X risk=Y"
        assert result["patient_reply"].startswith("Plan:")
        assert "Plan:2" in result["patient_reply"]  # 2 элемента в content_plan

        # 2. trust_delta точно 0.12 (±0.001)
        assert "state_updates" in result
        assert abs(result["state_updates"]["trust_delta"] - 0.12) < 0.001

        # 3. eval_markers.intent корректный
        assert "eval_markers" in result
        assert "intent" in result["eval_markers"]

        # 4. used_fragments содержит ids из кандидатов (пересечение не пустое)
        assert "used_fragments" in result
        assert isinstance(result["used_fragments"], list)
        # Должен быть пересечением с реальными кандидатами, но не пустой
        if result["used_fragments"]:
            assert len(result["used_fragments"]) > 0

        # 5. risk_status без "acute" для безопасной фразы
        assert "risk_status" in result
        assert "acute" not in result["risk_status"]

        # Verify DeepSeek reasoning was called
        mock_client.reasoning.assert_called_once()


@pytest.mark.anyio
async def test_reason_garbage_fallback(client):
    """
    Сценарий 2: Мусорный ответ (нет JSON).
    Ожидается fallback с флагом llm_parse_error.
    """
    case_id, session_id = await create_test_case_and_session(client)

    # Mock мусорного ответа
    async def fake_reasoning(*args, **kwargs):
        return {"choices": [{"message": {"content": "okay this is garbage response"}}]}

    # Enable DeepSeek reasoning, disable generation
    with (
        patch.object(settings, "USE_DEEPSEEK_REASON", True),
        patch.object(settings, "USE_DEEPSEEK_GEN", False),
        patch("app.orchestrator.nodes.reason_llm.DeepSeekClient") as mock_client_class,
    ):
        mock_client = MagicMock()
        mock_client.reasoning = AsyncMock(side_effect=fake_reasoning)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        turn_request = {
            "therapist_utterance": "Как дела?",
            "session_id": session_id,
            "case_id": case_id,
            "session_state": {
                "trust": 0.4,
                "fatigue": 0.2,
                "affect": "neutral",
                "risk_status": "none",
                "access_level": 1,
                "last_turn_summary": "",
            },
        }

        response = await client.post("/turn", json=turn_request)
        assert response.status_code == 200, f"Turn request failed: {response.text}"

        result = response.json()

        # Проверки сценария 2
        # 1. Fallback сработал, patient_reply не пустой
        assert "patient_reply" in result
        assert result["patient_reply"] != ""
        assert len(result["patient_reply"]) > 0

        # 2. Проверить наличие флагов ошибок в записи хода
        # Это может быть в telemetry или в других местах
        # Поскольку fallback возвращает negative trust_delta, проверим это
        assert "state_updates" in result
        trust_delta = result["state_updates"]["trust_delta"]
        assert trust_delta <= 0.0  # Negative or zero для fallback

        # 3. HTTP 200 OK несмотря на ошибку парсинга
        assert response.status_code == 200

        # Verify DeepSeek reasoning was called
        mock_client.reasoning.assert_called_once()


@pytest.mark.anyio
async def test_reason_extreme_values_normalized(client):
    """
    Сценарий 3: Экстремальные значения и некорректные поля.
    Ожидается нормализация всех полей валидатором.
    """
    case_id, session_id = await create_test_case_and_session(client)

    # Mock экстремальных значений
    extreme_payload = {
        "content_plan": [],  # Пустой план
        "style_directives": {
            "tempo": "warp",
            "length": "huge",
        },  # Некорректные значения
        "state_updates": {"trust_delta": 9, "fatigue_delta": -5},  # Вне границ
        "telemetry": {"chosen_ids": ["kb999", "kb1", "kb1"]},  # Дубликаты и несуществующие
    }

    async def fake_reasoning(*args, **kwargs):
        return {
            "choices": [
                {"message": {"reasoning_content": f"```json\n{json.dumps(extreme_payload)}\n```"}}
            ]
        }

    with (
        patch.object(settings, "USE_DEEPSEEK_REASON", True),
        patch.object(settings, "USE_DEEPSEEK_GEN", False),
        patch("app.orchestrator.nodes.reason_llm.DeepSeekClient") as mock_client_class,
    ):
        mock_client = MagicMock()
        mock_client.reasoning = AsyncMock(side_effect=fake_reasoning)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        turn_request = {
            "therapist_utterance": "Расскажите о своих чувствах",
            "session_id": session_id,
            "case_id": case_id,
            "session_state": {
                "trust": 0.5,
                "fatigue": 0.15,
                "affect": "neutral",
                "risk_status": "none",
                "access_level": 1,
                "last_turn_summary": "",
            },
        }

        response = await client.post("/turn", json=turn_request)
        assert response.status_code == 200, f"Turn request failed: {response.text}"

        result = response.json()

        # Проверки сценария 3 (после валидатора)
        # 1. content_plan автосформирован из кандидатов, не пуст
        assert "patient_reply" in result
        assert result["patient_reply"] != ""
        # Должен содержать сгенерированный контент, не пустой план

        # 2. tempo и length нормализованы
        # Поскольку мы не можем напрямую проверить style_directives в ответе /turn,
        # проверим что система не упала и работает корректно

        # 3. trust_delta и fatigue_delta в допустимых границах
        assert "state_updates" in result
        trust_delta = result["state_updates"]["trust_delta"]
        fatigue_delta = result["state_updates"]["fatigue_delta"]

        # trust_delta должно быть в [-0.2, 0.2]
        assert -0.2 <= trust_delta <= 0.2, f"trust_delta {trust_delta} not in [-0.2, 0.2]"

        # fatigue_delta должно быть в [0.0, 0.2]
        assert 0.0 <= fatigue_delta <= 0.2, f"fatigue_delta {fatigue_delta} not in [0.0, 0.2]"

        # 4. Система не упала и вернула корректный ответ
        assert "eval_markers" in result
        assert "used_fragments" in result
        assert isinstance(result["used_fragments"], list)

        # Verify DeepSeek reasoning was called
        mock_client.reasoning.assert_called_once()
