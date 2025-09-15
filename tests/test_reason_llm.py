import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.orchestrator.nodes.reason_llm import (
    _create_fallback_response,
    _truncate_candidates,
    reason_llm,
)


@pytest.mark.anyio
async def test_reason_llm_success():
    """
    Тест успешного reasoning через DeepSeek API.
    """
    # Mock данные для входа
    case_truth = {
        "dx_target": ["MDD"],
        "ddx": {"MDD": 0.8},
        "hidden_facts": ["episode in 2020"],
    }

    session_state = {
        "trust": 0.6,
        "fatigue": 0.2,
        "affect": "neutral",
        "risk_status": "none",
    }

    candidates = [
        {
            "id": "fragment-1",
            "text": "Patient shows signs of depression",
            "metadata": {"topic": "mood"},
        },
        {
            "id": "fragment-2",
            "text": "Sleep disturbances reported",
            "metadata": {"topic": "sleep"},
        },
    ]

    policies = {
        "disclosure_rules": {"min_trust_for_gated": 0.4},
        "risk_protocol": {"trigger_keywords": ["suicide"]},
    }

    # Mock успешный ответ от DeepSeek
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "content_plan": [
                                "I've been feeling down lately",
                                "Sleep has been difficult",
                            ],
                            "style_directives": {"tempo": "medium", "length": "medium"},
                            "state_updates": {
                                "trust_delta": 0.1,
                                "fatigue_delta": 0.05,
                            },
                            "telemetry": {"chosen_ids": ["fragment-1", "fragment-2"]},
                        }
                    )
                }
            }
        ]
    }

    # Mock DeepSeekClient using patch context
    with patch("app.orchestrator.nodes.reason_llm.DeepSeekClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.reasoning = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await reason_llm(case_truth, session_state, candidates, policies)

        # Проверяем структуру результата
        assert "content_plan" in result
        assert "style_directives" in result
        assert "state_updates" in result
        assert "telemetry" in result

        # Проверяем содержимое
        assert result["content_plan"] == [
            "I've been feeling down lately",
            "Sleep has been difficult",
        ]
        assert result["style_directives"]["tempo"] == "medium"
        assert result["style_directives"]["length"] == "medium"
        assert result["state_updates"]["trust_delta"] == 0.1
        assert result["state_updates"]["fatigue_delta"] == 0.05
        assert result["telemetry"]["chosen_ids"] == ["fragment-1", "fragment-2"]

        # Проверяем что API был вызван правильно
        mock_client.reasoning.assert_called_once()
        call_args = mock_client.reasoning.call_args[0][0]  # messages argument
        assert len(call_args) == 2
        assert call_args[0]["role"] == "system"
        assert call_args[1]["role"] == "user"


@pytest.mark.anyio
async def test_reason_llm_invalid_json_fallback():
    """
    Тест fallback при получении некорректного JSON от DeepSeek.
    """
    case_truth = {"dx_target": ["GAD"]}
    session_state = {"trust": 0.3}
    candidates = []
    policies = {}

    # Mock ответ с некорректным JSON
    mock_response = {
        "choices": [{"message": {"content": "This is not valid JSON content from the model"}}]
    }

    with patch("app.orchestrator.nodes.reason_llm.DeepSeekClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.reasoning = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await reason_llm(case_truth, session_state, candidates, policies)

        # Проверяем что использовался fallback
        assert result["content_plan"] == ["I'm feeling a bit confused right now"]
        assert result["style_directives"]["tempo"] == "calm"
        assert result["state_updates"]["trust_delta"] == -0.1  # Отрицательный при fallback
        assert result["telemetry"]["chosen_ids"] == []


@pytest.mark.anyio
async def test_reason_llm_empty_response_fallback():
    """
    Тест fallback при пустом ответе от DeepSeek.
    """
    case_truth = {"dx_target": ["PTSD"]}
    session_state = {"trust": 0.5}
    candidates = []
    policies = {}

    # Mock пустой ответ
    mock_response = {"choices": []}

    with patch("app.orchestrator.nodes.reason_llm.DeepSeekClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.reasoning = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await reason_llm(case_truth, session_state, candidates, policies)

        # Проверяем fallback response
        assert result["content_plan"] == ["I'm feeling a bit confused right now"]
        assert result["state_updates"]["trust_delta"] == -0.1


@pytest.mark.anyio
async def test_reason_llm_api_exception_fallback():
    """
    Тест fallback при исключении в API.
    """
    case_truth = {"dx_target": ["OCD"]}
    session_state = {"trust": 0.7}
    candidates = []
    policies = {}

    # Mock исключение при вызове API
    with patch("app.orchestrator.nodes.reason_llm.DeepSeekClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.reasoning = AsyncMock(side_effect=Exception("API connection failed"))
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await reason_llm(case_truth, session_state, candidates, policies)

        # Проверяем fallback response
        assert result["content_plan"] == ["I'm feeling a bit confused right now"]
        assert result["state_updates"]["trust_delta"] == -0.1


@pytest.mark.anyio
async def test_reason_llm_response_validation():
    """
    Тест валидации и исправления структуры ответа.
    """
    case_truth = {"dx_target": ["BPD"]}
    session_state = {"trust": 0.4}
    candidates = []
    policies = {}

    # Mock ответ с неполной структурой
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "content_plan": "Just a string instead of array",  # Неправильный тип
                            "style_directives": {"tempo": "medium"},  # Отсутствует length
                            # Отсутствуют state_updates и telemetry
                        }
                    )
                }
            }
        ]
    }

    with patch("app.orchestrator.nodes.reason_llm.DeepSeekClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.reasoning = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await reason_llm(case_truth, session_state, candidates, policies)

        # Проверяем что структура была исправлена
        assert isinstance(result["content_plan"], list)
        assert result["content_plan"] == ["Just a string instead of array"]

        assert isinstance(result["style_directives"], dict)
        assert "tempo" in result["style_directives"]

        assert isinstance(result["state_updates"], dict)
        assert "trust_delta" in result["state_updates"]
        assert "fatigue_delta" in result["state_updates"]

        assert isinstance(result["telemetry"], dict)
        assert "chosen_ids" in result["telemetry"]
        assert isinstance(result["telemetry"]["chosen_ids"], list)


def test_truncate_candidates():
    """
    Тест функции обрезания текста кандидатов.
    """
    long_text = "a" * 1000  # 1000 символов
    candidates = [
        {"id": "1", "text": "short text", "metadata": {}},
        {"id": "2", "text": long_text, "metadata": {"topic": "test"}},
        {"id": "3", "text": "another short", "metadata": {}},
    ]

    result = _truncate_candidates(candidates, max_length=500)

    assert len(result) == 3
    assert result[0]["text"] == "short text"  # Не обрезан
    assert len(result[1]["text"]) == 503  # Обрезан до 500 + "..."
    assert result[1]["text"].endswith("...")
    assert result[2]["text"] == "another short"  # Не обрезан

    # Проверяем что метаданные сохранились
    assert result[1]["metadata"]["topic"] == "test"


def test_create_fallback_response():
    """
    Тест создания fallback ответа.
    """
    case_truth = {"dx_target": ["Depression"]}
    session_state = {"trust": 0.6}

    result = _create_fallback_response(case_truth, session_state)

    assert isinstance(result, dict)
    assert "content_plan" in result
    assert "style_directives" in result
    assert "state_updates" in result
    assert "telemetry" in result

    # Fallback должен уменьшать доверие
    assert result["state_updates"]["trust_delta"] == -0.1
    assert result["state_updates"]["fatigue_delta"] == 0.05

    # Стиль должен быть консервативным
    assert result["style_directives"]["tempo"] == "calm"
    assert result["style_directives"]["length"] == "short"
