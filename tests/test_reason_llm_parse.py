"""
Tests for reason_llm JSON parsing integration.

Tests LLM response parsing with different content formats using mocked DeepSeekClient.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from app.orchestrator.nodes.reason_llm import reason_llm


@pytest.fixture
def sample_inputs():
    """Common test inputs for reason_llm function."""
    return {
        "case_truth": {
            "dx_target": ["MDD"],
            "ddx": {"MDD": 0.8},
            "hidden_facts": ["episode in 2020"],
        },
        "session_state": {
            "trust": 0.6,
            "fatigue": 0.2,
            "affect": "neutral",
            "risk_status": "none",
        },
        "candidates": [
            {
                "id": "fragment-1",
                "text": "Patient shows signs of depression",
                "metadata": {"topic": "mood"},
            }
        ],
        "policies": {
            "disclosure_rules": {"min_trust_for_gated": 0.4},
            "risk_protocol": {"trigger_keywords": ["suicide"]},
        },
    }


@pytest.mark.anyio
async def test_reason_llm_parses_reasoning_content_json_block(sample_inputs):
    """
    Scenario A: Response with reasoning_content containing ```json block.
    Should return normalized plan with correct keys and types.
    """
    # Mock response with reasoning_content containing JSON code block
    mock_response = {
        "choices": [
            {
                "message": {
                    "reasoning_content": """```json
{
    "content_plan": ["I understand your concerns about mood"],
    "style_directives": {
        "tempo": "slow",
        "length": "medium"
    },
    "state_updates": {
        "trust_delta": 0.15,
        "fatigue_delta": 0.05
    },
    "telemetry": {
        "chosen_ids": ["fragment-1"]
    }
}
```"""
                }
            }
        ]
    }

    with patch("app.orchestrator.nodes.reason_llm.DeepSeekClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.reasoning = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await reason_llm(
            sample_inputs["case_truth"],
            sample_inputs["session_state"],
            sample_inputs["candidates"],
            sample_inputs["policies"],
        )

        # Verify normalized structure
        assert isinstance(result, dict)
        assert "content_plan" in result
        assert "style_directives" in result
        assert "state_updates" in result
        assert "telemetry" in result

        # Verify content types and values
        assert isinstance(result["content_plan"], list)
        assert result["content_plan"] == ["I understand your concerns about mood"]

        assert isinstance(result["style_directives"], dict)
        assert result["style_directives"]["tempo"] == "slow"
        assert result["style_directives"]["length"] == "medium"

        assert isinstance(result["state_updates"], dict)
        assert result["state_updates"]["trust_delta"] == 0.15
        assert result["state_updates"]["fatigue_delta"] == 0.05

        assert isinstance(result["telemetry"], dict)
        assert result["telemetry"]["chosen_ids"] == ["fragment-1"]


@pytest.mark.anyio
async def test_reason_llm_parses_content_with_text_and_json(sample_inputs):
    """
    Scenario B: Response with content containing text and raw JSON.
    Should be parsed successfully.
    """
    # Mock response with content containing explanatory text and JSON
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": """Here is the plan based on the patient input:

{
    "content_plan": ["Let's explore your feelings further", "How has this been affecting you?"],
    "style_directives": {
        "tempo": "slow",
        "length": "long"
    },
    "state_updates": {
        "trust_delta": 0.05,
        "fatigue_delta": 0.02
    },
    "telemetry": {
        "chosen_ids": ["fragment-1"]
    }
}

This approach should help build trust."""
                }
            }
        ]
    }

    with patch("app.orchestrator.nodes.reason_llm.DeepSeekClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.reasoning = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await reason_llm(
            sample_inputs["case_truth"],
            sample_inputs["session_state"],
            sample_inputs["candidates"],
            sample_inputs["policies"],
        )

        # Verify parsing succeeded and structure is correct
        assert isinstance(result, dict)
        assert result["content_plan"] == [
            "Let's explore your feelings further",
            "How has this been affecting you?",
        ]
        assert result["style_directives"]["tempo"] == "slow"
        assert result["style_directives"]["length"] == "long"
        assert result["state_updates"]["trust_delta"] == 0.05
        assert result["state_updates"]["fatigue_delta"] == 0.02
        assert result["telemetry"]["chosen_ids"] == ["fragment-1"]


@pytest.mark.anyio
async def test_reason_llm_fallback_on_garbage_response(sample_inputs):
    """
    Scenario C: Response without parseable JSON.
    Should activate fallback with llm_parse_error=True.
    """
    # Mock response with unparseable content
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": """The patient seems to be experiencing some difficulties.
I would recommend a compassionate approach, but this response contains
no valid JSON structure that can be parsed into the required format.
Just random text without proper structure."""
                }
            }
        ]
    }

    with patch("app.orchestrator.nodes.reason_llm.DeepSeekClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.reasoning = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await reason_llm(
            sample_inputs["case_truth"],
            sample_inputs["session_state"],
            sample_inputs["candidates"],
            sample_inputs["policies"],
        )

        # Verify fallback was activated
        assert isinstance(result, dict)
        assert "content_plan" in result
        assert "style_directives" in result
        assert "state_updates" in result
        assert "telemetry" in result

        # Check fallback values match stub logic
        assert result["content_plan"] == ["I'm feeling a bit confused right now"]
        assert result["style_directives"]["tempo"] == "calm"
        assert result["style_directives"]["length"] == "short"
        assert result["state_updates"]["trust_delta"] == -0.1  # Error penalty
        assert result["state_updates"]["fatigue_delta"] == 0.05

        # Most importantly - verify parse error is marked
        assert "llm_parse_error" in result["telemetry"]
        assert result["telemetry"]["llm_parse_error"] is True
        assert result["telemetry"]["chosen_ids"] == []


@pytest.mark.anyio
async def test_reason_llm_parses_json5_with_comments_and_trailing_commas(sample_inputs):
    """
    Additional test: JSON5 format with comments and trailing commas should be handled.
    """
    mock_response = {
        "choices": [
            {
                "message": {
                    "reasoning_content": """```json
{
    // Patient response analysis
    "content_plan": [
        "I hear what you're saying", // Empathetic opening
        "Tell me more about that", // Exploration
    ],
    "style_directives": {
        "tempo": "medium", // Not too fast
        "length": "short", // Keep it brief
    },
    "state_updates": {
        "trust_delta": 0.1, // Small trust increase
        "fatigue_delta": 0.0, // No change
    },
    "telemetry": {
        "chosen_ids": ["fragment-1"], // Used fragments
    },
}
```"""
                }
            }
        ]
    }

    with patch("app.orchestrator.nodes.reason_llm.DeepSeekClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.reasoning = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await reason_llm(
            sample_inputs["case_truth"],
            sample_inputs["session_state"],
            sample_inputs["candidates"],
            sample_inputs["policies"],
        )

        # Verify JSON5 parsing worked
        assert result["content_plan"] == [
            "I hear what you're saying",
            "Tell me more about that",
        ]
        assert result["style_directives"]["tempo"] == "medium"
        assert result["state_updates"]["trust_delta"] == 0.1
        assert result["telemetry"]["chosen_ids"] == ["fragment-1"]
        # Should not have parse error
        assert result["telemetry"].get("llm_parse_error") is None
