import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from app.core.settings import settings


@pytest.mark.anyio
async def test_pipeline_with_deepseek_reason_enabled(client):
    """
    Тест pipeline с включенным DeepSeek reasoning.
    Проверяем что reason_llm вызывается и trust_delta передается из LLM.
    """
    # Setup: Create case and session
    case_data = {
        "case_truth": {
            "dx_target": ["MDD"],
            "ddx": {"MDD": 0.8, "GAD": 0.2},
            "hidden_facts": ["episode in 2020"],
            "red_flags": ["depression symptoms"],
            "trajectories": ["improving with therapy"],
        },
        "policies": {
            "disclosure_rules": {
                "full_on_valid_question": True,
                "partial_if_low_trust": True,
                "min_trust_for_gated": 0.4,
            },
            "distortion_rules": {"enabled": True, "by_defense": {}},
            "risk_protocol": {
                "trigger_keywords": ["suicide"],
                "response_style": "stable",
                "lock_topics": [],
            },
            "style_profile": {
                "register": "casual",
                "tempo": "medium",
                "length": "short",
            },
        },
    }

    case_response = await client.post("/case", json=case_data)
    assert case_response.status_code == 200
    case_id = case_response.json()["case_id"]

    session_data = {"case_id": case_id}
    session_response = await client.post("/session", json=session_data)
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]

    # Mock DeepSeek reasoning response
    mock_reasoning_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "content_plan": [
                                "I've been feeling down lately",
                                "Sleep has been difficult",
                            ],
                            "style_directives": {"tempo": "calm", "length": "medium"},
                            "state_updates": {
                                "trust_delta": 0.15,
                                "fatigue_delta": 0.05,
                            },
                            "telemetry": {"chosen_ids": ["fragment-1", "fragment-2"]},
                        }
                    )
                }
            }
        ]
    }

    # Enable DeepSeek reasoning
    with patch.object(settings, "USE_DEEPSEEK_REASON", True), patch.object(
        settings, "USE_DEEPSEEK_GEN", False
    ), patch("app.orchestrator.nodes.reason_llm.DeepSeekClient") as mock_client_class:
        # Mock the client
        mock_client = MagicMock()
        mock_client.reasoning = AsyncMock(return_value=mock_reasoning_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        # Make turn request
        turn_request = {
            "therapist_utterance": "How are you feeling today?",
            "session_id": session_id,
            "case_id": case_id,
            "session_state": {
                "trust": 0.5,
                "fatigue": 0.2,
                "affect": "neutral",
                "risk_status": "none",
                "access_level": 1,
                "last_turn_summary": "",
            },
        }

        response = await client.post("/turn", json=turn_request)
        assert response.status_code == 200

        result = response.json()

        # Verify that DeepSeek reasoning was used
        mock_client.reasoning.assert_called_once()

        # Check that trust_delta from LLM is present
        assert "trust_delta" in result["state_updates"]
        assert result["state_updates"]["trust_delta"] == 0.15

        # Since generation is disabled, should still have Plan: format
        assert result["patient_reply"].startswith("Plan:")
        assert "intent=" in result["patient_reply"]

        # Should have eval_markers with intent
        assert "eval_markers" in result
        assert "intent" in result["eval_markers"]


@pytest.mark.anyio
async def test_pipeline_with_deepseek_generation_enabled(client):
    """
    Тест pipeline с включенным DeepSeek generation.
    Проверяем что generate_llm вызывается и возвращает строку вместо Plan: формата.
    """
    # Setup: Create case and session
    case_data = {
        "case_truth": {
            "dx_target": ["GAD"],
            "ddx": {"GAD": 0.7, "MDD": 0.3},
            "hidden_facts": ["anxiety episodes"],
            "red_flags": ["panic attacks"],
            "trajectories": ["managing anxiety with CBT"],
        },
        "policies": {
            "disclosure_rules": {
                "full_on_valid_question": True,
                "partial_if_low_trust": True,
                "min_trust_for_gated": 0.4,
            },
            "distortion_rules": {"enabled": True, "by_defense": {}},
            "risk_protocol": {
                "trigger_keywords": ["panic"],
                "response_style": "stable",
                "lock_topics": [],
            },
            "style_profile": {
                "register": "casual",
                "tempo": "medium",
                "length": "short",
            },
        },
    }

    case_response = await client.post("/case", json=case_data)
    assert case_response.status_code == 200
    case_id = case_response.json()["case_id"]

    session_data = {"case_id": case_id}
    session_response = await client.post("/session", json=session_data)
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]

    # Mock DeepSeek generation response
    mock_generation_response = {
        "choices": [
            {
                "message": {
                    "content": "I've been really anxious lately, especially at night when I try to sleep."
                }
            }
        ]
    }

    # Enable DeepSeek generation (but not reasoning)
    with patch.object(settings, "USE_DEEPSEEK_REASON", False), patch.object(
        settings, "USE_DEEPSEEK_GEN", True
    ), patch("app.orchestrator.nodes.generate_llm.DeepSeekClient") as mock_client_class:
        # Mock the client
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value=mock_generation_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        # Make turn request
        turn_request = {
            "therapist_utterance": "Tell me more about your anxiety",
            "session_id": session_id,
            "case_id": case_id,
            "session_state": {
                "trust": 0.6,
                "fatigue": 0.1,
                "affect": "anxious",
                "risk_status": "none",
                "access_level": 1,
                "last_turn_summary": "",
            },
        }

        response = await client.post("/turn", json=turn_request)
        assert response.status_code == 200

        result = response.json()

        # Verify that DeepSeek generation was used
        mock_client.generate.assert_called_once()

        # Should have natural response instead of Plan: format
        assert not result["patient_reply"].startswith("Plan:")
        assert (
            result["patient_reply"]
            == "I've been really anxious lately, especially at night when I try to sleep."
        )

        # Should still have eval_markers
        assert "eval_markers" in result
        assert "intent" in result["eval_markers"]


@pytest.mark.anyio
async def test_pipeline_with_both_deepseek_flags_enabled(client):
    """
    Тест pipeline с обоими флагами DeepSeek включенными.
    """
    # Setup: Create case and session
    case_data = {
        "case_truth": {
            "dx_target": ["PTSD"],
            "ddx": {"PTSD": 0.9},
            "hidden_facts": ["trauma history"],
            "red_flags": ["flashbacks", "nightmares"],
            "trajectories": ["PTSD recovery through therapy"],
        },
        "policies": {
            "disclosure_rules": {"min_trust_for_gated": 0.4},
            "risk_protocol": {
                "trigger_keywords": ["flashback"],
                "response_style": "stable",
                "lock_topics": [],
            },
            "style_profile": {
                "register": "casual",
                "tempo": "slow",
                "length": "medium",
            },
        },
    }

    case_response = await client.post("/case", json=case_data)
    assert case_response.status_code == 200
    case_id = case_response.json()["case_id"]

    session_data = {"case_id": case_id}
    session_response = await client.post("/session", json=session_data)
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]

    # Mock responses
    mock_reasoning_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "content_plan": [
                                "The nightmares are getting worse",
                                "I keep having flashbacks",
                            ],
                            "style_directives": {"tempo": "slow", "length": "medium"},
                            "state_updates": {"trust_delta": 0.1, "fatigue_delta": 0.1},
                            "telemetry": {"chosen_ids": ["fragment-trauma"]},
                        }
                    )
                }
            }
        ]
    }

    mock_generation_response = {
        "choices": [
            {
                "message": {
                    "content": "The nightmares have been really intense lately. Sometimes I wake up in a cold sweat from flashbacks."
                }
            }
        ]
    }

    # Enable both flags
    with patch.object(settings, "USE_DEEPSEEK_REASON", True), patch.object(
        settings, "USE_DEEPSEEK_GEN", True
    ), patch(
        "app.orchestrator.nodes.reason_llm.DeepSeekClient"
    ) as mock_reason_client_class, patch(
        "app.orchestrator.nodes.generate_llm.DeepSeekClient"
    ) as mock_gen_client_class:
        # Mock reasoning client
        mock_reason_client = MagicMock()
        mock_reason_client.reasoning = AsyncMock(return_value=mock_reasoning_response)
        mock_reason_client_class.return_value.__aenter__ = AsyncMock(
            return_value=mock_reason_client
        )
        mock_reason_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock generation client
        mock_gen_client = MagicMock()
        mock_gen_client.generate = AsyncMock(return_value=mock_generation_response)
        mock_gen_client_class.return_value.__aenter__ = AsyncMock(
            return_value=mock_gen_client
        )
        mock_gen_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        # Make turn request
        turn_request = {
            "therapist_utterance": "How have you been sleeping?",
            "session_id": session_id,
            "case_id": case_id,
            "session_state": {
                "trust": 0.4,
                "fatigue": 0.3,
                "affect": "distressed",
                "risk_status": "none",
                "access_level": 1,
                "last_turn_summary": "",
            },
        }

        response = await client.post("/turn", json=turn_request)
        assert response.status_code == 200

        result = response.json()

        # Verify both APIs were called
        mock_reason_client.reasoning.assert_called_once()
        mock_gen_client.generate.assert_called_once()

        # Should have natural response and LLM trust_delta
        assert not result["patient_reply"].startswith("Plan:")
        assert (
            result["patient_reply"]
            == "The nightmares have been really intense lately. Sometimes I wake up in a cold sweat from flashbacks."
        )
        assert result["state_updates"]["trust_delta"] == 0.1


@pytest.mark.anyio
async def test_pipeline_deepseek_reasoning_fallback(client):
    """
    Тест fallback при ошибке в DeepSeek reasoning.
    """
    # Setup: Create case and session
    case_data = {
        "case_truth": {
            "dx_target": ["OCD"],
            "ddx": {"OCD": 0.8},
            "hidden_facts": ["compulsive behaviors"],
            "red_flags": ["obsessive thoughts"],
            "trajectories": ["OCD treatment progress"],
        },
        "policies": {
            "disclosure_rules": {"min_trust_for_gated": 0.4},
            "risk_protocol": {
                "trigger_keywords": ["intrusive"],
                "response_style": "stable",
                "lock_topics": [],
            },
            "style_profile": {
                "register": "casual",
                "tempo": "medium",
                "length": "short",
            },
        },
    }

    case_response = await client.post("/case", json=case_data)
    assert case_response.status_code == 200
    case_id = case_response.json()["case_id"]

    session_data = {"case_id": case_id}
    session_response = await client.post("/session", json=session_data)
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]

    # Enable DeepSeek reasoning with failing client
    with patch.object(settings, "USE_DEEPSEEK_REASON", True), patch.object(
        settings, "USE_DEEPSEEK_GEN", False
    ), patch("app.orchestrator.nodes.reason_llm.DeepSeekClient") as mock_client_class:
        # Mock client that throws exception
        mock_client = MagicMock()
        mock_client.reasoning = AsyncMock(
            side_effect=Exception("API connection failed")
        )
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        # Make turn request
        turn_request = {
            "therapist_utterance": "Tell me about your thoughts",
            "session_id": session_id,
            "case_id": case_id,
            "session_state": {
                "trust": 0.5,
                "fatigue": 0.2,
                "affect": "anxious",
                "risk_status": "none",
                "access_level": 1,
                "last_turn_summary": "",
            },
        }

        response = await client.post("/turn", json=turn_request)
        assert response.status_code == 200

        result = response.json()

        # Should fallback to stub reasoning and still work
        assert "patient_reply" in result
        assert result["patient_reply"].startswith("Plan:")
        assert "state_updates" in result
        assert "eval_markers" in result


@pytest.mark.anyio
async def test_pipeline_deepseek_generation_fallback(client):
    """
    Тест fallback при ошибке в DeepSeek generation.
    """
    # Setup: Create case and session
    case_data = {
        "case_truth": {
            "dx_target": ["BPD"],
            "ddx": {"BPD": 0.7},
            "hidden_facts": ["emotional instability"],
            "red_flags": ["mood swings"],
            "trajectories": ["BPD therapy progress"],
        },
        "policies": {
            "disclosure_rules": {"min_trust_for_gated": 0.4},
            "risk_protocol": {
                "trigger_keywords": ["self-harm"],
                "response_style": "stable",
                "lock_topics": [],
            },
            "style_profile": {"register": "casual", "tempo": "fast", "length": "long"},
        },
    }

    case_response = await client.post("/case", json=case_data)
    assert case_response.status_code == 200
    case_id = case_response.json()["case_id"]

    session_data = {"case_id": case_id}
    session_response = await client.post("/session", json=session_data)
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]

    # Enable DeepSeek generation with failing client
    with patch.object(settings, "USE_DEEPSEEK_REASON", False), patch.object(
        settings, "USE_DEEPSEEK_GEN", True
    ), patch("app.orchestrator.nodes.generate_llm.DeepSeekClient") as mock_client_class:
        # Mock client that throws exception
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(
            side_effect=Exception("Generation service unavailable")
        )
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        # Make turn request
        turn_request = {
            "therapist_utterance": "How are your emotions today?",
            "session_id": session_id,
            "case_id": case_id,
            "session_state": {
                "trust": 0.4,
                "fatigue": 0.3,
                "affect": "volatile",
                "risk_status": "none",
                "access_level": 1,
                "last_turn_summary": "",
            },
        }

        response = await client.post("/turn", json=turn_request)
        assert response.status_code == 200

        result = response.json()

        # Should fallback to Plan: format
        assert "patient_reply" in result
        assert result["patient_reply"].startswith("Plan:")
        assert "intent=" in result["patient_reply"]
        assert "state_updates" in result
        assert "eval_markers" in result


@pytest.mark.anyio
async def test_pipeline_default_behavior_unchanged(client):
    """
    Тест что поведение по умолчанию (флаги выключены) не изменилось.
    """
    # Setup: Create case and session
    case_data = {
        "case_truth": {
            "dx_target": ["MDD"],
            "ddx": {"MDD": 0.6, "GAD": 0.4},
            "hidden_facts": ["depression episode"],
            "red_flags": ["low mood"],
            "trajectories": ["treatment response tracking"],
        },
        "policies": {
            "disclosure_rules": {"min_trust_for_gated": 0.4},
            "risk_protocol": {
                "trigger_keywords": ["suicide"],
                "response_style": "stable",
                "lock_topics": [],
            },
            "style_profile": {
                "register": "formal",
                "tempo": "medium",
                "length": "medium",
            },
        },
    }

    case_response = await client.post("/case", json=case_data)
    assert case_response.status_code == 200
    case_id = case_response.json()["case_id"]

    session_data = {"case_id": case_id}
    session_response = await client.post("/session", json=session_data)
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]

    # Ensure both flags are disabled (default)
    with patch.object(settings, "USE_DEEPSEEK_REASON", False), patch.object(
        settings, "USE_DEEPSEEK_GEN", False
    ):
        # Make turn request
        turn_request = {
            "therapist_utterance": "How are you feeling?",
            "session_id": session_id,
            "case_id": case_id,
            "session_state": {
                "trust": 0.5,
                "fatigue": 0.2,
                "affect": "neutral",
                "risk_status": "none",
                "access_level": 1,
                "last_turn_summary": "",
            },
        }

        response = await client.post("/turn", json=turn_request)
        assert response.status_code == 200

        result = response.json()

        # Should have traditional Plan: format
        assert result["patient_reply"].startswith("Plan:")
        assert "intent=" in result["patient_reply"]
        assert "risk=" in result["patient_reply"]

        # Should have all expected fields
        assert "state_updates" in result
        assert "used_fragments" in result
        assert "risk_status" in result
        assert "eval_markers" in result
        assert "intent" in result["eval_markers"]
