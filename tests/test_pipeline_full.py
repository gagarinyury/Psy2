import pytest


@pytest.mark.anyio
async def test_turn_normal_flow(client):
    """Test complete pipeline: normal flow with full normalize→retrieve→reason→guard"""

    # Create case with policies and case truth
    case_data = {
        "case_truth": {
            "dx_target": ["MDD"],
            "ddx": {"MDD": 0.6, "GAD": 0.3, "AUD": 0.1},
            "hidden_facts": ["эпизод в 2016", "семейная история депрессии"],
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

    # Create case and session through API
    case_response = await client.post("/case", json=case_data)
    assert case_response.status_code == 200, f"Case creation failed: {case_response.text}"
    case_id = case_response.json()["case_id"]

    session_data = {"case_id": case_id}
    session_response = await client.post("/session", json=session_data)
    assert session_response.status_code == 200, f"Session creation failed: {session_response.text}"
    session_id = session_response.json()["session_id"]

    # Test turn: "Как вы спите последние недели?"
    turn_data = {
        "therapist_utterance": "Как вы спите последние недели?",
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

    response = await client.post("/turn", json=turn_data)
    assert response.status_code == 200, f"Turn request failed: {response.text}"

    data = response.json()

    # Validate response structure and content
    assert "patient_reply" in data
    assert "Plan:" in data["patient_reply"], "patient_reply should contain 'Plan:'"
    assert "eval_markers" in data
    assert "intent" in data["eval_markers"]
    assert data["eval_markers"]["intent"] == "clarify", (
        f"Expected intent 'clarify', got {data['eval_markers']['intent']}"
    )
    assert data["risk_status"] == "none", f"Expected risk_status 'none', got {data['risk_status']}"
    assert "used_fragments" in data
    # Note: used_fragments can be empty if no KB fragments are available for the case
    assert isinstance(data["used_fragments"], list), "used_fragments should be a list"


@pytest.mark.anyio
async def test_turn_risk_flow(client):
    """Test complete pipeline: risk flow with suicide-related question"""

    # Create case with policies including suicide triggers
    case_data = {
        "case_truth": {
            "dx_target": ["MDD"],
            "ddx": {"MDD": 0.6, "GAD": 0.3, "AUD": 0.1},
            "hidden_facts": ["эпизод в 2016"],
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

    # Create case and session through API
    case_response = await client.post("/case", json=case_data)
    assert case_response.status_code == 200, f"Case creation failed: {case_response.text}"
    case_id = case_response.json()["case_id"]

    session_data = {"case_id": case_id}
    session_response = await client.post("/session", json=session_data)
    assert session_response.status_code == 200, f"Session creation failed: {session_response.text}"
    session_id = session_response.json()["session_id"]

    # Test turn: "Бывают ли мысли о суициде?"
    turn_data = {
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

    response = await client.post("/turn", json=turn_data)
    assert response.status_code == 200, f"Risk turn request failed: {response.text}"

    data = response.json()

    # Validate risk flow results
    assert data["risk_status"] == "acute", (
        f"Expected risk_status 'acute', got {data['risk_status']}"
    )
    assert "eval_markers" in data
    assert "intent" in data["eval_markers"]
    assert data["eval_markers"]["intent"] == "risk_check", (
        f"Expected intent 'risk_check', got {data['eval_markers']['intent']}"
    )
    assert "patient_reply" in data
    assert "Plan:1" in data["patient_reply"], (
        "patient_reply should contain 'Plan:1' (guard replaced content)"
    )


@pytest.mark.anyio
async def test_turn_fallback_on_error(monkeypatch, client):
    """Test pipeline fallback behavior when normalize throws exception"""

    # Create case and session
    case_data = {
        "case_truth": {
            "dx_target": ["MDD"],
            "ddx": {"MDD": 0.6, "GAD": 0.3, "AUD": 0.1},
            "hidden_facts": ["эпизод в 2016"],
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
                "trigger_keywords": ["суицид"],
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

    case_response = await client.post("/case", json=case_data)
    assert case_response.status_code == 200
    case_id = case_response.json()["case_id"]

    session_data = {"case_id": case_id}
    session_response = await client.post("/session", json=session_data)
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]

    # Mock normalize to raise exception
    def mock_normalize_error(utterance, session_state):
        raise RuntimeError("Normalize failed for testing")

    monkeypatch.setattr("app.orchestrator.pipeline.normalize", mock_normalize_error)

    # Test turn that should trigger fallback
    turn_data = {
        "therapist_utterance": "Test utterance",
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

    response = await client.post("/turn", json=turn_data)
    assert response.status_code == 200, f"Fallback turn request failed: {response.text}"

    data = response.json()

    # Validate fallback response
    assert data["patient_reply"] == "safe-fallback", (
        f"Expected 'safe-fallback', got {data['patient_reply']}"
    )
    assert data["risk_status"] == "none", f"Expected risk_status 'none', got {data['risk_status']}"
    assert data["state_updates"] == {}, "state_updates should be empty dict in fallback"
    assert data["used_fragments"] == [], "used_fragments should be empty list in fallback"
    assert data["eval_markers"] == {}, "eval_markers should be empty dict in fallback"
