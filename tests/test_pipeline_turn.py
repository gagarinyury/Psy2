import pytest


@pytest.mark.anyio
async def test_pipeline_turn_integration(client):
    """Test complete pipeline integration: normal phrase and risk phrase scenarios"""

    # Setup: Create case and session using API instead of CLI
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

    # Create case
    case_response = await client.post("/case", json=case_data)
    assert (
        case_response.status_code == 200
    ), f"Case creation failed: {case_response.text}"
    case_id = case_response.json()["case_id"]

    # Create session for the case
    session_data = {"case_id": case_id}
    session_response = await client.post("/session", json=session_data)
    assert (
        session_response.status_code == 200
    ), f"Session creation failed: {session_response.text}"
    session_id = session_response.json()["session_id"]

    # Test 1: Normal phrase about sleep
    normal_turn_data = {
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

    response = await client.post("/turn", json=normal_turn_data)
    assert response.status_code == 200, f"Normal turn failed: {response.text}"

    turn_response = response.json()

    # Validate Test 1 requirements
    assert "patient_reply" in turn_response
    assert (
        "Echo:" in turn_response["patient_reply"]
    ), "patient_reply should contain 'Echo:'"
    assert (
        turn_response["risk_status"] == "none"
    ), f"Expected risk_status 'none', got {turn_response['risk_status']}"
    assert "used_fragments" in turn_response
    assert isinstance(
        turn_response["used_fragments"], list
    ), "used_fragments should be a list"

    # Validate response structure
    assert "state_updates" in turn_response
    assert "eval_markers" in turn_response

    # Test 2: Risk phrase about suicide
    risk_turn_data = {
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

    response = await client.post("/turn", json=risk_turn_data)
    assert response.status_code == 200, f"Risk turn failed: {response.text}"

    turn_response = response.json()

    # Validate Test 2 requirements
    assert (
        turn_response["risk_status"] == "acute"
    ), f"Expected risk_status 'acute', got {turn_response['risk_status']}"
    assert "eval_markers" in turn_response
    assert "intent" in turn_response["eval_markers"]
    assert (
        turn_response["eval_markers"]["intent"] == "risk_check"
    ), f"Expected intent 'risk_check', got {turn_response['eval_markers']['intent']}"

    # Validate response structure for risk scenario
    assert "patient_reply" in turn_response
    assert "used_fragments" in turn_response
    assert "state_updates" in turn_response


@pytest.mark.anyio
async def test_pipeline_turn_sleep_scenario(client):
    """Test 1: Normal phrase about sleep - detailed validation"""

    # Create test case with sleep-related fragment
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

    # Test normal phrase about sleep
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
    assert response.status_code == 200

    data = response.json()

    # Detailed validation
    assert data["patient_reply"].startswith(
        "Echo:"
    ), "patient_reply should start with 'Echo:'"
    assert data["risk_status"] == "none"
    assert isinstance(data["used_fragments"], list)
    assert isinstance(data["state_updates"], dict)
    assert isinstance(data["eval_markers"], dict)
    assert "intent" in data["eval_markers"]
    # Intent should be clarify for sleep question
    assert data["eval_markers"]["intent"] == "clarify"


@pytest.mark.anyio
async def test_pipeline_turn_risk_scenario(client):
    """Test 2: Risk phrase about suicide - detailed validation"""

    # Create test case
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

    case_response = await client.post("/case", json=case_data)
    assert case_response.status_code == 200
    case_id = case_response.json()["case_id"]

    session_data = {"case_id": case_id}
    session_response = await client.post("/session", json=session_data)
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]

    # Test risk phrase
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
    assert response.status_code == 200

    data = response.json()

    # Detailed validation for risk scenario
    assert data["risk_status"] == "acute"
    assert data["eval_markers"]["intent"] == "risk_check"
    assert isinstance(data["patient_reply"], str)
    assert len(data["patient_reply"]) > 0
    assert isinstance(data["used_fragments"], list)
    assert isinstance(data["state_updates"], dict)


@pytest.mark.anyio
async def test_pipeline_turn_response_structure(client):
    """Test TurnResponse structure validation"""

    # Create test case
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

    # Test with neutral utterance
    turn_data = {
        "therapist_utterance": "Расскажите о себе",
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
    assert response.status_code == 200

    data = response.json()

    # Validate complete TurnResponse structure
    required_fields = [
        "patient_reply",
        "state_updates",
        "used_fragments",
        "risk_status",
        "eval_markers",
    ]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"

    # Validate field types
    assert isinstance(data["patient_reply"], str)
    assert isinstance(data["state_updates"], dict)
    assert isinstance(data["used_fragments"], list)
    assert isinstance(data["risk_status"], str)
    assert isinstance(data["eval_markers"], dict)

    # Validate risk_status is valid value
    assert data["risk_status"] in ["none", "acute"]

    # Validate eval_markers has intent
    assert "intent" in data["eval_markers"]
    assert isinstance(data["eval_markers"]["intent"], str)
