"""
Tests for session report API endpoint.

Tests the GET /report/session/{id} endpoint including
successful report generation and error handling.
"""

import pytest

from app.core.tables import KBFragment, TelemetryTurn


@pytest.mark.anyio
async def test_get_session_report_success(client, db_session):
    """Test successful session report generation via API."""
    # Create test case via API
    case_data = {
        "case_truth": {
            "dx_target": ["MDD"],
            "ddx": {"MDD": 0.8},
            "red_flags": ["depression", "suicidal thoughts"],
            "hidden_facts": ["previous episode"],
        },
        "policies": {
            "disclosure_rules": {"min_trust_for_gated": 0.4},
            "risk_protocol": {"trigger_keywords": ["suicide"]},
        },
    }

    case_response = await client.post("/case", json=case_data)
    assert case_response.status_code == 200
    case_id = case_response.json()["case_id"]

    # Create session via API
    session_data = {"case_id": case_id}
    session_response = await client.post("/session", json=session_data)
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]

    # Create key fragments and telemetry data directly via database
    # (since we don't have API endpoints for these yet)
    case_uuid = case_response.json()["case_id"]
    session_uuid = session_response.json()["session_id"]

    # Add key fragment to database
    key_fragment = KBFragment(
        case_id=case_uuid,
        type="key_info",
        text="Important patient information",
        fragment_metadata={"tags": ["key"], "topic": "patient_history"},
        availability="public",
        consistency_keys={},
    )
    db_session.add(key_fragment)
    await db_session.flush()

    # Add telemetry turns
    telemetry_turns = [
        {
            "turn_no": 1,
            "used_fragments": [],
            "risk_status": "none",
            "eval_markers": {"intent": "greeting"},
        },
        {
            "turn_no": 2,
            "used_fragments": [str(key_fragment.id)],
            "risk_status": "acute",  # Risk detected on turn 2
            "eval_markers": {"intent": "risk_check"},
        },
    ]

    for turn_data in telemetry_turns:
        turn = TelemetryTurn(
            session_id=session_uuid,
            turn_no=turn_data["turn_no"],
            used_fragments=turn_data["used_fragments"],
            risk_status=turn_data["risk_status"],
            eval_markers=turn_data["eval_markers"],
            timings={},
            costs={},
        )
        db_session.add(turn)

    await db_session.commit()

    # Test the report API endpoint
    report_response = await client.get(f"/report/session/{session_id}")
    assert report_response.status_code == 200

    report_data = report_response.json()

    # Verify report structure
    assert "session_id" in report_data
    assert "case_id" in report_data
    assert "metrics" in report_data

    assert report_data["session_id"] == session_id
    assert report_data["case_id"] == case_id

    # Verify metrics content
    metrics = report_data["metrics"]
    assert "recall_keys" in metrics
    assert "risk_timeliness" in metrics
    assert "turns_total" in metrics
    assert "used_fragments_total" in metrics
    assert "key_fragments_total" in metrics

    # Verify specific values
    assert metrics["recall_keys"] == 1.0  # Used 1 of 1 key fragments
    assert metrics["risk_timeliness"] == 1.0  # Risk detected on turn 2 (≤3)
    assert metrics["turns_total"] == 2
    assert metrics["used_fragments_total"] == 1
    assert metrics["key_fragments_total"] == 1


@pytest.mark.anyio
async def test_get_session_report_session_not_found(client):
    """Test 404 response for non-existent session."""
    fake_session_id = "12345678-1234-1234-1234-123456789abc"

    response = await client.get(f"/report/session/{fake_session_id}")
    assert response.status_code == 404
    assert "Session not found" in response.json()["detail"]


@pytest.mark.anyio
async def test_get_session_report_invalid_session_id(client):
    """Test 400 response for invalid session ID format."""
    invalid_session_id = "not-a-valid-uuid"

    response = await client.get(f"/report/session/{invalid_session_id}")
    assert response.status_code == 400
    assert "Invalid session_id format" in response.json()["detail"]


@pytest.mark.anyio
async def test_get_session_report_empty_session(client, db_session):
    """Test report generation for session with no turns."""
    # Create case and session
    case_data = {
        "case_truth": {"dx_target": ["GAD"]},
        "policies": {},
    }

    case_response = await client.post("/case", json=case_data)
    assert case_response.status_code == 200

    session_data = {"case_id": case_response.json()["case_id"]}
    session_response = await client.post("/session", json=session_data)
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]

    # No telemetry turns created - session is empty
    await db_session.commit()

    # Test report generation
    response = await client.get(f"/report/session/{session_id}")
    assert response.status_code == 200

    report_data = response.json()
    metrics = report_data["metrics"]

    # Empty session should have zero turns
    assert metrics["turns_total"] == 0
    assert metrics["used_fragments_total"] == 0
    assert metrics["recall_keys"] in [0.0, 1.0]  # Depends on key fragments existence
    assert metrics["risk_timeliness"] in [0.0, 1.0]  # Depends on red_flags existence


@pytest.mark.anyio
async def test_get_session_report_comprehensive_metrics(client, db_session):
    """Test report with comprehensive metrics validation."""
    # Create case with multiple key fragments
    case_data = {
        "case_truth": {
            "dx_target": ["PTSD"],
            "red_flags": ["trauma", "nightmares"],
            "hidden_facts": ["military service", "combat exposure"],
        },
        "policies": {
            "disclosure_rules": {"min_trust_for_gated": 0.5},
            "risk_protocol": {"trigger_keywords": ["trauma", "flashback"]},
        },
    }

    case_response = await client.post("/case", json=case_data)
    assert case_response.status_code == 200
    case_id = case_response.json()["case_id"]

    session_data = {"case_id": case_id}
    session_response = await client.post("/session", json=session_data)
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]

    # Create multiple key fragments
    key_fragment1 = KBFragment(
        case_id=case_id,
        type="trauma_history",
        text="Patient has combat-related PTSD",
        fragment_metadata={"tags": ["hook", "trauma"], "topic": "trauma"},
        availability="gated",
        consistency_keys={},
    )

    key_fragment2 = KBFragment(
        case_id=case_id,
        type="symptoms",
        text="Experiences nightmares and flashbacks",
        fragment_metadata={"tags": ["key", "symptoms"], "topic": "symptoms"},
        availability="public",
        consistency_keys={},
    )

    regular_fragment = KBFragment(
        case_id=case_id,
        type="demographic",
        text="Patient is 45 years old veteran",
        fragment_metadata={"tags": ["demographic"], "topic": "basic"},
        availability="public",
        consistency_keys={},
    )

    db_session.add(key_fragment1)
    db_session.add(key_fragment2)
    db_session.add(regular_fragment)
    await db_session.flush()

    # Create telemetry with partial key usage and late risk detection
    turns_data = [
        {"turn_no": 1, "used_fragments": [], "risk_status": "none"},
        {
            "turn_no": 2,
            "used_fragments": [str(key_fragment1.id)],
            "risk_status": "none",
        },
        {
            "turn_no": 3,
            "used_fragments": [str(regular_fragment.id)],
            "risk_status": "none",
        },
        {"turn_no": 4, "used_fragments": [], "risk_status": "none"},
        {"turn_no": 5, "used_fragments": [], "risk_status": "acute"},  # Risk on turn 5
    ]

    for turn_data in turns_data:
        turn = TelemetryTurn(
            session_id=session_id,
            turn_no=turn_data["turn_no"],
            used_fragments=turn_data["used_fragments"],
            risk_status=turn_data["risk_status"],
            eval_markers={"intent": "clarify"},
            timings={},
            costs={},
        )
        db_session.add(turn)

    await db_session.commit()

    # Get report
    response = await client.get(f"/report/session/{session_id}")
    assert response.status_code == 200

    metrics = response.json()["metrics"]

    # Validate comprehensive metrics
    assert metrics["turns_total"] == 5
    assert metrics["key_fragments_total"] == 2  # 2 key fragments
    assert metrics["used_fragments_total"] == 2  # key_fragment1 + regular_fragment
    assert metrics["recall_keys"] == 0.5  # Used 1 of 2 key fragments
    assert metrics["risk_timeliness"] == 0.5  # Risk detected on turn 5 (4-6 range)
    assert metrics["first_acute_turn"] == 5

    # Verify fragment IDs
    assert str(key_fragment1.id) in metrics["used_key_ids"]
    assert str(key_fragment2.id) not in metrics["used_key_ids"]
    assert len(metrics["all_key_ids"]) == 2
    assert len(metrics["used_key_ids"]) == 1
