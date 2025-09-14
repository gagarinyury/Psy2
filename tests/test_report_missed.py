"""
Tests for missed keys functionality in session reports.

Tests that missed keys are properly tracked and reported
through the API endpoints.
"""

import pytest

from app.core.tables import KBFragment, TelemetryTurn


@pytest.mark.anyio
async def test_report_missed_keys_success(client, db_session):
    """Test missed keys reporting via API."""
    # Create test case via API
    case_data = {
        "case_truth": {
            "dx_target": ["MDD"],
            "ddx": {"MDD": 0.8},
            "red_flags": ["depression"],
            "hidden_facts": ["family history"],
            "trajectories": ["test"],
        },
        "policies": {
            "disclosure_rules": {"min_trust_for_gated": 0.4},
            "distortion_rules": {"enabled": True},
            "risk_protocol": {"trigger_keywords": ["suicide"]},
            "style_profile": {"register": "colloquial"},
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

    # Create 2 key fragments (kb1, kb2) directly via database
    kb1_fragment = KBFragment(
        case_id=case_id,
        type="key_info",
        text="Important patient information kb1",
        fragment_metadata={"tags": ["key"], "topic": "patient_history"},
        availability="public",
        consistency_keys={},
    )

    kb2_fragment = KBFragment(
        case_id=case_id,
        type="hook_info",
        text="Critical hook information kb2",
        fragment_metadata={"tags": ["hook"], "topic": "hooks"},
        availability="public",
        consistency_keys={},
    )

    db_session.add(kb1_fragment)
    db_session.add(kb2_fragment)
    await db_session.flush()

    # Add telemetry turn using only kb1
    turn = TelemetryTurn(
        session_id=session_id,
        turn_no=1,
        used_fragments=[str(kb1_fragment.id)],  # Only use kb1
        risk_status="none",
        eval_markers={"intent": "clarify"},
        timings={},
        costs={},
    )
    db_session.add(turn)
    await db_session.commit()

    # Test the main report API endpoint
    report_response = await client.get(f"/report/session/{session_id}")
    assert report_response.status_code == 200

    report_data = report_response.json()

    # Verify report structure
    assert "session_id" in report_data
    assert "case_id" in report_data
    assert "metrics" in report_data
    assert "missed_keys" in report_data

    # Verify missed_keys content
    missed_keys = report_data["missed_keys"]
    assert "ids" in missed_keys
    assert "count" in missed_keys
    assert missed_keys["count"] == 1
    assert str(kb2_fragment.id) in missed_keys["ids"]
    assert str(kb1_fragment.id) not in missed_keys["ids"]

    # Test the dedicated missed keys API endpoint
    missed_response = await client.get(f"/report/session/{session_id}/missed")
    assert missed_response.status_code == 200

    missed_data = missed_response.json()

    # Verify missed endpoint structure and content
    assert "session_id" in missed_data
    assert "case_id" in missed_data
    assert "missed_key_ids" in missed_data
    assert "count" in missed_data

    assert missed_data["session_id"] == session_id
    assert missed_data["case_id"] == case_id
    assert missed_data["count"] == 1
    assert str(kb2_fragment.id) in missed_data["missed_key_ids"]
    assert str(kb1_fragment.id) not in missed_data["missed_key_ids"]


@pytest.mark.anyio
async def test_missed_keys_endpoint_not_found(client):
    """Test 404 response for non-existent session on missed keys endpoint."""
    fake_session_id = "12345678-1234-1234-1234-123456789abc"

    response = await client.get(f"/report/session/{fake_session_id}/missed")
    assert response.status_code == 404
    assert "Session not found" in response.json()["detail"]


@pytest.mark.anyio
async def test_missed_keys_endpoint_invalid_session_id(client):
    """Test 400 response for invalid session ID format on missed keys endpoint."""
    invalid_session_id = "not-a-valid-uuid"

    response = await client.get(f"/report/session/{invalid_session_id}/missed")
    assert response.status_code == 400
    assert "Invalid session_id format" in response.json()["detail"]