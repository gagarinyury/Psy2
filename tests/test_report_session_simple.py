"""
Simplified tests for session report API endpoint.

Tests basic functionality without complex database setup.
"""

import pytest


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
async def test_get_session_report_basic_structure(client):
    """Test report structure for an existing session."""
    # Create case via API
    case_data = {
        "case_truth": {
            "dx_target": ["MDD"],
            "ddx": {"MDD": 0.8},
        },
        "policies": {
            "disclosure_rules": {"min_trust_for_gated": 0.4},
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

    # Basic validation - should be numbers
    assert isinstance(metrics["recall_keys"], (int, float))
    assert isinstance(metrics["risk_timeliness"], (int, float))
    assert isinstance(metrics["turns_total"], int)
    assert isinstance(metrics["used_fragments_total"], int)
    assert isinstance(metrics["key_fragments_total"], int)
