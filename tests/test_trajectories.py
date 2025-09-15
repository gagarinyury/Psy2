"""
Tests for trajectory progress tracking and API endpoints.

Tests the trajectory step completion logic, session trajectory progress tracking,
and GET /session/{session_id}/trajectory endpoint.
"""

import pytest

from app.core.tables import KBFragment, SessionTrajectory, TelemetryTurn


@pytest.mark.anyio
async def test_session_trajectory_progress(client, db_session):
    """Test trajectory progress tracking through API."""
    # Create test case with trajectories
    case_data = {
        "case_truth": {
            "dx_target": ["insomnia"],
            "ddx": {"insomnia": 0.8},
            "red_flags": [],
            "hidden_facts": [],
            "trajectories": [
                {
                    "id": "t1",
                    "name": "Test Trajectory",
                    "steps": [
                        {
                            "id": "sleep",
                            "name": "Sleep Step",
                            "condition_tags": ["sleep"],
                            "min_trust": 0.3
                        },
                        {
                            "id": "mood",
                            "name": "Mood Step",
                            "condition_tags": ["mood"],
                            "min_trust": 0.5
                        }
                    ]
                }
            ]
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
                "register": "colloquial",
                "tempo": "medium",
                "length": "short",
            },
        }
    }

    case_response = await client.post("/case", json=case_data)
    assert case_response.status_code == 200
    case_id = case_response.json()["case_id"]

    # Create session
    session_data = {"case_id": case_id}
    session_response = await client.post("/session", json=session_data)
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]

    # Create KB fragments with appropriate metadata tags
    sleep_fragment = KBFragment(
        case_id=case_id,
        type="symptom",
        text="Patient reports trouble sleeping",
        fragment_metadata={"tags": ["sleep"], "topic": "sleep_issues"},
        availability="public",
        consistency_keys={},
    )

    mood_fragment = KBFragment(
        case_id=case_id,
        type="symptom",
        text="Patient shows signs of mood changes",
        fragment_metadata={"tags": ["mood"], "topic": "mood_issues"},
        availability="public",
        consistency_keys={},
    )

    db_session.add(sleep_fragment)
    db_session.add(mood_fragment)
    await db_session.flush()

    # Turn 1: trust=0.4, use sleep fragment -> sleep step completed
    telemetry_turn1 = TelemetryTurn(
        session_id=session_id,
        turn_no=1,
        used_fragments=[str(sleep_fragment.id)],
        risk_status="none",
        eval_markers={"intent": "assess_sleep"},
        timings={},
        costs={},
    )
    db_session.add(telemetry_turn1)

    # Turn 2: trust=0.55, use mood fragment -> mood step completed
    telemetry_turn2 = TelemetryTurn(
        session_id=session_id,
        turn_no=2,
        used_fragments=[str(mood_fragment.id)],
        risk_status="none",
        eval_markers={"intent": "assess_mood"},
        timings={},
        costs={},
    )
    db_session.add(telemetry_turn2)

    # Create session trajectory record with completed steps
    session_trajectory = SessionTrajectory(
        session_id=session_id,
        trajectory_id="t1",
        completed_steps=["sleep", "mood"],  # Both steps completed
    )
    db_session.add(session_trajectory)

    await db_session.commit()

    # Test GET /session/{session_id}/trajectory endpoint
    trajectory_response = await client.get(f"/session/{session_id}/trajectory")
    assert trajectory_response.status_code == 200

    trajectory_data = trajectory_response.json()

    # Verify response structure
    assert "session_id" in trajectory_data
    assert "progress" in trajectory_data
    assert trajectory_data["session_id"] == session_id

    # Verify trajectory progress data
    progress = trajectory_data["progress"]
    assert len(progress) == 1  # One trajectory

    trajectory_progress = progress[0]
    assert trajectory_progress["trajectory_id"] == "t1"
    assert trajectory_progress["total"] == 2  # Two steps total

    # Both steps should be completed
    completed_steps = trajectory_progress["completed_steps"]
    assert len(completed_steps) == 2
    assert "sleep" in completed_steps
    assert "mood" in completed_steps


@pytest.mark.anyio
async def test_session_trajectory_partial_completion(client, db_session):
    """Test trajectory with partial step completion."""
    # Create case with trajectory
    case_data = {
        "case_truth": {
            "dx_target": ["anxiety"],
            "ddx": {"anxiety": 0.7},
            "red_flags": [],
            "hidden_facts": [],
            "trajectories": [
                {
                    "id": "partial_t1",
                    "name": "Partial Test Trajectory",
                    "steps": [
                        {
                            "id": "low_trust_step",
                            "name": "Low Trust Step",
                            "condition_tags": ["anxiety"],
                            "min_trust": 0.2
                        },
                        {
                            "id": "high_trust_step",
                            "name": "High Trust Step",
                            "condition_tags": ["anxiety"],
                            "min_trust": 0.8
                        }
                    ]
                }
            ]
        },
        "policies": {
            "disclosure_rules": {
                "full_on_valid_question": True,
                "partial_if_low_trust": True,
                "min_trust_for_gated": 0.4,
            },
            "distortion_rules": {"enabled": True, "by_defense": {}},
            "risk_protocol": {
                "trigger_keywords": ["anxiety"],
                "response_style": "stable",
                "lock_topics": [],
            },
            "style_profile": {
                "register": "colloquial",
                "tempo": "medium",
                "length": "short",
            },
        }
    }

    case_response = await client.post("/case", json=case_data)
    assert case_response.status_code == 200
    case_id = case_response.json()["case_id"]

    session_data = {"case_id": case_id}
    session_response = await client.post("/session", json=session_data)
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]

    # Create KB fragment with anxiety tag
    anxiety_fragment = KBFragment(
        case_id=case_id,
        type="symptom",
        text="Patient shows anxiety symptoms",
        fragment_metadata={"tags": ["anxiety"], "topic": "anxiety_assessment"},
        availability="public",
        consistency_keys={},
    )

    db_session.add(anxiety_fragment)

    # Create session trajectory record with only low trust step completed
    session_trajectory = SessionTrajectory(
        session_id=session_id,
        trajectory_id="partial_t1",
        completed_steps=["low_trust_step"],  # Only low trust step completed
    )
    db_session.add(session_trajectory)

    await db_session.commit()

    # Test trajectory endpoint
    trajectory_response = await client.get(f"/session/{session_id}/trajectory")
    assert trajectory_response.status_code == 200

    trajectory_data = trajectory_response.json()
    progress = trajectory_data["progress"]

    assert len(progress) == 1
    trajectory_progress = progress[0]
    assert trajectory_progress["trajectory_id"] == "partial_t1"
    assert trajectory_progress["total"] == 2

    # Only low trust step should be completed
    completed_steps = trajectory_progress["completed_steps"]
    assert len(completed_steps) == 1
    assert "low_trust_step" in completed_steps
    assert "high_trust_step" not in completed_steps


@pytest.mark.anyio
async def test_session_trajectory_no_matching_tags(client, db_session):
    """Test trajectory with no matching fragment tags."""
    case_data = {
        "case_truth": {
            "dx_target": ["depression"],
            "ddx": {"depression": 0.9},
            "red_flags": [],
            "hidden_facts": [],
            "trajectories": [
                {
                    "id": "no_match_t1",
                    "name": "No Match Trajectory",
                    "steps": [
                        {
                            "id": "specific_step",
                            "name": "Specific Step",
                            "condition_tags": ["very_specific_tag"],
                            "min_trust": 0.1
                        }
                    ]
                }
            ]
        },
        "policies": {
            "disclosure_rules": {
                "full_on_valid_question": True,
                "partial_if_low_trust": True,
                "min_trust_for_gated": 0.4,
            },
            "distortion_rules": {"enabled": True, "by_defense": {}},
            "risk_protocol": {
                "trigger_keywords": ["anxiety"],
                "response_style": "stable",
                "lock_topics": [],
            },
            "style_profile": {
                "register": "colloquial",
                "tempo": "medium",
                "length": "short",
            },
        }
    }

    case_response = await client.post("/case", json=case_data)
    assert case_response.status_code == 200
    case_id = case_response.json()["case_id"]

    session_data = {"case_id": case_id}
    session_response = await client.post("/session", json=session_data)
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]

    # Create KB fragment with different tags
    different_fragment = KBFragment(
        case_id=case_id,
        type="symptom",
        text="Different symptom information",
        fragment_metadata={"tags": ["different_tag"], "topic": "different_topic"},
        availability="public",
        consistency_keys={},
    )

    db_session.add(different_fragment)
    await db_session.commit()

    # No session trajectory created since no steps should be completed
    # (fragment tags don't match trajectory step condition tags)

    # Test trajectory endpoint
    trajectory_response = await client.get(f"/session/{session_id}/trajectory")
    assert trajectory_response.status_code == 200

    trajectory_data = trajectory_response.json()
    progress = trajectory_data["progress"]

    # Should have trajectory record but no completed steps
    assert len(progress) == 0  # No session trajectory created since no steps completed


@pytest.mark.anyio
async def test_session_trajectory_endpoint_session_not_found(client):
    """Test trajectory endpoint with non-existent session."""
    fake_session_id = "12345678-1234-1234-1234-123456789abc"

    response = await client.get(f"/session/{fake_session_id}/trajectory")
    assert response.status_code == 404
    assert "Session not found" in response.json()["detail"]


@pytest.mark.anyio
async def test_session_trajectory_endpoint_invalid_session_id(client):
    """Test trajectory endpoint with invalid session ID format."""
    invalid_session_id = "not-a-valid-uuid"

    response = await client.get(f"/session/{invalid_session_id}/trajectory")
    assert response.status_code == 400
    assert "Invalid session_id format" in response.json()["detail"]


@pytest.mark.anyio
async def test_session_trajectory_no_trajectories(client):
    """Test trajectory endpoint for session with no trajectories in case."""
    # Create case without trajectories
    case_data = {
        "case_truth": {
            "dx_target": ["no_trajectory_case"],
            "ddx": {"no_trajectory_case": 1.0},
            "red_flags": [],
            "hidden_facts": [],
            "trajectories": []  # Empty trajectories
        },
        "policies": {
            "disclosure_rules": {
                "full_on_valid_question": True,
                "partial_if_low_trust": True,
                "min_trust_for_gated": 0.4,
            },
            "distortion_rules": {"enabled": True, "by_defense": {}},
            "risk_protocol": {
                "trigger_keywords": ["anxiety"],
                "response_style": "stable",
                "lock_topics": [],
            },
            "style_profile": {
                "register": "colloquial",
                "tempo": "medium",
                "length": "short",
            },
        }
    }

    case_response = await client.post("/case", json=case_data)
    assert case_response.status_code == 200
    case_id = case_response.json()["case_id"]

    session_data = {"case_id": case_id}
    session_response = await client.post("/session", json=session_data)
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]

    # Test trajectory endpoint
    trajectory_response = await client.get(f"/session/{session_id}/trajectory")
    assert trajectory_response.status_code == 200

    trajectory_data = trajectory_response.json()
    assert trajectory_data["session_id"] == session_id
    assert trajectory_data["progress"] == []  # Empty progress list