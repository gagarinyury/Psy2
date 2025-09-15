"""
Tests for multi-session trajectory reporting functionality.

Tests the scenario where multiple sessions contribute to a single case trajectory:
1. Two sessions for the same case
2. First session completes "sleep" step
3. Second session completes "mood" step
4. POST /session/link to link sessions in a chain
5. GET /report/case/{case_id}/trajectories should return coverage == 1.0, union steps [sleep,mood]
6. compute_session_metrics should contain trajectory_progress with correct numbers
"""

import uuid

import pytest

from app.core.tables import KBFragment, SessionTrajectory, TelemetryTurn
from app.eval.metrics import compute_session_metrics


@pytest.mark.anyio
async def test_multi_session_trajectory_report(client, db_session):
    """Test multi-session trajectory completion and reporting."""
    # 1. Create case with trajectory t1 having steps: sleep(min_trust=0.3, tags=["sleep"]), mood(0.5, ["mood"])
    case_data = {
        "case_truth": {
            "dx_target": ["insomnia"],
            "ddx": {"insomnia": 0.8},
            "red_flags": [],
            "hidden_facts": [],
            "trajectories": [
                {
                    "id": "t1",
                    "name": "Sleep-Mood Trajectory",
                    "steps": [
                        {
                            "id": "sleep",
                            "name": "Sleep Assessment",
                            "condition_tags": ["sleep"],
                            "min_trust": 0.3
                        },
                        {
                            "id": "mood",
                            "name": "Mood Assessment",
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

    # 2. Create session1 and session2 for the same case
    session_data = {"case_id": case_id}

    session1_response = await client.post("/session", json=session_data)
    assert session1_response.status_code == 200
    session1_id = session1_response.json()["session_id"]

    session2_response = await client.post("/session", json=session_data)
    assert session2_response.status_code == 200
    session2_id = session2_response.json()["session_id"]

    # 3. Create KB fragments with metadata tags ["sleep"] and ["mood"]
    sleep_fragment = KBFragment(
        case_id=case_id,
        type="symptom",
        text="Patient reports difficulty falling asleep",
        fragment_metadata={"tags": ["sleep"], "topic": "sleep_issues"},
        availability="public",
        consistency_keys={},
    )

    mood_fragment = KBFragment(
        case_id=case_id,
        type="symptom",
        text="Patient shows signs of depressed mood",
        fragment_metadata={"tags": ["mood"], "topic": "mood_issues"},
        availability="public",
        consistency_keys={},
    )

    db_session.add(sleep_fragment)
    db_session.add(mood_fragment)
    await db_session.flush()

    # 4. Make turn calls in session1 to complete sleep step
    # Turn 1 in session1: trust=0.4 (>= 0.3), use sleep fragment -> sleep step completed
    turn1_data = {
        "therapist_utterance": "How has your sleep been lately?",
        "session_state": {
            "affect": "neutral",
            "trust": 0.4,  # >= min_trust for sleep step (0.3)
            "fatigue": 0.1,
            "access_level": 1,
            "risk_status": "none",
            "last_turn_summary": "",
        },
        "case_id": case_id,
        "session_id": session1_id,
        "options": {},
    }

    turn1_response = await client.post("/turn", json=turn1_data)
    assert turn1_response.status_code == 200

    # Manually create telemetry turn for session1 with sleep fragment usage
    telemetry_turn1 = TelemetryTurn(
        session_id=uuid.UUID(session1_id),
        turn_no=1,
        used_fragments=[str(sleep_fragment.id)],
        risk_status="none",
        eval_markers={"intent": "assess_sleep"},
        timings={},
        costs={},
    )
    db_session.add(telemetry_turn1)

    # Create session trajectory record for session1 with completed sleep step
    session_trajectory1 = SessionTrajectory(
        session_id=uuid.UUID(session1_id),
        trajectory_id="t1",
        completed_steps=["sleep"],  # Sleep step completed
    )
    db_session.add(session_trajectory1)

    # 5. Make turn calls in session2 to complete mood step
    # Turn 1 in session2: trust=0.6 (>= 0.5), use mood fragment -> mood step completed
    turn2_data = {
        "therapist_utterance": "How are you feeling emotionally?",
        "session_state": {
            "affect": "neutral",
            "trust": 0.6,  # >= min_trust for mood step (0.5)
            "fatigue": 0.1,
            "access_level": 1,
            "risk_status": "none",
            "last_turn_summary": "",
        },
        "case_id": case_id,
        "session_id": session2_id,
        "options": {},
    }

    turn2_response = await client.post("/turn", json=turn2_data)
    assert turn2_response.status_code == 200

    # Manually create telemetry turn for session2 with mood fragment usage
    telemetry_turn2 = TelemetryTurn(
        session_id=uuid.UUID(session2_id),
        turn_no=1,
        used_fragments=[str(mood_fragment.id)],
        risk_status="none",
        eval_markers={"intent": "assess_mood"},
        timings={},
        costs={},
    )
    db_session.add(telemetry_turn2)

    # Create session trajectory record for session2 with completed mood step
    session_trajectory2 = SessionTrajectory(
        session_id=uuid.UUID(session2_id),
        trajectory_id="t1",
        completed_steps=["mood"],  # Mood step completed
    )
    db_session.add(session_trajectory2)

    await db_session.commit()

    # 6. POST /session/link to link the sessions
    # First, link session1 to the case (no previous session)
    link_data1 = {
        "session_id": session1_id,
        "case_id": case_id,
        "prev_session_id": None,
    }

    link_response1 = await client.post("/session/link", json=link_data1)
    assert link_response1.status_code == 200

    # Then, link session2 to the case with session1 as previous
    link_data2 = {
        "session_id": session2_id,
        "case_id": case_id,
        "prev_session_id": session1_id,
    }

    link_response2 = await client.post("/session/link", json=link_data2)
    assert link_response2.status_code == 200

    link_result = link_response2.json()
    assert link_result["case_id"] == case_id
    assert len(link_result["sessions"]) == 2
    assert session1_id in link_result["sessions"]
    assert session2_id in link_result["sessions"]

    # 7. Test GET /report/case/{case_id}/trajectories endpoint
    case_trajectory_response = await client.get(f"/report/case/{case_id}/trajectories")
    assert case_trajectory_response.status_code == 200

    case_trajectory_data = case_trajectory_response.json()
    assert case_trajectory_data["case_id"] == case_id
    assert len(case_trajectory_data["sessions"]) == 2
    assert session1_id in case_trajectory_data["sessions"]
    assert session2_id in case_trajectory_data["sessions"]

    # Verify trajectory aggregation
    trajectories = case_trajectory_data["trajectories"]
    assert len(trajectories) == 1  # One trajectory t1

    t1_trajectory = trajectories[0]
    assert t1_trajectory["trajectory_id"] == "t1"
    assert t1_trajectory["coverage"] == 1.0  # Both steps completed across sessions
    assert set(t1_trajectory["completed_steps_union"]) == {"sleep", "mood"}  # Union of both steps

    # 8. Test that compute_session_metrics includes trajectory_progress
    # Check session1 metrics
    session1_metrics = await compute_session_metrics(db_session, uuid.UUID(session1_id))
    assert "trajectory_progress" in session1_metrics

    trajectory_progress1 = session1_metrics["trajectory_progress"]
    assert len(trajectory_progress1) == 1

    traj_progress = trajectory_progress1[0]
    assert traj_progress["trajectory_id"] == "t1"
    assert traj_progress["total"] == 2  # Total steps
    assert traj_progress["completed"] == 1  # Only sleep completed in session1
    assert traj_progress["completed_steps"] == ["sleep"]

    # Check session2 metrics
    session2_metrics = await compute_session_metrics(db_session, uuid.UUID(session2_id))
    assert "trajectory_progress" in session2_metrics

    trajectory_progress2 = session2_metrics["trajectory_progress"]
    assert len(trajectory_progress2) == 1

    traj_progress2 = trajectory_progress2[0]
    assert traj_progress2["trajectory_id"] == "t1"
    assert traj_progress2["total"] == 2  # Total steps
    assert traj_progress2["completed"] == 1  # Only mood completed in session2
    assert traj_progress2["completed_steps"] == ["mood"]

    # 9. Verify coverage calculation and union logic
    # The case-level aggregation should show full coverage (1.0) from union of both sessions
    assert t1_trajectory["coverage"] == 1.0  # 2 completed steps / 2 total steps = 1.0
    assert len(t1_trajectory["completed_steps_union"]) == 2  # Both sleep and mood


