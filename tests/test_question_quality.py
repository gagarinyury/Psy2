"""
Tests for Question Quality metric functionality.

Tests the question_quality calculation in session metrics
based on intent distribution from telemetry turns.
"""

import pytest

from app.core.tables import Case, Session, TelemetryTurn
from app.eval.metrics import compute_session_metrics


@pytest.mark.anyio
async def test_question_quality_calculation(db_session):
    """
    Test question quality metric with various intent types.

    Creates a session with 5 turns with different intents:
    - open_question: good
    - clarify: good
    - rapport: neutral
    - risk_check: neutral
    - unknown: no eval_markers

    Expected:
    - good == 2
    - known == 4
    - score == 0.5
    """
    # Create test case
    case = Case(
        case_truth={
            "dx_target": ["MDD"],
            "ddx": {"MDD": 0.8},
            "red_flags": ["depression"],
            "hidden_facts": ["family history"],
        },
        policies={
            "disclosure_rules": {"min_trust_for_gated": 0.4},
            "risk_protocol": {"trigger_keywords": ["suicide"]},
        },
    )
    db_session.add(case)
    await db_session.flush()

    # Create session
    session = Session(
        case_id=case.id,
        session_state={
            "trust": 0.5,
            "fatigue": 0.1,
            "affect": "neutral",
            "risk_status": "none",
        },
    )
    db_session.add(session)
    await db_session.flush()

    # Create telemetry turns with different intents
    turns_data = [
        {
            "turn_no": 1,
            "eval_markers": {"intent": "open_question"},
            "used_fragments": [],
            "risk_status": "none",
        },
        {
            "turn_no": 2,
            "eval_markers": {"intent": "clarify"},
            "used_fragments": [],
            "risk_status": "none",
        },
        {
            "turn_no": 3,
            "eval_markers": {"intent": "rapport"},
            "used_fragments": [],
            "risk_status": "none",
        },
        {
            "turn_no": 4,
            "eval_markers": {"intent": "risk_check"},
            "used_fragments": [],
            "risk_status": "none",
        },
        {
            "turn_no": 5,
            "eval_markers": None,  # This will be treated as unknown
            "used_fragments": [],
            "risk_status": "none",
        },
    ]

    for turn_data in turns_data:
        turn = TelemetryTurn(
            session_id=session.id,
            turn_no=turn_data["turn_no"],
            used_fragments=turn_data["used_fragments"],
            risk_status=turn_data["risk_status"],
            eval_markers=turn_data["eval_markers"],
            timings={},
            costs={},
        )
        db_session.add(turn)

    await db_session.commit()

    # Test metrics computation
    metrics = await compute_session_metrics(db_session, session.id)

    # Verify question_quality structure exists
    assert "question_quality" in metrics

    question_quality = metrics["question_quality"]

    # Verify structure
    assert "score" in question_quality
    assert "counts" in question_quality
    assert "known" in question_quality
    assert "good" in question_quality

    # Verify counts
    counts = question_quality["counts"]
    assert counts["open_question"] == 1
    assert counts["clarify"] == 1
    assert counts["rapport"] == 1
    assert counts["risk_check"] == 1
    assert counts["unknown"] == 1

    # Verify calculated metrics
    assert question_quality["good"] == 2  # open_question + clarify
    assert question_quality["known"] == 4  # 5 total - 1 unknown
    assert question_quality["score"] == 0.5  # 2 good / 4 known

    # Additional verifications
    assert 0.0 <= question_quality["score"] <= 1.0
    assert question_quality["known"] >= 0
    assert question_quality["good"] >= 0


@pytest.mark.anyio
async def test_question_quality_all_good(db_session):
    """Test case where all turns have good intents."""
    # Create test case and session
    case = Case(
        case_truth={"dx_target": ["GAD"], "ddx": {"GAD": 0.9}, "red_flags": [], "hidden_facts": []},
        policies={"disclosure_rules": {"min_trust_for_gated": 0.3}},
    )
    db_session.add(case)
    await db_session.flush()

    session = Session(
        case_id=case.id,
        session_state={"trust": 0.5, "fatigue": 0.1, "affect": "neutral", "risk_status": "none"},
    )
    db_session.add(session)
    await db_session.flush()

    # Create turns with only good intents
    for i, intent in enumerate(["open_question", "clarify", "open_question"], 1):
        turn = TelemetryTurn(
            session_id=session.id,
            turn_no=i,
            used_fragments=[],
            risk_status="none",
            eval_markers={"intent": intent},
            timings={},
            costs={},
        )
        db_session.add(turn)

    await db_session.commit()

    # Test metrics computation
    metrics = await compute_session_metrics(db_session, session.id)
    question_quality = metrics["question_quality"]

    # All turns are good and known
    assert question_quality["good"] == 3
    assert question_quality["known"] == 3
    assert question_quality["score"] == 1.0


@pytest.mark.anyio
async def test_question_quality_no_known_intents(db_session):
    """Test case where all turns have unknown intent."""
    # Create test case and session
    case = Case(
        case_truth={
            "dx_target": ["PTSD"],
            "ddx": {"PTSD": 0.7},
            "red_flags": [],
            "hidden_facts": [],
        },
        policies={"disclosure_rules": {"min_trust_for_gated": 0.3}},
    )
    db_session.add(case)
    await db_session.flush()

    session = Session(
        case_id=case.id,
        session_state={"trust": 0.5, "fatigue": 0.1, "affect": "neutral", "risk_status": "none"},
    )
    db_session.add(session)
    await db_session.flush()

    # Create turns with no eval_markers (unknown intent)
    for i in range(1, 3):
        turn = TelemetryTurn(
            session_id=session.id,
            turn_no=i,
            used_fragments=[],
            risk_status="none",
            eval_markers=None,  # Unknown intent
            timings={},
            costs={},
        )
        db_session.add(turn)

    await db_session.commit()

    # Test metrics computation
    metrics = await compute_session_metrics(db_session, session.id)
    question_quality = metrics["question_quality"]

    # No known intents, score should be 0/1 = 0
    assert question_quality["good"] == 0
    assert question_quality["known"] == 0
    assert question_quality["score"] == 0.0  # 0 / max(0, 1) = 0
    assert question_quality["counts"]["unknown"] == 2
