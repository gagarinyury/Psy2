"""
Tests for session evaluation metrics.

Tests the compute_session_metrics function with various scenarios
including key fragment coverage and risk detection timeliness.
"""

from uuid import uuid4

import pytest

from app.core.tables import Case, KBFragment, Session, TelemetryTurn
from app.eval.metrics import compute_session_metrics


@pytest.mark.anyio
async def test_compute_session_metrics_with_key_fragments(db_session):
    """
    Test metrics computation with key fragments and risk detection.

    Creates demo case with one key fragment, session with 4 turns:
    - turn 1: no fragments, no risk
    - turn 2: uses kb1 (key fragment), no risk
    - turn 3: risk detected (acute)
    - turn 4: additional turn

    Expects:
    - recall_keys = 1.0 (used 1 of 1 key fragments)
    - risk_timeliness = 1.0 (first acute on turn ≤3)
    """
    # Create test case with red flags
    case = Case(
        case_truth={
            "dx_target": ["MDD"],
            "red_flags": ["suicidal ideation"],
            "hidden_facts": ["depression episode"],
        },
        policies={
            "disclosure_rules": {"min_trust_for_gated": 0.4},
            "risk_protocol": {"trigger_keywords": ["suicide"]},
        },
    )
    db_session.add(case)
    await db_session.flush()  # Get case.id

    # Create KB fragments - one with key tag, one without
    key_fragment = KBFragment(
        case_id=case.id,
        type="patient_history",
        text="Patient has history of depression and suicidal thoughts",
        fragment_metadata={"tags": ["hook"], "topic": "mental_health"},
        availability="public",
        consistency_keys={},
    )

    regular_fragment = KBFragment(
        case_id=case.id,
        type="patient_info",
        text="Patient is 35 years old",
        fragment_metadata={"tags": ["demographic"], "topic": "basic_info"},
        availability="public",
        consistency_keys={},
    )

    db_session.add(key_fragment)
    db_session.add(regular_fragment)
    await db_session.flush()  # Get fragment IDs

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
    await db_session.flush()  # Get session.id

    # Create telemetry turns
    turns_data = [
        # Turn 1: no fragments, no risk
        {
            "turn_no": 1,
            "used_fragments": [],
            "risk_status": "none",
            "eval_markers": {"intent": "greeting"},
        },
        # Turn 2: uses key fragment, no risk
        {
            "turn_no": 2,
            "used_fragments": [str(key_fragment.id)],
            "risk_status": "none",
            "eval_markers": {"intent": "clarify"},
        },
        # Turn 3: risk detected (acute)
        {
            "turn_no": 3,
            "used_fragments": [],
            "risk_status": "acute",
            "eval_markers": {"intent": "risk_check"},
        },
        # Turn 4: additional turn
        {
            "turn_no": 4,
            "used_fragments": [str(regular_fragment.id)],
            "risk_status": "none",
            "eval_markers": {"intent": "clarify"},
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

    # Verify metrics
    assert metrics["recall_keys"] == 1.0  # Used 1 of 1 key fragments
    assert metrics["risk_timeliness"] == 1.0  # First acute on turn 3 (≤3)
    assert metrics["turns_total"] == 4
    assert metrics["used_fragments_total"] == 2  # key_fragment + regular_fragment
    assert metrics["key_fragments_total"] == 1  # Only one key fragment
    assert metrics["first_acute_turn"] == 3

    # Check fragment IDs in results
    assert str(key_fragment.id) in metrics["used_key_ids"]
    assert str(key_fragment.id) in metrics["all_key_ids"]
    assert len(metrics["used_key_ids"]) == 1
    assert len(metrics["all_key_ids"]) == 1


@pytest.mark.anyio
async def test_compute_session_metrics_partial_recall(db_session):
    """Test metrics with partial key fragment coverage."""
    # Create case
    case = Case(
        case_truth={
            "dx_target": ["GAD"],
            "red_flags": ["anxiety attacks"],
        },
        policies={},
    )
    db_session.add(case)
    await db_session.flush()

    # Create multiple key fragments
    key_fragment1 = KBFragment(
        case_id=case.id,
        type="symptom",
        text="Anxiety symptoms present",
        fragment_metadata={"tags": ["key"], "topic": "symptoms"},
        availability="public",
        consistency_keys={},
    )

    key_fragment2 = KBFragment(
        case_id=case.id,
        type="trigger",
        text="Panic trigger identified",
        fragment_metadata={"tags": ["hook"], "topic": "triggers"},
        availability="public",
        consistency_keys={},
    )

    db_session.add(key_fragment1)
    db_session.add(key_fragment2)
    await db_session.flush()

    # Create session
    session = Session(case_id=case.id, session_state={})
    db_session.add(session)
    await db_session.flush()

    # Create turn that uses only one key fragment
    turn = TelemetryTurn(
        session_id=session.id,
        turn_no=1,
        used_fragments=[str(key_fragment1.id)],  # Only uses first key fragment
        risk_status="none",
        eval_markers={"intent": "clarify"},
        timings={},
        costs={},
    )
    db_session.add(turn)
    await db_session.commit()

    # Test metrics
    metrics = await compute_session_metrics(db_session, session.id)

    # Should have 50% recall (1 of 2 key fragments used)
    assert metrics["recall_keys"] == 0.5
    assert metrics["key_fragments_total"] == 2
    assert len(metrics["used_key_ids"]) == 1
    assert len(metrics["all_key_ids"]) == 2


@pytest.mark.anyio
async def test_compute_session_metrics_late_risk_detection(db_session):
    """Test risk timeliness with late detection."""
    # Create case with red flags
    case = Case(
        case_truth={"red_flags": ["self_harm"]},
        policies={},
    )
    db_session.add(case)
    await db_session.flush()

    # Create session
    session = Session(case_id=case.id, session_state={})
    db_session.add(session)
    await db_session.flush()

    # Create 7 turns, risk detected only on turn 7
    for turn_no in range(1, 8):
        risk_status = "acute" if turn_no == 7 else "none"
        turn = TelemetryTurn(
            session_id=session.id,
            turn_no=turn_no,
            used_fragments=[],
            risk_status=risk_status,
            eval_markers={"intent": "clarify"},
            timings={},
            costs={},
        )
        db_session.add(turn)

    await db_session.commit()

    # Test metrics
    metrics = await compute_session_metrics(db_session, session.id)

    # Risk detected after turn 6, should be 0.0
    assert metrics["risk_timeliness"] == 0.0
    assert metrics["first_acute_turn"] == 7


@pytest.mark.anyio
async def test_compute_session_metrics_no_key_fragments(db_session):
    """Test metrics when no key fragments exist."""
    # Create case without key fragments
    case = Case(case_truth={}, policies={})
    db_session.add(case)
    await db_session.flush()

    # Create regular (non-key) fragment
    regular_fragment = KBFragment(
        case_id=case.id,
        type="info",
        text="Basic information",
        fragment_metadata={"tags": ["info"]},  # No "key" or "hook" tags
        availability="public",
        consistency_keys={},
    )
    db_session.add(regular_fragment)
    await db_session.flush()

    # Create session and turn
    session = Session(case_id=case.id, session_state={})
    db_session.add(session)
    await db_session.flush()

    turn = TelemetryTurn(
        session_id=session.id,
        turn_no=1,
        used_fragments=[str(regular_fragment.id)],
        risk_status="none",
        eval_markers={"intent": "clarify"},
        timings={},
        costs={},
    )
    db_session.add(turn)
    await db_session.commit()

    # Test metrics
    metrics = await compute_session_metrics(db_session, session.id)

    # No key fragments = perfect recall
    assert metrics["recall_keys"] == 1.0
    assert metrics["key_fragments_total"] == 0
    assert len(metrics["used_key_ids"]) == 0
    assert len(metrics["all_key_ids"]) == 0


@pytest.mark.anyio
async def test_compute_session_metrics_session_not_found():
    """Test error handling for non-existent session."""
    from app.core.db import AsyncSessionLocal

    non_existent_id = uuid4()

    session = AsyncSessionLocal()
    try:
        with pytest.raises(ValueError, match="Session .* not found"):
            await compute_session_metrics(session, non_existent_id)
    finally:
        await session.close()


@pytest.mark.anyio
async def test_compute_session_metrics_medium_risk_timeliness(db_session):
    """Test risk timeliness with detection in turns 4-6."""
    # Create case with red flags
    case = Case(
        case_truth={"red_flags": ["trauma"]},
        policies={},
    )
    db_session.add(case)
    await db_session.flush()

    # Create session
    session = Session(case_id=case.id, session_state={})
    db_session.add(session)
    await db_session.flush()

    # Create 5 turns, risk detected on turn 5 (should get 0.5 score)
    for turn_no in range(1, 6):
        risk_status = "acute" if turn_no == 5 else "none"
        turn = TelemetryTurn(
            session_id=session.id,
            turn_no=turn_no,
            used_fragments=[],
            risk_status=risk_status,
            eval_markers={"intent": "clarify"},
            timings={},
            costs={},
        )
        db_session.add(turn)

    await db_session.commit()

    # Test metrics
    metrics = await compute_session_metrics(db_session, session.id)

    # Risk detected on turn 5 (4-6 range) should be 0.5
    assert metrics["risk_timeliness"] == 0.5
    assert metrics["first_acute_turn"] == 5
