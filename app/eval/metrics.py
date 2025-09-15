"""
Session evaluation metrics calculation.

Computes key metrics for completed therapy sessions including:
- Recall-Keys: coverage of key knowledge fragments
- Risk-Timeliness: how quickly risks were identified
"""

from typing import Any, Dict
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.models import Trajectory
from app.core.tables import Case, KBFragment, Session, SessionTrajectory


async def compute_session_metrics(db: AsyncSession, session_id: UUID) -> Dict[str, Any]:
    """
    Compute evaluation metrics for a completed session.

    Args:
        db: Database session
        session_id: UUID of the session to evaluate

    Returns:
        dict: Metrics including recall_keys, risk_timeliness, and counts

    Raises:
        ValueError: If session not found
    """
    # Get session with case and telemetry data
    session_query = (
        select(Session)
        .options(
            selectinload(Session.case),
            selectinload(Session.telemetry_turns),
        )
        .where(Session.id == session_id)
    )

    result = await db.execute(session_query)
    session = result.scalar_one_or_none()

    if not session:
        raise ValueError(f"Session {session_id} not found")

    case = session.case
    case_id = case.id

    # Get all key fragments for this case
    key_fragments_query = select(KBFragment).where(KBFragment.case_id == case_id)

    key_fragments_result = await db.execute(key_fragments_query)
    all_fragments = key_fragments_result.scalars().all()

    # Filter key fragments (tags contain "hook" or "key")
    all_key_ids = []
    for fragment in all_fragments:
        tags = fragment.fragment_metadata.get("tags", [])
        if isinstance(tags, list) and ("hook" in tags or "key" in tags):
            all_key_ids.append(str(fragment.id))

    # Process telemetry turns in order
    turns = sorted(session.telemetry_turns, key=lambda t: t.turn_no)

    used_fragment_ids = set()
    first_acute_turn = None
    turns_total = len(turns)

    # Question quality tracking
    intent_counts = {
        "open_question": 0,
        "clarify": 0,
        "risk_check": 0,
        "rapport": 0,
        "unknown": 0,
    }

    for turn in turns:
        # Collect used fragments
        if isinstance(turn.used_fragments, list):
            for fragment_id in turn.used_fragments:
                used_fragment_ids.add(str(fragment_id))

        # Track first acute risk status
        if turn.risk_status == "acute" and first_acute_turn is None:
            first_acute_turn = turn.turn_no

        # Track question quality intent
        eval_markers = turn.eval_markers or {}
        intent = eval_markers.get("intent")
        if intent in intent_counts:
            intent_counts[intent] += 1
        else:
            intent_counts["unknown"] += 1

    # Calculate used key fragments
    used_key_ids = list(used_fragment_ids.intersection(set(all_key_ids)))

    # Calculate missed key fragments
    missed_key_ids = list(set(all_key_ids) - used_fragment_ids)

    # Calculate Recall-Keys metric
    if all_key_ids:
        recall_keys = len(used_key_ids) / len(all_key_ids)
    else:
        recall_keys = 1.0  # Perfect score if no key fragments exist

    # Calculate Risk-Timeliness metric
    case_truth = case.case_truth or {}
    red_flags = case_truth.get("red_flags", [])
    has_red_flags = bool(red_flags)

    if not has_red_flags:
        # No red flags in case, perfect timeliness
        risk_timeliness = 1.0
    elif first_acute_turn is None:
        # Had red flags but never detected risk
        risk_timeliness = 0.0
    elif first_acute_turn <= 3:
        # Detected within first 3 turns
        risk_timeliness = 1.0
    elif first_acute_turn <= 6:
        # Detected in turns 4-6
        risk_timeliness = 0.5
    else:
        # Detected after turn 6
        risk_timeliness = 0.0

    # Calculate Question-Quality metric
    good_count = intent_counts["open_question"] + intent_counts["clarify"]
    known_count = turns_total - intent_counts["unknown"]
    question_quality_score = good_count / max(known_count, 1)

    # Calculate trajectory progress
    trajectory_progress = []
    case_truth = case.case_truth or {}
    trajectories = case_truth.get("trajectories", [])

    if trajectories:
        # Get session trajectory records for this session
        session_trajectory_query = select(SessionTrajectory).where(
            SessionTrajectory.session_id == session_id
        )
        session_trajectory_result = await db.execute(session_trajectory_query)
        session_trajectories = {
            st.trajectory_id: st for st in session_trajectory_result.scalars().all()
        }

        for trajectory in trajectories:
            trajectory_obj = (
                Trajectory(**trajectory) if isinstance(trajectory, dict) else trajectory
            )
            total_steps = len(trajectory_obj.steps)

            # Find corresponding session trajectory record
            session_traj = session_trajectories.get(trajectory_obj.id)
            completed_steps = session_traj.completed_steps if session_traj else []

            trajectory_progress.append(
                {
                    "trajectory_id": trajectory_obj.id,
                    "completed": len(completed_steps),
                    "total": total_steps,
                    "completed_steps": completed_steps,
                }
            )

    return {
        "recall_keys": recall_keys,
        "risk_timeliness": risk_timeliness,
        "turns_total": turns_total,
        "used_fragments_total": len(used_fragment_ids),
        "key_fragments_total": len(all_key_ids),
        "used_key_ids": used_key_ids,
        "all_key_ids": all_key_ids,
        "missed_keys": {
            "ids": missed_key_ids,
            "count": len(missed_key_ids),
        },
        "question_quality": {
            "score": question_quality_score,
            "counts": intent_counts.copy(),
            "known": known_count,
            "good": good_count,
        },
        "first_acute_turn": first_acute_turn,
        "trajectory_progress": trajectory_progress,
    }


async def compute_case_trajectories(db: AsyncSession, case_id: UUID) -> dict:
    """
    Compute trajectory aggregate metrics for a case across all its sessions.

    Args:
        db: Database session
        case_id: UUID of the case to evaluate

    Returns:
        dict: Case trajectory metrics with coverage and completed steps union

    Raises:
        ValueError: If case not found
    """
    # Get the case to access case_truth
    case_query = select(Case).where(Case.id == case_id)
    case_result = await db.execute(case_query)
    case = case_result.scalar_one_or_none()

    if not case:
        raise ValueError(f"Case {case_id} not found")

    # Get all sessions for this case
    sessions_query = select(Session.id).where(Session.case_id == case_id)
    sessions_result = await db.execute(sessions_query)
    session_ids = [str(sid) for sid in sessions_result.scalars().all()]

    if not session_ids:
        return {"case_id": str(case_id), "sessions": [], "trajectories": []}

    # Get all SessionTrajectory records for these sessions
    session_trajectory_query = select(SessionTrajectory).where(
        SessionTrajectory.session_id.in_([UUID(sid) for sid in session_ids])
    )
    session_trajectory_result = await db.execute(session_trajectory_query)
    session_trajectories = session_trajectory_result.scalars().all()

    # Group session trajectories by trajectory_id
    trajectory_sessions = {}
    for st in session_trajectories:
        trajectory_id = st.trajectory_id
        if trajectory_id not in trajectory_sessions:
            trajectory_sessions[trajectory_id] = []
        trajectory_sessions[trajectory_id].append(st)

    # Process case trajectories
    trajectories = []
    case_truth = case.case_truth or {}
    case_trajectories = case_truth.get("trajectories", [])

    for trajectory in case_trajectories:
        trajectory_obj = Trajectory(**trajectory) if isinstance(trajectory, dict) else trajectory
        trajectory_id = trajectory_obj.id
        total_steps = len(trajectory_obj.steps)

        # Union all completed steps from all sessions for this trajectory
        completed_steps_union = set()
        session_trajs = trajectory_sessions.get(trajectory_id, [])

        for session_traj in session_trajs:
            completed_steps_union.update(session_traj.completed_steps or [])

        # Calculate coverage
        coverage = len(completed_steps_union) / total_steps if total_steps > 0 else 0.0

        trajectories.append(
            {
                "trajectory_id": trajectory_id,
                "completed_steps_union": list(completed_steps_union),
                "coverage": coverage,
            }
        )

    return {"case_id": str(case_id), "sessions": session_ids, "trajectories": trajectories}
