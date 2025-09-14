"""
Session evaluation metrics calculation.

Computes key metrics for completed therapy sessions including:
- Recall-Keys: coverage of key knowledge fragments
- Risk-Timeliness: how quickly risks were identified
"""

from typing import Dict, Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.tables import Session, KBFragment


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

    for turn in turns:
        # Collect used fragments
        if isinstance(turn.used_fragments, list):
            for fragment_id in turn.used_fragments:
                used_fragment_ids.add(str(fragment_id))

        # Track first acute risk status
        if turn.risk_status == "acute" and first_acute_turn is None:
            first_acute_turn = turn.turn_no

    # Calculate used key fragments
    used_key_ids = list(used_fragment_ids.intersection(set(all_key_ids)))

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

    return {
        "recall_keys": recall_keys,
        "risk_timeliness": risk_timeliness,
        "turns_total": turns_total,
        "used_fragments_total": len(used_fragment_ids),
        "key_fragments_total": len(all_key_ids),
        "used_key_ids": used_key_ids,
        "all_key_ids": all_key_ids,
        "first_acute_turn": first_acute_turn,
    }
