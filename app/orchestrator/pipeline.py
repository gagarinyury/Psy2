"""
Pipeline orchestrator for processing therapy session turns.
Coordinates normalize and retrieve nodes to generate patient responses.
"""

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.exc import SQLAlchemyError

from app.core.models import TurnRequest, TurnResponse
from app.core.tables import Session, TelemetryTurn, Case
from app.orchestrator.nodes.normalize import normalize
from app.orchestrator.nodes.retrieve import retrieve
from app.orchestrator.nodes.reason import reason
from app.orchestrator.nodes.guard import guard

logger = logging.getLogger(__name__)


async def get_case_truth(db: AsyncSession, case_id: str) -> dict:
    """
    Retrieve case truth from database.

    Args:
        db: Database session
        case_id: UUID string of the case

    Returns:
        dict: Case truth data
    """
    case_query = select(Case).where(Case.id == case_id)
    case_result = await db.execute(case_query)
    case = case_result.scalar_one_or_none()

    if not case:
        raise ValueError(f"Case {case_id} not found")

    return case.case_truth


async def get_policies(db: AsyncSession, case_id: str) -> dict:
    """
    Retrieve policies from database.

    Args:
        db: Database session
        case_id: UUID string of the case

    Returns:
        dict: Policies data
    """
    case_query = select(Case).where(Case.id == case_id)
    case_result = await db.execute(case_query)
    case = case_result.scalar_one_or_none()

    if not case:
        raise ValueError(f"Case {case_id} not found")

    return case.policies


async def run_turn(request: TurnRequest, db: AsyncSession) -> TurnResponse:
    """
    Process a therapy session turn through the full orchestration pipeline.
    Pipeline: normalize → retrieve → reason → guard

    Args:
        request: TurnRequest containing therapist utterance, session state, and case_id
        db: AsyncSession for database operations

    Returns:
        TurnResponse with patient reply, state updates, fragments, and telemetry

    Raises:
        ValueError: If session doesn't exist or invalid input
        SQLAlchemyError: Database operation errors
    """
    try:
        # Verify session exists
        session_query = select(Session).where(Session.id == request.session_id)
        session_result = await db.execute(session_query)
        session = session_result.scalar_one_or_none()

        if not session:
            raise ValueError(f"Session {request.session_id} not found")

        # Convert session state to dict for pipeline nodes
        session_state_dict = request.session_state.model_dump()

        # Step 1: Normalize therapist utterance
        n = normalize(request.therapist_utterance, session_state_dict)

        # Step 2: Retrieve relevant knowledge fragments
        cands = await retrieve(
            db=db,
            case_id=request.case_id,
            intent=n["intent"],
            topics=n["topics"],
            session_state_compact=session_state_dict,
        )

        # Step 3: Reason - get case_truth and policies from DB
        case_truth = await get_case_truth(db, request.case_id)
        policies = await get_policies(db, request.case_id)
        r = reason(case_truth, session_state_dict, cands, policies)

        # Step 4: Guard - apply risk filtering
        g = guard(r, policies, n["risk_flags"])

        # Step 5: Form response
        patient_reply = f"Plan:{len(g['safe_output']['content_plan'])} intent={n['intent']} risk={'acute' if g['risk_status']=='acute' else 'none'}"

        # State updates - combine from reason and normalize
        state_updates = r["state_updates"] | {
            "last_turn_summary": n["last_turn_summary"]
        }

        # Fragment IDs from reason telemetry
        used_fragments = r["telemetry"].get("chosen_ids", [])

        # Risk status from guard
        risk_status = g["risk_status"]

        # Eval markers - include topics for enhanced evaluation
        eval_markers = {"intent": n["intent"], "topics": n["topics"]}

        # Step 6: Record telemetry
        await _record_telemetry(
            db=db,
            session_id=request.session_id,
            used_fragments=used_fragments,
            risk_status=risk_status,
            eval_markers=eval_markers,
        )

        return TurnResponse(
            patient_reply=patient_reply,
            state_updates=state_updates,
            used_fragments=used_fragments,
            risk_status=risk_status,
            eval_markers=eval_markers,
        )

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        # Safe fallback response
        return TurnResponse(
            patient_reply="safe-fallback",
            state_updates={},
            used_fragments=[],
            risk_status="none",
            eval_markers={},
        )


async def _record_telemetry(
    db: AsyncSession,
    session_id: str,
    used_fragments: list[str],
    risk_status: str,
    eval_markers: dict[str, Any],
) -> None:
    """
    Record telemetry data for the current turn.

    Args:
        db: Database session
        session_id: UUID of the session
        used_fragments: List of fragment IDs used in this turn
        risk_status: Risk status ("acute" or "none")
        eval_markers: Evaluation markers dict
    """
    try:
        # Get current turn number (MAX + 1)
        turn_no_query = select(
            func.coalesce(func.max(TelemetryTurn.turn_no), 0) + 1
        ).where(TelemetryTurn.session_id == session_id)
        turn_no_result = await db.execute(turn_no_query)
        turn_no = turn_no_result.scalar()

        # Create telemetry record
        telemetry_turn = TelemetryTurn(
            session_id=session_id,
            turn_no=turn_no,
            used_fragments=used_fragments,
            risk_status=risk_status,
            eval_markers=eval_markers,
            timings={},
            costs={},
        )

        db.add(telemetry_turn)
        await db.commit()

        logger.info(
            f"Recorded telemetry for session {session_id}, turn {turn_no}, "
            f"fragments: {len(used_fragments)}, risk: {risk_status}"
        )

    except SQLAlchemyError as e:
        logger.exception(f"Failed to record telemetry: {e}")
        await db.rollback()
        raise
