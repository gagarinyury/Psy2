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
from app.core.tables import Session, TelemetryTurn
from app.orchestrator.nodes.normalize import normalize
from app.orchestrator.nodes.retrieve import retrieve

logger = logging.getLogger(__name__)


async def run_turn(request: TurnRequest, db: AsyncSession) -> TurnResponse:
    """
    Process a therapy session turn through the orchestration pipeline.

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

        # Convert session state to dict for normalize and retrieve
        session_state_dict = request.session_state.model_dump()

        # Step 1: Normalize therapist utterance
        normalize_result = normalize(request.therapist_utterance, session_state_dict)
        intent = normalize_result["intent"]
        topics = normalize_result["topics"]
        risk_flags = normalize_result["risk_flags"]
        last_turn_summary = normalize_result["last_turn_summary"]

        # Step 2: Retrieve relevant knowledge fragments
        candidates = await retrieve(
            db=db,
            case_id=request.case_id,
            intent=intent,
            topics=topics,
            session_state_compact=session_state_dict,
        )

        # Step 3: Form TurnResponse
        # Patient reply format: "Echo: {intent}, candidates={N}"
        patient_reply = f"Echo: {intent}, candidates={len(candidates)}"

        # State updates - update last_turn_summary from normalize
        state_updates = {"last_turn_summary": last_turn_summary}

        # Extract fragment IDs for used_fragments
        used_fragments = [candidate["id"] for candidate in candidates]

        # Determine risk status
        risk_status = "acute" if risk_flags else "none"

        # Create eval markers
        eval_markers = {"intent": intent}

        # Step 4: Record telemetry
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

    except ValueError as e:
        logger.error(f"Validation error in run_turn: {e}")
        raise
    except SQLAlchemyError as e:
        logger.exception(f"Database error in run_turn: {e}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in run_turn: {e}")
        raise


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
