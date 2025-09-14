"""
Pipeline orchestrator for processing therapy session turns.
Coordinates normalize and retrieve nodes to generate patient responses.
"""

import logging
import uuid
from typing import Any
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import TurnRequest, TurnResponse, CaseTruth, Trajectory, TrajectoryStep
from app.core.settings import settings
from app.core.tables import Case, Session, TelemetryTurn, SessionTrajectory, KBFragment
from app.orchestrator.nodes.generate_llm import generate_llm
from app.orchestrator.nodes.guard import guard
from app.orchestrator.nodes.normalize import normalize
from app.orchestrator.nodes.reason import reason
from app.orchestrator.nodes.reason_llm import reason_llm
from app.orchestrator.nodes.retrieve import retrieve

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


async def update_trajectory_progress(
    db: AsyncSession,
    session_id: str,
    case_truth: dict,
    session_state_trust: float,
    used_fragments: list[str],
) -> None:
    """
    Update trajectory progress based on current turn results.

    For each trajectory in case_truth, check if any steps should be completed
    based on trust level and used fragments with matching tags.

    Args:
        db: Database session
        session_id: UUID of the session
        case_truth: Case truth data containing trajectories
        session_state_trust: Current session trust level
        used_fragments: List of fragment IDs used in this turn
    """
    try:
        # Parse case_truth into CaseTruth model to access trajectories
        case_truth_model = CaseTruth(**case_truth)

        if not case_truth_model.trajectories or not used_fragments:
            return

        # Get metadata for used fragments to check tags
        fragment_query = select(KBFragment).where(
            KBFragment.id.in_([uuid.UUID(fid) for fid in used_fragments])
        )
        fragment_result = await db.execute(fragment_query)
        used_fragment_records = fragment_result.scalars().all()

        # Collect all tags from used fragments
        used_tags = set()
        for fragment in used_fragment_records:
            if fragment.fragment_metadata and "tags" in fragment.fragment_metadata:
                used_tags.update(fragment.fragment_metadata["tags"])

        if not used_tags:
            return

        # Process each trajectory
        for trajectory in case_truth_model.trajectories:
            # Get current session trajectory record
            trajectory_query = select(SessionTrajectory).where(
                SessionTrajectory.session_id == uuid.UUID(session_id),
                SessionTrajectory.trajectory_id == trajectory.id
            )
            trajectory_result = await db.execute(trajectory_query)
            session_trajectory = trajectory_result.scalar_one_or_none()

            # Track new steps to complete
            new_completed_steps = []
            existing_completed_steps = []

            if session_trajectory:
                existing_completed_steps = session_trajectory.completed_steps or []

            # Check each step in the trajectory
            for step in trajectory.steps:
                # Skip if step already completed
                if step.id in existing_completed_steps:
                    continue

                # Check trust threshold
                if session_state_trust < step.min_trust:
                    continue

                # Check if step condition tags intersect with used fragment tags
                if step.condition_tags and not set(step.condition_tags).intersection(used_tags):
                    continue

                # Step conditions met - mark for completion
                new_completed_steps.append(step.id)
                logger.info(
                    f"Trajectory step completed: {trajectory.id}/{step.id} "
                    f"(trust: {session_state_trust:.2f}, tags: {list(set(step.condition_tags).intersection(used_tags))})"
                )

            if new_completed_steps:
                if session_trajectory:
                    # Update existing record
                    session_trajectory.completed_steps = existing_completed_steps + new_completed_steps
                    session_trajectory.updated_at = func.now()
                else:
                    # Create new session trajectory record
                    session_trajectory = SessionTrajectory(
                        session_id=uuid.UUID(session_id),
                        trajectory_id=trajectory.id,
                        completed_steps=new_completed_steps
                    )
                    db.add(session_trajectory)

        await db.commit()

    except Exception as e:
        logger.error(f"Failed to update trajectory progress: {e}")
        await db.rollback()
        # Don't raise - trajectory tracking is non-critical


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

        # Step 1: Get policies first to pass to normalize
        policies = await get_policies(db, request.case_id)

        # Step 2: Normalize therapist utterance with policies
        n = normalize(request.therapist_utterance, session_state_dict, policies)

        # Step 3: Retrieve relevant knowledge fragments
        cands = await retrieve(
            db=db,
            case_id=request.case_id,
            intent=n["intent"],
            topics=n["topics"],
            session_state_compact=session_state_dict,
        )

        # Step 4: Reason - get case_truth (policies already loaded)
        case_truth = await get_case_truth(db, request.case_id)

        # Use LLM reasoning if enabled, otherwise use stub
        if settings.USE_DEEPSEEK_REASON:
            try:
                r = await reason_llm(case_truth, session_state_dict, cands, policies)
                logger.info("Used DeepSeek reasoning")
            except Exception as e:
                logger.error(f"DeepSeek reasoning failed, falling back to stub: {e}")
                r = reason(case_truth, session_state_dict, cands, policies)
        else:
            r = reason(case_truth, session_state_dict, cands, policies)

        # Step 5: Guard - apply risk filtering
        g = guard(r, policies, n["risk_flags"])

        # Step 6: Form response
        if settings.USE_DEEPSEEK_GEN:
            try:
                # Use LLM generation for natural patient response
                content_plan = g["safe_output"]["content_plan"]
                style_directives = g["safe_output"]["style_directives"]
                patient_context = f"Patient with {case_truth.get('dx_target', ['unknown condition'])[0]}"

                patient_reply = await generate_llm(
                    content_plan, style_directives, patient_context
                )
                logger.info("Used DeepSeek generation")
            except Exception as e:
                logger.error(
                    f"DeepSeek generation failed, falling back to plan format: {e}"
                )
                patient_reply = f"Plan:{len(g['safe_output']['content_plan'])} intent={n['intent']} risk={'acute' if g['risk_status']=='acute' else 'none'}"
        else:
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

        # Step 7: Record telemetry
        await _record_telemetry(
            db=db,
            session_id=request.session_id,
            used_fragments=used_fragments,
            risk_status=risk_status,
            eval_markers=eval_markers,
        )

        # Step 8: Update trajectory progress
        await update_trajectory_progress(
            db=db,
            session_id=request.session_id,
            case_truth=case_truth,
            session_state_trust=request.session_state.trust,
            used_fragments=used_fragments,
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
