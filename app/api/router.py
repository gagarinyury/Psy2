import uuid
from typing import Annotated, Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.core.models import (
    CaseRequest,
    CaseResponse,
    CaseTrajectoryResponse,
    LLMFlagsRequest,
    LLMFlagsResponse,
    RAGModeRequest,
    RAGModeResponse,
    SessionLinkRequest,
    SessionLinkResponse,
    SessionRequest,
    SessionResponse,
    SessionStateCompact,
    SessionTrajectoryResponse,
    TrajectoryAggregateItem,
    TrajectoryProgressItem,
    TurnRequest,
    TurnResponse,
)
from app.core.settings import settings
from app.core.tables import Case, Session, SessionLink, SessionTrajectory
from app.eval.metrics import compute_session_metrics
from app.infra.logging import get_logger
from app.infra.metrics import CASE_OPERATIONS, SESSION_OPERATIONS, TURN_OPERATIONS
from app.orchestrator.pipeline import run_turn

logger = get_logger()
router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "ok"}


@router.post("/case", response_model=CaseResponse)
async def create_case(
    request: CaseRequest, db: Annotated[AsyncSession, Depends(get_db)]
) -> CaseResponse:
    """Create a new case with CaseTruth validation"""
    try:
        # Create case record
        case = Case(
            case_truth=request.case_truth.model_dump(),
            policies=request.policies.model_dump() if request.policies else {},
            version="1.0",
        )

        db.add(case)
        await db.commit()
        await db.refresh(case)

        # Metrics
        CASE_OPERATIONS.labels(operation="create").inc()

        logger.info(f"Created case with ID: {case.id}")

        return CaseResponse(case_id=str(case.id))

    except Exception as e:
        logger.error(f"Error creating case: {e}")
        raise HTTPException(status_code=500, detail="Failed to create case")


@router.post("/session", response_model=SessionResponse)
async def create_session(
    request: SessionRequest, db: Annotated[AsyncSession, Depends(get_db)]
) -> SessionResponse:
    """Create a new session for a case"""
    try:
        # Validate case exists
        case_uuid = uuid.UUID(request.case_id)
        case = await db.get(Case, case_uuid)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

        # Create default session state
        default_state = SessionStateCompact(
            affect="neutral",
            trust=0.3,
            fatigue=0.1,
            access_level=1,
            risk_status="none",
            last_turn_summary="",
        )

        # Create session record
        session = Session(case_id=case_uuid, session_state=default_state.model_dump())

        db.add(session)
        await db.commit()
        await db.refresh(session)

        # Metrics
        SESSION_OPERATIONS.labels(operation="create").inc()

        logger.info(
            f"Created session with ID: {session.id} for case: {request.case_id}"
        )

        return SessionResponse(session_id=str(session.id))

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid case_id format")
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")


@router.post("/turn", response_model=TurnResponse)
async def process_turn(
    request: TurnRequest, db: Annotated[AsyncSession, Depends(get_db)]
) -> TurnResponse:
    """Process a turn request through normalize → retrieve pipeline"""
    try:
        # Validate case exists
        case_uuid = uuid.UUID(request.case_id)
        case = await db.get(Case, case_uuid)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

        # Process turn through pipeline
        response = await run_turn(request, db)

        # Metrics
        TURN_OPERATIONS.labels(operation="process").inc()

        logger.info(f"Processed turn through pipeline for case: {request.case_id}")

        return response

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid case_id format")
    except Exception as e:
        logger.error(f"Error processing turn: {e}")
        raise HTTPException(status_code=500, detail="Failed to process turn")


@router.post("/admin/rag_mode", response_model=RAGModeResponse)
async def set_rag_mode(request: RAGModeRequest) -> RAGModeResponse:
    """
    Переключает режим RAG между metadata и vector в runtime.

    Args:
        request: Запрос с флагом use_vector

    Returns:
        Текущий режим RAG после изменения
    """
    try:
        # Изменяем настройку в runtime
        settings.RAG_USE_VECTOR = request.use_vector

        # Определяем название режима для ответа
        current_mode = "vector" if settings.RAG_USE_VECTOR else "metadata"

        logger.info(
            f"RAG mode switched to {current_mode}", use_vector=settings.RAG_USE_VECTOR
        )

        return RAGModeResponse(
            current_mode=current_mode, use_vector=settings.RAG_USE_VECTOR
        )

    except Exception as e:
        logger.error(f"Error setting RAG mode: {e}")
        raise HTTPException(status_code=500, detail="Failed to set RAG mode")


@router.post("/admin/llm_flags", response_model=LLMFlagsResponse)
async def set_llm_flags(request: LLMFlagsRequest) -> LLMFlagsResponse:
    """Set DeepSeek LLM flags for runtime configuration"""
    try:
        # Update settings if provided
        if request.use_reason is not None:
            settings.USE_DEEPSEEK_REASON = request.use_reason
            logger.info(f"DeepSeek reasoning flag set to {request.use_reason}")

        if request.use_gen is not None:
            settings.USE_DEEPSEEK_GEN = request.use_gen
            logger.info(f"DeepSeek generation flag set to {request.use_gen}")

        return LLMFlagsResponse(
            use_reason=settings.USE_DEEPSEEK_REASON, use_gen=settings.USE_DEEPSEEK_GEN
        )

    except Exception as e:
        logger.error(f"Error setting LLM flags: {e}")
        raise HTTPException(status_code=500, detail="Failed to set LLM flags")


@router.get("/report/session/{session_id}")
async def get_session_report(
    session_id: str, db: Annotated[AsyncSession, Depends(get_db)]
) -> Dict[str, Any]:
    """Get evaluation report for a completed session"""
    try:
        # Validate session_id format
        session_uuid = uuid.UUID(session_id)

        # Check if session exists
        session = await db.get(Session, session_uuid)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Compute session metrics
        metrics = await compute_session_metrics(db, session_uuid)

        return {
            "session_id": session_id,
            "case_id": str(session.case_id),
            "metrics": metrics,
            "missed_keys": metrics["missed_keys"],
        }

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    except HTTPException:
        # Re-raise HTTPException to preserve status code
        raise
    except Exception as e:
        logger.error(f"Error generating session report: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate session report")


@router.get("/report/session/{session_id}/missed")
async def get_session_missed_keys(
    session_id: str, db: Annotated[AsyncSession, Depends(get_db)]
) -> Dict[str, Any]:
    """Get missed keys for a completed session"""
    try:
        # Validate session_id format
        session_uuid = uuid.UUID(session_id)

        # Check if session exists
        session = await db.get(Session, session_uuid)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Compute session metrics
        metrics = await compute_session_metrics(db, session_uuid)

        return {
            "session_id": session_id,
            "case_id": str(session.case_id),
            "missed_key_ids": metrics["missed_keys"]["ids"],
            "count": metrics["missed_keys"]["count"],
        }

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    except HTTPException:
        # Re-raise HTTPException to preserve status code
        raise
    except Exception as e:
        logger.error(f"Error getting missed keys: {e}")
        raise HTTPException(status_code=500, detail="Failed to get missed keys")


@router.post("/session/link", response_model=SessionLinkResponse)
async def create_session_link(
    request: SessionLinkRequest, db: Annotated[AsyncSession, Depends(get_db)]
) -> SessionLinkResponse:
    """Create or update session link and return the session chain for the case"""
    try:
        # Validate UUIDs
        session_uuid = uuid.UUID(request.session_id)
        case_uuid = uuid.UUID(request.case_id)
        prev_session_uuid = None
        if request.prev_session_id:
            prev_session_uuid = uuid.UUID(request.prev_session_id)

        # Validate case exists
        case = await db.get(Case, case_uuid)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

        # Validate session exists
        session = await db.get(Session, session_uuid)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Validate prev_session exists if provided
        if prev_session_uuid:
            prev_session = await db.get(Session, prev_session_uuid)
            if not prev_session:
                raise HTTPException(status_code=404, detail="Previous session not found")

        # Create or update session link
        session_link = SessionLink(
            session_id=session_uuid,
            case_id=case_uuid,
            prev_session_id=prev_session_uuid,
        )

        # Use merge to handle upsert behavior
        await db.merge(session_link)
        await db.commit()

        # Get session chain for the case (ordered by created_at)
        stmt = (
            select(SessionLink.session_id)
            .where(SessionLink.case_id == case_uuid)
            .order_by(SessionLink.created_at)
        )
        result = await db.execute(stmt)
        session_chain = [str(row.session_id) for row in result.fetchall()]

        logger.info(
            f"Created/updated session link for session {request.session_id} "
            f"in case {request.case_id}"
        )

        return SessionLinkResponse(case_id=request.case_id, sessions=session_chain)

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")
    except Exception as e:
        logger.error(f"Error creating session link: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session link")


@router.get("/session/{session_id}/trajectory", response_model=SessionTrajectoryResponse)
async def get_session_trajectory(
    session_id: str, db: Annotated[AsyncSession, Depends(get_db)]
) -> SessionTrajectoryResponse:
    """Get trajectory progress for a session"""
    try:
        # Validate session_id format
        session_uuid = uuid.UUID(session_id)

        # Check if session exists
        session = await db.get(Session, session_uuid)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get case and its trajectories to calculate total steps
        case = await db.get(Case, session.case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

        case_truth = case.case_truth
        trajectories = case_truth.get("trajectories", [])

        # Get session trajectory progress
        stmt = select(SessionTrajectory).where(SessionTrajectory.session_id == session_uuid)
        result = await db.execute(stmt)
        session_trajectories = result.scalars().all()

        # Build progress response
        progress = []
        trajectory_lookup = {traj["id"]: traj for traj in trajectories}

        for session_traj in session_trajectories:
            trajectory_id = session_traj.trajectory_id
            total_steps = 0
            if trajectory_id in trajectory_lookup:
                total_steps = len(trajectory_lookup[trajectory_id].get("steps", []))

            progress.append(
                TrajectoryProgressItem(
                    trajectory_id=trajectory_id,
                    completed_steps=session_traj.completed_steps,
                    total=total_steps,
                )
            )

        logger.info(f"Retrieved trajectory progress for session {session_id}")

        return SessionTrajectoryResponse(session_id=session_id, progress=progress)

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    except HTTPException:
        # Re-raise HTTPException to preserve status code
        raise
    except Exception as e:
        logger.error(f"Error getting session trajectory: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session trajectory")


@router.get("/report/case/{case_id}/trajectories", response_model=CaseTrajectoryResponse)
async def get_case_trajectory_report(
    case_id: str, db: Annotated[AsyncSession, Depends(get_db)]
) -> CaseTrajectoryResponse:
    """Get aggregated trajectory data across all sessions for a case"""
    try:
        # Validate case_id format
        case_uuid = uuid.UUID(case_id)

        # Check if case exists
        case = await db.get(Case, case_uuid)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

        # Get all sessions for the case through session links
        stmt = (
            select(SessionLink.session_id)
            .where(SessionLink.case_id == case_uuid)
            .order_by(SessionLink.created_at)
        )
        result = await db.execute(stmt)
        session_ids = [str(row.session_id) for row in result.fetchall()]

        if not session_ids:
            return CaseTrajectoryResponse(
                case_id=case_id, sessions=[], trajectories=[]
            )

        # Get all trajectory progress for these sessions
        session_uuids = [uuid.UUID(sid) for sid in session_ids]
        stmt = select(SessionTrajectory).where(
            SessionTrajectory.session_id.in_(session_uuids)
        )
        result = await db.execute(stmt)
        all_session_trajectories = result.scalars().all()

        # Get case trajectories for total step calculation
        case_truth = case.case_truth
        trajectories = case_truth.get("trajectories", [])
        trajectory_lookup = {traj["id"]: traj for traj in trajectories}

        # Aggregate by trajectory_id
        trajectory_aggregates = {}
        for session_traj in all_session_trajectories:
            trajectory_id = session_traj.trajectory_id
            if trajectory_id not in trajectory_aggregates:
                trajectory_aggregates[trajectory_id] = set()

            # Union of completed steps across all sessions
            trajectory_aggregates[trajectory_id].update(session_traj.completed_steps)

        # Build response with coverage calculation
        trajectory_items = []
        for trajectory_id, completed_steps_set in trajectory_aggregates.items():
            completed_steps_union = list(completed_steps_set)
            total_steps = 0
            if trajectory_id in trajectory_lookup:
                total_steps = len(trajectory_lookup[trajectory_id].get("steps", []))

            coverage = len(completed_steps_union) / total_steps if total_steps > 0 else 0.0

            trajectory_items.append(
                TrajectoryAggregateItem(
                    trajectory_id=trajectory_id,
                    completed_steps_union=completed_steps_union,
                    coverage=round(coverage, 2),
                )
            )

        logger.info(f"Generated trajectory report for case {case_id}")

        return CaseTrajectoryResponse(
            case_id=case_id, sessions=session_ids, trajectories=trajectory_items
        )

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid case_id format")
    except Exception as e:
        logger.error(f"Error generating case trajectory report: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to generate case trajectory report"
        )
