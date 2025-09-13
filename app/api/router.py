import uuid
from typing import Annotated, Dict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.core.models import (
    CaseRequest,
    CaseResponse,
    SessionRequest,
    SessionResponse,
    TurnRequest,
    TurnResponse,
    SessionStateCompact,
)
from app.core.tables import Case, Session
from app.infra.metrics import CASE_OPERATIONS, SESSION_OPERATIONS, TURN_OPERATIONS
from app.infra.logging import get_logger
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
