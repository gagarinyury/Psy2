import uuid
from typing import Annotated, Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.db import get_db
from app.core.models import (
    CaseRequest, CaseResponse, 
    SessionRequest, SessionResponse,
    TurnRequest, TurnResponse,
    SessionStateCompact
)
from app.core.tables import Case, Session, TelemetryTurn
from app.infra.metrics import CASE_OPERATIONS, SESSION_OPERATIONS, TURN_OPERATIONS
from app.infra.logging import get_logger

logger = get_logger()
router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "ok"}


@router.post("/case", response_model=CaseResponse)
async def create_case(
    request: CaseRequest,
    db: Annotated[AsyncSession, Depends(get_db)]
) -> CaseResponse:
    """Create a new case with CaseTruth validation"""
    try:
        # Create case record
        case = Case(
            case_truth=request.case_truth.model_dump(),
            policies=request.policies or {},
            version="1.0"
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
    request: SessionRequest,
    db: Annotated[AsyncSession, Depends(get_db)]
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
            last_turn_summary=""
        )
        
        # Create session record
        session = Session(
            case_id=case_uuid,
            session_state=default_state.model_dump()
        )
        
        db.add(session)
        await db.commit()
        await db.refresh(session)
        
        # Metrics
        SESSION_OPERATIONS.labels(operation="create").inc()
        
        logger.info(f"Created session with ID: {session.id} for case: {request.case_id}")
        
        return SessionResponse(session_id=str(session.id))
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid case_id format")
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")


@router.post("/turn", response_model=TurnResponse)
async def process_turn(
    request: TurnRequest,
    db: Annotated[AsyncSession, Depends(get_db)]
) -> TurnResponse:
    """Process a turn request - stub implementation"""
    try:
        # Validate case exists
        case_uuid = uuid.UUID(request.case_id)
        case = await db.get(Case, case_uuid)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Find latest session for this case (simplified approach)
        # In real implementation, session_id would be provided in request
        result = await db.execute(
            text("SELECT id FROM sessions WHERE case_id = :case_id ORDER BY created_at DESC LIMIT 1"),
            {"case_id": case_uuid}
        )
        session_row = result.fetchone()
        
        if not session_row:
            raise HTTPException(status_code=404, detail="No session found for case")
        
        session_id = session_row[0]
        
        # Get next turn number
        result = await db.execute(
            text("SELECT COALESCE(MAX(turn_no), 0) + 1 FROM telemetry_turns WHERE session_id = :session_id"),
            {"session_id": session_id}
        )
        turn_no = result.scalar()
        
        # Create telemetry record
        telemetry = TelemetryTurn(
            session_id=session_id,
            turn_no=turn_no,
            used_fragments=[],  # Empty for stub
            risk_status="none",
            eval_markers={"stub": True},
            timings={},
            costs={}
        )
        
        db.add(telemetry)
        await db.commit()
        
        # Generate stub response - echo first 20 chars of therapist_utterance
        patient_reply = f"Echo: {request.therapist_utterance[:20]}..."
        
        # Metrics
        TURN_OPERATIONS.labels(operation="process").inc()
        
        logger.info(f"Processed turn {turn_no} for session: {session_id}")
        
        return TurnResponse(
            patient_reply=patient_reply,
            state_updates={"trust_delta": 0.0},
            used_fragments=[],
            risk_status="none",
            eval_markers={"stub": True}
        )
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid case_id format")
    except Exception as e:
        logger.error(f"Error processing turn: {e}")
        raise HTTPException(status_code=500, detail="Failed to process turn")