from typing import Any, Optional
from pydantic import BaseModel


class CaseTruth(BaseModel):
    dx_target: list[str]
    ddx: dict[str, float]
    hidden_facts: list[str]
    red_flags: list[str]
    trajectories: list[str]


class SessionStateCompact(BaseModel):
    affect: str
    trust: float
    fatigue: float
    access_level: int
    risk_status: str
    last_turn_summary: str


class TurnRequest(BaseModel):
    therapist_utterance: str
    session_state: SessionStateCompact
    case_id: str
    options: Optional[dict[str, Any]] = None


class TurnResponse(BaseModel):
    patient_reply: str
    state_updates: dict[str, Any]
    used_fragments: list[str]
    risk_status: str
    eval_markers: dict[str, Any]


# API Request/Response models
class CaseRequest(BaseModel):
    case_truth: CaseTruth
    policies: dict[str, Any] = {}


class CaseResponse(BaseModel):
    case_id: str


class SessionRequest(BaseModel):
    case_id: str


class SessionResponse(BaseModel):
    session_id: str