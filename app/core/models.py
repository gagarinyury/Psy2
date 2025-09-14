from typing import Any, Optional

from pydantic import BaseModel

from app.core.policies import Policies


class TrajectoryStep(BaseModel):
    id: str
    name: str
    condition_tags: list[str] = []
    min_trust: float = 0.4


class Trajectory(BaseModel):
    id: str
    name: str
    steps: list[TrajectoryStep] = []


class CaseTruth(BaseModel):
    dx_target: list[str]
    ddx: dict[str, float]
    hidden_facts: list[str]
    red_flags: list[str]
    trajectories: list[Trajectory] = []


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
    session_id: str
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
    policies: Policies


class CaseResponse(BaseModel):
    case_id: str


class SessionRequest(BaseModel):
    case_id: str


class SessionResponse(BaseModel):
    session_id: str


class RAGModeRequest(BaseModel):
    use_vector: bool


class RAGModeResponse(BaseModel):
    current_mode: str
    use_vector: bool


class LLMFlagsRequest(BaseModel):
    use_reason: Optional[bool] = None
    use_gen: Optional[bool] = None


class LLMFlagsResponse(BaseModel):
    use_reason: bool
    use_gen: bool


# Trajectory-related API models
class SessionLinkRequest(BaseModel):
    session_id: str
    case_id: str
    prev_session_id: Optional[str] = None


class SessionLinkResponse(BaseModel):
    case_id: str
    sessions: list[str]


class TrajectoryProgressItem(BaseModel):
    trajectory_id: str
    completed_steps: list[str]
    total: int


class SessionTrajectoryResponse(BaseModel):
    session_id: str
    progress: list[TrajectoryProgressItem]


class TrajectoryAggregateItem(BaseModel):
    trajectory_id: str
    completed_steps_union: list[str]
    coverage: float


class CaseTrajectoryResponse(BaseModel):
    case_id: str
    sessions: list[str]
    trajectories: list[TrajectoryAggregateItem]
