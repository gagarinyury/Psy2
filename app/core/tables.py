import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    case_truth: Mapped[dict[str, Any]] = mapped_column(JSONB)
    policies: Mapped[dict[str, Any]] = mapped_column(JSONB)
    version: Mapped[str] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    kb_fragments = relationship("KBFragment", back_populates="case")
    sessions = relationship("Session", back_populates="case")


class KBFragment(Base):
    __tablename__ = "kb_fragments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cases.id")
    )
    type: Mapped[str] = mapped_column(String(100))
    text: Mapped[str] = mapped_column(Text)
    fragment_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB)
    availability: Mapped[str] = mapped_column(String(100))
    consistency_keys: Mapped[dict[str, Any]] = mapped_column(JSONB)
    embedding: Mapped[Any] = mapped_column(Vector(1024), nullable=True)

    # Relationships
    case = relationship("Case", back_populates="kb_fragments")


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cases.id")
    )
    session_state: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    case = relationship("Case", back_populates="sessions")
    telemetry_turns = relationship("TelemetryTurn", back_populates="session")


class TelemetryTurn(Base):
    __tablename__ = "telemetry_turns"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sessions.id")
    )
    turn_no: Mapped[int] = mapped_column(Integer)
    used_fragments: Mapped[dict[str, Any]] = mapped_column(JSONB)
    risk_status: Mapped[str] = mapped_column(String(100))
    eval_markers: Mapped[dict[str, Any]] = mapped_column(JSONB)
    timings: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=True)
    costs: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    session = relationship("Session", back_populates="telemetry_turns")


class SessionTrajectory(Base):
    __tablename__ = "session_trajectories"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sessions.id"), primary_key=True
    )
    trajectory_id: Mapped[str] = mapped_column(Text, primary_key=True)
    completed_steps: Mapped[list[str]] = mapped_column(
        ARRAY(Text), server_default="{}"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class SessionLink(Base):
    __tablename__ = "session_links"
    __table_args__ = (Index("ix_session_links_case_created", "case_id", "created_at"),)

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cases.id")
    )
    prev_session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
