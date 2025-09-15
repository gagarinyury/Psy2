"""add trajectory tables

Revision ID: 38768b431ef5
Revises: bb28c240b2bc
Create Date: 2025-09-14 09:32:56.425853

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '38768b431ef5'
down_revision = 'bb28c240b2bc'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create session_trajectories table
    op.create_table(
        "session_trajectories",
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("trajectory_id", sa.Text(), nullable=False),
        sa.Column(
            "completed_steps",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'")
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            # Note: onupdate trigger will be handled at application level
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.id"],
        ),
        sa.PrimaryKeyConstraint("session_id", "trajectory_id"),
    )

    # Create session_links table
    op.create_table(
        "session_links",
        sa.Column(
            "session_id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False
        ),
        sa.Column("case_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("prev_session_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["case_id"],
            ["cases.id"],
        ),
        sa.ForeignKeyConstraint(
            ["prev_session_id"],
            ["sessions.id"],
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.id"],
        ),
        sa.PrimaryKeyConstraint("session_id"),
    )

    # Create index on (case_id, created_at) for session_links
    op.create_index(
        "ix_session_links_case_created",
        "session_links",
        ["case_id", "created_at"]
    )


def downgrade() -> None:
    # Drop index first
    op.drop_index("ix_session_links_case_created")

    # Drop tables in reverse order
    op.drop_table("session_links")
    op.drop_table("session_trajectories")