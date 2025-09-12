"""Initial tables

Revision ID: 001
Revises: 
Create Date: 2025-09-12 20:58:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector
import uuid

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable vector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Create cases table
    op.create_table('cases',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('case_truth', postgresql.JSONB(), nullable=False),
        sa.Column('policies', postgresql.JSONB(), nullable=False),
        sa.Column('version', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create kb_fragments table
    op.create_table('kb_fragments',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('case_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('type', sa.String(length=100), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('metadata', postgresql.JSONB(), nullable=False),
        sa.Column('availability', sa.String(length=100), nullable=False),
        sa.Column('consistency_keys', postgresql.JSONB(), nullable=False),
        sa.Column('embedding', Vector(1024), nullable=True),
        sa.ForeignKeyConstraint(['case_id'], ['cases.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create sessions table  
    op.create_table('sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('case_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_state', postgresql.JSONB(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['case_id'], ['cases.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create telemetry_turns table
    op.create_table('telemetry_turns',
        sa.Column('id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('turn_no', sa.Integer(), nullable=False),
        sa.Column('used_fragments', postgresql.JSONB(), nullable=False),
        sa.Column('risk_status', sa.String(length=100), nullable=False),
        sa.Column('eval_markers', postgresql.JSONB(), nullable=False),
        sa.Column('timings', postgresql.JSONB(), nullable=True),
        sa.Column('costs', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create GIN indexes on JSONB columns
    op.create_index('ix_cases_case_truth', 'cases', ['case_truth'], postgresql_using='gin')
    op.create_index('ix_cases_policies', 'cases', ['policies'], postgresql_using='gin')
    op.create_index('ix_kb_fragments_metadata', 'kb_fragments', ['metadata'], postgresql_using='gin')
    op.create_index('ix_kb_fragments_consistency_keys', 'kb_fragments', ['consistency_keys'], postgresql_using='gin')
    op.create_index('ix_sessions_session_state', 'sessions', ['session_state'], postgresql_using='gin')
    op.create_index('ix_telemetry_used_fragments', 'telemetry_turns', ['used_fragments'], postgresql_using='gin')
    op.create_index('ix_telemetry_eval_markers', 'telemetry_turns', ['eval_markers'], postgresql_using='gin')
    op.create_index('ix_telemetry_timings', 'telemetry_turns', ['timings'], postgresql_using='gin')
    op.create_index('ix_telemetry_costs', 'telemetry_turns', ['costs'], postgresql_using='gin')
    
    # Create vector index on embeddings (HNSW for better performance)
    op.execute('CREATE INDEX ix_kb_fragments_embedding_hnsw ON kb_fragments USING hnsw (embedding vector_cosine_ops)')


def downgrade() -> None:
    # Drop indexes first
    op.drop_index('ix_kb_fragments_embedding_hnsw')
    op.drop_index('ix_telemetry_costs')
    op.drop_index('ix_telemetry_timings')  
    op.drop_index('ix_telemetry_eval_markers')
    op.drop_index('ix_telemetry_used_fragments')
    op.drop_index('ix_sessions_session_state')
    op.drop_index('ix_kb_fragments_consistency_keys')
    op.drop_index('ix_kb_fragments_metadata')
    op.drop_index('ix_cases_policies')
    op.drop_index('ix_cases_case_truth')
    
    # Drop tables in reverse order
    op.drop_table('telemetry_turns')
    op.drop_table('sessions')
    op.drop_table('kb_fragments')
    op.drop_table('cases')
    
    # Don't drop vector extension as it might be used by other applications