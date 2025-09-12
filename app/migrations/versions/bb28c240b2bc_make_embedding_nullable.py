"""Make embedding nullable

Revision ID: bb28c240b2bc
Revises: 001
Create Date: 2025-09-12 23:08:04.756118

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'bb28c240b2bc'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Make embedding column nullable
    op.alter_column('kb_fragments', 'embedding', nullable=True)


def downgrade() -> None:
    # Make embedding column not nullable (reverse operation)
    op.alter_column('kb_fragments', 'embedding', nullable=False)