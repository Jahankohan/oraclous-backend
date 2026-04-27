"""Add optimization tracking fields

Revision ID: be711928c61e
Revises: 
Create Date: 2025-08-27 06:51:07.841929

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'be711928c61e'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add optimization tracking fields to knowledge_graphs table."""
    # Add optimization tracking columns
    op.add_column('knowledge_graphs', sa.Column('last_optimized', sa.DateTime(timezone=True), nullable=True))
    op.add_column('knowledge_graphs', sa.Column('optimization_count', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('knowledge_graphs', sa.Column('last_optimization_type', sa.String(length=50), nullable=True))


def downgrade() -> None:
    """Remove optimization tracking fields from knowledge_graphs table."""
    op.drop_column('knowledge_graphs', 'last_optimization_type')
    op.drop_column('knowledge_graphs', 'optimization_count')
    op.drop_column('knowledge_graphs', 'last_optimized')
