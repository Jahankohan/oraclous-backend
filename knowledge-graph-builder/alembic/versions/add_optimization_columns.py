"""add optimization columns to knowledge graphs

Revision ID: add_optimization_columns
Revises: be711928c61e
Create Date: 2025-08-27 12:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "add_optimization_columns"
down_revision = "be711928c61e"
branch_labels = None
depends_on = None


def upgrade():
    # Add new columns to knowledge_graphs table
    op.add_column(
        "knowledge_graphs",
        sa.Column(
            "similarity_relationships", sa.Integer(), nullable=False, server_default="0"
        ),
    )
    op.add_column(
        "knowledge_graphs",
        sa.Column(
            "communities_count", sa.Integer(), nullable=False, server_default="0"
        ),
    )


def downgrade():
    # Remove the columns
    op.drop_column("knowledge_graphs", "communities_count")
    op.drop_column("knowledge_graphs", "similarity_relationships")
