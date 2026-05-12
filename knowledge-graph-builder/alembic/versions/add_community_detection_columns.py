"""add community detection columns to knowledge_graphs

Revision ID: add_community_detection_columns
Revises: add_optimization_columns
Create Date: 2026-04-07 00:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "add_community_detection_columns"
down_revision = "add_optimization_columns"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "knowledge_graphs",
        sa.Column("communities_detected_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "knowledge_graphs",
        sa.Column(
            "communities_status",
            sa.String(20),
            nullable=True,
            server_default="not_detected",
        ),
    )
    op.add_column(
        "knowledge_graphs",
        sa.Column(
            "entity_count_at_detection",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "knowledge_graphs",
        sa.Column(
            "entity_delta_since_detection",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade():
    op.drop_column("knowledge_graphs", "entity_delta_since_detection")
    op.drop_column("knowledge_graphs", "entity_count_at_detection")
    op.drop_column("knowledge_graphs", "communities_status")
    op.drop_column("knowledge_graphs", "communities_detected_at")
