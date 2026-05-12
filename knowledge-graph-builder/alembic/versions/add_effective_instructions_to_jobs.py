"""Add effective_instructions provenance column to ingestion_jobs

Revision ID: add_effective_instructions_to_jobs
Revises: add_optimization_columns
Create Date: 2026-04-07 00:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "add_effective_instructions_to_jobs"
down_revision = "add_optimization_columns"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "ingestion_jobs",
        sa.Column("effective_instructions", sa.JSON(), nullable=True),
    )


def downgrade():
    op.drop_column("ingestion_jobs", "effective_instructions")
