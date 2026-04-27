"""Add ingest_mode column to ingestion_jobs

Revision ID: add_ingest_mode_to_jobs
Revises: add_ontology_columns_to_ingestion_jobs
Create Date: 2026-04-27 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = 'add_ingest_mode_to_jobs'
down_revision = 'add_ontology_columns_to_ingestion_jobs'
branch_labels = None
depends_on = None


def upgrade():
    # Use execute() for IF NOT EXISTS support — safer on existing envs
    op.execute(
        "ALTER TABLE ingestion_jobs "
        "ADD COLUMN IF NOT EXISTS ingest_mode VARCHAR(20) NOT NULL DEFAULT 'incremental'"
    )


def downgrade():
    op.drop_column('ingestion_jobs', 'ingest_mode')
