"""Add filename to ingestion_jobs

Revision ID: add_filename_to_ingestion_jobs
Revises: add_ingest_mode_to_jobs
Create Date: 2026-04-27 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = 'add_filename_to_ingestion_jobs'
down_revision = 'add_ingest_mode_to_jobs'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('ingestion_jobs', sa.Column('filename', sa.String(length=512), nullable=True))


def downgrade() -> None:
    op.drop_column('ingestion_jobs', 'filename')
