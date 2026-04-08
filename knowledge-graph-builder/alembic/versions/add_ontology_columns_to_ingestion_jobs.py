"""Add ontology_violations and ontology_coercions to ingestion_jobs

Revision ID: add_ontology_columns_to_ingestion_jobs
Revises: add_community_detection_columns
Create Date: 2026-04-07 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_ontology_columns_to_ingestion_jobs'
down_revision = 'add_community_detection_columns'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'ingestion_jobs',
        sa.Column('ontology_violations', sa.Integer(), nullable=False, server_default='0'),
    )
    op.add_column(
        'ingestion_jobs',
        sa.Column('ontology_coercions', sa.Integer(), nullable=False, server_default='0'),
    )


def downgrade():
    op.drop_column('ingestion_jobs', 'ontology_coercions')
    op.drop_column('ingestion_jobs', 'ontology_violations')
