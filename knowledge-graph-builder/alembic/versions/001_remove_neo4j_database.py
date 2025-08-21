"""Remove neo4j_database column

Revision ID: 002_remove_neo4j_database
Revises: 001_initial_migration
Create Date: 2024-01-01 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '002_remove_neo4j_database'
down_revision = '001_initial_migration'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Remove the neo4j_database column since we're using single database
    op.drop_column('knowledge_graphs', 'neo4j_database')

def downgrade() -> None:
    # Add back the column if we need to rollback
    op.add_column('knowledge_graphs', 
                  sa.Column('neo4j_database', sa.String(100), nullable=True))
