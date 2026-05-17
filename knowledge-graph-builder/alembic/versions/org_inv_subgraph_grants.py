"""add subgraph_grants column to org_invitations

Revision ID: org_inv_subgraph_grants
Revises: create_org_invitations_table
Create Date: 2026-05-17 00:00:00.000000

Note: the revision id is kept short on purpose — alembic_version.version_num
is VARCHAR(32), so a longer id fails the version stamp and rolls the whole
migration back.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "org_inv_subgraph_grants"
down_revision = "create_org_invitations_table"
branch_labels = None
depends_on = None


def upgrade():
    # IF NOT EXISTS keeps this idempotent — safe to re-run on an environment
    # where the column was already added out of band.
    op.execute(
        "ALTER TABLE org_invitations ADD COLUMN IF NOT EXISTS subgraph_grants JSONB"
    )


def downgrade():
    op.execute("ALTER TABLE org_invitations DROP COLUMN IF EXISTS subgraph_grants")
