"""create org_invitations table (member invitations)

Revision ID: create_org_invitations_table
Revises: create_organizations_table
Create Date: 2026-05-17 00:00:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

# revision identifiers, used by Alembic.
revision = "create_org_invitations_table"
down_revision = "create_organizations_table"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "org_invitations",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("org_id", UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column(
            "org_role", sa.String(length=64), nullable=False, server_default="member"
        ),
        sa.Column("token", sa.String(length=64), nullable=False),
        sa.Column(
            "status", sa.String(length=32), nullable=False, server_default="pending"
        ),
        sa.Column("invited_by_user_id", UUID(as_uuid=True), nullable=False),
        sa.Column("accepted_by_user_id", UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_org_invitations_org_id", "org_invitations", ["org_id"])
    op.create_index(
        "idx_org_invitations_token", "org_invitations", ["token"], unique=True
    )


def downgrade():
    op.drop_index("idx_org_invitations_token", table_name="org_invitations")
    op.drop_index("idx_org_invitations_org_id", table_name="org_invitations")
    op.drop_table("org_invitations")
