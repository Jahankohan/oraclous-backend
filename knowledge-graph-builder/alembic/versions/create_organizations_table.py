"""create organizations table (TASK-201)

Revision ID: create_organizations_table
Revises: chat_persistence_drop_graph_fk
Create Date: 2026-05-17 00:00:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

# revision identifiers, used by Alembic.
revision = "create_organizations_table"
down_revision = "chat_persistence_drop_graph_fk"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "organizations",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("owner_user_id", UUID(as_uuid=True), nullable=False),
        sa.Column(
            "settings",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("status", sa.String(length=50), server_default="active"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")
        ),
    )
    op.create_index(
        "idx_organizations_owner_user_id", "organizations", ["owner_user_id"]
    )


def downgrade():
    op.drop_index("idx_organizations_owner_user_id", table_name="organizations")
    op.drop_table("organizations")
