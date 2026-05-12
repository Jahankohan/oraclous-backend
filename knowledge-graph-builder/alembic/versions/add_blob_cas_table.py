"""add blob_cas table for deliverable content-addressable store (TASK-082)

Revision ID: add_blob_cas_table
Revises: ('add_connector_tables', 'add_effective_instructions_to_jobs', 'add_filename_to_ingestion_jobs')
Create Date: 2026-05-12 00:00:00.000000

The :Deliverable nodes carry `content_uri` strings that point into this
table by `sha256`. Per STORY-026 the final HTML+PDF docs from `/docify`
exceed any reasonable Neo4j-property bound (~50 KB), so the bytes live in
Postgres bytea while Neo4j keeps only the small metadata + URI reference.

This revision also merges the three pre-existing alembic heads on
`develop` (connector tables, effective_instructions on jobs, filename on
jobs) into a single lineage so `alembic upgrade head` has one target
again.

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "add_blob_cas_table"
# Tuple form merges the three open heads on develop at the time of writing.
down_revision = (
    "add_connector_tables",
    "add_effective_instructions_to_jobs",
    "add_filename_to_ingestion_jobs",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Composite PK on (graph_id, sha256). The task spec describes `sha256` as
    # the PK and also requires cross-tenant identical content to stay separate
    # per the Data Ownership founding principle. A composite PK is the only
    # way to satisfy both: `sha256` is the natural identifier; `graph_id`
    # namespaces it. The `blob_cas_graph_id` index (single column) keeps the
    # tenant-scoped lookups required by `BlobCASService.get` fast.
    op.create_table(
        "blob_cas",
        sa.Column("sha256", sa.CHAR(length=64), nullable=False),
        sa.Column("graph_id", sa.Text(), nullable=False),
        sa.Column("mime_type", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("content", sa.LargeBinary(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("graph_id", "sha256", name="blob_cas_pkey"),
    )
    op.create_index("blob_cas_graph_id", "blob_cas", ["graph_id"])


def downgrade() -> None:
    op.drop_index("blob_cas_graph_id", table_name="blob_cas")
    op.drop_table("blob_cas")
