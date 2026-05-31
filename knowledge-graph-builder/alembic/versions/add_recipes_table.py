"""add recipes table — the ingestion-recipe library (TASK-224, STORY-034)

Revision ID: add_recipes_table
Revises: add_org_slug
Create Date: 2026-05-22 00:00:00.000000

Concern-driven ingestion (ADR-022). The recipe library stores authored
ingestion recipes, versions them, and looks them up by data shape.

Storage decision (STORY-034 open question 2): recipes live in **Postgres**,
not Neo4j — per ADR-020 a recipe is operational configuration, not
knowledge-graph content.

The table is keyed on `(id, version)`: each version is its own row, so the
library keeps every version (recipe-spec §10) and promotion never overwrites
a sibling. `idx_recipes_lookup` covers the tenant-scoped data-shape lookup
(`RecipeLibrary.lookup`).
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

# revision identifiers, used by Alembic.
revision = "add_recipes_table"
down_revision = "add_org_slug"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "recipes",
        # Composite PK — `id` is the recipe id (`rcp_...`); `version` is the
        # integer recipe version. A new version is a new row.
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        # Lifecycle status — draft | promoted (recipe-spec §2).
        sa.Column("status", sa.Text(), nullable=False, server_default="draft"),
        # The data-shape match (`applies_to`), denormalized for the lookup index.
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("shape_signature", sa.Text(), nullable=False),
        sa.Column("concern", sa.Text(), nullable=False),
        # The full recipe document — the source of truth for the recipe.
        sa.Column("recipe_json", JSONB(), nullable=False),
        # Provenance — author and authoring tenant.
        sa.Column("authored_by", sa.Text(), nullable=True),
        sa.Column("graph_id", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", "version", name="recipes_pkey"),
    )
    # Covers RecipeLibrary.lookup — latest promoted recipe for a tenant by
    # (source_type, shape_signature, status).
    op.create_index(
        "idx_recipes_lookup",
        "recipes",
        ["graph_id", "source_type", "shape_signature", "status"],
    )


def downgrade() -> None:
    op.drop_index("idx_recipes_lookup", table_name="recipes")
    op.drop_table("recipes")
