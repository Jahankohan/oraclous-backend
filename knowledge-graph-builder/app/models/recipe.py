"""SQLAlchemy model for the ingestion-recipe library (TASK-224, STORY-034).

Per ADR-022 an ingestion recipe is *data, not code*: a declarative JSON
document the recipe execution engine (TASK-223) interprets. The recipe
**library** is where authored recipes are stored, versioned, and looked up.

Storage decision (STORY-034 open question 2): recipes live in **Postgres**,
not Neo4j. Per ADR-020 a recipe is operational configuration — it describes
*how* a source projects into the graph; it is not knowledge-graph content.

A recipe is identified by `(id, version)`:

  * `id`     — the recipe id (`rcp_...`), stable across versions;
  * `version` — an integer, 1 on first store, `max(version) + 1` on each
    subsequent store of the same `id`. Promotion bumps nothing on its own —
    the library keeps *every* version (recipe-spec §10), so promotion never
    overwrites a sibling version silently.

Tenant isolation: every row carries `graph_id` (the authoring tenant). Every
`RecipeLibrary` query is `graph_id`-scoped. Cross-tenant recipe sharing is
deliberately deferred (see `app/recipes/library.py`).
"""

from sqlalchemy import Column, DateTime, Index, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.core.database import Base


class Recipe(Base):
    """One stored version of an ingestion recipe.

    The composite primary key `(id, version)` is what keeps every version: a
    new version is a new row, never an UPDATE of an existing one. `status`
    moves `draft -> promoted` in place — promotion is the only mutation the
    library performs, and it touches one (id, version) row.
    """

    __tablename__ = "recipes"

    # Composite PK — `(id, version)`. `id` is the recipe id (`rcp_...`);
    # `version` is the integer recipe version (recipe-spec §3, §10).
    id = Column(Text, primary_key=True, nullable=False)
    version = Column(Integer, primary_key=True, nullable=False)

    # Lifecycle status — draft | promoted (recipe-spec §2).
    status = Column(Text, nullable=False, server_default="draft")

    # The data-shape match (`applies_to`) the library looks up on
    # (recipe-spec §4). Denormalized out of `recipe_json` so the lookup index
    # can cover them.
    source_type = Column(Text, nullable=False)
    shape_signature = Column(Text, nullable=False)

    # The natural-language concern this recipe was authored for (recipe-spec §3).
    concern = Column(Text, nullable=False)

    # The full recipe document — the single source of truth for the recipe.
    recipe_json = Column(JSONB, nullable=False)

    # Provenance: who authored the recipe (typically the `data-specialist`
    # agent) and the authoring tenant.
    authored_by = Column(Text, nullable=True)
    graph_id = Column(Text, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        # Covers `RecipeLibrary.lookup` — find the latest promoted recipe for a
        # tenant matching a (source_type, shape_signature) under a status.
        Index(
            "idx_recipes_lookup",
            "graph_id",
            "source_type",
            "shape_signature",
            "status",
        ),
    )
