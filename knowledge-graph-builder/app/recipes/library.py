"""The ingestion-recipe library (TASK-224, STORY-034, ADR-022).

The library is where authored ingestion recipes are **stored, versioned, and
looked up**. It is the design-time counterpart to the recipe execution engine
(`app/recipes/engine.py`, TASK-223): the engine *runs* a recipe; the library
*holds* it.

Storage decision — STORY-034 open question 2
--------------------------------------------
Recipes live in **Postgres** (table `recipes`, see
`alembic/versions/add_recipes_table.py`), not Neo4j. Per ADR-020 a recipe is
operational configuration — it describes *how* a source projects into the
graph — and is not itself knowledge-graph content. Neo4j holds the projected
graph; Postgres holds the recipe that produced it.

Versioning — recipe-spec §10
----------------------------
A recipe is identified by `(id, version)`. `version` is an integer:

  * a brand-new recipe `id` is stored at `version = 1`;
  * storing again under an existing `id` writes `max(version) + 1`.

The library keeps **every** version — each version is its own row. `store`
always writes a `draft`; `promote` flips one `(id, version)` row to
`promoted`. Promotion never overwrites or deletes a sibling version
(recipe-spec §10: "promotion never overwrites silently").

Tenant scoping
--------------
Every row carries `graph_id` (the authoring tenant); every query here is
`graph_id`-scoped. **Cross-tenant recipe sharing is deliberately deferred** —
`lookup` only returns recipes authored by the asking tenant. A future
iteration may add an opt-in shared recipe catalog; until then a recipe
authored by tenant A is invisible to tenant B, consistent with the
Data Ownership founding principle.

Validation
----------
`store` validates the recipe document against `recipe.schema.json` (the same
schema the execution engine enforces) before persisting. An invalid recipe is
*refused* — `RecipeValidationError` — never coerced.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

import jsonschema
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.recipe import Recipe
from app.recipes.engine import RecipeValidationError

logger = get_logger(__name__)


def _load_schema() -> dict[str, Any]:
    """Load the recipe JSON Schema bundled with the package."""
    text = (
        resources.files("app.recipes")
        .joinpath("recipe.schema.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(text)


# Loaded once at import — the schema is static package data.
_RECIPE_SCHEMA = _load_schema()
_VALIDATOR = jsonschema.Draft202012Validator(_RECIPE_SCHEMA)


def _validate_recipe(recipe: dict[str, Any]) -> None:
    """Validate *recipe* against `recipe.schema.json`.

    Raises `RecipeValidationError` on the first schema violation. The library
    refuses an invalid recipe; it never sanitises one (recipe-spec §5.5).
    """
    if not isinstance(recipe, dict):
        raise RecipeValidationError("recipe must be a JSON object")
    errors = sorted(_VALIDATOR.iter_errors(recipe), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        path = "/".join(str(p) for p in first.path) or "<root>"
        raise RecipeValidationError(
            f"recipe failed schema validation at {path}: {first.message}"
        )


class RecipeLibrary:
    """Store, version, promote, and look up ingestion recipes.

    Stateless apart from the `AsyncSession` passed in at construction — one
    library instance is bound to one session/transaction, the same way the
    repo's other Postgres-backed services take a session from the caller. The
    caller owns commit/rollback; `store` and `promote` `flush()` so the new
    row is visible within the transaction but do not commit on the caller's
    behalf.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -- write paths -------------------------------------------------------

    async def store(self, recipe: dict[str, Any], graph_id: str) -> dict[str, Any]:
        """Insert *recipe* as a new **draft** version for tenant *graph_id*.

        Versioning (recipe-spec §10):
          * a recipe `id` not yet in the library  -> stored at `version = 1`;
          * an `id` already present               -> stored at `max(version) + 1`.

        The recipe is validated against `recipe.schema.json` first; an invalid
        document raises `RecipeValidationError` and nothing is written. The
        stored row is always `status = "draft"` — promotion is a separate,
        reviewed step (`promote`).

        Returns the stored recipe document with `id`, `version` (the
        library-assigned version), and `status` reflecting what was persisted.
        """
        _validate_recipe(recipe)

        recipe_id = recipe["id"]

        # Next version for this id, scoped to the authoring tenant — a recipe
        # id is namespaced by graph_id, so two tenants never collide on it.
        max_version = await self._session.scalar(
            select(func.max(Recipe.version)).where(
                Recipe.id == recipe_id,
                Recipe.graph_id == graph_id,
            )
        )
        next_version = 1 if max_version is None else int(max_version) + 1

        # The stored document is the recipe as authored, with version/status
        # forced to what the library actually persists — the library, not the
        # caller, owns the version counter and the draft status.
        stored_doc = dict(recipe)
        stored_doc["version"] = next_version
        stored_doc["status"] = "draft"

        applies_to = recipe.get("applies_to", {})
        authoring = recipe.get("authoring", {})

        row = Recipe(
            id=recipe_id,
            version=next_version,
            status="draft",
            source_type=applies_to.get("source_type"),
            shape_signature=applies_to.get("shape_signature"),
            concern=recipe.get("concern"),
            recipe_json=stored_doc,
            authored_by=authoring.get("authored_by"),
            graph_id=graph_id,
        )
        self._session.add(row)
        await self._session.flush()

        logger.info(
            "recipe_library: stored draft %s v%d for graph %s",
            recipe_id,
            next_version,
            graph_id,
        )
        return stored_doc

    async def promote(self, recipe_id: str, version: int, graph_id: str) -> None:
        """Mark `(recipe_id, version)` as **promoted** for tenant *graph_id*.

        Promotion flips exactly one row's `status` from `draft` to `promoted`.
        It never overwrites, deletes, or demotes another version — the library
        keeps every version (recipe-spec §10). Promoting an already-promoted
        version is idempotent.

        Raises `KeyError` if no such `(recipe_id, version)` exists for the
        tenant — the library does not silently create a row on promote.
        """
        row = await self._session.get(Recipe, (recipe_id, version))
        if row is None or row.graph_id != graph_id:
            # Tenant mismatch is masked as "not found" — a tenant cannot probe
            # for another tenant's recipe ids (fail-closed).
            raise KeyError(
                f"recipe {recipe_id!r} v{version} not found for graph {graph_id!r}"
            )

        row.status = "promoted"
        # Keep the embedded document's status field consistent with the column.
        doc = dict(row.recipe_json or {})
        doc["status"] = "promoted"
        row.recipe_json = doc
        await self._session.flush()

        logger.info(
            "recipe_library: promoted %s v%d for graph %s",
            recipe_id,
            version,
            graph_id,
        )

    # -- read paths --------------------------------------------------------

    async def get(
        self,
        recipe_id: str,
        graph_id: str,
        version: int | None = None,
    ) -> dict[str, Any] | None:
        """Fetch one recipe version, or the latest if *version* is None.

        Tenant-scoped by *graph_id*. Returns the stored recipe document, or
        `None` if no matching version exists for the tenant.
        """
        stmt = select(Recipe).where(
            Recipe.id == recipe_id,
            Recipe.graph_id == graph_id,
        )
        if version is not None:
            stmt = stmt.where(Recipe.version == version)
        else:
            stmt = stmt.order_by(Recipe.version.desc())
        stmt = stmt.limit(1)

        row = await self._session.scalar(stmt)
        return row.recipe_json if row is not None else None

    async def lookup(
        self,
        source_type: str,
        shape_signature: str,
        concern: str,
        graph_id: str,
    ) -> dict[str, Any] | None:
        """Find the latest **promoted** recipe matching a data shape + concern.

        This is the reuse path of concern-driven ingestion (recipe-spec §4):
        an incoming source with a known `(source_type, shape_signature)` and a
        stated `concern` can reuse an already-authored recipe with no agent
        involvement.

        Matching is exact on all three of `source_type`, `shape_signature`,
        and `concern`, scoped to `graph_id`, and restricted to
        `status = "promoted"` — a draft is a reviewable artifact, never
        executed. When several promoted versions match, the highest `version`
        wins.

        Returns the recipe document, or `None` on a miss.

        Cross-tenant sharing is deliberately deferred: a recipe authored by
        another tenant is never returned, even on an identical shape.
        """
        stmt = (
            select(Recipe)
            .where(
                Recipe.graph_id == graph_id,
                Recipe.source_type == source_type,
                Recipe.shape_signature == shape_signature,
                Recipe.concern == concern,
                Recipe.status == "promoted",
            )
            .order_by(Recipe.version.desc())
            .limit(1)
        )
        row = await self._session.scalar(stmt)
        return row.recipe_json if row is not None else None
