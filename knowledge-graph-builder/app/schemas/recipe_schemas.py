"""Pydantic request/response schemas for the recipe REST surface (TASK-236).

The STORY-034 recipe pipeline — the library (`app/recipes/library.py`) and the
execution engine (`app/recipes/engine.py`) — had no REST surface, so the
TASK-229 MCP projection could not expose `recipe.*`. These schemas type the
minimal REST surface added by TASK-236 (STORY-035, ADR-022/023/024):

  * list / get / store recipe documents (the library's design-time half);
  * run a stored recipe over an inline source into a graph (the engine).

Per ADR-023 D4 every projected MCP tool publishes a fully typed input schema;
the recipe endpoints therefore use typed request/response models — no untyped
`body: dict`. A recipe *document* itself is an open JSON object validated
against `recipe.schema.json` by the library, so `RecipeStoreRequest.recipe`
stays a `dict[str, Any]` deliberately — the JSON Schema, not Pydantic, is the
recipe contract (recipe-spec §5).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RecipeStoreRequest(BaseModel):
    """Request body for `POST /recipes` — store a caller-supplied recipe.

    `recipe` is the full recipe document. The library validates it against
    `recipe.schema.json` and refuses an invalid document; it always persists
    as a `draft` (ADR-022 — promotion is a separate gated act). The library
    owns the `version` counter, so a `version`/`status` in the supplied
    document is overwritten by what is actually persisted.

    Recipes are tenant-scoped — `graph_id` is the authoring tenant the library
    namespaces the recipe under, exactly as `RecipeLibrary.store` requires.
    Write access to `graph_id` is verified before the recipe is stored.
    """

    graph_id: str = Field(
        ...,
        description="The authoring tenant graph the recipe is stored under.",
    )
    recipe: dict[str, Any] = Field(
        ...,
        description=(
            "The recipe document — a JSON object validated against "
            "recipe.schema.json. Stored as a new draft version."
        ),
    )


class RecipeResponse(BaseModel):
    """One stored recipe document — the response of get / store.

    The recipe document is returned as-is (the `recipe_json` the library
    persisted), so the caller sees the library-assigned `version` and the
    `draft` status.
    """

    recipe: dict[str, Any] = Field(
        ..., description="The stored recipe document (recipe_json)."
    )


class RecipeListResponse(BaseModel):
    """Response of `GET /recipes` — the latest version of every recipe.

    Tenant-scoped: only recipes authored by the calling tenant's graph are
    listed (cross-tenant recipe sharing is deliberately deferred).
    """

    recipes: list[dict[str, Any]] = Field(
        default_factory=list,
        description="One recipe document per recipe id — the highest version.",
    )
    count: int = Field(..., description="Number of recipes returned.")


class RecipeRunRequest(BaseModel):
    """Request body for `POST /graphs/{graph_id}/recipes/{recipe_id}/run`.

    The execution engine projects a `StructuralRepresentation` into the graph.
    The simplest generic run input is the source content **inline** in the
    request: a list of structured records plus the source type. The endpoint
    runs the matching deterministic primitive (FULL mode) over those records
    to produce the representation, then enqueues the recipe-execution Celery
    task. No file upload, no connector — inline content keeps the run input
    minimal and generic (TASK-236 Notes/Decisions).
    """

    source_type: str = Field(
        ...,
        description=(
            "The primitive to decompose the inline records with — `csv` or "
            "`json`. Must match the recipe's `applies_to.source_type`."
        ),
    )
    records: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description=(
            "The source content, inline — a non-empty list of flat record "
            "objects. Each object is one row/record the recipe projects."
        ),
    )
    version: int | None = Field(
        default=None,
        description=("Recipe version to run. Omitted runs the latest stored version."),
    )


class RecipeRunResponse(BaseModel):
    """Response of the recipe-run endpoint — an enqueued async job.

    The run executes asynchronously in the Celery worker. A recipe run is a
    first-class `ingestion_jobs` row (TASK-237): `run_id` is that row's id,
    `graph_id`-scoped, and is pollable through the standard
    `GET /api/v1/graphs/{graph_id}/jobs/{job_id}` endpoint — not a raw Celery
    task id.
    """

    run_id: str = Field(
        ...,
        description=(
            "The recipe-run ingestion-job id. Poll "
            "`GET /api/v1/graphs/{graph_id}/jobs/{run_id}` for status."
        ),
    )
    recipe_id: str = Field(..., description="The recipe being run.")
    recipe_version: int = Field(..., description="The recipe version being run.")
    graph_id: str = Field(..., description="The target tenant graph.")
    status: str = Field(
        default="pending",
        description="Run lifecycle status — `pending` on enqueue.",
    )
