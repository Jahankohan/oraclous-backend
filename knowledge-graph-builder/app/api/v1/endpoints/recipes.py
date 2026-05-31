"""Recipe pipeline REST surface — TASK-236 / STORY-035 / ADR-022.

The STORY-034 recipe pipeline — the recipe **library** (store/version/lookup,
`app/recipes/library.py`) and the recipe **execution engine**
(`app/recipes/engine.py`) — was fully built but never exposed over REST. The
TASK-229 MCP projection projects *existing* REST routes only, so `recipe.*`
could not be projected and STORY-035 acceptance criterion 1 ("a Claude Code
client authors and runs an ingestion recipe") was unsatisfiable.

This router is the **minimal, generic** REST surface that closes the gap. It
exposes an *already-built* capability; it adds no recipe-engine behaviour:

  * `GET  /recipes?graph_id=...`             — list the tenant's recipes;
  * `GET  /recipes/{recipe_id}?graph_id=...` — get one recipe document;
  * `POST /recipes`                          — store a recipe (status `draft`);
  * `POST /graphs/{graph_id}/recipes/{recipe_id}/run`
                                             — run a recipe over an inline
                                               source into the graph (async).

Tenant scoping: recipes are tenant-scoped — every `RecipeLibrary` query is
`graph_id`-scoped (the library's own scope parameter, see
`app/recipes/library.py`). `graph_id` is therefore an explicit input on every
endpoint — a query parameter on list/get, a body field on store, the path
parameter on run — and the caller's access to it is verified before any
recipe row is touched. The list/get endpoints require `read` access; store
and run require `write`. A tenant mismatch inside the library is masked as
"not found" (fail-closed, ADR-022) — defence in depth behind the access check.
"""

from __future__ import annotations

import csv
import io
import json
import tempfile
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    get_current_user_id,
    get_database,
    verify_graph_access,
    verify_graph_write_access,
)
from app.core.logging import get_logger
from app.models.graph import IngestionJob
from app.recipes.authoring import PRIMITIVE_REGISTRY
from app.recipes.engine import RecipeValidationError
from app.recipes.library import RecipeLibrary
from app.recipes.primitives.interface import ExtractionMode
from app.schemas.recipe_schemas import (
    RecipeListResponse,
    RecipeResponse,
    RecipeRunRequest,
    RecipeRunResponse,
    RecipeStoreRequest,
)
from app.tasks.recipe_tasks import execute_recipe_task

router = APIRouter()
logger = get_logger(__name__)

# The run endpoint's inline-records input is decomposed by a deterministic
# primitive. Only the two row/record-shaped primitives accept inline records;
# file-only primitives (code, text, relational) are not a generic inline
# source and are rejected with a clear error.
_INLINE_PRIMITIVE_KINDS = ("csv", "json")


# ==================== RECIPE LIBRARY (design-time) ====================


@router.get(
    "/recipes",
    response_model=RecipeListResponse,
    summary="List a tenant's ingestion recipes",
)
async def list_recipes(
    graph_id: str = Query(..., description="The tenant graph to list recipes for."),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
) -> RecipeListResponse:
    """List the latest version of every recipe authored by *graph_id*.

    Recipes are tenant-scoped; `read` access to the graph is verified first.
    Cross-tenant recipes are never returned (ADR-022).
    """
    await verify_graph_access(graph_id, "read", user_id)
    library = RecipeLibrary(db)
    recipes = await library.list(graph_id)
    return RecipeListResponse(recipes=recipes, count=len(recipes))


@router.get(
    "/recipes/{recipe_id}",
    response_model=RecipeResponse,
    summary="Get one ingestion recipe document",
    responses={404: {"description": "Recipe not found for this tenant"}},
)
async def get_recipe(
    recipe_id: str,
    graph_id: str = Query(..., description="The tenant graph the recipe belongs to."),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
) -> RecipeResponse:
    """Get one recipe document (the latest version) for *graph_id*.

    `read` access to the graph is verified first. A recipe authored by another
    tenant is masked as 404 — the library is `graph_id`-scoped, so a tenant
    cannot probe for another tenant's recipe ids (fail-closed).
    """
    await verify_graph_access(graph_id, "read", user_id)
    library = RecipeLibrary(db)
    recipe = await library.get(recipe_id, graph_id)
    if recipe is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"recipe {recipe_id!r} not found",
        )
    return RecipeResponse(recipe=recipe)


@router.post(
    "/recipes",
    response_model=RecipeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Store a recipe document (as a new draft)",
    responses={400: {"description": "Recipe failed schema validation"}},
)
async def store_recipe(
    request: RecipeStoreRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
) -> RecipeResponse:
    """Store a caller-supplied recipe document as a new **draft** version.

    `write` access to `request.graph_id` is verified first. The library
    validates the document against `recipe.schema.json` and refuses an invalid
    one (`400`). The recipe is always stored as `draft` — promotion stays a
    separate, gated act (ADR-022). The library, not the caller, owns the
    version counter.
    """
    await verify_graph_access(request.graph_id, "write", user_id)
    library = RecipeLibrary(db)
    try:
        stored = await library.store(request.recipe, request.graph_id)
    except RecipeValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from None
    await db.commit()
    return RecipeResponse(recipe=stored)


# ==================== RECIPE EXECUTION (run-time) ====================


def _records_to_source_file(records: list[dict], source_type: str) -> str:
    """Write inline *records* to a temp file the chosen primitive can decompose.

    The primitives are file-path decomposers; the run endpoint takes the source
    **inline** as JSON records, so the records are materialized to a temp file
    of the matching format (CSV or JSON), decomposed FULL-mode, then the file
    is removed. This keeps the run input minimal and generic — no upload, no
    connector (TASK-236 Notes/Decisions).
    """
    if source_type == "csv":
        fieldnames: list[str] = []
        for record in records:
            for key in record:
                if key not in fieldnames:
                    fieldnames.append(key)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)
        suffix, content = ".csv", buffer.getvalue()
    else:  # json
        suffix, content = ".json", json.dumps(records)

    fd, path = tempfile.mkstemp(suffix=suffix, prefix="recipe-run-")
    with open(fd, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


@router.post(
    "/graphs/{graph_id}/recipes/{recipe_id}/run",
    response_model=RecipeRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run a recipe over an inline source into a graph (async)",
    responses={
        400: {"description": "Unsupported source type or empty source"},
        404: {"description": "Recipe not found for this graph"},
    },
)
async def run_recipe(
    graph_id: str,
    recipe_id: str,
    request: RecipeRunRequest,
    db: AsyncSession = Depends(get_database),
    _: str = Depends(verify_graph_write_access),
) -> RecipeRunResponse:
    """Execute a stored recipe over an inline source into a graph, async.

    `write` access to `graph_id` is verified first. The recipe is looked up
    `graph_id`-scoped (a recipe of another tenant is masked 404). The inline
    `records` are decomposed by the matching primitive in FULL mode into a
    `StructuralRepresentation`.

    A recipe run is a first-class `ingestion_jobs` row (TASK-237, closes
    TASK-233 INFO-2): the endpoint creates an `IngestionJob`
    (`source_type="recipe"`, `status="pending"`, `graph_id`-scoped) before
    enqueuing, and returns **that row's id** as `run_id` — not the Celery task
    id. The run is then pollable through the standard `graph_id`-scoped
    `GET /api/v1/graphs/{graph_id}/jobs/{job_id}` endpoint, so it has
    tenant-scoped status and run history like every other async job.
    """
    if request.source_type not in _INLINE_PRIMITIVE_KINDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"unsupported source_type {request.source_type!r} for an inline "
                f"recipe run; supported: {list(_INLINE_PRIMITIVE_KINDS)}"
            ),
        )

    # Look up the recipe, graph_id-scoped — a tenant mismatch masks as 404.
    library = RecipeLibrary(db)
    recipe = await library.get(recipe_id, graph_id, version=request.version)
    if recipe is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"recipe {recipe_id!r} not found",
        )

    # Decompose the inline records with the matching primitive, FULL mode.
    source_path = _records_to_source_file(request.records, request.source_type)
    try:
        primitive = PRIMITIVE_REGISTRY[request.source_type]
        representation = primitive.decompose(source_path, ExtractionMode.FULL)
    finally:
        Path(source_path).unlink(missing_ok=True)

    # Create the recipe-run as a first-class ingestion_jobs row before
    # enqueuing — the row id, not the Celery task id, is the run handle
    # (TASK-237, closes TASK-233 INFO-2). It is graph_id-scoped, so the run is
    # pollable through the standard tenant-scoped jobs endpoint.
    job_id = uuid4()
    job = IngestionJob(
        id=job_id,
        graph_id=UUID(graph_id),
        source_type="recipe",
        status="pending",
        effective_instructions={
            "recipe_id": recipe_id,
            "recipe_version": int(recipe.get("version", 0)),
        },
    )
    db.add(job)
    await db.commit()

    # Enqueue the recipe-execution Celery task — the run completes async and
    # the worker updates this job row to running → completed/failed.
    task = execute_recipe_task.delay(
        recipe=recipe,
        representation=representation.model_dump(mode="json"),
        graph_id=graph_id,
        job_id=str(job_id),
    )
    logger.info(
        "run_recipe: enqueued recipe %s v%s on graph %s — job %s, task %s",
        recipe_id,
        recipe.get("version"),
        graph_id,
        job_id,
        task.id,
    )
    return RecipeRunResponse(
        run_id=str(job_id),
        recipe_id=recipe_id,
        recipe_version=int(recipe.get("version", 0)),
        graph_id=graph_id,
        status="pending",
    )
