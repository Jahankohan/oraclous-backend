"""Recipe execution Celery task — TASK-223 / STORY-034 / ADR-022.

A **thin** wrapper around `RecipeExecutionEngine.execute`. All the logic — recipe
validation, structure materialization, mapping projection — lives in the engine
(`app/recipes/engine.py`); this module only:

  * runs in the Celery worker context with a task-scoped, NullPool sync Neo4j
    driver (the `WorkerNeo4jManager` pattern — never a FastAPI-shared driver);
  * adapts the engine's `StructuralRepresentation` input from a plain dict so a
    recipe run is queueable as ordinary JSON-serialisable task arguments;
  * updates the run's `ingestion_jobs` row — running → completed/failed — so a
    recipe run is an observable, `graph_id`-scoped job (TASK-237, closes
    TASK-233 INFO-2);
  * returns the `ExecutionResult` as a dict.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.core.logging import get_logger
from app.recipes.engine import RecipeExecutionEngine
from app.recipes.primitives.interface import StructuralRepresentation

# Reuse the shared Celery app (same broker/backend as every other task).
from app.services.background_jobs import WorkerNeo4jManager, celery_app

logger = get_logger(__name__)


def _update_job(job_id: str, updates: dict[str, Any]) -> None:
    """Update one `ingestion_jobs` row from the Celery worker.

    Uses a task-scoped sync PostgreSQL engine with `NullPool` — the project
    invariant for Celery worker DB access (Architecture Rule #5: no shared
    pool, each task owns its connection).
    """
    sync_pg_url = settings.POSTGRES_URL.replace(
        "postgresql+asyncpg://", "postgresql://"
    )
    sync_engine = create_engine(sync_pg_url, poolclass=NullPool)
    try:
        with sync_engine.connect() as conn:
            cols = ", ".join(f"{k} = :{k}" for k in updates)
            conn.execute(
                text(f"UPDATE ingestion_jobs SET {cols} WHERE id = :job_id"),
                {"job_id": job_id, **updates},
            )
            conn.commit()
    finally:
        sync_engine.dispose()


@celery_app.task(bind=True, name="recipe_tasks.execute_recipe_task")
def execute_recipe_task(
    self,
    recipe: dict[str, Any],
    representation: dict[str, Any],
    graph_id: str,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Execute a recipe over a structural representation and write the graph.

    Args:
        recipe: The recipe document (a JSON object — validated by the engine).
        representation: A `StructuralRepresentation` serialised as a dict
            (`model_dump()` output) — the FULL-mode primitive output.
        graph_id: UUID string of the target tenant graph.
        job_id: UUID string of the recipe-run `ingestion_jobs` row to update
            (TASK-237). When supplied, the row is moved to `processing` on
            start and to `completed` (with the engine's nodes/edges counts) or
            `failed` (with `error_message`) on finish. Optional so the legacy
            3-arg call path keeps working.

    Returns:
        The `ExecutionResult` as a dict (counts + warnings).
    """
    if job_id is not None:
        _update_job(
            job_id,
            {"status": "processing", "started_at": datetime.now(UTC)},
        )

    try:
        rep = StructuralRepresentation.model_validate(representation)
        engine = RecipeExecutionEngine()

        with WorkerNeo4jManager() as neo4j:
            driver = neo4j.get_sync_driver()
            result = engine.execute(recipe, rep, graph_id, driver)
    except Exception as exc:
        if job_id is not None:
            _update_job(
                job_id,
                {
                    "status": "failed",
                    "error_message": str(exc),
                    "completed_at": datetime.now(UTC),
                },
            )
        logger.error(
            "execute_recipe_task: recipe run on graph %s failed — %s",
            graph_id,
            exc,
        )
        raise

    logger.info(
        "execute_recipe_task: recipe %s v%s on graph %s — %d nodes, %d edges written",
        result.recipe_id,
        result.recipe_version,
        graph_id,
        result.nodes_written,
        result.edges_written,
    )

    if job_id is not None:
        _update_job(
            job_id,
            {
                "status": "completed",
                "extracted_entities": result.nodes_written,
                "extracted_relationships": result.edges_written,
                "completed_at": datetime.now(UTC),
            },
        )

    return result.as_dict()
