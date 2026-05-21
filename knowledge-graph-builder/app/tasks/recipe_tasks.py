"""Recipe execution Celery task — TASK-223 / STORY-034 / ADR-022.

A **thin** wrapper around `RecipeExecutionEngine.execute`. All the logic — recipe
validation, structure materialization, mapping projection — lives in the engine
(`app/recipes/engine.py`); this module only:

  * runs in the Celery worker context with a task-scoped, NullPool sync Neo4j
    driver (the `WorkerNeo4jManager` pattern — never a FastAPI-shared driver);
  * adapts the engine's `StructuralRepresentation` input from a plain dict so a
    recipe run is queueable as ordinary JSON-serialisable task arguments;
  * returns the `ExecutionResult` as a dict.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.recipes.engine import RecipeExecutionEngine
from app.recipes.primitives.interface import StructuralRepresentation

# Reuse the shared Celery app (same broker/backend as every other task).
from app.services.background_jobs import WorkerNeo4jManager, celery_app

logger = get_logger(__name__)


@celery_app.task(bind=True, name="recipe_tasks.execute_recipe_task")
def execute_recipe_task(
    self,
    recipe: dict[str, Any],
    representation: dict[str, Any],
    graph_id: str,
) -> dict[str, Any]:
    """Execute a recipe over a structural representation and write the graph.

    Args:
        recipe: The recipe document (a JSON object — validated by the engine).
        representation: A `StructuralRepresentation` serialised as a dict
            (`model_dump()` output) — the FULL-mode primitive output.
        graph_id: UUID string of the target tenant graph.

    Returns:
        The `ExecutionResult` as a dict (counts + warnings).
    """
    rep = StructuralRepresentation.model_validate(representation)
    engine = RecipeExecutionEngine()

    with WorkerNeo4jManager() as neo4j:
        driver = neo4j.get_sync_driver()
        result = engine.execute(recipe, rep, graph_id, driver)

    logger.info(
        "execute_recipe_task: recipe %s v%s on graph %s — %d nodes, %d edges written",
        result.recipe_id,
        result.recipe_version,
        graph_id,
        result.nodes_written,
        result.edges_written,
    )
    return result.as_dict()
