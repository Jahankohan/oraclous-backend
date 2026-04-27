"""
Federation Celery Tasks

Async SAME_AS entity resolution across two knowledge graphs.

Architecture rules:
- Uses async Neo4j driver via WorkerNeo4jManager — Celery worker context
- Never shares drivers with FastAPI
- Every Cypher query filters by graph_id (multi-tenancy)
- Task-scoped connections: open on entry, close in finally block
"""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.services.background_jobs import WorkerNeo4jManager, celery_app
from app.services.task_executor import AsyncTaskExecutor

logger = get_logger(__name__)


@celery_app.task(bind=True, name="federation_tasks.resolve_same_as_task")
def resolve_same_as_task(
    self,
    graph_id: str,
    target_graph_id: str,
    confidence_threshold: float = 0.85,
) -> dict[str, Any]:
    """Celery task: resolve SAME_AS links between graph_id and target_graph_id.

    Iterates all entities in *graph_id*, finds candidates in *target_graph_id*
    via exact-name + vector search, scores each pair with the four-signal
    EntityResolver, and creates SAME_AS links for scores at or above
    *confidence_threshold*.

    Args:
        graph_id:             Source graph to scan.
        target_graph_id:      Target graph to match entities against.
        confidence_threshold: Minimum confidence for link creation; clamped to
                              [0.60, 1.0] before this task is called (see
                              FederationResolveRequest.clamp_threshold).

    Returns:
        Dict with keys:
          created_links   — list of {entity_a_id, entity_b_id, confidence}
          ambiguous_count — pairs rejected by LLM disambiguation
          rejected_count  — pairs below AMBIGUOUS_LOWER threshold
    """
    return AsyncTaskExecutor.run_async_task(
        _resolve_same_as_async,
        self,
        graph_id,
        target_graph_id,
        confidence_threshold,
    )


async def _resolve_same_as_async(
    task,
    graph_id: str,
    target_graph_id: str,
    confidence_threshold: float,
) -> dict[str, Any]:
    """Async implementation of SAME_AS resolution.

    Opens a task-scoped async Neo4j connection (WorkerNeo4jManager), iterates
    entities in the source graph, finds candidates in the target graph, then
    delegates scoring and link creation to EntityResolver.resolve_and_link().
    """
    from app.components.entity_resolver import AMBIGUOUS_LOWER, EntityResolver
    from app.schemas.federation_schemas import SameAsCandidate

    created_links: list[dict[str, Any]] = []
    ambiguous_count = 0
    rejected_count = 0

    async with WorkerNeo4jManager() as neo4j:
        async_driver = neo4j.get_async_driver()

        async with async_driver.session(database=settings.NEO4J_DATABASE) as session:
            # Step 1: Fetch all entities (with embeddings) from the source graph.
            entities = await _fetch_entities_with_embeddings(session, graph_id)

            logger.info(
                "resolve_same_as_task: %d entities in source graph %s",
                len(entities),
                graph_id,
            )

            if not entities:
                return {
                    "created_links": [],
                    "ambiguous_count": 0,
                    "rejected_count": 0,
                }

            for entity in entities:
                entity_id = entity.get("entity_id", "")
                embedding = entity.get("embedding") or []

                # Step 2: Find candidates in target graph.
                candidates: list[SameAsCandidate] = await _find_candidates_for_entity(
                    session,
                    entity,
                    graph_id,
                    target_graph_id,
                    embedding,
                    confidence_threshold,
                )

                if not candidates:
                    continue

                # Step 3: Score and link via EntityResolver.
                # resolve_and_link returns the ambiguous-zone pairs rejected by LLM.
                ambiguous_rejected = await EntityResolver.resolve_and_link(
                    entity_a=entity,
                    candidates=candidates,
                    session=session,
                    graph_id_a=graph_id,
                    target_graph_ids=[target_graph_id],
                )
                ambiguous_count += len(ambiguous_rejected)

                # Count rejected (below AMBIGUOUS_LOWER) — these were discarded
                # inside resolve_and_link; approximate from candidates not linked.
                for candidate in candidates:
                    entity_b = candidate["entity"]
                    entity_b_id = entity_b.get("entity_id", "")
                    # A link was created if entity_b does NOT appear in ambiguous_rejected.
                    already_linked = _was_linked(entity_b_id, ambiguous_rejected)
                    if already_linked:
                        created_links.append({
                            "entity_a_id": entity_id,
                            "entity_b_id": entity_b_id,
                            "confidence": candidate.get("score", 0.0),
                        })
                    else:
                        # Check whether this candidate fell in the rejected zone
                        if candidate.get("score", 0.0) < AMBIGUOUS_LOWER:
                            rejected_count += 1

    logger.info(
        "resolve_same_as_task complete: graph_id=%s target=%s "
        "created=%d ambiguous=%d rejected=%d",
        graph_id,
        target_graph_id,
        len(created_links),
        ambiguous_count,
        rejected_count,
    )
    return {
        "created_links": created_links,
        "ambiguous_count": ambiguous_count,
        "rejected_count": rejected_count,
    }


async def _fetch_entities_with_embeddings(
    session,
    graph_id: str,
) -> list[dict[str, Any]]:
    """Return all __Entity__ nodes in *graph_id* that have a stored embedding."""
    query = """
    MATCH (e:__Entity__ {graph_id: $graph_id})
    WHERE e.embedding IS NOT NULL
    RETURN elementId(e) AS entity_id,
           e.name        AS name,
           e.type        AS type,
           e.embedding   AS embedding,
           e.graph_id    AS source_graph_id
    """
    try:
        result = await session.run(query, {"graph_id": graph_id})
        return await result.data()
    except Exception as exc:
        logger.error("_fetch_entities_with_embeddings failed for graph %s: %s", graph_id, exc)
        return []


async def _find_candidates_for_entity(
    session,
    entity: dict[str, Any],
    graph_id: str,
    target_graph_id: str,
    embedding: list[float],
    threshold: float,
) -> list[Any]:
    """Find SAME_AS candidates for *entity* in *target_graph_id*.

    Strategy (ordered):
    1. Exact name + type fast-path — score 0.99.
    2. Vector search using stored embedding — score from cosine similarity.

    Returns a list of SameAsCandidate TypedDicts.
    """
    from app.schemas.federation_schemas import SameAsCandidate

    name = entity.get("name") or ""
    etype = entity.get("type") or ""

    # Fast path: exact name + type match
    if name:
        exact_query = """
        MATCH (e:__Entity__ {graph_id: $target_graph_id})
        WHERE toLower(e.name) = toLower($name)
          AND ($etype = '' OR toLower(coalesce(e.type, '')) = toLower($etype))
        RETURN elementId(e) AS entity_id,
               e.name        AS name,
               e.type        AS type,
               e.graph_id    AS source_graph_id
        LIMIT 1
        """
        try:
            result = await session.run(
                exact_query,
                {
                    "target_graph_id": target_graph_id,
                    "name": name,
                    "etype": etype,
                },
            )
            rows = await result.data()
            if rows:
                return [SameAsCandidate(entity=rows[0], score=0.99, method="exact")]
        except Exception as exc:
            logger.warning("exact-match candidate search failed: %s", exc)

    # Vector search path
    if not embedding:
        return []

    vector_query = """
    CALL db.index.vector.queryNodes('entity-embedding-index', $top_k, $embedding)
    YIELD node AS e, score
    WHERE e.graph_id = $target_graph_id
      AND score >= $threshold
    RETURN elementId(e) AS entity_id,
           e.name        AS name,
           e.type        AS type,
           e.graph_id    AS source_graph_id,
           score
    ORDER BY score DESC
    LIMIT $top_k
    """
    try:
        result = await session.run(
            vector_query,
            {
                "target_graph_id": target_graph_id,
                "embedding": embedding,
                "threshold": threshold,
                "top_k": 10,
            },
        )
        rows = await result.data()
        return [
            SameAsCandidate(entity=row, score=row["score"], method="vector")
            for row in rows
        ]
    except Exception as exc:
        # Vector index may not exist — degrade gracefully
        logger.warning("vector candidate search failed for entity %s: %s", entity.get("entity_id"), exc)
        return []


def _was_linked(entity_b_id: str, ambiguous_rejected: list[dict]) -> bool:
    """Return True when entity_b_id is NOT in the ambiguous-rejected list.

    resolve_and_link returns only the pairs that were *not* linked (LLM said NO
    or score was in ambiguous zone but rejected).  A candidate that does NOT
    appear in the rejected list was either linked at high confidence or linked
    after LLM disambiguation.
    """
    for item in ambiguous_rejected:
        if item.get("entity_b", {}).get("entity_id") == entity_b_id:
            return False
    return True
