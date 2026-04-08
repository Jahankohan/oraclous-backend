"""
Ontology Celery Tasks

Retroactive ontology enforcement for large graphs (>10k entities).

Architecture rules:
- Uses SYNC Neo4j driver with NullPool — Celery worker context
- Never shares drivers with FastAPI
- Every Cypher query filters by graph_id (multi-tenancy)
"""

from __future__ import annotations

import difflib

from neo4j import GraphDatabase

from app.core.config import settings
from app.core.logging import get_logger
from app.services.background_jobs import celery_app

logger = get_logger(__name__)


@celery_app.task(bind=True)
def retroactive_apply_ontology_task(
    self,
    graph_id: str,
    allowed_types: list[str],
    mode: str,
) -> dict:
    """
    Apply ontology enforcement to all entities in a graph.

    Parameters
    ----------
    graph_id:      Graph identifier (multi-tenant filter).
    allowed_types: List of allowed entity label strings.
    mode:          One of "warn", "strict", "coerce".

    Returns a summary dict with violation/coercion/deletion counts.
    """
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
        max_connection_pool_size=1,
    )
    try:
        allowed_set = set(allowed_types)
        coercions = 0
        deletions = 0

        with driver.session() as session:
            if mode == "strict":
                result = session.run(
                    """
                    MATCH (e:__Entity__ {graph_id: $graph_id})
                    WHERE NOT e.label IN $allowed_types
                    DETACH DELETE e
                    RETURN count(e) AS deleted
                    """,
                    {"graph_id": graph_id, "allowed_types": allowed_types},
                )
                record = result.single()
                deletions = record["deleted"] if record else 0

            elif mode == "coerce":
                allowed_list = list(allowed_set)
                violators = session.run(
                    """
                    MATCH (e:__Entity__ {graph_id: $graph_id})
                    WHERE NOT e.label IN $allowed_types
                    RETURN elementId(e) AS eid, e.label AS label
                    """,
                    {"graph_id": graph_id, "allowed_types": allowed_types},
                ).data()

                for rec in violators:
                    label = rec.get("label", "")
                    eid = rec.get("eid")
                    if not label or not eid:
                        continue
                    best_match = max(
                        allowed_list,
                        key=lambda a: difflib.SequenceMatcher(
                            None, label.lower(), a.lower()
                        ).ratio(),
                    )
                    ratio = difflib.SequenceMatcher(
                        None, label.lower(), best_match.lower()
                    ).ratio()
                    if ratio >= 0.7:
                        session.run(
                            "MATCH (e:__Entity__) WHERE elementId(e) = $eid SET e.label = $new_label",
                            {"eid": eid, "new_label": best_match},
                        )
                        coercions += 1
                    else:
                        session.run(
                            "MATCH (e:__Entity__) WHERE elementId(e) = $eid DETACH DELETE e",
                            {"eid": eid},
                        )
                        deletions += 1

            # Count remaining violations regardless of mode
            record = session.run(
                """
                MATCH (e:__Entity__ {graph_id: $graph_id})
                WHERE NOT e.label IN $allowed_types
                RETURN count(e) AS cnt
                """,
                {"graph_id": graph_id, "allowed_types": allowed_types},
            ).single()
            remaining = record["cnt"] if record else 0

        logger.info(
            f"retroactive_apply_ontology_task complete for graph {graph_id}: "
            f"mode={mode}, coercions={coercions}, deletions={deletions}, remaining={remaining}"
        )
        return {
            "graph_id": graph_id,
            "mode": mode,
            "coercions_applied": coercions,
            "deletions_applied": deletions,
            "remaining_violations": remaining,
        }

    finally:
        driver.close()
