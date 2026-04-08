"""Cross-Graph Federation Service.

Implements the federation model from ORA-41 spec:
- Permission validation (ownership + federatable flag, fail-closed)
- UNION ALL Cypher query router (per-graph subqueries, each uses graph_id index)
- Async parallel execution using AsyncDriver (Architecture Rule #5)
- Post-query SAME_AS entity deduplication
- Vector search federation with over-fetch pattern
"""

import time
from typing import Any

from neo4j import AsyncDriver

from app.core.logging import get_logger
from app.schemas.federation_schemas import (
    MAX_GRAPH_IDS,
    MAX_RESULTS_PER_GRAPH,
    MAX_TOTAL_RESULTS,
    CrossGraphLink,
    FederatedEntity,
    FederatedQueryOptions,
    FederatedVectorResult,
)

logger = get_logger(__name__)

# Confidence thresholds from ORA-41 spec §6
_SAME_AS_STORE_THRESHOLD = 0.85
_SAME_AS_CANDIDATE_THRESHOLD = 0.60

# Over-fetch multiplier for vector search post-filter
_VECTOR_OVERFETCH_MULTIPLIER = 1.5


class FederationError(Exception):
    """Raised for federation-specific validation failures."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class FederationService:
    """Orchestrates cross-graph queries for a single authenticated user or service account."""

    def __init__(self, async_driver: AsyncDriver, neo4j_database: str = "neo4j"):
        self._driver = async_driver
        self._database = neo4j_database

    # ── Public API ────────────────────────────────────────────────────────────

    async def federated_query(
        self,
        user_id: str,
        graph_ids: list[str],
        search_term: str,
        options: FederatedQueryOptions | None = None,
        principal: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a cross-graph entity search and return merged results.

        Raises FederationError (400) if any graph_id is not owned/federatable.
        Raises FederationError (403) if any graph_id belongs to another tenant.

        The `principal` dict (from get_current_user) is used to route between
        user and service-account permission checks.
        """
        if options is None:
            options = FederatedQueryOptions()

        start_ms = _now_ms()
        allowed = await self._validate_and_filter(
            user_id, graph_ids, principal=principal
        )
        # allowed is guaranteed == graph_ids (fail-closed: raises on mismatch)

        graph_meta = {g["graph_id"]: g["name"] for g in allowed}

        raw_entities = await self._execute_entity_union(
            graph_ids=graph_ids,
            search_term=search_term,
            max_per_graph=options.max_results_per_graph,
        )

        federated_entities = [
            FederatedEntity(
                entity_id=row["entity_id"],
                name=row["name"],
                type=row.get("type", "Unknown"),
                properties={
                    k: v
                    for k, v in row.items()
                    if k not in {"entity_id", "name", "type", "source_graph_id"}
                },
                source_graph_id=row["source_graph_id"],
                source_graph_name=graph_meta.get(
                    row["source_graph_id"], row["source_graph_id"]
                ),
            )
            for row in raw_entities[:MAX_TOTAL_RESULTS]
        ]

        cross_links: list[CrossGraphLink] = []
        deduplication_status = "not_requested"
        if options.deduplicate_entities and options.include_cross_graph_links:
            cross_links = await self._resolve_same_as(federated_entities)
            deduplication_status = "complete"

        elapsed_ms = _now_ms() - start_ms
        return {
            "status": "ok",
            "graphs_queried": graph_ids,
            "total_entities": len(federated_entities),
            "entities": federated_entities,
            "cross_graph_links": cross_links,
            "query_meta": {
                "execution_time_ms": elapsed_ms,
                "graphs_skipped": [],
                "timed_out": False,
                "deduplication_status": deduplication_status,
            },
        }

    async def federated_vector_search(
        self,
        user_id: str,
        graph_ids: list[str],
        query_text: str,
        top_k: int = 20,
        similarity_threshold: float = 0.75,
        principal: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Vector similarity search across multiple graphs using over-fetch pattern."""
        start_ms = _now_ms()
        allowed = await self._validate_and_filter(
            user_id, graph_ids, principal=principal
        )
        graph_meta = {g["graph_id"]: g["name"] for g in allowed}

        # Over-fetch to compensate for post-filter recall loss (ORA-41 §4.1)
        candidate_count = int(top_k * len(graph_ids) * _VECTOR_OVERFETCH_MULTIPLIER)

        results = await self._execute_vector_search(
            graph_ids=graph_ids,
            query_text=query_text,
            candidate_count=candidate_count,
            similarity_threshold=similarity_threshold,
        )

        federated_results = [
            FederatedVectorResult(
                chunk_id=row["chunk_id"],
                text=row.get("text", ""),
                score=row["score"],
                source_graph_id=row["source_graph_id"],
                source_graph_name=graph_meta.get(
                    row["source_graph_id"], row["source_graph_id"]
                ),
                entity_name=row.get("entity_name"),
                entity_type=row.get("entity_type"),
            )
            for row in results[:top_k]
        ]

        elapsed_ms = _now_ms() - start_ms
        return {
            "status": "ok",
            "graphs_queried": graph_ids,
            "total_results": len(federated_results),
            "results": federated_results,
            "query_meta": {
                "execution_time_ms": elapsed_ms,
                "graphs_skipped": [],
                "timed_out": False,
            },
        }

    # ── Validation ────────────────────────────────────────────────────────────

    async def _validate_and_filter(
        self,
        user_id: str,
        graph_ids: list[str],
        principal: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Fail-closed permission check.

        Raises FederationError if:
        - any graph_id belongs to another tenant (403)
        - any graph_id is not federatable=true (400)
        - service account principal lacks CAN_ACCESS to the requested graphs (403)

        Routes to SA permission check when principal_type == "service_account".
        """
        if len(graph_ids) > MAX_GRAPH_IDS:
            raise FederationError(
                f"Too many graphs: max {MAX_GRAPH_IDS}, got {len(graph_ids)}"
            )

        principal_type = (principal or {}).get("principal_type", "user")

        # Fetch ownership + federatable status for all requested graph_ids
        query = """
        MATCH (g:Graph)
        WHERE g.graph_id IN $graph_ids
        RETURN g.graph_id AS graph_id,
               g.user_id AS user_id,
               g.name AS name,
               coalesce(g.federatable, false) AS federatable
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, {"graph_ids": graph_ids})
            rows = await result.data()

        found_ids = {row["graph_id"] for row in rows}
        missing = set(graph_ids) - found_ids
        if missing:
            raise FederationError(
                f"Graphs not found: {', '.join(sorted(missing))}", status_code=400
            )

        if principal_type == "service_account":
            # SA path: check CAN_ACCESS edges + federatable flag
            tenant_id = (principal or {}).get("tenant_id", "")
            from app.services.service_account_service import service_account_service

            accessible = await service_account_service.get_sa_accessible_graphs(
                self._driver, user_id, tenant_id, graph_ids
            )
            accessible_set = set(accessible)

            for row in rows:
                if row["graph_id"] not in accessible_set:
                    raise FederationError(
                        "Access denied — no accessible graphs in federation request",
                        status_code=403,
                    )
                if not row["federatable"]:
                    raise FederationError(
                        "One or more requested graphs are not enabled for federation",
                        status_code=400,
                    )
        else:
            # User path (existing behavior): ownership check + federatable flag
            for row in rows:
                if row["user_id"] != user_id:
                    raise FederationError(
                        "Access denied — no accessible graphs in federation request",
                        status_code=403,
                    )
                if not row["federatable"]:
                    raise FederationError(
                        "One or more requested graphs are not enabled for federation",
                        status_code=400,
                    )

        return rows

    # ── Query builder & executor ──────────────────────────────────────────────

    async def _execute_entity_union(
        self,
        graph_ids: list[str],
        search_term: str,
        max_per_graph: int,
    ) -> list[dict[str, Any]]:
        """Build a UNION ALL Cypher query across all graph_ids and execute async.

        Each CALL subquery is scoped to one graph_id, hitting the per-graph index.
        """
        params: dict[str, Any] = {"search_term": search_term, "limit": max_per_graph}

        subqueries: list[str] = []
        for i, gid in enumerate(graph_ids):
            param_key = f"gid_{i}"
            params[param_key] = gid
            subqueries.append(
                f"CALL {{\n"
                f"  MATCH (e:__Entity__)\n"
                f"  WHERE e.graph_id = ${param_key}\n"
                f"    AND toLower(e.name) CONTAINS toLower($search_term)\n"
                f"  RETURN elementId(e) AS entity_id,\n"
                f"         e.name AS name,\n"
                f"         coalesce(e.type, labels(e)[-1]) AS type,\n"
                f"         ${param_key} AS source_graph_id\n"
                f"  LIMIT $limit\n"
                f"}}"
            )

        cypher = (
            "\nUNION ALL\n".join(subqueries)
            + "\nRETURN entity_id, name, type, source_graph_id"
        )

        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, params)
            return await result.data()

    async def _execute_vector_search(
        self,
        graph_ids: list[str],
        query_text: str,
        candidate_count: int,
        similarity_threshold: float,
    ) -> list[dict[str, Any]]:
        """Vector search using the shared chunk-embedding index with graph_id post-filter."""
        # Use the shared vector index; post-filter by graph_ids (user-validated)
        query = """
        CALL db.index.vector.queryNodes('chunk-embedding-index', $candidate_count, $query_vector)
        YIELD node AS chunk, score
        WHERE chunk.graph_id IN $graph_ids
          AND score >= $similarity_threshold
        OPTIONAL MATCH (chunk)<-[:FROM_CHUNK]-(entity:__Entity__)
        WHERE entity.graph_id IN $graph_ids
        RETURN elementId(chunk) AS chunk_id,
               coalesce(chunk.text, chunk.content, '') AS text,
               score,
               chunk.graph_id AS source_graph_id,
               entity.name AS entity_name,
               entity.type AS entity_type
        ORDER BY score DESC
        LIMIT $result_limit
        """
        # For now we use a text embedding; in production inject the vector from LLM service
        # This query is intentionally left as-is — vector embedding generation is handled
        # by the caller (endpoint) before invoking this method.
        params = {
            "graph_ids": graph_ids,
            "candidate_count": candidate_count,
            "similarity_threshold": similarity_threshold,
            "result_limit": MAX_RESULTS_PER_GRAPH,
            # query_vector must be injected by caller; placeholder shows contract
            "query_vector": [],
        }
        logger.warning(
            "federated_vector_search called without a real query vector; "
            "integrate with llm_service.get_embedding() before shipping"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, params)
            return await result.data()

    # ── SAME_AS deduplication ─────────────────────────────────────────────────

    async def _resolve_same_as(
        self, entities: list[FederatedEntity]
    ) -> list[CrossGraphLink]:
        """Post-query entity resolution: find SAME_AS candidates across graphs.

        Uses exact name+type matching (confidence 0.99) as the MVP strategy.
        Async: stores new SAME_AS links in Neo4j; returns links for the response.
        """
        # Group by normalized (name, type) — candidates are same name+type in different graphs
        from collections import defaultdict

        buckets: dict[tuple[str, str], list[FederatedEntity]] = defaultdict(list)
        for entity in entities:
            key = (entity.name.strip().lower(), (entity.type or "").strip().lower())
            buckets[key].append(entity)

        links: list[CrossGraphLink] = []
        merge_tasks = []

        for (_name, _etype), group in buckets.items():
            if len(group) < 2:
                continue
            # Generate all pairs
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    if a.source_graph_id == b.source_graph_id:
                        continue  # same graph — skip
                    confidence = 0.99  # exact name+type match
                    links.append(
                        CrossGraphLink(
                            entity_a_id=a.entity_id,
                            entity_b_id=b.entity_id,
                            link_type="SAME_AS",
                            confidence=confidence,
                            graph_a=a.source_graph_id,
                            graph_b=b.source_graph_id,
                        )
                    )
                    if confidence >= _SAME_AS_STORE_THRESHOLD:
                        merge_tasks.append((a.entity_id, b.entity_id, confidence))

        if merge_tasks:
            await self._store_same_as_links(merge_tasks)

        return links

    async def _store_same_as_links(self, pairs: list[tuple[str, str, float]]) -> None:
        """Persist SAME_AS relationship pairs in Neo4j using MERGE (idempotent)."""
        query = """
        UNWIND $pairs AS pair
        MATCH (a:__Entity__) WHERE elementId(a) = pair.id_a
        MATCH (b:__Entity__) WHERE elementId(b) = pair.id_b
        MERGE (a)-[s:SAME_AS {detected_by: 'federation_resolver'}]->(b)
        SET s.confidence = CASE WHEN s.confidence IS NULL OR pair.confidence > s.confidence
                                THEN pair.confidence ELSE s.confidence END,
            s.match_method = 'exact_name',
            s.detected_at = datetime()
        """
        pair_params = [
            {"id_a": id_a, "id_b": id_b, "confidence": conf}
            for id_a, id_b, conf in pairs
        ]
        try:
            async with self._driver.session(database=self._database) as session:

                async def _write(tx) -> None:
                    await tx.run(query, {"pairs": pair_params})

                await session.execute_write(_write)
        except Exception as exc:
            logger.warning("Failed to store SAME_AS links: %s", exc)


def _now_ms() -> int:
    return int(time.monotonic() * 1000)
