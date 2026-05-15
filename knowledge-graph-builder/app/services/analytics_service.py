"""
Graph Analytics Service

This service provides advanced graph analytics capabilities including:
- Community detection using leidenalg at 5 resolutions (multi-resolution hierarchy)
- Centrality analysis (PageRank and degree centrality)
- Neighborhood analysis for entity relationships
- Pathway discovery between entities
- Temporal context analysis
- Graph statistics and metrics

All methods are multi-tenant safe with proper graph_id filtering.
"""

import hashlib
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client
from app.schemas.community_kinds import CommunityKindSpec, get_kind

logger = get_logger(__name__)


# Neo4j cannot parameterize labels or relationship types, so the registry's
# label/rel/property names are interpolated into Cypher via f-strings. This
# is safe because the values come from ``COMMUNITY_KINDS`` (compile-time
# constants), never from request input. Callers always resolve a kind via
# ``get_kind(...)`` which raises ``UnknownCommunityKindError`` for anything
# off the registry.


def _coerce_neo4j_datetime(value: Any) -> Any:
    """Convert Neo4j DateTime / Date values to ISO-8601 strings.

    Pydantic v2 cannot serialize ``neo4j.time.DateTime`` directly. Rows
    that the analytics service returns to FastAPI must be flat JSON, so
    coerce these properties before they reach the response model.
    """
    if value is None:
        return None
    if hasattr(value, "to_native"):
        try:
            return value.to_native().isoformat()
        except Exception:
            return str(value)
    return value


def _pick_entity_type(labels: list[str] | None, member_label: str) -> str:
    """Pick the most domain-meaningful label from a Neo4j node's labels list.

    Skips internal/utility labels — both the generic ``__Entity__`` tag and
    neo4j_graphrag's `:__KGBuilder__` marker — and the member-label itself
    (which is already implied by the community kind). Falls back to the
    member label when no domain label survives the filter.
    """
    if not labels:
        return member_label
    skip = {"__Entity__", "__KGBuilder__", member_label}
    for lbl in labels:
        if lbl in skip:
            continue
        # Skip any other "__Foo__" Neo4j-internal-style marker labels.
        if lbl.startswith("__") and lbl.endswith("__"):
            continue
        return lbl
    return member_label


class GraphAnalyticsService:
    """
    Dedicated service for graph analytics and algorithm execution.
    Extracted from ChatService for better separation of concerns.
    """

    def __init__(self):
        self.cached_statistics = {}

    # ==================== COMMUNITY DETECTION ====================

    async def get_community_context(
        self,
        entities: list[dict[str, Any]],
        graph_id: UUID,
    ) -> dict[str, Any]:
        """
        Return community context from persisted __Community__ nodes (post-Leiden).

        Falls back to simple shared-neighbor detection when no active communities exist.
        """
        if not entities:
            return {"communities": []}

        entity_ids = [e["id"] for e in entities if e.get("id")]

        try:
            query = """
            MATCH (entity:__Entity__)-[r:IN_COMMUNITY {graph_id: $graph_id, level: 1}]->(community:__Community__)
            WHERE entity.id IN $entity_ids
              AND community.graph_id = $graph_id
              AND community.status = 'active'
            WITH community, count(entity) AS member_hits
            WHERE member_hits >= 2
            RETURN community.id AS community_id,
                   community.summary AS summary,
                   community.level AS level,
                   community.entity_count AS entity_count,
                   community.status AS status,
                   member_hits
            ORDER BY member_hits DESC, community.entity_count ASC
            LIMIT 3
            """
            results = await neo4j_client.execute_query(
                query,
                {
                    "entity_ids": entity_ids,
                    "graph_id": str(graph_id),
                },
            )

            if results:
                communities = [
                    {
                        "community_id": r["community_id"],
                        "summary": r["summary"],
                        "level": r["level"],
                        "entity_count": r["entity_count"],
                        "member_hits": r["member_hits"],
                        "type": "leiden_community",
                    }
                    for r in results
                ]
                return {"communities": communities}

        except Exception as exc:
            logger.warning(f"Persisted community lookup failed: {exc}")

        # Fallback when no active communities
        return await self.get_simple_community_context(entities, graph_id)

    # ==================== NEW: COMMUNITY MANAGEMENT METHODS ====================

    async def detect_communities_async(
        self,
        graph_id: UUID,
        levels: int = 3,
        force_rebuild: bool = False,
        kind: str = "entity",
    ) -> dict[str, Any]:
        """
        Queue a Celery community detection job and return the task ID.

        Args:
            graph_id: Target graph
            levels: Number of hierarchy levels (1-5). Ignored for flat
                community kinds (Louvain over chunks does not have levels).
            force_rebuild: Run even if status == 'active'
            kind: Which community kind to detect. Must be in the registry
                AND have ``detector_task_name`` set. Read-only kinds (e.g.,
                chunk-Louvain today) raise ``NotImplementedError`` which
                the API endpoint maps to HTTP 405.

        Returns:
            Dict with job_id, graph_id, kind, status
        """
        spec = get_kind(kind)
        if spec.detector_task_name is None:
            raise NotImplementedError(
                f"Community detection for kind {kind!r} is not implemented "
                f"yet. Existing {spec.community_label} nodes remain readable "
                "via the list/detail endpoints."
            )

        # The registry's detector_task_name is "module.task" form. We only
        # have one detector wired today (entity-Leiden); rather than build
        # a generic dotted-path importer that adds attack surface for the
        # ~one extra task we expect, just resolve the known case here. If
        # a second wired kind ever lands, swap this for an importlib lookup.
        if spec.detector_task_name == "community_tasks.detect_communities_task":
            from app.tasks.community_tasks import detect_communities_task

            level_indices = list(range(levels))
            resolutions = [0.5, 1.0, 2.0, 3.0, 4.0][:levels]
            result = detect_communities_task.apply_async(
                args=[str(graph_id)],
                kwargs={
                    "levels": level_indices,
                    "resolutions": resolutions,
                    "force_rebuild": force_rebuild,
                },
                countdown=0,
            )
            return {
                "job_id": result.id,
                "graph_id": str(graph_id),
                "kind": kind,
                "status": "queued",
            }

        # Defensive — should never hit; means a registry entry declared a
        # detector name we don't know how to dispatch.
        raise NotImplementedError(
            f"Detector {spec.detector_task_name!r} for kind {kind!r} is "
            "registered but no dispatch path is wired in detect_communities_async."
        )

    async def get_community_status(
        self,
        graph_id: UUID,
        kind: str = "entity",
    ) -> dict[str, Any]:
        """Return current community detection status for a graph.

        For hierarchical kinds (entity-Leiden), reports per-level counts and
        consults the Postgres ``knowledge_graphs`` row for the
        detected-at / staleness metadata which only the entity-Leiden
        detector currently writes.

        For flat kinds (chunk-Louvain), reports a single bucket under
        level "0", returns ``status="read_only"`` to signal there's no
        detector to drive a status machine, and skips the Postgres lookup.
        """
        spec = get_kind(kind)
        community_label = spec.community_label

        if spec.hierarchical:
            level_query = (
                f"MATCH (c:`{community_label}` {{graph_id: $graph_id}}) "
                "RETURN c.level AS level, count(c) AS cnt, c.status AS status "
                "ORDER BY level"
            )
            level_results = await neo4j_client.execute_query(
                level_query, {"graph_id": str(graph_id)}
            )

            communities_by_level: dict[str, int] = {}
            detected_status = "not_detected"
            for r in level_results:
                communities_by_level[str(r["level"])] = r["cnt"]
                if r["status"] == "active":
                    detected_status = "active"
                elif r["status"] == "rebuilding":
                    detected_status = "rebuilding"
                elif detected_status == "not_detected" and r["status"] == "stale":
                    detected_status = "stale"
        else:
            # Flat kinds have no per-level breakdown; report a single bucket.
            count_query = (
                f"MATCH (c:`{community_label}` {{graph_id: $graph_id}}) "
                "RETURN count(c) AS cnt"
            )
            count_results = await neo4j_client.execute_query(
                count_query, {"graph_id": str(graph_id)}
            )
            cnt = count_results[0]["cnt"] if count_results else 0
            communities_by_level = {"0": cnt}
            detected_status = "read_only" if cnt > 0 else "not_detected"

        # Postgres detection-history columns only track entity-Leiden today;
        # skip the lookup for flat read-only kinds.
        entity_count_at_detection = 0
        last_detected_at = None
        current_entity_count = 0
        if spec.hierarchical and spec.detector_task_name is not None:
            try:
                pg_engine = create_engine(
                    settings.POSTGRES_URL.replace("+asyncpg", ""),
                    poolclass=NullPool,
                )
                with pg_engine.connect() as conn:
                    row = conn.execute(
                        text(
                            "SELECT communities_detected_at, entity_count_at_detection, "
                            "entity_delta_since_detection, communities_status "
                            "FROM knowledge_graphs WHERE id = :gid"
                        ),
                        {"gid": str(graph_id)},
                    ).fetchone()
                    if row:
                        last_detected_at = row[0]
                        entity_count_at_detection = row[1] or 0
                        delta = row[2] or 0
                        current_entity_count = entity_count_at_detection + delta
                        if row[3]:
                            detected_status = row[3]
                pg_engine.dispose()
            except Exception as exc:
                logger.warning(f"Postgres community status lookup failed: {exc}")

        staleness_pct = 0.0
        if entity_count_at_detection > 0:
            staleness_pct = (
                current_entity_count - entity_count_at_detection
            ) / entity_count_at_detection

        return {
            "kind": kind,
            "status": detected_status,
            "last_detected_at": (
                last_detected_at.isoformat() if last_detected_at else None
            ),
            "communities_by_level": communities_by_level,
            "entity_count_at_detection": entity_count_at_detection,
            "current_entity_count": current_entity_count,
            "staleness_pct": round(staleness_pct, 4),
        }

    async def get_communities_list(
        self,
        graph_id: UUID,
        level: int | None = None,
        min_size: int = 2,
        limit: int = 50,
        offset: int = 0,
        include_summary: bool = True,
        kind: str = "entity",
    ) -> dict[str, Any]:
        """Return paginated list of communities for a graph.

        The shape of each list entry is uniform across kinds: ``community_id``,
        ``kind``, ``level``, ``entity_count`` (renamed at the result level to
        match the existing API even when the underlying property is ``size``
        for chunk-Louvain), ``weight``, ``parent_id``, ``status``, and
        optionally ``summary``. Fields not present on a given kind come back
        as ``None``.

        ``level`` filter is only applied when the kind is hierarchical.
        """
        spec = get_kind(kind)
        community_label = spec.community_label
        id_prop = spec.id_property
        size_prop = spec.size_property

        where_clauses = [
            "c.graph_id = $graph_id",
            f"c.{size_prop} >= $min_size",
        ]
        params: dict[str, Any] = {
            "graph_id": str(graph_id),
            "min_size": min_size,
            "limit": limit,
            "offset": offset,
        }
        if level is not None and spec.hierarchical:
            where_clauses.append("c.level = $level")
            params["level"] = level

        where = " AND ".join(where_clauses)

        # Build the RETURN list. For hierarchical kinds we pull the
        # additional metadata properties (level/weight/parent_id/status);
        # for flat kinds those come back as NULL.
        return_fields = [
            f"c.{id_prop} AS community_id",
            f"c.{size_prop} AS entity_count",
        ]
        if spec.hierarchical:
            return_fields.extend(
                [
                    "c.level AS level",
                    "c.weight AS weight",
                    "c.parent_id AS parent_id",
                    "c.status AS status",
                ]
            )
        else:
            return_fields.extend(
                [
                    "NULL AS level",
                    "NULL AS weight",
                    "NULL AS parent_id",
                    "NULL AS status",
                ]
            )
        if include_summary:
            return_fields.append("c.summary AS summary")
            # Chunk communities (STORY-4b) also have summary_keywords +
            # summary_excerpt; entity-Leiden returns NULL for these.
            if spec.kind == "chunk":
                return_fields.extend(
                    [
                        "c.summary_keywords AS summary_keywords",
                        "c.summary_excerpt AS summary_excerpt",
                    ]
                )
            else:
                return_fields.extend(
                    [
                        "NULL AS summary_keywords",
                        "NULL AS summary_excerpt",
                    ]
                )
        fields = ", ".join(return_fields)

        # ORDER BY: hierarchical kinds sort by level first (so callers can
        # walk the hierarchy); flat kinds sort by size descending only.
        order_by = (
            f"c.level, c.{size_prop} DESC"
            if spec.hierarchical
            else f"c.{size_prop} DESC"
        )

        count_query = (
            f"MATCH (c:`{community_label}`) WHERE {where} RETURN count(c) AS total"
        )
        list_query = (
            f"MATCH (c:`{community_label}`) WHERE {where} "
            f"RETURN {fields} "
            f"ORDER BY {order_by} "
            "SKIP $offset LIMIT $limit"
        )

        total_results = await neo4j_client.execute_query(count_query, params)
        total = total_results[0]["total"] if total_results else 0

        list_results = await neo4j_client.execute_query(list_query, params)

        communities = []
        for r in list_results:
            item = {
                "community_id": r["community_id"],
                "kind": kind,
                "level": r["level"],
                "entity_count": r["entity_count"],
                "weight": r["weight"],
                "parent_id": r["parent_id"],
                "status": r["status"],
            }
            if include_summary:
                item["summary"] = r.get("summary")
                # Parse summary_keywords (stored as JSON string in Neo4j).
                raw_kw = r.get("summary_keywords")
                if raw_kw:
                    try:
                        import json as _json

                        parsed_kw = _json.loads(raw_kw)
                        if isinstance(parsed_kw, list):
                            item["summary_keywords"] = parsed_kw
                    except (ValueError, TypeError):
                        item["summary_keywords"] = None
                item["summary_excerpt"] = r.get("summary_excerpt")
            communities.append(item)

        status_info = await self.get_community_status(graph_id, kind=kind)

        return {
            "communities": communities,
            "kind": kind,
            "total": total,
            "detection_status": status_info["status"],
            "last_detected_at": status_info["last_detected_at"],
        }

    async def get_community_detail(
        self,
        graph_id: UUID,
        community_id: str,
        kind: str | None = None,
    ) -> dict[str, Any] | None:
        """Return full community detail with members and parent/child links.

        When ``kind`` is ``None`` (legacy callers), each registered kind is
        probed in turn — entity-Leiden first so the common case stays a
        single round trip. The first kind whose ``community_label`` and
        ``id_property`` match the given ``community_id`` wins.
        """
        candidate_specs: list[CommunityKindSpec]
        if kind is None:
            from app.schemas.community_kinds import all_kinds

            candidate_specs = all_kinds()
        else:
            candidate_specs = [get_kind(kind)]

        spec: CommunityKindSpec | None = None
        community_row: dict[str, Any] | None = None
        for candidate in candidate_specs:
            row = await self._fetch_community_row(
                graph_id=graph_id,
                community_id=community_id,
                spec=candidate,
            )
            if row is not None:
                spec = candidate
                community_row = row
                break

        if spec is None or community_row is None:
            return None

        members = await self._fetch_community_members(
            graph_id=graph_id,
            community_id=community_id,
            spec=spec,
            level=community_row.get("level"),
        )

        parent_community = None
        child_communities: list[dict[str, Any]] = []
        if spec.hierarchical:
            parent_community = await self._fetch_parent_community(
                graph_id=graph_id,
                parent_id=community_row.get("parent_id"),
                spec=spec,
            )
            child_communities = await self._fetch_child_communities(
                graph_id=graph_id,
                community_id=community_id,
                spec=spec,
            )

        # Parse summary_keywords (stored as JSON string in Neo4j).
        keywords: list[str] | None = None
        raw_kw = community_row.get("summary_keywords")
        if raw_kw:
            try:
                import json as _json

                parsed = _json.loads(raw_kw)
                if isinstance(parsed, list):
                    keywords = parsed
            except (ValueError, TypeError):
                keywords = None

        return {
            "community_id": community_row["community_id"],
            "kind": spec.kind,
            "level": community_row.get("level"),
            "summary": community_row.get("summary"),
            "summary_keywords": keywords,
            "summary_excerpt": community_row.get("summary_excerpt"),
            "entity_count": community_row.get("entity_count"),
            "algorithm": community_row.get("algorithm"),
            "status": community_row.get("status"),
            "parent_community": parent_community,
            "child_communities": child_communities,
            "members": members,
            "created_at": _coerce_neo4j_datetime(community_row.get("created_at")),
            "last_updated": _coerce_neo4j_datetime(community_row.get("last_updated")),
        }

    async def _fetch_community_row(
        self,
        graph_id: UUID,
        community_id: str,
        spec: CommunityKindSpec,
    ) -> dict[str, Any] | None:
        """Pull the community node for one kind. Returns None if absent."""
        # Build RETURN list. Fields that don't exist on the kind come back
        # as NULL so the caller can treat the result shape uniformly.
        if spec.hierarchical:
            extra_fields = (
                "c.level AS level, c.algorithm AS algorithm, "
                "c.parent_id AS parent_id, c.status AS status, "
                "c.last_updated AS last_updated"
            )
        else:
            # Flat kinds may not have these — return NULLs to keep callers simple.
            extra_fields = (
                "NULL AS level, NULL AS algorithm, NULL AS parent_id, "
                "NULL AS status, c.updated_at AS last_updated"
            )

        # STORY-4b chunk-community summary fields. Entity-Leiden returns
        # NULL for these today.
        if spec.kind == "chunk":
            summary_extras = (
                "c.summary_keywords AS summary_keywords, "
                "c.summary_excerpt AS summary_excerpt"
            )
        else:
            summary_extras = "NULL AS summary_keywords, NULL AS summary_excerpt"

        query = (
            f"MATCH (c:`{spec.community_label}` "
            f"{{`{spec.id_property}`: $community_id, graph_id: $graph_id}}) "
            f"RETURN c.`{spec.id_property}` AS community_id, "
            f"c.summary AS summary, "
            f"{summary_extras}, "
            f"c.`{spec.size_property}` AS entity_count, "
            f"c.created_at AS created_at, "
            f"{extra_fields}"
        )
        rows = await neo4j_client.execute_query(
            query,
            {"community_id": community_id, "graph_id": str(graph_id)},
        )
        return rows[0] if rows else None

    async def _fetch_community_members(
        self,
        graph_id: UUID,
        community_id: str,
        spec: CommunityKindSpec,
        level: int | None,
    ) -> list[dict[str, Any]]:
        """Pull up to 100 members of one community."""
        # Member-rel filter: hierarchical kinds (Leiden) tag the rel with
        # {graph_id, level} so we can isolate one level; flat kinds (Louvain)
        # have no rel properties and we just walk the edge.
        if spec.member_rel_has_level and level is not None:
            rel_filter = (
                f"[:`{spec.member_rel}` {{graph_id: $graph_id, level: $level}}]"
            )
            params: dict[str, Any] = {
                "graph_id": str(graph_id),
                "level": level,
                "cid": community_id,
            }
        else:
            rel_filter = f"[:`{spec.member_rel}`]"
            params = {"graph_id": str(graph_id), "cid": community_id}

        members_query = (
            f"MATCH (m:`{spec.member_label}` {{graph_id: $graph_id}})"
            f"-{rel_filter}->"
            f"(c:`{spec.community_label}` "
            f"{{`{spec.id_property}`: $cid, graph_id: $graph_id}}) "
            "RETURN coalesce(m.id, elementId(m)) AS entity_id, "
            "coalesce(m.name, m.text, '') AS entity_name, "
            "labels(m) AS entity_labels "
            "LIMIT 100"
        )
        members_result = await neo4j_client.execute_query(members_query, params)
        return [
            {
                "entity_id": m["entity_id"],
                "entity_name": m["entity_name"],
                "entity_type": _pick_entity_type(m["entity_labels"], spec.member_label),
            }
            for m in members_result
        ]

    async def _fetch_parent_community(
        self,
        graph_id: UUID,
        parent_id: str | None,
        spec: CommunityKindSpec,
    ) -> dict[str, Any] | None:
        if not parent_id:
            return None
        query = (
            f"MATCH (p:`{spec.community_label}` "
            f"{{`{spec.id_property}`: $pid, graph_id: $gid}}) "
            f"RETURN p.`{spec.id_property}` AS community_id, p.summary AS summary"
        )
        result = await neo4j_client.execute_query(
            query, {"pid": parent_id, "gid": str(graph_id)}
        )
        if not result:
            return None
        return {
            "community_id": result[0]["community_id"],
            "summary": result[0]["summary"],
        }

    async def _fetch_child_communities(
        self,
        graph_id: UUID,
        community_id: str,
        spec: CommunityKindSpec,
    ) -> list[dict[str, Any]]:
        query = (
            f"MATCH (child:`{spec.community_label}` {{graph_id: $graph_id}})"
            "-[:PARENT_COMMUNITY]->"
            f"(parent:`{spec.community_label}` "
            f"{{`{spec.id_property}`: $cid, graph_id: $graph_id}}) "
            f"RETURN child.`{spec.id_property}` AS community_id, "
            "child.summary AS summary, "
            f"child.`{spec.size_property}` AS entity_count "
            "LIMIT 20"
        )
        child_results = await neo4j_client.execute_query(
            query, {"graph_id": str(graph_id), "cid": community_id}
        )
        return [
            {
                "community_id": c["community_id"],
                "summary": c["summary"],
                "entity_count": c["entity_count"],
            }
            for c in child_results
        ]

    async def get_simple_community_context(
        self, entities: list[dict[str, Any]], graph_id: UUID
    ) -> dict[str, Any]:
        """
        Fallback community detection based on shared neighbors with graph_id filtering.

        Args:
            entities: List of entity dictionaries with 'id' and 'name' keys
            graph_id: UUID of the specific graph to analyze

        Returns:
            Dictionary containing simple community information
        """
        if not entities:
            return {"communities": []}

        entity_ids = [e["id"] for e in entities]

        query = """
        MATCH (entity)
        WHERE entity.id IN $entity_ids AND entity.graph_id = $graph_id

        MATCH (entity)-[r1]-(neighbor)-[r2]-(community_member)
        WHERE r1.graph_id = $graph_id AND r2.graph_id = $graph_id
        AND neighbor.graph_id = $graph_id AND community_member.graph_id = $graph_id
        AND community_member.id <> entity.id

        WITH entity, neighbor, collect(DISTINCT community_member) as members
        WHERE size(members) >= 2

        RETURN entity.id as entity_id,
            entity.name as entity_name,
            neighbor.name as hub_name,
            [m IN members | {id: m.id, name: m.name}][..5] as community_members
        LIMIT 10
        """

        results = await neo4j_client.execute_query(
            query, {"entity_ids": entity_ids, "graph_id": str(graph_id)}
        )

        communities = []
        for result in results:
            communities.append(
                {
                    "entity": result["entity_name"],
                    "hub": result["hub_name"],
                    "members": result["community_members"],
                    "type": "shared_neighbor_community",
                }
            )

        return {"communities": communities}

    # ==================== COMMUNITY PERSISTENCE ====================

    async def create_community_nodes(self, graph_id: UUID) -> dict[str, Any]:
        """
        Create persistent community nodes using leidenalg at 5 resolutions.

        This method:
        1. Fetches __Entity__ nodes and relationships for the graph from Neo4j
        2. Builds an igraph graph in-process (no GDS dependency)
        3. Runs leidenalg.find_partition at 5 resolutions (0.25, 0.5, 1.0, 2.0, 4.0)
        4. Writes __Community__ nodes with IN_COMMUNITY and PARENT_COMMUNITY
           edges to Neo4j (same edge names the Celery detector and all
           readers use — see ``community_tasks.py`` and the
           ``CommunityKindSpec`` registry).

        Args:
            graph_id: UUID of the specific graph to analyze

        Returns:
            Dictionary containing creation results and statistics
        """
        try:
            logger.info(f"Creating Leiden community nodes for graph {graph_id}")
            result = await self._run_leiden_community_detection(graph_id)
            return result
        except Exception as e:
            logger.error(f"Failed to create community nodes: {e}")
            return await self._create_simple_communities(graph_id)

    async def _run_leiden_community_detection(self, graph_id: UUID) -> dict[str, Any]:
        """
        Run leidenalg at 5 resolutions and store hierarchical communities.

        Uses igraph + leidenalg — no Neo4j GDS required.
        All Cypher queries are parameterized and graph_id-scoped.

        Args:
            graph_id: UUID of the specific graph to analyze

        Returns:
            Dictionary with detection statistics
        """
        import igraph as ig
        import leidenalg

        graph_id_str = str(graph_id)

        # Step 1 — Fetch entity nodes
        nodes_result = await neo4j_client.execute_query(
            "MATCH (e:__Entity__ {graph_id: $graph_id}) "
            "RETURN elementId(e) AS eid, e.id AS entity_id, e.name AS name",
            {"graph_id": graph_id_str},
        )

        if not nodes_result:
            return {
                "communities_created": 0,
                "relationships_created": 0,
                "algorithm_used": "leiden",
                "message": "No entities found for this graph",
            }

        MAX_SYNC_ENTITIES = getattr(settings, "LEIDEN_SYNC_MAX_ENTITIES", 10_000)
        if len(nodes_result) > MAX_SYNC_ENTITIES:
            logger.warning(
                f"Graph {graph_id} has {len(nodes_result)} entities, exceeding sync limit {MAX_SYNC_ENTITIES}. "
                "Use async detect_communities_task for large graphs."
            )
            return {
                "communities_created": 0,
                "relationships_created": 0,
                "message": f"Graph too large for sync detection ({len(nodes_result)} entities). Use async task.",
                "graph_id": str(graph_id),
            }

        # Step 2 — Fetch relationships
        edges_result = await neo4j_client.execute_query(
            "MATCH (a:__Entity__ {graph_id: $graph_id})-[r]->(b:__Entity__ {graph_id: $graph_id}) "
            "RETURN elementId(a) AS src, elementId(b) AS tgt",
            {"graph_id": graph_id_str},
        )

        # Step 3 — Build igraph
        eid_to_idx = {n["eid"]: i for i, n in enumerate(nodes_result)}
        ig_edges = [
            (eid_to_idx[e["src"]], eid_to_idx[e["tgt"]])
            for e in edges_result
            if e["src"] in eid_to_idx and e["tgt"] in eid_to_idx
        ]

        g = ig.Graph(n=len(nodes_result), edges=ig_edges, directed=False)

        # Step 4 — Run at 5 resolutions
        RESOLUTIONS = [0.25, 0.5, 1.0, 2.0, 4.0]
        all_communities: dict[int, dict] = {}

        for level, resolution in enumerate(RESOLUTIONS):
            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=resolution,
                n_iterations=10,
                seed=42,
            )
            all_communities[level] = {
                "resolution": resolution,
                "membership": partition.membership,  # list: vertex_index -> community_id
            }

        # Step 5 — Write __Community__ nodes and IN_COMMUNITY / PARENT_COMMUNITY
        # edges. These names match the Celery production detector
        # (community_tasks.py) and every reader path; the registry's
        # ``CommunityKindSpec.member_rel`` is also ``"IN_COMMUNITY"``.
        total_communities = 0
        total_relationships = 0

        for level, data in all_communities.items():
            resolution = data["resolution"]
            membership = data["membership"]

            # Group entities by community
            community_groups: dict[int, list[str]] = {}
            for idx, comm_id in enumerate(membership):
                entity_id = (
                    nodes_result[idx].get("entity_id") or nodes_result[idx]["eid"]
                )
                community_groups.setdefault(comm_id, []).append(entity_id)

            for comm_id, member_entity_ids in community_groups.items():
                community_node_id = f"{graph_id_str}_l{level}_c{comm_id}"

                # Create __Community__ node
                await neo4j_client.execute_query(
                    """
                    MERGE (c:__Community__ {id: $community_id, graph_id: $graph_id})
                    SET c.level = $level,
                        c.resolution = $resolution,
                        c.algorithm = 'leiden',
                        c.entity_count = $member_count,
                        c.status = 'active',
                        c.last_updated = datetime()
                    """,
                    {
                        "community_id": community_node_id,
                        "graph_id": graph_id_str,
                        "level": level,
                        "resolution": resolution,
                        "member_count": len(member_entity_ids),
                    },
                )
                total_communities += 1

                # Link entity members via IN_COMMUNITY (matches the
                # Celery detector + every reader; carries graph_id+level
                # on the relationship so per-level queries can filter).
                for entity_id in member_entity_ids:
                    await neo4j_client.execute_query(
                        """
                        MATCH (e:__Entity__ {graph_id: $graph_id})
                        WHERE e.id = $entity_id OR elementId(e) = $entity_id
                        MATCH (c:__Community__ {id: $community_id, graph_id: $graph_id})
                        MERGE (e)-[:IN_COMMUNITY {graph_id: $graph_id, level: $level}]->(c)
                        """,
                        {
                            "graph_id": graph_id_str,
                            "entity_id": entity_id,
                            "community_id": community_node_id,
                            "level": level,
                        },
                    )
                    total_relationships += 1

                # Link to parent community (level - 1) via PARENT_COMMUNITY,
                # direction child->parent — matches community_tasks.py
                # _upsert_communities and the analytics readers.
                if level > 0:
                    first_idx = next(
                        (
                            i
                            for i, n in enumerate(nodes_result)
                            if (n.get("entity_id") or n["eid"]) == member_entity_ids[0]
                        ),
                        None,
                    )
                    if first_idx is not None:
                        parent_comm_id = all_communities[level - 1]["membership"][
                            first_idx
                        ]
                        parent_node_id = (
                            f"{graph_id_str}_l{level - 1}_c{parent_comm_id}"
                        )
                        await neo4j_client.execute_query(
                            """
                            MATCH (child:__Community__ {id: $child_id, graph_id: $graph_id})
                            MATCH (parent:__Community__ {id: $parent_id, graph_id: $graph_id})
                            MERGE (child)-[:PARENT_COMMUNITY {graph_id: $graph_id}]->(parent)
                            """,
                            {
                                "child_id": community_node_id,
                                "parent_id": parent_node_id,
                                "graph_id": graph_id_str,
                            },
                        )

        logger.info(
            f"Leiden detection complete for graph {graph_id}: "
            f"{total_communities} communities across {len(RESOLUTIONS)} levels"
        )

        return {
            "communities_created": total_communities,
            "relationships_created": total_relationships,
            "total_entities_processed": len(nodes_result),
            "graph_id": graph_id_str,
            "algorithm_used": "leiden",
            "levels": len(RESOLUTIONS),
            "resolutions": RESOLUTIONS,
        }

    async def _create_simple_communities(self, graph_id: UUID) -> dict[str, Any]:
        """
        Fallback method to create communities using simple clustering based on shared neighbors.

        Args:
            graph_id: UUID of the specific graph to analyze

        Returns:
            Dictionary containing creation results
        """
        try:
            logger.info(f"Creating simple communities for graph {graph_id}")

            # Find entities with shared neighbors using more flexible matching
            shared_neighbors_query = """
            MATCH (a)-[r1]-(common)-[r2]-(b)
            WHERE a.graph_id = $graph_id
            AND b.graph_id = $graph_id
            AND common.graph_id = $graph_id
            AND a.id < b.id  // Avoid duplicates
            AND a.id IS NOT NULL
            AND b.id IS NOT NULL

            WITH a, b, count(DISTINCT common) as shared_count
            WHERE shared_count >= 2  // At least 2 shared neighbors

            RETURN a.id as entity_a_id,
                coalesce(a.name, a.id) as entity_a_name,
                b.id as entity_b_id,
                coalesce(b.name, b.id) as entity_b_name,
                shared_count
            ORDER BY shared_count DESC
            LIMIT 50
            """

            results = await neo4j_client.execute_query(
                shared_neighbors_query, {"graph_id": str(graph_id)}
            )

            if not results:
                return {
                    "communities_created": 0,
                    "relationships_created": 0,
                    "message": "No shared neighbor communities found",
                }

            # Group entities into simple communities
            communities_created = 0
            relationships_created = 0

            for i, result in enumerate(results[:10]):  # Limit to 10 communities
                community_uuid = f"simple_community_{graph_id}_{i}"

                # Create community node
                create_community_query = """
                MERGE (community:__Community__ {
                    id: $community_id,
                    graph_id: $graph_id
                })
                SET community.summary = $summary,
                    community.entity_count = 2,
                    community.detection_algorithm = 'shared_neighbors',
                    community.weight = $weight,
                    community.creation_date = datetime(),
                    community.last_updated = datetime()
                """

                summary = f"Community of {result['entity_a_name']} and {result['entity_b_name']} (shared {result['shared_count']} connections)"

                await neo4j_client.execute_query(
                    create_community_query,
                    {
                        "community_id": community_uuid,
                        "graph_id": str(graph_id),
                        "summary": summary,
                        "weight": result["shared_count"] / 10.0,  # Normalize weight
                    },
                )

                communities_created += 1

                # Create relationships for both entities using flexible matching
                for entity_id in [result["entity_a_id"], result["entity_b_id"]]:
                    relationship_query = """
                    MATCH (entity)
                    WHERE entity.id = $entity_id AND entity.graph_id = $graph_id
                    MATCH (community:__Community__ {id: $community_id, graph_id: $graph_id})
                    MERGE (entity)-[:IN_COMMUNITY]->(community)
                    """

                    await neo4j_client.execute_query(
                        relationship_query,
                        {
                            "entity_id": entity_id,
                            "community_id": community_uuid,
                            "graph_id": str(graph_id),
                        },
                    )

                    relationships_created += 1

            logger.info(
                f"Created {communities_created} simple communities and {relationships_created} relationships for graph {graph_id}"
            )

            return {
                "communities_created": communities_created,
                "relationships_created": relationships_created,
                "total_entities_processed": len(results) * 2,
                "graph_id": str(graph_id),
                "algorithm_used": "shared_neighbors",
            }

        except Exception as e:
            logger.error(
                f"Failed to create simple communities for {graph_id}: {e}",
                exc_info=True,
            )
            return {
                "communities_created": 0,
                "relationships_created": 0,
                "error": "community_creation_failed",
                "graph_id": str(graph_id),
            }

    def _generate_community_id(
        self, graph_id: UUID, community_id: int, members: list[dict]
    ) -> str:
        """
        Generate a unique, deterministic community ID based on members.

        Args:
            graph_id: UUID of the graph
            community_id: Original community ID from algorithm
            members: List of community members

        Returns:
            Unique community ID string
        """
        # Sort members by ID for consistency
        sorted_ids = sorted([member["entity_id"] for member in members])

        # Create hash from graph_id + community_id + member IDs
        content = f"{graph_id}_{community_id}_{'_'.join(sorted_ids)}"
        community_hash = hashlib.md5(content.encode()).hexdigest()[:12]

        return f"community_{graph_id}_{community_hash}"

    def _generate_community_summary(self, members: list[dict]) -> str:
        """
        Generate a human-readable summary for a community.

        Args:
            members: List of community members with entity info

        Returns:
            Community summary string
        """
        entity_names = [member["entity_name"] for member in members]

        if len(entity_names) <= 3:
            names_str = ", ".join(entity_names)
        else:
            names_str = (
                f"{', '.join(entity_names[:3])} and {len(entity_names) - 3} others"
            )

        # Determine primary entity types
        all_labels = []
        for member in members:
            all_labels.extend(member.get("entity_labels", []))

        # Count label frequency (excluding __Entity__)
        label_counts = {}
        for label in all_labels:
            if label != "__Entity__":
                label_counts[label] = label_counts.get(label, 0) + 1

        if label_counts:
            primary_type = max(label_counts, key=label_counts.get)
            summary = f"Community of {len(members)} entities primarily about {primary_type.lower()}: {names_str}"
        else:
            summary = f"Community of {len(members)} related entities: {names_str}"

        return summary

    async def get_community_search_context(
        self, query_text: str, graph_id: UUID, top_k: int = 5
    ) -> dict[str, Any]:
        """
        Search communities by text similarity and return context for RAG.

        Args:
            query_text: Text query to search communities
            graph_id: UUID of the specific graph
            top_k: Number of top communities to return

        Returns:
            Dictionary containing community search results
        """
        try:
            # For now, search by text matching in community summaries
            # TODO: Implement embedding-based search when embeddings are ready
            search_query = """
            MATCH (community:__Community__)
            WHERE community.graph_id = $graph_id
            AND (
                community.summary CONTAINS $query_text
                OR any(word IN split(toLower($query_text), ' ')
                       WHERE community.summary CONTAINS word)
            )

            // Get community members
            MATCH (entity:__Entity__)-[:IN_COMMUNITY]->(community)
            WHERE entity.graph_id = $graph_id

            RETURN community.id as community_id,
                   community.summary as summary,
                   community.entity_count as entity_count,
                   community.weight as weight,
                   community.detection_algorithm as algorithm,
                   collect({id: entity.id, name: entity.name}) as members
            ORDER BY community.weight DESC, community.entity_count DESC
            LIMIT $top_k
            """

            results = await neo4j_client.execute_query(
                search_query,
                {
                    "query_text": query_text.lower(),
                    "graph_id": str(graph_id),
                    "top_k": top_k,
                },
            )

            communities = []
            for result in results:
                communities.append(
                    {
                        "community_id": result["community_id"],
                        "summary": result["summary"],
                        "entity_count": result["entity_count"],
                        "weight": result["weight"],
                        "algorithm": result["algorithm"],
                        "members": result["members"],
                    }
                )

            return {
                "communities": communities,
                "search_type": "text_matching",
                "query": query_text,
                "graph_id": str(graph_id),
            }

        except Exception as e:
            logger.error(f"Community search failed: {e}")
            return {"communities": [], "error": str(e), "graph_id": str(graph_id)}

    # ==================== CENTRALITY ANALYSIS ====================

    async def get_influential_context(
        self, query: str, graph_id: UUID
    ) -> dict[str, Any]:
        """
        Get highly connected nodes using Neo4j GDS PageRank with graph_id filtering.

        Args:
            query: User query for context (used for logging/debugging)
            graph_id: UUID of the specific graph to analyze

        Returns:
            Dictionary containing influential nodes information
        """
        try:
            # Try advanced PageRank centrality with Neo4j GDS
            pagerank_query = """
            CALL {
                // Create temporary graph projection
                CALL gds.graph.project.cypher(
                    'temp-pagerank-' + $graph_id,
                    'MATCH (n) WHERE n.graph_id = "' + $graph_id + '" AND NOT n:__Chat__ RETURN id(n) AS id, n.name AS name',
                    'MATCH (a)-[r]-(b) WHERE a.graph_id = "' + $graph_id + '" AND b.graph_id = "' + $graph_id + '" AND r.graph_id = "' + $graph_id + '" AND NOT a:__Chat__ AND NOT b:__Chat__ RETURN id(a) AS source, id(b) AS target'
                )
                YIELD graphName

                // Run PageRank algorithm
                CALL gds.pageRank.stream('temp-pagerank-' + $graph_id)
                YIELD nodeId, score

                // Get original nodes with scores. Exclude chat-namespaced
                // projection nodes (STORY-031 / ADR-020).
                MATCH (node)
                WHERE id(node) = nodeId
                  AND node.graph_id = $graph_id
                  AND NOT node:__Chat__

                WITH node, score
                ORDER BY score DESC
                LIMIT 10

                RETURN node.id as entity_id,
                    node.name as entity_name,
                    score as pagerank_score,
                    labels(node) as labels
            }

            // Clean up temporary graph
            CALL gds.graph.drop('temp-pagerank-' + $graph_id, false)
            YIELD graphName as droppedGraph

            RETURN entity_id, entity_name, pagerank_score, labels
            """

            results = await neo4j_client.execute_query(
                pagerank_query, {"graph_id": str(graph_id)}
            )

            influential = []
            for result in results:
                influential.append(
                    {
                        "id": result["entity_id"],
                        "name": result["entity_name"],
                        "pagerank_score": result["pagerank_score"],
                        "labels": result["labels"],
                        "influence_type": "pagerank_centrality",
                    }
                )

            return {"influential": influential}

        except Exception as e:
            logger.warning(f"Advanced PageRank failed: {e}")
            # Fallback to simple degree centrality
            return await self.get_simple_influential_context(query, graph_id)

    async def get_simple_influential_context(
        self, query: str, graph_id: UUID
    ) -> dict[str, Any]:
        """
        Fallback influential nodes based on degree centrality with graph_id filtering.

        Args:
            query: User query for context (used for logging/debugging)
            graph_id: UUID of the specific graph to analyze

        Returns:
            Dictionary containing influential nodes based on degree centrality
        """
        # Find high-degree nodes in this specific graph. Exclude
        # chat-namespaced nodes so the conversation shadow doesn't
        # inflate degree centrality (STORY-031 / ADR-020).
        cypher_query = """
        MATCH (n)
        WHERE n.graph_id = $graph_id
          AND NOT n:__Chat__

        OPTIONAL MATCH (n)-[r]-()
        WHERE r.graph_id = $graph_id

        WITH n, count(r) as degree
        WHERE degree > 3  // Nodes with more than 3 connections
        ORDER BY degree DESC
        LIMIT 10

        RETURN n.name as name,
            n.id as id,
            degree,
            labels(n) as labels,
            n{.*} as properties
        """

        results = await neo4j_client.execute_query(
            cypher_query, {"graph_id": str(graph_id)}
        )

        influential = []
        for result in results:
            influential.append(
                {
                    "name": result["name"],
                    "id": result["id"],
                    "degree": result["degree"],
                    "labels": result["labels"],
                    "properties": result["properties"],
                    "influence_type": "degree_centrality",
                }
            )

        return {"influential": influential}

    # ==================== NEIGHBORHOOD ANALYSIS ====================

    async def get_neighborhood_context(
        self, entities: list[dict[str, Any]], graph_id: UUID
    ) -> dict[str, Any]:
        """
        Get 1-2 hop neighbor context with proper graph_id filtering.

        Args:
            entities: List of entity dictionaries with 'id' and 'name' keys
            graph_id: UUID of the specific graph to analyze

        Returns:
            Dictionary containing neighborhood information
        """
        if not entities:
            return {"neighborhoods": [], "relationships": []}

        entity_ids = [e["id"] for e in entities]

        query = """
        MATCH (start)
        WHERE start.id IN $entity_ids AND start.graph_id = $graph_id

        MATCH (start)-[r1]-(neighbor1)
        WHERE r1.graph_id = $graph_id AND neighbor1.graph_id = $graph_id

        OPTIONAL MATCH (neighbor1)-[r2]-(neighbor2)
        WHERE r2.graph_id = $graph_id AND neighbor2.graph_id = $graph_id
        AND neighbor2.id <> start.id

        WITH start, r1, neighbor1,
            collect(DISTINCT {
                node: neighbor2{.id, .name, labels: labels(neighbor2)},
                relationship: type(r2)
            })[..3] as second_hop

        RETURN start.id as center_id,
            start.name as center_name,
            {
                relationship: type(r1),
                neighbor: neighbor1{.id, .name, labels: labels(neighbor1)},
                second_hop: second_hop
            } as neighborhood_info
        LIMIT 30
        """

        results = await neo4j_client.execute_query(
            query, {"entity_ids": entity_ids, "graph_id": str(graph_id)}
        )

        neighborhoods = {}
        relationships = []

        for result in results:
            center_id = result["center_id"]
            if center_id not in neighborhoods:
                neighborhoods[center_id] = {
                    "center": {"id": center_id, "name": result["center_name"]},
                    "neighbors": [],
                }

            neighborhoods[center_id]["neighbors"].append(result["neighborhood_info"])
            relationships.append(
                {
                    "source": center_id,
                    "target": result["neighborhood_info"]["neighbor"]["id"],
                    "type": result["neighborhood_info"]["relationship"],
                }
            )

        return {
            "neighborhoods": list(neighborhoods.values()),
            "relationships": relationships,
        }

    # ==================== PATHWAY ANALYSIS ====================

    async def get_pathway_context(
        self, entities: list[dict[str, Any]], graph_id: UUID, max_depth: int = 3
    ) -> dict[str, Any]:
        """
        Get pathways between entities with proper graph_id filtering.

        Args:
            entities: List of entity dictionaries with 'id' and 'name' keys
            graph_id: UUID of the specific graph to analyze
            max_depth: Maximum depth for path discovery (default: 3)

        Returns:
            Dictionary containing pathway information
        """
        if len(entities) < 2:
            return {"pathways": []}

        pathways = []

        # Get pathways between all pairs
        for i, start_entity in enumerate(entities):
            for _j, end_entity in enumerate(entities[i + 1 :], i + 1):
                paths = await self.find_paths_between_entities(
                    start_entity["id"], end_entity["id"], graph_id, max_depth
                )
                pathways.extend(paths)

        # Remove duplicates and sort
        unique_pathways = []
        seen_paths = set()

        for pathway in pathways:
            path_signature = tuple(pathway.get("nodes", []))
            if path_signature not in seen_paths:
                seen_paths.add(path_signature)
                unique_pathways.append(pathway)

        unique_pathways.sort(key=lambda x: x.get("length", 999))

        return {"pathways": unique_pathways[:10]}

    async def find_shortest_paths(
        self, start_id: str, end_id: str, graph_id: UUID
    ) -> list[dict[str, Any]]:
        """
        Find shortest paths between entities with proper graph_id filtering.

        Args:
            start_id: Starting entity ID
            end_id: Target entity ID
            graph_id: UUID of the specific graph to analyze

        Returns:
            List of shortest path dictionaries
        """
        query = """
        MATCH (start {id: $start_id, graph_id: $graph_id})
        MATCH (end {id: $end_id, graph_id: $graph_id})

        MATCH path = shortestPath((start)-[*1..3]-(end))
        WHERE ALL(r IN relationships(path) WHERE r.graph_id = $graph_id)
        AND ALL(n IN nodes(path) WHERE n.graph_id = $graph_id)

        RETURN [n.name FOR n IN nodes(path)] as node_names,
               [type(r) FOR r IN relationships(path)] as relationship_types,
               length(path) as path_length
        ORDER BY path_length
        LIMIT 3
        """

        results = await neo4j_client.execute_query(
            query, {"start_id": start_id, "end_id": end_id, "graph_id": str(graph_id)}
        )

        return [
            {
                "start": start_id,
                "end": end_id,
                "nodes": result["node_names"],
                "relationships": result["relationship_types"],
                "length": result["path_length"],
                "type": "shortest_path",
            }
            for result in results
        ]

    async def find_paths_between_entities(
        self, start_id: str, end_id: str, graph_id: UUID, max_depth: int
    ) -> list[dict[str, Any]]:
        """
        Advanced pathfinding with proper graph_id filtering.

        Args:
            start_id: Starting entity ID
            end_id: Target entity ID
            graph_id: UUID of the specific graph to analyze
            max_depth: Maximum depth for path discovery

        Returns:
            List of path dictionaries
        """
        query = """
        MATCH (start {id: $start_id, graph_id: $graph_id})
        MATCH (end {id: $end_id, graph_id: $graph_id})

        CALL apoc.path.allSimplePaths(start, end, '', $max_depth) YIELD path
        WHERE ALL(n IN nodes(path) WHERE n.graph_id = $graph_id)
        AND ALL(r IN relationships(path) WHERE r.graph_id = $graph_id)

        WITH path, length(path) as path_length
        ORDER BY path_length
        LIMIT 5

        RETURN [n.name FOR n IN nodes(path)] as path_nodes,
               [{type: type(r), properties: properties(r)} FOR r IN relationships(path)] as path_relationships,
               path_length
        """

        results = await neo4j_client.execute_query(
            query,
            {
                "start_id": start_id,
                "end_id": end_id,
                "graph_id": str(graph_id),
                "max_depth": max_depth,
            },
        )

        return [
            {
                "start": start_id,
                "end": end_id,
                "nodes": result["path_nodes"],
                "relationships": result["path_relationships"],
                "length": result["path_length"],
                "type": "advanced_path",
            }
            for result in results
        ]

    # ==================== TEMPORAL ANALYSIS ====================

    async def get_temporal_context(
        self, entities: list[dict[str, Any]], graph_id: UUID
    ) -> dict[str, Any]:
        """
        Get temporal/time-based context with graph_id filtering.

        Args:
            entities: List of entity dictionaries with 'id' and 'name' keys
            graph_id: UUID of the specific graph to analyze

        Returns:
            Dictionary containing temporal context information
        """
        if not entities:
            return {"temporal": []}

        entity_ids = [e["id"] for e in entities]

        # Look for date-related properties in relationships and nodes
        query = """
        MATCH (n)-[r]-(m)
        WHERE n.id IN $entity_ids
        AND n.graph_id = $graph_id
        AND r.graph_id = $graph_id
        AND m.graph_id = $graph_id
        AND (
            r.start_date IS NOT NULL OR
            r.end_date IS NOT NULL OR
            r.date IS NOT NULL OR
            r.year IS NOT NULL OR
            r.created_at IS NOT NULL OR
            r.timestamp IS NOT NULL OR
            n.birth_date IS NOT NULL OR
            n.founded IS NOT NULL OR
            m.birth_date IS NOT NULL OR
            m.founded IS NOT NULL
        )

        WITH n, r, m,
            coalesce(
                r.start_date,
                r.end_date,
                r.date,
                r.year,
                r.created_at,
                r.timestamp,
                n.birth_date,
                n.founded,
                m.birth_date,
                m.founded
            ) as date_info

        WHERE date_info IS NOT NULL

        RETURN n.name as entity_name,
            type(r) as relationship_type,
            m.name as connected_entity,
            date_info,
            r{.*} as relationship_properties
        ORDER BY date_info DESC
        LIMIT 20
        """

        results = await neo4j_client.execute_query(
            query, {"entity_ids": entity_ids, "graph_id": str(graph_id)}
        )

        temporal = []
        for result in results:
            temporal.append(
                {
                    "entity": result["entity_name"],
                    "relationship": result["relationship_type"],
                    "connected_to": result["connected_entity"],
                    "date": str(
                        result["date_info"]
                    ),  # Convert to string for JSON serialization
                    "relationship_properties": result["relationship_properties"],
                    "context_type": "temporal",
                }
            )

        return {"temporal": temporal}

    # ==================== GRAPH STATISTICS ====================

    async def get_graph_statistics(self, graph_id: UUID) -> dict[str, Any]:
        """
        Get comprehensive graph statistics for the specified graph.

        Args:
            graph_id: UUID of the specific graph to analyze

        Returns:
            Dictionary containing graph statistics
        """
        try:
            # Basic graph statistics. Exclude chat-namespaced nodes so
            # the conversation shadow projected by TASK-106 doesn't
            # inflate node counts (STORY-031 / ADR-020).
            stats_query = """
            MATCH (n)
            WHERE n.graph_id = $graph_id AND NOT n:__Chat__
            WITH count(n) as node_count

            MATCH ()-[r]-()
            WHERE r.graph_id = $graph_id
            WITH node_count, count(r) as rel_count

            MATCH (n)
            WHERE n.graph_id = $graph_id AND NOT n:__Chat__
            WITH node_count, rel_count, labels(n) as node_labels
            UNWIND node_labels as label
            WITH node_count, rel_count, label
            WHERE label <> 'Entity'  // Skip generic labels

            WITH node_count, rel_count, collect(DISTINCT label) as entity_types

            MATCH ()-[r]-()
            WHERE r.graph_id = $graph_id
            WITH node_count, rel_count, entity_types, type(r) as rel_type

            RETURN node_count,
                rel_count,
                entity_types,
                collect(DISTINCT rel_type) as relationship_types
            """

            result = await neo4j_client.execute_query(
                stats_query, {"graph_id": str(graph_id)}
            )

            if result:
                stats = result[0]
                node_count = stats["node_count"]
                rel_count = stats["rel_count"]

                return {
                    "node_count": node_count,
                    "relationship_count": rel_count,
                    "density": (
                        rel_count / (node_count * (node_count - 1))
                        if node_count > 1
                        else 0
                    ),
                    "entity_types": stats["entity_types"][:10],  # Top 10 types
                    "relationship_types": stats["relationship_types"][
                        :10
                    ],  # Top 10 types
                    "avg_degree": (2 * rel_count / node_count) if node_count > 0 else 0,
                    "computed_at": datetime.now().isoformat(),
                }
            else:
                return {"node_count": 0, "relationship_count": 0}

        except Exception as e:
            logger.warning(f"Failed to get graph statistics: {e}")
            return {"error": str(e)}

    async def precompute_and_cache_statistics(self, graph_id: UUID) -> None:
        """
        Precompute and cache graph statistics for better performance.

        Args:
            graph_id: UUID of the specific graph to analyze
        """
        try:
            # Get comprehensive statistics
            stats = await self.get_graph_statistics(graph_id)

            # Cache with timestamp
            self.cached_statistics[str(graph_id)] = {
                **stats,
                "cached_at": datetime.now(),
            }

            logger.info(
                f"Precomputed statistics for graph {graph_id}: "
                f"{stats.get('node_count', 0)} nodes, "
                f"{stats.get('relationship_count', 0)} relationships"
            )

        except Exception as e:
            logger.warning(f"Failed to precompute graph statistics: {e}")
            self.cached_statistics[str(graph_id)] = {
                "error": str(e),
                "node_count": 0,
                "relationship_count": 0,
                "cached_at": datetime.now(),
            }

    def get_cached_statistics(self, graph_id: UUID) -> dict[str, Any] | None:
        """
        Get cached statistics for a graph if available.

        Args:
            graph_id: UUID of the specific graph

        Returns:
            Cached statistics dictionary or None if not cached
        """
        return self.cached_statistics.get(str(graph_id))

    # ==================== PER-LEVEL LLM SUMMARIES ====================

    async def _generate_level_summaries(self, graph_id: str) -> dict[str, Any]:
        """
        Generate LLM summaries for every __Community__ node across all levels.

        Strategy:
        - Level 0 (finest): prompt built from member __Entity__ names + types.
        - Level 1+: prompt built from child community summaries (hierarchical roll-up).
          Children are linked via PARENT_COMMUNITY edges to their parent community.

        Stale summaries are cleared before generation so that re-runs after new
        community detection always produce fresh output.

        Security rules enforced:
        - Every Cypher query includes {graph_id: $graph_id} in MATCH clause.
        - No f-strings in Cypher query text — all values passed as parameters.

        Args:
            graph_id: UUID string of the target graph.

        Returns:
            Dict with per-level counts of communities summarised.
        """
        import asyncio as _asyncio

        from openai import AsyncOpenAI

        openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        llm_model = "gpt-4o-mini"

        # ── Step 1: Clear stale summaries ─────────────────────────────────────
        await neo4j_client.execute_query(
            "MATCH (c:__Community__ {graph_id: $graph_id}) SET c.summary = null",
            {"graph_id": graph_id},
        )

        # ── Step 2: Discover all levels present (ascending order) ─────────────
        level_rows = await neo4j_client.execute_query(
            "MATCH (c:__Community__ {graph_id: $graph_id}) "
            "RETURN DISTINCT c.level AS level ORDER BY level",
            {"graph_id": graph_id},
        )
        levels = [r["level"] for r in level_rows if r["level"] is not None]

        summaries_per_level: dict[str, int] = {}
        LLM_CONCURRENCY = getattr(settings, "LLM_SUMMARY_CONCURRENCY", 5)
        semaphore = _asyncio.Semaphore(LLM_CONCURRENCY)

        async def _call_llm(prompt: str) -> str:
            """Call the LLM with the given user prompt; return generated text."""
            async with semaphore:
                response = await openai_client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a knowledge graph analyst. "
                                "Generate a concise, informative summary of a cluster "
                                "of related entities from a knowledge graph."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=200,
                )
                return response.choices[0].message.content.strip()

        # Process levels in ascending order so child summaries exist before parents
        for level in sorted(levels):
            # Fetch all community IDs at this level
            community_rows = await neo4j_client.execute_query(
                "MATCH (c:__Community__ {graph_id: $graph_id, level: $level}) "
                "RETURN c.id AS community_id",
                {"graph_id": graph_id, "level": level},
            )
            community_ids = [r["community_id"] for r in community_rows]

            async def _summarise_community(cid: str, lvl: int) -> bool:
                """Build prompt, call LLM, write summary. Returns True on success."""
                if lvl == 0:
                    # Level-0: summarise from raw entity names + types
                    entity_rows = await neo4j_client.execute_query(
                        """
                        MATCH (e:__Entity__ {graph_id: $graph_id})
                              -[:IN_COMMUNITY {graph_id: $graph_id, level: $level}]->
                              (c:__Community__ {id: $community_id, graph_id: $graph_id})
                        RETURN e.name AS name, labels(e) AS lbls
                        LIMIT 50
                        """,
                        {
                            "graph_id": graph_id,
                            "level": lvl,
                            "community_id": cid,
                        },
                    )
                    entity_list = ", ".join(
                        "{name} ({etype})".format(
                            name=r["name"] or cid,
                            etype=next(
                                (
                                    lbl
                                    for lbl in (r["lbls"] or [])
                                    if lbl != "__Entity__"
                                ),
                                "Entity",
                            ),
                        )
                        for r in entity_rows
                    )
                    if not entity_list:
                        return False
                    prompt = (
                        "Summarize the entities and their relationships in this group: "
                        + entity_list
                    )
                else:
                    # Level 1+: roll up from child community summaries.
                    # Children (level lvl-1) link to parent via PARENT_COMMUNITY.
                    child_rows = await neo4j_client.execute_query(
                        """
                        MATCH (child:__Community__ {graph_id: $graph_id, level: $child_level})
                              -[:PARENT_COMMUNITY {graph_id: $graph_id}]->
                              (parent:__Community__ {id: $community_id, graph_id: $graph_id})
                        WHERE child.summary IS NOT NULL
                        RETURN child.summary AS summary
                        LIMIT 20
                        """,
                        {
                            "graph_id": graph_id,
                            "child_level": lvl - 1,
                            "community_id": cid,
                        },
                    )
                    child_summaries = [r["summary"] for r in child_rows if r["summary"]]
                    if not child_summaries:
                        return False
                    joined = " | ".join(child_summaries)
                    if lvl == 1:
                        prompt = (
                            "Summarize the themes and patterns across these sub-communities: "
                            + joined
                        )
                    else:
                        prompt = (
                            "What are the overarching topics and insights? " + joined
                        )

                try:
                    summary_text = await _call_llm(prompt)
                except Exception as exc:
                    logger.warning(
                        f"LLM summary failed for community {cid} at level {lvl}: {exc}"
                    )
                    summary_text = f"Community at level {lvl} (summary unavailable)"

                # Write summary back — all values as parameters, no Cypher interpolation
                await neo4j_client.execute_query(
                    """
                    MATCH (c:__Community__ {id: $community_id, graph_id: $graph_id})
                    SET c.summary = $summary,
                        c.summary_level = $level,
                        c.last_updated = datetime()
                    """,
                    {
                        "community_id": cid,
                        "graph_id": graph_id,
                        "summary": summary_text,
                        "level": lvl,
                    },
                )
                return True

            results = await _asyncio.gather(
                *(_summarise_community(cid, level) for cid in community_ids),
                return_exceptions=False,
            )
            count_written = sum(1 for r in results if r is True)
            summaries_per_level[str(level)] = count_written
            logger.info(
                f"Level {level}: summarised {count_written}/{len(community_ids)} "
                f"communities for graph {graph_id}"
            )

        return {
            "status": "completed",
            "graph_id": graph_id,
            "summaries_per_level": summaries_per_level,
            "total_summarised": sum(summaries_per_level.values()),
        }

    # ==================== COMPREHENSIVE ANALYSIS ====================

    async def comprehensive_graph_analysis(
        self,
        entities: list[dict[str, Any]],
        graph_id: UUID,
        include_communities: bool = True,
        include_influence: bool = True,
        include_pathways: bool = True,
        include_temporal: bool = True,
    ) -> dict[str, Any]:
        """
        Perform comprehensive graph analysis combining multiple analytics methods.

        Args:
            entities: List of entity dictionaries to analyze
            graph_id: UUID of the specific graph to analyze
            include_communities: Whether to include community detection
            include_influence: Whether to include influence analysis
            include_pathways: Whether to include pathway analysis
            include_temporal: Whether to include temporal analysis

        Returns:
            Dictionary containing comprehensive analysis results
        """
        analysis_results = {
            "graph_id": str(graph_id),
            "entities_analyzed": len(entities),
            "analysis_timestamp": datetime.now().isoformat(),
        }

        try:
            # Always include neighborhood analysis
            neighborhood_ctx = await self.get_neighborhood_context(entities, graph_id)
            analysis_results["neighborhoods"] = neighborhood_ctx

            if include_communities:
                community_ctx = await self.get_community_context(entities, graph_id)
                analysis_results["communities"] = community_ctx

            if include_influence:
                influence_ctx = await self.get_influential_context("", graph_id)
                analysis_results["influential"] = influence_ctx

            if include_pathways and len(entities) >= 2:
                pathway_ctx = await self.get_pathway_context(entities, graph_id)
                analysis_results["pathways"] = pathway_ctx

            if include_temporal:
                temporal_ctx = await self.get_temporal_context(entities, graph_id)
                analysis_results["temporal"] = temporal_ctx

            # Include graph statistics
            stats = await self.get_graph_statistics(graph_id)
            analysis_results["statistics"] = stats

            analysis_results["success"] = True

        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            analysis_results["error"] = str(e)
            analysis_results["success"] = False

        return analysis_results


# Create global instance
analytics_service = GraphAnalyticsService()
