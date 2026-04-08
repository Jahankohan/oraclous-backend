"""
Snapshot Service

Manages GraphVersion nodes in Neo4j — zero-copy pointer-based snapshots.
Materialized snapshots are deferred to Phase 4; snapshot_strategy is a no-op
placeholder per CTO design decision (ORA-55).

Architecture rules:
- Rule #4: All Cypher queries filter by graph_id
- Rule #5: FastAPI endpoints use AsyncDriver (this service); Celery uses sync driver
- Rule #6: GraphVersion nodes live in Neo4j, not PostgreSQL
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client

logger = get_logger(__name__)


class SnapshotService:
    """
    CRUD and diff operations for GraphVersion nodes.

    All methods are async and use the shared async Neo4j driver.
    Celery snapshot tasks create their own task-scoped sync driver (see background_jobs.py).
    """

    # ------------------------------------------------------------------
    # Index setup (idempotent, called at startup)
    # ------------------------------------------------------------------

    async def ensure_indexes(self) -> None:
        """Create Neo4j indexes required for versioning queries (idempotent)."""
        index_queries = [
            "CREATE INDEX entity_transaction_time_idx IF NOT EXISTS FOR (e:__Entity__) ON (e.graph_id, e.transaction_time)",
            "CREATE INDEX entity_invalidated_at_idx IF NOT EXISTS FOR (e:__Entity__) ON (e.graph_id, e.invalidated_at)",
            "CREATE INDEX version_graph_idx IF NOT EXISTS FOR (v:GraphVersion) ON (v.graph_id, v.captured_at)",
            "CREATE INDEX version_number_idx IF NOT EXISTS FOR (v:GraphVersion) ON (v.graph_id, v.version_number)",
            # Composite index on relationships — (graph_id, transaction_time, invalidated_at)
            "CREATE INDEX rel_version_composite_idx IF NOT EXISTS FOR ()-[r]-() ON (r.graph_id, r.transaction_time, r.invalidated_at)",
            # ORA-138: Relationship temporal (valid-time) indexes — composite for
            # queries that filter by r.graph_id + temporal props, and standalone
            # for traversal queries where graph_id is only on the node side.
            "CREATE INDEX rel_temporal_idx IF NOT EXISTS FOR ()-[r]-() ON (r.graph_id, r.valid_from, r.valid_to)",
            "CREATE INDEX rel_valid_from_idx IF NOT EXISTS FOR ()-[r]-() ON (r.valid_from)",
            "CREATE INDEX rel_valid_to_idx IF NOT EXISTS FOR ()-[r]-() ON (r.valid_to)",
        ]
        for q in index_queries:
            try:
                await neo4j_client.execute_query(q, {})
            except Exception as e:
                logger.warning(f"Index creation skipped: {e}")

    # ------------------------------------------------------------------
    # Create snapshot
    # ------------------------------------------------------------------

    async def create_snapshot(
        self,
        graph_id: str,
        label: str | None,
        description: str | None,
        created_by: str,
        is_auto: bool = False,
        parent_version_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a GraphVersion node anchored to the current datetime.

        This is a zero-copy (pointer-based) snapshot — no entity data is duplicated.
        snapshot_strategy is a no-op placeholder for Phase 4 materialized snapshots.
        """
        count_q = """
        MATCH (e:__Entity__ {graph_id: $graph_id})
        WHERE e.invalidated_at IS NULL
        WITH count(e) AS entity_count
        OPTIONAL MATCH (a:__Entity__ {graph_id: $graph_id})-[r {graph_id: $graph_id}]->(b:__Entity__ {graph_id: $graph_id})
        WHERE r.invalidated_at IS NULL
        RETURN entity_count, count(r) AS relationship_count
        """
        counts = await neo4j_client.execute_query(count_q, {"graph_id": graph_id})
        entity_count = int(counts[0]["entity_count"]) if counts else 0
        relationship_count = int(counts[0]["relationship_count"]) if counts else 0

        num_q = """
        MATCH (v:GraphVersion {graph_id: $graph_id})
        RETURN coalesce(max(v.version_number), 0) + 1 AS next_num
        """
        num_result = await neo4j_client.execute_query(num_q, {"graph_id": graph_id})
        version_number = int(num_result[0]["next_num"]) if num_result else 1

        version_id = str(uuid.uuid4())
        now = datetime.now(UTC)

        create_q = """
        MATCH (g:Graph {graph_id: $graph_id})
        CREATE (v:GraphVersion {
            version_id:         $version_id,
            graph_id:           $graph_id,
            version_number:     $version_number,
            label:              $label,
            description:        $description,
            captured_at:        datetime($captured_at),
            created_by:         $created_by,
            parent_version_id:  $parent_version_id,
            is_auto:            $is_auto,
            snapshot_strategy:  'pointer',
            entity_count:       $entity_count,
            relationship_count: $relationship_count,
            created_at:         datetime($created_at)
        })
        CREATE (g)-[:HAS_VERSION]->(v)
        RETURN v
        """
        params = {
            "graph_id": graph_id,
            "version_id": version_id,
            "version_number": version_number,
            "label": label,
            "description": description,
            "captured_at": now.isoformat(),
            "created_by": created_by,
            "parent_version_id": parent_version_id,
            "is_auto": is_auto,
            "entity_count": entity_count,
            "relationship_count": relationship_count,
            "created_at": now.isoformat(),
        }
        result = await neo4j_client.execute_query(create_q, params)

        if not result:
            # Graph node may not exist — create without MATCH on Graph
            create_no_graph_q = """
            CREATE (v:GraphVersion {
                version_id:         $version_id,
                graph_id:           $graph_id,
                version_number:     $version_number,
                label:              $label,
                description:        $description,
                captured_at:        datetime($captured_at),
                created_by:         $created_by,
                parent_version_id:  $parent_version_id,
                is_auto:            $is_auto,
                snapshot_strategy:  'pointer',
                entity_count:       $entity_count,
                relationship_count: $relationship_count,
                created_at:         datetime($created_at)
            })
            RETURN v
            """
            result = await neo4j_client.execute_query(create_no_graph_q, params)

        logger.info(
            f"Created snapshot {version_id} (v{version_number}) for graph {graph_id}"
        )
        return self._node_to_dict(result[0]["v"])

    # ------------------------------------------------------------------
    # List snapshots
    # ------------------------------------------------------------------

    async def list_snapshots(self, graph_id: str) -> list[dict[str, Any]]:
        """Return all snapshots for a graph, newest first."""
        q = """
        MATCH (v:GraphVersion {graph_id: $graph_id})
        RETURN v
        ORDER BY v.captured_at DESC
        """
        results = await neo4j_client.execute_query(q, {"graph_id": graph_id})
        return [self._node_to_dict(r["v"]) for r in results]

    # ------------------------------------------------------------------
    # Get single snapshot
    # ------------------------------------------------------------------

    async def get_snapshot(
        self, graph_id: str, snapshot_id: str
    ) -> dict[str, Any] | None:
        """Fetch a single snapshot by ID, scoped to graph_id."""
        q = """
        MATCH (v:GraphVersion {graph_id: $graph_id, version_id: $version_id})
        RETURN v
        """
        results = await neo4j_client.execute_query(
            q, {"graph_id": graph_id, "version_id": snapshot_id}
        )
        return self._node_to_dict(results[0]["v"]) if results else None

    # ------------------------------------------------------------------
    # Delete snapshot
    # ------------------------------------------------------------------

    async def delete_snapshot(self, graph_id: str, snapshot_id: str) -> bool:
        """Delete a GraphVersion node. Returns True if deleted, False if not found."""
        existing = await self.get_snapshot(graph_id, snapshot_id)
        if not existing:
            return False
        q = """
        MATCH (v:GraphVersion {graph_id: $graph_id, version_id: $version_id})
        DETACH DELETE v
        """
        await neo4j_client.execute_query(
            q, {"graph_id": graph_id, "version_id": snapshot_id}
        )
        logger.info(f"Deleted snapshot {snapshot_id} for graph {graph_id}")
        return True

    # ------------------------------------------------------------------
    # Diff two snapshots
    # ------------------------------------------------------------------

    async def diff_snapshots(
        self,
        graph_id: str,
        snapshot_id: str,
        compare_to: str,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        Compute entities and relationships added/deleted between two snapshots.

        The older snapshot is always treated as the 'from' side.
        Cross-graph diff is out of scope for Phase 3.
        """
        v1 = await self.get_snapshot(graph_id, snapshot_id)
        v2 = await self.get_snapshot(graph_id, compare_to)

        if not v1 or not v2:
            raise ValueError("One or both snapshots not found")

        t1 = _snapshot_ts(v1)
        t2 = _snapshot_ts(v2)

        # Ensure v1 is older
        if t1 > t2:
            v1, v2, t1, t2 = v2, v1, t2, t1

        count_params = {"graph_id": graph_id, "v1_ts": t1, "v2_ts": t2}

        ea_q = """
        MATCH (e:__Entity__ {graph_id: $graph_id})
        WHERE e.transaction_time > datetime($v1_ts)
          AND e.transaction_time <= datetime($v2_ts)
          AND (e.invalidated_at IS NULL OR e.invalidated_at > datetime($v2_ts))
        RETURN count(e) AS cnt
        """
        ed_q = """
        MATCH (e:__Entity__ {graph_id: $graph_id})
        WHERE e.invalidated_at > datetime($v1_ts)
          AND e.invalidated_at <= datetime($v2_ts)
        RETURN count(e) AS cnt
        """
        ra_q = """
        MATCH (a:__Entity__ {graph_id: $graph_id})-[r {graph_id: $graph_id}]->(b:__Entity__ {graph_id: $graph_id})
        WHERE r.transaction_time > datetime($v1_ts)
          AND r.transaction_time <= datetime($v2_ts)
          AND (r.invalidated_at IS NULL OR r.invalidated_at > datetime($v2_ts))
        RETURN count(r) AS cnt
        """
        rd_q = """
        MATCH (a:__Entity__ {graph_id: $graph_id})-[r {graph_id: $graph_id}]->(b:__Entity__ {graph_id: $graph_id})
        WHERE r.invalidated_at > datetime($v1_ts)
          AND r.invalidated_at <= datetime($v2_ts)
        RETURN count(r) AS cnt
        """

        ea_cnt = int(
            (await neo4j_client.execute_query(ea_q, count_params) or [{"cnt": 0}])[0][
                "cnt"
            ]
        )
        ed_cnt = int(
            (await neo4j_client.execute_query(ed_q, count_params) or [{"cnt": 0}])[0][
                "cnt"
            ]
        )
        ra_cnt = int(
            (await neo4j_client.execute_query(ra_q, count_params) or [{"cnt": 0}])[0][
                "cnt"
            ]
        )
        rd_cnt = int(
            (await neo4j_client.execute_query(rd_q, count_params) or [{"cnt": 0}])[0][
                "cnt"
            ]
        )

        changes_q = """
        CALL {
            MATCH (e:__Entity__ {graph_id: $graph_id})
            WHERE e.transaction_time > datetime($v1_ts)
              AND e.transaction_time <= datetime($v2_ts)
              AND (e.invalidated_at IS NULL OR e.invalidated_at > datetime($v2_ts))
            RETURN 'entity_added' AS type,
                   coalesce(e.entity_id, elementId(e)) AS id,
                   coalesce(e.name, '') AS name,
                   coalesce(e.type, '') AS entity_type,
                   '' AS subject, '' AS predicate, '' AS object,
                   e.transaction_time AS ts
            UNION ALL
            MATCH (e:__Entity__ {graph_id: $graph_id})
            WHERE e.invalidated_at > datetime($v1_ts)
              AND e.invalidated_at <= datetime($v2_ts)
            RETURN 'entity_deleted' AS type,
                   coalesce(e.entity_id, elementId(e)) AS id,
                   coalesce(e.name, '') AS name,
                   coalesce(e.type, '') AS entity_type,
                   '' AS subject, '' AS predicate, '' AS object,
                   e.invalidated_at AS ts
            UNION ALL
            MATCH (a:__Entity__ {graph_id: $graph_id})-[r {graph_id: $graph_id}]->(b:__Entity__ {graph_id: $graph_id})
            WHERE r.transaction_time > datetime($v1_ts)
              AND r.transaction_time <= datetime($v2_ts)
              AND (r.invalidated_at IS NULL OR r.invalidated_at > datetime($v2_ts))
            RETURN 'relationship_added' AS type,
                   '' AS id, '' AS name, '' AS entity_type,
                   coalesce(a.name, '') AS subject,
                   type(r) AS predicate,
                   coalesce(b.name, '') AS object,
                   r.transaction_time AS ts
            UNION ALL
            MATCH (a:__Entity__ {graph_id: $graph_id})-[r {graph_id: $graph_id}]->(b:__Entity__ {graph_id: $graph_id})
            WHERE r.invalidated_at > datetime($v1_ts)
              AND r.invalidated_at <= datetime($v2_ts)
            RETURN 'relationship_deleted' AS type,
                   '' AS id, '' AS name, '' AS entity_type,
                   coalesce(a.name, '') AS subject,
                   type(r) AS predicate,
                   coalesce(b.name, '') AS object,
                   r.invalidated_at AS ts
        }
        RETURN type, id, name, entity_type, subject, predicate, object, ts
        ORDER BY ts
        SKIP $offset LIMIT $limit
        """
        change_rows = await neo4j_client.execute_query(
            changes_q,
            {
                "graph_id": graph_id,
                "v1_ts": t1,
                "v2_ts": t2,
                "offset": offset,
                "limit": limit,
            },
        )
        changes = [
            {
                "type": r["type"],
                "entity_id": r["id"] or None,
                "name": r["name"] or None,
                "entity_type": r["entity_type"] or None,
                "subject": r["subject"] or None,
                "predicate": r["predicate"] or None,
                "object": r["object"] or None,
                "timestamp": str(r["ts"]) if r["ts"] else None,
            }
            for r in (change_rows or [])
        ]

        total_changes = ea_cnt + ed_cnt + ra_cnt + rd_cnt
        return {
            "from_version": {
                "version_id": v1["version_id"],
                "label": v1.get("label"),
                "captured_at": v1.get("captured_at"),
            },
            "to_version": {
                "version_id": v2["version_id"],
                "label": v2.get("label"),
                "captured_at": v2.get("captured_at"),
            },
            "summary": {
                "entities_added": ea_cnt,
                "entities_deleted": ed_cnt,
                "relationships_added": ra_cnt,
                "relationships_deleted": rd_cnt,
                "property_changes": 0,
            },
            "changes": changes,
            "offset": offset,
            "limit": limit,
            "has_more": (offset + limit) < total_changes,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _node_to_dict(node: Any) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for key, val in node.items():
            if hasattr(val, "iso_format"):
                d[key] = val.iso_format()
            else:
                d[key] = val
        return d


def _snapshot_ts(snapshot: dict[str, Any]) -> str:
    """Return captured_at as an ISO-8601 string for Cypher datetime() parameter."""
    ts = snapshot.get("captured_at")
    if ts is None:
        raise ValueError("Snapshot missing captured_at")
    if isinstance(ts, str):
        return ts
    if hasattr(ts, "iso_format"):
        return ts.iso_format()
    return str(ts)


# Module-level singleton
snapshot_service = SnapshotService()
