"""
Graph Versioning Service

Implements the metadata-anchored hybrid versioning strategy from ORA-51:
- Zero-copy snapshots anchored by a captured_at timestamp
- Diff via transaction_time window queries
- Full and partial rollback via soft-invalidation (invalidated_at) semantics

Architecture rules enforced:
- Rule #4: All Cypher queries filter by graph_id
- Rule #5: Celery snapshot tasks use sync Driver / NullPool (handled in background_jobs.py)
- Rule #6: GraphVersion and VersionDiff nodes live in Neo4j — no PostgreSQL
- Rule #7: No premature abstraction — direct Cypher per operation
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.neo4j_client import neo4j_client
from app.core.logging import get_logger

logger = get_logger(__name__)


class VersioningService:
    """
    Creates, lists, diffs, and rolls back graph versions.

    All methods are async and use the shared async Neo4j driver from neo4j_client.
    Celery snapshot tasks create their own task-scoped sync driver (see background_jobs.py).
    """

    # ------------------------------------------------------------------
    # Index / migration
    # ------------------------------------------------------------------

    async def ensure_indexes(self) -> None:
        """Create Neo4j indexes required for versioning queries (idempotent)."""
        index_queries = [
            "CREATE INDEX entity_invalidated_at_idx IF NOT EXISTS FOR (e:__Entity__) ON (e.graph_id, e.invalidated_at)",
            "CREATE INDEX version_graph_idx IF NOT EXISTS FOR (v:GraphVersion) ON (v.graph_id, v.captured_at)",
            "CREATE INDEX version_number_idx IF NOT EXISTS FOR (v:GraphVersion) ON (v.graph_id, v.version_number)",
            # Relationship indexes — wildcard syntax (Neo4j 5.x)
            "CREATE INDEX rel_transaction_time_idx IF NOT EXISTS FOR ()-[r]-() ON (r.graph_id, r.transaction_time)",
            "CREATE INDEX rel_invalidated_at_idx IF NOT EXISTS FOR ()-[r]-() ON (r.graph_id, r.invalidated_at)",
            # Composite index on relationships per CTO design decision (ORA-55)
            "CREATE INDEX rel_version_composite_idx IF NOT EXISTS FOR ()-[r]-() ON (r.graph_id, r.transaction_time, r.invalidated_at)",
        ]
        for q in index_queries:
            try:
                await neo4j_client.execute_query(q, {})
            except Exception as e:
                # Index creation errors are non-fatal (e.g. not supported on this edition)
                logger.warning(f"Index creation skipped: {e}")

    # ------------------------------------------------------------------
    # Create snapshot
    # ------------------------------------------------------------------

    async def create_version(
        self,
        graph_id: str,
        label: Optional[str],
        description: Optional[str],
        created_by: str,
        is_auto: bool = False,
        parent_version_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a GraphVersion node anchored to datetime().

        Returns the version metadata dict.
        """
        # Count current live entities and relationships
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

        # Auto-increment version_number within this graph
        num_q = """
        MATCH (v:GraphVersion {graph_id: $graph_id})
        RETURN coalesce(max(v.version_number), 0) + 1 AS next_num
        """
        num_result = await neo4j_client.execute_query(num_q, {"graph_id": graph_id})
        version_number = int(num_result[0]["next_num"]) if num_result else 1

        version_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        create_q = """
        MATCH (g:Graph {graph_id: $graph_id})
        CREATE (v:GraphVersion {
            version_id:        $version_id,
            graph_id:          $graph_id,
            version_number:    $version_number,
            label:             $label,
            description:       $description,
            captured_at:       datetime($captured_at),
            created_by:        $created_by,
            parent_version_id: $parent_version_id,
            is_auto:           $is_auto,
            entity_count:      $entity_count,
            relationship_count: $relationship_count,
            created_at:        datetime($created_at)
        })
        CREATE (g)-[:HAS_VERSION]->(v)
        RETURN v
        """
        result = await neo4j_client.execute_query(create_q, {
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
        })

        if not result:
            # Graph node may not exist — fallback without MATCH on Graph
            create_no_graph_q = """
            CREATE (v:GraphVersion {
                version_id:        $version_id,
                graph_id:          $graph_id,
                version_number:    $version_number,
                label:             $label,
                description:       $description,
                captured_at:       datetime($captured_at),
                created_by:        $created_by,
                parent_version_id: $parent_version_id,
                is_auto:           $is_auto,
                entity_count:      $entity_count,
                relationship_count: $relationship_count,
                created_at:        datetime($created_at)
            })
            RETURN v
            """
            result = await neo4j_client.execute_query(create_no_graph_q, {
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
            })

        logger.info(f"Created version {version_id} (v{version_number}) for graph {graph_id}")
        return self._node_to_dict(result[0]["v"])

    # ------------------------------------------------------------------
    # List versions
    # ------------------------------------------------------------------

    async def list_versions(self, graph_id: str) -> List[Dict[str, Any]]:
        """Return all versions for a graph, newest first."""
        q = """
        MATCH (v:GraphVersion {graph_id: $graph_id})
        RETURN v
        ORDER BY v.captured_at DESC
        """
        results = await neo4j_client.execute_query(q, {"graph_id": graph_id})
        return [self._node_to_dict(r["v"]) for r in results]

    # ------------------------------------------------------------------
    # Get single version
    # ------------------------------------------------------------------

    async def get_version(self, graph_id: str, version_id: str) -> Optional[Dict[str, Any]]:
        q = """
        MATCH (v:GraphVersion {graph_id: $graph_id, version_id: $version_id})
        RETURN v
        """
        results = await neo4j_client.execute_query(q, {"graph_id": graph_id, "version_id": version_id})
        return self._node_to_dict(results[0]["v"]) if results else None

    # ------------------------------------------------------------------
    # Diff two versions
    # ------------------------------------------------------------------

    async def diff_versions(
        self,
        graph_id: str,
        v1_id: str,
        v2_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Compute added/deleted entities and relationships between v1 and v2.
        v1 must be older than v2.
        """
        v1 = await self.get_version(graph_id, v1_id)
        v2 = await self.get_version(graph_id, v2_id)

        if not v1 or not v2:
            raise ValueError("One or both versions not found")

        # Normalise captured_at to comparable strings
        t1 = _version_ts(v1)
        t2 = _version_ts(v2)

        # Ensure v1 is older
        if t1 > t2:
            v1, v2, t1, t2 = v2, v1, t2, t1

        params = {
            "graph_id": graph_id,
            "v1_ts": t1,
            "v2_ts": t2,
            "offset": offset,
            "limit": limit,
        }

        # Entities added in window
        ea_q = """
        MATCH (e:__Entity__ {graph_id: $graph_id})
        WHERE e.transaction_time > datetime($v1_ts)
          AND e.transaction_time <= datetime($v2_ts)
          AND (e.invalidated_at IS NULL OR e.invalidated_at > datetime($v2_ts))
        RETURN count(e) AS cnt
        """
        # Entities deleted in window
        ed_q = """
        MATCH (e:__Entity__ {graph_id: $graph_id})
        WHERE e.invalidated_at > datetime($v1_ts)
          AND e.invalidated_at <= datetime($v2_ts)
        RETURN count(e) AS cnt
        """
        # Relationships added in window
        ra_q = """
        MATCH (a:__Entity__ {graph_id: $graph_id})-[r {graph_id: $graph_id}]->(b:__Entity__ {graph_id: $graph_id})
        WHERE r.transaction_time > datetime($v1_ts)
          AND r.transaction_time <= datetime($v2_ts)
          AND (r.invalidated_at IS NULL OR r.invalidated_at > datetime($v2_ts))
        RETURN count(r) AS cnt
        """
        # Relationships deleted in window
        rd_q = """
        MATCH (a:__Entity__ {graph_id: $graph_id})-[r {graph_id: $graph_id}]->(b:__Entity__ {graph_id: $graph_id})
        WHERE r.invalidated_at > datetime($v1_ts)
          AND r.invalidated_at <= datetime($v2_ts)
        RETURN count(r) AS cnt
        """

        count_params = {"graph_id": graph_id, "v1_ts": t1, "v2_ts": t2}
        ea_cnt = int((await neo4j_client.execute_query(ea_q, count_params) or [{"cnt": 0}])[0]["cnt"])
        ed_cnt = int((await neo4j_client.execute_query(ed_q, count_params) or [{"cnt": 0}])[0]["cnt"])
        ra_cnt = int((await neo4j_client.execute_query(ra_q, count_params) or [{"cnt": 0}])[0]["cnt"])
        rd_cnt = int((await neo4j_client.execute_query(rd_q, count_params) or [{"cnt": 0}])[0]["cnt"])

        # Paginated change list — added entities first, then deleted, then rels
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
        change_rows = await neo4j_client.execute_query(changes_q, params)
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
            "from_version": {"version_id": v1["version_id"], "label": v1.get("label"), "captured_at": v1.get("captured_at")},
            "to_version": {"version_id": v2["version_id"], "label": v2.get("label"), "captured_at": v2.get("captured_at")},
            "summary": {
                "entities_added": ea_cnt,
                "entities_deleted": ed_cnt,
                "relationships_added": ra_cnt,
                "relationships_deleted": rd_cnt,
                "property_changes": 0,  # Phase 3 scope: Option A — not tracked at value level
            },
            "changes": changes,
            "offset": offset,
            "limit": limit,
            "has_more": (offset + limit) < total_changes,
        }

    # ------------------------------------------------------------------
    # Delete snapshot
    # ------------------------------------------------------------------

    async def delete_version(self, graph_id: str, version_id: str) -> bool:
        """Delete a GraphVersion node. Returns True if deleted, False if not found."""
        existing = await self.get_version(graph_id, version_id)
        if not existing:
            return False
        q = """
        MATCH (v:GraphVersion {graph_id: $graph_id, version_id: $version_id})
        DETACH DELETE v
        """
        await neo4j_client.execute_query(q, {"graph_id": graph_id, "version_id": version_id})
        logger.info(f"Deleted version {version_id} for graph {graph_id}")
        return True

    # ------------------------------------------------------------------
    # Rollback job management (PostgreSQL — async large-graph tracking)
    # ------------------------------------------------------------------

    async def create_rollback_job(
        self,
        db: AsyncSession,
        graph_id: str,
        version_id: str,
        mode: str,
        performed_by: str,
        scope: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a pending GraphRollbackJob record in PostgreSQL."""
        from app.models.graph import GraphRollbackJob
        import uuid as _uuid

        job = GraphRollbackJob(
            id=_uuid.uuid4(),
            graph_id=graph_id,
            version_id=version_id,
            mode=mode,
            status="pending",
            performed_by=performed_by,
            scope=scope,
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)
        return self._job_to_dict(job)

    async def get_rollback_job(
        self, db: AsyncSession, graph_id: str, job_id: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch a rollback job by id, scoped to graph_id for multi-tenancy."""
        from app.models.graph import GraphRollbackJob
        import uuid as _uuid

        try:
            gid = _uuid.UUID(graph_id)
            jid = _uuid.UUID(job_id)
        except ValueError:
            return None

        result = await db.execute(
            select(GraphRollbackJob).where(
                GraphRollbackJob.id == jid,
                GraphRollbackJob.graph_id == gid,
            )
        )
        job = result.scalar_one_or_none()
        return self._job_to_dict(job) if job else None

    @staticmethod
    def _job_to_dict(job: Any) -> Dict[str, Any]:
        return {
            "rollback_job_id": str(job.id),
            "graph_id": str(job.graph_id),
            "version_id": job.version_id,
            "mode": job.mode,
            "status": job.status,
            "progress": job.progress,
            "entities_restored": job.entities_restored,
            "entities_soft_deleted": job.entities_soft_deleted,
            "relationships_restored": job.relationships_restored,
            "relationships_soft_deleted": job.relationships_soft_deleted,
            "checkpoint_version_id": job.checkpoint_version_id,
            "error_message": job.error_message,
            "performed_by": job.performed_by,
            "scope": job.scope,
            "celery_task_id": job.celery_task_id,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "created_at": job.created_at.isoformat() if job.created_at else None,
        }

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    async def rollback(
        self,
        graph_id: str,
        version_id: str,
        mode: str,
        performed_by: str,
        create_checkpoint: bool = True,
        scope: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Roll back a graph to the state at version_id.

        Full rollback:
        1. Auto-checkpoint (unless disabled)
        2. Soft-delete entities added after captured_at
        3. Restore entities deleted after captured_at
        4. Same for relationships

        Partial rollback: restricts to entities traceable to scope.document_ids.
        """
        version = await self.get_version(graph_id, version_id)
        if not version:
            raise ValueError(f"Version {version_id} not found")

        captured_at = _version_ts(version)
        now_iso = datetime.now(timezone.utc).isoformat()

        checkpoint_vid: Optional[str] = None
        if create_checkpoint:
            chk = await self.create_version(
                graph_id=graph_id,
                label=f"pre-rollback-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
                description=f"Auto-checkpoint before rollback to version {version_id}",
                created_by=performed_by,
                is_auto=True,
                parent_version_id=version_id,
            )
            checkpoint_vid = chk["version_id"]

        if mode == "partial" and scope and scope.get("document_ids"):
            return await self._partial_rollback(
                graph_id, captured_at, scope["document_ids"], performed_by, now_iso, checkpoint_vid
            )

        return await self._full_rollback(graph_id, captured_at, performed_by, now_iso, checkpoint_vid)

    async def _full_rollback(
        self,
        graph_id: str,
        captured_at: str,
        performed_by: str,
        now_iso: str,
        checkpoint_vid: Optional[str],
    ) -> Dict[str, Any]:
        params = {"graph_id": graph_id, "captured_at": captured_at, "performed_by": performed_by, "now": now_iso}

        # Step 2: soft-delete entities added after captured_at
        del_new_entities_q = """
        MATCH (e:__Entity__ {graph_id: $graph_id})
        WHERE e.transaction_time > datetime($captured_at) AND e.invalidated_at IS NULL
        SET e.invalidated_at = datetime($now), e.deleted_by = $performed_by
        RETURN count(e) AS cnt
        """
        r = await neo4j_client.execute_query(del_new_entities_q, params)
        entities_soft_deleted = int((r or [{"cnt": 0}])[0]["cnt"])

        # Step 3: restore entities deleted after captured_at
        restore_entities_q = """
        MATCH (e:__Entity__ {graph_id: $graph_id})
        WHERE e.invalidated_at > datetime($captured_at)
        REMOVE e.invalidated_at, e.deleted_by
        RETURN count(e) AS cnt
        """
        r = await neo4j_client.execute_query(restore_entities_q, params)
        entities_restored = int((r or [{"cnt": 0}])[0]["cnt"])

        # Step 4: soft-delete relationships added after captured_at
        del_new_rels_q = """
        MATCH (a:__Entity__ {graph_id: $graph_id})-[r {graph_id: $graph_id}]->(b:__Entity__ {graph_id: $graph_id})
        WHERE r.transaction_time > datetime($captured_at) AND r.invalidated_at IS NULL
        SET r.invalidated_at = datetime($now), r.deleted_by = $performed_by
        RETURN count(r) AS cnt
        """
        r = await neo4j_client.execute_query(del_new_rels_q, params)
        rels_soft_deleted = int((r or [{"cnt": 0}])[0]["cnt"])

        # Step 5: restore relationships deleted after captured_at
        restore_rels_q = """
        MATCH (a:__Entity__ {graph_id: $graph_id})-[r {graph_id: $graph_id}]->(b:__Entity__ {graph_id: $graph_id})
        WHERE r.invalidated_at > datetime($captured_at)
        REMOVE r.invalidated_at, r.deleted_by
        RETURN count(r) AS cnt
        """
        r = await neo4j_client.execute_query(restore_rels_q, params)
        rels_restored = int((r or [{"cnt": 0}])[0]["cnt"])

        # Integrity check: cascade-delete rels whose endpoints are now soft-deleted
        integrity_q = """
        MATCH (a:__Entity__ {graph_id: $graph_id})-[r {graph_id: $graph_id}]->(b:__Entity__ {graph_id: $graph_id})
        WHERE (a.invalidated_at IS NOT NULL OR b.invalidated_at IS NOT NULL) AND r.invalidated_at IS NULL
        SET r.invalidated_at = datetime($now), r.deleted_by = $performed_by
        RETURN count(r) AS cnt
        """
        await neo4j_client.execute_query(integrity_q, params)

        logger.info(
            f"Full rollback to {captured_at} for graph {graph_id}: "
            f"entities soft-deleted={entities_soft_deleted}, restored={entities_restored}"
        )
        return {
            "checkpoint_version_id": checkpoint_vid,
            "entities_restored": entities_restored,
            "entities_soft_deleted": entities_soft_deleted,
            "relationships_restored": rels_restored,
            "relationships_soft_deleted": rels_soft_deleted,
            "message": "Full rollback completed",
        }

    async def _partial_rollback(
        self,
        graph_id: str,
        captured_at: str,
        document_ids: List[str],
        performed_by: str,
        now_iso: str,
        checkpoint_vid: Optional[str],
    ) -> Dict[str, Any]:
        """Rollback only entities traceable to specified documents."""
        params = {
            "graph_id": graph_id,
            "captured_at": captured_at,
            "performed_by": performed_by,
            "now": now_iso,
            "document_ids": document_ids,
        }

        # Soft-delete entities from these documents added after captured_at
        del_q = """
        MATCH (e:__Entity__ {graph_id: $graph_id})<-[:MENTIONS]-(:Chunk {graph_id: $graph_id})-[:FROM_DOCUMENT]->(d {graph_id: $graph_id})
        WHERE d.document_id IN $document_ids
          AND e.transaction_time > datetime($captured_at)
          AND e.invalidated_at IS NULL
        SET e.invalidated_at = datetime($now), e.deleted_by = $performed_by
        RETURN count(e) AS cnt
        """
        r = await neo4j_client.execute_query(del_q, params)
        entities_soft_deleted = int((r or [{"cnt": 0}])[0]["cnt"])

        logger.info(f"Partial rollback for graph {graph_id} docs {document_ids}: deleted {entities_soft_deleted}")
        return {
            "checkpoint_version_id": checkpoint_vid,
            "entities_restored": 0,
            "entities_soft_deleted": entities_soft_deleted,
            "relationships_restored": 0,
            "relationships_soft_deleted": 0,
            "message": f"Partial rollback completed for {len(document_ids)} document(s)",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _node_to_dict(node: Any) -> Dict[str, Any]:
        """Convert Neo4j node to plain dict, serialising datetime objects."""
        d: Dict[str, Any] = {}
        for key, val in node.items():
            if hasattr(val, "iso_format"):
                d[key] = val.iso_format()
            else:
                d[key] = val
        return d


def _version_ts(version: Dict[str, Any]) -> str:
    """Return captured_at as an ISO-8601 string for Cypher datetime() parameter."""
    ts = version.get("captured_at")
    if ts is None:
        raise ValueError("Version missing captured_at")
    if isinstance(ts, str):
        return ts
    if hasattr(ts, "iso_format"):
        return ts.iso_format()
    return str(ts)


# Module-level singleton
versioning_service = VersioningService()
