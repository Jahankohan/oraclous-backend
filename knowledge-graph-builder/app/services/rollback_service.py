"""
Rollback Service

Handles whole-graph rollback via soft-invalidation (invalidated_at) semantics.

CTO design decisions (ORA-55):
- Rollback scope: whole-graph only. No subgraph rollback in Phase 3.
- Property naming: invalidated_at (not system_end_time, not deleted_at).
- Async path for graphs > 10K entities via Celery; sync path for smaller graphs.

Architecture rules:
- Rule #4: All Cypher queries filter by graph_id
- Rule #5: FastAPI uses AsyncDriver (this service); Celery sync driver in background_jobs.py
- Rule #6: Rollback job tracking in PostgreSQL (GraphRollbackJob table)
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client
from app.services.snapshot_service import snapshot_service

logger = get_logger(__name__)


class RollbackService:
    """
    Full-graph rollback operations.

    Sync path (≤ 10K entities): runs inline, returns result immediately.
    Async path (> 10K entities): creates a GraphRollbackJob record, dispatches
    Celery task, caller polls GET /graphs/{graphId}/rollbacks/{rollbackId}.
    """

    # ------------------------------------------------------------------
    # Rollback job management (PostgreSQL)
    # ------------------------------------------------------------------

    async def create_rollback_job(
        self,
        db: AsyncSession,
        graph_id: str,
        version_id: str,
        performed_by: str,
    ) -> dict[str, Any]:
        """Create a pending GraphRollbackJob record in PostgreSQL."""
        from app.models.graph import GraphRollbackJob

        job = GraphRollbackJob(
            id=uuid.uuid4(),
            graph_id=graph_id,
            version_id=version_id,
            mode="full",
            status="pending",
            performed_by=performed_by,
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)
        return self._job_to_dict(job)

    async def get_rollback_job(
        self, db: AsyncSession, graph_id: str, job_id: str
    ) -> dict[str, Any] | None:
        """Fetch a rollback job by id, scoped to graph_id for multi-tenancy."""
        from app.models.graph import GraphRollbackJob

        try:
            gid = uuid.UUID(graph_id)
            jid = uuid.UUID(job_id)
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

    # ------------------------------------------------------------------
    # Rollback execution (sync, for ≤ 10K entities)
    # ------------------------------------------------------------------

    async def rollback(
        self,
        graph_id: str,
        version_id: str,
        performed_by: str,
        create_checkpoint: bool = True,
    ) -> dict[str, Any]:
        """
        Roll back a graph to the state captured at version_id.

        Steps:
        1. Auto-checkpoint current state (unless disabled)
        2. Soft-delete entities added after captured_at
        3. Restore entities deleted (invalidated) after captured_at
        4. Same for relationships
        5. Integrity pass: cascade-invalidate rels with soft-deleted endpoints
        """
        version = await snapshot_service.get_snapshot(graph_id, version_id)
        if not version:
            raise ValueError(f"Snapshot {version_id} not found")

        from app.services.snapshot_service import _snapshot_ts

        captured_at = _snapshot_ts(version)
        now_iso = datetime.now(UTC).isoformat()

        checkpoint_vid: str | None = None
        if create_checkpoint:
            chk = await snapshot_service.create_snapshot(
                graph_id=graph_id,
                label=f"pre-rollback-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}",
                description=f"Auto-checkpoint before rollback to snapshot {version_id}",
                created_by=performed_by,
                is_auto=True,
                parent_version_id=version_id,
            )
            checkpoint_vid = chk["version_id"]

        return await self._full_rollback(
            graph_id, captured_at, performed_by, now_iso, checkpoint_vid
        )

    async def _full_rollback(
        self,
        graph_id: str,
        captured_at: str,
        performed_by: str,
        now_iso: str,
        checkpoint_vid: str | None,
    ) -> dict[str, Any]:
        params = {
            "graph_id": graph_id,
            "captured_at": captured_at,
            "performed_by": performed_by,
            "now": now_iso,
        }

        # Soft-delete entities added after captured_at
        r = await neo4j_client.execute_query(
            """
            MATCH (e:__Entity__ {graph_id: $graph_id})
            WHERE e.transaction_time > datetime($captured_at) AND e.invalidated_at IS NULL
            SET e.invalidated_at = datetime($now), e.deleted_by = $performed_by
            RETURN count(e) AS cnt
            """,
            params,
        )
        entities_soft_deleted = int((r or [{"cnt": 0}])[0]["cnt"])

        # Restore entities invalidated after captured_at
        r = await neo4j_client.execute_query(
            """
            MATCH (e:__Entity__ {graph_id: $graph_id})
            WHERE e.invalidated_at > datetime($captured_at)
            REMOVE e.invalidated_at, e.deleted_by
            RETURN count(e) AS cnt
            """,
            params,
        )
        entities_restored = int((r or [{"cnt": 0}])[0]["cnt"])

        # Soft-delete relationships added after captured_at
        r = await neo4j_client.execute_query(
            """
            MATCH (a:__Entity__ {graph_id: $graph_id})-[r {graph_id: $graph_id}]->(b:__Entity__ {graph_id: $graph_id})
            WHERE r.transaction_time > datetime($captured_at) AND r.invalidated_at IS NULL
            SET r.invalidated_at = datetime($now), r.deleted_by = $performed_by
            RETURN count(r) AS cnt
            """,
            params,
        )
        rels_soft_deleted = int((r or [{"cnt": 0}])[0]["cnt"])

        # Restore relationships invalidated after captured_at
        r = await neo4j_client.execute_query(
            """
            MATCH (a:__Entity__ {graph_id: $graph_id})-[r {graph_id: $graph_id}]->(b:__Entity__ {graph_id: $graph_id})
            WHERE r.invalidated_at > datetime($captured_at)
            REMOVE r.invalidated_at, r.deleted_by
            RETURN count(r) AS cnt
            """,
            params,
        )
        rels_restored = int((r or [{"cnt": 0}])[0]["cnt"])

        # Integrity pass: cascade-invalidate rels whose endpoints are now soft-deleted
        await neo4j_client.execute_query(
            """
            MATCH (a:__Entity__ {graph_id: $graph_id})-[r {graph_id: $graph_id}]->(b:__Entity__ {graph_id: $graph_id})
            WHERE (a.invalidated_at IS NOT NULL OR b.invalidated_at IS NOT NULL) AND r.invalidated_at IS NULL
            SET r.invalidated_at = datetime($now), r.deleted_by = $performed_by
            RETURN count(r) AS cnt
            """,
            params,
        )

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _job_to_dict(job: Any) -> dict[str, Any]:
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


# Module-level singleton
rollback_service = RollbackService()
