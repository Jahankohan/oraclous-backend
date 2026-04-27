"""
Refactored Background Jobs using Pipeline Service
Clean implementation with Neo4j GraphRAG pipeline and multi-tenant support
"""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from celery import Celery
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.core.logging import get_logger
from app.models.graph import IngestionJob  # Only IngestionJob, not KnowledgeGraph
from app.schemas.graph_schemas import IngestionOverrides, IngestMode, TemporalContext
from app.services.document_processor import document_processor
from app.services.graph_node_service import GraphNodeService
from app.services.pipeline_service import pipeline_service
from app.services.task_executor import AsyncTaskExecutor

logger = get_logger(__name__)

worker_engine = create_async_engine(
    settings.POSTGRES_URL,
    poolclass=NullPool,  # No connection pooling in workers
    echo=False,
    future=True,
)

worker_session_maker = async_sessionmaker(
    bind=worker_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ==================== WORKER NEO4J CONNECTION MANAGER ====================


class WorkerNeo4jManager:
    """
    Neo4j connection manager for Celery workers using task-scoped connections.

    Follows the PostgreSQL NullPool pattern to ensure each Celery task gets
    its own Neo4j connection, preventing connection pool conflicts between
    FastAPI and worker processes.

    Features:
    - Task-scoped connections (no connection pooling between tasks)
    - Automatic cleanup after task completion
    - Support for both sync (GraphRAG) and async operations
    - Isolation from FastAPI connection pools

    Usage:
        async with WorkerNeo4jManager() as neo4j:
            driver = neo4j.get_sync_driver()  # For GraphRAG components
            # Use driver for the task
            # Automatic cleanup when exiting context
    """

    def __init__(self):
        """Initialize worker Neo4j manager."""
        self.sync_driver = None
        self.async_driver = None
        self._logger = get_logger(f"{__name__}.WorkerNeo4jManager")

    async def __aenter__(self):
        """Async context manager entry - create fresh connections."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup connections."""
        await self.cleanup()

    def __enter__(self):
        """Sync context manager entry - create fresh connections."""
        self.connect_sync_only()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit - cleanup connections."""
        self.cleanup_sync()

    async def connect(self):
        """
        Create fresh Neo4j connections for this task.

        Creates both async and sync drivers with minimal connection pools
        to ensure task isolation following the NullPool pattern.
        """
        try:
            # Import here to avoid circular imports
            from neo4j import AsyncGraphDatabase, GraphDatabase

            # Create sync driver for GraphRAG components (1 connection max)
            if not self.sync_driver:
                self.sync_driver = GraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
                    max_connection_pool_size=1,  # Minimal pool like NullPool
                    connection_acquisition_timeout=30,
                )
                # Test sync connection
                self.sync_driver.verify_connectivity()
                self._logger.debug("Worker sync driver connected")

            # Create async driver for any async operations (1 connection max)
            if not self.async_driver:
                self.async_driver = AsyncGraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
                    max_connection_pool_size=1,  # Minimal pool like NullPool
                    connection_acquisition_timeout=30,
                )
                # Test async connection
                await self.async_driver.verify_connectivity()
                self._logger.debug("Worker async driver connected")

        except Exception as e:
            self._logger.error(f"Failed to create worker Neo4j connections: {e}")
            await self.cleanup()
            raise

    def connect_sync_only(self):
        """
        Create only sync Neo4j connection for sync-only tasks.

        Optimized for tasks that only need GraphRAG components.
        """
        try:
            from neo4j import GraphDatabase

            if not self.sync_driver:
                self.sync_driver = GraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
                    max_connection_pool_size=1,  # Minimal pool like NullPool
                    connection_acquisition_timeout=30,
                )
                # Test connection
                self.sync_driver.verify_connectivity()
                self._logger.debug("Worker sync-only driver connected")

        except Exception as e:
            self._logger.error(f"Failed to create worker sync Neo4j connection: {e}")
            self.cleanup_sync()
            raise

    def get_sync_driver(self):
        """
        Get sync driver for GraphRAG components.

        Returns:
            Neo4j sync driver instance

        Raises:
            RuntimeError: If sync driver is not available
        """
        if not self.sync_driver:
            raise RuntimeError(
                "Sync driver not available. Use context manager to connect."
            )
        return self.sync_driver

    def get_async_driver(self):
        """
        Get async driver for async operations.

        Returns:
            Neo4j async driver instance

        Raises:
            RuntimeError: If async driver is not available
        """
        if not self.async_driver:
            raise RuntimeError(
                "Async driver not available. Use async context manager to connect."
            )
        return self.async_driver

    async def cleanup(self):
        """Clean up both sync and async connections."""
        if self.async_driver:
            try:
                await self.async_driver.close()
                self._logger.debug("Worker async driver closed")
            except Exception as e:
                self._logger.warning(f"Error closing worker async driver: {e}")
            finally:
                self.async_driver = None

        self.cleanup_sync()

    def cleanup_sync(self):
        """Clean up only sync connections."""
        if self.sync_driver:
            try:
                self.sync_driver.close()
                self._logger.debug("Worker sync driver closed")
            except Exception as e:
                self._logger.warning(f"Error closing worker sync driver: {e}")
            finally:
                self.sync_driver = None


# Configure Celery
celery_app = Celery(
    "knowledge_graph_builder",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    beat_schedule={
        # Hard-delete stale chunks whose TTL has expired (default: 7 days after soft-delete).
        # Runs once daily at 03:00 UTC to avoid peak hours.
        "cleanup-stale-chunks-daily": {
            "task": "app.services.background_jobs.cleanup_stale_chunks",
            "schedule": 86400,  # every 24 hours in seconds
            "args": [7],  # stale_ttl_days
        },
        # Consolidate near-duplicate memories across all graphs nightly at 02:00 UTC.
        "consolidate-memories-nightly": {
            "task": "app.services.background_jobs.consolidate_all_memories",
            "schedule": 86400,
            "args": [],
        },
        # Connector framework: poll for due scheduled syncs every 60 seconds.
        "poll-due-connectors": {
            "task": "connectors.poll_due_connectors",
            "schedule": 60.0,
        },
        # Connector framework: retry failed webhook events every 5 minutes.
        "retry-failed-webhook-events": {
            "task": "connectors.retry_failed_events",
            "schedule": 300.0,
        },
    },
)

# Attach OpenTelemetry Celery instrumentation (no-op when OTEL_ENABLED=false)
try:
    from app.core.telemetry import instrument_celery

    instrument_celery()
except Exception as _otel_exc:
    logger.warning(f"Could not attach Celery OTel instrumentation: {_otel_exc}")

# ==================== MAIN CELERY TASKS ====================


@celery_app.task(bind=True)
def cleanup_stale_chunks(self, stale_ttl_days: int = 7) -> dict[str, Any]:
    """
    Celery beat task: hard-delete Chunk nodes whose staleAt has exceeded the TTL.

    Follows the dual-driver rule: sync Neo4j driver with NullPool for Celery workers.
    Orphan-protection: only deletes entity nodes whose ALL chunks are stale.

    Default TTL: 7 days. Scheduled to run daily (see beat_schedule).
    """
    from neo4j import GraphDatabase

    logger.info(f"Starting stale chunk cleanup (TTL={stale_ttl_days} days)")
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
        max_connection_pool_size=1,
    )
    chunks_deleted = 0
    entities_deleted = 0
    try:
        with driver.session() as session:
            # 1. Hard-delete chunks past TTL (DETACH DELETE removes their relationships too)
            result = session.run(
                """
                MATCH (c:Chunk)
                WHERE c.staleAt IS NOT NULL
                  AND c.staleAt < datetime() - duration({days: $ttl_days})
                CALL { WITH c DETACH DELETE c } IN TRANSACTIONS OF 500 ROWS
                RETURN count(c) AS deleted
                """,
                {"ttl_days": stale_ttl_days},
            )
            record = result.single()
            chunks_deleted = int(record["deleted"]) if record else 0

            # 2. Delete orphan entity nodes whose ALL chunks are now stale/deleted
            result = session.run(
                """
                MATCH (e:__Entity__)
                WHERE NOT EXISTS {
                    MATCH (e)<-[:HAS_ENTITY]-(c:Chunk)
                    WHERE c.staleAt IS NULL
                }
                AND EXISTS { MATCH (e)<-[:HAS_ENTITY]-() }
                CALL { WITH e DETACH DELETE e } IN TRANSACTIONS OF 500 ROWS
                RETURN count(e) AS deleted
                """,
            )
            record = result.single()
            entities_deleted = int(record["deleted"]) if record else 0

        logger.info(
            f"Stale chunk cleanup complete: {chunks_deleted} chunks, {entities_deleted} orphan entities deleted"
        )
        return {
            "status": "done",
            "chunks_deleted": chunks_deleted,
            "entities_deleted": entities_deleted,
        }

    except Exception as exc:
        logger.error(f"Stale chunk cleanup failed: {exc}")
        raise
    finally:
        driver.close()


@celery_app.task(bind=True)
def process_ingestion_job(self, job_id: str, user_id: str):
    """
    Process ingestion job using clean pipeline service.
    Follows your established pattern: Celery task -> AsyncTaskExecutor -> async implementation
    """
    return AsyncTaskExecutor.run_async_task(
        _process_pipeline_ingestion_async, self, job_id, user_id
    )


async def _process_pipeline_ingestion_async(
    task, job_id: str, user_id: str
) -> dict[str, Any]:
    """
    CLEAN REFACTOR: End-to-end document processing using pipeline_service.

    FLOW: Document -> Pipeline Service -> Neo4j Storage -> Job Completion
    - Uses your new pipeline_service.py (Neo4j GraphRAG foundation)
    - Maintains your excellent progress tracking patterns
    - Keeps database session management
    - Delivers end-to-end results (no complex features for now)
    """
    job = None

    try:
        logger.info(f"Starting clean pipeline ingestion for job {job_id}")

        # STEP 1: Get job from database and verify graph exists in Neo4j
        async with worker_session_maker() as session:
            job_query = select(IngestionJob).where(IngestionJob.id == UUID(job_id))
            result = await session.execute(job_query)
            job = result.scalar_one_or_none()

            if not job:
                logger.error(f"Job {job_id} not found")
                return {"error": f"Job {job_id} not found", "status": "failed"}

            # Verify graph exists in Neo4j using GraphNodeService
            # Use sync operations since this is a worker context
            async with WorkerNeo4jManager() as neo4j:
                try:
                    graph_service = GraphNodeService(neo4j.get_sync_driver())
                    graph = graph_service.get_graph(str(job.graph_id))

                    if not graph:
                        logger.error(
                            f"Graph {job.graph_id} not found in Neo4j for job {job_id}"
                        )
                        return {"status": "error", "message": "Graph not found"}

                    # Verify user ownership
                    if graph["user_id"] != user_id:
                        logger.error(
                            f"User {user_id} does not own graph {job.graph_id}"
                        )
                        return {"status": "error", "message": "Access denied"}

                    logger.info(
                        f"Verified graph {job.graph_id} exists in Neo4j for user {user_id}"
                    )

                except Exception as e:
                    logger.error(f"Failed to verify graph {job.graph_id}: {e}")
                    return {
                        "status": "error",
                        "message": f"Graph verification failed: {str(e)}",
                    }

            # Update job status to processing
            await _update_job_status_async(session, job_id, "processing")

        logger.info(f"Processing job {job_id} for graph {job.graph_id}")

        # STEP 2: Progress tracking (keep your excellent pattern)
        if task:
            task.update_state(
                state="PROGRESS",
                meta={
                    "progress": 10,
                    "status": "Starting Neo4j GraphRAG pipeline processing",
                },
            )

        # STEP 3: Process document content based on source type
        try:
            processed_doc = document_processor.process_document(
                content=job.source_content,
                source_type=job.source_type or "text",
                metadata={
                    "job_id": job_id,
                    "graph_id": str(job.graph_id),
                    "user_id": user_id,
                },
            )

            # Enhanced document metadata from processor
            document_text = processed_doc["text"]
            base_metadata = processed_doc["metadata"]

        except Exception as e:
            logger.error(f"Document processing failed for job {job_id}: {e}")
            async with worker_session_maker() as session:
                await _update_job_status_async(
                    session,
                    job_id,
                    "failed",
                    error=f"Document processing failed: {str(e)}",
                )
            return {
                "status": "failed",
                "job_id": job_id,
                "error": f"Document processing failed: {str(e)}",
            }

        # STEP 4: Convert processed document to GraphRAG format
        documents = [
            {
                "text": document_text,
                "source": f"job_{job_id}",
                "title": f"Document from job {job_id}",
                "id": job_id,  # Add explicit document ID
                "metadata": {
                    **base_metadata,  # Include processed metadata
                    "job_id": job_id,
                    "graph_id": str(job.graph_id),
                    "user_id": user_id,
                    "source_type": job.source_type or "text",
                    "created_at": datetime.now(UTC).isoformat(),
                    "document_title": f"Document from job {job_id}",
                    "document_source": f"job_{job_id}",
                },
            }
        ]

        if task:
            task.update_state(
                state="PROGRESS",
                meta={
                    "progress": 30,
                    "status": (
                        f"Processing {len(documents)} document(s) through pipeline"
                    ),
                },
            )

        # STEP 4: Process through clean pipeline service (END-TO-END)
        logger.info(
            f"Processing documents through pipeline service for graph {job.graph_id}"
        )

        # Reconstruct IngestionOverrides, TemporalContext, and IngestMode from stored effective_instructions
        overrides: IngestionOverrides | None = None
        temporal_context: TemporalContext | None = None
        ingest_mode: IngestMode = IngestMode.INCREMENTAL
        if job.effective_instructions and isinstance(job.effective_instructions, dict):
            raw_overrides = job.effective_instructions.get("overrides")
            if raw_overrides:
                try:
                    overrides = IngestionOverrides(**raw_overrides)
                except Exception as e:
                    logger.warning(
                        f"Could not reconstruct IngestionOverrides for job {job_id}: {e}"
                    )
            raw_temporal = job.effective_instructions.get("temporal_context")
            if raw_temporal:
                try:
                    temporal_context = TemporalContext(**raw_temporal)
                except Exception as e:
                    logger.warning(
                        f"Could not reconstruct TemporalContext for job {job_id}: {e}"
                    )
            raw_mode = job.effective_instructions.get("ingest_mode")
            if raw_mode:
                try:
                    ingest_mode = IngestMode(raw_mode)
                except ValueError:
                    logger.warning(
                        f"Unknown ingest_mode '{raw_mode}' for job {job_id}, defaulting to incremental"
                    )

        pipeline_result = await pipeline_service.process_documents(
            documents=documents,
            graph_id=job.graph_id,
            user_id=user_id,
            overrides=overrides,
            temporal_context=temporal_context,
            mode=ingest_mode,
            job_id=job_id,
        )

        logger.info(f"Pipeline processing result: {pipeline_result}")

        # STEP 5: Extract results
        if pipeline_result["status"] == "completed":
            entities_created = pipeline_result.get("entities_created", 0)
            relationships_created = pipeline_result.get("relationships_created", 0)
            chunks_created = pipeline_result.get("chunks_created", 0)

            if task:
                task.update_state(
                    state="PROGRESS",
                    meta={
                        "progress": 80,
                        "status": (
                            f"Pipeline completed: {entities_created} entities, {relationships_created} relationships, {chunks_created} chunks"
                        ),
                    },
                )
        elif pipeline_result["status"] == "processing":
            # Handle background processing case
            if task:
                task.update_state(
                    state="PROGRESS",
                    meta={
                        "progress": 50,
                        "status": (
                            "Documents processing in background - monitoring progress"
                        ),
                    },
                )

            # For now, we'll consider this a success but note it's async
            entities_created = pipeline_result.get("documents_queued", 0)
            relationships_created = 0
            chunks_created = 0
        else:
            # Failed processing
            error_msg = pipeline_result.get("error", "Pipeline processing failed")
            logger.error(f"Pipeline processing failed for job {job_id}: {error_msg}")

            # Update job status to failed
            async with worker_session_maker() as session:
                await _update_job_status_async(
                    session, job_id, "failed", error=error_msg
                )

            return {"status": "failed", "job_id": job_id, "error": error_msg}

        # STEP 6: Complete job (keep your existing pattern)
        if task:
            task.update_state(
                state="PROGRESS",
                meta={
                    "progress": 100,
                    "status": "Completing job and updating database",
                },
            )

        async with worker_session_maker() as session:
            await _update_job_status_async(
                session,
                job_id,
                "completed",
                entities=entities_created,
                relationships=relationships_created,
                chunks=chunks_created,
            )

        logger.info(
            f"✅ Completed pipeline ingestion job {job_id}: "
            f"{entities_created} entities, {relationships_created} relationships, {chunks_created} chunks"
        )

        # STEP 7: Post-ingestion community detection trigger
        await _maybe_trigger_community_detection(str(job.graph_id))

        # STEP 8: Auto-snapshot (if enabled and 24h cap not exceeded)
        await _maybe_auto_snapshot(str(job.graph_id))

        # STEP 9: Invalidate query cache for this graph so stale results
        # are not served after new documents are ingested.
        await _invalidate_query_cache(str(job.graph_id))

        return {
            "status": "completed",
            "job_id": job_id,
            "graph_id": str(job.graph_id),
            "entities_created": entities_created,
            "relationships_created": relationships_created,
            "chunks_created": chunks_created,
            "pipeline_version": "neo4j_graphrag_clean",
        }

    except Exception as e:
        logger.error(f"Pipeline ingestion job {job_id} failed: {e}")

        # Update job status to failed
        try:
            async with worker_session_maker() as session:
                await _update_job_status_async(session, job_id, "failed", error=str(e))
        except Exception as status_error:
            logger.error(f"Failed to update job status: {status_error}")

        return {"status": "failed", "job_id": job_id, "error": str(e)}


async def _update_job_status_async(
    session: AsyncSession,
    job_id: str,
    status: str,
    entities: int = None,
    relationships: int = None,
    chunks: int = None,
    error: str = None,
) -> None:
    """
    Update job status in database (keep your existing pattern).

    Args:
        session: Database session
        job_id: Job identifier
        status: New status ('processing', 'completed', 'failed')
        entities: Number of entities created (optional)
        relationships: Number of relationships created (optional)
        chunks: Number of chunks created (optional)
        error: Error message if failed (optional)
    """
    try:
        # Prepare update data
        update_data = {"status": status}

        # Add metrics if provided
        if entities is not None:
            update_data["extracted_entities"] = entities
        if relationships is not None:
            update_data["extracted_relationships"] = relationships
        if chunks is not None:
            update_data["processed_chunks"] = chunks
        if error is not None:
            update_data["error_message"] = error

        # Update job in database
        stmt = (
            update(IngestionJob)
            .where(IngestionJob.id == UUID(job_id))
            .values(**update_data)
        )

        await session.execute(stmt)
        await session.commit()

        logger.debug(f"Updated job {job_id} status to {status}")

    except Exception as e:
        logger.error(f"Failed to update job {job_id} status: {e}")
        await session.rollback()
        raise


# ==================== POST-INGESTION COMMUNITY TRIGGER ====================


async def _maybe_trigger_community_detection(graph_id: str) -> None:
    """
    Queue community (re-)detection after ingestion if thresholds are met.

    Logic:
    - Count current entity nodes for the graph
    - If count >= COMMUNITY_DETECTION_MIN_ENTITIES AND communities are not
      currently rebuilding → queue detection with a 30s delay to batch
      rapid ingestions.
    - If entity_delta / entity_count_at_detection > 0.10 AND status == 'active'
      → mark communities stale and queue re-detection.
    """
    try:
        from neo4j import GraphDatabase
        from sqlalchemy import create_engine, text
        from sqlalchemy.pool import NullPool

        from app.tasks.community_tasks import (
            COMMUNITY_DETECTION_MIN_ENTITIES,
            _get_communities_status_pg,
            _update_communities_status_pg,
            detect_communities_task,
        )

        # Count current entities via NullPool sync engine
        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
            max_connection_pool_size=1,
        )
        try:
            with driver.session() as session:
                result = session.run(
                    "MATCH (e:__Entity__ {graph_id: $gid}) RETURN count(e) AS cnt",
                    {"gid": graph_id},
                )
                entity_count = result.single()["cnt"]
        finally:
            driver.close()

        if entity_count < COMMUNITY_DETECTION_MIN_ENTITIES:
            logger.debug(
                f"Skipping community detection for {graph_id}: "
                f"{entity_count} < {COMMUNITY_DETECTION_MIN_ENTITIES} entities"
            )
            return

        pg_engine = create_engine(
            settings.POSTGRES_URL.replace("+asyncpg", ""),
            poolclass=NullPool,
        )
        try:
            current_status = _get_communities_status_pg(pg_engine, graph_id)
            if current_status == "rebuilding":
                logger.debug(
                    f"Community detection already running for {graph_id}, skipping trigger"
                )
                return

            # Check staleness: if entity delta > 10% mark stale
            if current_status == "active":
                with pg_engine.connect() as conn:
                    row = conn.execute(
                        text(
                            "SELECT entity_count_at_detection, entity_delta_since_detection "
                            "FROM knowledge_graphs WHERE id = :gid"
                        ),
                        {"gid": graph_id},
                    ).fetchone()
                    if row and row[0] and row[0] > 0:
                        new_delta = (row[1] or 0) + 1  # bump delta by new entities
                        staleness = abs(new_delta) / max(row[0], 1)
                        if staleness > 0.10:
                            _update_communities_status_pg(pg_engine, graph_id, "stale")
                            logger.info(
                                f"Graph {graph_id} communities marked stale "
                                f"({staleness:.1%} entity delta)"
                            )
                        else:
                            # Increment delta counter
                            conn.execute(
                                text(
                                    "UPDATE knowledge_graphs "
                                    "SET entity_delta_since_detection = entity_delta_since_detection + 1 "
                                    "WHERE id = :gid"
                                ),
                                {"gid": graph_id},
                            )
                            conn.commit()
                            return  # Not stale yet, skip re-detection

            # Queue detection with 30s delay (batches rapid successive ingests)
            detect_communities_task.apply_async(
                args=[graph_id],
                kwargs={
                    "levels": [0, 1, 2, 3, 4],
                    "resolutions": [0.25, 0.5, 1.0, 2.0, 4.0],
                    "force_rebuild": False,
                },
                countdown=30,
            )
            logger.info(f"Queued community detection for graph {graph_id} (delay=30s)")

        finally:
            pg_engine.dispose()

    except Exception as exc:
        # Non-critical — ingestion already completed
        logger.warning(f"Post-ingestion community trigger failed for {graph_id}: {exc}")


# ==================== QUERY CACHE INVALIDATION ====================


async def _invalidate_query_cache(graph_id: str) -> None:
    """
    Invalidate all Redis query cache entries for graph_id after ingest.

    Uses redis.asyncio directly — not the FastAPI singleton — to avoid sharing
    an async connection across the Celery worker fork boundary.

    Non-critical: failures are logged as warnings and never propagate.
    """
    try:
        import redis.asyncio as _aioredis

        from app.services.query_cache_service import QueryCacheService

        r = _aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        cache = QueryCacheService(r)
        try:
            deleted = await cache.invalidate_graph(graph_id)
            logger.info(
                f"Query cache invalidated for graph {graph_id}: {deleted} key(s) deleted"
            )
        finally:
            await r.aclose()
    except Exception as exc:
        logger.warning(
            f"Query cache invalidation failed for graph {graph_id}: {exc}"
        )


# ==================== VERSIONING TASKS ====================


@celery_app.task(bind=True, name="create_graph_snapshot")
def create_graph_snapshot(
    self,
    graph_id: str,
    label: str | None = None,
    description: str | None = None,
    created_by: str = "system",
    is_auto: bool = False,
    parent_version_id: str | None = None,
) -> dict[str, Any]:
    """
    Celery task: create a GraphVersion snapshot for large graphs.

    Uses a task-scoped sync Neo4j driver (NullPool pattern — Architecture Rule #5).
    Returns version metadata dict.
    """
    import uuid as _uuid
    from datetime import datetime

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
        max_connection_pool_size=1,
    )
    try:
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            # Count live entities and relationships
            count_r = session.run(
                "MATCH (e:__Entity__ {graph_id: $graph_id}) WHERE e.invalidated_at IS NULL "
                "WITH count(e) AS ec "
                "OPTIONAL MATCH (a:__Entity__ {graph_id: $graph_id})-[r {graph_id: $graph_id}]->(b:__Entity__ {graph_id: $graph_id}) "
                "WHERE r.invalidated_at IS NULL "
                "RETURN ec, count(r) AS rc",
                {"graph_id": graph_id},
            ).single()
            entity_count = int(count_r["ec"]) if count_r else 0
            relationship_count = int(count_r["rc"]) if count_r else 0

            # Auto-increment version_number
            num_r = session.run(
                "MATCH (v:GraphVersion {graph_id: $graph_id}) "
                "RETURN coalesce(max(v.version_number), 0) + 1 AS next_num",
                {"graph_id": graph_id},
            ).single()
            version_number = int(num_r["next_num"]) if num_r else 1

            version_id = str(_uuid.uuid4())
            now_iso = datetime.now(UTC).isoformat()

            session.run(
                """
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
                """,
                {
                    "version_id": version_id,
                    "graph_id": graph_id,
                    "version_number": version_number,
                    "label": label,
                    "description": description,
                    "captured_at": now_iso,
                    "created_by": created_by,
                    "parent_version_id": parent_version_id,
                    "is_auto": is_auto,
                    "entity_count": entity_count,
                    "relationship_count": relationship_count,
                    "created_at": now_iso,
                },
            )
            logger.info(
                f"Snapshot task created version {version_id} (v{version_number}) for graph {graph_id}"
            )
            return {
                "version_id": version_id,
                "version_number": version_number,
                "entity_count": entity_count,
                "relationship_count": relationship_count,
            }
    finally:
        driver.close()


# ==================== AUTO-SNAPSHOT HOOK ====================


async def _maybe_auto_snapshot(graph_id: str) -> None:
    """
    Trigger a zero-copy snapshot after ingestion if:
    1. `auto_snapshot_on_ingestion` is enabled for the graph (PostgreSQL flag).
    2. The 24h cap has not been exceeded (no auto snapshot in the last 24h).
    """
    try:
        from datetime import timedelta

        from sqlalchemy import update as sa_update

        from app.models.graph import KnowledgeGraph

        async with worker_session_maker() as session:
            import uuid as _uuid

            result = await session.execute(
                select(KnowledgeGraph).where(KnowledgeGraph.id == _uuid.UUID(graph_id))
            )
            kg = result.scalar_one_or_none()

            if not kg or not kg.auto_snapshot_on_ingestion:
                return

            now = datetime.now(UTC)
            if kg.auto_snapshot_last_at and (
                now - kg.auto_snapshot_last_at
            ) < timedelta(hours=24):
                logger.debug(
                    f"Auto-snapshot cap: graph {graph_id} already snapshotted in last 24h, skipping"
                )
                return

            # Dispatch async snapshot task
            create_graph_snapshot.delay(
                graph_id=graph_id,
                label=f"auto-{now.strftime('%Y%m%dT%H%M%S')}",
                description="Automatic snapshot triggered on ingestion",
                created_by="system",
                is_auto=True,
            )

            # Update last snapshot timestamp
            await session.execute(
                sa_update(KnowledgeGraph)
                .where(KnowledgeGraph.id == _uuid.UUID(graph_id))
                .values(auto_snapshot_last_at=now)
            )
            await session.commit()
            logger.info(f"Auto-snapshot queued for graph {graph_id}")

    except Exception as exc:
        logger.warning(f"Auto-snapshot hook failed for graph {graph_id}: {exc}")


# ==================== ASYNC ROLLBACK TASK (LARGE GRAPHS) ====================


@celery_app.task(bind=True, name="async_rollback_graph")
def async_rollback_graph(
    self,
    job_id: str,
    graph_id: str,
    version_id: str,
    mode: str = "full",
    performed_by: str = "system",
    create_checkpoint: bool = True,
    scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Celery task: roll back a large graph (>10K entities) asynchronously.

    Uses task-scoped sync Neo4j driver (NullPool — Architecture Rule #5).
    Tracks progress in PostgreSQL GraphRollbackJob table.
    """

    from neo4j import GraphDatabase
    from sqlalchemy import create_engine, text
    from sqlalchemy.pool import NullPool as SyncNullPool

    # Sync PostgreSQL engine for Celery worker
    sync_pg_url = settings.POSTGRES_URL.replace(
        "postgresql+asyncpg://", "postgresql://"
    )
    sync_engine = create_engine(sync_pg_url, poolclass=SyncNullPool)

    def _pg_update(updates: dict) -> None:
        with sync_engine.connect() as conn:
            cols = ", ".join(f"{k} = :{k}" for k in updates)
            conn.execute(
                text(f"UPDATE graph_rollback_jobs SET {cols} WHERE id = :job_id"),
                {"job_id": job_id, **updates},
            )
            conn.commit()

    # Mark running
    _pg_update({"status": "running", "started_at": datetime.now(UTC).isoformat()})

    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
        max_connection_pool_size=1,
    )
    try:
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            # Fetch captured_at from GraphVersion
            version_row = session.run(
                "MATCH (v:GraphVersion {graph_id: $gid, version_id: $vid}) RETURN v.captured_at AS ts",
                {"gid": graph_id, "vid": version_id},
            ).single()
            if not version_row:
                raise ValueError(f"Version {version_id} not found for graph {graph_id}")

            captured_at = version_row["ts"]
            if hasattr(captured_at, "iso_format"):
                captured_at = captured_at.iso_format()

            now_iso = datetime.now(UTC).isoformat()

            # Optional checkpoint before rollback
            checkpoint_vid: str | None = None
            if create_checkpoint:
                import uuid as _uuid2

                chk_id = str(_uuid2.uuid4())
                session.run(
                    """
                    CREATE (v:GraphVersion {
                        version_id: $vid, graph_id: $gid,
                        version_number: coalesce((MATCH (x:GraphVersion {graph_id: $gid}) RETURN max(x.version_number))[0], 0) + 1,
                        label: $label, description: $desc,
                        captured_at: datetime($ts), created_by: $by,
                        is_auto: true, entity_count: 0, relationship_count: 0,
                        created_at: datetime($ts)
                    })
                    """,
                    {
                        "vid": chk_id,
                        "gid": graph_id,
                        "label": f"pre-rollback-{now_iso[:19]}",
                        "desc": (
                            f"Auto-checkpoint before async rollback to {version_id}"
                        ),
                        "ts": now_iso,
                        "by": performed_by,
                    },
                )
                checkpoint_vid = chk_id

            params = {
                "graph_id": graph_id,
                "captured_at": captured_at,
                "performed_by": performed_by,
                "now": now_iso,
            }

            # Soft-delete entities added after captured_at
            r = session.run(
                "MATCH (e:__Entity__ {graph_id: $graph_id}) "
                "WHERE e.transaction_time > datetime($captured_at) AND e.invalidated_at IS NULL "
                "SET e.invalidated_at = datetime($now), e.deleted_by = $performed_by "
                "RETURN count(e) AS cnt",
                params,
            ).single()
            entities_soft_deleted = int(r["cnt"]) if r else 0

            # Restore entities invalidated after captured_at
            r = session.run(
                "MATCH (e:__Entity__ {graph_id: $graph_id}) "
                "WHERE e.invalidated_at > datetime($captured_at) "
                "REMOVE e.invalidated_at, e.deleted_by "
                "RETURN count(e) AS cnt",
                params,
            ).single()
            entities_restored = int(r["cnt"]) if r else 0

            # Soft-delete relationships added after captured_at
            r = session.run(
                "MATCH (a:__Entity__ {graph_id: $graph_id})-[r {graph_id: $graph_id}]->(b:__Entity__ {graph_id: $graph_id}) "
                "WHERE r.transaction_time > datetime($captured_at) AND r.invalidated_at IS NULL "
                "SET r.invalidated_at = datetime($now), r.deleted_by = $performed_by "
                "RETURN count(r) AS cnt",
                params,
            ).single()
            rels_soft_deleted = int(r["cnt"]) if r else 0

            # Restore relationships invalidated after captured_at
            r = session.run(
                "MATCH (a:__Entity__ {graph_id: $graph_id})-[r {graph_id: $graph_id}]->(b:__Entity__ {graph_id: $graph_id}) "
                "WHERE r.invalidated_at > datetime($captured_at) "
                "REMOVE r.invalidated_at, r.deleted_by "
                "RETURN count(r) AS cnt",
                params,
            ).single()
            rels_restored = int(r["cnt"]) if r else 0

            # Cascade-invalidate rels with soft-deleted endpoints
            session.run(
                "MATCH (a:__Entity__ {graph_id: $graph_id})-[r {graph_id: $graph_id}]->(b:__Entity__ {graph_id: $graph_id}) "
                "WHERE (a.invalidated_at IS NOT NULL OR b.invalidated_at IS NOT NULL) AND r.invalidated_at IS NULL "
                "SET r.invalidated_at = datetime($now), r.deleted_by = $performed_by",
                params,
            )

        _pg_update(
            {
                "status": "done",
                "entities_restored": entities_restored,
                "entities_soft_deleted": entities_soft_deleted,
                "relationships_restored": rels_restored,
                "relationships_soft_deleted": rels_soft_deleted,
                "checkpoint_version_id": checkpoint_vid,
                "completed_at": datetime.now(UTC).isoformat(),
            }
        )
        logger.info(f"Async rollback job {job_id} completed for graph {graph_id}")
        return {"status": "done", "job_id": job_id}

    except Exception as exc:
        _pg_update(
            {
                "status": "failed",
                "error_message": str(exc),
                "completed_at": datetime.now(UTC).isoformat(),
            }
        )
        logger.error(f"Async rollback job {job_id} failed: {exc}")
        raise
    finally:
        driver.close()
        sync_engine.dispose()


# ==================== MEMORY CONSOLIDATION TASK ====================


@celery_app.task(
    bind=True, name="app.services.background_jobs.consolidate_memories_task"
)
def consolidate_memories_task(self, graph_id: str) -> dict[str, Any]:
    """
    Celery task: consolidate duplicate Memory nodes for a graph.

    Follows the dual-driver rule: async neo4j_client is used inside
    memory_service.consolidate(); we run it via asyncio.run() to bridge
    the sync Celery worker context.

    Scheduled nightly via beat_schedule. Also triggered manually via
    POST /graphs/{graphId}/memories/consolidate.
    """
    import asyncio as _asyncio

    logger.info(f"Starting memory consolidation for graph {graph_id}")
    try:
        result = _asyncio.run(_consolidate_async(graph_id))
        logger.info(f"Memory consolidation done for graph {graph_id}: {result}")
        return result
    except Exception as exc:
        logger.error(f"Memory consolidation failed for graph {graph_id}: {exc}")
        raise


async def _consolidate_async(graph_id: str) -> dict[str, Any]:
    from app.core.neo4j_client import neo4j_client as _client
    from app.services.memory_service import memory_service as _ms

    if not _client.async_driver:
        await _client.connect_async()
    return await _ms.consolidate(graph_id)


@celery_app.task(
    bind=True, name="app.services.background_jobs.consolidate_all_memories"
)
def consolidate_all_memories(self) -> dict[str, Any]:
    """
    Nightly beat task: fan out consolidation to all active graphs.

    Fetches all distinct graph_ids that have Memory nodes, then dispatches
    a consolidate_memories_task per graph.
    """
    import asyncio as _asyncio

    async def _fetch_graphs() -> list:
        from app.core.neo4j_client import neo4j_client as _client

        if not _client.async_driver:
            await _client.connect_async()
        return await _client.execute_query(
            "MATCH (m:Memory) RETURN DISTINCT m.graph_id AS graph_id"
        )

    try:
        records = _asyncio.run(_fetch_graphs())
        graph_ids = [r["graph_id"] for r in records if r.get("graph_id")]
        for gid in graph_ids:
            consolidate_memories_task.delay(gid)
        logger.info(f"Queued memory consolidation for {len(graph_ids)} graphs")
        return {"graphs_queued": len(graph_ids)}
    except Exception as exc:
        logger.error(f"consolidate_all_memories failed: {exc}")
        raise


# ==================== MULTI-MODAL IMAGE INGESTION TASK ====================


@celery_app.task(bind=True, name="app.services.background_jobs.ingest_image_task")
def ingest_image_task(self, job_id: str, user_id: str) -> dict[str, Any]:
    """
    Celery task: process an image ingestion job.

    Flow:
    1. Load image from file path stored in IngestionJob.source_content
    2. Call vision_extractor to get entities/relationships (Claude Vision primary)
    3. Serialise vision output as structured text
    4. Feed text through the existing pipeline_service (same path as text ingestion)

    Follows the dual-driver rule — bridges sync Celery context to async pipeline
    via AsyncTaskExecutor.
    """
    return AsyncTaskExecutor.run_async_task(
        _process_image_ingestion_async, self, job_id, user_id
    )


async def _process_image_ingestion_async(
    task, job_id: str, user_id: str
) -> dict[str, Any]:
    """Async implementation of the image ingestion task."""
    job = None
    try:
        logger.info(f"Starting image ingestion for job {job_id}")

        # STEP 1: Load job from database
        async with worker_session_maker() as session:
            job_query = select(IngestionJob).where(IngestionJob.id == UUID(job_id))
            result = await session.execute(job_query)
            job = result.scalar_one_or_none()

            if not job:
                logger.error(f"Image job {job_id} not found")
                return {"error": f"Job {job_id} not found", "status": "failed"}

            # Verify graph exists in Neo4j
            async with WorkerNeo4jManager() as neo4j:
                graph_service = GraphNodeService(neo4j.get_sync_driver())
                graph = graph_service.get_graph(str(job.graph_id))
                if not graph:
                    return {"status": "error", "message": "Graph not found"}
                if graph["user_id"] != user_id:
                    return {"status": "error", "message": "Access denied"}

            await _update_job_status_async(session, job_id, "processing")

        if task:
            task.update_state(
                state="PROGRESS",
                meta={
                    "progress": 10,
                    "status": "Extracting entities from image via vision model",
                },
            )

        # STEP 2: Run vision extraction
        file_path = job.source_content
        effective = job.effective_instructions or {}
        context = effective.get("context", "")
        vision_model = effective.get("vision_model", "claude")

        from app.services.vision_extractor import vision_extractor

        try:
            vision_result = vision_extractor.extract_from_image(
                file_path, context=context, model=vision_model
            )
        except Exception as exc:
            logger.error(f"Vision extraction failed for job {job_id}: {exc}")
            async with worker_session_maker() as session:
                await _update_job_status_async(
                    session, job_id, "failed", error=f"Vision extraction failed: {exc}"
                )
            return {"status": "failed", "job_id": job_id, "error": str(exc)}

        entity_count = len(vision_result.get("entities", []))
        rel_count = len(vision_result.get("relationships", []))
        logger.info(
            f"Vision extraction for job {job_id}: "
            f"{entity_count} entities, {rel_count} relationships"
        )

        if task:
            task.update_state(
                state="PROGRESS",
                meta={
                    "progress": 40,
                    "status": (
                        f"Vision extraction complete: {entity_count} entities, {rel_count} relationships"
                    ),
                },
            )

        # STEP 3: Convert to text and feed through existing pipeline
        text = vision_extractor.to_text(vision_result, context=context)
        if not text.strip():
            # Nothing was extracted — mark complete with zero counts
            async with worker_session_maker() as session:
                await _update_job_status_async(
                    session, job_id, "completed", entities=0, relationships=0, chunks=0
                )
            return {
                "status": "completed",
                "job_id": job_id,
                "entities_created": 0,
                "relationships_created": 0,
                "note": "No entities could be extracted from this image",
            }

        documents = [
            {
                "text": text,
                "source": f"job_{job_id}",
                "title": f"Image from job {job_id}",
                "id": job_id,
                "metadata": {
                    "job_id": job_id,
                    "graph_id": str(job.graph_id),
                    "user_id": user_id,
                    "source_type": "image",
                    "vision_model": vision_model,
                    "context": context,
                    "created_at": datetime.now(UTC).isoformat(),
                },
            }
        ]

        if task:
            task.update_state(
                state="PROGRESS",
                meta={
                    "progress": 60,
                    "status": "Running knowledge graph pipeline on vision output",
                },
            )

        pipeline_result = await pipeline_service.process_documents(
            documents=documents,
            graph_id=job.graph_id,
            user_id=user_id,
            overrides=None,
            temporal_context=None,
            mode=IngestMode.INCREMENTAL,
            job_id=job_id,
        )

        if pipeline_result["status"] not in ("completed", "processing"):
            error_msg = pipeline_result.get("error", "Pipeline failed")
            async with worker_session_maker() as session:
                await _update_job_status_async(
                    session, job_id, "failed", error=error_msg
                )
            return {"status": "failed", "job_id": job_id, "error": error_msg}

        entities_created = pipeline_result.get("entities_created", entity_count)
        relationships_created = pipeline_result.get("relationships_created", rel_count)
        chunks_created = pipeline_result.get("chunks_created", 0)

        async with worker_session_maker() as session:
            await _update_job_status_async(
                session,
                job_id,
                "completed",
                entities=entities_created,
                relationships=relationships_created,
                chunks=chunks_created,
            )

        await _maybe_trigger_community_detection(str(job.graph_id))
        await _maybe_auto_snapshot(str(job.graph_id))
        await _invalidate_query_cache(str(job.graph_id))

        logger.info(
            f"✅ Image ingestion job {job_id} complete: "
            f"{entities_created} entities, {relationships_created} relationships"
        )
        return {
            "status": "completed",
            "job_id": job_id,
            "graph_id": str(job.graph_id),
            "entities_created": entities_created,
            "relationships_created": relationships_created,
            "chunks_created": chunks_created,
        }

    except Exception as exc:
        logger.error(f"Image ingestion job {job_id} failed: {exc}")
        try:
            async with worker_session_maker() as session:
                await _update_job_status_async(
                    session, job_id, "failed", error=str(exc)
                )
        except Exception as status_error:
            logger.error(f"Failed to update image job status: {status_error}")
        return {"status": "failed", "job_id": job_id, "error": str(exc)}


# ==================== CODE KNOWLEDGE GRAPH INGESTION TASK ====================


@celery_app.task(bind=True, name="app.services.background_jobs.code_ingest_task")
def code_ingest_task(self, job_id: str, user_id: str) -> dict[str, Any]:
    """
    Celery task: ingest a code repository into the Code Knowledge Graph.

    Stages (per ORA-69 spec):
      0. Bootstrap   — file discovery, dependency manifest parsing
      1. Delta       — SHA-256 hash comparison, stale node marking (sync)
      2. AST Parse   — Tree-sitter symbol extraction (ThreadPoolExecutor)
      3. Resolve     — cross-file call/import/inherit edges
      4. Embed       — Function + Class embeddings (OpenAI text-embedding-3-small)
      5. Write       — deterministic MERGE to Neo4j (batched, sync driver)
      6. Cleanup     — stale node TTL sweep

    Dual-driver rule: sync Neo4j driver with NullPool via WorkerNeo4jManager.
    PostgreSQL job tracking via IngestionJob (source_type='code').
    """
    import json as _json

    logger.info(f"Starting code ingestion job {job_id}")

    from sqlalchemy.pool import NullPool as _NullPool

    from app.services.code_parser_service import (
        IngestStats,
        bootstrap_repository,
        cleanup_stale_code_nodes_sync,
        detect_deltas_sync,
        generate_embeddings,
        parse_files_parallel,
        resolve_symbols,
        write_code_graph_sync,
    )

    # Sync PostgreSQL engine for this task
    sync_pg_url = settings.POSTGRES_URL.replace(
        "postgresql+asyncpg://", "postgresql+psycopg2://"
    )
    from sqlalchemy import create_engine as _create_engine
    from sqlalchemy import text as _text

    pg_engine = _create_engine(sync_pg_url, poolclass=_NullPool, echo=False)

    def _pg_get_job(job_id_str: str) -> Any | None:
        with pg_engine.connect() as conn:
            row = conn.execute(
                _text("SELECT * FROM ingestion_jobs WHERE id = :id"),
                {"id": job_id_str},
            ).fetchone()
        return row

    def _pg_update_job(
        job_id_str: str, status: str, progress: int, meta: dict[str, Any]
    ) -> None:
        with pg_engine.begin() as conn:
            conn.execute(
                _text("""
                    UPDATE ingestion_jobs
                    SET status = :status,
                        progress = :progress,
                        extracted_entities = :symbols_added,
                        error_message = :error,
                        completed_at = CASE WHEN :status IN ('completed','failed') THEN NOW() ELSE completed_at END,
                        started_at = CASE WHEN :status = 'running' AND started_at IS NULL THEN NOW() ELSE started_at END
                    WHERE id = :id
                """),
                {
                    "id": job_id_str,
                    "status": status,
                    "progress": progress,
                    "symbols_added": meta.get("symbols_added", 0),
                    "error": meta.get("error"),
                },
            )

    stats = IngestStats()
    job_row = None

    try:
        job_row = _pg_get_job(job_id)
        if not job_row:
            logger.error(f"Code job {job_id} not found in PostgreSQL")
            return {"status": "failed", "error": f"Job {job_id} not found"}

        graph_id = str(job_row.graph_id)
        params: dict[str, Any] = _json.loads(job_row.source_content or "{}")
        _pg_update_job(job_id, "running", 5, {})

        # ── Stage 0: Bootstrap ─────────────────────────────────────────────
        repo_path, file_metas, dep_nodes = bootstrap_repository(
            repo_path=params.get("repo_path"),
            git_url=params.get("git_url"),
            branch=params.get("branch", "main"),
            allowed_languages=(
                set(params["languages"]) if params.get("languages") else None
            ),
            exclude_patterns=params.get("exclude_patterns", []),
        )
        stats.files_scanned = len(file_metas)
        _pg_update_job(job_id, "running", 15, {"symbols_added": 0})
        logger.info(f"[{job_id}] Stage 0 done: {stats.files_scanned} files found")

        # Auto-switch depth for large repos (CTO approved: Q1 in ORA-69)
        depth_str: str = params.get("depth") or "function"
        depth_explicitly_set: bool = bool(params.get("depth"))
        depth_auto_switched = False
        if (
            not depth_explicitly_set
            and stats.files_scanned > settings.CODE_LARGE_REPO_DEPTH_THRESHOLD
        ):
            depth_str = "file"
            depth_auto_switched = True
            stats.warnings.append(
                f"Depth auto-switched to 'file': {stats.files_scanned} files "
                f"> threshold {settings.CODE_LARGE_REPO_DEPTH_THRESHOLD}"
            )
            logger.info(f"[{job_id}] Large repo auto-switch: depth='file'")

        mode = params.get("mode", "incremental")

        with WorkerNeo4jManager() as neo4j_mgr:
            driver = neo4j_mgr.get_sync_driver()

            # ── Stage 1: Delta Detection ───────────────────────────────────
            if mode == "incremental":
                with driver.session() as session:
                    new_files, changed_files = detect_deltas_sync(
                        graph_id, file_metas, session
                    )
            else:
                # Full mode: treat all files as new
                new_files, changed_files = file_metas, []

            to_parse = new_files + changed_files
            stats.files_changed = len(to_parse)
            _pg_update_job(job_id, "running", 25, {"symbols_added": 0})
            logger.info(f"[{job_id}] Stage 1 done: {len(to_parse)} files to parse")

            if depth_str == "file":
                # File-depth only: write File + dependency nodes, skip symbol extraction
                with driver.session() as session:
                    from app.services.code_parser_service import write_code_graph_sync

                    write_code_graph_sync(
                        graph_id,
                        session,
                        to_parse,
                        [],
                        dep_nodes,
                        [],
                        [],
                        [],
                        {},
                        stats,
                    )
                _pg_update_job(job_id, "completed", 100, {"symbols_added": 0})
                logger.info(f"[{job_id}] File-depth ingest complete")
                return {
                    "status": "completed",
                    "job_id": job_id,
                    "graph_id": graph_id,
                    "files_scanned": stats.files_scanned,
                    "files_changed": stats.files_changed,
                    "symbols_added": 0,
                    "depth_auto_switched": depth_auto_switched,
                    "warnings": stats.warnings,
                }

            # ── Stage 2: AST Parsing ───────────────────────────────────────
            symbols = parse_files_parallel(to_parse, max_workers=4)
            _pg_update_job(job_id, "running", 50, {"symbols_added": 0})
            logger.info(
                f"[{job_id}] Stage 2 done: {len(symbols)} raw symbols extracted"
            )

            # ── Stage 3: Cross-File Resolution ────────────────────────────
            symbols, calls_edges, imports_edges, inherits_edges = resolve_symbols(
                symbols, file_metas
            )
            _pg_update_job(job_id, "running", 65, {"symbols_added": 0})
            logger.info(
                f"[{job_id}] Stage 3 done: "
                f"{len(calls_edges)} CALLS, {len(imports_edges)} IMPORTS, "
                f"{len(inherits_edges)} INHERITS edges"
            )

            # ── Stage 4: Embedding Generation ─────────────────────────────
            embeddings = generate_embeddings(symbols)
            _pg_update_job(job_id, "running", 80, {"symbols_added": 0})
            logger.info(
                f"[{job_id}] Stage 4 done: {len(embeddings)} embeddings generated"
            )

            # ── Stage 5: Neo4j Write ───────────────────────────────────────
            with driver.session() as session:
                write_code_graph_sync(
                    graph_id,
                    session,
                    to_parse,
                    symbols,
                    dep_nodes,
                    calls_edges,
                    imports_edges,
                    inherits_edges,
                    embeddings,
                    stats,
                )
            _pg_update_job(
                job_id, "running", 95, {"symbols_added": stats.symbols_added}
            )
            logger.info(
                f"[{job_id}] Stage 5 done: {stats.symbols_added} symbols written"
            )

            # ── Stage 6: Data Flow Analysis (Python only) ─────────────────
            # Runs after write_code_graph_sync so all Function/Variable/Class
            # nodes are already in Neo4j and can be referenced by FLOWS_TO edges.
            # Uses the same sync driver (WorkerNeo4jManager / NullPool).
            try:
                from app.services.data_flow_analyzer import DataFlowAnalyzer

                python_files = [f for f in to_parse if f.language == "python"]
                if python_files:
                    dfa = DataFlowAnalyzer(sync_driver=driver)
                    flows_written = dfa.analyze_files(python_files, graph_id)
                    logger.info(
                        f"[{job_id}] Stage 6: DataFlowAnalyzer wrote "
                        f"{flows_written} FLOWS_TO edges for "
                        f"{len(python_files)} Python file(s)"
                    )
            except Exception as dfa_exc:
                # Non-fatal — data flow analysis failure must not abort ingestion
                logger.warning(
                    f"[{job_id}] DataFlowAnalyzer failed (non-fatal): {dfa_exc}"
                )

            # ── Stage 7: Stale Cleanup ─────────────────────────────────────
            if mode == "full":
                with driver.session() as session:
                    deleted = cleanup_stale_code_nodes_sync(graph_id, session)
                logger.info(f"[{job_id}] Stage 7: cleaned up {deleted} stale nodes")

        _pg_update_job(job_id, "completed", 100, {"symbols_added": stats.symbols_added})
        logger.info(
            f"Code ingestion {job_id} complete — "
            f"files={stats.files_changed}, symbols={stats.symbols_added}"
        )
        return {
            "status": "completed",
            "job_id": job_id,
            "graph_id": graph_id,
            "files_scanned": stats.files_scanned,
            "files_changed": stats.files_changed,
            "symbols_added": stats.symbols_added,
            "symbols_updated": stats.symbols_updated,
            "errors": stats.errors,
            "warnings": stats.warnings,
            "depth_auto_switched": depth_auto_switched,
        }

    except Exception as exc:
        logger.error(f"Code ingestion job {job_id} failed: {exc}", exc_info=True)
        try:
            _pg_update_job(job_id, "failed", 0, {"error": str(exc)})
        except Exception as status_err:
            logger.error(f"Failed to update code job status: {status_err}")
        stats.errors.append(str(exc))
        return {"status": "failed", "job_id": job_id, "error": str(exc)}


# ===========================================================================
# Database connector sync task (ORA-77)
# ===========================================================================


@celery_app.task(bind=True)
def sync_database_connector(
    self,
    graph_id: str,
    connector_id: str,
    user_id: str,
    sync_mode_override: str | None = None,
    table_filter_override: list | None = None,
) -> dict[str, Any]:
    """Celery task: sync a database connector (PostgreSQL/MySQL/MongoDB) into Neo4j.

    Uses WorkerNeo4jManager (async, NullPool) for task-scoped Neo4j connections.
    Credentials are fetched from the credential broker — never stored.
    """
    return AsyncTaskExecutor.run_async_task(
        _sync_database_connector_async,
        self,
        graph_id,
        connector_id,
        user_id,
        sync_mode_override,
        table_filter_override,
    )


async def _sync_database_connector_async(
    task,
    graph_id: str,
    connector_id: str,
    user_id: str,
    sync_mode_override: str | None,
    table_filter_override: list | None,
) -> dict[str, Any]:
    """Async implementation of the database connector sync task."""
    from app.services.credential_service import credential_service
    from app.services.database_connector_service import (
        DatabaseConnectorType,
        DbSyncMode,
        SchemaSnapshot,
        make_connector,
        write_table_to_kg,
    )

    logger.info(
        f"Starting database connector sync: connector={connector_id} graph={graph_id}"
    )

    async with WorkerNeo4jManager() as neo4j:
        async_driver = neo4j.get_async_driver()

        # ------------------------------------------------------------------
        # Load connector config from Neo4j
        # ------------------------------------------------------------------
        records, _, _ = await async_driver.execute_query(
            """
            MATCH (c:Connector {graph_id: $graph_id, connector_id: $connector_id})
            WHERE c.status <> 'deleted'
            RETURN c
            """,
            {"graph_id": graph_id, "connector_id": connector_id},
        )
        if not records:
            return {"status": "failed", "error": f"Connector {connector_id} not found."}

        connector_node = dict(records[0]["c"])
        connector_type = DatabaseConnectorType(connector_node["connector_type"])
        sync_mode = DbSyncMode(sync_mode_override or connector_node["sync_mode"])
        table_filter = table_filter_override or (
            __import__("json").loads(connector_node["table_filter"])
            if connector_node.get("table_filter")
            else None
        )

        config = {
            "host": connector_node["host"],
            "port": connector_node["port"],
            "database": connector_node["database"],
            "schema_filter": connector_node.get("schema_filter"),
            "table_filter": table_filter,
            "sample_row_limit": connector_node.get("sample_row_limit", 100),
        }

        # ------------------------------------------------------------------
        # Fetch credentials from broker
        # ------------------------------------------------------------------
        creds = await credential_service.get_user_credentials(
            user_id, f"db:{connector_id}"
        )
        if not creds:
            await _worker_record_sync_error(
                async_driver,
                graph_id,
                connector_id,
                "auth_error",
                "No credentials found in credential broker.",
            )
            await _worker_update_sync_status(
                async_driver,
                graph_id,
                connector_id,
                "failed",
                error_msg="No credentials registered for this connector.",
            )
            return {"status": "failed", "error": "Missing credentials."}

        db_user = (
            creds.get("username") or creds.get("user") or creds.get("access_token", "")
        )
        db_password = creds.get("password") or creds.get("secret", "")

        # ------------------------------------------------------------------
        # Connect to source database
        # ------------------------------------------------------------------
        connector = make_connector(connector_type, config)
        try:
            await connector.connect(db_user, db_password)
        except Exception as exc:
            logger.error(f"DB connector {connector_id} connect failed: {exc}")
            await _worker_record_sync_error(
                async_driver, graph_id, connector_id, "connection_failed", str(exc)
            )
            await _worker_update_sync_status(
                async_driver, graph_id, connector_id, "failed", error_msg=str(exc)
            )
            return {"status": "failed", "error": str(exc)}

        total_entities = 0
        tables_failed: list = []

        try:
            # ------------------------------------------------------------------
            # Schema introspection
            # ------------------------------------------------------------------
            snapshot = await connector.introspect_schema()

            # ------------------------------------------------------------------
            # CDC: detect schema changes against previous snapshot
            # ------------------------------------------------------------------
            if sync_mode == DbSyncMode.CDC:
                prev_snapshot_json = connector_node.get("last_schema_snapshot")
                if prev_snapshot_json:
                    prev_snapshot = SchemaSnapshot.from_json(prev_snapshot_json)
                    changes = await connector.detect_schema_changes(prev_snapshot)
                    # For CDC: only process added/altered tables, soft-delete removed ones
                    added_names = set(changes.get("added_tables", []))
                    removed_names = set(changes.get("removed_tables", []))
                    altered_names = {
                        a["table"] for a in changes.get("altered_tables", [])
                    }
                    snapshot.tables = [
                        t
                        for t in snapshot.tables
                        if t.name in added_names or t.name in altered_names
                    ]
                    # Soft-delete entities from removed tables
                    for tname in removed_names:
                        try:
                            await async_driver.execute_query(
                                """
                                MATCH (e:__Entity__ {
                                    graph_id: $graph_id,
                                    source_connector_id: $connector_id,
                                    source_table: $table_name
                                })
                                SET e.staleAt = datetime().epochMillis
                                """,
                                {
                                    "graph_id": graph_id,
                                    "connector_id": connector_id,
                                    "table_name": tname,
                                },
                            )
                        except Exception as e:
                            logger.warning(f"CDC soft-delete failed for {tname}: {e}")

            # ------------------------------------------------------------------
            # Write each table to Neo4j
            # ------------------------------------------------------------------
            for table in snapshot.tables:
                sample_rows = []
                if sync_mode != DbSyncMode.SCHEMA_ONLY:
                    try:
                        sample_rows = await connector.extract_sample_data(
                            table.name, config["sample_row_limit"]
                        )
                    except Exception as e:
                        logger.warning(
                            f"Sample extraction failed for {table.name}: {e}"
                        )
                        tables_failed.append(table.name)
                        continue

                try:
                    count = await write_table_to_kg(
                        graph_id=graph_id,
                        connector_id=connector_id,
                        table=table,
                        sync_mode=sync_mode,
                        sample_rows=sample_rows,
                        driver=async_driver,
                    )
                    total_entities += count
                except Exception as e:
                    logger.warning(f"KG write failed for {table.name}: {e}")
                    tables_failed.append(table.name)

            # Store snapshot for future CDC runs
            await async_driver.execute_query(
                """
                MATCH (c:Connector {graph_id: $graph_id, connector_id: $connector_id})
                SET c.last_schema_snapshot = $snapshot
                """,
                {
                    "graph_id": graph_id,
                    "connector_id": connector_id,
                    "snapshot": snapshot.to_json(),
                },
            )

            final_status = "success" if not tables_failed else "partial"
            await _worker_update_sync_status(
                async_driver,
                graph_id,
                connector_id,
                final_status,
                row_count=total_entities,
            )
            if tables_failed:
                await _worker_record_sync_error(
                    async_driver,
                    graph_id,
                    connector_id,
                    "write_error",
                    f"Failed to process {len(tables_failed)} table(s).",
                    tables_failed=tables_failed,
                )

            logger.info(
                f"DB connector sync complete: connector={connector_id} "
                f"entities={total_entities} failed_tables={tables_failed}"
            )
            return {
                "status": final_status,
                "connector_id": connector_id,
                "sync_mode": sync_mode.value,
                "entities_written": total_entities,
                "tables_failed": tables_failed,
            }

        except Exception as exc:
            logger.error(f"DB connector sync failed: {exc}", exc_info=True)
            await _worker_record_sync_error(
                async_driver, graph_id, connector_id, "schema_error", str(exc)
            )
            await _worker_update_sync_status(
                async_driver, graph_id, connector_id, "failed", error_msg=str(exc)
            )
            return {"status": "failed", "error": str(exc)}

        finally:
            await connector.close()


async def _worker_update_sync_status(
    driver,
    graph_id: str,
    connector_id: str,
    sync_status: str,
    row_count: int | None = None,
    error_msg: str | None = None,
) -> None:
    """Update connector sync metadata using the worker's async driver."""
    try:
        await driver.execute_query(
            """
            MATCH (c:Connector {graph_id: $graph_id, connector_id: $connector_id})
            SET c.last_sync_at        = datetime().epochMillis,
                c.last_sync_status    = $sync_status,
                c.last_sync_error     = $error_msg,
                c.last_sync_row_count = $row_count,
                c.updated_at          = datetime().epochMillis
            """,
            {
                "graph_id": graph_id,
                "connector_id": connector_id,
                "sync_status": sync_status,
                "error_msg": error_msg,
                "row_count": row_count,
            },
        )
    except Exception as e:
        logger.error(f"Failed to update sync status for {connector_id}: {e}")


async def _worker_record_sync_error(
    driver,
    graph_id: str,
    connector_id: str,
    error_type: str,
    error_message: str,
    tables_failed: list | None = None,
) -> None:
    """Record a ConnectorSyncError node using the worker's async driver."""
    import json as _json
    import uuid as _uuid

    try:
        await driver.execute_query(
            """
            MATCH (c:Connector {graph_id: $graph_id, connector_id: $connector_id})
            CREATE (e:ConnectorSyncError {
                error_id:      $error_id,
                connector_id:  $connector_id,
                graph_id:      $graph_id,
                occurred_at:   datetime().epochMillis,
                error_type:    $error_type,
                error_message: $error_message,
                tables_failed: $tables_failed
            })
            CREATE (c)-[:HAD_SYNC_ERROR {occurred_at: datetime().epochMillis}]->(e)
            WITH c
            MATCH (c)-[:HAD_SYNC_ERROR]->(old:ConnectorSyncError)
            WITH c, old ORDER BY old.occurred_at ASC
            WITH c, collect(old) AS all_errors
            WHERE size(all_errors) > 10
            UNWIND all_errors[..size(all_errors) - 10] AS stale
            DETACH DELETE stale
            """,
            {
                "graph_id": graph_id,
                "connector_id": connector_id,
                "error_id": str(_uuid.uuid4()),
                "error_type": error_type,
                "error_message": error_message,
                "tables_failed": _json.dumps(tables_failed) if tables_failed else None,
            },
        )
    except Exception as e:
        logger.error(f"Failed to record sync error for {connector_id}: {e}")


# ==================== BITEMPORAL MIGRATION TASK ====================


@celery_app.task(
    bind=True,
    name="app.services.background_jobs._run_bitemporal_migration_for_graph",
)
def _run_bitemporal_migration_for_graph(self, graph_id: str) -> dict[str, Any]:
    """
    Per-graph worker: backfill event_time, ingestion_time, and ingestion_source
    onto __Entity__ nodes and relationships for a single graph_id.

    Guard: atomic MERGE on (:Migration {id: 'bitemporal-v1-<graph_id>'}) with
    ON CREATE SET done=false ensures only one worker runs per graph even under
    concurrent dispatch.  Sets done=true on completion.

    Follows the dual-driver rule: task-scoped sync Neo4j driver with NullPool.
    """
    from neo4j import GraphDatabase

    migration_id = f"bitemporal-v1-{graph_id}"

    logger.info(
        f"Starting bitemporal migration for graph '{graph_id}' (guard={migration_id})"
    )

    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
        max_connection_pool_size=1,
    )
    try:
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            # Atomic guard: create the Migration node if it doesn't exist and
            # return whether it was already marked done.  A single MERGE avoids
            # the TOCTOU race between a separate check and create.
            guard_result = session.run(
                """
                MERGE (m:Migration {id: $migration_id})
                ON CREATE SET m.done = false, m.started_at = datetime()
                RETURN m.done AS already_done
                """,
                {"migration_id": migration_id},
            ).single()
            if guard_result and guard_result["already_done"]:
                logger.info(
                    f"Bitemporal migration '{migration_id}' already completed — skipping"
                )
                return {
                    "status": "skipped",
                    "migration_id": migration_id,
                    "graph_id": graph_id,
                }

            # Backfill entity nodes scoped to this graph_id only.
            entity_result = session.run(
                """
                MATCH (e:__Entity__)
                WHERE e.event_time IS NULL
                  AND e.graph_id = $graph_id
                SET e.event_time       = coalesce(e.ingested_at, e.ingestedAt, datetime()),
                    e.ingestion_time   = coalesce(e.ingested_at, e.ingestedAt, datetime()),
                    e.ingestion_source = 'pre-migration'
                RETURN count(e) AS updated
                """,
                {"graph_id": graph_id},
            ).single()
            entities_updated = int(entity_result["updated"]) if entity_result else 0

            # Backfill relationships scoped to this graph_id only.
            rel_result = session.run(
                """
                MATCH ()-[r]->()
                WHERE r.event_time IS NULL
                  AND (r.ingested_at IS NOT NULL OR r.transaction_time IS NOT NULL)
                  AND r.graph_id = $graph_id
                SET r.event_time       = coalesce(r.ingested_at, r.transaction_time),
                    r.ingestion_time   = coalesce(r.ingested_at, r.transaction_time),
                    r.ingestion_source = 'pre-migration'
                RETURN count(r) AS updated
                """,
                {"graph_id": graph_id},
            ).single()
            rels_updated = int(rel_result["updated"]) if rel_result else 0

            # Mark migration complete with stats.
            ran_at = datetime.now(UTC).isoformat()
            session.run(
                """
                MATCH (m:Migration {id: $migration_id})
                SET m.done                  = true,
                    m.completed_at          = datetime($ran_at),
                    m.entities_updated      = $entities_updated,
                    m.relationships_updated = $rels_updated
                """,
                {
                    "migration_id": migration_id,
                    "ran_at": ran_at,
                    "entities_updated": entities_updated,
                    "rels_updated": rels_updated,
                },
            )

        logger.info(
            f"Bitemporal migration '{migration_id}' complete: "
            f"{entities_updated} entities, {rels_updated} relationships updated"
        )
        return {
            "status": "done",
            "migration_id": migration_id,
            "graph_id": graph_id,
            "entities_updated": entities_updated,
            "relationships_updated": rels_updated,
        }

    except Exception as exc:
        logger.error(f"Bitemporal migration '{migration_id}' failed: {exc}")
        raise
    finally:
        driver.close()


@celery_app.task(bind=True, name="app.services.background_jobs.run_bitemporal_migration_v1")
def run_bitemporal_migration_v1(self) -> dict[str, Any]:
    """
    Fan-out orchestrator: fetch all distinct graph_ids from Neo4j and dispatch
    one _run_bitemporal_migration_for_graph task per graph.

    This ensures the migration never touches data across tenant boundaries —
    each sub-task is scoped to a single graph_id.
    """
    from neo4j import GraphDatabase

    logger.info("Starting bitemporal migration fan-out across all graphs")

    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
        max_connection_pool_size=1,
    )
    try:
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            records = session.run(
                "MATCH (e:__Entity__) RETURN DISTINCT e.graph_id AS graph_id"
            )
            graph_ids = [r["graph_id"] for r in records if r["graph_id"]]
    finally:
        driver.close()

    for gid in graph_ids:
        _run_bitemporal_migration_for_graph.delay(gid)

    logger.info(f"Bitemporal migration dispatched for {len(graph_ids)} graph(s)")
    return {"status": "dispatched", "graphs_queued": len(graph_ids)}
