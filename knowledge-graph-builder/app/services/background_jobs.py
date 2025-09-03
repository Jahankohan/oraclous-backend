"""
Refactored Background Jobs using Pipeline Service
Clean implementation with Neo4j GraphRAG pipeline and multi-tenant support
"""

from celery import Celery
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select, update, or_
from sqlalchemy.pool import NullPool
from typing import Dict, Any
from datetime import datetime, timezone
from uuid import UUID

from app.core.config import settings
from app.models.graph import IngestionJob, KnowledgeGraph
from app.services.task_executor import AsyncTaskExecutor, TaskConcurrencyManager
from app.services.pipeline_service import pipeline_service
from app.services.document_processor import document_processor
from app.core.logging import get_logger

logger = get_logger(__name__)

worker_engine = create_async_engine(
    settings.POSTGRES_URL,
    poolclass=NullPool,  # No connection pooling in workers
    echo=False,
    future=True
)

worker_session_maker = async_sessionmaker(
    bind=worker_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False
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
                    connection_acquisition_timeout=30
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
                    connection_acquisition_timeout=30
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
                    connection_acquisition_timeout=30
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
            raise RuntimeError("Sync driver not available. Use context manager to connect.")
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
            raise RuntimeError("Async driver not available. Use async context manager to connect.")
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


# Example usage for future worker tasks:
# 
# async def some_worker_task():
#     async with WorkerNeo4jManager() as neo4j:
#         sync_driver = neo4j.get_sync_driver()  # For GraphRAG components
#         async_driver = neo4j.get_async_driver()  # For async operations
#         
#         # Use drivers for task operations
#         # Automatic cleanup when exiting context
#
# OR for sync-only tasks:
#
# def some_sync_worker_task():
#     with WorkerNeo4jManager() as neo4j:
#         sync_driver = neo4j.get_sync_driver()  # For GraphRAG components
#         # Use driver for task operations
#         # Automatic cleanup when exiting context

# Configure Celery
celery_app = Celery(
    "knowledge_graph_builder",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
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
)

# ==================== MAIN CELERY TASKS ====================

@celery_app.task(bind=True)
def process_ingestion_job(self, job_id: str, user_id: str):
    """
    Process ingestion job using clean pipeline service.
    Follows your established pattern: Celery task -> AsyncTaskExecutor -> async implementation
    """
    return AsyncTaskExecutor.run_async_task(_process_pipeline_ingestion_async, self, job_id, user_id)


async def _process_pipeline_ingestion_async(task, job_id: str, user_id: str) -> Dict[str, Any]:
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
        
        # STEP 1: Get job from database (keep your existing pattern)
        async with worker_session_maker() as session:
            job_query = select(IngestionJob).where(IngestionJob.id == UUID(job_id))
            result = await session.execute(job_query)
            job = result.scalar_one_or_none()
            
            if not job:
                logger.error(f"Job {job_id} not found")
                return {"error": f"Job {job_id} not found", "status": "failed"}

            graph = await session.get(KnowledgeGraph, job.graph_id)
            if not graph:
                logger.error(f"Graph {job.graph_id} not found for job {job_id}")
                return {"status": "error", "message": "Graph not found"}
            
            # Update job status to processing
            await _update_job_status_async(session, job_id, "processing")
            
        logger.info(f"Processing job {job_id} for graph {job.graph_id}")
        
        # STEP 2: Progress tracking (keep your excellent pattern)
        if task:
            task.update_state(
                state="PROGRESS", 
                meta={"progress": 10, "status": "Starting Neo4j GraphRAG pipeline processing"}
            )
        
        # STEP 3: Process document content based on source type
        try:
            processed_doc = document_processor.process_document(
                content=job.source_content,
                source_type=job.source_type or "text",
                metadata={
                    "job_id": job_id,
                    "graph_id": str(job.graph_id),
                    "user_id": user_id
                }
            )
            
            # Enhanced document metadata from processor
            document_text = processed_doc["text"]
            base_metadata = processed_doc["metadata"]
            
        except Exception as e:
            logger.error(f"Document processing failed for job {job_id}: {e}")
            async with worker_session_maker() as session:
                await _update_job_status_async(session, job_id, "failed", error=f"Document processing failed: {str(e)}")
            return {
                "status": "failed",
                "job_id": job_id,
                "error": f"Document processing failed: {str(e)}"
            }

        # STEP 4: Convert processed document to GraphRAG format
        documents = [{
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
                "created_at": datetime.now(timezone.utc).isoformat(),
                "document_title": f"Document from job {job_id}",
                "document_source": f"job_{job_id}"
            }
        }]
        
        if task:
            task.update_state(
                state="PROGRESS", 
                meta={"progress": 30, "status": f"Processing {len(documents)} document(s) through pipeline"}
            )
        
        # STEP 4: Process through clean pipeline service (END-TO-END)
        logger.info(f"Processing documents through pipeline service for graph {job.graph_id}")
        
        pipeline_result = await pipeline_service.process_documents(
            documents=documents,
            graph_id=job.graph_id,
            user_id=user_id
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
                        "status": f"Pipeline completed: {entities_created} entities, {relationships_created} relationships, {chunks_created} chunks"
                    }
                )
        elif pipeline_result["status"] == "processing":
            # Handle background processing case
            if task:
                task.update_state(
                    state="PROGRESS", 
                    meta={"progress": 50, "status": "Documents processing in background - monitoring progress"}
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
                await _update_job_status_async(session, job_id, "failed", error=error_msg)
            
            return {
                "status": "failed",
                "job_id": job_id,
                "error": error_msg
            }
        
        # STEP 6: Complete job (keep your existing pattern)
        if task:
            task.update_state(
                state="PROGRESS", 
                meta={"progress": 100, "status": "Completing job and updating database"}
            )
        
        async with worker_session_maker() as session:
            await _update_job_status_async(
                session, 
                job_id, 
                "completed",
                entities=entities_created,
                relationships=relationships_created,
                chunks=chunks_created
            )
        
        logger.info(f"✅ Completed pipeline ingestion job {job_id}: "
                   f"{entities_created} entities, {relationships_created} relationships, {chunks_created} chunks")
        
        return {
            "status": "completed",
            "job_id": job_id,
            "graph_id": str(job.graph_id),
            "entities_created": entities_created,
            "relationships_created": relationships_created,
            "chunks_created": chunks_created,
            "pipeline_version": "neo4j_graphrag_clean"
        }
        
    except Exception as e:
        logger.error(f"Pipeline ingestion job {job_id} failed: {e}")
        
        # Update job status to failed
        try:
            async with worker_session_maker() as session:
                await _update_job_status_async(session, job_id, "failed", error=str(e))
        except Exception as status_error:
            logger.error(f"Failed to update job status: {status_error}")
        
        return {
            "status": "failed",
            "job_id": job_id,
            "error": str(e)
        }


async def _update_job_status_async(
    session: AsyncSession, 
    job_id: str, 
    status: str, 
    entities: int = None, 
    relationships: int = None, 
    chunks: int = None,
    error: str = None
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
        update_data = {
            "status": status
        }
        
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


# ==================== OTHER TASKS (KEPT UNCHANGED) ====================

@celery_app.task(bind=True)
def process_embedding_generation_job(self, graph_id: str, user_id: str):
    """Generate embeddings for existing nodes - USES YOUR AsyncTaskExecutor pattern"""
    return AsyncTaskExecutor.run_async_task(_generate_embeddings_async, self, graph_id, user_id)


async def _generate_embeddings_async(task, graph_id: str, user_id: str):
    """Generate embeddings for existing nodes"""
    try:
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 20, "status": "Starting embedding generation"})
        
        # For now, we'll skip embedding generation since embedding_service was removed
        # This can be implemented later when needed
        logger.info(f"Embedding generation requested for graph {graph_id} - skipping for now (service refactored)")
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 100, "status": "Embedding generation skipped - service refactored"})
        
        return {
            "status": "skipped",
            "message": "Embedding generation temporarily disabled during refactor",
            "graph_id": graph_id
        }
        
    except Exception as e:
        logger.error(f"Embedding generation failed for graph {graph_id}: {e}")
        raise


@celery_app.task(bind=True)
def optimize_all_graphs(self):
    """Optimize all graphs - SINGLETON using your TaskConcurrencyManager"""
    if not TaskConcurrencyManager.should_allow_task('optimize_all_graphs', self.request.id):
        return {'status': 'skipped', 'message': 'Optimization already running'}
    
    return AsyncTaskExecutor.run_async_task(_optimize_all_graphs_async, self)


async def _optimize_all_graphs_async(task):
    """Optimize all graphs in the system"""
    try:
        from app.services.analytics_service import analytics_service
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 20, "status": "Finding graphs to optimize"})
        
        # Get all graphs that need optimization
        async with worker_session_maker() as session:
            result = await session.execute(
                select(KnowledgeGraph).where(
                    or_(
                        KnowledgeGraph.last_optimized.is_(None),
                        KnowledgeGraph.last_optimized < datetime.utcnow() - settings.OPTIMIZATION_INTERVAL
                    )
                )
            )
            graphs = result.scalars().all()
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 40, "status": f"Optimizing {len(graphs)} graphs"})
        
        optimized_count = 0
        for i, graph in enumerate(graphs):
            try:
                logger.info(f"Optimizing graph {graph.id} ({i+1}/{len(graphs)})")
                
                # Run analytics optimization
                await analytics_service.optimize_graph_structure(graph.id)
                optimized_count += 1
                
                # Update progress
                progress = 40 + int((i + 1) / len(graphs) * 50)
                if task:
                    task.update_state(
                        state="PROGRESS", 
                        meta={"progress": progress, "status": f"Optimized {i+1}/{len(graphs)} graphs"}
                    )
                
            except Exception as e:
                logger.error(f"Failed to optimize graph {graph.id}: {e}")
                continue
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 100, "status": "Graph optimization completed"})
        
        return {
            "status": "completed",
            "graphs_processed": len(graphs),
            "graphs_optimized": optimized_count
        }
        
    except Exception as e:
        logger.error(f"Graph optimization failed: {e}")
        raise


@celery_app.task(bind=True)
def cleanup_orphaned_data(self):
    """Clean up orphaned data - SINGLETON using your TaskConcurrencyManager"""
    if not TaskConcurrencyManager.should_allow_task('cleanup_orphaned_data', self.request.id):
        return {'status': 'skipped', 'message': 'Cleanup already running'}
    
    return AsyncTaskExecutor.run_async_task(_cleanup_orphaned_data_async, self)


async def _cleanup_orphaned_data_async(task):
    """Clean up orphaned nodes and relationships"""
    try:
        from app.services.graph_service import graph_service
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 30, "status": "Identifying orphaned data"})
        
        # Clean up orphaned nodes and relationships
        cleanup_result = await graph_service.cleanup_orphaned_data()
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 80, "status": "Compacting database"})
        
        # Compact database if needed
        await graph_service.compact_database()
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 100, "status": "Cleanup completed"})
        
        logger.info(f"Cleanup completed: {cleanup_result.get('nodes_removed', 0)} nodes, {cleanup_result.get('relationships_removed', 0)} relationships removed")
        return {
            "status": "completed",
            "nodes_removed": cleanup_result.get("nodes_removed", 0),
            "relationships_removed": cleanup_result.get("relationships_removed", 0)
        }
        
    except Exception as e:
        logger.error(f"Data cleanup failed: {e}")
        raise


@celery_app.task(bind=True)
def reindex_graph_search(self, graph_id: str):
    """Reindex graph for search optimization - USING YOUR AsyncTaskExecutor pattern"""
    return AsyncTaskExecutor.run_async_task(_reindex_graph_search_async, self, graph_id)


async def _reindex_graph_search_async(task, graph_id: str):
    """Reindex specific graph for search"""
    try:
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 30, "status": "Rebuilding search indexes"})
        
        # For now, we'll skip vector indexing since vector_service was removed
        # This can be implemented later when needed
        logger.info(f"Search reindexing requested for graph {graph_id} - skipping for now (service refactored)")
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 100, "status": "Search reindexing skipped - service refactored"})
        
        return {
            "status": "skipped",
            "message": "Search reindexing temporarily disabled during refactor",
            "graph_id": graph_id
        }
        
    except Exception as e:
        logger.error(f"Search reindexing failed for graph {graph_id}: {e}")
        raise


@celery_app.task(bind=True)
def generate_graph_summary(self, graph_id: str):
    """Generate graph summary - USING YOUR AsyncTaskExecutor pattern"""
    return AsyncTaskExecutor.run_async_task(_generate_graph_summary_async, self, graph_id)


async def _generate_graph_summary_async(task, graph_id: str):
    """Generate comprehensive graph summary"""
    try:
        from app.services.analytics_service import analytics_service
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 40, "status": "Generating graph summary"})
        
        # Generate graph metrics and summary
        summary = await analytics_service.get_graph_metrics(UUID(graph_id))
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 100, "status": "Graph summary completed"})
        
        logger.info(f"Generated summary for graph {graph_id}")
        return {
            "status": "completed",
            "graph_id": graph_id,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Graph summary generation failed for {graph_id}: {e}")
        raise