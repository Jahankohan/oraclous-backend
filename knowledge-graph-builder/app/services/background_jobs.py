"""
Refactored Background Jobs using Universal Task Executor
Handles all async tasks with proper event loop isolation
"""

from celery import Celery
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select, update, or_
from sqlalchemy.pool import NullPool
from typing import Dict, Any, List
from datetime import datetime
from uuid import UUID

from app.core.config import settings
from app.models.graph import IngestionJob, KnowledgeGraph
from app.services.vector_service import vector_service
from app.services.task_executor import AsyncTaskExecutor, TaskConcurrencyManager
from app.services.ingestion_service import graphrag_ingestion_service
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

# ====== MAIN CELERY TASKS ======

@celery_app.task(bind=True)
def process_ingestion_job(self, job_id: str, user_id: str):
    """
    CORRECTED: Process ingestion job using YOUR AsyncTaskExecutor with Neo4j GraphRAG.
    Follows your established pattern: Celery task -> AsyncTaskExecutor -> async implementation
    """
    return AsyncTaskExecutor.run_async_task(_run_graphrag_ingestion_async, self, job_id, user_id)


async def _run_graphrag_ingestion_async(task, job_id: str, user_id: str) -> Dict[str, Any]:
    """
    CORRECTED: Async implementation using Neo4j GraphRAG pipeline within your task framework.
    Replaces the old 4-phase custom pipeline with Neo4j GraphRAG components.
    """
    job = None
    
    try:
        graph = None
        
        # Use your existing database session pattern
        async with worker_session_maker() as session:
            # Get job details from database
            job_query = select(IngestionJob).where(IngestionJob.id == UUID(job_id))
            result = await session.execute(job_query)
            job = result.scalar_one_or_none()
            
            if not job:
                logger.error(f"Job {job_id} not found")
                return {"error": f"Job {job_id} not found"}

            graph = await session.get(KnowledgeGraph, job.graph_id)
            if not graph:
                return {"status": "error", "message": "Graph not found"}
            
            # Update job status using your existing pattern
            await _update_job_status_async(session, job_id, "processing")
            
        # Progress reporting using your existing pattern
        if task:
            task.update_state(
                state="PROGRESS", 
                meta={"progress": 10, "status": "Starting Neo4j GraphRAG pipeline"}
            )
        
        # PHASES 1-3: Use Neo4j GraphRAG pipeline (replaces old extract/enrich/store phases)
        if task:
            task.update_state(
                state="PROGRESS", 
                meta={"progress": 30, "status": "Processing with Neo4j GraphRAG components"}
            )
        
        # Convert job content to documents format expected by GraphRAG service
        documents = [{
            "id": str(job.graph_id),
            "content": job.source_content,
            "title": "Document Content", 
            "filename": f"job_{job_id}.txt",
            "content_type": "text/plain",
            "summary": "",
            "created_at": datetime.utcnow()
        }]
        
        # Use the new GraphRAG ingestion service (replaces old entity_extractor logic)
        if task:
            task.update_state(
                state="PROGRESS", 
                meta={"progress": 50, "status": "Running Neo4j entity extraction and graph building"}
            )
        
        graphrag_result = await graphrag_ingestion_service.process_documents(
            documents=documents,
            graph_id=job.graph_id,
            user_id=user_id,
            schema_config=graph.schema_config,
            domain_context=graph.schema_config.get("domain") if graph.schema_config else None
        )
        
        entities_count = graphrag_result.get("entities_stored", 0)
        relationships_count = graphrag_result.get("relationships_stored", 0) 
        chunks_stored = graphrag_result.get("chunks_stored", 0)
        
        if task:
            task.update_state(
                state="PROGRESS", 
                meta={"progress": 80, "status": f"Neo4j GraphRAG completed: {entities_count} entities, {relationships_count} relationships"}
            )
        
        # PHASE 4: INDEX - Create vector indexes for search (unchanged)
        if task:
            task.update_state(
                state="PROGRESS", 
                meta={"progress": 90, "status": "Creating search indexes"}
            )
        
        await vector_service.ensure_indexes_exist(job.graph_id)
        
        # Update job status to completed with metrics
        if task:
            task.update_state(
                state="PROGRESS", 
                meta={"progress": 100, "status": "Completed Neo4j GraphRAG ingestion"}
            )
        
        async with worker_session_maker() as session:
            await _update_job_status_async(
                session, job_id, "completed", 
                entities=entities_count, 
                relationships=relationships_count,
                chunks=chunks_stored
            )
        
        logger.info(f"Completed Neo4j GraphRAG ingestion job {job_id}: {entities_count} entities, {relationships_count} relationships, {chunks_stored} chunks")
        
        return {
            "status": "completed", 
            "job_id": job_id,
            "entities": entities_count,
            "relationships": relationships_count,
            "chunks": chunks_stored,
            "pipeline": "neo4j_graphrag"
        }
        
    except Exception as e:
        logger.error(f"Neo4j GraphRAG ingestion job {job_id} failed: {e}")
        
        # Update job status to failed using your existing pattern
        try:
            async with worker_session_maker() as session:
                await _update_job_status_async(session, job_id, "failed", error=str(e))
        except Exception as status_error:
            logger.error(f"Failed to update job status: {status_error}")
        
        raise


async def _update_job_status_async(
    session: AsyncSession, 
    job_id: str, 
    status: str, 
    entities: int = None,
    relationships: int = None, 
    chunks: int = None,
    error: str = None
):
    """Update job status in database - keeping your existing pattern"""
    
    update_data = {
        "status": status,
        "updated_at": datetime.utcnow()
    }
    
    if entities is not None:
        update_data["entities_count"] = entities
    if relationships is not None:
        update_data["relationships_count"] = relationships
    if chunks is not None:
        update_data["chunks_count"] = chunks
    if error:
        update_data["error_message"] = error
    
    await session.execute(
        update(IngestionJob)
        .where(IngestionJob.id == UUID(job_id))
        .values(**update_data)
    )
    await session.commit()


# ==================== OTHER BACKGROUND JOBS (KEEPING YOUR PATTERNS) ====================

@celery_app.task(bind=True)
def process_embedding_generation_job(self, graph_id: str, user_id: str):
    """Generate embeddings for existing graph nodes - USING YOUR AsyncTaskExecutor pattern"""
    return AsyncTaskExecutor.run_async_task(_process_embedding_generation, self, graph_id, user_id)


async def _process_embedding_generation(task, graph_id: str, user_id: str):
    """Generate embeddings for ALL node types: Documents, Chunks, and Entities"""
    
    try:
        logger.info(f"Starting embedding generation for graph: {graph_id}")
        
        # Import here to avoid circular imports
        from app.services.enhanced_graph_service import enhanced_graph_service
        
        # Generate embeddings for all node types
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 20, "status": "Generating embeddings for entities"})
        entity_result = await enhanced_graph_service.generate_embeddings_for_entities(graph_id)
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 60, "status": "Generating embeddings for chunks"})  
        chunk_result = await enhanced_graph_service.generate_embeddings_for_chunks(graph_id)
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 80, "status": "Creating vector indexes"})
        await vector_service.ensure_indexes_exist(graph_id)
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 100, "status": "Embedding generation completed"})
        
        total_embeddings = entity_result.get("embeddings_generated", 0) + chunk_result.get("embeddings_generated", 0)
        
        logger.info(f"Generated {total_embeddings} embeddings for graph {graph_id}")
        return {
            "status": "completed",
            "embeddings_generated": total_embeddings,
            "entities_embedded": entity_result.get("embeddings_generated", 0),
            "chunks_embedded": chunk_result.get("embeddings_generated", 0)
        }
        
    except Exception as e:
        logger.error(f"Embedding generation failed for graph {graph_id}: {e}")
        raise


@celery_app.task(bind=True) 
def optimize_all_graphs(self):
    """System-wide graph optimization - SINGLETON using your TaskConcurrencyManager"""
    if not TaskConcurrencyManager.should_allow_task('optimize_all_graphs', self.request.id):
        return {'status': 'skipped', 'message': 'Optimization already running'}
    
    return AsyncTaskExecutor.run_async_task(_optimize_all_graphs_async, self)


async def _optimize_all_graphs_async(task):
    """Perform system-wide graph optimization"""
    
    try:
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 20, "status": "Optimizing vector indexes"})
        
        # Import here to avoid circular imports
        from app.services.analytics_service import analytics_service
        
        # Optimize vector indexes across all graphs
        await vector_service.optimize_all_indexes()
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 60, "status": "Running graph analytics"})
        
        # Run community detection and centrality analysis
        optimization_result = await analytics_service.optimize_all_graphs()
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 100, "status": "Optimization completed"})
        
        logger.info("System optimization completed")
        return {
            "status": "completed",
            "graphs_optimized": optimization_result.get("graphs_processed", 0),
            "indexes_optimized": True,
            "communities_detected": optimization_result.get("communities_created", 0)
        }
        
    except Exception as e:
        logger.error(f"System optimization failed: {e}")
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
            task.update_state(state="PROGRESS", meta={"progress": 30, "status": "Rebuilding vector indexes"})
        
        # Rebuild vector indexes for the graph
        await vector_service.rebuild_indexes(graph_id)
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 70, "status": "Updating search metadata"})
        
        # Update search metadata and rankings
        from app.services.search_service import search_service
        await search_service.update_search_metadata(graph_id)
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 100, "status": "Reindexing completed"})
        
        logger.info(f"Reindexing completed for graph {graph_id}")
        return {
            "status": "completed",
            "graph_id": graph_id,
            "indexes_rebuilt": True
        }
        
    except Exception as e:
        logger.error(f"Reindexing failed for graph {graph_id}: {e}")
        raise


@celery_app.task(bind=True)
def generate_graph_summary(self, graph_id: str):
    """Generate comprehensive graph summary - USING YOUR AsyncTaskExecutor pattern"""
    return AsyncTaskExecutor.run_async_task(_generate_graph_summary_async, self, graph_id)


async def _generate_graph_summary_async(task, graph_id: str):
    """Generate summary and insights for graph"""
    
    try:
        from app.services.analytics_service import analytics_service
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 20, "status": "Analyzing graph structure"})
        
        # Generate comprehensive graph analysis
        analysis_result = await analytics_service.comprehensive_graph_analysis(
            entities=[], graph_id=graph_id  # entities fetched within the method
        )
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 60, "status": "Generating summary"})
        
        # Generate LLM-based summary
        summary_result = await analytics_service.generate_graph_summary(graph_id)
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 100, "status": "Summary completed"})
        
        logger.info(f"Graph summary generated for {graph_id}")
        return {
            "status": "completed",
            "graph_id": graph_id,
            "summary": summary_result.get("summary", ""),
            "key_insights": summary_result.get("insights", []),
            "metrics": analysis_result.get("metrics", {})
        }
        
    except Exception as e:
        logger.error(f"Summary generation failed for graph {graph_id}: {e}")
        raise