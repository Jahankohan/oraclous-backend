from celery import Celery
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from typing import Dict, Any
import asyncio
import os
from datetime import datetime
from uuid import UUID

from app.core.config import settings
from app.core.database import async_session_maker
from app.core.neo4j_client import neo4j_client
from app.models.graph import IngestionJob, KnowledgeGraph
from app.services.entity_extractor import entity_extractor
from app.services.graph_service import graph_service
from app.core.logging import get_logger

logger = get_logger(__name__)

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

@celery_app.task(bind=True)
def process_ingestion_job(self, job_id: str, user_id: str):
    """Synchronous background task to process ingestion job"""
    import asyncio
    
    # Create a new event loop for this task
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Import here to avoid circular imports
        from app.services.sync_ingestion_processor import process_job_sync
        return process_job_sync(self, job_id, user_id)
    finally:
        loop.close()

@celery_app.task(bind=True)
def process_embedding_generation_job(self, graph_id: str, user_id: str):
    """Celery task to process embedding generation"""
    import asyncio
    
    # Create a new event loop for this task
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(
            _process_embedding_generation_async(self, graph_id, user_id)
        )
    finally:
        loop.close()


async def _process_embedding_generation_async(task, graph_id: str, user_id: str):
    """Async function to process embedding generation"""
    
    async with async_session_maker() as db:
        try:
            # Import here to avoid circular imports
            from app.services.enhanced_graph_service import enhanced_graph_service
            
            logger.info(f"Starting embedding generation for graph {graph_id}")
            
            # Update progress - Starting
            if task:
                task.update_state(state="PROGRESS", meta={
                    "progress": 10, 
                    "status": "Starting embedding generation",
                    "graph_id": graph_id
                })
            
            # Verify graph exists and user has access
            result = await db.execute(
                select(KnowledgeGraph).where(
                    KnowledgeGraph.id == UUID(graph_id),
                    KnowledgeGraph.user_id == UUID(user_id)
                )
            )
            graph = result.scalar_one_or_none()
            
            if not graph:
                logger.error(f"Graph {graph_id} not found or user {user_id} doesn't have access")
                if task:
                    task.update_state(state="FAILURE", meta={"error": "Graph not found or access denied"})
                return {"status": "error", "message": "Graph not found"}
            
            # Update progress - Processing
            if task:
                task.update_state(state="PROGRESS", meta={
                    "progress": 30, 
                    "status": "Processing nodes for embedding generation",
                    "graph_id": graph_id
                })
            
            # Initialize Neo4j connection
            await neo4j_client.connect()
            
            # Generate embeddings for existing nodes
            nodes_processed = await enhanced_graph_service.generate_embeddings_for_existing_nodes(
                graph_id=UUID(graph_id),
                user_id=user_id,
                batch_size=50
            )
            
            # Update progress - Completed
            if task:
                task.update_state(state="PROGRESS", meta={
                    "progress": 100, 
                    "status": "Embedding generation completed",
                    "nodes_processed": nodes_processed,
                    "graph_id": graph_id
                })
            
            logger.info(f"Background embedding generation completed for graph {graph_id}: {nodes_processed} nodes processed")
            
            return {
                "status": "completed", 
                "nodes_processed": nodes_processed,
                "graph_id": graph_id
            }
            
        except Exception as e:
            logger.error(f"Background embedding generation failed for graph {graph_id}: {e}")
            if task:
                task.update_state(state="FAILURE", meta={
                    "error": str(e),
                    "graph_id": graph_id
                })
            return {"status": "error", "message": str(e)}


async def _process_ingestion_job_async(task, job_id: str, user_id: str):
    """Async function to process ingestion job with Diffbot support"""
    
    async with async_session_maker() as db:
        try:
            # Get the ingestion job
            result = await db.execute(
                select(IngestionJob).where(IngestionJob.id == UUID(job_id))
            )
            job = result.scalar_one_or_none()
            
            if not job:
                logger.error(f"Ingestion job {job_id} not found")
                return {"status": "error", "message": "Job not found"}
            
            # Get the associated graph
            result = await db.execute(
                select(KnowledgeGraph).where(KnowledgeGraph.id == job.graph_id)
            )
            graph = result.scalar_one_or_none()
            
            if not graph:
                logger.error(f"Graph {job.graph_id} not found")
                await _update_job_status(db, job_id, "failed", "Graph not found")
                return {"status": "error", "message": "Graph not found"}
            
            # Update job status to processing
            await _update_job_status(db, job_id, "processing", None, 10)
            task.update_state(state="PROGRESS", meta={"progress": 10})
            
            # Initialize Neo4j connection
            await neo4j_client.connect()
            
            # Extract entities and relationships using hybrid approach
            logger.info(f"Starting hybrid entity extraction for job {job_id}")
            
            graph_documents = await entity_extractor.extract_entities_hybrid(
                text=job.source_content,
                user_id=user_id,
                graph_id=job.graph_id,
                schema=graph.schema_config,
                use_diffbot=True,  # Enable Diffbot
                provider="openai"  # TODO: Make configurable
            )
            
            # Update progress
            await _update_job_status(db, job_id, "processing", None, 60)
            task.update_state(state="PROGRESS", meta={"progress": 60})
            
            # Store graph documents in Neo4j
            logger.info(f"Storing {len(graph_documents)} graph documents")
            
            entities_count, relationships_count = await graph_service.store_graph_documents(
                graph_id=job.graph_id,
                graph_documents=graph_documents
            )
            
            # Update progress
            await _update_job_status(db, job_id, "processing", None, 90)
            task.update_state(state="PROGRESS", meta={"progress": 90})
            
            # Update graph statistics
            await db.execute(
                update(KnowledgeGraph)
                .where(KnowledgeGraph.id == job.graph_id)
                .values(
                    node_count=KnowledgeGraph.node_count + entities_count,
                    relationship_count=KnowledgeGraph.relationship_count + relationships_count
                )
            )
            
            # Mark job as completed
            await db.execute(
                update(IngestionJob)
                .where(IngestionJob.id == UUID(job_id))
                .values(
                    status="completed",
                    progress=100,
                    extracted_entities=entities_count,
                    extracted_relationships=relationships_count,
                    completed_at=datetime.utcnow()
                )
            )
            await db.commit()
            
            logger.info(f"Ingestion job {job_id} completed successfully with Diffbot+LLM")
            return {
                "status": "completed",
                "entities_count": entities_count,
                "relationships_count": relationships_count
            }
            
        except Exception as e:
            logger.error(f"Ingestion job {job_id} failed: {e}")
            await _update_job_status(db, job_id, "failed", str(e))
            return {"status": "error", "message": str(e)}

async def _update_job_status(
    db: AsyncSession, 
    job_id: str, 
    status: str, 
    error_message: str = None,
    progress: int = None
):
    """Update job status in database"""
    update_data = {"status": status}
    
    if error_message:
        update_data["error_message"] = error_message
    if progress is not None:
        update_data["progress"] = progress
    if status == "processing" and progress == 10:
        update_data["started_at"] = datetime.utcnow()
    
    await db.execute(
        update(IngestionJob)
        .where(IngestionJob.id == UUID(job_id))
        .values(**update_data)
    )
    await db.commit()
