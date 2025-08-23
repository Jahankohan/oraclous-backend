import asyncio
from datetime import datetime
from uuid import UUID
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.models.graph import IngestionJob, KnowledgeGraph
from app.core.logging import get_logger

logger = get_logger(__name__)

# Sync database connection
sync_engine = create_engine(settings.POSTGRES_URL.replace("+asyncpg", ""))
SyncSession = sessionmaker(bind=sync_engine)

def process_job_sync(task, job_id: str, user_id: str):
    """Synchronous job processing - FIXED for event loop conflicts"""
    
    with SyncSession() as db:
        try:
            # Get job
            job = db.get(IngestionJob, UUID(job_id))
            if not job:
                return {"status": "error", "message": "Job not found"}
            
            # Get graph
            graph = db.get(KnowledgeGraph, job.graph_id)
            if not graph:
                update_job_status_sync(db, job_id, "failed", "Graph not found")
                return {"status": "error", "message": "Graph not found"}
            
            # Update to processing
            update_job_status_sync(db, job_id, "processing", None, 10)
            task.update_state(state="PROGRESS", meta={"progress": 10})
            
            # FIXED: Create clean event loop context
            try:
                result = run_extraction_sync(
                    content=job.source_content,
                    user_id=user_id,
                    graph_id=job.graph_id,
                    schema=graph.schema_config,
                    task=task
                )
                
                if result["success"]:
                    # Update success with chunk count
                    update_job_status_sync(
                        db, job_id, "completed", None, 100, 
                        result["entities"], result["relationships"], 
                        result.get("chunks", 0)
                    )
                    
                    # Update graph counts
                    graph.node_count += result["entities"]
                    graph.relationship_count += result["relationships"]
                    db.commit()
                    
                    return {
                        "status": "completed", 
                        "entities_count": result["entities"], 
                        "relationships_count": result["relationships"],
                        "chunks_count": result.get("chunks", 0)
                    }
                else:
                    update_job_status_sync(db, job_id, "failed", result["error"])
                    return {"status": "error", "message": result["error"]}
                    
            except Exception as e:
                logger.error(f"Extraction failed: {e}")
                update_job_status_sync(db, job_id, "failed", str(e))
                return {"status": "error", "message": str(e)}
                
        except Exception as e:
            logger.error(f"Job processing failed: {e}")
            update_job_status_sync(db, job_id, "failed", str(e))
            return {"status": "error", "message": str(e)}

def run_extraction_sync(content: str, user_id: str, graph_id: UUID, schema: dict, task=None):
    """
    FIXED: Synchronous wrapper for extraction that properly handles event loops
    """
    
    # Check if we already have an event loop
    try:
        loop = asyncio.get_running_loop()
        logger.info("Using existing event loop")
        
        # We're in a running loop context - need to use asyncio.create_task
        # But since we're in Celery worker, we should create a new loop
        raise RuntimeError("Need new loop")
        
    except RuntimeError:
        # No loop running, create a new one
        logger.info("Creating new event loop for extraction")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async extraction in the new loop
            result = loop.run_until_complete(
                _run_extraction_async(content, user_id, graph_id, schema, task)
            )
            return result
        finally:
            # Clean up the loop
            loop.close()
            asyncio.set_event_loop(None)

async def _run_extraction_async(content: str, user_id: str, graph_id: UUID, schema: dict, task=None):
    """
    FIXED: Pure async extraction function that avoids loop conflicts
    """
    try:
        from app.services.entity_extractor import entity_extractor
        from app.services.vector_service import vector_service
        from app.core.neo4j_client import neo4j_client
        from neo4j import AsyncGraphDatabase
        from app.core.config import settings
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 20, "status": "Initializing connections"})
        
        # Initialize connections in this loop context
        await neo4j_client.connect()
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 40, "status": "Extracting entities"})
        
        # Use new dual graph extraction method
        entity_graph_docs, lexical_chunks = await entity_extractor.extract_with_dual_graph(
            text=content,
            user_id=user_id,
            graph_id=graph_id,
            domain_context=schema.get("domain") if schema else None
        )
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 60, "status": "Storing document chunks"})
        
        # Store lexical graph (document chunks)
        chunks_stored = 0
        try:
            if lexical_chunks:
                chunks_stored = await vector_service.create_text_chunks(
                    graph_id=graph_id,
                    text_chunks=lexical_chunks
                )
        except Exception as e:
            logger.warning(f"Failed to store chunks: {e}")
            # Continue without chunks
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 80, "status": "Storing entities and relationships"})
        
        # Store entity graph with direct driver
        driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
        
        try:
            await driver.verify_connectivity()
            
            entities_count = 0
            relationships_count = 0
            
            for graph_doc in entity_graph_docs:
                # Filter out orphaned nodes (nodes without relationships)
                connected_nodes = set()
                valid_relationships = []
                
                # Find all nodes that have relationships
                for rel in graph_doc.relationships:
                    if rel.source and rel.target:
                        connected_nodes.add(rel.source.id)
                        connected_nodes.add(rel.target.id)
                        valid_relationships.append(rel)
                
                # Keep only connected nodes
                filtered_nodes = [node for node in graph_doc.nodes if node.id in connected_nodes]
                
                # Store nodes
                for node in filtered_nodes:
                    await _store_node_async(driver, node, graph_id)
                    entities_count += 1
                
                # Store relationships
                for relationship in valid_relationships:
                    await _store_relationship_async(driver, relationship, graph_id)
                    relationships_count += 1
            
            logger.info(f"Stored {entities_count} entities and {relationships_count} relationships")
            
        finally:
            await driver.close()
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 100, "status": "Completed"})
        
        return {
            "success": True,
            "entities": entities_count,
            "relationships": relationships_count,
            "chunks": chunks_stored
        }
        
    except Exception as e:
        logger.error(f"Async extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "entities": 0,
            "relationships": 0,
            "chunks": 0
        }

async def _store_node_async(driver, node, graph_id: UUID):
    """Store individual node in Neo4j"""
    query = f"""
    MERGE (n:{node.type} {{id: $node_id}})
    ON CREATE SET 
        n.name = $name,
        n.graph_id = $graph_id,
        n.created_at = datetime(),
        n += $properties
    ON MATCH SET 
        n.updated_at = datetime(),
        n += $properties
    """
    
    async with driver.session() as session:
        await session.run(query, {
            "node_id": node.id,
            "name": getattr(node, 'id', str(node.id)),
            "graph_id": str(graph_id),
            "properties": node.properties or {}
        })

async def _store_relationship_async(driver, relationship, graph_id: UUID):
    """Store individual relationship in Neo4j"""
    query = f"""
    MATCH (source {{id: $source_id}})
    MATCH (target {{id: $target_id}})
    MERGE (source)-[r:{relationship.type}]->(target)
    ON CREATE SET 
        r.graph_id = $graph_id,
        r.created_at = datetime(),
        r += $properties
    ON MATCH SET 
        r.updated_at = datetime(),
        r += $properties
    """
    
    async with driver.session() as session:
        await session.run(query, {
            "source_id": relationship.source.id,
            "target_id": relationship.target.id,
            "graph_id": str(graph_id),
            "properties": relationship.properties or {}
        })

def update_job_status_sync(db, job_id: str, status: str, error: str = None, 
                          progress: int = None, entities: int = None, 
                          relationships: int = None, chunks: int = None):
    """Update job status synchronously with new chunk tracking"""
    job = db.get(IngestionJob, UUID(job_id))
    if job:
        job.status = status
        if error:
            job.error_message = error
        if progress is not None:
            job.progress = progress
        if entities is not None:
            job.extracted_entities = entities
        if relationships is not None:
            job.extracted_relationships = relationships
        if chunks is not None:
            # Add this field to your IngestionJob model if it doesn't exist
            if hasattr(job, 'processed_chunks'):
                job.processed_chunks = chunks
        
        if status == "processing" and progress == 10:
            job.started_at = datetime.utcnow()
        elif status == "completed":
            job.completed_at = datetime.utcnow()
        
        db.commit()

# Legacy support - keep the old function name for backward compatibility
async def run_extraction(content: str, user_id: str, graph_id: UUID, schema: dict):
    """
    Legacy function - now calls the sync version to avoid loop conflicts
    This should NOT be called from within Celery workers
    """
    logger.warning("run_extraction() called - this should be run_extraction_sync() in Celery context")
    
    # If we're already in an async context, run directly
    return await _run_extraction_async(content, user_id, graph_id, schema)