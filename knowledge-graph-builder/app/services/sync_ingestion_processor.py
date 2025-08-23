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
    """Synchronous job processing"""
    
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
            
            # Run extraction in async context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    run_extraction(job.source_content, user_id, job.graph_id, graph.schema_config)
                )
                
                if result["success"]:
                    # Update success with chunk count
                    update_job_status_sync(db, job_id, "completed", None, 100, 
                                        result["entities"], result["relationships"], 
                                        result.get("chunks", 0))  # NEW
                    
                    # Update graph counts
                    graph.node_count += result["entities"]
                    graph.relationship_count += result["relationships"]
                    # Note: We might want to add chunk_count to KnowledgeGraph model too
                    db.commit()
                    
                    return {"status": "completed", 
                        "entities_count": result["entities"], 
                        "relationships_count": result["relationships"],
                        "chunks_count": result.get("chunks", 0)}  # NEW
                else:
                    update_job_status_sync(db, job_id, "failed", result["error"])
                    return {"status": "error", "message": result["error"]}
                    
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Job processing failed: {e}")
            update_job_status_sync(db, job_id, "failed", str(e))
            return {"status": "error", "message": str(e)}

def update_job_status_sync(db, job_id: str, status: str, error: str = None, 
                          progress: int = None, entities: int = None, 
                          relationships: int = None, chunks: int = None):  # NEW parameter
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
            job.processed_chunks = chunks  # NEW field
        if status == "processing" and progress == 10:
            job.started_at = datetime.utcnow()
        if status == "completed":
            job.completed_at = datetime.utcnow()
        db.commit()

async def run_extraction(content: str, user_id: str, graph_id: UUID, schema: dict):
    """Run the actual extraction logic with new dual graph approach"""
    try:
        # Import the new entity extractor
        from app.services.entity_extractor import entity_extractor
        from app.services.vector_service import vector_service
        from app.core.neo4j_client import neo4j_client
        
        # Initialize Neo4j connection
        await neo4j_client.connect()
        
        # Use new dual graph extraction method
        entity_graph_docs, lexical_chunks = await entity_extractor.extract_with_dual_graph(
            text=content,
            user_id=user_id,
            graph_id=graph_id,
            domain_context=schema.get("domain") if schema else None
        )
        
        # Store lexical graph (document chunks) - NEW
        chunks_stored = 0
        if lexical_chunks:
            chunks_stored = await vector_service.create_text_chunks(
                graph_id=graph_id,
                text_chunks=lexical_chunks
            )
        
        # Store entity graph with direct driver (keep existing pattern)
        from neo4j import AsyncGraphDatabase
        from app.core.config import settings
        
        driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
        
        try:
            await driver.verify_connectivity()

            entities_count = 0
            relationships_count = 0
            
            for graph_doc in entity_graph_docs:
                # Filter out orphan nodes (keep existing logic)
                connected_node_ids = set()
                for rel in graph_doc.relationships:
                    connected_node_ids.add(rel.source.id)
                    connected_node_ids.add(rel.target.id)
                
                # Store only connected nodes
                for node in graph_doc.nodes:
                    if node.id in connected_node_ids:
                        await store_node_direct(driver, node, str(graph_id))
                        entities_count += 1
                    else:
                        logger.info(f"Skipping orphan node: {node.id} ({node.properties.get('name', 'Unknown')})")
                
                # Store relationships
                for rel in graph_doc.relationships:
                    await store_relationship_direct(driver, rel, str(graph_id))
                    relationships_count += 1
            
            return {
                "success": True, 
                "entities": entities_count, 
                "relationships": relationships_count,
                "chunks": chunks_stored  # NEW field
            }
            
        finally:
            await driver.close()
        
    except Exception as e:
        logger.error(f"Enhanced extraction failed: {e}")
        return {"success": False, "error": str(e)}


async def store_node_direct(driver, node, graph_id: str):
    """Store node directly with driver"""
    properties = dict(node.properties) if hasattr(node, 'properties') and node.properties else {}
    properties["graph_id"] = graph_id
    properties["id"] = node.id

    # ADD: Log what's being stored
    logger.info(f"Storing node: ID={node.id}, Name={properties.get('name')}")
    
    # Sanitize labels
    import re
    if isinstance(node.type, list):
        labels = [re.sub(r'[^a-zA-Z0-9_]', '_', label).strip('_') for label in node.type]
    else:
        labels = [re.sub(r'[^a-zA-Z0-9_]', '_', node.type).strip('_')]
    
    labels = [label for label in labels if label]
    if not labels:
        labels = ["Entity"]
    
    labels_str = ":".join(labels)
    
    query = f"""
    MERGE (n:{labels_str} {{id: $id, graph_id: $graph_id}})
    SET n += $properties
    """
    
    async with driver.session() as session:
        await session.run(query, {"id": node.id, "graph_id": graph_id, "properties": properties})

async def store_relationship_direct(driver, rel, graph_id: str):
    """Store relationship directly with driver"""
    properties = dict(rel.properties) if hasattr(rel, 'properties') and rel.properties else {}
    properties["graph_id"] = graph_id

    logger.info(f"Looking for nodes: source_id={rel.source.id}, target_id={rel.target.id}")
    
    # ADD: Sanitize relationship type (same as node labels)
    import re
    rel_type = re.sub(r'[^a-zA-Z0-9_]', '_', rel.type).strip('_') if rel.type else "RELATED_TO"
    if not rel_type:
        rel_type = "RELATED_TO"
    
    query = f"""
    MATCH (source {{id: $source_id, graph_id: $graph_id}})
    MATCH (target {{id: $target_id, graph_id: $graph_id}})
    MERGE (source)-[r:{rel_type}]->(target)
    SET r += $properties
    """
    
    # ADD: Error handling and logging
    try:
        async with driver.session() as session:
            await session.run(query, {
                "source_id": rel.source.id,
                "target_id": rel.target.id,
                "graph_id": graph_id,
                "properties": properties
            })
        logger.info(f"Stored relationship: {rel.source.id} -[{rel_type}]-> {rel.target.id}")
    except Exception as e:
        logger.error(f"Failed to store relationship {rel_type}: {e}")
        logger.error(f"Source: {rel.source.id}, Target: {rel.target.id}")
