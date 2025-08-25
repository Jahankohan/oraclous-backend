from celery import Celery
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from typing import Dict, Any, List
from datetime import datetime
from uuid import UUID

from app.core.config import settings
from app.core.database import async_session_maker
from app.core.neo4j_client import neo4j_client
from app.models.graph import IngestionJob, KnowledgeGraph
from app.services.entity_extractor import entity_extractor
from app.services.enhanced_graph_service import enhanced_graph_service
from app.services.vector_service import vector_service
from app.services.embedding_service import embedding_service
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
    """Celery task to process embedding generation - FIXED for event loop conflicts"""
    import asyncio
    
    # Create a new event loop for this task
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(
            _process_embedding_generation_sync(self, graph_id, user_id)
        )
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        loop.close()
        asyncio.set_event_loop(None)

@celery_app.task(bind=True)
def optimize_all_graphs(self):
    """Background task to optimize all graphs periodically"""
    import asyncio
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(_optimize_all_graphs_async(self))
    finally:
        loop.close()

async def _process_ingestion_job_async(task, job_id: str, user_id: str):
    """Enhanced async ingestion processing with dual graph"""
    
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
            
            # Step 1: Extract using new dual graph approach
            logger.info(f"Starting dual graph extraction for job {job_id}")
            task.update_state(state="PROGRESS", meta={"progress": 30, "status": "Analyzing document structure"})
            
            entity_graph_docs, lexical_chunks = await entity_extractor.extract_with_dual_graph(
                text=job.source_content,
                user_id=user_id,
                graph_id=job.graph_id,
                domain_context=graph.schema_config.get("domain") if graph.schema_config else None
            )
            
            # Step 2: Store lexical graph (document chunks)
            task.update_state(state="PROGRESS", meta={"progress": 50, "status": "Storing document chunks"})
            
            chunks_stored = 0
            if lexical_chunks:
                chunks_stored = await vector_service.create_text_chunks(
                    graph_id=job.graph_id,
                    text_chunks=lexical_chunks
                )
            logger.info(f"Stored {chunks_stored} document chunks")
            
            # Step 3: Store entity graph (entities + relationships)
            task.update_state(state="PROGRESS", meta={"progress": 70, "status": "Storing entities and relationships"})
            
            entities_count, relationships_count = await enhanced_graph_service.store_graph_documents_with_embeddings(
                graph_id=job.graph_id,
                graph_documents=entity_graph_docs,
                user_id=user_id,
                generate_embeddings=True
            )

            similarity_count = 0
            if settings.ENABLE_SIMILARITY_PROCESSING:  # Config flag
                task.update_state(state="PROGRESS", meta={"progress": 80, "status": "Creating similarity relationships"})
                
                similarity_count = await _create_similarity_relationships(
                    graph_id=job.graph_id,
                    chunks=lexical_chunks,
                    entities_count=entities_count
                )
                logger.info(f"Created {similarity_count} similarity relationships")
            
            # NEW Step 5: Community detection and cleanup (OPTIONAL)
            communities_found = 0
            if settings.ENABLE_COMMUNITY_DETECTION:  # Config flag
                task.update_state(state="PROGRESS", meta={"progress": 90, "status": "Graph optimization"})
                
                communities_found = await _detect_communities_and_cleanup(job.graph_id)
                logger.info(f"Detected {communities_found} communities")
            
            # Step 6: Update job completion with new metrics
            await db.execute(
                update(IngestionJob)
                .where(IngestionJob.id == UUID(job_id))
                .values(
                    status="completed",
                    progress=100,
                    extracted_entities=entities_count,
                    extracted_relationships=relationships_count,
                    processed_chunks=chunks_stored,
                    similarity_relationships=similarity_count,
                    communities_detected=communities_found,
                    completed_at=datetime.utcnow()
                )
            )
            await db.commit()
            
            logger.info(f"Enhanced ingestion job {job_id} completed successfully")
            return {
                "status": "completed",
                "entities_count": entities_count,
                "relationships_count": relationships_count,
                "chunks_count": chunks_stored,
                "similarity_relationships": similarity_count,
                "communities_detected": communities_found
            }
            
        except Exception as e:
            logger.error(f"Enhanced ingestion job {job_id} failed: {e}")
            await _update_job_status(db, job_id, "failed", str(e))
            return {"status": "error", "message": str(e)}

async def _process_embedding_generation_sync(task, graph_id: str, user_id: str):
    """FIXED: Simplified sync embedding generation without loop conflicts"""
    
    try:
        logger.info(f"Starting embedding generation for graph {graph_id}")
        
        # Initialize Neo4j connection in this loop
        await neo4j_client.connect()
        
        # Initialize embedding service with proper credentials
        from app.services.credential_service import credential_service
        
        # Get OpenAI credentials directly (avoiding service call conflicts)
        try:
            # Try to initialize embedding service
            from app.services.embedding_service import embedding_service
            success = await embedding_service.initialize_embeddings(
                provider="openai",
                model="text-embedding-3-small", 
                user_id=user_id
            )
            
            if not success:
                # Fallback: try with environment variable
                import os
                openai_key = os.getenv('OPENAI_API_KEY')
                if openai_key:
                    success = await embedding_service.initialize_embeddings(
                        provider="openai",
                        model="text-embedding-3-small",
                        user_id=user_id,
                        api_key=openai_key  # Direct API key
                    )
        
        except Exception as cred_error:
            logger.error(f"Credential retrieval error: {cred_error}")
            # Try with environment variable as fallback
            import os
            openai_key = os.getenv('OPENAI_API_KEY')
            if not openai_key:
                return {
                    "status": "error", 
                    "message": "OpenAI API key not available"
                }
            
            from app.services.embedding_service import embedding_service
            success = await embedding_service.initialize_embeddings(
                provider="openai",
                model="text-embedding-3-small",
                user_id=user_id,
                api_key=openai_key
            )
        
        if not success:
            return {
                "status": "error", 
                "message": "Failed to initialize embedding service"
            }
            
        logger.info(f"Embeddings initialized: {embedding_service.provider} - {embedding_service.model_name} (dim: {embedding_service.dimension})")
        
        # Process nodes in batches
        batch_size = 10
        total_processed = 0
        
        while True:
            # Get nodes without embeddings - FIXED query execution
            try:
                get_nodes_query = """
                MATCH (n)
                WHERE n.graph_id = $graph_id 
                AND n.embedding IS NULL
                AND n.name IS NOT NULL
                RETURN n.id as id, n.name as name, 
                       coalesce(n.description, '') as description
                LIMIT $batch_size
                """
                
                # Execute query with proper error handling
                result = await neo4j_client.execute_query(get_nodes_query, {
                    "graph_id": graph_id,
                    "batch_size": batch_size
                })
                
                if not result:
                    logger.info("No more nodes to process")
                    break
                
                logger.info(f"Processing batch of {len(result)} nodes")
                
                # Generate embeddings for this batch
                texts = []
                node_ids = []
                
                for node in result:
                    # Combine name and description for embedding
                    text = f"{node['name']}"
                    if node['description']:
                        text += f" {node['description']}"
                    
                    texts.append(text)
                    node_ids.append(node['id'])
                
                # Generate embeddings
                try:
                    embeddings = await embedding_service.embed_documents(texts)
                    
                    # Update nodes with embeddings
                    for node_id, embedding in zip(node_ids, embeddings):
                        update_query = """
                        MATCH (n {id: $node_id, graph_id: $graph_id})
                        SET n.embedding = $embedding
                        """
                        
                        await neo4j_client.execute_write_query(update_query, {
                            "node_id": node_id,
                            "graph_id": graph_id,
                            "embedding": embedding
                        })
                    
                    total_processed += len(embeddings)
                    logger.info(f"Updated {len(embeddings)} nodes with embeddings. Total: {total_processed}")
                    
                    # Update task progress
                    task.update_state(
                        state="PROGRESS", 
                        meta={
                            "progress": min(90, total_processed * 2),  # Rough progress
                            "nodes_processed": total_processed
                        }
                    )
                
                except Exception as embed_error:
                    logger.error(f"Embedding generation failed for batch: {embed_error}")
                    continue
                    
            except Exception as query_error:
                logger.error(f"Query execution failed: {query_error}")
                logger.error(f"Query: {get_nodes_query}")
                logger.error(f"Parameters: {{'graph_id': '{graph_id}', 'batch_size': {batch_size}}}")
                break
        
        # Create vector indexes if needed
        try:
            from app.services.vector_service import vector_service
            await vector_service.create_vector_indexes(
                dimension=embedding_service.dimension
            )
        except Exception as e:
            logger.warning(f"Failed to create vector indexes: {e}")
        
        logger.info(f"Embedding generation completed. Total nodes processed: {total_processed}")
        
        return {
            "status": "completed",
            "nodes_processed": total_processed,
            "message": f"Successfully generated embeddings for {total_processed} nodes"
        }
        
    except Exception as e:
        error_msg = f"Background embedding generation failed for graph {graph_id}: {e}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg
        }

async def _create_similarity_relationships(
    graph_id: UUID,
    chunks: List[Dict[str, Any]],
    entities_count: int
) -> int:
    """Create SIMILAR relationships between semantically related chunks and entities"""
    
    if not embedding_service.is_initialized():
        logger.warning("Embedding service not initialized, skipping similarity relationships")
        return 0
    
    try:
        # Create chunk-to-chunk similarities
        chunk_similarities = await _create_chunk_similarities(graph_id, chunks)
        
        # Create entity-to-chunk similarities
        entity_similarities = await _create_entity_chunk_similarities(graph_id)
        
        total_similarities = chunk_similarities + entity_similarities
        logger.info(f"Created {total_similarities} similarity relationships")
        
        return total_similarities
        
    except Exception as e:
        logger.error(f"Failed to create similarity relationships: {e}")
        return 0


async def _create_chunk_similarities(graph_id: UUID, chunks: List[Dict[str, Any]]) -> int:
    """Create SIMILAR relationships between semantically related chunks"""
    
    similarities_created = 0
    similarity_threshold = 0.85  # Configurable threshold
    
    # Compare each chunk with others
    for i, chunk1 in enumerate(chunks):
        if not chunk1.get("embedding"):
            continue
            
        for j, chunk2 in enumerate(chunks[i+1:], i+1):
            if not chunk2.get("embedding"):
                continue
            
            # Calculate similarity
            similarity = entity_extractor._cosine_similarity(
                chunk1["embedding"], 
                chunk2["embedding"]
            )
            
            if similarity >= similarity_threshold:
                # Create SIMILAR relationship
                query = """
                MATCH (c1:DocumentChunk {id: $chunk1_id, graph_id: $graph_id})
                MATCH (c2:DocumentChunk {id: $chunk2_id, graph_id: $graph_id})
                MERGE (c1)-[s:SIMILAR]->(c2)
                SET s.similarity_score = $similarity, s.graph_id = $graph_id
                RETURN s
                """
                
                await neo4j_client.execute_write_query(query, {
                    "chunk1_id": chunk1["id"],
                    "chunk2_id": chunk2["id"],
                    "graph_id": str(graph_id),
                    "similarity": similarity
                })
                
                similarities_created += 1
    
    return similarities_created


async def _create_entity_chunk_similarities(graph_id: UUID) -> int:
    """Create SIMILAR relationships between entities and chunks"""
    
    # Query to find entities and chunks, create similarities based on embeddings
    query = """
    MATCH (e:Entity {graph_id: $graph_id})
    WHERE e.embedding IS NOT NULL
    MATCH (c:DocumentChunk {graph_id: $graph_id})
    WHERE c.embedding IS NOT NULL
    WITH e, c, gds.similarity.cosine(e.embedding, c.embedding) AS similarity
    WHERE similarity >= $threshold
    MERGE (e)-[s:SIMILAR_TO_CHUNK]->(c)
    SET s.similarity_score = similarity, s.graph_id = $graph_id
    RETURN count(s) as similarities_created
    """
    
    try:
        result = await neo4j_client.execute_write_query(query, {
            "graph_id": str(graph_id),
            "threshold": 0.8
        })
        
        return result[0]["similarities_created"] if result else 0
        
    except Exception as e:
        logger.error(f"Failed to create entity-chunk similarities: {e}")
        return 0


async def _detect_communities_and_cleanup(graph_id: UUID) -> int:
    """Detect communities and perform graph cleanup"""
    
    try:
        # Community detection using Louvain algorithm
        community_query = """
        CALL gds.graph.project(
            'tempGraph_' + $graph_id,
            ['Entity', 'DocumentChunk'],
            ['RELATED_TO', 'SIMILAR', 'CONTAINS'],
            {nodeProperties: ['embedding'], relationshipProperties: ['similarity_score']}
        )
        YIELD graphName
        
        CALL gds.louvain.write(graphName, {writeProperty: 'community'})
        YIELD communityCount
        
        CALL gds.graph.drop(graphName)
        
        RETURN communityCount
        """
        
        result = await neo4j_client.execute_write_query(community_query, {
            "graph_id": str(graph_id).replace("-", "_")  # Neo4j graph names can't have hyphens
        })
        
        community_count = result[0]["communityCount"] if result else 0
        
        # Cleanup orphaned nodes
        cleanup_query = """
        MATCH (n {graph_id: $graph_id})
        WHERE NOT (n)-[]-()
        DELETE n
        RETURN count(n) as orphans_removed
        """
        
        cleanup_result = await neo4j_client.execute_write_query(cleanup_query, {
            "graph_id": str(graph_id)
        })
        
        orphans_removed = cleanup_result[0]["orphans_removed"] if cleanup_result else 0
        
        logger.info(f"Detected {community_count} communities, removed {orphans_removed} orphaned nodes")
        
        return community_count
        
    except Exception as e:
        logger.error(f"Community detection failed: {e}")
        return 0

async def _process_embedding_generation_async(task, graph_id: str, user_id: str):
    """Async function to process embedding generation with better error tracking"""
    
    async with async_session_maker() as db:
        try:
            from app.services.enhanced_graph_service import enhanced_graph_service
            from app.services.embedding_service import embedding_service
            from app.services.vector_service import vector_service
            
            logger.info(f"Starting embedding generation for graph {graph_id}")
            
            # ... [previous code for graph verification] ...
            
            # Initialize Neo4j connection
            await neo4j_client.connect()
            
            # Initialize embedding service
            success = await embedding_service.initialize_embeddings(
                provider="openai",
                model="text-embedding-3-small", 
                user_id=user_id
            )
            
            if not success:
                raise Exception("Failed to initialize embedding service in worker")
            
            # Process nodes with better tracking
            batch_size = 10
            nodes_processed = 0
            nodes_succeeded = 0
            nodes_failed = 0
            
            while True:
                # Get nodes without embeddings
                get_nodes_query = """
                MATCH (n)
                WHERE n.graph_id = $graph_id 
                AND n.embedding IS NULL
                AND n.name IS NOT NULL
                RETURN n.id as id, n.name as name, 
                       coalesce(n.description, '') as description
                LIMIT $batch_size
                """
                
                result = await neo4j_client.execute_query(get_nodes_query, {
                    "graph_id": graph_id,
                    "batch_size": batch_size
                })
                
                if not result:
                    break
                
                logger.info(f"Processing batch of {len(result)} nodes")
                
                # Process each node in the batch
                for record in result:
                    try:
                        entity_id = record["id"]
                        name = record["name"]
                        description = record["description"]
                        
                        logger.info(f"🔄 Processing node: '{name}' (ID: {entity_id})")
                        
                        # Create text for embedding
                        text = name
                        if description:
                            text += f" {description}"
                        
                        # Generate embedding
                        embedding = await embedding_service.embed_text(text)
                        logger.info(f"✅ Generated embedding for '{name}': {len(embedding)} dimensions")
                        
                        # Update node
                        update_result = await vector_service.add_entity_embedding(
                            entity_id=entity_id,
                            embedding=embedding,
                            graph_id=UUID(graph_id)
                        )
                        
                        if update_result:
                            nodes_succeeded += 1
                            logger.info(f"✅ Successfully updated '{name}' with embedding")
                        else:
                            nodes_failed += 1
                            logger.error(f"❌ Failed to update '{name}' - node not found or not updated")
                        
                        nodes_processed += 1
                        
                    except Exception as e:
                        nodes_failed += 1
                        logger.error(f"❌ Failed to process node '{record.get('name', record.get('id'))}': {e}")
                
                # If we processed fewer than batch_size, we're done
                if len(result) < batch_size:
                    break
                
                # Update progress
                if task:
                    progress = min(90, 30 + (nodes_processed * 60 / 20))  # Assuming ~20 total nodes
                    task.update_state(state="PROGRESS", meta={
                        "progress": progress, 
                        "status": f"Processed {nodes_processed} nodes ({nodes_succeeded} succeeded, {nodes_failed} failed)",
                        "graph_id": graph_id
                    })
            
            logger.info(f"📊 Final results - Processed: {nodes_processed}, Succeeded: {nodes_succeeded}, Failed: {nodes_failed}")
            
            # Update progress - Completed
            if task:
                task.update_state(state="PROGRESS", meta={
                    "progress": 100, 
                    "status": "Embedding generation completed",
                    "nodes_processed": nodes_processed,
                    "nodes_succeeded": nodes_succeeded,
                    "nodes_failed": nodes_failed,
                    "graph_id": graph_id
                })
            
            return {
                "status": "completed", 
                "nodes_processed": nodes_processed,
                "nodes_succeeded": nodes_succeeded,
                "nodes_failed": nodes_failed,
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

async def _optimize_all_graphs_async(task):
    """Optimize all graphs that haven't been optimized recently"""
    
    async with async_session_maker() as db:
        try:
            # Find graphs that need optimization (no optimization in last 7 days)
            from datetime import timedelta
            week_ago = datetime.utcnow() - timedelta(days=7)
            
            result = await db.execute(
                select(KnowledgeGraph).where(
                    or_(
                        KnowledgeGraph.last_optimized.is_(None),
                        KnowledgeGraph.last_optimized < week_ago
                    )
                )
            )
            graphs = result.scalars().all()
            
            total_optimized = 0
            
            for i, graph in enumerate(graphs):
                try:
                    logger.info(f"Optimizing graph {graph.id} ({i+1}/{len(graphs)})")
                    
                    # Get chunks for similarity processing
                    chunks_query = """
                    MATCH (c:DocumentChunk {graph_id: $graph_id})
                    RETURN c.id as id, c.embedding as embedding
                    """
                    
                    chunks_result = await neo4j_client.execute_query(chunks_query, {
                        "graph_id": str(graph.id)
                    })
                    
                    chunks = [{"id": r["id"], "embedding": r["embedding"]} for r in chunks_result if r["embedding"]]
                    
                    # Create similarities
                    similarity_count = await _create_similarity_relationships(
                        graph_id=graph.id,
                        chunks=chunks,
                        entities_count=graph.node_count
                    )
                    
                    # Detect communities
                    communities_found = await _detect_communities_and_cleanup(graph.id)
                    
                    # Update graph optimization timestamp
                    await db.execute(
                        update(KnowledgeGraph)
                        .where(KnowledgeGraph.id == graph.id)
                        .values(
                            last_optimized=datetime.utcnow(),
                            similarity_relationships=similarity_count,
                            communities_count=communities_found
                        )
                    )
                    
                    total_optimized += 1
                    
                except Exception as e:
                    logger.error(f"Failed to optimize graph {graph.id}: {e}")
                    continue
            
            await db.commit()
            
            return {
                "status": "completed",
                "graphs_processed": len(graphs),
                "graphs_optimized": total_optimized
            }
            
        except Exception as e:
            logger.error(f"Batch optimization failed: {e}")
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
