from celery import Celery
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, or_
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
from app.services.analytics_service import analytics_service
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
    import threading
    
    try:
        # Check if there's already a running event loop
        try:
            loop = asyncio.get_running_loop()
            logger.warning("Event loop already running, using thread-based execution")
            
            # Use thread-based execution to avoid loop conflicts
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(_process_embedding_generation_sync(self, graph_id, user_id))
                )
                return future.result()
                
        except RuntimeError:
            # No running loop, safe to create new one
            return asyncio.run(_process_embedding_generation_sync(self, graph_id, user_id))
            
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return {"status": "error", "message": str(e)}

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
            
            # Convert text to documents format for extract_from_documents
            documents = [{
                "id": str(job.id),
                "content": job.source_content,
                "title": f"Ingestion Job {job.id}",
                "filename": f"job_{job.id}.txt",
                "content_type": "text/plain",
                "summary": "",
                "created_at": job.created_at.isoformat() if job.created_at else None
            }]
            
            entity_graph_docs, lexical_chunks = await entity_extractor.extract_from_documents(
                documents=documents,
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
    """Generate embeddings for ALL node types: Documents, Chunks, and Entities (__Entity__)"""
    
    try:
        logger.info(f"Starting comprehensive embedding generation for graph {graph_id}")
        
        # Initialize Neo4j connection fresh in this event loop
        from app.core.neo4j_client import Neo4jClient
        neo4j_client_local = Neo4jClient()
        await neo4j_client_local.connect()
        
        # Initialize embedding service
        from app.services.embedding_service import EmbeddingService
        embedding_service_local = EmbeddingService()
        
        # Get OpenAI credentials with fallback
        try:
            success = await embedding_service_local.initialize_embeddings(
                provider="openai",
                model="text-embedding-3-small", 
                user_id=user_id
            )
            
            if not success:
                import os
                openai_key = os.getenv('OPENAI_API_KEY')
                if openai_key:
                    success = await embedding_service_local.initialize_embeddings(
                        provider="openai",
                        model="text-embedding-3-small",
                        user_id=user_id,
                        api_key=openai_key
                    )
        
        except Exception as cred_error:
            logger.error(f"Error retrieving credentials: {cred_error}")
            import os
            openai_key = os.getenv('OPENAI_API_KEY')
            if not openai_key:
                return {"status": "error", "message": "OpenAI API key not available"}
            
            from app.services.embedding_service import EmbeddingService
            embedding_service_local = EmbeddingService()
            success = await embedding_service_local.initialize_embeddings(
                provider="openai",
                model="text-embedding-3-small",
                user_id=user_id,
                api_key=openai_key
            )
        
        if not success:
            return {"status": "error", "message": "Failed to initialize embedding service"}
        
        logger.info(f"Embeddings initialized: openai - text-embedding-3-small (dim: {embedding_service_local.dimension})")
        
        # Process embeddings for all node types
        total_processed = 0
        
        # 1. Process Document nodes
        doc_count = await _process_document_embeddings(neo4j_client_local, embedding_service_local, graph_id)
        total_processed += doc_count
        
        # 2. Process DocumentChunk nodes  
        chunk_count = await _process_chunk_embeddings(neo4j_client_local, embedding_service_local, graph_id)
        total_processed += chunk_count
        
        # 3. Process Entity nodes (using __Entity__ base type)
        entity_count = await _process_entity_embeddings(neo4j_client_local, embedding_service_local, graph_id)
        total_processed += entity_count
        
        # 4. Create vector indexes if needed
        try:
            from app.services.vector_service import VectorService
            vector_service_local = VectorService()
            await vector_service_local.create_vector_indexes(
                dimension=embedding_service_local.dimension
            )
        except Exception as e:
            logger.warning(f"Failed to create vector indexes: {e}")
        
        logger.info(f"Comprehensive embedding generation completed. Documents: {doc_count}, Chunks: {chunk_count}, Entities: {entity_count}, Total: {total_processed}")
        
        return {
            "status": "completed",
            "documents_processed": doc_count,
            "chunks_processed": chunk_count,
            "entities_processed": entity_count,
            "total_processed": total_processed,
            "message": f"Successfully generated embeddings for {doc_count} documents, {chunk_count} chunks, and {entity_count} entities"
        }
        
    except Exception as e:
        error_msg = f"Background embedding generation failed for graph {graph_id}: {e}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}
    
    finally:
        # Cleanup connections
        try:
            if 'neo4j_client_local' in locals():
                await neo4j_client_local.close()
        except Exception as e:
            logger.warning(f"Failed to close Neo4j connection: {e}")

async def _process_document_embeddings(neo4j_client, embedding_service, graph_id: str) -> int:
    """Process Document node embeddings"""
    batch_size = 10
    processed = 0
    
    while True:
        # Get documents without embeddings
        get_documents_query = """
        MATCH (d:Document)
        WHERE d.graph_id = $graph_id 
        AND (d.embedding IS NULL OR d.has_embedding = false)
        AND (d.title IS NOT NULL OR d.summary IS NOT NULL)
        RETURN d.id as id, 
               coalesce(d.title, '') as title,
               coalesce(d.summary, '') as summary,
               coalesce(d.filename, '') as filename
        LIMIT $batch_size
        """
        
        result = await neo4j_client.execute_query(get_documents_query, {
            "graph_id": graph_id,
            "batch_size": batch_size
        })
        
        if not result:
            break
            
        # Generate embeddings for documents
        texts = []
        doc_ids = []
        
        for doc in result:
            # Combine title, summary, filename for document embedding
            text_parts = [doc['title'], doc['summary'], doc['filename']]
            text = " ".join([part for part in text_parts if part.strip()])
            
            if text.strip():
                texts.append(text)
                doc_ids.append(doc['id'])
        
        if texts:
            embeddings = await embedding_service.embed_documents(texts)
            
            # Update documents with embeddings
            for doc_id, embedding in zip(doc_ids, embeddings):
                update_query = """
                MATCH (d:Document {id: $doc_id, graph_id: $graph_id})
                SET d.embedding = $embedding, d.has_embedding = true
                """
                
                await neo4j_client.execute_write_query(update_query, {
                    "doc_id": doc_id,
                    "graph_id": graph_id,
                    "embedding": embedding
                })
            
            processed += len(embeddings)
            logger.info(f"Updated {len(embeddings)} documents with embeddings. Total documents: {processed}")
        
        if len(result) < batch_size:
            break
    
    return processed

async def _process_chunk_embeddings(neo4j_client, embedding_service, graph_id: str) -> int:
    """Process DocumentChunk node embeddings"""
    batch_size = 5  # Smaller batch for chunks (they're larger)
    processed = 0
    
    while True:
        # Get chunks without embeddings
        get_chunks_query = """
        MATCH (c:DocumentChunk)
        WHERE c.graph_id = $graph_id 
        AND (c.embedding IS NULL OR c.has_embedding = false)
        AND c.text IS NOT NULL
        RETURN c.id as id, 
               c.text as text
        LIMIT $batch_size
        """
        
        result = await neo4j_client.execute_query(get_chunks_query, {
            "graph_id": graph_id,
            "batch_size": batch_size
        })
        
        if not result:
            break
            
        # Generate embeddings for chunks
        texts = []
        chunk_ids = []
        
        for chunk in result:
            # Use chunk text directly (truncate if too long)
            text = chunk['text'][:4000]  # Truncate for token limits
            texts.append(text)
            chunk_ids.append(chunk['id'])
        
        embeddings = await embedding_service.embed_documents(texts)
        
        # Update chunks with embeddings
        for chunk_id, embedding in zip(chunk_ids, embeddings):
            update_query = """
            MATCH (c:DocumentChunk {id: $chunk_id, graph_id: $graph_id})
            SET c.embedding = $embedding, c.has_embedding = true
            """
            
            await neo4j_client.execute_write_query(update_query, {
                "chunk_id": chunk_id,
                "graph_id": graph_id,
                "embedding": embedding
            })
        
        processed += len(embeddings)
        logger.info(f"Updated {len(embeddings)} chunks with embeddings. Total chunks: {processed}")
        
        if len(result) < batch_size:
            break
    
    return processed

async def _process_entity_embeddings(neo4j_client, embedding_service, graph_id: str) -> int:
    """Process Entity node embeddings using __Entity__ base type"""
    batch_size = 10
    processed = 0
    
    while True:
        # Get __Entity__ nodes without embeddings
        get_entities_query = """
        MATCH (e:`__Entity__`)
        WHERE e.graph_id = $graph_id 
        AND (e.embedding IS NULL OR e.has_embedding = false)
        AND e.name IS NOT NULL
        RETURN e.id as id, e.name as name, 
               coalesce(e.description, '') as description,
               coalesce(e.entity_type, 'Entity') as entity_type
        LIMIT $batch_size
        """
        
        result = await neo4j_client.execute_query(get_entities_query, {
            "graph_id": graph_id,
            "batch_size": batch_size
        })
        
        if not result:
            break
            
        # Generate embeddings for entities
        texts = []
        entity_ids = []
        
        for entity in result:
            # Combine name, description, and type for entity embedding
            text_parts = [entity['name']]
            if entity['description']:
                text_parts.append(entity['description'])
            text_parts.append(f"Type: {entity['entity_type']}")
            
            text = " ".join(text_parts)
            texts.append(text)
            entity_ids.append(entity['id'])
        
        embeddings = await embedding_service.embed_documents(texts)
        
        # Update entities with embeddings
        for entity_id, embedding in zip(entity_ids, embeddings):
            update_query = """
            MATCH (e:`__Entity__` {id: $entity_id, graph_id: $graph_id})
            SET e.embedding = $embedding, e.has_embedding = true
            """
            
            await neo4j_client.execute_write_query(update_query, {
                "entity_id": entity_id,
                "graph_id": graph_id,
                "embedding": embedding
            })
        
        processed += len(embeddings)
        logger.info(f"Updated {len(embeddings)} entities with embeddings. Total entities: {processed}")
        
        if len(result) < batch_size:
            break
    
    return processed

async def _create_similarity_relationships(
    graph_id: UUID,
    chunks: List[Dict[str, Any]],
    entities_count: int
) -> int:
    """Create SIMILAR relationships between semantically related chunks and entities"""
    
    # Try to initialize embedding service if not already initialized
    if not embedding_service.is_initialized():
        logger.info("Initializing embedding service for similarity relationships")
        success = await embedding_service.initialize_embeddings(
            provider="openai",
            model="text-embedding-3-small"
        )
        
        if not success:
            logger.warning("Failed to initialize embedding service, skipping similarity relationships")
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
    """Create persistent community nodes and perform graph cleanup"""
    
    try:
        logger.info(f"Starting persistent community creation for graph {graph_id}")
        
        # Step 1: Create persistent community nodes using analytics service
        # This replaces the old temporary GDS approach with full persistence
        try:
            persistence_result = await analytics_service.create_community_nodes(graph_id)
            
            logger.info(f"Community persistence results: {persistence_result}")
            
            persistent_communities = persistence_result.get("communities_created", 0)
            persistent_relationships = persistence_result.get("relationships_created", 0)
            
            if persistent_communities > 0:
                logger.info(f"Successfully created {persistent_communities} persistent community nodes "
                           f"with {persistent_relationships} IN_COMMUNITY relationships")
            else:
                logger.warning("No persistent communities were created")
                
        except Exception as persistence_error:
            logger.error(f"Failed to create persistent community nodes: {persistence_error}")
            # Continue with cleanup even if persistence fails
            persistent_communities = 0
            
        # Step 2: Cleanup orphaned nodes
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
        
        logger.info(f"Persistent community creation completed: {persistent_communities} communities created, "
                   f"{orphans_removed} orphaned nodes removed")
        
        return persistent_communities
        
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
            logger.error(f"Failed to optimize graphs: {e}")
            await db.rollback()
            return {
                "status": "error",
                "message": str(e),
                "graphs_processed": 0,
                "graphs_optimized": 0
            }


@celery_app.task(bind=True)
def create_persistent_communities_task(self, graph_id: str, user_id: str):
    """Background task to create persistent community nodes for a specific graph"""
    import asyncio
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            _create_persistent_communities_async(self, UUID(graph_id), user_id)
        )
        return result
    finally:
        loop.close()


async def _create_persistent_communities_async(task, graph_id: UUID, user_id: str):
    """Async function to create persistent community nodes"""
    
    async with async_session_maker() as db:
        try:
            # Update task status
            task.update_state(
                state="PROGRESS",
                meta={"status": "Starting community node creation", "progress": 0}
            )
            
            logger.info(f"Creating persistent communities for graph {graph_id}")
            
            # Step 1: Create community nodes
            task.update_state(
                state="PROGRESS", 
                meta={"status": "Running community detection", "progress": 25}
            )
            
            community_result = await analytics_service.create_community_nodes(graph_id)
            
            # Step 2: Generate community embeddings (placeholder for now)
            task.update_state(
                state="PROGRESS",
                meta={"status": "Generating community embeddings", "progress": 75}
            )
            
            # TODO: Uncomment when embedding service is ready
            # embedding_result = await analytics_service.generate_community_embeddings(graph_id)
            
            # Step 3: Update graph status
            task.update_state(
                state="PROGRESS",
                meta={"status": "Updating graph status", "progress": 90}
            )
            
            # Update the knowledge graph record
            await db.execute(
                update(KnowledgeGraph)
                .where(KnowledgeGraph.id == graph_id)
                .values(
                    communities_enabled=True,
                    last_optimized=datetime.utcnow()
                )
            )
            
            await db.commit()
            
            task.update_state(
                state="SUCCESS",
                meta={
                    "status": "Community creation completed",
                    "progress": 100,
                    "communities_created": community_result.get("communities_created", 0),
                    "relationships_created": community_result.get("relationships_created", 0),
                    "algorithm_used": community_result.get("algorithm_used", "unknown")
                }
            )
            
            return community_result
            
        except Exception as e:
            logger.error(f"Community creation task failed for graph {graph_id}: {e}")
            
            task.update_state(
                state="FAILURE",
                meta={"status": f"Failed: {str(e)}", "error": str(e)}
            )
            
            raise


@celery_app.task(bind=True)
def update_community_embeddings_task(self, graph_id: str, user_id: str):
    """Background task to update community embeddings for better search"""
    import asyncio
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            _update_community_embeddings_async(self, UUID(graph_id), user_id)
        )
        return result
    finally:
        loop.close()


async def _update_community_embeddings_async(task, graph_id: UUID, user_id: str):
    """Async function to update community embeddings"""
    
    try:
        task.update_state(
            state="PROGRESS",
            meta={"status": "Starting embedding generation", "progress": 0}
        )
        
        logger.info(f"Updating community embeddings for graph {graph_id}")
        
        # Generate embeddings for community summaries
        # TODO: Implement when embedding service is integrated
        # embedding_result = await analytics_service.generate_community_embeddings(graph_id)
        
        # For now, return a placeholder result
        embedding_result = {
            "embeddings_generated": 0,
            "message": "Embedding generation not yet implemented",
            "graph_id": str(graph_id)
        }
        
        task.update_state(
            state="SUCCESS",
            meta={
                "status": "Embedding update completed",
                "progress": 100,
                "embeddings_generated": embedding_result.get("embeddings_generated", 0)
            }
        )
        
        return embedding_result
        
    except Exception as e:
        logger.error(f"Community embedding update failed for graph {graph_id}: {e}")
        
        task.update_state(
            state="FAILURE",
            meta={"status": f"Failed: {str(e)}", "error": str(e)}
        )
        
        raise


@celery_app.task(bind=True) 
def refresh_all_communities_task(self, user_id: str):
    """Background task to refresh communities for all graphs"""
    import asyncio
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            _refresh_all_communities_async(self, user_id)
        )
        return result
    finally:
        loop.close()


async def _refresh_all_communities_async(task, user_id: str):
    """Async function to refresh communities for all user graphs"""
    
    async with async_session_maker() as db:
        try:
            # Get all graphs for the user
            result = await db.execute(
                select(KnowledgeGraph)
                .where(KnowledgeGraph.user_id == user_id)
                .where(KnowledgeGraph.status == "completed")
            )
            
            graphs = result.scalars().all()
            
            if not graphs:
                return {
                    "status": "completed",
                    "message": "No graphs found for user",
                    "graphs_processed": 0
                }
            
            task.update_state(
                state="PROGRESS",
                meta={"status": f"Processing {len(graphs)} graphs", "progress": 0}
            )
            
            communities_created = 0
            graphs_processed = 0
            
            for i, graph in enumerate(graphs):
                try:
                    logger.info(f"Refreshing communities for graph {graph.id}")
                    
                    # Create persistent community nodes
                    community_result = await analytics_service.create_community_nodes(graph.id)
                    
                    communities_created += community_result.get("communities_created", 0)
                    graphs_processed += 1
                    
                    # Update progress
                    progress = int(((i + 1) / len(graphs)) * 100)
                    task.update_state(
                        state="PROGRESS",
                        meta={
                            "status": f"Processed {i + 1}/{len(graphs)} graphs",
                            "progress": progress,
                            "communities_created": communities_created
                        }
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to refresh communities for graph {graph.id}: {e}")
                    continue
            
            final_result = {
                "status": "completed",
                "graphs_processed": graphs_processed,
                "total_communities_created": communities_created,
                "user_id": user_id
            }
            
            task.update_state(
                state="SUCCESS",
                meta=final_result
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"Refresh all communities failed for user {user_id}: {e}")
            
            task.update_state(
                state="FAILURE", 
                meta={"status": f"Failed: {str(e)}", "error": str(e)}
            )
            
            raise
            
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
