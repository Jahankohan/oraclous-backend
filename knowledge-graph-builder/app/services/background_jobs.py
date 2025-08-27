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
from app.core.neo4j_client import neo4j_client
from app.models.graph import IngestionJob, KnowledgeGraph
from app.services.entity_extractor import entity_extractor
from app.services.enhanced_graph_service import enhanced_graph_service
from app.services.vector_service import vector_service
from app.services.embedding_service import embedding_service
from app.services.analytics_service import analytics_service
from app.services.task_executor import AsyncTaskExecutor, TaskConcurrencyManager
from app.core.logging import get_logger
import traceback

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
    """Process ingestion job - allows concurrency"""
    return AsyncTaskExecutor.run_async_task(_run_ingestion_async, self, job_id, user_id)

@celery_app.task(bind=True)
def process_embedding_generation_job(self, graph_id: str, user_id: str):
    """Process embedding generation - allows concurrency"""
    return AsyncTaskExecutor.run_async_task(_process_embedding_generation, self, graph_id, user_id)

@celery_app.task(bind=True)
def optimize_all_graphs(self):
    """Optimize all graphs - singleton task"""
    if not TaskConcurrencyManager.should_allow_task('optimize_all_graphs', self.request.id):
        return {
            'status': 'skipped',
            'message': 'Graph optimization already running'
        }
    
    return AsyncTaskExecutor.run_async_task(_optimize_all_graphs_async, self)

@celery_app.task(bind=True)
def reindex_graph_search(self, graph_id: str):
    """Reindex graph for search - allows concurrency"""
    return AsyncTaskExecutor.run_async_task(_reindex_graph_search_async, self, graph_id)

@celery_app.task(bind=True)
def generate_graph_summary(self, graph_id: str):
    """Generate graph summary - allows concurrency"""
    return AsyncTaskExecutor.run_async_task(_generate_graph_summary_async, self, graph_id)

@celery_app.task(bind=True)
def create_similarity_relationships_job(self, graph_id: str):
    """Create similarity relationships - allows concurrency"""
    return AsyncTaskExecutor.run_async_task(_create_similarity_relationships, self, graph_id)

@celery_app.task(bind=True)
def detect_communities_job(self, graph_id: str):
    """Detect communities - allows concurrency"""
    return AsyncTaskExecutor.run_async_task(_detect_communities_and_cleanup, self, graph_id)

@celery_app.task(bind=True)
def update_community_embeddings_job(self, graph_id: str, user_id: str):
    """Update community embeddings - allows concurrency"""
    return AsyncTaskExecutor.run_async_task(_update_community_embeddings_async, self, graph_id, user_id)

@celery_app.task(bind=True)
def refresh_all_communities_job(self, user_id: str):
    """Refresh all communities for user - allows concurrency"""
    return AsyncTaskExecutor.run_async_task(_refresh_all_communities_async, self, user_id)

@celery_app.task(bind=True)
def cleanup_orphaned_data(self, user_id: str):
    """Cleanup orphaned data for user - allows concurrency"""
    return AsyncTaskExecutor.run_async_task(_cleanup_orphaned_data_async, self, user_id)

# ====== ASYNC TASK IMPLEMENTATIONS ======

async def _run_extraction_async(content: str, user_id: str, graph_id: UUID, schema: dict, task=None):
    """
    Pure async extraction function with proper cleanup
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
        
        # Convert content to documents format for extract_from_documents
        documents = [{
            "id": str(graph_id),
            "content": content,
            "title": "Document Content",
            "filename": "content.txt",
            "content_type": "text/plain",
            "summary": "",
            "created_at": None
        }]
        
        # Use new dual graph extraction method
        entity_graph_docs, lexical_chunks = await entity_extractor.extract_from_documents(
            documents=documents,
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
        
        # Create dedicated driver for this task to avoid connection conflicts
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
            
                for relationship in valid_relationships:
                    await _store_relationship_async(driver, relationship, graph_id)
                    relationships_count += 1
                
            logger.info(f"Stored {entities_count} entities and {relationships_count} relationships")
        
        finally:
            await driver.close()
        
        if task:
            task.update_state(state="PROGRESS", meta={
                "progress": 100, 
                "status": f"Completed: {entities_count} entities, {relationships_count} relationships, {chunks_stored} chunks"
            })
        
        return {
            "success": True,
            "entities": entities_count,
            "relationships": relationships_count,
            "chunks": chunks_stored
        }
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        logger.error(traceback.format_exc())
        if task:
            task.update_state(state="FAILURE", meta={"error": str(e)})
        raise

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
    
    # Handle multiple labels properly
    if isinstance(node.type, list):
        # Join multiple labels with colon for Cypher syntax
        labels_str = ":".join([label.replace(" ", "_").replace("-", "_") for label in node.type if label])
    else:
        # Single label
        labels_str = str(node.type).replace(" ", "_").replace("-", "_")
    
    # Fallback to Entity if no valid labels
    if not labels_str:
        labels_str = "Entity"
    
    query = f"""
    MERGE (n:{labels_str} {{id: $node_id}})
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

async def _run_ingestion_async(task, job_id: str, user_id: str):
    """Process ingestion job with proper async setup"""
    job = None
    
    try:
        graph = None
        # Use async database session
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
            
            # Update job status
            await _update_job_status_async(session, job_id, "processing")
            
        task.update_state(state="PROGRESS", meta={"progress": 10, "status": "Starting ingestion"})
        
        # Use the built-in extraction function below
        await _run_extraction_async(
            content=job.source_content,
            user_id=user_id,
            graph_id=job.graph_id,
            schema=graph.schema_config,
            task=task
        )
        
        # Update job status to completed
        async with worker_session_maker() as session:
            await _update_job_status_async(session, job_id, "completed")
        
        logger.info(f"Completed ingestion job {job_id}")
        return {"status": "completed", "job_id": job_id}
        
    except Exception as e:
        logger.error(f"Ingestion job {job_id} failed: {e}")
        
        # Update job status to failed
        try:
            async with worker_session_maker() as session:
                await _update_job_status_async(session, job_id, "failed", error=str(e))
        except Exception as status_error:
            logger.error(f"Failed to update job status: {status_error}")
        
        raise

async def _process_embedding_generation(task, graph_id: str, user_id: str):
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

async def _optimize_all_graphs_async(task):
    """Optimize all graphs that haven't been optimized recently"""
    
    try:
        logger.info("Starting graph optimization for all graphs")
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 10, "status": "Initializing"})
        
        async with worker_session_maker() as db:
            # Find graphs that need optimization (no optimization in last 7 days)
            from datetime import timedelta
            week_ago = datetime.utcnow() - timedelta(days=7)
            
            result = await db.execute(
                select(KnowledgeGraph).where(
                    KnowledgeGraph.status == "active"  # Keep the status filter we added
                ).where(
                    or_(
                        KnowledgeGraph.last_optimized.is_(None),
                        KnowledgeGraph.last_optimized < week_ago
                    )
                )
            )
            graphs = result.scalars().all()
            
            total_graphs = len(graphs)
            total_optimized = 0
            
            logger.info(f"Found {total_graphs} graphs to optimize")
            
            for i, graph in enumerate(graphs):
                try:
                    if task:
                        progress = int(20 + (i / total_graphs) * 70)
                        task.update_state(
                            state="PROGRESS", 
                            meta={
                                "progress": progress, 
                                "status": f"Optimizing graph {graph.name}",
                                "graphs_processed": i,
                                "graphs_optimized": total_optimized
                            }
                        )
                    
                    logger.info(f"Optimizing graph {graph.id} ({i+1}/{total_graphs})")
                    
                    # Call the single graph optimization (based on original logic)
                    optimization_result = await _optimize_single_graph(graph.id, db)
                    
                    if optimization_result:
                        total_optimized += 1
                    
                except Exception as e:
                    logger.error(f"Failed to optimize graph {graph.id}: {e}")
                    continue
            
            await db.commit()
            
            if task:
                task.update_state(state="PROGRESS", meta={"progress": 100, "status": "Completed"})
            
            return {
                'status': 'success',
                'graphs_processed': total_graphs,
                'graphs_optimized': total_optimized,
                'message': f'Optimized {total_optimized} out of {total_graphs} graphs'
            }
        
    except Exception as e:
        logger.error(f"Graph optimization failed: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'graphs_processed': 0,
            'graphs_optimized': 0
        }

async def _reindex_graph_search_async(task, graph_id: str):
    """Reindex search capabilities for a graph"""
    
    try:
        logger.info(f"Starting search reindexing for graph {graph_id}")
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 10, "status": "Connecting to Neo4j"})
        
        await neo4j_client.connect()
        
        # Step 1: Rebuild vector indexes
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 30, "status": "Rebuilding vector indexes"})
        
        # Create/recreate entity embeddings index
        entity_index_query = """
        CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
        FOR (n:__Entity__)
        ON (n.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }}
        """
        
        # Create/recreate chunk embeddings index  
        chunk_index_query = """
        CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
        FOR (n:DocumentChunk)
        ON (n.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }}
        """
        
        await neo4j_client.execute_write_query(entity_index_query)
        await neo4j_client.execute_write_query(chunk_index_query)
        
        # Step 2: Rebuild fulltext indexes
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 60, "status": "Rebuilding fulltext indexes"})
        
        # Create fulltext index for entities
        fulltext_query = """
        CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
        FOR (n:__Entity__)
        ON EACH [n.name, n.description]
        """
        
        await neo4j_client.execute_write_query(fulltext_query)
        
        # Step 3: Count indexed nodes
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 90, "status": "Verifying indexes"})
        
        # Count entities with embeddings
        entity_count_query = """
        MATCH (e:__Entity__ {graph_id: $graph_id})
        WHERE e.embedding IS NOT NULL
        RETURN count(e) as entities_indexed
        """
        
        entity_result = await neo4j_client.execute_query(entity_count_query, {
            "graph_id": graph_id
        })
        entities_indexed = entity_result[0]["entities_indexed"] if entity_result else 0
        
        # Count chunks with embeddings
        chunk_count_query = """
        MATCH (c:DocumentChunk {graph_id: $graph_id})
        WHERE c.embedding IS NOT NULL
        RETURN count(c) as chunks_indexed
        """
        
        chunk_result = await neo4j_client.execute_query(chunk_count_query, {
            "graph_id": graph_id
        })
        chunks_indexed = chunk_result[0]["chunks_indexed"] if chunk_result else 0
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 100, "status": "Reindexing completed"})
        
        logger.info(f"Search reindexing completed for graph {graph_id}: {entities_indexed} entities, {chunks_indexed} chunks indexed")
        
        return {
            'status': 'success',
            'entities_indexed': entities_indexed,
            'chunks_indexed': chunks_indexed,
            'message': f'Reindexed {entities_indexed} entities and {chunks_indexed} chunks'
        }
        
    except Exception as e:
        logger.error(f"Search reindexing failed for graph {graph_id}: {e}")
        return {
            'status': 'error',
            'entities_indexed': 0,
            'chunks_indexed': 0,
            'message': str(e)
        }
    finally:
        try:
            await neo4j_client.disconnect()
        except Exception as e:
            logger.warning(f"Neo4j cleanup warning: {e}")

async def _generate_graph_summary_async(task, graph_id: str):
    """Generate summary for a graph"""
    
    try:
        logger.info(f"Generating summary for graph {graph_id}")
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 10, "status": "Analyzing graph"})
        
        # Initialize connections
        await neo4j_client.connect()
        
        # Get graph statistics
        driver = neo4j_client.driver
        async with driver.session() as session:
            result = await session.run("""
                MATCH (n)
                WHERE n.graph_id = $graph_id
                RETURN count(n) as node_count, 
                       collect(DISTINCT labels(n)[0]) as node_types
            """, {"graph_id": graph_id})
            
            stats = await result.single()
            
            if task:
                task.update_state(state="PROGRESS", meta={"progress": 50, "status": "Generating summary"})
            
            # Generate summary (implement your summary logic)
            summary = {
                "node_count": stats["node_count"],
                "node_types": stats["node_types"],
                "generated_at": datetime.utcnow().isoformat()
            }
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 100, "status": "Completed"})
        
        return {
            'status': 'success',
            'summary': summary,
            'message': f'Generated summary for graph {graph_id}'
        }
        
    except Exception as e:
        logger.error(f"Summary generation failed for graph {graph_id}: {e}")
        return {
            'status': 'error',
            'message': str(e)
        }
    finally:
        try:
            await neo4j_client.disconnect()
        except Exception as e:
            logger.warning(f"Neo4j cleanup warning: {e}")

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

async def _refresh_all_communities_async(task, user_id: str):
    """Async function to refresh communities for all user graphs"""
    
    async with worker_session_maker() as db:
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

async def _cleanup_orphaned_data_async(task, graph_id: str):
    """Clean up orphaned nodes and relationships in a graph"""
    
    try:
        logger.info(f"Starting orphaned data cleanup for graph {graph_id}")
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 10, "status": "Connecting to Neo4j"})
        
        await neo4j_client.connect()
        
        # Step 1: Remove orphaned Entity nodes (nodes with no relationships)
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 30, "status": "Removing orphaned entity nodes"})
        
        orphaned_entities_query = """
        MATCH (n:`__Entity__` {graph_id: $graph_id})
        WHERE NOT (n)-[]-()
        DELETE n
        RETURN count(n) as orphaned_entities_removed
        """
        
        orphaned_entities_result = await neo4j_client.execute_write_query(orphaned_entities_query, {
            "graph_id": graph_id
        })
        orphaned_entities = orphaned_entities_result[0]["orphaned_entities_removed"] if orphaned_entities_result else 0
        
        # Step 2: Remove orphaned DocumentChunk nodes (chunks not connected to anything)
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 50, "status": "Removing orphaned document chunks"})
        
        orphaned_chunks_query = """
        MATCH (c:DocumentChunk {graph_id: $graph_id})
        WHERE NOT (c)-[]-()
        DELETE c
        RETURN count(c) as orphaned_chunks_removed
        """
        
        orphaned_chunks_result = await neo4j_client.execute_write_query(orphaned_chunks_query, {
            "graph_id": graph_id
        })
        orphaned_chunks = orphaned_chunks_result[0]["orphaned_chunks_removed"] if orphaned_chunks_result else 0
        
        # Step 3: Remove duplicate RELATED_TO relationships
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 70, "status": "Removing duplicate relationships"})
        
        duplicate_rels_query = """
        MATCH (a:`__Entity__` {graph_id: $graph_id})-[r1:RELATED_TO]->(b:`__Entity__` {graph_id: $graph_id})
        MATCH (a)-[r2:RELATED_TO]->(b)
        WHERE id(r1) < id(r2) AND r1.graph_id = $graph_id AND r2.graph_id = $graph_id
        DELETE r2
        RETURN count(r2) as duplicate_relationships_removed
        """
        
        duplicate_rels_result = await neo4j_client.execute_write_query(duplicate_rels_query, {
            "graph_id": graph_id
        })
        duplicate_rels = duplicate_rels_result[0]["duplicate_relationships_removed"] if duplicate_rels_result else 0
        
        # Step 4: Remove duplicate SIMILAR relationships
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 85, "status": "Removing duplicate similarity relationships"})
        
        duplicate_similar_query = """
        MATCH (a {graph_id: $graph_id})-[r1:SIMILAR]->(b {graph_id: $graph_id})
        MATCH (a)-[r2:SIMILAR]->(b)
        WHERE id(r1) < id(r2) AND r1.graph_id = $graph_id AND r2.graph_id = $graph_id
        DELETE r2
        RETURN count(r2) as duplicate_similar_removed
        """
        
        duplicate_similar_result = await neo4j_client.execute_write_query(duplicate_similar_query, {
            "graph_id": graph_id
        })
        duplicate_similar = duplicate_similar_result[0]["duplicate_similar_removed"] if duplicate_similar_result else 0
        
        # Step 5: Get final statistics
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 95, "status": "Updating graph statistics"})
        
        # Count remaining nodes
        stats_query = """
        MATCH (n {graph_id: $graph_id})
        RETURN 
            count(CASE WHEN n:`__Entity__` THEN 1 END) as entities_count,
            count(CASE WHEN n:DocumentChunk THEN 1 END) as chunks_count,
            count(CASE WHEN n:Community THEN 1 END) as communities_count
        """
        
        stats_result = await neo4j_client.execute_query(stats_query, {"graph_id": graph_id})
        stats = stats_result[0] if stats_result else {}
        
        # Count relationships
        rel_stats_query = """
        MATCH ()-[r {graph_id: $graph_id}]->()
        RETURN 
            count(CASE WHEN type(r) = 'RELATED_TO' THEN 1 END) as relationships_count,
            count(CASE WHEN type(r) = 'SIMILAR' THEN 1 END) as similarity_relationships_count
        """
        
        rel_stats_result = await neo4j_client.execute_query(rel_stats_query, {"graph_id": graph_id})
        rel_stats = rel_stats_result[0] if rel_stats_result else {}
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 100, "status": "Cleanup completed"})
        
        total_removed = orphaned_entities + orphaned_chunks + duplicate_rels + duplicate_similar
        
        logger.info(f"Orphaned data cleanup completed for graph {graph_id}: "
                   f"{orphaned_entities} orphaned entities, {orphaned_chunks} orphaned chunks, "
                   f"{duplicate_rels} duplicate relationships, {duplicate_similar} duplicate similarities removed")
        
        return {
            'status': 'success',
            'orphaned_entities_removed': orphaned_entities,
            'orphaned_chunks_removed': orphaned_chunks,
            'duplicate_relationships_removed': duplicate_rels,
            'duplicate_similarities_removed': duplicate_similar,
            'total_items_removed': total_removed,
            'final_stats': {
                'entities_count': stats.get('entities_count', 0),
                'chunks_count': stats.get('chunks_count', 0),
                'communities_count': stats.get('communities_count', 0),
                'relationships_count': rel_stats.get('relationships_count', 0),
                'similarity_relationships_count': rel_stats.get('similarity_relationships_count', 0)
            },
            'message': f'Removed {total_removed} orphaned/duplicate items from graph {graph_id}'
        }
        
    except Exception as e:
        logger.error(f"Orphaned data cleanup failed for graph {graph_id}: {e}")
        return {
            'status': 'error',
            'orphaned_entities_removed': 0,
            'orphaned_chunks_removed': 0,
            'duplicate_relationships_removed': 0,
            'duplicate_similarities_removed': 0,
            'total_items_removed': 0,
            'message': str(e)
        }
    finally:
        try:
            await neo4j_client.disconnect()
        except Exception as e:
            logger.warning(f"Neo4j cleanup warning: {e}")

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

async def _create_similarity_relationships(
    graph_id: UUID,
    chunks: List[Dict[str, Any]]) -> int:
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

# ====== HELPER FUNCTIONS ======

async def _update_job_status_async(session: AsyncSession, job_id: str, status: str, 
                                  error: str = None, progress: int = None, 
                                  entities: int = None, relationships: int = None, 
                                  chunks: int = None):
    """Update job status asynchronously"""
    
    result = await session.execute(select(IngestionJob).where(IngestionJob.id == UUID(job_id)))
    job = result.scalar_one_or_none()
    
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
        if chunks is not None and hasattr(job, 'processed_chunks'):
            job.processed_chunks = chunks
        
        if status == "processing" and progress == 10:
            job.started_at = datetime.utcnow()
        elif status == "completed":
            job.completed_at = datetime.utcnow()
        
        await session.commit()

async def _optimize_single_graph(graph_id: UUID, session: AsyncSession):
    """Optimize a single graph based on original implementation"""
    
    try:
        # Get chunks for similarity processing
        chunks_query = """
            MATCH (c:DocumentChunk {graph_id: $graph_id})
            RETURN c.id as id, c.embedding as embedding
        """
                    
        chunks_result = await neo4j_client.execute_query(chunks_query, {
            "graph_id": str(graph_id)
        })
                    
        chunks = [{"id": r["id"], "embedding": r["embedding"]} for r in chunks_result if r["embedding"]]
                    
        # Create similarities
        similarity_count = await _create_similarity_relationships(
            graph_id=graph_id,
            chunks=chunks
        )
                    
        # Detect communities
        communities_found = await _detect_communities_and_cleanup(graph_id)
                    
        # Update graph optimization timestamp
        await session.execute(
            update(KnowledgeGraph)
            .where(KnowledgeGraph.id == graph_id)
            .values(
                last_optimized=datetime.utcnow(),
                similarity_relationships=similarity_count,
                communities_count=communities_found
            )
        )

        return True

    except Exception as e:
        logger.error(f"Failed to optimize graph {graph_id}: {e}")
        return False
    finally:
        try:
            await neo4j_client.disconnect()
        except Exception as e:
            logger.warning(f"Neo4j cleanup warning during optimization: {e}")
