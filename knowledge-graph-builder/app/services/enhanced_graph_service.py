from typing import Dict, Any, List
from langchain_core.documents import Document
from app.core.logging import get_logger
from app.services.embedding_service import embedding_service
from app.services.vector_service import vector_service
from app.services.graph_service import graph_service
from uuid import UUID

logger = get_logger(__name__)

class EnhancedGraphService:
    """
    Service for graph data enrichment and coordinated storage operations.
    
    RESPONSIBILITIES:
    - Enrich graph documents with embeddings and metadata
    - Coordinate atomic storage operations across services
    - Validate and preprocess graph data before storage
    - Handle complex graph creation workflows
    
    DOES NOT:
    - Perform direct Neo4j operations (delegates to graph_service)
    - Generate embeddings (delegates to embedding_service)
    - Handle vector indexing (delegates to vector_service)
    """
    
    def __init__(self):
        self.neo4j_client = None  # Will be injected
    
    async def store_complete_graph(
        self, 
        graph_documents: List[Any], 
        chunks: List[Document], 
        graph_id: UUID,
        user_id = str,
    ) -> Dict[str, Any]:
        """
        Orchestrate complete graph storage with enrichment.
        
        This is the main coordination method that handles the full pipeline:
        1. Enrich entities with embeddings
        2. Store enriched graph data
        3. Store chunks with embeddings  
        4. Create vector indexes
        5. Link chunks to entities
        """
        try:
            logger.info(f"Starting complete graph storage for graph_id: {graph_id}")

            if not embedding_service.is_initialized():
                await embedding_service.initialize_embeddings(
                    provider="openai",
                    user_id=user_id
                )
            
            # Phase 1: Enrich and store entities
            entity_result = await self.enrich_and_store_entities(graph_documents, str(graph_id))
            
            # Phase 2: Store chunks with embeddings
            chunk_ids = await self.store_chunks_with_embeddings(chunks, str(graph_id))
            
            # Phase 3: Link chunks to entities (if both exist)
            if chunk_ids and graph_documents:
                await self._link_chunks_to_entities(chunk_ids, graph_documents, str(graph_id))
            
            # Phase 4: Ensure vector indexes exist
            await vector_service.create_vector_indexes()
            
            result = {
                "status": "success",
                "graph_id": str(graph_id),
                "entities_processed": entity_result.get("entities_count", 0),
                "chunks_processed": len(chunk_ids),
                "relationships_created": entity_result.get("relationships_count", 0)
            }
            
            logger.info(f"Successfully completed graph storage: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to store complete graph for {graph_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "graph_id": str(graph_id)
            }
    
    async def enrich_and_store_entities(
        self, 
        graph_documents: List[Any], 
        graph_id: str
    ) -> Dict[str, Any]:
        """
        Enrich graph documents with embeddings and store them.
        
        ENRICHMENT PROCESS:
        1. Extract entity text for embedding
        2. Generate embeddings for entities
        3. Add embeddings to entity properties
        4. Delegate storage to graph_service
        5. Delegate vector storage to vector_service
        """
        if not graph_documents:
            return {"entities_count": 0, "relationships_count": 0}
        
        try:
            entities_count = 0
            relationships_count = 0
            
            # Process each graph document
            for graph_doc in graph_documents:
                # Enrich nodes with embeddings
                for node in graph_doc.nodes:
                    # Generate embedding for entity
                    entity_text = f"{node.id} {getattr(node, 'type', '')} {str(getattr(node, 'properties', {}))}"
                    embedding = await embedding_service.embed_text(entity_text)
                    
                    # Add embedding to node properties
                    if not hasattr(node, 'properties') or not node.properties:
                        node.properties = {}
                    node.properties["embedding"] = embedding
                    node.properties["graph_id"] = graph_id
                    entities_count += 1
                
                # Add graph_id to relationships
                for rel in graph_doc.relationships:
                    if not hasattr(rel, 'properties') or not rel.properties:
                        rel.properties = {}
                    rel.properties["graph_id"] = graph_id
                    relationships_count += 1
            
            # Delegate actual storage to graph_service
            storage_result = await graph_service.store_graph_documents(graph_id, graph_documents)
            
            return {
                "entities_count": entities_count,
                "relationships_count": relationships_count,
                "storage_result": storage_result
            }
            
        except Exception as e:
            logger.error(f"Failed to enrich and store entities: {e}")
            raise
    
    async def store_chunks_with_embeddings(
        self, 
        chunks: List[Document], 
        graph_id: str
    ) -> List[str]:
        """
        Store text chunks with embeddings.
        
        PROCESS:
        1. Generate embeddings for chunk text
        2. Create chunk nodes via graph_service
        3. Store chunk embeddings via vector_service
        """
        if not chunks:
            return []
        
        try:
            # Generate embeddings for all chunks
            chunk_texts = [chunk.page_content for chunk in chunks]
            embeddings = await embedding_service.embed_documents(chunk_texts)
            
            # Create chunk nodes with embeddings via graph_service
            chunk_ids = await self._create_chunk_nodes_with_embeddings(chunks, embeddings, graph_id)
            
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Failed to store chunks with embeddings: {e}")
            raise
    
    async def _create_chunk_nodes_with_embeddings(
        self, 
        chunks: List[Document], 
        embeddings: List[List[float]], 
        graph_id: str
    ) -> List[str]:
        """
        Create chunk nodes with embeddings in the graph.
        DELEGATES to graph_service for actual Neo4j operations.
        """
        chunk_ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{graph_id}_{i}"
            
            # Prepare chunk metadata with embedding
            metadata = getattr(chunk, 'metadata', {})
            metadata['embedding'] = embeddings[i]
            
            # Delegate to graph_service for actual node creation
            await graph_service.create_chunk_node(
                chunk_id=chunk_id,
                text=chunk.page_content,
                graph_id=graph_id,
                metadata=metadata
            )
            
            chunk_ids.append(chunk_id)
        
        return chunk_ids
    
    async def _link_chunks_to_entities(
        self, 
        chunk_ids: List[str], 
        graph_documents: List[Any], 
        graph_id: str
    ) -> None:
        """
        Create relationships between chunks and entities.
        DELEGATES to graph_service for actual relationship creation.
        """
        try:
            # Simple linking strategy: connect chunks to all entities in the same graph
            entity_ids = []
            for graph_doc in graph_documents:
                for node in graph_doc.nodes:
                    entity_ids.append(node.id)
            
            # Delegate relationship creation to graph_service
            for chunk_id in chunk_ids:
                for entity_id in entity_ids:
                    await graph_service.create_chunk_entity_relationship(
                        chunk_id=chunk_id,
                        entity_id=entity_id,
                        relationship_type="CONTAINS",
                        graph_id=graph_id
                    )
                    
        except Exception as e:
            logger.error(f"Failed to link chunks to entities: {e}")
            # Don't raise - linking is optional
            pass

# Create singleton instance
enhanced_graph_service = EnhancedGraphService()