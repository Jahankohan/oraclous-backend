from typing import Dict, Any, List, Optional
from uuid import UUID
from app.core.neo4j_client import neo4j_client
from app.core.logging import get_logger

logger = get_logger(__name__)

class GraphService:
    """
    Service for direct Neo4j graph operations.
    
    RESPONSIBILITIES:
    - Direct Neo4j node and relationship creation
    - Graph document storage operations  
    - Raw graph data persistence
    - Graph schema operations
    
    DOES NOT:
    - Generate embeddings (delegates to embedding_service)
    - Handle vector indexing (delegates to vector_service)
    - Coordinate enrichment workflows (delegates to enhanced_graph_service)
    """
    
    def __init__(self):
        pass
    
    async def store_graph_documents(
        self, 
        graph_id: str, 
        graph_documents: List[Any]
    ) -> Dict[str, Any]:
        """
        Store graph documents (nodes and relationships) in Neo4j.
        This is the core graph storage operation.
        """
        if not graph_documents:
            return {"nodes_created": 0, "relationships_created": 0}
        
        try:
            nodes_created = 0
            relationships_created = 0
            
            for graph_doc in graph_documents:
                # Store nodes
                for node in graph_doc.nodes:
                    await self._create_node(node, graph_id)
                    nodes_created += 1
                
                # Store relationships
                for rel in graph_doc.relationships:
                    await self._create_relationship(rel, graph_id)
                    relationships_created += 1
            
            logger.info(f"Stored {nodes_created} nodes and {relationships_created} relationships for graph {graph_id}")
            return {
                "nodes_created": nodes_created,
                "relationships_created": relationships_created,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Failed to store graph documents: {e}")
            raise
    
    async def _create_node(self, node: Any, graph_id: str) -> bool:
        try:
            # Normalize labels
            node_type = getattr(node, "type", "Entity")
            if isinstance(node_type, (list, tuple)):
                node_type = ":".join(node_type)

            node_id = getattr(node, "id", None)
            if not node_id:
                logger.warning("Node missing ID, skipping")
                return False

            properties = dict(getattr(node, "properties", {}) or {})

            query = f"""
            MERGE (n:{node_type} {{id: $node_id}})
            ON CREATE SET 
                n.graph_id = $graph_id,
                n.created_at = datetime(),
                n += $properties
            ON MATCH SET 
                n.updated_at = datetime(),
                n += $properties
            """

            await neo4j_client.execute_write_query(query, {
                "node_id": node_id,
                "graph_id": graph_id,
                "properties": properties
            })

            return True

        except Exception as e:
            logger.error(f"Failed to create node {getattr(node, 'id', 'unknown')}: {e}")
            return False

    
    async def _create_relationship(self, relationship: Any, graph_id: str) -> bool:
        try:
            # Extract IDs safely
            source = getattr(relationship, "source", None)
            target = getattr(relationship, "target", None)

            source_id = source.id if hasattr(source, "id") else source
            target_id = target.id if hasattr(target, "id") else target

            if not source_id or not target_id:
                logger.warning("Relationship missing source or target, skipping")
                return False

            rel_type = getattr(relationship, "type", "RELATED_TO")
            properties = dict(getattr(relationship, "properties", {}) or {})

            query = f"""
            MATCH (source {{id: $source_id, graph_id: $graph_id}})
            MATCH (target {{id: $target_id, graph_id: $graph_id}})
            MERGE (source)-[r:{rel_type}]->(target)
            ON CREATE SET 
                r.graph_id = $graph_id,
                r.created_at = datetime(),
                r += $properties
            ON MATCH SET 
                r.updated_at = datetime(),
                r += $properties
            """

            await neo4j_client.execute_write_query(query, {
                "source_id": source_id,
                "target_id": target_id,
                "graph_id": graph_id,
                "properties": properties
            })

            return True

        except Exception as e:
            logger.error(f"Failed to create relationship {getattr(relationship, 'type', 'UNKNOWN')}: {e}")
            return False
        
    async def create_chunk_node(
        self, 
        chunk_id: str, 
        text: str, 
        graph_id: str, 
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Create a chunk node in the graph.
        Used by enhanced_graph_service for chunk storage.
        """
        try:
            properties = {
                "id": chunk_id,
                "text": text,
                "graph_id": graph_id,
                "created_at": "datetime()",
                **(metadata or {})
            }
            
            query = """
            CREATE (c:Chunk $properties)
            """
            
            await neo4j_client.execute_write_query(query, {
                "properties": properties
            })
            
            logger.debug(f"Created chunk node: {chunk_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create chunk node {chunk_id}: {e}")
            return False
    
    async def create_chunk_entity_relationship(
        self,
        chunk_id: str,
        entity_id: str, 
        relationship_type: str,
        graph_id: str
    ) -> bool:
        """
        Create relationship between chunk and entity.
        Used by enhanced_graph_service for linking.
        """
        try:
            query = f"""
            MATCH (c:Chunk {{id: $chunk_id, graph_id: $graph_id}})
            MATCH (e {{id: $entity_id, graph_id: $graph_id}})
            MERGE (c)-[:{relationship_type} {{graph_id: $graph_id}}]->(e)
            """
            
            await neo4j_client.execute_write_query(query, {
                "chunk_id": chunk_id,
                "entity_id": entity_id,
                "graph_id": graph_id
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create chunk-entity relationship: {e}")
            return False
    
    async def create_text_chunks(
        self, 
        graph_id: UUID,
        text_chunks: List[Dict[str, Any]]
    ) -> int:
        """
        Create text chunk nodes with embeddings in Neo4j.
        MOVED FROM vector_service - this is a graph operation.
        """
        if not text_chunks:
            return 0
        
        query = """
        UNWIND $chunks as chunk
        CREATE (c:DocumentChunk {
            id: chunk.id,
            text: chunk.text,
            graph_id: chunk.graph_id,
            chunk_index: chunk.chunk_index,
            char_start: chunk.char_start,
            char_end: chunk.char_end,
            word_count: chunk.word_count,
            embedding: chunk.embedding
        })
        RETURN count(c) as created
        """
        
        try:
            result = await neo4j_client.execute_write_query(query, {
                "chunks": text_chunks
            })
            
            created_count = result[0]["created"] if result else 0
            logger.info(f"Created {created_count} text chunks with embeddings")
            return created_count
            
        except Exception as e:
            logger.error(f"Failed to create text chunks: {e}")
            raise
    
    async def update_entity_embedding(
        self, 
        entity_id: str, 
        embedding: List[float], 
        graph_id: UUID
    ) -> bool:
        """
        Update entity with embedding data.
        MOVED FROM vector_service - this is a graph operation.
        """
        try:
            query = """
            MATCH (e:Entity {id: $entity_id, graph_id: $graph_id})
            SET e.embedding = $embedding
            RETURN e.id as entity_id
            """
            
            result = await neo4j_client.execute_write_query(query, {
                "entity_id": entity_id,
                "embedding": embedding,
                "graph_id": str(graph_id)
            })
            
            if result:
                logger.debug(f"Updated embedding for entity: {entity_id}")
                return True
            else:
                logger.warning(f"Entity not found for embedding update: {entity_id} in graph {graph_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update entity embedding {entity_id}: {e}")
            raise

# Create singleton instance
graph_service = GraphService()
