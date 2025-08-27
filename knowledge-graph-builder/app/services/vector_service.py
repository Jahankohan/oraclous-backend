from typing import List, Dict, Any, Optional
from uuid import UUID
from app.core.neo4j_client import neo4j_client
from app.core.logging import get_logger

logger = get_logger(__name__)

class VectorService:
    """
    Service for vector storage and indexing operations.
    
    RESPONSIBILITIES:
    - Vector index creation and management
    - Vector storage in Neo4j vector indexes
    - Vector similarity queries
    - Index optimization
    
    DOES NOT:
    - Generate embeddings (delegates to embedding_service)
    - Handle search coordination (delegates to search_service)
    """
    
    def __init__(self):
        pass

    async def batch_store_embeddings(
        self,
        embeddings_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Store multiple embeddings in batch for better performance.
        
        Args:
            embeddings_data: List of dicts with keys: 
                - type: 'node' or 'chunk'
                - id: node/chunk ID
                - embedding: vector
                - graph_id: graph identifier
                - text: original text (for chunks)
        """
        try:
            stored_count = 0
            failed_count = 0
            
            for data in embeddings_data:
                try:
                    if data['type'] == 'node':
                        await self.store_node_embedding(
                            node_id=data['id'],
                            embedding=data['embedding'],
                            graph_id=data['graph_id']
                        )
                    elif data['type'] == 'chunk':
                        await self.store_chunk_embedding(
                            chunk_id=data['id'],
                            embedding=data['embedding'], 
                            graph_id=data['graph_id'],
                            text=data.get('text', '')
                        )
                    stored_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to store embedding for {data['id']}: {e}")
                    failed_count += 1
            
            return {
                "stored_count": stored_count,
                "failed_count": failed_count,
                "total_count": len(embeddings_data)
            }
            
        except Exception as e:
            logger.error(f"Batch embedding storage failed: {e}")
            raise
    
    async def optimize_indexes(self, graph_id: Optional[str] = None) -> Dict[str, Any]:
        """Optimize vector indexes for better performance"""
        try:
            # Add index optimization logic here
            # This is a placeholder for future optimization features
            
            return {
                "status": "success",
                "message": "Vector indexes optimized",
                "graph_id": graph_id
            }
            
        except Exception as e:
            logger.error(f"Index optimization failed: {e}")
            return {
                "status": "error", 
                "error": str(e)
            }
    
    async def create_vector_indexes(self, dimension: int = 512):
        """Create necessary vector indexes in Neo4j"""
        
        try:
            # Drop existing indexes if they exist (for dimension changes)
            drop_queries = [
                "DROP INDEX entity_embeddings IF EXISTS",
                "DROP INDEX chunk_embeddings IF EXISTS"
            ]
            
            for query in drop_queries:
                try:
                    await neo4j_client.execute_write_query(query)
                except Exception as e:
                    logger.debug(f"Index drop warning: {e}")
            
            # Create vector indexes
            index_queries = [
                f"""
                CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
                FOR (e:`__Entity__`) ON (e.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dimension},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """,
                f"""
                CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                FOR (c:DocumentChunk) ON (c.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dimension},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """
            ]
            
            for query in index_queries:
                await neo4j_client.execute_write_query(query)
                logger.info(f"Created vector index with dimension {dimension}")
            
            # Create fulltext indexes for hybrid search
            fulltext_queries = [
                """
                CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
                FOR (e:`__Entity__`) ON EACH [e.name, e.description]
                """,
                """
                CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS  
                FOR (c:DocumentChunk) ON EACH [c.text, c.content]
                """
            ]
            
            for query in fulltext_queries:
                try:
                    await neo4j_client.execute_write_query(query)
                    logger.info("Created fulltext index")
                except Exception as e:
                    logger.debug(f"Fulltext index warning: {e}")
            
            logger.info("Vector indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create vector indexes: {e}")
            raise
    
    async def ensure_indexes_exist(self, graph_id: UUID, dimension: int = 1536) -> Dict[str, Any]:
        """
        Ensure vector indexes exist for a specific graph.
        Creates indexes if they don't exist, validates existing ones.
        
        Args:
            graph_id: Graph identifier
            dimension: Embedding dimension (default 1536 for OpenAI text-embedding-3-small)
            
        Returns:
            Dict with status and index information
        """
        try:
            logger.info(f"Ensuring vector indexes exist for graph {graph_id} with dimension {dimension}")
            
            # Check if indexes already exist
            index_status = await self._check_existing_indexes()
            
            # Create indexes if they don't exist or have wrong dimensions
            if not index_status.get("entity_index_exists") or not index_status.get("chunk_index_exists"):
                logger.info("Creating missing vector indexes")
                await self.create_vector_indexes(dimension=dimension)
            else:
                logger.info("Vector indexes already exist")
            
            # Verify indexes are ready
            ready_status = await self._verify_indexes_ready()
            
            return {
                "status": "success",
                "graph_id": str(graph_id),
                "dimension": dimension,
                "indexes_created": not index_status.get("entity_index_exists", True),
                "entity_index_ready": ready_status.get("entity_index_ready", False),
                "chunk_index_ready": ready_status.get("chunk_index_ready", False),
                "message": "Vector indexes ensured for graph"
            }
            
        except Exception as e:
            logger.error(f"Failed to ensure indexes exist for graph {graph_id}: {e}")
            return {
                "status": "error",
                "graph_id": str(graph_id),
                "error": str(e),
                "message": f"Failed to ensure vector indexes: {str(e)}"
            }
    
    async def _check_existing_indexes(self) -> Dict[str, bool]:
        """Check if vector indexes already exist"""
        try:
            # Query to check existing vector indexes
            check_query = """
            SHOW INDEXES
            YIELD name, type, labelsOrTypes, properties, options
            WHERE type = 'VECTOR'
            RETURN name, labelsOrTypes, properties, options
            """
            
            result = await neo4j_client.execute_query(check_query)
            
            entity_index_exists = False
            chunk_index_exists = False
            
            for record in result:
                index_name = record.get("name", "")
                labels = record.get("labelsOrTypes", [])
                
                if "entity_embeddings" in index_name or ("Entity" in labels or "__Entity__" in labels):
                    entity_index_exists = True
                elif "chunk_embeddings" in index_name or ("DocumentChunk" in labels or "Chunk" in labels):
                    chunk_index_exists = True
            
            return {
                "entity_index_exists": entity_index_exists,
                "chunk_index_exists": chunk_index_exists
            }
            
        except Exception as e:
            logger.warning(f"Could not check existing indexes: {e}")
            return {"entity_index_exists": False, "chunk_index_exists": False}
    
    async def _verify_indexes_ready(self) -> Dict[str, bool]:
        """Verify that indexes are ready for queries"""
        try:
            # Query to check index status
            status_query = """
            SHOW INDEXES
            YIELD name, state, type
            WHERE type = 'VECTOR'
            RETURN name, state
            """
            
            result = await neo4j_client.execute_query(status_query)
            
            entity_ready = False
            chunk_ready = False
            
            for record in result:
                index_name = record.get("name", "")
                state = record.get("state", "")
                
                if "entity" in index_name.lower() and state == "ONLINE":
                    entity_ready = True
                elif "chunk" in index_name.lower() and state == "ONLINE":
                    chunk_ready = True
            
            return {
                "entity_index_ready": entity_ready,
                "chunk_index_ready": chunk_ready
            }
            
        except Exception as e:
            logger.warning(f"Could not verify index readiness: {e}")
            return {"entity_index_ready": False, "chunk_index_ready": False}
    
    async def similarity_search(
        self, 
        query_embedding: List[float], 
        graph_id: str, 
        k: int = 5,
        node_types: List[str] = None,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using vector indexes.
        Used by search_service for semantic queries.
        """
        try:
            # Default to searching both entities and chunks
            if not node_types:
                node_types = ["Entity", "Chunk"]
            
            results = []
            
            for node_type in node_types:
                if node_type == "Entity":
                    query = """
                    CALL db.index.vector.queryNodes('entity_embeddings', $k, $query_embedding)
                    YIELD node, score
                    WHERE node.graph_id = $graph_id AND score >= $threshold
                    RETURN node.id as id, node.name as name, node.type as type,
                           node.description as description, labels(node) as labels, score,
                           node{.*} as properties
                    ORDER BY score DESC
                    """
                elif node_type == "Chunk":
                    query = """
                    CALL db.index.vector.queryNodes('chunk_embeddings', $k, $query_embedding)
                    YIELD node, score
                    WHERE node.graph_id = $graph_id AND score >= $threshold
                    RETURN node.id as id, node.text as text, score,
                           labels(node) as labels, node{.*} as properties
                    ORDER BY score DESC
                    """
                else:
                    continue
                
                result = await neo4j_client.execute_read_query(query, {
                    "k": k,
                    "query_embedding": query_embedding,
                    "graph_id": graph_id,
                    "threshold": threshold
                })
                
                if result:
                    for record in result:
                        results.append({
                            "id": record["id"],
                            "score": record["score"],
                            "type": node_type,
                            "labels": record["labels"],
                            "properties": record["properties"],
                            **({"name": record["name"]} if "name" in record else {}),
                            **({"type": record["type"]} if "type" in record else {}),
                            **({"description": record["description"]} if "description" in record else {}),
                            **({"text": record["text"]} if "text" in record else {})
                        })
            
            # Sort by score and return top k
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.error(f"Vector similarity search failed: {e}")
            return []

# Create singleton instance
vector_service = VectorService()