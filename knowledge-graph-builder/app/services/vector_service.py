from typing import List, Dict, Any, Optional
from uuid import UUID
from app.core.neo4j_client import neo4j_client
from app.core.logging import get_logger

logger = get_logger(__name__)

class VectorService:
    """Service for managing vector indexes and operations in Neo4j"""
    
    def __init__(self):
        pass
    
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
                FOR (e:Entity) ON (e.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dimension},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """,
                f"""
                CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
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
                FOR (e:Entity) ON EACH [e.name, e.description]
                """,
                """
                CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS  
                FOR (c:Chunk) ON EACH [c.text, c.content]
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
    
    async def add_entity_embedding(
        self, 
        entity_id: str, 
        embedding: List[float],
        graph_id: UUID
    ):
        """Add embedding to an entity"""
        
        query = """
        MATCH (e:Entity {id: $entity_id, graph_id: $graph_id})
        SET e.embedding = $embedding
        RETURN e
        """
        
        try:
            await neo4j_client.execute_write_query(query, {
                "entity_id": entity_id,
                "graph_id": str(graph_id),
                "embedding": embedding
            })
            logger.debug(f"Added embedding to entity {entity_id}")
            
        except Exception as e:
            logger.error(f"Failed to add embedding to entity {entity_id}: {e}")
            raise
    
    async def create_text_chunks(
        self, 
        graph_id: UUID,
        text_chunks: List[Dict[str, Any]]
    ):
        """Create text chunk nodes with embeddings"""
        
        query = """
        UNWIND $chunks as chunk
        CREATE (c:Chunk {
            id: chunk.id,
            text: chunk.text,
            graph_id: $graph_id,
            source: chunk.source,
            embedding: chunk.embedding,
            chunk_index: chunk.chunk_index
        })
        RETURN count(c) as created
        """
        
        try:
            result = await neo4j_client.execute_write_query(query, {
                "graph_id": str(graph_id),
                "chunks": text_chunks
            })
            
            created_count = result[0]["created"] if result else 0
            logger.info(f"Created {created_count} text chunks with embeddings")
            return created_count
            
        except Exception as e:
            logger.error(f"Failed to create text chunks: {e}")
            raise

vector_service = VectorService()
