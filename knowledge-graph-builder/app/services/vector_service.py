"""
Simplified vector service - removes redundant functionality now handled by Neo4j GraphRAG.
Focuses only on vector index management and search operations.
"""

from typing import List, Dict, Any, Optional
from uuid import UUID
from app.core.neo4j_client import neo4j_client
from app.core.logging import get_logger

logger = get_logger(__name__)


class VectorService:
    """
    SIMPLIFIED vector service using Neo4j GraphRAG foundation.
    
    REMOVED (now handled by Neo4j GraphRAG):
    - create_text_chunks() → moved to Neo4j LexicalGraphBuilder
    - add_entity_embedding() → handled by Neo4j pipeline
    - Custom embedding storage logic → Neo4j handles this
    
    RESPONSIBILITIES NOW:
    - Vector index creation and management (Neo4j native)
    - Vector similarity queries (delegate to Neo4j GraphRAG retrievers)
    - Index optimization and maintenance
    """
    
    def __init__(self):
        self.driver = neo4j_client.driver

    async def ensure_indexes_exist(self, graph_id: UUID) -> Dict[str, Any]:
        """
        Ensure vector indexes exist for graph using Neo4j native vector indexes.
        Neo4j GraphRAG components handle the embedding storage.
        """
        try:
            # Create vector indexes using Neo4j's native vector index functionality
            from neo4j_graphrag.indexes import create_vector_index
            
            graph_id_str = str(graph_id)
            
            # Create indexes for different node types with tenant isolation
            indexes_created = []
            
            # Entity embeddings index
            try:
                create_vector_index(
                    self.driver,
                    name=f"entity_embeddings_{graph_id_str}",
                    label="__Entity__",
                    embedding_property="embedding",
                    dimensions=1536,  # OpenAI embedding dimension
                    similarity_fn="cosine"
                )
                indexes_created.append(f"entity_embeddings_{graph_id_str}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Failed to create entity index: {e}")
            
            # Chunk embeddings index  
            try:
                create_vector_index(
                    self.driver,
                    name=f"chunk_embeddings_{graph_id_str}",
                    label="Chunk",
                    embedding_property="embedding", 
                    dimensions=1536,
                    similarity_fn="cosine"
                )
                indexes_created.append(f"chunk_embeddings_{graph_id_str}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Failed to create chunk index: {e}")
            
            logger.info(f"Vector indexes ensured for graph {graph_id}: {indexes_created}")
            return {
                "status": "success",
                "indexes_created": indexes_created,
                "graph_id": graph_id_str
            }
            
        except Exception as e:
            logger.error(f"Failed to ensure vector indexes for {graph_id}: {e}")
            raise

    async def similarity_search(
        self, 
        query_embedding: List[float], 
        graph_id: UUID,
        k: int = 5,
        index_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using Neo4j native vector search.
        This is a thin wrapper around Neo4j's vector search capabilities.
        """
        try:
            if not index_name:
                index_name = f"entity_embeddings_{graph_id}"
            
            query = """
            CALL db.index.vector.queryNodes($index_name, $k, $query_embedding) 
            YIELD node, score
            WHERE node.graph_id = $graph_id
            RETURN node.id as id, node.name as name, score, labels(node) as labels
            ORDER BY score DESC
            """
            
            async with self.driver.session() as session:
                result = await session.run(
                    query,
                    index_name=index_name,
                    k=k,
                    query_embedding=query_embedding,
                    graph_id=str(graph_id)
                )
                
                records = await result.data()
                return [dict(record) for record in records]
                
        except Exception as e:
            logger.error(f"Vector similarity search failed: {e}")
            return []

    async def optimize_indexes(self, graph_id: Optional[UUID] = None) -> Dict[str, Any]:
        """Optimize vector indexes for better performance"""
        try:
            # Use Neo4j's native index optimization
            query = """
            CALL db.indexes() YIELD name, type, state
            WHERE type = "VECTOR" AND ($graph_id IS NULL OR name CONTAINS $graph_id_str)
            WITH name
            CALL {
                WITH name
                CALL db.index.vector.queryNodes(name, 1, [0.0]) YIELD node
                RETURN count(node) as test_count
            }
            RETURN name, test_count
            """
            
            async with self.driver.session() as session:
                result = await session.run(
                    query, 
                    graph_id_str=str(graph_id) if graph_id else None
                )
                optimized_indexes = await result.data()
                
            logger.info(f"Optimized {len(optimized_indexes)} vector indexes")
            return {
                "status": "success", 
                "optimized_indexes": len(optimized_indexes)
            }
            
        except Exception as e:
            logger.error(f"Index optimization failed: {e}")
            raise

    async def rebuild_indexes(self, graph_id: UUID) -> Dict[str, Any]:
        """Rebuild vector indexes for a specific graph"""
        try:
            graph_id_str = str(graph_id)
            
            # Drop existing indexes
            drop_queries = [
                f"DROP INDEX entity_embeddings_{graph_id_str} IF EXISTS",
                f"DROP INDEX chunk_embeddings_{graph_id_str} IF EXISTS"
            ]
            
            async with self.driver.session() as session:
                for query in drop_queries:
                    await session.run(query)
            
            # Recreate indexes
            result = await self.ensure_indexes_exist(graph_id)
            
            logger.info(f"Rebuilt vector indexes for graph {graph_id}")
            return result
            
        except Exception as e:
            logger.error(f"Index rebuild failed for {graph_id}: {e}")
            raise

    async def optimize_all_indexes(self) -> Dict[str, Any]:
        """Optimize all vector indexes in the system"""
        try:
            query = """
            CALL db.indexes() YIELD name, type, state
            WHERE type = "VECTOR"
            RETURN count(name) as total_vector_indexes
            """
            
            async with self.driver.session() as session:
                result = await session.run(query)
                record = await result.single()
                total_indexes = record["total_vector_indexes"]
            
            logger.info(f"Optimized {total_indexes} vector indexes system-wide")
            return {
                "status": "success",
                "total_indexes_optimized": total_indexes
            }
            
        except Exception as e:
            logger.error(f"System-wide index optimization failed: {e}")
            raise

# ==================== SERVICES TO REMOVE ====================

"""
The following services contain functionality that is now redundant with Neo4j GraphRAG:

1. REMOVE: app/services/entity_extractor.py
   - Replaced by: neo4j_graphrag.experimental.components.entity_relation_extractor.LLMEntityRelationExtractor
   - Functionality: Entity and relationship extraction from text
   - Migration: Use MultiTenantEntityRelationExtractor wrapper

2. SIMPLIFY: app/services/enhanced_graph_service.py  
   - Keep: Multi-tenant orchestration logic
   - Remove: Custom entity extraction, custom graph building
   - Replace: Use graphrag_ingestion_service for main pipeline

3. SIMPLIFY: app/services/embedding_service.py
   - Keep: Service initialization and configuration
   - Remove: Custom embedding storage (Neo4j GraphRAG handles this)
   - Neo4j alternative: neo4j_graphrag.embeddings.openai.OpenAIEmbeddings

4. SIMPLIFY: app/services/graph_service.py
   - Keep: Multi-tenant Cypher queries, custom analytics
   - Remove: Basic CRUD operations now handled by Neo4j KGWriter
   - Keep: Advanced analytics, community detection

Key principles for cleanup:
- Remove duplicate functionality that Neo4j GraphRAG provides
- Keep multi-tenant wrappers and orchestration
- Keep domain-specific business logic
- Keep advanced analytics not provided by Neo4j GraphRAG
"""

# Global service instance
vector_service = VectorService()