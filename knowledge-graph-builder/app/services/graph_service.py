from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from langchain_community.graphs.graph_document import GraphDocument
from app.models.graph import KnowledgeGraph, IngestionJob
from app.core.neo4j_client import neo4j_client
from app.services.schema_service import schema_service
from app.core.logging import get_logger
import json

logger = get_logger(__name__)

class GraphService:
    """Enhanced service for managing knowledge graphs"""
    
    def __init__(self):
        pass
    
    async def store_graph_documents(
        self,
        graph_id: UUID,
        graph_documents: List[GraphDocument],
        neo4j_database: str = None
    ) -> Tuple[int, int]:
        """Store graph documents in Neo4j and return counts"""
        
        try:
            entities_count = 0
            relationships_count = 0
            
            for graph_doc in graph_documents:
                # Store nodes
                for node in graph_doc.nodes:
                    await self._store_node(node, str(graph_id), neo4j_database)
                    entities_count += 1
                
                # Store relationships
                for rel in graph_doc.relationships:
                    await self._store_relationship(rel, str(graph_id), neo4j_database)
                    relationships_count += 1
            
            logger.info(f"Stored {entities_count} entities and {relationships_count} relationships")
            return entities_count, relationships_count
            
        except Exception as e:
            logger.error(f"Error storing graph documents: {e}")
            raise
    
    async def _store_node(self, node, graph_id: str, neo4j_database: str = None):
        """Store a single node in Neo4j"""
        
        # Prepare node properties
        properties = dict(node.properties)
        properties["graph_id"] = graph_id
        properties["id"] = node.id
        
        # Create node with label
        labels = ":".join(node.type) if isinstance(node.type, list) else node.type
        
        query = f"""
        MERGE (n:{labels} {{id: $id, graph_id: $graph_id}})
        SET n += $properties
        RETURN n
        """
        
        await neo4j_client.execute_write_query(
            query,
            {
                "id": node.id,
                "graph_id": graph_id,
                "properties": properties
            },
            database=neo4j_database
        )
    
    async def _store_relationship(self, rel, graph_id: str, neo4j_database: str = None):
        """Store a single relationship in Neo4j"""
        
        # Prepare relationship properties
        properties = dict(rel.properties)
        properties["graph_id"] = graph_id
        
        query = f"""
        MATCH (source {{id: $source_id, graph_id: $graph_id}})
        MATCH (target {{id: $target_id, graph_id: $graph_id}})
        MERGE (source)-[r:{rel.type}]->(target)
        SET r += $properties
        RETURN r
        """
        
        await neo4j_client.execute_write_query(
            query,
            {
                "source_id": rel.source.id,
                "target_id": rel.target.id,
                "graph_id": graph_id,
                "properties": properties
            },
            database=neo4j_database
        )
    
    async def get_graph_stats(self, graph_id: UUID, db: AsyncSession) -> Dict[str, int]:
        """Get statistics for a graph"""
        result = await db.execute(
            select(KnowledgeGraph).where(KnowledgeGraph.id == graph_id)
        )
        graph = result.scalar_one_or_none()
        
        if not graph:
            return {"node_count": 0, "relationship_count": 0}
        
        try:
            # Query Neo4j for actual counts
            node_query = """
            MATCH (n) 
            WHERE n.graph_id = $graph_id 
            RETURN count(n) as count
            """
            
            rel_query = """
            MATCH ()-[r]->() 
            WHERE r.graph_id = $graph_id 
            RETURN count(r) as count
            """
            
            node_result = await neo4j_client.execute_query(
                node_query, 
                {"graph_id": str(graph_id)},
                database=graph.neo4j_database
            )
            
            rel_result = await neo4j_client.execute_query(
                rel_query,
                {"graph_id": str(graph_id)},
                database=graph.neo4j_database
            )
            
            node_count = node_result[0]["count"] if node_result else 0
            rel_count = rel_result[0]["count"] if rel_result else 0
            
            # Update cached counts
            await db.execute(
                update(KnowledgeGraph)
                .where(KnowledgeGraph.id == graph_id)
                .values(node_count=node_count, relationship_count=rel_count)
            )
            await db.commit()
            
            return {"node_count": node_count, "relationship_count": rel_count}
            
        except Exception as e:
            logger.error(f"Error getting graph stats for {graph_id}: {e}")
            return {"node_count": graph.node_count, "relationship_count": graph.relationship_count}
    
    async def create_graph_schema(self, graph_id: UUID, schema_config: Dict[str, Any]):
        """Create or update graph schema in Neo4j"""
        try:
            # Get the graph to find Neo4j database
            # This would need the database session - simplified for now
            await schema_service.create_graph_constraints(schema_config)
            logger.info(f"Created graph schema for {graph_id}")
            
        except Exception as e:
            logger.error(f"Error creating graph schema: {e}")
            raise

graph_service = GraphService()
