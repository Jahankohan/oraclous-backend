from typing import List, Dict, Any, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from app.models.graph import KnowledgeGraph, IngestionJob
from app.core.neo4j_client import neo4j_client
from app.core.logging import get_logger

logger = get_logger(__name__)

class GraphService:
    """Service for managing knowledge graphs"""
    
    def __init__(self):
        pass
    
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
            node_query = f"""
            MATCH (n) 
            WHERE n.graph_id = $graph_id 
            RETURN count(n) as count
            """
            
            rel_query = f"""
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
            # Create constraints and indexes based on schema
            constraints = []
            
            # Entity uniqueness constraints
            if "entities" in schema_config:
                for entity_type in schema_config["entities"]:
                    constraint_query = f"""
                    CREATE CONSTRAINT {entity_type.lower()}_id_unique 
                    FOR (n:{entity_type}) 
                    REQUIRE n.id IS UNIQUE
                    """
                    constraints.append(constraint_query)
            
            # Execute constraints
            for constraint in constraints:
                try:
                    await neo4j_client.execute_write_query(constraint)
                except Exception as e:
                    # Constraint might already exist
                    logger.debug(f"Constraint creation warning: {e}")
            
            logger.info(f"Created graph schema for {graph_id}")
            
        except Exception as e:
            logger.error(f"Error creating graph schema: {e}")
            raise

graph_service = GraphService()
