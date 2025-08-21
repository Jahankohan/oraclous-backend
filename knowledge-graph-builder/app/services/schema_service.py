from typing import Dict, Any, List, Optional
from langchain_community.graphs.neo4j_graph import Neo4jGraph
from app.core.neo4j_client import neo4j_client
from app.services.llm_service import llm_service
from app.core.logging import get_logger

logger = get_logger(__name__)

class SchemaService:
    """Service for managing graph schemas"""
    
    def __init__(self):
        self.neo4j_graph = None
    
    async def get_existing_schema(self, neo4j_database: str = None) -> Dict[str, Any]:
        """Get current database schema"""
        try:
            # Get node labels
            labels_query = "CALL db.labels()"
            labels_result = await neo4j_client.execute_query(
                labels_query, 
                database=neo4j_database
            )
            node_labels = [record["label"] for record in labels_result]
            
            # Get relationship types
            rels_query = "CALL db.relationshipTypes()"
            rels_result = await neo4j_client.execute_query(
                rels_query,
                database=neo4j_database
            )
            relationship_types = [record["relationshipType"] for record in rels_result]
            
            return {
                "entities": node_labels,
                "relationships": relationship_types,
                "node_count": len(node_labels),
                "relationship_count": len(relationship_types)
            }
            
        except Exception as e:
            logger.error(f"Failed to get existing schema: {e}")
            return {"entities": [], "relationships": [], "node_count": 0, "relationship_count": 0}
    
    async def consolidate_schema(
        self, 
        existing_entities: List[str],
        new_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Consolidate existing and new schema elements"""
        
        # Combine and deduplicate entities
        all_entities = list(set(existing_entities + new_schema.get("entities", [])))
        
        # Get existing relationships
        existing_relationships = await self._get_existing_relationships()
        all_relationships = list(set(
            existing_relationships + new_schema.get("relationships", [])
        ))
        
        return {
            "entities": sorted(all_entities),
            "relationships": sorted(all_relationships)
        }
    
    async def _get_existing_relationships(self) -> List[str]:
        """Get existing relationship types from database"""
        try:
            query = "CALL db.relationshipTypes()"
            result = await neo4j_client.execute_query(query)
            return [record["relationshipType"] for record in result]
        except Exception:
            return []
    
    async def create_graph_constraints(
        self, 
        schema_config: Dict[str, Any],
        neo4j_database: str = None
    ):
        """Create constraints and indexes based on schema"""
        try:
            entities = schema_config.get("entities", [])
            
            for entity_type in entities:
                # Create uniqueness constraint on id property
                constraint_query = f"""
                CREATE CONSTRAINT {entity_type.lower()}_id_unique IF NOT EXISTS
                FOR (n:{entity_type}) 
                REQUIRE n.id IS UNIQUE
                """
                
                try:
                    await neo4j_client.execute_write_query(
                        constraint_query,
                        database=neo4j_database
                    )
                    logger.debug(f"Created constraint for {entity_type}")
                except Exception as e:
                    logger.debug(f"Constraint for {entity_type} might already exist: {e}")
            
            # Create index on graph_id for efficient filtering
            graph_id_index = """
            CREATE INDEX graph_id_index IF NOT EXISTS
            FOR (n)
            ON (n.graph_id)
            """
            
            try:
                await neo4j_client.execute_write_query(
                    graph_id_index,
                    database=neo4j_database
                )
                logger.debug("Created graph_id index")
            except Exception as e:
                logger.debug(f"Graph ID index might already exist: {e}")
                
            logger.info(f"Schema constraints created for {len(entities)} entity types")
            
        except Exception as e:
            logger.error(f"Error creating graph constraints: {e}")
            raise

schema_service = SchemaService()
