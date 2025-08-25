from typing import Dict, Any, List, Optional
from langchain_community.graphs.neo4j_graph import Neo4jGraph
from app.core.neo4j_client import neo4j_client
from app.services.llm_service import llm_service
from uuid import UUID
from app.core.logging import get_logger

logger = get_logger(__name__)

class SchemaService:
    """Service for managing graph schemas - CORRECTED for single Neo4j database"""
    
    def __init__(self):
        self.neo4j_graph = None
    
    async def get_existing_schema(self) -> Dict[str, Any]:
        """
        CORRECTED: Get current database schema without neo4j_database parameter
        Since you're now using community version with single database
        """
        try:
            # Get node labels - REMOVED database parameter
            labels_query = "CALL db.labels()"
            labels_result = await neo4j_client.execute_query(labels_query)
            node_labels = [record["label"] for record in labels_result]
            
            # Get relationship types - REMOVED database parameter
            rels_query = "CALL db.relationshipTypes()"
            rels_result = await neo4j_client.execute_query(rels_query)
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
    
    async def get_graph_specific_schema(self, graph_id: str) -> Dict[str, Any]:
        """
        NEW: Get schema for a specific graph using graph_id filtering
        This replaces the old per-database approach
        """
        try:
            # Get node labels for specific graph
            labels_query = """
            MATCH (n)
            WHERE n.graph_id = $graph_id
            RETURN DISTINCT labels(n) as labels
            """
            
            # Get relationship types for specific graph
            rels_query = """
            MATCH ()-[r]->()
            WHERE r.graph_id = $graph_id
            RETURN DISTINCT type(r) as relationship_type
            """
            
            params = {"graph_id": str(graph_id)}
            
            labels_result = await neo4j_client.execute_query(labels_query, params)
            rels_result = await neo4j_client.execute_query(rels_query, params)
            
            # Extract unique entity types
            entity_types = set()
            for record in labels_result:
                for label in record["labels"]:
                    if label != "Entity":  # Filter out generic Entity label
                        entity_types.add(label)
            
            # Extract relationship types
            relationship_types = [record["relationship_type"] for record in rels_result]
            
            return {
                "entities": sorted(list(entity_types)),
                "relationships": sorted(relationship_types),
                "node_count": len(entity_types),
                "relationship_count": len(relationship_types)
            }
            
        except Exception as e:
            logger.error(f"Failed to get schema for graph {graph_id}: {e}")
            return {"entities": [], "relationships": [], "node_count": 0, "relationship_count": 0}
    
    async def get_graph_schema(self, graph_id: UUID) -> Dict[str, Any]:
        """
        Get schema for specific graph (method that chat_service expects)
        This is a wrapper around get_graph_specific_schema for consistency
        """
        try:
            return await self.get_graph_specific_schema(str(graph_id))
        except Exception as e:
            logger.error(f"Failed to get graph schema: {e}")
            return {"entities": [], "relationships": []}

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
    
    async def consolidate_graph_schema(
        self,
        graph_id: str,
        new_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        NEW: Consolidate schema for a specific graph
        More accurate than global schema consolidation
        """
        
        # Get existing schema for this specific graph
        existing_schema = await self.get_graph_specific_schema(graph_id)
        
        # Combine and deduplicate entities
        all_entities = list(set(
            existing_schema.get("entities", []) + new_schema.get("entities", [])
        ))
        
        # Combine and deduplicate relationships
        all_relationships = list(set(
            existing_schema.get("relationships", []) + new_schema.get("relationships", [])
        ))
        
        return {
            "entities": sorted(all_entities),
            "relationships": sorted(all_relationships)
        }
    
    async def _get_existing_relationships(self) -> List[str]:
        """CORRECTED: Get existing relationship types without database parameter"""
        try:
            query = "CALL db.relationshipTypes()"
            result = await neo4j_client.execute_query(query)
            return [record["relationshipType"] for record in result]
        except Exception:
            return []
    
    async def _get_graph_relationships(self, graph_id: str) -> List[str]:
        """NEW: Get relationship types for specific graph"""
        try:
            query = """
            MATCH ()-[r]->()
            WHERE r.graph_id = $graph_id
            RETURN DISTINCT type(r) as relationship_type
            """
            result = await neo4j_client.execute_query(query, {"graph_id": str(graph_id)})
            return [record["relationship_type"] for record in result]
        except Exception:
            return []
    
    async def create_graph_constraints(
        self, 
        schema_config: Dict[str, Any]
    ):
        """
        CORRECTED: Create constraints without neo4j_database parameter
        """
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
                    # REMOVED: database parameter
                    await neo4j_client.execute_write_query(constraint_query)
                    logger.debug(f"Created constraint for {entity_type}")
                except Exception as e:
                    logger.debug(f"Constraint for {entity_type} might already exist: {e}")
            
            # Create index on graph_id for efficient filtering
            graph_id_index = """
            CREATE INDEX graph_id_entities_general IF NOT EXISTS
            FOR (n:Entity)
            ON (n.graph_id)
            """
            
            try:
                await neo4j_client.execute_write_query(graph_id_index)
                logger.debug("Created graph_id index for entities")
            except Exception as e:
                logger.debug(f"Graph ID index might already exist: {e}")
                
            logger.info(f"Schema constraints created for {len(entities)} entity types")
            
        except Exception as e:
            logger.error(f"Error creating graph constraints: {e}")
            raise

    # ==================== NEW HELPER METHODS ====================
    
    async def validate_schema_for_graph(
        self,
        graph_id: str,
        proposed_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate proposed schema against existing graph data"""
        
        try:
            # Get current graph schema
            current_schema = await self.get_graph_specific_schema(graph_id)
            
            # Check for conflicts or issues
            warnings = []
            errors = []
            
            proposed_entities = set(proposed_schema.get("entities", []))
            current_entities = set(current_schema.get("entities", []))
            
            # Warn about removing existing entities
            removed_entities = current_entities - proposed_entities
            if removed_entities:
                warnings.append(f"Schema removes existing entities: {list(removed_entities)}")
            
            # Validate entity names
            for entity in proposed_entities:
                if not entity.replace("_", "").isalnum():
                    errors.append(f"Invalid entity name: {entity}")
            
            return {
                "valid": len(errors) == 0,
                "warnings": warnings,
                "errors": errors,
                "proposed_schema": proposed_schema,
                "current_schema": current_schema
            }
            
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return {
                "valid": False,
                "warnings": [],
                "errors": [f"Validation failed: {str(e)}"],
                "proposed_schema": proposed_schema,
                "current_schema": {}
            }
    
    async def get_schema_evolution_suggestions(
        self,
        graph_id: str,
        text_sample: str
    ) -> Dict[str, Any]:
        """
        NEW: Get suggestions for schema evolution based on text analysis
        """
        
        try:
            # Get current schema
            current_schema = await self.get_graph_specific_schema(graph_id)
            
            # Analyze text for potential new types (simplified version)
            # This could be enhanced with more sophisticated NLP
            potential_entities = []
            potential_relationships = []
            
            # Simple keyword-based detection (you can enhance this)
            common_entity_patterns = [
                "person", "organization", "company", "location", "product",
                "technology", "system", "project", "document", "event"
            ]
            
            common_relation_patterns = [
                "works_for", "located_in", "develops", "uses", "manages",
                "belongs_to", "creates", "owns", "partners_with"
            ]
            
            text_lower = text_sample.lower()
            
            for pattern in common_entity_patterns:
                if pattern in text_lower and pattern.title() not in current_schema.get("entities", []):
                    potential_entities.append(pattern.title())
            
            for pattern in common_relation_patterns:
                if pattern in text_lower and pattern.upper() not in current_schema.get("relationships", []):
                    potential_relationships.append(pattern.upper())
            
            return {
                "current_schema": current_schema,
                "suggested_additions": {
                    "entities": potential_entities[:5],  # Limit suggestions
                    "relationships": potential_relationships[:5]
                },
                "confidence": "low"  # Since this is a simple implementation
            }
            
        except Exception as e:
            logger.error(f"Schema evolution suggestions failed: {e}")
            return {
                "current_schema": {},
                "suggested_additions": {"entities": [], "relationships": []},
                "confidence": "none",
                "error": str(e)
            }

# Create singleton instance
schema_service = SchemaService()

# ==================== MIGRATION HELPER FUNCTIONS ====================

async def migrate_legacy_neo4j_references():
    """
    Helper function to identify and fix any remaining neo4j_database references
    Run this once during migration
    """
    
    logger.info("Checking for legacy neo4j_database references...")
    
    try:
        # Test basic connectivity
        result = await neo4j_client.execute_query("RETURN 1 as test")
        if result:
            logger.info("✅ Neo4j connectivity confirmed - single database mode")
        
        # Check if graph_id indexing is working
        index_query = "SHOW INDEXES YIELD name, labelsOrTypes, properties WHERE name CONTAINS 'graph_id'"
        indexes = await neo4j_client.execute_query(index_query)
        
        if indexes:
            logger.info(f"Found {len(indexes)} graph_id indexes")
        else:
            logger.warning("No graph_id indexes found - consider creating them")
            
        return True
        
    except Exception as e:
        logger.error(f"Legacy migration check failed: {e}")
        return False

async def create_missing_graph_indexes():
    """Create any missing indexes for graph isolation"""
    
    try:
        # ✅ FIXED: Create graph_id indexes for specific node types
        node_index_queries = [
            "CREATE INDEX graph_id_entities IF NOT EXISTS FOR (n:Entity) ON (n.graph_id)",
            "CREATE INDEX graph_id_chunks IF NOT EXISTS FOR (n:Chunk) ON (n.graph_id)", 
            "CREATE INDEX graph_id_documents IF NOT EXISTS FOR (n:Document) ON (n.graph_id)"
        ]
        
        for query in node_index_queries:
            try:
                await neo4j_client.execute_write_query(query)
                logger.debug(f"Created node index: {query}")
            except Exception as e:
                logger.debug(f"Node index might already exist: {e}")
        
        # ✅ FIXED: Create graph_id indexes for common relationship types  
        rel_index_queries = [
            "CREATE INDEX graph_id_rels_general IF NOT EXISTS FOR ()-[r]-() ON (r.graph_id)",
        ]
        
        # Try general relationship index first (works in Neo4j 5.13+)
        try:
            await neo4j_client.execute_write_query(rel_index_queries[0])
            logger.debug("Created general relationship index")
        except Exception as e:
            logger.debug(f"General relationship index not supported, creating specific types: {e}")
            
            # Fallback: Create indexes for common relationship types
            specific_rel_queries = [
                "CREATE INDEX graph_id_related IF NOT EXISTS FOR ()-[r:RELATED_TO]-() ON (r.graph_id)",
                "CREATE INDEX graph_id_mentions IF NOT EXISTS FOR ()-[r:MENTIONS]-() ON (r.graph_id)",
                "CREATE INDEX graph_id_contains IF NOT EXISTS FOR ()-[r:CONTAINS]-() ON (r.graph_id)"
            ]
            
            for query in specific_rel_queries:
                try:
                    await neo4j_client.execute_write_query(query)
                except Exception as e:
                    logger.debug(f"Specific relationship index warning: {e}")
        
        logger.info("Graph isolation indexes created/verified")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create graph indexes: {e}")
        return False