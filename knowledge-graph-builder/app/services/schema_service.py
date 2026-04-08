"""
Neo4j Schema Management Service

Service for extracting, caching, and managing Neo4j database schemas
to improve Text2CypherRetriever performance with dynamic schema updates.
"""
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timezone

from app.core.neo4j_client import neo4j_client
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NodeSchema:
    """Schema information for a node type"""
    label: str
    properties: Dict[str, str]  # property_name -> type
    sample_count: int
    indexes: List[str]


@dataclass
class RelationshipSchema:
    """Schema information for a relationship type"""
    type: str
    properties: Dict[str, str]  # property_name -> type
    start_labels: Set[str]
    end_labels: Set[str]
    sample_count: int


@dataclass
class GraphSchema:
    """Complete graph schema information"""
    graph_id: str
    nodes: Dict[str, NodeSchema]
    relationships: Dict[str, RelationshipSchema]
    constraints: List[Dict[str, Any]]
    indexes: List[Dict[str, Any]]
    last_updated: datetime
    schema_version: str


class Neo4jSchemaManager:
    """
    Manager for Neo4j database schemas with multi-tenant support.
    
    Features:
    - Dynamic schema extraction from live database
    - Multi-tenant schema isolation
    - Schema caching for performance
    - Automatic schema updates
    - Text2Cypher optimization
    """
    
    def __init__(self):
        """Initialize schema manager"""
        self._schema_cache: Dict[str, GraphSchema] = {}
        self._cache_ttl_minutes = 60  # Cache for 1 hour
        
    async def _ensure_connections(self):
        """Ensure Neo4j connections are available"""
        await neo4j_client.connect_async()
        
        if neo4j_client.async_driver is None:
            raise ConnectionError("Neo4j async driver not available for schema operations")
    
    async def extract_schema(self, graph_id: str, force_refresh: bool = False) -> GraphSchema:
        """
        Extract comprehensive schema for a specific graph.
        
        Args:
            graph_id: Graph identifier for multi-tenant isolation
            force_refresh: Force schema refresh even if cached
            
        Returns:
            Complete graph schema information
        """
        try:
            # Check cache first
            if not force_refresh and graph_id in self._schema_cache:
                cached_schema = self._schema_cache[graph_id]
                age_minutes = (datetime.now(timezone.utc) - cached_schema.last_updated).total_seconds() / 60
                if age_minutes < self._cache_ttl_minutes:
                    logger.info(f"Using cached schema for graph {graph_id} (age: {age_minutes:.1f}min)")
                    return cached_schema
            
            await self._ensure_connections()
            
            logger.info(f"Extracting schema for graph {graph_id}")
            
            # Extract node schemas
            nodes = await self._extract_node_schemas(graph_id)
            
            # Extract relationship schemas
            relationships = await self._extract_relationship_schemas(graph_id)
            
            # Get constraints and indexes
            constraints = await self._get_constraints()
            indexes = await self._get_indexes()
            
            # Create schema object
            schema = GraphSchema(
                graph_id=graph_id,
                nodes=nodes,
                relationships=relationships,
                constraints=constraints,
                indexes=indexes,
                last_updated=datetime.now(timezone.utc),
                schema_version=f"{len(nodes)}n_{len(relationships)}r_{hash(graph_id) % 10000}"
            )
            
            # Cache the schema
            self._schema_cache[graph_id] = schema
            
            logger.info(f"Schema extracted for graph {graph_id}: {len(nodes)} node types, {len(relationships)} relationship types")
            return schema
            
        except Exception as e:
            logger.error(f"Failed to extract schema for graph {graph_id}: {e}")
            raise
    
    async def _extract_node_schemas(self, graph_id: str) -> Dict[str, NodeSchema]:
        """Extract node type schemas"""
        try:
            # Get all node labels and their property schemas
            query = """
            MATCH (n {graph_id: $graph_id})
            WITH labels(n) as nodeLabels, keys(n) as props, n
            UNWIND nodeLabels as label
            WITH label, props, n
            UNWIND props as prop
            WITH label, prop, n[prop] as value
            WHERE prop <> 'graph_id'
            RETURN label,
                   prop,
                   apoc.meta.cypher.type(value) as propType,
                   count(*) as frequency
            ORDER BY label, prop
            """
            
            records = await neo4j_client.execute_query(query, {"graph_id": graph_id})
            
            # Group by label with proper typing
            label_schemas: Dict[str, Dict[str, Any]] = {}
            for record in records:
                label = str(record['label'])
                if label not in label_schemas:
                    label_schemas[label] = {"properties": {}, "total_count": 0}
                
                prop_name = str(record['prop'])
                prop_type = str(record['propType'])
                frequency = int(record['frequency'])
                
                label_schemas[label]["properties"][prop_name] = prop_type
                label_schemas[label]["total_count"] = max(int(label_schemas[label]["total_count"]), frequency)
            
            # Get node counts and indexes
            nodes: Dict[str, NodeSchema] = {}
            for label, schema_info in label_schemas.items():
                # Guard: backtick-escaped labels are safe for Neo4j, but reject any label that
                # itself contains a backtick (Neo4j never produces such labels normally).
                if '`' in label:
                    logger.warning(f"Skipping label with unexpected backtick character: {label!r}")
                    continue
                # Get sample count for this label ($graph_id is parameterized)
                count_query = f"MATCH (n:`{label}` {{graph_id: $graph_id}}) RETURN count(n) as count"
                count_records = await neo4j_client.execute_query(count_query, {"graph_id": graph_id})
                sample_count = int(count_records[0]['count']) if count_records else 0
                
                # Get indexes for this label (simplified)
                indexes: List[str] = []  # Would need to query SHOW INDEXES for actual indexes
                
                nodes[label] = NodeSchema(
                    label=label,
                    properties=dict(schema_info["properties"]),
                    sample_count=sample_count,
                    indexes=indexes
                )
            
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to extract node schemas: {e}")
            return {}
    
    async def _extract_relationship_schemas(self, graph_id: str) -> Dict[str, RelationshipSchema]:
        """Extract relationship type schemas"""
        try:
            # Get all relationship types and their schemas
            query = """
            MATCH (start {graph_id: $graph_id})-[r]->(end {graph_id: $graph_id})
            WITH type(r) as relType, labels(start) as startLabels, labels(end) as endLabels, 
                 keys(r) as props, r
            UNWIND props as prop
            WITH relType, startLabels, endLabels, prop, r[prop] as value
            WHERE prop <> 'graph_id'
            RETURN relType,
                   collect(DISTINCT startLabels) as allStartLabels,
                   collect(DISTINCT endLabels) as allEndLabels,
                   prop,
                   apoc.meta.cypher.type(value) as propType,
                   count(*) as frequency
            ORDER BY relType, prop
            """
            
            records = await neo4j_client.execute_query(query, {"graph_id": graph_id})
            
            # Group by relationship type with proper typing
            rel_schemas: Dict[str, Dict[str, Any]] = {}
            for record in records:
                rel_type = str(record['relType'])
                if rel_type not in rel_schemas:
                    rel_schemas[rel_type] = {
                        "properties": {},
                        "start_labels": set(),
                        "end_labels": set(),
                        "total_count": 0
                    }
                
                # Add property info
                if record.get('prop'):
                    rel_schemas[rel_type]["properties"][str(record['prop'])] = str(record['propType'])
                
                # Add label info
                for start_labels in record.get('allStartLabels', []):
                    if isinstance(start_labels, list):
                        rel_schemas[rel_type]["start_labels"].update(start_labels)
                for end_labels in record.get('allEndLabels', []):
                    if isinstance(end_labels, list):
                        rel_schemas[rel_type]["end_labels"].update(end_labels)
                
                rel_schemas[rel_type]["total_count"] = max(int(rel_schemas[rel_type]["total_count"]), int(record['frequency']))
            
            # Create relationship schema objects
            relationships: Dict[str, RelationshipSchema] = {}
            for rel_type, schema_info in rel_schemas.items():
                relationships[rel_type] = RelationshipSchema(
                    type=rel_type,
                    properties=dict(schema_info["properties"]),
                    start_labels=set(schema_info["start_labels"]),
                    end_labels=set(schema_info["end_labels"]),
                    sample_count=int(schema_info["total_count"])
                )
            
            return relationships
            
        except Exception as e:
            logger.error(f"Failed to extract relationship schemas: {e}")
            return {}
    
    async def _get_constraints(self) -> List[Dict[str, Any]]:
        """Get database constraints"""
        try:
            query = "SHOW CONSTRAINTS"
            records = await neo4j_client.execute_query(query)
            
            return [
                {
                    "name": record.get("name"),
                    "type": record.get("type"),
                    "entityType": record.get("entityType"),
                    "labelsOrTypes": record.get("labelsOrTypes"),
                    "properties": record.get("properties")
                }
                for record in records
            ]
            
        except Exception as e:
            logger.error(f"Failed to get constraints: {e}")
            return []
    
    async def _get_indexes(self) -> List[Dict[str, Any]]:
        """Get database indexes"""
        try:
            query = "SHOW INDEXES"
            records = await neo4j_client.execute_query(query)
            
            return [
                {
                    "name": record.get("name"),
                    "type": record.get("type"),
                    "entityType": record.get("entityType"),
                    "labelsOrTypes": record.get("labelsOrTypes"),
                    "properties": record.get("properties"),
                    "state": record.get("state")
                }
                for record in records
            ]
            
        except Exception as e:
            logger.error(f"Failed to get indexes: {e}")
            return []
    
    def format_schema_for_text2cypher(self, schema: GraphSchema) -> str:
        """
        Format schema information for Text2CypherRetriever.
        
        Args:
            schema: Graph schema to format
            
        Returns:
            Formatted schema string for LLM consumption
        """
        try:
            schema_parts = [
                f"# Neo4j Knowledge Graph Schema for {schema.graph_id}",
                f"# Last Updated: {schema.last_updated.isoformat()}",
                f"# Schema Version: {schema.schema_version}",
                "",
                "## Node Types"
            ]
            
            # Add node information
            for label, node in schema.nodes.items():
                schema_parts.append(f"### {label} ({node.sample_count:,} nodes)")
                if node.properties:
                    schema_parts.append("Properties:")
                    for prop, prop_type in node.properties.items():
                        schema_parts.append(f"  - {prop}: {prop_type}")
                if node.indexes:
                    schema_parts.append("Indexes:")
                    for index in node.indexes:
                        schema_parts.append(f"  - {index}")
                schema_parts.append("")
            
            # Add relationship information
            schema_parts.append("## Relationship Types")
            for rel_type, rel in schema.relationships.items():
                start_labels = ', '.join(sorted(rel.start_labels))
                end_labels = ', '.join(sorted(rel.end_labels))
                schema_parts.append(f"### {rel_type} ({rel.sample_count:,} relationships)")
                schema_parts.append(f"Pattern: ({start_labels})-[:{rel_type}]->({end_labels})")
                if rel.properties:
                    schema_parts.append("Properties:")
                    for prop, prop_type in rel.properties.items():
                        schema_parts.append(f"  - {prop}: {prop_type}")
                schema_parts.append("")
            
            # Add constraints
            if schema.constraints:
                schema_parts.append("## Constraints")
                for constraint in schema.constraints:
                    constraint_desc = f"{constraint.get('type', 'Unknown')} on {constraint.get('labelsOrTypes', [])} ({constraint.get('properties', [])})"
                    schema_parts.append(f"- {constraint_desc}")
                schema_parts.append("")
            
            # Add important notes
            schema_parts.extend([
                "## Important Notes",
                "- All nodes and relationships have a 'graph_id' property for multi-tenant isolation",
                f"- Always include 'graph_id: \"{schema.graph_id}\"' in WHERE clauses",
                "- Use proper Cypher syntax and be mindful of performance",
                "- Prefer specific patterns over broad MATCH statements"
            ])
            
            return "\n".join(schema_parts)
            
        except Exception as e:
            logger.error(f"Failed to format schema: {e}")
            return f"Error formatting schema for graph {schema.graph_id}"
    
    async def get_schema_for_text2cypher(self, graph_id: str) -> str:
        """
        Get formatted schema string for Text2CypherRetriever.
        
        Args:
            graph_id: Graph identifier
            
        Returns:
            Formatted schema string ready for Text2CypherRetriever
        """
        try:
            schema = await self.extract_schema(graph_id)
            return self.format_schema_for_text2cypher(schema)
            
        except Exception as e:
            logger.error(f"Failed to get schema for Text2Cypher: {e}")
            # Return a basic fallback schema
            return f"""
            # Basic Schema for {graph_id}
            
            ## Node Types
            - Entity (name: string, type: string, graph_id: string)
            - Chunk (text: string, graph_id: string)
            - Document (path: string, title: string, graph_id: string)
            
            ## Relationship Types
            - (Entity)-[:FOUNDED_BY]->(Entity)
            - (Entity)-[:CEO_OF]->(Entity)
            - (Entity)-[:LOCATED_IN]->(Entity)
            - (Entity)-[:FROM_CHUNK]->(Chunk)
            - (Chunk)-[:FROM_DOCUMENT]->(Document)
            
            ## Important Notes
            - Always include 'graph_id: "{graph_id}"' in WHERE clauses
            """
    
    def clear_cache(self, graph_id: Optional[str] = None):
        """Clear schema cache for specific graph or all graphs"""
        if graph_id:
            self._schema_cache.pop(graph_id, None)
            logger.info(f"Cleared schema cache for graph {graph_id}")
        else:
            self._schema_cache.clear()
            logger.info("Cleared all schema cache")
    
    def get_cached_schemas(self) -> Dict[str, str]:
        """Get summary of cached schemas"""
        return {
            graph_id: f"Version {schema.schema_version}, updated {schema.last_updated.isoformat()}"
            for graph_id, schema in self._schema_cache.items()
        }
    
    def get_cache_details(self) -> Dict[str, GraphSchema]:
        """Get detailed cached schema information"""
        return dict(self._schema_cache)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cached_count": len(self._schema_cache),
            "cache_ttl_minutes": self._cache_ttl_minutes
        }


# ==================== GLOBAL MANAGER INSTANCE ====================

# Global manager instance for dependency injection
schema_manager = Neo4jSchemaManager()


# ==================== CONVENIENCE FUNCTIONS ====================

async def get_text2cypher_schema(graph_id: str) -> str:
    """Convenience function to get schema for Text2CypherRetriever"""
    return await schema_manager.get_schema_for_text2cypher(graph_id)


async def refresh_schema_cache(graph_id: str) -> GraphSchema:
    """Convenience function to force refresh schema cache"""
    return await schema_manager.extract_schema(graph_id, force_refresh=True)


def clear_schema_cache(graph_id: Optional[str] = None):
    """Convenience function to clear schema cache"""
    schema_manager.clear_cache(graph_id)
