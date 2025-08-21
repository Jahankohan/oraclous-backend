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
    """Enhanced service for managing knowledge graphs - Single Database Edition"""
    
    def __init__(self):
        pass
    
    async def store_graph_documents(
        self,
        graph_id: UUID,
        graph_documents: List[GraphDocument],
        neo4j_database: str = None  # Ignored - always use default
    ) -> Tuple[int, int]:
        """Store graph documents in Neo4j and return counts"""
        
        try:
            entities_count = 0
            relationships_count = 0
            
            for graph_doc in graph_documents:
                # Store nodes
                for node in graph_doc.nodes:
                    await self._store_node(node, str(graph_id))
                    entities_count += 1
                
                # Store relationships
                for rel in graph_doc.relationships:
                    await self._store_relationship(rel, str(graph_id))
                    relationships_count += 1
            
            logger.info(f"Stored {entities_count} entities and {relationships_count} relationships for graph {graph_id}")
            return entities_count, relationships_count
            
        except Exception as e:
            logger.error(f"Error storing graph documents: {e}")
            raise
    
    async def _store_node(self, node, graph_id: str):
        """Store a single node in Neo4j with graph_id isolation"""
        
        # Prepare node properties
        properties = dict(node.properties) if hasattr(node, 'properties') and node.properties else {}
        properties["graph_id"] = graph_id
        properties["id"] = node.id
        
        # Sanitize node labels - Neo4j labels cannot contain spaces or special characters
        if isinstance(node.type, list):
            labels = [self._sanitize_label(label) for label in node.type]
        else:
            labels = [self._sanitize_label(node.type)]

        labels = [label for label in labels if label]

        # If no valid labels, use default
        if not labels:
            labels = ["Entity"]

        labels_str = ":".join(labels)

        # Use MERGE to avoid duplicates
        query = f"""
        MERGE (n:{labels_str} {{id: $id, graph_id: $graph_id}})
        SET n += $properties
        RETURN n
        """
        
        await neo4j_client.execute_write_query(
            query,
            {
                "id": node.id,
                "graph_id": graph_id,
                "properties": properties
            }
        )
    
    async def _store_relationship(self, rel, graph_id: str):
        """Store a single relationship in Neo4j with graph_id isolation"""
        
        # Prepare relationship properties
        properties = dict(rel.properties) if hasattr(rel, 'properties') and rel.properties else {}
        properties["graph_id"] = graph_id

        rel_type = self._sanitize_label(rel.type) if rel.type else "RELATED_TO"

        print("Relationship:", rel)

        query = f"""
        MATCH (source {{id: $source_id, graph_id: $graph_id}})
        MATCH (target {{id: $target_id, graph_id: $graph_id}})
        MERGE (source)-[r:{rel_type}]->(target)
        SET r += $properties
        RETURN r
        """
        
        try:
            await neo4j_client.execute_write_query(
                query,
                {
                    "source_id": rel.source.id,
                    "target_id": rel.target.id,
                    "graph_id": graph_id,
                    "properties": properties
                }
            )
        except Exception as e:
            logger.error(f"Failed to store relationship {rel.type}: {e}")
            logger.error(f"Source: {rel.source.id}, Target: {rel.target.id}")
    
    async def get_graph_stats(self, graph_id: UUID, db: AsyncSession) -> Dict[str, int]:
        """Get statistics for a graph using graph_id filtering"""
        result = await db.execute(
            select(KnowledgeGraph).where(KnowledgeGraph.id == graph_id)
        )
        graph = result.scalar_one_or_none()
        
        if not graph:
            return {"node_count": 0, "relationship_count": 0}
        
        try:
            # Query Neo4j for actual counts using graph_id filter
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
                {"graph_id": str(graph_id)}
            )
            
            rel_result = await neo4j_client.execute_query(
                rel_query,
                {"graph_id": str(graph_id)}
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
    
    async def delete_graph_data(self, graph_id: UUID):
        """Delete all nodes and relationships for a specific graph"""
        try:
            # Delete all relationships first
            rel_query = """
            MATCH ()-[r]->()
            WHERE r.graph_id = $graph_id
            DELETE r
            """
            
            # Delete all nodes
            node_query = """
            MATCH (n)
            WHERE n.graph_id = $graph_id
            DELETE n
            """
            
            await neo4j_client.execute_write_query(rel_query, {"graph_id": str(graph_id)})
            await neo4j_client.execute_write_query(node_query, {"graph_id": str(graph_id)})
            
            logger.info(f"Deleted all data for graph {graph_id}")
            
        except Exception as e:
            logger.error(f"Error deleting graph data for {graph_id}: {e}")
            raise
    
    def _sanitize_label(self, label: str) -> str:
        """Sanitize Neo4j labels to be valid identifiers"""
        if not label:
            return ""
        
        import re
        
        # Replace spaces and special characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', label)
        sanitized = sanitized.strip('_')
        
        # Ensure it starts with a letter
        if sanitized and sanitized[0].isdigit():
            sanitized = f"Type_{sanitized}"
        
        # Handle common cases
        label_mapping = {
            "Job_title": "JobTitle",
            "Organization_type": "OrganizationType", 
            "AI_models": "AIModels",
            "software_engineer": "SoftwareEngineer",
            "Job_Title": "JobTitle",
            "Organization_Type": "OrganizationType"
        }
        
        return label_mapping.get(sanitized, sanitized) if sanitized else "Entity"

graph_service = GraphService()
