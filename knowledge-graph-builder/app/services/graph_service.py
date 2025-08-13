import logging
from typing import List, Dict, Any, Optional, Tuple
import asyncio

from app.core.neo4j_client import Neo4jClient
from app.core.exceptions import ServiceError
from app.config.settings import get_settings
from app.models.responses import GraphVisualization, GraphNode, GraphRelationship, DuplicateNode

logger = logging.getLogger(__name__)

class GraphService:
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
        self.settings = get_settings()
    
    async def get_graph_visualization(
        self, 
        file_names: Optional[List[str]] = None,
        limit: int = 100
    ) -> GraphVisualization:
        """Get graph visualization data"""
        try:
            # Build query based on file filters
            if file_names:
                query = """
                MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:HAS_ENTITY]->(e:Entity)
                WHERE d.fileName IN $fileNames
                WITH collect(DISTINCT e) as entities
                UNWIND entities as e1
                MATCH (e1)-[r]-(e2)
                WHERE e2 IN entities
                RETURN DISTINCT e1, r, e2
                LIMIT $limit
                """
                params = {"fileNames": file_names, "limit": limit}
            else:
                query = """
                MATCH (e1:Entity)-[r]-(e2:Entity)
                RETURN DISTINCT e1, r, e2
                LIMIT $limit
                """
                params = {"limit": limit}
            
            result = self.neo4j.execute_query(query, params)
            
            nodes = {}
            relationships = []
            
            for record in result:
                # Process nodes
                for node_key in ['e1', 'e2']:
                    node_data = record[node_key]
                    node_id = str(node_data['id'])
                    
                    if node_id not in nodes:
                        nodes[node_id] = GraphNode(
                            id=node_id,
                            labels=list(node_data.labels),
                            properties=dict(node_data)
                        )
                
                # Process relationship
                rel_data = record['r']
                relationships.append(GraphRelationship(
                    id=str(rel_data.id),
                    type=rel_data.type,
                    start_node_id=str(record['e1']['id']),
                    end_node_id=str(record['e2']['id']),
                    properties=dict(rel_data)
                ))
            
            return GraphVisualization(
                nodes=list(nodes.values()),
                relationships=relationships
            )
            
        except Exception as e:
            logger.error(f"Error getting graph visualization: {e}")
            raise ServiceError(f"Failed to get graph visualization: {e}")
    
    async def get_node_neighbors(self, node_id: str, depth: int = 1) -> GraphVisualization:
        """Get neighbors of a specific node"""
        try:
            query = f"""
            MATCH (n)-[r*1..{depth}]-(neighbor)
            WHERE elementId(n) = $nodeId
            WITH n, r, neighbor
            UNWIND r as rel
            RETURN DISTINCT n, rel, neighbor
            """
            
            result = self.neo4j.execute_query(query, {"nodeId": node_id})
            
            nodes = {}
            relationships = []
            
            for record in result:
                # Add center node
                center_node = record['n']
                center_id = str(center_node.element_id)
                
                if center_id not in nodes:
                    nodes[center_id] = GraphNode(
                        id=center_id,
                        labels=list(center_node.labels),
                        properties=dict(center_node)
                    )
                
                # Add neighbor node
                neighbor_node = record['neighbor']
                neighbor_id = str(neighbor_node.element_id)
                
                if neighbor_id not in nodes:
                    nodes[neighbor_id] = GraphNode(
                        id=neighbor_id,
                        labels=list(neighbor_node.labels),
                        properties=dict(neighbor_node)
                    )
                
                # Add relationship
                rel = record['rel']
                relationships.append(GraphRelationship(
                    id=str(rel.element_id),
                    type=rel.type,
                    start_node_id=str(rel.start_node.element_id),
                    end_node_id=str(rel.end_node.element_id),
                    properties=dict(rel)
                ))
            
            return GraphVisualization(
                nodes=list(nodes.values()),
                relationships=relationships
            )
            
        except Exception as e:
            logger.error(f"Error getting node neighbors: {e}")
            raise ServiceError(f"Failed to get node neighbors: {e}")
    
    async def delete_documents(self, file_names: List[str], delete_entities: bool = False) -> int:
        """Delete documents and optionally their entities"""
        try:
            if delete_entities:
                # Delete everything related to the documents
                query = """
                MATCH (d:Document)
                WHERE d.fileName IN $fileNames
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e:Entity)
                DETACH DELETE d, c, e
                RETURN count(d) as deletedCount
                """
            else:
                # Delete only documents and chunks, keep entities
                query = """
                MATCH (d:Document)
                WHERE d.fileName IN $fileNames
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                DETACH DELETE d, c
                RETURN count(d) as deletedCount
                """
            
            result = self.neo4j.execute_write_query(query, {"fileNames": file_names})
            return result[0]["deletedCount"] if result else 0
            
        except Exception as e:
            logger.error
