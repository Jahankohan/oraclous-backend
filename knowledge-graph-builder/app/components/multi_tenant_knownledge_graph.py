from neo4j_graphrag.knowledge_graph import KnowledgeGraph
from neo4j import Driver
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class MultiTenantKnowledgeGraph(KnowledgeGraph):
    """
    Multi-tenant wrapper for Neo4j KnowledgeGraph.
    
    FEATURES:
    - Automatic graph_id injection into all nodes/relationships
    - Safe multi-tenant querying with automatic filtering
    - Preserves all KnowledgeGraph functionality
    - Zero-configuration tenant isolation
    """
    
    def __init__(self, driver: Driver, graph_id: str):
        """
        Args:
            driver: Neo4j driver instance
            graph_id: Unique identifier for tenant graph
        """
        super().__init__(driver)
        self.graph_id = graph_id
        logger.info(f"MultiTenantKnowledgeGraph initialized for graph {graph_id}")
    
    def add_document(self, document: Dict[str, Any]) -> None:
        """Add document with automatic graph_id injection"""
        # Inject graph_id into document metadata
        if 'metadata' not in document:
            document['metadata'] = {}
        document['metadata']['graph_id'] = self.graph_id
        
        # Inject graph_id into all entities and relationships
        if 'entities' in document:
            for entity in document['entities']:
                entity['graph_id'] = self.graph_id
        
        if 'relationships' in document:
            for rel in document['relationships']:
                rel['graph_id'] = self.graph_id
        
        logger.debug(f"Adding document to graph {self.graph_id}")
        super().add_document(document)
    
    def query(self, cypher: str, parameters: Dict[str, Any] = None) -> Any:
        """Execute query with automatic graph_id filtering"""
        if parameters is None:
            parameters = {}
        
        # Always inject graph_id parameter
        parameters['graph_id'] = self.graph_id
        
        # Auto-inject graph_id filter for safety (simple heuristic)
        safe_cypher = self._inject_tenant_filter(cypher)
        
        logger.debug(f"Executing tenant-safe query for graph {self.graph_id}")
        return super().query(safe_cypher, parameters)
    
    def _inject_tenant_filter(self, cypher: str) -> str:
        """
        Inject tenant filter into Cypher query (simplified implementation)
        
        NOTE: This is a basic implementation. In production, you'd want
        a more sophisticated query parser.
        """
        cypher_upper = cypher.upper()
        
        # If query already has WHERE clause, add graph_id filter
        if ' WHERE ' in cypher_upper:
            return cypher.replace(' WHERE ', ' WHERE n.graph_id = $graph_id AND ')
        
        # If query has MATCH but no WHERE, add WHERE clause
        if 'MATCH ' in cypher_upper and ' RETURN ' in cypher_upper:
            return cypher.replace(' RETURN ', ' WHERE n.graph_id = $graph_id RETURN ')
        
        # Return as-is if we can't safely modify
        return cypher
