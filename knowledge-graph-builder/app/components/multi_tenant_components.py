from typing import List, Dict, Any, Optional
from uuid import UUID
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.retrievers import VectorRetriever, VectorCypherRetriever
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter, KGWriter
from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor
from neo4j_graphrag.experimental.components.types import Neo4jGraph, Neo4jNode, Neo4jRelationship
from neo4j_graphrag.types import RetrieverResult, RetrieverResultItem
from app.core.logging import get_logger

logger = get_logger(__name__)


class MultiTenantRetriever(Retriever):
    """
    Simple wrapper that adds graph_id filtering to any Neo4j retriever.
    Follows the composition over inheritance principle.
    """
    
    def __init__(self, base_retriever: Retriever, graph_id: str):
        """
        Args:
            base_retriever: Any Neo4j GraphRAG retriever (VectorRetriever, etc.)
            graph_id: Tenant identifier for filtering
        """
        self.base_retriever = base_retriever
        self.graph_id = graph_id
        super().__init__(base_retriever.driver)
    
    def get_search_results(self, query_vector=None, query_text=None, **kwargs) -> RetrieverResult:
        """Add graph_id filter to search and delegate to base retriever"""
        
        # Add multi-tenant filtering
        filters = kwargs.get('filters', {})
        filters['graph_id'] = self.graph_id
        kwargs['filters'] = filters
        
        # Delegate to base retriever
        results = self.base_retriever.get_search_results(
            query_vector=query_vector, 
            query_text=query_text, 
            **kwargs
        )
        
        # Additional filtering in case base retriever doesn't support filters properly
        filtered_items = []
        for item in results.items:
            # Check if item has graph_id property matching our tenant
            if hasattr(item, 'metadata') and item.metadata.get('graph_id') == self.graph_id:
                filtered_items.append(item)
            elif hasattr(item, 'content') and self.graph_id in str(item.content):
                filtered_items.append(item)
                
        return RetrieverResult(items=filtered_items)

class MultiTenantKnowledgeGraphWriter(KGWriter):
    """
    Multi-tenant wrapper for Neo4j KG Writer.
    Automatically injects graph_id into all nodes and relationships.
    """
    
    def __init__(self, base_writer: Neo4jWriter, graph_id: str):
        """
        Args:
            base_writer: Neo4j GraphRAG writer
            graph_id: Tenant identifier to inject
        """
        self.base_writer = base_writer
        self.graph_id = graph_id
    
    async def run(self, graph: Neo4jGraph) -> Dict[str, Any]:
        """Inject graph_id into all nodes/relationships and delegate to base writer"""
        
        # Inject graph_id into all nodes
        for node in graph.nodes:
            if not node.properties:
                node.properties = {}
            node.properties['graph_id'] = self.graph_id
            
        # Inject graph_id into all relationships  
        for rel in graph.relationships:
            if not rel.properties:
                rel.properties = {}
            rel.properties['graph_id'] = self.graph_id
            
        # Delegate to base writer
        result = await self.base_writer.run(graph)
        return result

class MultiTenantEntityRelationExtractor(LLMEntityRelationExtractor):
    """
    Multi-tenant wrapper for LLM entity extraction.
    Adds graph_id to all extracted entities and relationships.
    """
    
    def __init__(self, base_extractor: LLMEntityRelationExtractor, graph_id: str):
        """
        Args:
            base_extractor: Neo4j GraphRAG entity extractor
            graph_id: Tenant identifier
        """
        self.base_extractor = base_extractor
        self.graph_id = graph_id
        # Copy all attributes from base extractor
        super().__init__(
            llm=base_extractor.llm,
            prompt_template=base_extractor.prompt_template,
            create_lexical_graph=base_extractor.create_lexical_graph,
            on_error=base_extractor.on_error,
            max_concurrency=base_extractor.max_concurrency
        )
    
    async def run(self, **kwargs) -> Neo4jGraph:
        """Extract entities and inject graph_id into all results"""
        
        # Delegate to base extractor
        graph = await self.base_extractor.run(**kwargs)
        
        # Inject graph_id into all nodes
        for node in graph.nodes:
            if not node.properties:
                node.properties = {}
            node.properties['graph_id'] = self.graph_id
            
        # Inject graph_id into all relationships
        for rel in graph.relationships:
            if not rel.properties:
                rel.properties = {}
            rel.properties['graph_id'] = self.graph_id
            
        return graph


class MultiTenantVectorRetriever(MultiTenantRetriever):
    """Specialized multi-tenant vector retriever"""
    
    def __init__(self, driver, index_name: str, embedder, graph_id: str, **kwargs):
        """Create multi-tenant vector retriever"""
        
        base_retriever = VectorRetriever(
            driver=driver,
            index_name=f"{index_name}_{graph_id}",  # Tenant-specific index
            embedder=embedder,
            **kwargs
        )
        super().__init__(base_retriever, graph_id)


class MultiTenantVectorCypherRetriever(MultiTenantRetriever):
    """Specialized multi-tenant vector+cypher retriever"""
    
    def __init__(self, driver, index_name: str, embedder, retrieval_query: str, graph_id: str, **kwargs):
        """Create multi-tenant vector+cypher retriever with graph_id filtering in query"""
        
        # Modify retrieval query to include graph_id filter
        filtered_query = self._add_graph_id_filter(retrieval_query, graph_id)
        
        base_retriever = VectorCypherRetriever(
            driver=driver,
            index_name=f"{index_name}_{graph_id}",  # Tenant-specific index
            embedder=embedder,
            retrieval_query=filtered_query,
            **kwargs
        )
        super().__init__(base_retriever, graph_id)
    
    def _add_graph_id_filter(self, query: str, graph_id: str) -> str:
        """Add WHERE graph_id = $graph_id filter to Cypher queries"""
        
        # Simple approach: add filter after first MATCH
        if "MATCH" in query:
            lines = query.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith("MATCH"):
                    # Insert graph_id filter after this MATCH
                    lines.insert(i + 1, f"WHERE node.graph_id = '{graph_id}'")
                    break
            return '\n'.join(lines)
        return query
