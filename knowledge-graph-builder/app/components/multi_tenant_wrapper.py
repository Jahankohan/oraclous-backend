# app/components/multi_tenant_wrapper.py
"""
Multi-tenant wrapper components that inject graph_id into all operations
"""
from typing import Dict, Any, Optional
from neo4j_graphrag.experimental.pipeline.component import Component
from neo4j_graphrag.experimental.components.types import Neo4jGraph, Neo4jNode, Neo4jRelationship
from neo4j_graphrag.retrievers.base import Retriever
from pydantic import validate_call


class MultiTenantGraphWrapper(Component):
    """Injects graph_id into all nodes and relationships for perfect tenant isolation"""
    
    def __init__(self, graph_id: str, user_id: str):
        self.graph_id = graph_id
        self.user_id = user_id
    
    @validate_call
    async def run(self, graph: Neo4jGraph) -> Neo4jGraph:
        """Add tenant metadata to all graph elements"""
        
        # Add graph_id to all nodes
        for node in graph.nodes:
            if not node.properties:
                node.properties = {}
            node.properties.update({
                'graph_id': self.graph_id,
                'user_id': self.user_id,
                'tenant_created_at': datetime.now().isoformat()
            })
        
        # Add graph_id to all relationships  
        for rel in graph.relationships:
            if not rel.properties:
                rel.properties = {}
            rel.properties.update({
                'graph_id': self.graph_id,
                'user_id': self.user_id
            })
        
        return graph


class MultiTenantRetriever:
    """Wrapper that adds graph_id filtering to any Neo4j retriever"""
    
    def __init__(self, base_retriever: Retriever, graph_id: str):
        self.base_retriever = base_retriever
        self.graph_id = graph_id
    
    def search(self, query_text: str = None, query_vector = None, top_k: int = 5, **kwargs):
        """Add graph_id filter to all search operations"""
        
        # Inject graph_id filter into Cypher queries
        if hasattr(self.base_retriever, 'retrieval_query'):
            # Modify retrieval query to include graph_id filter
            original_query = self.base_retriever.retrieval_query
            
            # Add WHERE clause with graph_id filter
            if 'WHERE' in original_query:
                modified_query = original_query.replace(
                    'WHERE', 
                    f'WHERE node.graph_id = "{self.graph_id}" AND '
                )
            else:
                # Add WHERE clause after MATCH
                modified_query = original_query.replace(
                    'RETURN',
                    f'WHERE node.graph_id = "{self.graph_id}"\nRETURN'
                )
            
            # Temporarily modify the retriever's query
            self.base_retriever.retrieval_query = modified_query
        
        # Add filters for vector search
        filters = kwargs.get('filters', {})
        filters['graph_id'] = self.graph_id
        kwargs['filters'] = filters
        
        # Call original search
        result = self.base_retriever.search(
            query_text=query_text, 
            query_vector=query_vector, 
            top_k=top_k, 
            **kwargs
        )
        
        # Restore original query
        if hasattr(self.base_retriever, 'retrieval_query'):
            self.base_retriever.retrieval_query = original_query
        
        return result
    
    def __getattr__(self, name):
        """Delegate other attributes to wrapped retriever"""
        return getattr(self.base_retriever, name)


class MultiTenantKGWriter(Component):
    """Wrapper around Neo4jWriter that ensures tenant isolation"""
    
    def __init__(self, base_writer, graph_id: str):
        self.base_writer = base_writer
        self.graph_id = graph_id
    
    @validate_call
    async def run(self, graph: Neo4jGraph) -> None:
        """Write graph with tenant isolation validation"""
        
        # Validate all nodes have graph_id
        for node in graph.nodes:
            if not node.properties or node.properties.get('graph_id') != self.graph_id:
                raise ValueError(f"Node missing graph_id: {node}")
        
        # Validate all relationships have graph_id
        for rel in graph.relationships:
            if not rel.properties or rel.properties.get('graph_id') != self.graph_id:
                raise ValueError(f"Relationship missing graph_id: {rel}")
        
        # Delegate to base writer
        return await self.base_writer.run(graph)


# Enhanced pipeline with multi-tenancy
class MultiTenantAdvancedGraphRAGPipeline(AdvancedGraphRAGPipeline):
    """Your pipeline enhanced with automatic multi-tenancy"""
    
    def __init__(self, config: AdvancedPipelineConfig, graph_id: str, user_id: str):
        super().__init__(config)
        self.graph_id = graph_id
        self.user_id = user_id
    
    def _create_advanced_pipeline(self, schema: GraphSchema) -> Pipeline:
        """Enhanced pipeline with multi-tenant components"""
        pipeline = Pipeline()
        
        # Your existing components
        self.text_splitter = FixedSizeSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            approximate=self.config.approximate_chunking
        )
        
        chunk_embedder = TextChunkEmbedder(embedder=self.embedder)
        
        self.extractor = LLMEntityRelationExtractor(
            llm=self.llm,
            create_lexical_graph=True,
            max_concurrency=self.config.max_concurrency,
            on_error=self.config.on_error
        )
        
        entity_embedder = EntityEmbedder(embedder=self.embedder)
        relationship_embedder = RelationshipEmbedder(embedder=self.embedder)
        
        # NEW: Multi-tenant wrapper
        tenant_wrapper = MultiTenantGraphWrapper(self.graph_id, self.user_id)
        
        # NEW: Multi-tenant KG writer
        base_writer = Neo4jWriter(
            driver=self.driver,
            batch_size=self.config.batch_size,
            neo4j_database=self.config.neo4j_database
        )
        tenant_writer = MultiTenantKGWriter(base_writer, self.graph_id)
        
        # Build pipeline with multi-tenancy
        pipeline.add_component(self.text_splitter, "splitter")
        pipeline.add_component(chunk_embedder, "chunk_embedder")
        pipeline.add_component(self.extractor, "extractor")
        pipeline.add_component(entity_embedder, "entity_embedder")
        pipeline.add_component(relationship_embedder, "relationship_embedder")
        pipeline.add_component(tenant_wrapper, "tenant_wrapper")  # NEW
        pipeline.add_component(tenant_writer, "entity_writer")
        
        # Connect components
        pipeline.connect("splitter", "chunk_embedder", input_config={"text_chunks": "splitter"})
        pipeline.connect("chunk_embedder", "extractor", input_config={"chunks": "chunk_embedder"})
        pipeline.connect("extractor", "entity_embedder", input_config={"graph": "extractor"})
        pipeline.connect("entity_embedder", "relationship_embedder", input_config={"graph": "entity_embedder"})
        pipeline.connect("relationship_embedder", "tenant_wrapper", input_config={"graph": "relationship_embedder"})  # NEW
        pipeline.connect("tenant_wrapper", "entity_writer", input_config={"graph": "tenant_wrapper"})  # NEW
        
        return pipeline
    
    async def create_retrieval_system(self) -> Dict[str, Any]:
        """Enhanced retrieval with multi-tenancy"""
        base_system = await super().create_retrieval_system()
        
        # Wrap all retrievers with multi-tenant filtering
        base_system["vector_retriever"] = MultiTenantRetriever(
            base_system["vector_retriever"], 
            self.graph_id
        )
        base_system["hybrid_retriever"] = MultiTenantRetriever(
            base_system["hybrid_retriever"], 
            self.graph_id
        )
        # ... wrap other retrievers
        
        return base_system
    