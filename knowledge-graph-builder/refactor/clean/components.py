"""Custom components following neo4j-graphrag patterns - maintaining exact original functionality."""

from typing import Any, Dict, List
from pydantic import validate_call

from neo4j_graphrag.experimental.pipeline.component import Component, DataModel
from neo4j_graphrag.experimental.components.types import Neo4jGraph
from neo4j_graphrag.embeddings.base import Embedder


class LoggingText2CypherRetriever:
    """Wrapper for Text2CypherRetriever that logs generated Cypher queries - exact copy from original"""
    
    def __init__(self, text2cypher_retriever, logger):
        self._retriever = text2cypher_retriever
        self.logger = logger
        self.last_generated_cypher = None
    
    def search(self, query_text: str, **kwargs):
        """Override search to capture and log generated Cypher"""
        try:
            # Call the original search method
            result = self._retriever.search(query_text=query_text, **kwargs)
            
            # Try to capture the generated Cypher from various possible locations
            generated_cypher = None
            
            # Method 1: Check result metadata (most common location)
            if hasattr(result, 'metadata') and result.metadata and 'cypher' in result.metadata:
                generated_cypher = result.metadata['cypher']
                self.logger.info(f"🔍 Found Cypher in result.metadata['cypher']")
            
            # Method 2: Check result attributes
            elif hasattr(result, 'cypher_query'):
                generated_cypher = result.cypher_query
                self.logger.info(f"🔍 Found Cypher in result.cypher_query")
            elif hasattr(result, 'query'):
                generated_cypher = result.query
                self.logger.info(f"🔍 Found Cypher in result.query")
            
            # Method 3: Check retriever attributes
            elif hasattr(self._retriever, 'last_query'):
                generated_cypher = self._retriever.last_query
                self.logger.info(f"🔍 Found Cypher in _retriever.last_query")
            elif hasattr(self._retriever, '_last_cypher'):
                generated_cypher = self._retriever._last_cypher
                self.logger.info(f"🔍 Found Cypher in _retriever._last_cypher")
            
            # Store for access
            self.last_generated_cypher = generated_cypher
            
            # Log the query
            if generated_cypher:
                self.logger.info(f"🔍 Generated Cypher for '{query_text}': {generated_cypher}")
            else:
                self.logger.warning(f"⚠️  Could not capture generated Cypher for query: '{query_text}'")
                # Debug info
                if hasattr(result, 'metadata'):
                    self.logger.debug(f"Result metadata keys: {list(result.metadata.keys()) if result.metadata else 'None'}")
            
            # Enhance result with generated cypher if possible
            if hasattr(result, '__dict__') and generated_cypher:
                result.generated_cypher = generated_cypher
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error in Text2CypherRetriever for '{query_text}': {e}")
            raise
    
    def __getattr__(self, name):
        """Delegate other attributes to the wrapped retriever"""
        return getattr(self._retriever, name)


class EntityEmbedder(Component):
    """Simple entity embedder following neo4j-graphrag patterns - exact logic from original"""
    
    def __init__(self, embedder):
        """Initialize with just an embedder like TextChunkEmbedder"""
        self.embedder = embedder
    
    @validate_call
    async def run(self, graph: Neo4jGraph) -> Neo4jGraph:
        """Add entity embeddings to entities without existing embeddings."""
        if not graph.nodes:
            return graph
        
        # Find entities without embeddings (excluding Document and Chunk nodes)
        entities_to_embed = [
            node for node in graph.nodes
            if (
                node.label not in ['Document', 'Chunk'] and 
                (not node.embedding_properties or 
                 not node.embedding_properties.get('embedding'))
            )
        ]
        
        if entities_to_embed:
            # Create embedding text for each entity
            embedding_texts = []
            for entity in entities_to_embed:
                # Create descriptive text for entity embedding
                name = entity.properties.get('name', '')
                label = entity.label
                text = f"{label}: {name}" if name else label
                embedding_texts.append(text)
            
            # Get embeddings in batch
            embeddings = []
            for text in embedding_texts:
                embedding = self.embedder.embed_query(text)
                embeddings.append(embedding)
            
            # Add embeddings to entities
            for entity, embedding in zip(entities_to_embed, embeddings):
                if not entity.embedding_properties:
                    entity.embedding_properties = {}
                entity.embedding_properties['embedding'] = embedding
            
            print(f"✅ Added embeddings to {len(entities_to_embed)} entities")
        
        return graph


class RelationshipEmbedder(Component):
    """Simple relationship embedder following neo4j-graphrag patterns - exact logic from original"""
    
    def __init__(self, embedder):
        """Initialize with just an embedder like TextChunkEmbedder"""
        self.embedder = embedder
    
    @validate_call
    async def run(self, graph: Neo4jGraph) -> Neo4jGraph:
        """Add relationship embeddings to relationships without existing embeddings."""
        if not graph.relationships:
            return graph
        
        # Find relationships without embeddings
        rels_to_embed = [
            rel for rel in graph.relationships
            if (not rel.embedding_properties or 
                not rel.embedding_properties.get('embedding'))
        ]
        
        if rels_to_embed:
            # Create embedding text for each relationship
            embedding_texts = []
            for rel in rels_to_embed:
                text = f"{rel.start_node_id} {rel.type} {rel.end_node_id}"
                embedding_texts.append(text)
            
            # Get embeddings in batch
            embeddings = []
            for text in embedding_texts:
                embedding = self.embedder.embed_query(text)
                embeddings.append(embedding)
            
            # Add embeddings to relationships
            for rel, embedding in zip(rels_to_embed, embeddings):
                if not rel.embedding_properties:
                    rel.embedding_properties = {}
                rel.embedding_properties['embedding'] = embedding
            
            print(f"✅ Added embeddings to {len(rels_to_embed)} relationships")
        
        return graph
