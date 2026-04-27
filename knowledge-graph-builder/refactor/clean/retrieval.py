"""Retrieval system factory following neo4j-graphrag patterns - exact same functionality as original."""

from typing import Dict, Any
from neo4j import Driver

from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.retrievers import (
    VectorRetriever,
    VectorCypherRetriever,
    HybridRetriever,
    Text2CypherRetriever
)
from components import LoggingText2CypherRetriever


class RetrievalSystemFactory:
    """Factory for creating various retrievers for GraphRAG chatbot - exact copy from original functionality"""
    
    def __init__(self, driver: Driver, embedder: Embedder, llm: LLMInterface, database: str = "neo4j"):
        self.driver = driver
        self.embedder = embedder
        self.llm = llm
        self.database = database
        import logging
        self.logger = logging.getLogger(__name__)
    
    def create_vector_retriever(self) -> VectorRetriever:
        """Create vector retriever for semantic chunk search."""
        return VectorRetriever(
            driver=self.driver,
            index_name="text_embeddings_primary",
            embedder=self.embedder,
            return_properties=["text", "chunk_index"]
        )
    
    def create_vector_cypher_retriever(self) -> VectorCypherRetriever:
        """Create vector + cypher retriever for graph-aware context."""
        return VectorCypherRetriever(
            driver=self.driver,
            index_name="text_embeddings_primary",
            embedder=self.embedder,
            retrieval_query="""
            WITH node AS chunk, score
            MATCH (chunk)<-[:FROM_CHUNK]-(entity:__Entity__)-[r]->(related_entity:__Entity__)
            WHERE r.confidence > 0.5
            RETURN 
                chunk.text AS context,
                chunk.chunk_index AS chunk_index,
                collect(DISTINCT {
                    entity: entity.name,
                    type: labels(entity)[0],
                    relationship: type(r),
                    related_entity: related_entity.name,
                    confidence: r.confidence
                }) AS knowledge_graph_context,
                score
            ORDER BY score DESC
            """
        )
    
    def create_entity_retriever(self) -> VectorRetriever:
        """Create entity-focused vector retriever."""
        return VectorRetriever(
            driver=self.driver,
            index_name="entity_embeddings",
            embedder=self.embedder,
            return_properties=["name", "description", "type"]
        )
    
    def create_hybrid_retriever(self) -> HybridRetriever:
        """Create hybrid retriever combining vector and full-text search."""
        return HybridRetriever(
            driver=self.driver,
            vector_index_name="text_embeddings_primary", 
            fulltext_index_name="chunk_text_fulltext",
            embedder=self.embedder
        )
    
    def create_text2cypher_retriever(self) -> LoggingText2CypherRetriever:
        """Create text-to-cypher retriever for natural language queries - with logging wrapper like original."""
        # Get enhanced schema from database
        with self.driver.session(database=self.database) as session:
            schema_result = session.run("CALL db.schema.visualization()")
            enhanced_schema = self._build_enhanced_schema(schema_result.data())
        
        custom_prompt = """
        Generate a Cypher query for a Neo4j graph database based on this natural language request.

        Available Schema:
        {schema}

        Query Examples:
        {examples}

        User Question:
        {query_text}

        Generate a single Cypher query that answers the user's question. The query must:
        - Use only the nodes, relationships, and properties shown in the schema
        - Start with MATCH, CREATE, MERGE, or other valid Cypher keywords
        - Be syntactically correct and executable
        - Return relevant data to answer the question

        Output ONLY the raw Cypher query without any JSON formatting, code blocks, or explanations. Example:
        MATCH (n:Entity) RETURN n LIMIT 10
        """
        
        text2cypher_retriever = Text2CypherRetriever(
            driver=self.driver,
            llm=self.llm,
            neo4j_schema=enhanced_schema,
            custom_prompt=custom_prompt,
            examples=[
                "Find all people working for TechNova Corporation -> MATCH (e1:__Entity__)-[r:WORKS_FOR|EMPLOYED_BY|CEO_OF|CTO_OF]->(e2:__Entity__ {name: 'TechNova'}) RETURN e1.name, type(r), e2.name",
                "Show companies founded in Austin -> MATCH (e1:__Entity__)-[r:LOCATED_IN|FOUNDED_IN]->(e2:__Entity__ {name: 'Austin'}) RETURN e1.name, type(r)",
                "Find entities related to artificial intelligence -> MATCH (e:__Entity__) WHERE e.name CONTAINS 'AI' OR e.name CONTAINS 'artificial intelligence' OR e.description CONTAINS 'artificial intelligence' RETURN e.name, e.description LIMIT 10",
                "Get document context for entity -> MATCH (e:__Entity__ {name: $entity_name})-[:FROM_CHUNK]->(c:Chunk)-[:FROM_DOCUMENT]->(d:Document) RETURN d.path, c.text",
                "Show entity relationships with high confidence -> MATCH (e1:__Entity__)-[r]->(e2:__Entity__) WHERE r.confidence > 0.8 RETURN e1.name, type(r), e2.name, r.confidence ORDER BY r.confidence DESC LIMIT 20"
            ]
        )
        
        # Wrap with logging to capture generated Cypher queries - exact same as original
        return LoggingText2CypherRetriever(text2cypher_retriever, self.logger)
    
    def create_all_retrievers(self) -> Dict[str, Any]:
        """Create all retrieval strategies - exact same as original."""
        try:
            # 1. Primary Vector Retriever for semantic chunk search
            vector_retriever = VectorRetriever(
                driver=self.driver,
                index_name="text_embeddings_primary",
                embedder=self.embedder,
                return_properties=["text", "chunk_index"]
            ) 
            
            # 2. Enhanced Vector + Cypher Retriever for graph-aware context
            vector_cypher_retriever = VectorCypherRetriever(
                driver=self.driver,
                index_name="text_embeddings_primary",
                embedder=self.embedder,
                retrieval_query="""
                WITH node AS chunk, score
                MATCH (chunk)<-[:FROM_CHUNK]-(entity:__Entity__)-[r]->(related_entity:__Entity__)
                WHERE r.confidence > 0.5
                RETURN 
                    chunk.text AS context,
                    chunk.chunk_index AS chunk_index,
                    collect(DISTINCT {
                        entity: entity.name,
                        type: labels(entity)[0],
                        relationship: type(r),
                        related_entity: related_entity.name,
                        confidence: r.confidence
                    }) AS knowledge_graph_context,
                    score
                ORDER BY score DESC
                """
            ) 
            
            # 3. Entity-focused Vector Retriever for entity-centric queries
            entity_retriever = VectorRetriever(
                driver=self.driver,
                index_name="entity_embeddings",
                embedder=self.embedder,
                return_properties=["name", "description", "type"]
            )
            
            # 4. Hybrid Retriever combining vector and full-text search
            hybrid_retriever = HybridRetriever(
                driver=self.driver,
                vector_index_name="text_embeddings_primary", 
                fulltext_index_name="chunk_text_fulltext",
                embedder=self.embedder
            ) 
            
            # 5. Enhanced Text2Cypher with comprehensive schema
            text2cypher_retriever = self.create_text2cypher_retriever()
            
            return {
                "retrievers_created": 5,
                "vector_retriever": vector_retriever,
                "vector_cypher_retriever": vector_cypher_retriever,
                "entity_retriever": entity_retriever,
                "hybrid_retriever": hybrid_retriever,
                "text2cypher_retriever": text2cypher_retriever,
                "chatbot_ready": True,
                "supported_queries": [
                    "semantic_similarity",
                    "graph_traversal", 
                    "entity_search",
                    "hybrid_search",
                    "natural_language_to_cypher"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Retrieval system creation failed: {e}")
            raise
    
    def _build_enhanced_schema(self, schema_data) -> str:
        """Build enhanced schema description for Text2Cypher - exact same as original."""
        schema_description = """
        # Knowledge Graph Schema for GraphRAG
        
        ## Core Node Types:
        - Document: Represents source documents
          Properties: path (string), metadata (map)
        
        - Chunk: Text chunks from documents  
          Properties: text (string), chunk_index (integer), embedding (vector)
        
        - __Entity__: Extracted entities (people, organizations, concepts, etc.)
          Properties: name (string), description (string), type (string), embedding (vector)
        
        ## Key Relationships:
        - FROM_DOCUMENT: Links chunks to their source documents (Document)-[:FROM_DOCUMENT]->(Chunk)
        - FROM_CHUNK: Links entities to the chunks they were extracted from (Chunk)<-[:FROM_CHUNK]-(__Entity__)
        - Dynamic relationships between entities: (__Entity__)-[various_types]->(__Entity__)
          Common types: WORKS_FOR, LOCATED_IN, FOUNDED_BY, DEVELOPS, PARTNERS_WITH, etc.
        - SAME_AS: Entity resolution links for duplicate entities
        
        ## Important Notes:
        - All extracted entities use the __Entity__ label regardless of type (person, organization, concept)
        - Entity relationships have confidence scores (r.confidence property)
        - Chunks do not have a 'source' property - use path from related Document instead
        - Use vector similarity searches for semantic matching
        
        ## Vector Indexes Available:
        - text_embeddings_primary: For semantic search on chunks
        - entity_embeddings: For semantic search on entities
        
        ## Fulltext Indexes Available:  
        - chunk_text_fulltext: For keyword search on chunk text
        - entity_text_fulltext: For keyword search on entity names/descriptions
        """
        
        return schema_description
