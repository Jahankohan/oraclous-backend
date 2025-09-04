"""
Chat Service using Neo4j GraphRAG
Simple, clean implementation following Neo4j GraphRAG patterns
"""
from typing import Dict, Any, Optional
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import VectorCypherRetriever 
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.generation.types import RagResultModel

from app.core.neo4j_client import neo4j_client
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ChatService:
    """
    Chat service using Neo4j GraphRAG
    
    Simple wrapper around Neo4j GraphRAG components for multi-tenant usage
    """
    
    def __init__(self, graph_id: str):
        """
        Initialize chat service for specific graph
        
        Args:
            graph_id: Target graph identifier for multi-tenant isolation
        """
        self.graph_id = graph_id
        
        # Initialize embedder
        self.embedder = OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY,
            model="text-embedding-3-large"
        )
        
        # Initialize LLM
        self.llm = OpenAILLM(
            model_name="gpt-4o",
            api_key=settings.OPENAI_API_KEY,
            model_params={"temperature": 0.1}
        )
        
        # Initialize retriever with graph traversal
        self.retriever = self._create_retriever()
        
        # Initialize GraphRAG
        self.rag = GraphRAG(
            retriever=self.retriever,
            llm=self.llm
        )
        
        logger.info(f"ChatService initialized for graph {graph_id}")
    
    def _create_retriever(self) -> VectorCypherRetriever:
        """Create VectorCypherRetriever with multi-tenant support"""
        
        # Retrieval query that leverages your graph structure
        retrieval_query = f"""
        // Multi-tenant filter for graph_id
        WHERE node.graph_id = '{self.graph_id}'
        
        // Get entities that are connected to this chunk
        MATCH (entity {{graph_id: '{self.graph_id}'}})-[:FROM_CHUNK]->(node)
        OPTIONAL MATCH (node)-[:FROM_DOCUMENT]->(document {{graph_id: '{self.graph_id}'}})
        
        // Traverse entity relationships for context
        OPTIONAL MATCH (entity)-[r]-(related_entity {{graph_id: '{self.graph_id}'}})
        
        RETURN node.text as text,
               document.path as document_path,
               collect(DISTINCT entity.name) as entities,
               collect(DISTINCT {{
                   entity: related_entity.name,
                   relationship: type(r)
               }}) as relationships,
               score
        ORDER BY score DESC
        """
        
        # Ensure sync driver is connected
        if neo4j_client.sync_driver is None:
            neo4j_client.connect_sync()
            
        if neo4j_client.sync_driver is None:
            raise ConnectionError("Failed to establish Neo4j sync driver connection")
            
        return VectorCypherRetriever(
            driver=neo4j_client.sync_driver,
            index_name="text_embeddings_primary",  # Use the correct index name for chunks
            retrieval_query=retrieval_query,
            embedder=self.embedder,
            neo4j_database=settings.NEO4J_DATABASE
        )
    
    async def search(
        self,
        query_text: str,
        retriever_config: Optional[Dict[str, Any]] = None,
        return_context: bool = False,
        examples: str = ""
    ) -> RagResultModel:
        """
        Perform GraphRAG search
        
        Args:
            query_text: User's question
            retriever_config: Configuration for retriever (e.g., top_k)
            return_context: Whether to return retrieval context
            examples: Examples for few-shot learning
            
        Returns:
            GraphRAG result with answer and optional context
        """
        try:
            logger.info(f"Processing search query for graph {self.graph_id}")
            
            # Use Neo4j GraphRAG search
            result = self.rag.search(
                query_text=query_text,
                retriever_config=retriever_config or {"top_k": 5},
                return_context=return_context,
                examples=examples
            )
            
            logger.info(f"GraphRAG search completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"GraphRAG search failed: {e}")
            # Return fallback response
            return RagResultModel(
                answer=f"I encountered an error while searching the knowledge graph: {str(e)}",
                retriever_result=None
            )
