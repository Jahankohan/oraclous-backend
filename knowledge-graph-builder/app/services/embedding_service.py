import asyncio
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings

from app.config.settings import get_settings, EmbeddingModel
from app.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating embeddings"""
    
    def __init__(self):
        self.settings = get_settings()
        self._embedding_model = None
        self._dimensions = self._get_embedding_dimensions()
    
    def _get_embedding_dimensions(self) -> int:
        """Get embedding dimensions based on model"""
        if self.settings.embedding_model == EmbeddingModel.SENTENCE_TRANSFORMER:
            return 384
        elif self.settings.embedding_model == EmbeddingModel.OPENAI:
            return 1536
        elif self.settings.embedding_model == EmbeddingModel.VERTEX_AI:
            return 768
        return 384
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    def _get_embedding_model(self):
        """Get or create embedding model"""
        if self._embedding_model is not None:
            return self._embedding_model
        
        try:
            if self.settings.embedding_model == EmbeddingModel.SENTENCE_TRANSFORMER:
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            elif self.settings.embedding_model == EmbeddingModel.OPENAI:
                if not self.settings.openai_api_key:
                    raise EmbeddingError("OpenAI API key not configured")
                
                self._embedding_model = OpenAIEmbeddings(
                    model="text-embedding-ada-002",
                    api_key=self.settings.openai_api_key
                )
            
            elif self.settings.embedding_model == EmbeddingModel.VERTEX_AI:
                if not self.settings.gemini_api_key:
                    raise EmbeddingError("Vertex AI credentials not configured")
                
                self._embedding_model = VertexAIEmbeddings(
                    model_name="textembedding-gecko@003"
                )
            
            return self._embedding_model
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise EmbeddingError(f"Failed to initialize embedding model: {e}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            if not texts:
                return []
            
            model = self._get_embedding_model()
            
            if self.settings.embedding_model == EmbeddingModel.SENTENCE_TRANSFORMER:
                # Use sentence transformers
                embeddings = await asyncio.to_thread(model.encode, texts)
                return embeddings.tolist()
            
            else:
                # Use LangChain embeddings (OpenAI, Vertex AI)
                embeddings = await model.aembed_documents(texts)
                return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {e}")
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a single query"""
        try:
            model = self._get_embedding_model()
            
            if self.settings.embedding_model == EmbeddingModel.SENTENCE_TRANSFORMER:
                embedding = await asyncio.to_thread(model.encode, [query])
                return embedding[0].tolist()
            else:
                embedding = await model.aembed_query(query)
                return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise EmbeddingError(f"Failed to generate query embedding: {e}")
    
    async def similarity_search(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Perform vector similarity search in Neo4j"""
        from app.core.neo4j_client import get_neo4j_client
        
        try:
            # This would be injected properly in a real implementation
            # For now, we'll assume access to Neo4j client
            query = """
            CALL db.index.vector.queryNodes('vector', $limit, $queryEmbedding) 
            YIELD node AS chunk, score
            MATCH (d:Document)-[:HAS_CHUNK]->(chunk)
            RETURN chunk.id as chunkId, 
                   chunk.text as text, 
                   d.fileName as fileName,
                   score
            ORDER BY score DESC
            """
            
            # This is a placeholder - would need proper dependency injection
            # result = neo4j.execute_query(query, {
            #     "queryEmbedding": query_embedding,
            #     "limit": limit
            # })
            
            # return result
            return []  # Placeholder
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise EmbeddingError(f"Vector search failed: {e}")
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm_a = np.linalg.norm(vec1)
            norm_b = np.linalg.norm(vec2)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            similarity = dot_product / (norm_a * norm_b)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
