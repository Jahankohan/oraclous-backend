from typing import List, Dict, Any, Optional, Union
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import asyncio
from app.core.config import settings
from app.core.logging import get_logger
from app.services.credential_service import credential_service

logger = get_logger(__name__)

class EmbeddingService:
    """Service for generating text embeddings using various providers"""
    
    def __init__(self):
        self.embeddings = None
        self.provider = None
        self.model_name = None
        self.dimension = None
    
    async def initialize_embeddings(
        self, 
        provider: str = "openai",
        model: str = "text-embedding-3-small",
        user_id: Optional[str] = None
    ) -> bool:
        """Initialize embedding model"""
        
        try:
            if provider == "openai":
                # Get API key
                api_key = None
                if user_id:
                    api_key = await credential_service.get_openai_token(user_id)
                if not api_key:
                    api_key = settings.OPENAI_API_KEY
                
                if not api_key:
                    logger.error("No OpenAI API key available for embeddings")
                    return False
                
                self.embeddings = OpenAIEmbeddings(
                    api_key=api_key,
                    model=model,
                    dimensions=512 if model == "text-embedding-3-small" else 1536
                )
                self.dimension = 512 if model == "text-embedding-3-small" else 1536
                
            elif provider == "sentence-transformers":
                # Use local sentence transformers (no API key needed)
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model or "all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                self.dimension = 384  # Default for all-MiniLM-L6-v2
                
            else:
                logger.error(f"Unsupported embedding provider: {provider}")
                return False
            
            self.provider = provider
            self.model_name = model
            
            logger.info(f"Embeddings initialized: {provider} - {model} (dim: {self.dimension})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings {provider}: {e}")
            return False
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not self.embeddings:
            raise ValueError("Embeddings not initialized")
        
        try:
            if hasattr(self.embeddings, 'aembed_query'):
                embedding = await self.embeddings.aembed_query(text)
            else:
                # Fallback to sync method
                embedding = await asyncio.get_event_loop().run_in_executor(
                    None, self.embeddings.embed_query, text
                )
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not self.embeddings:
            raise ValueError("Embeddings not initialized")
        
        try:
            if hasattr(self.embeddings, 'aembed_documents'):
                embeddings = await self.embeddings.aembed_documents(texts)
            else:
                # Fallback to sync method
                embeddings = await asyncio.get_event_loop().run_in_executor(
                    None, self.embeddings.embed_documents, texts
                )
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def is_initialized(self) -> bool:
        """Check if embeddings are initialized"""
        return self.embeddings is not None

# Global embedding service
embedding_service = EmbeddingService()
