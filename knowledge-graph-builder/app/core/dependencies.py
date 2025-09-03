# app/core/dependencies.py
"""
FastAPI dependencies for Neo4j GraphRAG components.
Simple, maintainable dependency injection following Neo4j GraphRAG patterns.
"""

from functools import lru_cache

from fastapi import Depends, HTTPException, status
from neo4j import Driver
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM

from app.core.neo4j_client import neo4j_client
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ==================== NEO4J GRAPHRAG CORE DEPENDENCIES ====================

@lru_cache()
def get_neo4j_driver() -> Driver:
    """
    Get Neo4j driver instance for Neo4j GraphRAG components.
    
    Returns:
        Raw Neo4j driver compatible with Neo4j GraphRAG retrievers
    """
    if not neo4j_client.driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j connection not available"
        )
    
    return neo4j_client.driver


@lru_cache()  
def get_openai_embedder() -> OpenAIEmbeddings:
    """
    Get OpenAI embedder instance for Neo4j GraphRAG components.
    
    Returns:
        Configured OpenAI embeddings instance
    """
    try:
        embedder = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL or "text-embedding-3-large",
            api_key=settings.OPENAI_API_KEY
        )
        
        logger.debug("OpenAI embedder created successfully")
        return embedder
        
    except Exception as e:
        logger.error(f"Failed to create OpenAI embedder: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI embeddings service not available"
        )


@lru_cache()
def get_openai_llm() -> OpenAILLM:
    """
    Get OpenAI LLM instance for Neo4j GraphRAG components.
    
    Returns:
        Configured OpenAI LLM instance
    """
    try:
        llm = OpenAILLM(
            model_name=settings.LLM_MODEL or "gpt-4",
            api_key=settings.OPENAI_API_KEY,
            model_params={
                "temperature": settings.LLM_TEMPERATURE or 0.1,
                "max_tokens": settings.LLM_MAX_TOKENS or 1500
            }
        )
        
        logger.debug("OpenAI LLM created successfully")
        return llm
        
    except Exception as e:
        logger.error(f"Failed to create OpenAI LLM: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI LLM service not available"
        )


# ==================== MULTI-TENANT RETRIEVAL DEPENDENCIES ====================

def get_retrieval_service_factory():
    """
    Factory for creating RetrievalService instances.
    
    Usage:
        @router.get("/search")
        async def search_endpoint(
            retrieval_factory = Depends(get_retrieval_service_factory),
            driver: Driver = Depends(get_neo4j_driver),
            embedder: OpenAIEmbeddings = Depends(get_openai_embedder)
        ):
            retrieval_service = retrieval_factory(driver=driver, embedder=embedder)
            return await retrieval_service.similarity_search_entities(...)
    """
    from app.services.retriever_service import RetrievalService
    
    def factory(
        driver: Driver = Depends(get_neo4j_driver),
        embedder: OpenAIEmbeddings = Depends(get_openai_embedder)
    ) -> RetrievalService:
        return RetrievalService(driver=driver, embedder=embedder)
    
    return factory


# ==================== PIPELINE DEPENDENCIES ====================

def get_kg_pipeline_factory():
    """
    Factory for creating multi-tenant KG pipeline instances.
    
    Usage in future pipeline service:
        @router.post("/process")
        async def process_documents(
            pipeline_factory = Depends(get_kg_pipeline_factory),
            graph_id: str = Path(...),
            driver: Driver = Depends(get_neo4j_driver),
            llm: OpenAILLM = Depends(get_openai_llm),
            embedder: OpenAIEmbeddings = Depends(get_openai_embedder)
        ):
            pipeline = pipeline_factory(
                driver=driver, llm=llm, embedder=embedder, graph_id=graph_id
            )
            return await pipeline.process_documents(documents)
    """
    # Import here to avoid circular imports
    def factory(
        graph_id: str,
        driver: Driver = Depends(get_neo4j_driver),
        llm: OpenAILLM = Depends(get_openai_llm),
        embedder: OpenAIEmbeddings = Depends(get_openai_embedder)
    ):
        # This will be implemented when we create the pipeline service
        from app.components.multi_tenant_components import create_multi_tenant_kg_writer
        
        return create_multi_tenant_kg_writer(
            driver=driver,
            graph_id=graph_id
        )
    
    return factory


# ==================== HEALTH CHECK DEPENDENCIES ====================

async def check_neo4j_health(driver: Driver = Depends(get_neo4j_driver)) -> bool:
    """
    Check Neo4j connection health for dependency validation.
    
    Returns:
        True if Neo4j is healthy, raises HTTPException otherwise
    """
    try:
        # Simple health check query
        with driver.session() as session:
            result = session.run("RETURN 1 as health")
            health_value = result.single()["health"]
            
        if health_value != 1:
            raise Exception("Health check returned unexpected value")
        
        return True
        
    except Exception as e:
        logger.error(f"Neo4j health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j database is not available"
        )


async def check_openai_health(embedder: OpenAIEmbeddings = Depends(get_openai_embedder)) -> bool:
    """
    Check OpenAI service health for dependency validation.
    
    Returns:
        True if OpenAI is healthy, raises HTTPException otherwise
    """
    try:
        # Simple test embedding
        test_embedding = embedder.embed_query("test")
        
        if not test_embedding or len(test_embedding) == 0:
            raise Exception("Embedding service returned empty result")
        
        return True
        
    except Exception as e:
        logger.error(f"OpenAI health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI embeddings service is not available"
        )


# ==================== BACKWARD COMPATIBILITY ====================

# Legacy aliases for existing code
get_neo4j_client = get_neo4j_driver  # Alias for backward compatibility
get_openai_embeddings = get_openai_embedder  # Alias for backward compatibility