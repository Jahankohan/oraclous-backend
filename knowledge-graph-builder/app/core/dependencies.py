# app/core/dependencies.py
"""
FastAPI dependencies for Neo4j GraphRAG components.
Simple, maintainable dependency injection following Neo4j GraphRAG patterns.
Updated to use dual driver architecture without @lru_cache for stateful connections.
"""

from fastapi import Depends, HTTPException, status
from neo4j import Driver, AsyncDriver
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM

from app.core.neo4j_client import neo4j_client
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ==================== NEO4J DRIVER DEPENDENCIES ====================

async def get_neo4j_async_driver() -> AsyncDriver:
    """
    Get Neo4j async driver for FastAPI endpoints.
    
    Returns:
        AsyncDriver instance for FastAPI web requests
        
    Raises:
        HTTPException: If connection is not available
        
    Note:
        Removed @lru_cache to prevent stale connections.
        Driver connection is managed by the Neo4jClient instance.
    """
    try:
        if not neo4j_client.async_driver:
            await neo4j_client.connect_async()
        
        if not neo4j_client.async_driver:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Neo4j async connection not available"
            )
        
        return neo4j_client.async_driver
        
    except Exception as e:
        logger.error(f"Failed to get async driver: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j async connection failed"
        )


def get_neo4j_driver() -> Driver:
    """
    Get Neo4j sync driver for Neo4j GraphRAG components.
    
    Returns:
        Sync Driver instance compatible with Neo4j GraphRAG retrievers
        
    Raises:
        HTTPException: If connection is not available
        
    Note:
        Removed @lru_cache to prevent stale connections.
        GraphRAG components require synchronous drivers.
    """
    try:
        if not neo4j_client.sync_driver:
            neo4j_client.connect_sync()
        
        if not neo4j_client.sync_driver:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Neo4j sync connection not available"
            )
        
        return neo4j_client.sync_driver
        
    except Exception as e:
        logger.error(f"Failed to get sync driver: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j sync connection failed"
        )


# ==================== NEO4J GRAPHRAG CORE DEPENDENCIES ====================

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


def get_openai_llm() -> OpenAILLM:
    """
    Get OpenAI LLM instance for Neo4j GraphRAG components.
    
    Returns:
        Configured OpenAI LLM instance
    """
    try:
        llm = OpenAILLM(
            model_name=getattr(settings, 'LLM_MODEL', "gpt-4"),
            api_key=settings.OPENAI_API_KEY,
            model_params={
                "temperature": getattr(settings, 'LLM_TEMPERATURE', 0.1),
                "max_tokens": getattr(settings, 'LLM_MAX_TOKENS', 1500)
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

async def check_neo4j_health() -> bool:
    """
    Check Neo4j connection health for dependency validation.
    
    Returns:
        True if Neo4j is healthy, raises HTTPException otherwise
        
    Note:
        Uses the Neo4jClient's built-in health check which tests both drivers.
    """
    try:
        health_info = await neo4j_client.health_check()
        
        if health_info["status"] != "healthy":
            raise Exception(f"Neo4j health check failed: {health_info.get('error', 'Unknown error')}")
        
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

# Legacy aliases for existing code that expects the old function names
get_neo4j_client = get_neo4j_driver  # Alias for sync driver (GraphRAG compatibility)
get_openai_embeddings = get_openai_embedder  # Alias for embedder