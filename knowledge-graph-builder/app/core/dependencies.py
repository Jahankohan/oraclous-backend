"""
Dependency injection for FastAPI
"""
from typing import AsyncGenerator
from fastapi import Depends

from app.core.neo4j_pool import Neo4jPool, get_neo4j_pool
from app.services.document_service import DocumentService
from app.services.extraction_service import ExtractionService
from app.services.graph_service import GraphService
from app.services.chat_service import ChatService
from app.services.embedding_service import EmbeddingService


async def get_neo4j() -> AsyncGenerator[Neo4jPool, None]:
    """Get Neo4j pool dependency"""
    pool = await get_neo4j_pool()
    yield pool


async def get_document_service(
    neo4j: Neo4jPool = Depends(get_neo4j)
) -> DocumentService:
    """Get document service dependency"""
    return DocumentService(neo4j)


async def get_extraction_service(
    neo4j: Neo4jPool = Depends(get_neo4j)
) -> ExtractionService:
    """Get extraction service dependency"""
    return ExtractionService(neo4j)


async def get_graph_service(
    neo4j: Neo4jPool = Depends(get_neo4j)
) -> GraphService:
    """Get graph service dependency"""
    return GraphService(neo4j)


async def get_chat_service(
    neo4j: Neo4jPool = Depends(get_neo4j)
) -> ChatService:
    """Get chat service dependency"""
    return ChatService(neo4j)


async def get_embedding_service() -> EmbeddingService:
    """Get embedding service dependency"""
    return EmbeddingService()