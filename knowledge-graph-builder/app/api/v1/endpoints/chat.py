"""
Chat API endpoints using Neo4j GraphRAG
"""
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.api.dependencies import get_current_user
from app.services.chat_service import ChatService
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    """Chat request model"""
    query: str
    graph_id: str
    retriever_config: Optional[Dict[str, Any]] = None
    return_context: bool = False
    examples: str = ""


class ChatResponse(BaseModel):
    """Chat response model"""
    answer: str
    graph_id: str
    query: str
    success: bool
    retriever_result: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@router.post("/chat", response_model=ChatResponse)
async def chat_with_graph(
    request: ChatRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ChatResponse:
    """
    Chat with knowledge graph using Neo4j GraphRAG
    
    Args:
        request: Chat request with query and configuration
        current_user: Current authenticated user
        
    Returns:
        Chat response with answer and optional context
    """
    try:
        logger.info(f"Processing chat request for graph {request.graph_id}")
        
        # Initialize chat service for the specific graph
        chat_service = ChatService(graph_id=request.graph_id)
        
        # Perform GraphRAG search
        result = await chat_service.search(
            query_text=request.query,
            retriever_config=request.retriever_config or {"top_k": 5},
            return_context=request.return_context,
            examples=request.examples
        )
        
        return ChatResponse(
            answer=result.answer,
            graph_id=request.graph_id,
            query=request.query,
            success=True,
            retriever_result=result.retriever_result.model_dump() if result.retriever_result else None,
            metadata={
                "timestamp": "2024-01-01T00:00:00Z",  # Add proper timestamp
                "model": "gpt-4o"
            }
        )
        
    except Exception as e:
        logger.error(f"Chat error for graph {request.graph_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        )


@router.get("/modes", response_model=Dict[str, Any])
async def get_chat_modes(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get available chat modes and configurations"""
    return {
        "modes": ["vector_cypher"],
        "mode_descriptions": {
            "vector_cypher": "Vector similarity search combined with graph traversal"
        },
        "default": "vector_cypher"
    }