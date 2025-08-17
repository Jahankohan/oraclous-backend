from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional

from app.core.neo4j_client import Neo4jClient, get_neo4j_client
from app.models.requests import ChatRequest
from app.models.responses import ChatResponse, BaseResponse
from app.services.enhanced_chat_service import EnhancedChatService

router = APIRouter()

@router.post("/chat_bot", response_model=ChatResponse)
async def chat_with_bot(
    request: ChatRequest,
    neo4j: Neo4jClient = Depends(get_neo4j_client)
) -> ChatResponse:
    """Chat with the knowledge graph bot"""
    try:
        chat_service = EnhancedChatService(neo4j)
        return await chat_service.chat(
            message=request.message,
            mode=request.mode,
            file_names=request.file_names,
            session_id=request.session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@router.post("/clear_chat_bot", response_model=BaseResponse)
async def clear_chat_history(
    session_id: str,
    neo4j: Neo4jClient = Depends(get_neo4j_client)
) -> BaseResponse:
    """Clear chat history for a session"""
    try:
        chat_service = EnhancedChatService(neo4j)
        await chat_service.clear_chat_history(session_id)
        return BaseResponse(
            success=True,
            message="Chat history cleared successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear chat history: {str(e)}")

@router.get("/chat_history/{session_id}")
async def get_chat_history(
    session_id: str,
    limit: int = 50,
    neo4j: Neo4jClient = Depends(get_neo4j_client)
):
    """Get chat history for a session"""
    try:
        chat_service = EnhancedChatService(neo4j)
        return await chat_service.get_chat_history(session_id, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chat history: {str(e)}")

@router.post("/metric")
async def evaluate_chat_response(
    question: str,
    answer: str,
    contexts: List[str],
    neo4j: Neo4jClient = Depends(get_neo4j_client)
):
    """Evaluate chatbot response using RAGAS metrics"""
    try:
        # This would integrate with RAGAS for evaluation
        # For now, return placeholder metrics
        
        metrics = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.92,
            "context_precision": 0.78,
            "context_recall": 0.89
        }
        
        return {
            "metrics": metrics,
            "overall_score": sum(metrics.values()) / len(metrics)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@router.post("/additional_metrics")
async def evaluate_additional_metrics(
    question: str,
    answer: str,
    ground_truth: str,
    contexts: List[str],
    neo4j: Neo4jClient = Depends(get_neo4j_client)
):
    """Evaluate additional metrics with ground truth"""
    try:
        # This would integrate with RAGAS for additional metrics
        # For now, return placeholder metrics
        
        metrics = {
            "context_entity_recall": 0.87,
            "semantic_similarity": 0.91,
            "rouge_score": 0.76
        }
        
        return {
            "metrics": metrics,
            "ground_truth_comparison": {
                "exact_match": answer.strip().lower() == ground_truth.strip().lower(),
                "similarity_score": 0.89
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Additional evaluation failed: {str(e)}")
