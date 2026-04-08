"""
Evaluation API endpoint.

POST /graphs/{graph_id}/evaluate

Scores a question/answer pair against a knowledge graph using RAGAS metrics:
  - faithfulness
  - answer_relevance
  - context_precision
  - context_recall (requires ground_truth)
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials

from app.api.dependencies import security
from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client
from app.schemas.evaluation_schemas import EvaluationRequest, EvaluationResponse
from app.services.auth_service import auth_service
from app.services.evaluation_service import EvaluationService
from app.services.graph_node_service import GraphNodeService

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/graphs/{graph_id}/evaluate",
    response_model=EvaluationResponse,
    summary="Evaluate chat quality with RAGAS metrics",
    responses={
        403: {"description": "Graph not found or access denied"},
        422: {"description": "Request body validation failed"},
        500: {"description": "Evaluation pipeline error"},
    },
)
async def evaluate_graph_response(
    graph_id: str,
    request: EvaluationRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> EvaluationResponse:
    """
    Score a question/answer pair against the specified knowledge graph.

    **How it works:**
    1. Retrieves relevant context from the graph using the default retriever.
    2. If `answer` is omitted, generates one via the chat pipeline.
    3. Runs RAGAS metrics against the retrieved context and answer.

    **Metrics:**
    - `faithfulness` — fraction of answer claims supported by retrieved context
    - `answer_relevance` — how well the answer addresses the question
    - `context_precision` — fraction of retrieved chunks that are relevant
    - `context_recall` — fraction of ground-truth facts captured (requires `ground_truth`)

    **Multi-tenancy:** graph ownership is verified before evaluation.
    Attempting to evaluate another user's graph returns 403.
    """
    user = await auth_service.verify_token(credentials.credentials)
    user_id = str(user["id"])

    graph_service = GraphNodeService(neo4j_client.sync_driver)
    graph = graph_service.get_graph(graph_id)
    if not graph or graph["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        svc = EvaluationService(graph_id=graph_id, user_id=user_id)
        result = await svc.evaluate(
            question=request.question,
            answer=request.answer,
            ground_truth=request.ground_truth,
            metrics=request.metrics,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Evaluation error for graph {graph_id}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(exc)}",
        )

    return EvaluationResponse(
        graph_id=graph_id,
        question=request.question,
        answer=result["answer"],
        retrieved_contexts=result["retrieved_contexts"],
        scores=result["scores"],
        overall=result["overall"],
        metrics_computed=result["metrics_computed"],
        is_grounded=result["is_grounded"],
        warnings=result["warnings"],
    )
