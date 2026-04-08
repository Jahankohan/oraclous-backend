"""
Chat API endpoints using Neo4j GraphRAG with Enhanced Retriever Support
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials

from app.api.dependencies import security, verify_graph_access
from app.services.auth_service import auth_service
from app.services.chat_service import ChatService
from app.core.logging import get_logger
from app.schemas.chat_schemas import (
    ChatRequest, ChatResponse, ChatModesResponse, ErrorResponse,
    get_mode_mapping, get_all_modes, ChatMode, SourceInfo, RetrievalContext
)
from app.services.retriever_factory import RetrieverType

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with a knowledge graph",
    responses={
        403: {"description": "Graph not found or access denied"},
        422: {"description": "Request body validation failed"},
        500: {"description": "LLM or retriever error"},
    },
)
async def chat_with_graph(
    request: ChatRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> ChatResponse:
    """
    Send a natural language query to a knowledge graph and receive a grounded answer.

    Responses are strictly grounded in retrieved graph data — when the graph does
    not contain sufficient context, the response sets `is_grounded: false` and
    explains what information is missing, rather than allowing the LLM to speculate.

    **Choosing a mode:**
    - `enhanced` (default) — vector search + graph traversal, best for most questions
    - `simple` — pure vector search, fastest
    - `hybrid` / `hybrid_plus` — adds full-text search; requires full-text indexes
    - `natural` — translates natural language to Cypher for precise graph queries

    Set `retriever_type` directly to bypass the mode system and use a specific
    retriever with custom configuration.
    """
    # Verify token after Pydantic body validation so invalid bodies return 422, not 401.
    user = await auth_service.verify_token(credentials.credentials)
    user_id = str(user["id"])

    # ReBAC access check — read level required for chat
    await verify_graph_access(request.graph_id, "read", user_id)

    try:
        logger.info(f"Processing chat request for graph {request.graph_id}")

        # Determine retriever type from mode or explicit type.
        retriever_type = request.retriever_type
        if not retriever_type:
            mode_mapping = get_mode_mapping()
            retriever_type = mode_mapping.get(
                request.mode or ChatMode.ENHANCED, RetrieverType.VECTOR_CYPHER
            )

        chat_service = ChatService(
            graph_id=request.graph_id,
            retriever_type=retriever_type,
            retriever_config=request.retriever_config.model_dump() if request.retriever_config else None
        )
        await chat_service.initialize()

        result = await chat_service.search(
            query_text=request.query,
            retriever_config=request.retriever_config.model_dump() if request.retriever_config else {"top_k": 5},
            return_context=request.return_context,
            examples=request.examples,
            temporal_filter=request.temporal_filter,
        )

        # Map GroundedSearchResult sources → SourceInfo schema objects.
        sources: List[SourceInfo] = []
        if request.include_sources:
            for src in result.sources:
                sources.append(SourceInfo(
                    node_id=src.get("node_id"),
                    node_labels=src.get("node_labels"),
                    relevance_score=src.get("relevance_score"),
                    content=src.get("content"),
                    properties=src.get("properties"),
                ))

        context = None
        if request.return_context and result.retriever_result:
            context = RetrievalContext(
                retriever_type=retriever_type.value,
                sources=sources,
                total_results=len(result.sources),
            )

        return ChatResponse(
            answer=result.answer,
            query=request.query,
            graph_id=request.graph_id,
            success=True,
            mode=request.mode.value if request.mode else "enhanced",
            retriever_type=result.retriever_used,
            is_grounded=result.is_grounded,
            confidence=result.confidence,
            context=context,
            sources=sources if request.include_sources else None,
            conversation_id=request.conversation_id,
            metadata={
                "model": "gpt-4o",
                "include_cypher": request.include_cypher,
                "return_context": request.return_context,
            }
        )

    except Exception as e:
        logger.error(f"Chat error for graph {request.graph_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        )


@router.post(
    "/chat/stream",
    summary="Stream chat responses via SSE",
    response_description="Server-Sent Events stream (text/event-stream)",
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "SSE stream"},
        403: {"description": "Graph not found or access denied"},
    },
)
async def stream_chat_with_graph(
    request: ChatRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> StreamingResponse:
    """
    Stream a chat response in real-time using Server-Sent Events (SSE).

    Identical to `POST /chat` but returns a streaming response for progressive
    display in UIs. Events are emitted in this order:

    1. One `source` event per retrieved graph node (before the answer begins)
    2. Multiple `answer_chunk` events with word-level answer text
    3. A single `done` event with final metadata
    4. An `error` event only on failure

    **Event shapes:**
    ```
    {"type": "source", "node_id": "...", "node_labels": [...], "relevance_score": 0.94, "content": "..."}
    {"type": "answer_chunk", "text": "TechNova "}
    {"type": "done", "confidence": 0.91, "is_grounded": true, "retriever_used": "vector_cypher"}
    {"type": "error", "message": "..."}
    ```

    The stream is strictly graph-grounded — if the knowledge graph lacks sufficient
    context, the answer chunks will say so and `is_grounded` will be `false` in the
    `done` event.
    """
    # Verify token after Pydantic body validation so invalid bodies return 422, not 401.
    user = await auth_service.verify_token(credentials.credentials)
    user_id = str(user["id"])

    # ReBAC access check — read level required for streaming chat
    await verify_graph_access(request.graph_id, "read", user_id)

    retriever_type = request.retriever_type
    if not retriever_type:
        mode_mapping = get_mode_mapping()
        retriever_type = mode_mapping.get(
            request.mode or ChatMode.ENHANCED, RetrieverType.VECTOR_CYPHER
        )

    chat_service = ChatService(
        graph_id=request.graph_id,
        retriever_type=retriever_type,
        retriever_config=request.retriever_config.model_dump() if request.retriever_config else None,
    )

    async def event_generator():
        try:
            await chat_service.initialize()
            async for chunk in chat_service.stream_search(
                query_text=request.query,
                retriever_config=request.retriever_config.model_dump() if request.retriever_config else {"top_k": 5},
            ):
                yield chunk
        except Exception as e:
            import json
            logger.error(f"Stream chat error for graph {request.graph_id}: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get(
    "/modes",
    response_model=ChatModesResponse,
    summary="List available chat modes",
)
async def get_chat_modes(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> ChatModesResponse:
    """
    Return all available chat modes with descriptions, use cases, and requirements.

    Use this endpoint to discover which modes are available and whether your
    graph has the indexes required for `hybrid` and `hybrid_plus` modes.
    The `default_mode` field indicates the recommended starting point.
    """
    await auth_service.verify_token(credentials.credentials)
    return ChatModesResponse(
        modes=get_all_modes(),
        default_mode=ChatMode.ENHANCED,
        graph_capabilities={
            "has_fulltext_indexes": True,  # This should be checked per graph
            "has_vector_indexes": True,
            "supports_cypher": True
        }
    )