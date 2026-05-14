"""
Chat API endpoints (STORY-8 unified through AgentExecutor).

After STORY-8 ``/chat`` and ``/chat/stream`` route through the
``AgentExecutor`` engine, the same path that powers
``/agents/{id}/chat``. A synthetic in-memory "default agent" is built
per request — never persisted — and the user-facing ``mode`` field
maps to one of the retrieval tools added by STORY-8 (or earlier).

This file is intentionally short — most logic now lives in:
- ``app/services/chat_engine.py`` — mode→tool mapping + ChatResponse adaptation
- ``app/services/agent_executor.py`` — the unified execution engine
- ``app/services/agent_tools.py`` — retrieval tools

The legacy ``ChatService`` remains importable but is deprecated.
"""

import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials

from app.api.dependencies import security, verify_graph_access
from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client
from app.core.rate_limiter import limiter
from app.schemas.chat_schemas import (
    ChatMode,
    ChatModesResponse,
    ChatRequest,
    ChatResponse,
    RetrievalContext,
    SourceInfo,
    get_all_modes,
)
from app.services.agent_executor import AgentExecutor
from app.services.auth_service import auth_service
from app.services.chat_engine import (
    build_default_agent_config,
    derive_grounding,
    mode_to_retriever_label,
    tool_for_mode,
)

logger = get_logger(__name__)

router = APIRouter()


def _provenance_node_to_source(node: dict) -> SourceInfo:
    """Adapt a provenance node entry to the SourceInfo shape the
    /chat response surfaces.

    Provenance nodes have ``id`` and ``label`` only (the provenance
    collector strips everything else for size). For now we surface
    these as ``node_id`` + ``content``; richer fields (properties,
    relevance_score) would require provenance to track more.
    """
    return SourceInfo(
        node_id=node.get("id"),
        node_labels=None,
        relevance_score=None,
        content=node.get("label"),
        properties=None,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with a knowledge graph",
    responses={
        403: {"description": "Graph not found or access denied"},
        422: {"description": "Request body validation failed"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "LLM or retriever error"},
    },
)
@limiter.limit("30/minute")
async def chat_with_graph(
    request: Request,
    body: ChatRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> ChatResponse:
    """
    Send a natural language query to a knowledge graph and receive a
    grounded answer. After STORY-8 this routes through ``AgentExecutor``
    — the same engine that powers ``/agents/{id}/chat``.

    The ``mode`` field selects which retrieval tool the agent will use:
    - ``simple`` — fast vector search (``graph_search`` tool)
    - ``enhanced`` (default) — vector + graph traversal (``vector_cypher_search``)
    - ``hybrid`` / ``hybrid_plus`` — vector + fulltext + graph traversal
      (``hybrid_cypher_search``)
    - ``natural`` — text-to-Cypher (``cypher_query``)
    """
    user = await auth_service.verify_token(credentials.credentials)
    user_id = str(user["id"])
    await verify_graph_access(body.graph_id, "read", user_id)

    try:
        logger.info(f"chat: graph={body.graph_id} mode={body.mode}")
        agent_def = build_default_agent_config(
            graph_id=str(body.graph_id),
            mode=body.mode.value if body.mode else None,
        )
        executor = await AgentExecutor.from_chat_config(
            driver=neo4j_client.async_driver, agent_def=agent_def
        )
        agent_response = await executor.run(message=body.query, session_id=None)
    except Exception as e:
        logger.error(f"Chat error for graph {body.graph_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Chat processing failed: {e!s}"
        ) from None

    # Adapt AgentChatResponse → ChatResponse
    provenance = agent_response.provenance
    nodes = provenance.nodes or []
    is_grounded, confidence = derive_grounding(agent_response.response, nodes)

    sources: list[SourceInfo] = []
    if body.include_sources:
        sources = [_provenance_node_to_source(n) for n in nodes]

    context = None
    if body.return_context:
        context = RetrievalContext(
            retriever_type=mode_to_retriever_label(body.mode),
            sources=sources,
            total_results=len(nodes),
        )

    return ChatResponse(
        answer=agent_response.response,
        query=body.query,
        graph_id=body.graph_id,
        success=True,
        mode=body.mode.value if body.mode else "enhanced",
        retriever_type=mode_to_retriever_label(body.mode),
        is_grounded=is_grounded,
        confidence=confidence,
        # STORY-8 doesn't yet wrap the executor in a cache; the layer
        # is a clean follow-up (caching belongs around AgentExecutor.run
        # at the chat endpoint, not inside the executor itself).
        cache_hit=False,
        temporal_mode_applied=(
            body.temporal_mode.value if body.temporal_mode else None
        ),
        context=context,
        sources=sources if body.include_sources else None,
        conversation_id=body.conversation_id,
        metadata={
            "engine": "agent_executor",
            "tool_used": tool_for_mode(body.mode.value if body.mode else None),
        },
    )


@router.post(
    "/chat/stream",
    summary="Stream chat responses via SSE",
    response_description="Server-Sent Events stream (text/event-stream)",
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "SSE stream"},
        403: {"description": "Graph not found or access denied"},
        429: {"description": "Rate limit exceeded"},
    },
)
@limiter.limit("30/minute")
async def stream_chat_with_graph(
    request: Request,
    body: ChatRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> StreamingResponse:
    """Stream a chat response via Server-Sent Events.

    Event order (preserved from pre-STORY-8 ChatService contract):
    1. One ``source`` event per retrieved node
    2. Multiple ``answer_chunk`` events as the LLM streams its answer
    3. A single ``done`` event with final metadata
    4. An ``error`` event only on failure

    Implementation note: we dispatch the retrieval tool once
    synchronously to emit ``source`` events, then route the LLM call
    through ``executor.run_stream`` for token-by-token streaming. The
    executor re-dispatches the same tool during run_stream — one extra
    round trip in exchange for the source-first contract.
    """
    user = await auth_service.verify_token(credentials.credentials)
    user_id = str(user["id"])
    await verify_graph_access(body.graph_id, "read", user_id)

    async def event_generator():
        try:
            agent_def = build_default_agent_config(
                graph_id=str(body.graph_id),
                mode=body.mode.value if body.mode else None,
            )
            executor = await AgentExecutor.from_chat_config(
                driver=neo4j_client.async_driver, agent_def=agent_def
            )

            # Pre-fetch retrieval results so source events emit first.
            tool_name = tool_for_mode(body.mode.value if body.mode else None)
            tool_method = getattr(executor._toolkit, tool_name, None)
            pre_nodes = []
            if tool_method is not None:
                try:
                    pre_nodes = await tool_method(str(body.graph_id), query=body.query)
                except Exception as pre_exc:
                    logger.warning(
                        "pre-stream retrieval via %s failed: %s — continuing "
                        "without source events",
                        tool_name,
                        pre_exc,
                    )

            for node in pre_nodes:
                node_props = getattr(node, "properties", {}) or {}
                source_payload = {
                    "type": "source",
                    "node_id": node.id,
                    "node_labels": [node.label] if node.label else None,
                    "relevance_score": node_props.get("score"),
                    "content": node_props.get("text") or node.label,
                }
                yield f"data: {json.dumps(source_payload)}\n\n"

            # Stream the answer tokens via the executor.
            full_response = ""
            async for token, prov_payload in executor.run_stream(
                message=body.query, session_id=None
            ):
                if token is not None:
                    full_response += token
                    yield (
                        "data: "
                        + json.dumps({"type": "answer_chunk", "text": token})
                        + "\n\n"
                    )
                else:
                    nodes = prov_payload.nodes if prov_payload else []
                    is_grounded, confidence = derive_grounding(
                        full_response,
                        [{"id": n.get("id"), "label": n.get("label")} for n in nodes],
                    )
                    done_payload = {
                        "type": "done",
                        "confidence": confidence,
                        "is_grounded": is_grounded,
                        "retriever_used": mode_to_retriever_label(body.mode),
                    }
                    yield f"data: {json.dumps(done_payload)}\n\n"
        except Exception as e:
            logger.error(f"Stream chat error for graph {body.graph_id}: {e}")
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
    Return all available chat modes with descriptions and use cases.
    """
    await auth_service.verify_token(credentials.credentials)
    return ChatModesResponse(
        modes=get_all_modes(),
        default_mode=ChatMode.ENHANCED,
        graph_capabilities={
            "has_fulltext_indexes": True,
            "has_vector_indexes": True,
            "supports_cypher": True,
        },
    )
