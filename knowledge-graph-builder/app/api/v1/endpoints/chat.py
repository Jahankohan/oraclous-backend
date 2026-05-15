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
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    get_chat_db,
    get_current_user_id,
    verify_graph_access,
)
from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client
from app.core.rate_limiter import limiter
from app.schemas.chat_schemas import (
    ChatMode,  # noqa: F401  (re-exported for backwards-compat callers)
    ChatModesResponse,
    ChatRequest,
    ChatResponse,
    RetrievalContext,
    SourceInfo,
    get_all_modes,
)
from app.services.agent_executor import AgentExecutor
from app.services.chat_engine import (
    build_default_agent_config,
    derive_grounding,
    mode_to_retriever_label,
    tool_for_mode,
)
from app.services.chat_history_service import ChatHistoryService

logger = get_logger(__name__)

router = APIRouter()

_chat_history = ChatHistoryService()


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
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_chat_db),
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

    STORY-031: the turn (user + assistant + tool calls) is persisted
    to Postgres. If ``conversation_id`` is omitted the backend creates
    a fresh conversation and returns its id in the response.
    """
    await verify_graph_access(body.graph_id, "read", user_id)

    conv = await _chat_history.get_or_create_conversation(
        db,
        user_id=user_id,
        graph_id=body.graph_id,
        agent_id=None,
        conversation_id=body.conversation_id,
        first_message=body.query,
    )
    # Load prior turns BEFORE writing the new user message so the
    # executor sees only the historical context (not the message it's
    # about to answer).
    from app.core.config import settings as _settings

    context_history = await _chat_history.load_context(
        db,
        conversation_id=conv.id,
        max_turns=_settings.CHAT_CONTEXT_MAX_TURNS,
        max_tokens=_settings.CHAT_CONTEXT_MAX_TOKENS,
    )
    await _chat_history.write_user_message(
        db, conversation_id=conv.id, content=body.query
    )

    started = time.monotonic()
    try:
        logger.info(f"chat: graph={body.graph_id} mode={body.mode}")
        agent_def = build_default_agent_config(
            graph_id=str(body.graph_id),
            mode=body.mode.value if body.mode else None,
        )
        executor = await AgentExecutor.from_chat_config(
            driver=neo4j_client.async_driver, agent_def=agent_def
        )
        agent_response = await executor.run(
            message=body.query,
            session_id=None,
            history=context_history or None,
        )
    except Exception as e:
        latency_ms = int((time.monotonic() - started) * 1000)
        logger.error(f"Chat error for graph {body.graph_id}: {e}")
        # Persist the failed assistant turn so the conversation history
        # reflects the attempt rather than dropping it silently.
        await _chat_history.write_assistant_message(
            db,
            conversation_id=conv.id,
            content="",
            latency_ms=latency_ms,
            reasoning_mode=body.mode.value if body.mode else "enhanced",
            retriever_used=mode_to_retriever_label(body.mode),
            error=str(e),
        )
        await db.commit()
        raise HTTPException(
            status_code=500, detail=f"Chat processing failed: {e!s}"
        ) from None

    latency_ms = int((time.monotonic() - started) * 1000)
    provenance = agent_response.provenance
    nodes = provenance.nodes or []
    is_grounded, confidence = derive_grounding(agent_response.response, nodes)

    sources: list[SourceInfo] = []
    if body.include_sources:
        sources = [_provenance_node_to_source(n) for n in nodes]

    asst_msg = await _chat_history.write_assistant_message(
        db,
        conversation_id=conv.id,
        content=agent_response.response,
        latency_ms=latency_ms,
        reasoning_mode=body.mode.value if body.mode else "enhanced",
        retriever_used=mode_to_retriever_label(body.mode),
        sources=[s.model_dump(mode="json") for s in sources] if sources else None,
    )

    for idx, tc in enumerate(provenance.tool_calls or []):
        try:
            await _chat_history.write_tool_call(
                db,
                message_id=asst_msg.id,
                sequence_index=idx,
                tool_name=tc.get("name", "unknown"),
                args=None,  # provenance doesn't surface tool args today
                result={"node_count": tc.get("node_count")}
                if tc.get("node_count") is not None
                else None,
            )
        except Exception:
            # Tool-call audit is best-effort — never fail the user
            # response because a tool row couldn't be written.
            logger.exception("failed to persist tool call %s", tc.get("name"))

    await db.commit()

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
        conversation_id=str(conv.id),
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
    user_id: str = Depends(get_current_user_id),
) -> StreamingResponse:
    """Stream a chat response via Server-Sent Events.

    Event order:

    1. A ``meta`` event carrying the persisted ``conversation_id``
    2. One ``source`` event per retrieved node
    3. Multiple ``answer_chunk`` events as the LLM streams its answer
    4. A single ``done`` event with final metadata
    5. An ``error`` event only on failure

    Implementation note: we dispatch the retrieval tool once
    synchronously to emit ``source`` events, then route the LLM call
    through ``executor.run_stream`` for token-by-token streaming.

    STORY-031: the user turn is persisted in a setup transaction
    before streaming begins. The assistant turn + tool calls are
    persisted in a separate transaction after ``done`` fires, or as a
    cancelled/error turn if the stream aborts.
    """
    from sqlalchemy import text as sql_text

    from app.core.database import async_session_maker

    await verify_graph_access(body.graph_id, "read", user_id)

    # Persist the user turn + resolve conversation BEFORE streaming.
    # This commits in its own transaction so the user's message is
    # never lost even if the stream dies before the first chunk.
    from app.core.config import settings as _settings

    async with async_session_maker() as setup_session:
        await setup_session.execute(
            sql_text("SELECT set_config('app.current_user_id', :uid, true)").bindparams(
                uid=user_id
            )
        )
        conv = await _chat_history.get_or_create_conversation(
            setup_session,
            user_id=user_id,
            graph_id=body.graph_id,
            agent_id=None,
            conversation_id=body.conversation_id,
            first_message=body.query,
        )
        # Load context BEFORE writing the new user message.
        context_history = await _chat_history.load_context(
            setup_session,
            conversation_id=conv.id,
            max_turns=_settings.CHAT_CONTEXT_MAX_TURNS,
            max_tokens=_settings.CHAT_CONTEXT_MAX_TOKENS,
        )
        await _chat_history.write_user_message(
            setup_session, conversation_id=conv.id, content=body.query
        )
        await setup_session.commit()
        conversation_id_str = str(conv.id)

    async def event_generator():
        # Emit meta event so the client knows which conversation this
        # stream belongs to (especially important when conversation_id
        # was omitted in the request and the backend created one).
        yield (
            "data: "
            + json.dumps({"type": "meta", "conversation_id": conversation_id_str})
            + "\n\n"
        )

        full_response = ""
        retrieved_sources: list[dict] = []
        tool_calls_seen: list[dict] = []
        started = time.monotonic()
        had_error: str | None = None

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
                # Buffer source rows for later persistence as the
                # assistant turn's ``sources`` JSONB column.
                retrieved_sources.append(
                    {
                        "node_id": node.id,
                        "node_labels": [node.label] if node.label else None,
                        "relevance_score": node_props.get("score"),
                        "content": node_props.get("text") or node.label,
                    }
                )
                yield f"data: {json.dumps(source_payload)}\n\n"

            async for token, prov_payload in executor.run_stream(
                message=body.query,
                session_id=None,
                history=context_history or None,
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
                    if prov_payload:
                        tool_calls_seen = list(prov_payload.tool_calls or [])
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
            had_error = str(e)
            logger.error(f"Stream chat error for graph {body.graph_id}: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        # Persist the assistant turn (or its failure) regardless of
        # how the stream ended. Uses its own transaction with the GUC
        # set so RLS lets the writes through.
        latency_ms = int((time.monotonic() - started) * 1000)
        try:
            async with async_session_maker() as write_session:
                await write_session.execute(
                    sql_text(
                        "SELECT set_config('app.current_user_id', :uid, true)"
                    ).bindparams(uid=user_id)
                )
                # If the client disconnected mid-stream, full_response
                # contains the partial content; mark cancelled.
                cancelled = (
                    had_error is None
                    and full_response != ""
                    and await request.is_disconnected()
                )
                asst_msg = await _chat_history.write_assistant_message(
                    write_session,
                    conversation_id=conv.id,
                    content=full_response,
                    latency_ms=latency_ms,
                    reasoning_mode=body.mode.value if body.mode else "enhanced",
                    retriever_used=mode_to_retriever_label(body.mode),
                    sources=retrieved_sources or None,
                    error=had_error,
                    cancelled=cancelled,
                )
                for idx, tc in enumerate(tool_calls_seen):
                    try:
                        await _chat_history.write_tool_call(
                            write_session,
                            message_id=asst_msg.id,
                            sequence_index=idx,
                            tool_name=tc.get("name", "unknown"),
                            args=None,
                            result={"node_count": tc.get("node_count")}
                            if tc.get("node_count") is not None
                            else None,
                        )
                    except Exception:
                        logger.exception(
                            "stream: failed to persist tool call %s",
                            tc.get("name"),
                        )
                await write_session.commit()
        except Exception:
            # Persistence failure on the write path must not break the
            # client connection — the response stream has already
            # completed. Log and move on.
            logger.exception(
                "stream: failed to persist assistant turn for conversation %s",
                conversation_id_str,
            )

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
    user_id: str = Depends(get_current_user_id),
) -> ChatModesResponse:
    """
    Return all available chat modes with descriptions and use cases.
    """
    # user_id is unused but its dependency chain enforces authentication.
    del user_id
    return ChatModesResponse(
        modes=get_all_modes(),
        default_mode=ChatMode.ENHANCED,
        graph_capabilities={
            "has_fulltext_indexes": True,
            "has_vector_indexes": True,
            "supports_cypher": True,
        },
    )
