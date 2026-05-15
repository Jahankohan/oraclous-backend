"""Graph-Native Agent endpoints (STORY-020).

CRUD for :Agent nodes and the chat execution endpoint.
All operations are scoped to a single graph_id — no cross-graph access.
"""

import time

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    get_chat_db,
    get_current_user_id,
    verify_graph_access,
)
from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client
from app.schemas.agent_schemas import (
    AgentChatResponse,
    AgentCreate,
    AgentCreateResponse,
    AgentResponse,
    AgentUpdate,
    ChatRequest,
)
from app.services.agent_executor import AgentExecutor
from app.services.agent_service import AgentService
from app.services.agent_tools import ToolNotPermittedError
from app.services.chat_history_service import ChatHistoryService
from app.tasks.chat_projection import fire_and_forget as _project_chat_message

router = APIRouter()
logger = get_logger(__name__)

_chat_history = ChatHistoryService()


def _agent_service() -> AgentService:
    if not neo4j_client.async_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j connection not available",
        )
    return AgentService(neo4j_client.async_driver)


def _to_response(d: dict) -> AgentResponse:
    return AgentResponse(**d)


# ── CRUD ──────────────────────────────────────────────────────────────────────


@router.post(
    "/graphs/{graph_id}/agents",
    response_model=AgentCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a Graph-Native Agent",
    responses={
        400: {"description": "Invalid reasoning_mode or unknown tool name"},
        403: {"description": "Access denied"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def create_agent(
    graph_id: str,
    data: AgentCreate,
    user_id: str = Depends(get_current_user_id),
    svc: AgentService = Depends(_agent_service),
):
    await verify_graph_access(graph_id, "admin", user_id)
    agent_id = await svc.create_agent(graph_id, user_id, data)
    return AgentCreateResponse(agent_id=agent_id)


@router.get(
    "/graphs/{graph_id}/agents",
    response_model=list[AgentResponse],
    summary="List agents for a graph",
    responses={403: {"description": "Access denied"}},
)
async def list_agents(
    graph_id: str,
    user_id: str = Depends(get_current_user_id),
    svc: AgentService = Depends(_agent_service),
):
    await verify_graph_access(graph_id, "read", user_id)
    agents = await svc.list_agents(graph_id)
    return [_to_response(a) for a in agents]


@router.get(
    "/graphs/{graph_id}/agents/{agent_id}",
    response_model=AgentResponse,
    summary="Get a single agent definition",
    responses={
        403: {"description": "Access denied"},
        404: {"description": "Agent not found"},
    },
)
async def get_agent(
    graph_id: str,
    agent_id: str,
    user_id: str = Depends(get_current_user_id),
    svc: AgentService = Depends(_agent_service),
):
    await verify_graph_access(graph_id, "read", user_id)
    agent = await svc.get_agent(graph_id, agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found"
        )
    return _to_response(agent)


@router.patch(
    "/graphs/{graph_id}/agents/{agent_id}",
    response_model=AgentResponse,
    summary="Update an agent definition",
    responses={
        400: {"description": "Invalid field value"},
        403: {"description": "Access denied"},
        404: {"description": "Agent not found"},
    },
)
async def update_agent(
    graph_id: str,
    agent_id: str,
    data: AgentUpdate,
    user_id: str = Depends(get_current_user_id),
    svc: AgentService = Depends(_agent_service),
):
    await verify_graph_access(graph_id, "admin", user_id)
    updated = await svc.update_agent(graph_id, agent_id, data)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found"
        )
    return _to_response(updated)


@router.delete(
    "/graphs/{graph_id}/agents/{agent_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Deactivate an agent (soft-delete)",
    responses={
        403: {"description": "Access denied"},
        404: {"description": "Agent not found"},
    },
)
async def delete_agent(
    graph_id: str,
    agent_id: str,
    user_id: str = Depends(get_current_user_id),
    svc: AgentService = Depends(_agent_service),
):
    await verify_graph_access(graph_id, "admin", user_id)
    deleted = await svc.deactivate_agent(graph_id, agent_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found"
        )


# ── Chat ──────────────────────────────────────────────────────────────────────


@router.post(
    "/graphs/{graph_id}/agents/{agent_id}/chat",
    response_model=AgentChatResponse,
    summary="Run a chat turn against a graph-native agent",
    responses={
        400: {"description": "Tool not permitted for this agent"},
        403: {"description": "Access denied"},
        404: {"description": "Agent not found or deactivated"},
        503: {"description": "Neo4j or LLM unavailable"},
    },
)
async def agent_chat(
    graph_id: str,
    agent_id: str,
    body: ChatRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_chat_db),
):
    """Run a chat turn against an agent and persist the conversation.

    STORY-031: the user turn, the assistant turn, and per-tool audit
    rows are written to Postgres before the response is returned.
    A new conversation is created if ``body.conversation_id`` is None.
    """
    await verify_graph_access(graph_id, "read", user_id)

    if not neo4j_client.async_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j connection not available",
        )

    conv = await _chat_history.get_or_create_conversation(
        db,
        user_id=user_id,
        graph_id=graph_id,
        agent_id=agent_id,
        conversation_id=body.conversation_id,
        first_message=body.message,
    )
    # Load conversation context BEFORE writing the new user message so
    # the executor sees only historical turns (STORY-031 / TASK-105).
    from app.core.config import settings as _settings

    context_history = await _chat_history.load_context(
        db,
        conversation_id=conv.id,
        max_turns=_settings.CHAT_CONTEXT_MAX_TURNS,
        max_tokens=_settings.CHAT_CONTEXT_MAX_TOKENS,
    )
    await _chat_history.write_user_message(
        db, conversation_id=conv.id, content=body.message
    )

    try:
        executor = await AgentExecutor.from_neo4j(
            neo4j_client.async_driver, graph_id, agent_id
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found"
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )

    started = time.monotonic()
    try:
        result: AgentChatResponse = await executor.run(
            body.message,
            body.session_id,
            history=context_history or None,
        )
    except ToolNotPermittedError as exc:
        latency_ms = int((time.monotonic() - started) * 1000)
        await _chat_history.write_assistant_message(
            db,
            conversation_id=conv.id,
            content="",
            latency_ms=latency_ms,
            error=str(exc),
        )
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        latency_ms = int((time.monotonic() - started) * 1000)
        await _chat_history.write_assistant_message(
            db,
            conversation_id=conv.id,
            content="",
            latency_ms=latency_ms,
            error=str(exc),
        )
        await db.commit()
        raise

    latency_ms = int((time.monotonic() - started) * 1000)
    asst_msg = await _chat_history.write_assistant_message(
        db,
        conversation_id=conv.id,
        content=result.response,
    )
    # NOTE: model/provider/tokens are not surfaced by AgentExecutor today.
    # A follow-up backend task should thread them through provenance so
    # cost accounting becomes meaningful here. For now the audit row
    # records latency only.
    asst_msg.latency_ms = latency_ms

    for idx, tc in enumerate(result.provenance.tool_calls or []):
        try:
            await _chat_history.write_tool_call(
                db,
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
                "agent_chat: failed to persist tool call %s", tc.get("name")
            )

    await db.commit()

    # Fire async Neo4j projection (TASK-106). Postgres is the source of
    # truth; this is best-effort and never blocks the response.
    _project_chat_message(str(asst_msg.id), user_id)

    # Surface the persisted conversation id to the caller.
    result.conversation_id = str(conv.id)
    return result
