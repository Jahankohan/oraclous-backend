"""Graph-Native Agent endpoints (STORY-020).

CRUD for :Agent nodes and the chat execution endpoint.
All operations are scoped to a single graph_id — no cross-graph access.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response

from app.api.dependencies import get_current_user_id, verify_graph_access
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

router = APIRouter()
logger = get_logger(__name__)


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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")


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
):
    await verify_graph_access(graph_id, "read", user_id)

    if not neo4j_client.async_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j connection not available",
        )

    try:
        executor = await AgentExecutor.from_neo4j(
            neo4j_client.async_driver, graph_id, agent_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )

    try:
        return await executor.run(body.message, body.session_id)
    except ToolNotPermittedError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
