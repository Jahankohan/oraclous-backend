"""
Agent Memory API Endpoints

6 endpoints under /graphs/{graphId}/memories:
  POST   /graphs/{graphId}/memories
  GET    /graphs/{graphId}/memories/search
  GET    /graphs/{graphId}/memories/context
  PATCH  /graphs/{graphId}/memories/{memoryId}
  DELETE /graphs/{graphId}/memories/{memoryId}
  POST   /graphs/{graphId}/memories/consolidate
"""
from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response

from app.api.dependencies import get_current_user_id, verify_graph_access
from app.core.logging import get_logger
from app.schemas.memory import (
    ConsolidateResponse,
    MemoryContext,
    MemoryCreate,
    MemoryCreateResponse,
    MemoryRetrieverType,
    MemoryScope,
    MemorySearchResponse,
    MemoryType,
    MemoryUpdate,
    MemoryUpdateResponse,
)
from app.services.memory_service import memory_service

router = APIRouter()
logger = get_logger(__name__)


# ==================== STORE ====================

@router.post(
    "/graphs/{graph_id}/memories",
    response_model=MemoryCreateResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["memories"],
    summary="Store a memory",
)
async def store_memory(
    graph_id: str,
    body: MemoryCreate,
    user_id: str = Depends(get_current_user_id),
) -> MemoryCreateResponse:
    await verify_graph_access(graph_id=graph_id, required_level="viewer", user_id=user_id)
    try:
        return await memory_service.store_memory(graph_id=graph_id, req=body)
    except Exception as e:
        logger.error(f"store_memory failed for graph {graph_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store memory: {e}",
        )


# ==================== SEARCH ====================

@router.get(
    "/graphs/{graph_id}/memories/search",
    response_model=MemorySearchResponse,
    tags=["memories"],
    summary="Semantic search + graph traversal",
)
async def search_memories(
    graph_id: str,
    query: str = Query(..., description="Search query"),
    type: Optional[MemoryType] = Query(default=None, description="Filter by memory type"),
    scope: Optional[MemoryScope] = Query(default=None, description="Filter by scope"),
    temporal: str = Query(default="current", description="'current' | 'all'"),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    limit: int = Query(default=20, ge=1, le=100),
    include_graph_facts: bool = Query(default=False),
    user_id: str = Depends(get_current_user_id),
) -> MemorySearchResponse:
    await verify_graph_access(graph_id=graph_id, required_level="viewer", user_id=user_id)
    try:
        return await memory_service.search_memories(
            graph_id=graph_id,
            query=query,
            memory_type=type,
            scope=scope,
            temporal=temporal,
            min_confidence=min_confidence,
            limit=limit,
            include_graph_facts=include_graph_facts,
        )
    except Exception as e:
        logger.error(f"search_memories failed for graph {graph_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory search failed: {e}",
        )


# ==================== CONTEXT ====================

@router.get(
    "/graphs/{graph_id}/memories/context",
    response_model=MemoryContext,
    tags=["memories"],
    summary="Assemble agent context window",
)
async def get_memory_context(
    graph_id: str,
    query: str = Query(..., description="Current agent query or topic"),
    agent_id: Optional[str] = Query(default=None),
    session_id: Optional[str] = Query(default=None),
    scope: Optional[str] = Query(default=None, description="Comma-separated scopes"),
    max_tokens: int = Query(default=2000, ge=100, le=8000),
    include_types: Optional[str] = Query(
        default=None, description="Comma-separated memory types"
    ),
    user_id: str = Depends(get_current_user_id),
) -> MemoryContext:
    await verify_graph_access(graph_id=graph_id, required_level="viewer", user_id=user_id)

    scopes = [s.strip() for s in scope.split(",")] if scope else None
    types = [t.strip() for t in include_types.split(",")] if include_types else None

    try:
        return await memory_service.get_context(
            graph_id=graph_id,
            query=query,
            agent_id=agent_id,
            session_id=session_id,
            scopes=scopes,
            max_tokens=max_tokens,
            include_types=types,
        )
    except Exception as e:
        logger.error(f"get_memory_context failed for graph {graph_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Context assembly failed: {e}",
        )


# ==================== UPDATE ====================

@router.patch(
    "/graphs/{graph_id}/memories/{memory_id}",
    response_model=MemoryUpdateResponse,
    tags=["memories"],
    summary="Update memory (creates temporal version)",
)
async def update_memory(
    graph_id: str,
    memory_id: str,
    body: MemoryUpdate,
    user_id: str = Depends(get_current_user_id),
) -> MemoryUpdateResponse:
    await verify_graph_access(graph_id=graph_id, required_level="editor", user_id=user_id)
    try:
        return await memory_service.update_memory(
            graph_id=graph_id, memory_id=memory_id, req=body
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"update_memory failed for graph {graph_id}, memory {memory_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory update failed: {e}",
        )


# ==================== DELETE ====================

@router.delete(
    "/graphs/{graph_id}/memories/{memory_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    tags=["memories"],
    summary="Forget a memory (soft or hard delete)",
)
async def delete_memory(
    graph_id: str,
    memory_id: str,
    hard: bool = Query(default=False, description="Hard delete removes the node entirely"),
    user_id: str = Depends(get_current_user_id),
):
    await verify_graph_access(graph_id=graph_id, required_level="editor", user_id=user_id)
    try:
        await memory_service.delete_memory(
            graph_id=graph_id, memory_id=memory_id, hard=hard
        )
    except Exception as e:
        logger.error(f"delete_memory failed for graph {graph_id}, memory {memory_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory deletion failed: {e}",
        )


# ==================== CONSOLIDATE ====================

@router.post(
    "/graphs/{graph_id}/memories/consolidate",
    response_model=ConsolidateResponse,
    tags=["memories"],
    summary="Trigger async memory consolidation",
)
async def consolidate_memories(
    graph_id: str,
    user_id: str = Depends(get_current_user_id),
) -> ConsolidateResponse:
    await verify_graph_access(graph_id=graph_id, required_level="editor", user_id=user_id)
    try:
        from app.services.background_jobs import consolidate_memories_task
        task = consolidate_memories_task.delay(graph_id)
        return ConsolidateResponse(
            job_id=task.id,
            message=f"Consolidation job queued for graph {graph_id}",
        )
    except Exception as e:
        logger.error(f"consolidate_memories failed for graph {graph_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue consolidation: {e}",
        )
