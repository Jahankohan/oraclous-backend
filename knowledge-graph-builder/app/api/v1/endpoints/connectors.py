"""Connector management endpoints (ORA-78).

All endpoints enforce graph-level ownership via verify_graph_access (ReBAC).
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user_id, get_database, verify_graph_access
from app.schemas.connector_schemas import (
    CONNECTOR_TEMPLATES,
    ConnectorListResponse,
    ConnectorResponse,
    ConnectorTemplate,
    RegisterConnectorRequest,
    SyncLogListResponse,
    UpdateConnectorRequest,
)
from app.services.connector_service import connector_service

router = APIRouter()


# ---------------------------------------------------------------------------
# Connector templates
# ---------------------------------------------------------------------------

@router.get(
    "/connector-templates",
    response_model=Dict[str, ConnectorTemplate],
    summary="List built-in connector templates",
)
async def list_connector_templates() -> Dict[str, ConnectorTemplate]:
    return CONNECTOR_TEMPLATES


@router.get(
    "/connector-templates/{connector_type}",
    response_model=ConnectorTemplate,
    summary="Get connector template schema",
)
async def get_connector_template(connector_type: str) -> ConnectorTemplate:
    template = CONNECTOR_TEMPLATES.get(connector_type)
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No template for connector_type '{connector_type}'",
        )
    return template


# ---------------------------------------------------------------------------
# Connector CRUD
# ---------------------------------------------------------------------------

@router.post(
    "/graphs/{graph_id}/connectors",
    response_model=ConnectorResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new connector for a graph",
)
async def register_connector(
    graph_id: str,
    request: RegisterConnectorRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
) -> ConnectorResponse:
    await verify_graph_access(graph_id=graph_id, required_level="write", user_id=user_id)
    return await connector_service.register_connector(db, graph_id, user_id, request)


@router.get(
    "/graphs/{graph_id}/connectors",
    response_model=ConnectorListResponse,
    summary="List connectors for a graph",
)
async def list_connectors(
    graph_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
) -> ConnectorListResponse:
    await verify_graph_access(graph_id=graph_id, required_level="read", user_id=user_id)
    connectors = await connector_service.list_connectors(db, graph_id, user_id)
    return ConnectorListResponse(connectors=connectors, total=len(connectors))


@router.get(
    "/graphs/{graph_id}/connectors/{connector_id}",
    response_model=ConnectorResponse,
    summary="Get connector details",
)
async def get_connector(
    graph_id: str,
    connector_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
) -> ConnectorResponse:
    await verify_graph_access(graph_id=graph_id, required_level="read", user_id=user_id)
    return await connector_service.get_connector(db, graph_id, user_id, connector_id)


@router.patch(
    "/graphs/{graph_id}/connectors/{connector_id}",
    response_model=ConnectorResponse,
    summary="Update connector config or schedule",
)
async def update_connector(
    graph_id: str,
    connector_id: str,
    request: UpdateConnectorRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
) -> ConnectorResponse:
    await verify_graph_access(graph_id=graph_id, required_level="write", user_id=user_id)
    return await connector_service.update_connector(db, graph_id, user_id, connector_id, request)


@router.delete(
    "/graphs/{graph_id}/connectors/{connector_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a connector",
)
async def delete_connector(
    graph_id: str,
    connector_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
) -> None:
    await verify_graph_access(graph_id=graph_id, required_level="write", user_id=user_id)
    await connector_service.delete_connector(db, graph_id, user_id, connector_id)


# ---------------------------------------------------------------------------
# Sync operations
# ---------------------------------------------------------------------------

@router.post(
    "/graphs/{graph_id}/connectors/{connector_id}/sync",
    response_model=Dict[str, Any],
    summary="Trigger manual connector sync",
)
async def trigger_sync(
    graph_id: str,
    connector_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
) -> Dict[str, Any]:
    await verify_graph_access(graph_id=graph_id, required_level="write", user_id=user_id)
    return await connector_service.trigger_sync(db, graph_id, user_id, connector_id)


@router.get(
    "/graphs/{graph_id}/connectors/{connector_id}/logs",
    response_model=SyncLogListResponse,
    summary="Get sync history for a connector",
)
async def list_sync_logs(
    graph_id: str,
    connector_id: str,
    limit: int = 20,
    offset: int = 0,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
) -> SyncLogListResponse:
    await verify_graph_access(graph_id=graph_id, required_level="read", user_id=user_id)
    result = await connector_service.list_sync_logs(db, graph_id, user_id, connector_id, limit, offset)
    return SyncLogListResponse(**result)
