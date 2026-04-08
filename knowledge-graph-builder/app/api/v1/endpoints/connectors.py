"""Connector management endpoints (ORA-78, ORA-77).

All endpoints enforce graph-level ownership via verify_graph_access (ReBAC).

Routes:
  /connector-templates             — built-in connector templates
  /graphs/{id}/connectors/database — database connector CRUD (ORA-77, Neo4j-backed)
  /graphs/{id}/connectors          — API/webhook connector CRUD (ORA-78, PostgreSQL-backed)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user_id, get_database, verify_graph_access
from app.schemas.connector_schemas import (
    CONNECTOR_TEMPLATES,
    ConnectorListResponse,
    ConnectorResponse,
    ConnectorTemplate,
    DbConnectorDetailResponse,
    DbConnectorListResponse,
    DbConnectorResponse,
    RegisterConnectorRequest,
    RegisterDbConnectorRequest,
    SyncLogListResponse,
    TriggerDbSyncRequest,
    TriggerDbSyncResponse,
    UpdateConnectorRequest,
)
from app.services.connector_service import connector_service
from app.services.database_connector_service import (
    DatabaseConnectorType,
    DbSyncMode,
    database_connector_service,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Connector templates
# ---------------------------------------------------------------------------


@router.get(
    "/connector-templates",
    response_model=dict[str, ConnectorTemplate],
    summary="List built-in connector templates",
)
async def list_connector_templates() -> dict[str, ConnectorTemplate]:
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


# ===========================================================================
# Database connector endpoints (ORA-77) — Neo4j-backed
# NOTE: /connectors/database routes MUST appear before /connectors/{connector_id}
#       to avoid FastAPI treating "database" as a connector_id path parameter.
# ===========================================================================


@router.post(
    "/graphs/{graph_id}/connectors/database",
    response_model=DbConnectorResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a database connector (PostgreSQL / MySQL / MongoDB)",
    tags=["database-connectors"],
)
async def register_db_connector(
    graph_id: str,
    request: RegisterDbConnectorRequest,
    user_id: str = Depends(get_current_user_id),
) -> DbConnectorResponse:
    await verify_graph_access(graph_id=graph_id, required_level="write", user_id=user_id)
    data = await database_connector_service.register(
        graph_id=graph_id,
        user_id=user_id,
        display_name=request.display_name,
        connector_type=DatabaseConnectorType(request.connector_type),
        host=request.host,
        port=request.port,
        database=request.database,
        sync_mode=DbSyncMode(request.sync_mode),
        schema_filter=request.schema_filter,
        table_filter=request.table_filter,
        sample_row_limit=request.sample_row_limit,
    )
    return DbConnectorResponse(**data)


@router.get(
    "/graphs/{graph_id}/connectors/database",
    response_model=DbConnectorListResponse,
    summary="List database connectors for a graph",
    tags=["database-connectors"],
)
async def list_db_connectors(
    graph_id: str,
    user_id: str = Depends(get_current_user_id),
) -> DbConnectorListResponse:
    await verify_graph_access(graph_id=graph_id, required_level="read", user_id=user_id)
    connectors = await database_connector_service.list_connectors(graph_id)
    return DbConnectorListResponse(
        connectors=[DbConnectorResponse(**c) for c in connectors],
        total=len(connectors),
    )


@router.get(
    "/graphs/{graph_id}/connectors/database/{connector_id}",
    response_model=DbConnectorDetailResponse,
    summary="Get database connector detail (with last 5 sync errors)",
    tags=["database-connectors"],
)
async def get_db_connector(
    graph_id: str,
    connector_id: str,
    user_id: str = Depends(get_current_user_id),
) -> DbConnectorDetailResponse:
    await verify_graph_access(graph_id=graph_id, required_level="read", user_id=user_id)
    data = await database_connector_service.get_connector(graph_id, connector_id)
    return DbConnectorDetailResponse(**data)


@router.delete(
    "/graphs/{graph_id}/connectors/database/{connector_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Soft-delete a database connector",
    tags=["database-connectors"],
)
async def delete_db_connector(
    graph_id: str,
    connector_id: str,
    user_id: str = Depends(get_current_user_id),
) -> None:
    await verify_graph_access(graph_id=graph_id, required_level="write", user_id=user_id)
    await database_connector_service.delete_connector(graph_id, connector_id)


@router.post(
    "/graphs/{graph_id}/connectors/database/{connector_id}/sync",
    response_model=TriggerDbSyncResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger a database connector sync (enqueues Celery task)",
    tags=["database-connectors"],
)
async def trigger_db_sync(
    graph_id: str,
    connector_id: str,
    request: TriggerDbSyncRequest = TriggerDbSyncRequest(),
    user_id: str = Depends(get_current_user_id),
) -> TriggerDbSyncResponse:
    from datetime import datetime, timezone

    await verify_graph_access(graph_id=graph_id, required_level="write", user_id=user_id)

    # Confirm connector exists and belongs to this graph before queuing
    connector = await database_connector_service.get_connector(graph_id, connector_id)
    sync_mode = request.sync_mode or connector["sync_mode"]

    if not request.dry_run:
        from app.services.background_jobs import sync_database_connector
        task = sync_database_connector.delay(
            graph_id=graph_id,
            connector_id=connector_id,
            user_id=user_id,
            sync_mode_override=sync_mode,
            table_filter_override=request.table_filter,
        )
        job_id = task.id
    else:
        job_id = "dry-run"

    return TriggerDbSyncResponse(
        job_id=job_id,
        connector_id=connector_id,
        sync_mode=sync_mode,
        status="queued" if not request.dry_run else "dry_run",
        queued_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# API/webhook connector CRUD (ORA-78) — PostgreSQL-backed
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
    await verify_graph_access(
        graph_id=graph_id, required_level="write", user_id=user_id
    )
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
    await verify_graph_access(
        graph_id=graph_id, required_level="write", user_id=user_id
    )
    return await connector_service.update_connector(
        db, graph_id, user_id, connector_id, request
    )


@router.delete(
    "/graphs/{graph_id}/connectors/{connector_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    summary="Delete a connector",
)
async def delete_connector(
    graph_id: str,
    connector_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
):
    await verify_graph_access(
        graph_id=graph_id, required_level="write", user_id=user_id
    )
    await connector_service.delete_connector(db, graph_id, user_id, connector_id)


# ---------------------------------------------------------------------------
# Sync operations
# ---------------------------------------------------------------------------


@router.post(
    "/graphs/{graph_id}/connectors/{connector_id}/sync",
    response_model=dict[str, Any],
    summary="Trigger manual connector sync",
)
async def trigger_sync(
    graph_id: str,
    connector_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
) -> dict[str, Any]:
    await verify_graph_access(
        graph_id=graph_id, required_level="write", user_id=user_id
    )
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
    result = await connector_service.list_sync_logs(
        db, graph_id, user_id, connector_id, limit, offset
    )
    return SyncLogListResponse(**result)
