# app/api/v1/endpoints/graphs.py - NEO4J-ONLY VERSION

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Core dependencies
from app.api.dependencies import get_current_user, get_current_user_id, get_database, verify_graph_access
from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client
from app.core.rate_limiter import limiter
from app.models.graph import (  # Keep for job tracking only
    IngestionJob,
)
from app.schemas.graph_schemas import (
    AsyncRollbackResponse,
    CommunityDetailResponse,
    CommunityDetectRequest,
    CommunityDetectResponse,
    CommunityListResponse,
    CommunityStatusResponse,
    DocumentResponse,
    GraphCreate,
    GraphInstructions,
    GraphInstructionsResponse,
    GraphLLMConfigResponse,
    GraphResponse,
    GraphUpdate,
    IngestDataRequest,
    IngestionJobResponse,
    IngestMode,
    OntologyPatchRequest,
    OntologyResponse,
    OntologySetRequest,
    OntologyValidationReport,
    RetroactiveApplyRequest,
    RetroactiveApplyResponse,
    RollbackJobResponse,
    RollbackRequest,
    RollbackResponse,
    TimelineEvent,
    TimelineResponse,
    UpdateTemporalBoundsRequest,
    VersionCreateRequest,
    VersionDiffResponse,
    VersionListResponse,
    VersionResponse,
)
from app.services.background_job_service import background_job_service

# Neo4j Services
from app.services.graph_node_service import GraphNodeService
from app.services.llm_config_service import LLMConfigService
from app.services.rollback_service import rollback_service
from app.services.snapshot_service import snapshot_service

router = APIRouter()
logger = get_logger(__name__)

# ==================== HELPER FUNCTIONS ====================


def convert_neo4j_datetime_to_python(neo4j_datetime: Any) -> datetime:
    """Convert Neo4j DateTime to Python datetime for Pydantic validation"""
    if isinstance(neo4j_datetime, str):
        # Handle ISO format strings from GraphNodeService
        return datetime.fromisoformat(neo4j_datetime.replace("Z", "+00:00"))
    elif hasattr(neo4j_datetime, "to_native"):
        # Handle Neo4j DateTime objects
        return neo4j_datetime.to_native()
    elif isinstance(neo4j_datetime, datetime):
        # Already a Python datetime
        return neo4j_datetime
    else:
        # Fallback - try to parse as string
        return datetime.fromisoformat(str(neo4j_datetime).replace("Z", "+00:00"))


# ==================== NEO4J GRAPH CRUD ENDPOINTS ====================


@router.post(
    "/graphs",
    response_model=GraphResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a knowledge graph",
    responses={
        503: {"description": "Neo4j service unavailable"},
    },
)
async def create_graph(
    graph_data: GraphCreate, user_id: str = Depends(get_current_user_id)
):
    """
    Create a new knowledge graph in Neo4j.

    Each graph is scoped to the authenticated user and acts as an isolated
    container for entities and relationships extracted from your documents.
    After creating a graph, use `POST /graphs/{id}/ingest` to populate it.
    """

    try:
        # Use GraphNodeService with Neo4j sync driver for Neo4j operations
        if not neo4j_client.sync_driver:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Neo4j connection not available",
            )

        graph_service = GraphNodeService(neo4j_client.sync_driver)

        # Generate unique graph_id
        from uuid import uuid4

        graph_id = str(uuid4())

        # Create graph in Neo4j
        graph_result = graph_service.create_graph(
            graph_id=graph_id,
            user_id=user_id,
            name=graph_data.name,
            description=graph_data.description,
        )

        logger.info(
            f"Created new Neo4j graph: {graph_result['graph_id']} for user: {user_id}"
        )

        # Register in ReBAC system so creator immediately has admin access
        from app.core.neo4j_client import neo4j_client as _neo4j_client
        from app.services.rebac_service import rebac_service

        if _neo4j_client.async_driver:
            try:
                await rebac_service.register_new_graph(
                    driver=_neo4j_client.async_driver,
                    user_id=user_id,
                    graph_id=graph_id,
                    name=graph_data.name,
                )
            except Exception as rebac_exc:
                logger.warning(
                    f"ReBAC graph registration failed (non-fatal): {rebac_exc}"
                )

        # Return response in expected format
        return GraphResponse(
            id=UUID(graph_result["graph_id"]),
            name=graph_result["name"],
            description=graph_result.get("description"),
            user_id=UUID(user_id),
            created_at=convert_neo4j_datetime_to_python(graph_result["created_at"]),
            updated_at=convert_neo4j_datetime_to_python(graph_result["updated_at"]),
            node_count=graph_result.get("node_count", 0),
            relationship_count=graph_result.get("relationship_count", 0),
            status=graph_result.get("status", "active"),
            schema_config=graph_data.schema_config or {},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create graph: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create graph: {str(e)}",
        )


@router.get(
    "/graphs",
    response_model=list[GraphResponse],
    summary="List knowledge graphs",
)
async def list_graphs(user_id: str = Depends(get_current_user_id)):
    """
    Return all knowledge graphs owned by the authenticated user.

    Results are scoped to the current user — graphs owned by other users
    are never returned.
    """

    try:
        if not neo4j_client.sync_driver:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Neo4j connection not available",
            )

        graph_service = GraphNodeService(neo4j_client.sync_driver)
        graphs = graph_service.list_user_graphs(user_id)

        # Convert to GraphResponse format
        graph_responses = []
        for graph in graphs:
            graph_responses.append(
                GraphResponse(
                    id=UUID(graph["graph_id"]),
                    name=graph["name"],
                    description=graph.get("description", ""),
                    user_id=UUID(user_id),
                    created_at=convert_neo4j_datetime_to_python(graph["created_at"]),
                    updated_at=convert_neo4j_datetime_to_python(graph["updated_at"]),
                    node_count=graph.get("node_count", 0),
                    relationship_count=graph.get("relationship_count", 0),
                    status=graph.get("status", "active"),
                    schema_config={},  # Will be populated from graph metadata later
                )
            )

        return graph_responses

    except Exception as e:
        logger.error(f"Failed to list graphs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list graphs: {str(e)}",
        )


@router.get(
    "/graphs/{graph_id}",
    response_model=GraphResponse,
    summary="Get a knowledge graph",
    responses={
        403: {"description": "Graph belongs to another user"},
        404: {"description": "Graph not found"},
    },
)
async def get_graph(graph_id: UUID, user_id: str = Depends(get_current_user_id)):
    """
    Return details for a specific knowledge graph.

    Returns `403` if the graph is inaccessible (including if it does not exist —
    never 404, to prevent graph_id enumeration).
    """
    # ReBAC check — always 403 on denial, prevents graph_id enumeration
    await verify_graph_access(str(graph_id), "read", user_id)

    try:
        if not neo4j_client.sync_driver:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Neo4j connection not available",
            )

        graph_service = GraphNodeService(neo4j_client.sync_driver)
        graph = graph_service.get_graph(str(graph_id))

        if not graph:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Graph not found"
            )

        return GraphResponse(
            id=UUID(graph["graph_id"]),
            name=graph["name"],
            description=graph.get("description", ""),
            user_id=UUID(user_id),
            created_at=convert_neo4j_datetime_to_python(graph["created_at"]),
            updated_at=convert_neo4j_datetime_to_python(graph["updated_at"]),
            node_count=graph.get("node_count", 0),
            relationship_count=graph.get("relationship_count", 0),
            status=graph.get("status", "active"),
            schema_config={},  # Will be populated from graph metadata later
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get graph {graph_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get graph: {str(e)}",
        )


@router.put(
    "/graphs/{graph_id}",
    response_model=GraphResponse,
    summary="Update a knowledge graph",
    responses={
        403: {"description": "Graph belongs to another user"},
        404: {"description": "Graph not found"},
    },
)
async def update_graph(
    graph_id: UUID,
    graph_update: GraphUpdate,
    user_id: str = Depends(get_current_user_id),
):
    """
    Update the name or description of a knowledge graph.

    All fields are optional — only fields provided in the request are updated.
    Graph content (entities and relationships in Neo4j) is not affected.
    """
    # ReBAC check — write level required for updates
    await verify_graph_access(str(graph_id), "write", user_id)

    try:
        if not neo4j_client.sync_driver:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Neo4j connection not available",
            )

        graph_service = GraphNodeService(neo4j_client.sync_driver)

        existing_graph = graph_service.get_graph(str(graph_id))
        if not existing_graph:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Graph not found"
            )

        # Update graph
        updated_graph = graph_service.update_graph(
            graph_id=str(graph_id),
            user_id=user_id,
            name=graph_update.name,
            description=graph_update.description,
            federatable=graph_update.federatable,
            federation_group=graph_update.federation_group,
        )

        if not updated_graph:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update graph",
            )

        return GraphResponse(
            id=UUID(updated_graph["graph_id"]),
            name=updated_graph["name"],
            description=updated_graph.get("description", ""),
            user_id=UUID(user_id),
            created_at=convert_neo4j_datetime_to_python(updated_graph["created_at"]),
            updated_at=convert_neo4j_datetime_to_python(updated_graph["updated_at"]),
            node_count=updated_graph.get("node_count", 0),
            relationship_count=updated_graph.get("relationship_count", 0),
            status=updated_graph.get("status", "active"),
            schema_config=graph_update.schema_config or {},
            federatable=updated_graph.get("federatable", False),
            federation_group=updated_graph.get("federation_group"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update graph {graph_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update graph: {str(e)}",
        )


# ==================== SIMPLIFIED INGESTION ENDPOINT ====================


@router.post(
    "/graphs/{graph_id}/ingest",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document into a graph",
    responses={
        403: {"description": "Graph belongs to another user"},
        404: {"description": "Graph not found"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Neo4j service unavailable"},
    },
)
@limiter.limit("10/minute")
async def ingest_data_corrected(
    request: Request,
    graph_id: UUID,
    data: IngestDataRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
):
    """
    Submit document content for entity and relationship extraction.

    Ingestion runs as a background job using the Neo4j GraphRAG pipeline.
    The response contains a job `id` — poll `GET /graphs/{id}/jobs/{job_id}`
    until `status` is `completed` or `failed`.

    **Per-job overrides** let you tune extraction for a single document without
    changing the graph's persistent instructions. Use `overrides.additional_focus`
    to steer the LLM toward specific topics, or `overrides.override_density` to
    extract more or fewer entities than the graph default.

    **Deprecated:** The top-level `instructions` string field is deprecated.
    Use `overrides.additional_focus` instead.
    """

    # ReBAC check — write level required for ingestion
    await verify_graph_access(str(graph_id), "write", user_id)

    # Verify graph exists in Neo4j (authorized users only reach this point)
    try:
        if not neo4j_client.sync_driver:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Neo4j connection not available",
            )

        graph_service = GraphNodeService(neo4j_client.sync_driver)
        neo4j_graph = graph_service.get_graph(str(graph_id))

        if not neo4j_graph:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Graph not found"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to verify graph {graph_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to verify graph: {str(e)}",
        )

    # Capture per-job overrides (including backwards-compat wrapping of deprecated field)
    effective_overrides = data.resolved_overrides()
    effective_instructions_payload: dict = {}
    if effective_overrides is not None:
        effective_instructions_payload["overrides"] = effective_overrides.model_dump(
            exclude_none=True
        )
    if data.temporal_context is not None:
        effective_instructions_payload["temporal_context"] = (
            data.temporal_context.model_dump(mode="json", exclude_none=True)
        )
    # Store ingest mode so the background worker can reconstruct it
    effective_instructions_payload["ingest_mode"] = data.mode.value
    if not effective_instructions_payload:
        effective_instructions_payload = None  # type: ignore[assignment]

    # Create ingestion job record (still in PostgreSQL for job tracking)
    job = IngestionJob(
        graph_id=graph_id,
        source_type=data.source_type or "text",
        filename=data.filename,
        source_content=data.content,
        status="pending",
        ingest_mode=data.mode.value,
        effective_instructions=effective_instructions_payload,
    )

    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Start background ingestion job using pipeline service
    try:
        job_result = await background_job_service.start_ingestion_job(str(job.id), user_id)

        if job_result["status"] == "failed":
            logger.error(f"Failed to start background job: {job_result['message']}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to start background ingestion job",
            )

        logger.info(f"Started ingestion job {job.id} for graph {graph_id}")

        return IngestionJobResponse(
            id=job.id,  # type: ignore
            graph_id=job.graph_id,  # type: ignore
            status=job.status,  # type: ignore
            progress=job.progress,  # type: ignore
            created_at=job.created_at,  # type: ignore
            source_type=job.source_type,  # type: ignore
            extracted_entities=0,
            extracted_relationships=0,
        )

    except Exception as e:
        logger.error(f"Ingestion job creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create ingestion job: {str(e)}",
        )


@router.get(
    "/graphs/{graph_id}/llm-config",
    response_model=list[GraphLLMConfigResponse],
    summary="List LLM configs visible to this graph (org-level)",
)
async def list_graph_llm_configs(
    graph_id: UUID,
    current_user: dict = Depends(get_current_user),
):
    await verify_graph_access(str(graph_id), "read", str(current_user["id"]))
    org_id = str(current_user.get("tenant_id") or current_user.get("org_id") or current_user["id"])
    if not neo4j_client.async_driver:
        return []
    svc = LLMConfigService(neo4j_client.async_driver)
    configs = await svc.list_org_configs(org_id)
    return [
        GraphLLMConfigResponse(
            config_id=c["config_id"],
            provider=c["provider"],
            model_name=c.get("model", ""),
            is_active=c.get("deactivated_at") is None,
        )
        for c in configs
    ]


def _job_to_document(job: IngestionJob) -> DocumentResponse:
    status_map = {"pending": "processing", "running": "processing", "completed": "ready", "failed": "error"}
    ext = job.filename.rsplit(".", 1)[-1].lower() if job.filename and "." in job.filename else (job.source_type or "txt")
    return DocumentResponse(
        document_id=str(job.id),
        filename=job.filename or f"document_{str(job.id)[:8]}",
        file_type=ext,
        status=status_map.get(job.status, "processing"),
        node_count=job.extracted_entities or 0,
        created_at=job.created_at,
        error_message=job.error_message,
    )


@router.get(
    "/graphs/{graph_id}/documents",
    response_model=list[DocumentResponse],
    summary="List documents ingested into a graph",
)
async def list_documents(
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
):
    await verify_graph_access(str(graph_id), "read", user_id)
    result = await db.execute(
        select(IngestionJob)
        .where(IngestionJob.graph_id == graph_id)
        .order_by(IngestionJob.created_at.desc())
    )
    jobs = result.scalars().all()
    return [_job_to_document(j) for j in jobs]


@router.post(
    "/graphs/{graph_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a file and ingest it into a graph",
)
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    graph_id: UUID,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
):
    await verify_graph_access(str(graph_id), "write", user_id)

    raw = await file.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw.decode("latin-1")

    if len(content.strip()) < 10:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="File content too short")

    ext = (file.filename or "").rsplit(".", 1)[-1].lower() if file.filename and "." in (file.filename or "") else "txt"

    job = IngestionJob(
        graph_id=graph_id,
        source_type=ext,
        filename=file.filename,
        source_content=content,
        status="pending",
        ingest_mode="incremental",
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    job_result = await background_job_service.start_ingestion_job(str(job.id), user_id)
    if job_result.get("status") == "failed":
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to start ingestion job")

    return _job_to_document(job)


@router.post(
    "/graphs/{graph_id}/ingest/incremental",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Incrementally ingest a document (entity-level delta)",
    responses={
        403: {"description": "Graph belongs to another user"},
        404: {"description": "Graph not found"},
        429: {"description": "Rate limit exceeded"},
    },
)
@limiter.limit("10/minute")
async def ingest_incremental(
    request: Request,
    graph_id: UUID,
    data: IngestDataRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
):
    """
    Convenience endpoint that forces `mode=incremental`.

    Identical to `POST /graphs/{graphId}/ingest` with `mode=incremental` —
    skips unchanged entities (fingerprint + prop_hash comparison), updates
    changed properties in place, and soft-deletes orphaned relationships.
    """
    data.mode = IngestMode.INCREMENTAL
    return await ingest_data_corrected(
        request, graph_id, data, background_tasks, user_id, db
    )


# ==================== GRAPH INSTRUCTIONS ENDPOINTS ====================


@router.put(
    "/graphs/{graph_id}/instructions",
    response_model=GraphInstructionsResponse,
    summary="Set graph extraction instructions",
    responses={
        403: {"description": "Graph belongs to another user"},
        404: {"description": "Graph not found"},
    },
)
async def set_graph_instructions(
    graph_id: UUID,
    instructions: GraphInstructions,
    user_id: str = Depends(get_current_user_id),
):
    """
    Set or replace graph-level extraction instructions.

    Instructions are persisted on the graph and applied to all future ingestion
    jobs. Each update increments the version counter. When `entity_types` is
    provided, extraction switches from free-form to ontology-guided mode —
    only the specified entity types will be extracted.

    Use `edge_property_fields` to list property names (e.g. `job_title`, `role`)
    that must be stored on relationship edges, not entity nodes.
    """
    # ReBAC check — write level required to set instructions
    await verify_graph_access(str(graph_id), "write", user_id)

    if not neo4j_client.sync_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available",
        )

    from app.services.instructions_service import instructions_service

    return await instructions_service.set_instructions(str(graph_id), instructions)


@router.get(
    "/graphs/{graph_id}/instructions",
    response_model=GraphInstructionsResponse,
    summary="Get graph extraction instructions",
    responses={
        403: {"description": "Graph belongs to another user"},
        404: {"description": "Graph not found or no instructions configured"},
    },
)
async def get_graph_instructions(
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id),
):
    """
    Retrieve the current graph-level extraction instructions.

    Returns `404` if no instructions have been set on this graph.
    Check `has_instructions` on the graph object before calling this endpoint.
    """
    # ReBAC check — read level required to view instructions
    await verify_graph_access(str(graph_id), "read", user_id)

    if not neo4j_client.sync_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available",
        )

    from app.services.instructions_service import instructions_service

    result = await instructions_service.get_instructions(str(graph_id))
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No instructions configured for this graph",
        )
    return result


@router.delete(
    "/graphs/{graph_id}/instructions",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete graph extraction instructions",
    responses={
        403: {"description": "Graph belongs to another user"},
        404: {"description": "Graph not found"},
    },
)
async def delete_graph_instructions(
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id),
):
    """
    Delete graph-level extraction instructions and revert to free-form extraction.

    After deletion, future ingestion jobs will extract all entities and
    relationships without domain-specific guidance. Previously extracted
    graph data is not affected.
    """
    # ReBAC check — admin level required to delete instructions
    await verify_graph_access(str(graph_id), "admin", user_id)

    if not neo4j_client.sync_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available",
        )

    from app.services.instructions_service import instructions_service

    await instructions_service.delete_instructions(str(graph_id))


# ==================== MIGRATION ENDPOINT ====================


@router.post(
    "/graphs/{graph_id}/migrate-properties",
    summary="Migrate node properties to relationships",
    responses={
        403: {"description": "Graph belongs to another user"},
        404: {"description": "Graph not found"},
    },
)
async def migrate_graph_properties(
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id),
):
    """
    Run a 3-phase migration to move contextual node properties onto relationship edges.

    Contextual properties (such as `job_title` and `role`) semantically belong on
    the relationship between two entities, not on the entity node itself. This
    migration finds nodes with banned properties, moves them to the appropriate
    relationships, and logs any orphans that could not be automatically migrated.

    Safe to run multiple times — phases 1 (scan) and 2 (migrate) are idempotent.
    Phase 3 (orphan logging) appends new entries without duplicating existing ones.
    """
    # ReBAC check — admin level required for property migration
    await verify_graph_access(str(graph_id), "admin", user_id)

    if not neo4j_client.sync_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available",
        )

    graph_service = GraphNodeService(neo4j_client.sync_driver)
    try:
        result = graph_service.migrate_relationship_properties(str(graph_id))
        return result
    except Exception as e:
        logger.error(f"Migration failed for graph {graph_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


# ==================== JOB MANAGEMENT ENDPOINTS ====================


@router.get(
    "/graphs/{graph_id}/jobs",
    response_model=list[IngestionJobResponse],
    summary="List ingestion jobs",
    responses={
        404: {"description": "Graph not found"},
    },
)
async def list_ingestion_jobs(
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
):
    """
    Return all ingestion jobs for a graph, ordered newest-first.

    Use job `status` and `progress` fields to track processing. When a job
    fails, `error_message` contains the reason.
    """
    # ReBAC check — read level required to list jobs
    await verify_graph_access(str(graph_id), "read", user_id)

    # Get jobs
    result = await db.execute(
        select(IngestionJob)
        .where(IngestionJob.graph_id == graph_id)
        .order_by(IngestionJob.created_at.desc())
    )
    jobs = result.scalars().all()

    return jobs


@router.get(
    "/graphs/{graph_id}/jobs/{job_id}",
    response_model=IngestionJobResponse,
    summary="Get ingestion job status",
    responses={
        404: {"description": "Graph or job not found"},
    },
)
async def get_ingestion_job(
    graph_id: UUID,
    job_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
):
    """
    Return the current status and statistics for a specific ingestion job.

    Poll this endpoint after calling `POST /graphs/{id}/ingest` until
    `status` becomes `completed` (success) or `failed` (check `error_message`).
    Typical ingestion takes 10–60 seconds depending on content length.
    """
    # ReBAC check — read level required to get job status
    await verify_graph_access(str(graph_id), "read", user_id)

    # Get the specific job
    result = await db.execute(
        select(IngestionJob).where(
            IngestionJob.id == job_id, IngestionJob.graph_id == graph_id
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Ingestion job not found"
        )

    return job


# ==================== ONTOLOGY ENDPOINTS ====================


@router.post(
    "/graphs/{graph_id}/ontology",
    response_model=OntologyResponse,
    status_code=status.HTTP_200_OK,
    summary="Set (replace) graph ontology",
    responses={
        403: {"description": "Graph belongs to another user"},
        404: {"description": "Graph not found"},
    },
)
async def set_graph_ontology(
    graph_id: UUID,
    request: OntologySetRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Replace the ontology on a graph — entity types, relationship types, and enforcement mode.

    After setting the ontology all future ingestion jobs will enforce it.
    Existing graph data is NOT modified; use the retroactive-apply endpoint for that.
    Invalidates the schema cache.
    """
    if not neo4j_client.sync_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available",
        )

    from app.services.instructions_service import instructions_service

    await _verify_graph_ownership(graph_id, user_id)
    return await instructions_service.set_ontology(str(graph_id), request)


@router.get(
    "/graphs/{graph_id}/ontology",
    response_model=OntologyResponse,
    summary="Get graph ontology",
    responses={
        403: {"description": "Graph belongs to another user"},
        404: {"description": "Graph not found or no ontology configured"},
    },
)
async def get_graph_ontology(
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id),
):
    """
    Retrieve the current ontology configuration for a graph.

    Returns `404` if no ontology has been configured. Free-form graphs have no ontology.
    """
    if not neo4j_client.sync_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available",
        )

    from app.services.instructions_service import instructions_service

    await _verify_graph_ownership(graph_id, user_id)
    result = await instructions_service.get_ontology(str(graph_id))
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No ontology configured for this graph",
        )
    return result


@router.patch(
    "/graphs/{graph_id}/ontology",
    response_model=OntologyResponse,
    summary="Patch graph ontology",
    responses={
        403: {"description": "Graph belongs to another user"},
        404: {"description": "Graph not found"},
    },
)
async def patch_graph_ontology(
    graph_id: UUID,
    patch: OntologyPatchRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Merge-update the graph ontology: add/remove individual type definitions or change the mode.

    Types are matched by name. Adding a type that already exists replaces it.
    Invalidates the schema cache.
    """
    if not neo4j_client.sync_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available",
        )

    from app.services.instructions_service import instructions_service

    await _verify_graph_ownership(graph_id, user_id)
    return await instructions_service.patch_ontology(str(graph_id), patch)


@router.delete(
    "/graphs/{graph_id}/ontology",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete graph ontology",
    responses={
        403: {"description": "Graph belongs to another user"},
        404: {"description": "Graph not found"},
    },
)
async def delete_graph_ontology(
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id),
):
    """
    Remove the ontology from a graph, reverting it to free-form extraction.

    Other graph-level instructions (domain, density, focus areas) are preserved.
    Previously extracted entities are NOT removed.
    Invalidates the schema cache.
    """
    if not neo4j_client.sync_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available",
        )

    from app.services.instructions_service import instructions_service

    await _verify_graph_ownership(graph_id, user_id)
    await instructions_service.delete_ontology(str(graph_id))


@router.post(
    "/graphs/{graph_id}/ontology/validate",
    response_model=OntologyValidationReport,
    summary="Dry-run ontology validation scan",
    responses={
        403: {"description": "Graph belongs to another user"},
        404: {"description": "Graph not found or no ontology configured"},
    },
)
async def validate_graph_ontology(
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id),
):
    """
    Scan existing entities in the graph against the current ontology — no modifications.

    Returns violation counts and a sample of offending entities. Use this to assess
    the impact before running retroactive-apply.
    """
    if not neo4j_client.sync_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available",
        )

    from app.services.instructions_service import instructions_service

    await _verify_graph_ownership(graph_id, user_id)
    ontology = await instructions_service.get_ontology(str(graph_id))
    if ontology is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No ontology configured for this graph",
        )

    allowed_types = {et.name for et in ontology.entity_types}
    scan_query = """
    MATCH (e:__Entity__ {graph_id: $graph_id})
    WHERE NOT e.label IN $allowed_types
    RETURN e.name AS name, e.label AS label, elementId(e) AS element_id
    LIMIT 100
    """
    count_query = """
    MATCH (e:__Entity__ {graph_id: $graph_id})
    RETURN count(e) AS total,
           count(CASE WHEN NOT e.label IN $allowed_types THEN 1 END) AS violations
    """

    try:
        count_records = await neo4j_client.execute_query(
            count_query,
            {"graph_id": str(graph_id), "allowed_types": list(allowed_types)},
        )
        violation_records = await neo4j_client.execute_query(
            scan_query,
            {"graph_id": str(graph_id), "allowed_types": list(allowed_types)},
        )
    except Exception as e:
        logger.error(f"Ontology validation scan failed for graph {graph_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

    total = count_records[0]["total"] if count_records else 0
    violations = count_records[0]["violations"] if count_records else 0

    import difflib

    coercion_candidates = 0
    allowed_list = list(allowed_types)
    violation_samples = []
    for rec in violation_records:
        label = rec.get("label", "")
        if label:
            best_ratio = max(
                (
                    difflib.SequenceMatcher(None, label.lower(), a.lower()).ratio()
                    for a in allowed_list
                ),
                default=0.0,
            )
            if best_ratio >= 0.7:
                coercion_candidates += 1
        violation_samples.append(
            {
                "name": rec.get("name"),
                "label": label,
                "element_id": rec.get("element_id"),
            }
        )

    return OntologyValidationReport(
        graph_id=graph_id,
        scanned_entities=total,
        violation_count=violations,
        coercion_candidates=coercion_candidates,
        violations=violation_samples,
    )


@router.post(
    "/graphs/{graph_id}/ontology/retroactive-apply",
    response_model=RetroactiveApplyResponse,
    summary="Apply current ontology to existing graph entities",
    responses={
        403: {"description": "Graph belongs to another user"},
        404: {"description": "Graph not found or no ontology configured"},
    },
)
async def retroactive_apply_ontology(
    graph_id: UUID,
    request: RetroactiveApplyRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Apply the current ontology enforcement to entities already in the graph.

    - `dry_run=true` (default): scan only, returns counts without modifying the graph.
    - `dry_run=false, ≤10k entities`: runs inline and returns results synchronously.
    - `dry_run=false, >10k entities`: dispatches a Celery background task and returns
      `celery_task_id`. Poll job status separately.
    """
    if not neo4j_client.sync_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available",
        )

    from app.services.instructions_service import instructions_service

    await _verify_graph_ownership(graph_id, user_id)
    ontology = await instructions_service.get_ontology(str(graph_id))
    if ontology is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No ontology configured for this graph",
        )

    allowed_types = {et.name for et in ontology.entity_types}
    mode = ontology.ontology_mode

    # Count entities to decide inline vs Celery
    count_records = await neo4j_client.execute_query(
        "MATCH (e:__Entity__ {graph_id: $graph_id}) RETURN count(e) AS cnt",
        {"graph_id": str(graph_id)},
    )
    entity_count = count_records[0]["cnt"] if count_records else 0

    if request.dry_run:
        # Scan only
        count_query = """
        MATCH (e:__Entity__ {graph_id: $graph_id})
        RETURN count(CASE WHEN NOT e.label IN $allowed_types THEN 1 END) AS violations
        """
        records = await neo4j_client.execute_query(
            count_query,
            {"graph_id": str(graph_id), "allowed_types": list(allowed_types)},
        )
        violation_count = records[0]["violations"] if records else 0
        return RetroactiveApplyResponse(
            graph_id=graph_id,
            dry_run=True,
            mode=mode,
            violations_found=violation_count,
            coercions_applied=0,
            deletions_applied=0,
        )

    # Live apply
    if entity_count > 10_000:
        # Dispatch as Celery task
        from app.tasks.ontology_tasks import retroactive_apply_ontology_task

        task = retroactive_apply_ontology_task.delay(
            str(graph_id), list(allowed_types), mode.value
        )
        return RetroactiveApplyResponse(
            graph_id=graph_id,
            dry_run=False,
            mode=mode,
            violations_found=0,
            coercions_applied=0,
            deletions_applied=0,
            celery_task_id=task.id,
        )

    # Inline apply (≤10k)
    import difflib

    allowed_list = list(allowed_types)

    from app.schemas.graph_schemas import OntologyValidationMode as OVM

    coercions = 0
    deletions = 0

    if mode == OVM.STRICT:
        delete_query = """
        MATCH (e:__Entity__ {graph_id: $graph_id})
        WHERE NOT e.label IN $allowed_types
        DETACH DELETE e
        RETURN count(e) AS deleted
        """
        del_records = await neo4j_client.execute_query(
            delete_query,
            {"graph_id": str(graph_id), "allowed_types": list(allowed_types)},
        )
        deletions = del_records[0]["deleted"] if del_records else 0

    elif mode == OVM.COERCE:
        violators_query = """
        MATCH (e:__Entity__ {graph_id: $graph_id})
        WHERE NOT e.label IN $allowed_types
        RETURN elementId(e) AS eid, e.label AS label
        """
        violators = await neo4j_client.execute_query(
            violators_query,
            {"graph_id": str(graph_id), "allowed_types": list(allowed_types)},
        )
        for rec in violators:
            label = rec.get("label", "")
            eid = rec.get("eid")
            if not label or not eid:
                continue
            best_match = max(
                allowed_list,
                key=lambda a: difflib.SequenceMatcher(
                    None, label.lower(), a.lower()
                ).ratio(),
            )
            ratio = difflib.SequenceMatcher(
                None, label.lower(), best_match.lower()
            ).ratio()
            if ratio >= 0.7:
                await neo4j_client.execute_query(
                    "MATCH (e:__Entity__) WHERE elementId(e) = $eid SET e.label = $new_label",
                    {"eid": eid, "new_label": best_match},
                )
                coercions += 1
            else:
                await neo4j_client.execute_query(
                    "MATCH (e:__Entity__) WHERE elementId(e) = $eid DETACH DELETE e",
                    {"eid": eid},
                )
                deletions += 1

    # For WARN mode, just count violations and return
    violations_query = """
    MATCH (e:__Entity__ {graph_id: $graph_id})
    WHERE NOT e.label IN $allowed_types
    RETURN count(e) AS cnt
    """
    v_records = await neo4j_client.execute_query(
        violations_query,
        {"graph_id": str(graph_id), "allowed_types": list(allowed_types)},
    )
    remaining_violations = v_records[0]["cnt"] if v_records else 0

    return RetroactiveApplyResponse(
        graph_id=graph_id,
        dry_run=False,
        mode=mode,
        violations_found=remaining_violations + deletions,
        coercions_applied=coercions,
        deletions_applied=deletions,
    )


# ==================== COMMUNITY DETECTION ENDPOINTS ====================

from app.services.analytics_service import GraphAnalyticsService
from app.tasks.community_tasks import COMMUNITY_DETECTION_MIN_ENTITIES


def _get_analytics_service() -> GraphAnalyticsService:
    return GraphAnalyticsService()


async def _verify_graph_ownership(
    graph_id: UUID, user_id: str, required_level: str = "read"
) -> None:
    """
    Verify user has at least required_level access to graph_id via ReBAC.
    Always raises HTTP 403 on denial — never 404 — to prevent graph_id enumeration.
    """
    await verify_graph_access(str(graph_id), required_level, user_id)


@router.post(
    "/graphs/{graph_id}/communities/detect",
    response_model=CommunityDetectResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger hierarchical community detection",
)
async def detect_communities(
    graph_id: UUID,
    request: CommunityDetectRequest,
    user_id: str = Depends(get_current_user_id),
    analytics: GraphAnalyticsService = Depends(_get_analytics_service),
):
    """
    Queue a background Leiden community detection job for the graph.

    Returns 202 with job_id when queued.
    Returns 409 when a detection job is already running.
    Returns 400 when graph has fewer than the minimum required entities.
    """
    await _verify_graph_ownership(graph_id, user_id)

    # Check current status
    current_status = await analytics.get_community_status(graph_id)
    if current_status["status"] == "rebuilding" and not request.force_rebuild:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Community detection already in progress",
        )

    # Check entity count
    count_result = await neo4j_client.execute_query(
        "MATCH (e:__Entity__ {graph_id: $graph_id}) RETURN count(e) AS cnt",
        {"graph_id": str(graph_id)},
    )
    entity_count = count_result[0]["cnt"] if count_result else 0
    min_entities = request.min_entities or COMMUNITY_DETECTION_MIN_ENTITIES
    if entity_count < min_entities:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Graph has {entity_count} entities — minimum is {min_entities} for community detection",
        )

    result = await analytics.detect_communities_async(
        graph_id=graph_id,
        levels=request.levels,
        force_rebuild=request.force_rebuild,
    )

    return CommunityDetectResponse(
        job_id=result["job_id"],
        graph_id=result["graph_id"],
        status=result["status"],
        estimated_entities=entity_count,
    )


@router.get(
    "/graphs/{graph_id}/communities/status",
    response_model=CommunityStatusResponse,
    summary="Get community detection status for a graph",
)
async def get_community_status(
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id),
    analytics: GraphAnalyticsService = Depends(_get_analytics_service),
):
    """Return current detection status, entity counts, and staleness."""
    await _verify_graph_ownership(graph_id, user_id)
    result = await analytics.get_community_status(graph_id)
    return CommunityStatusResponse(**result)


@router.get(
    "/graphs/{graph_id}/communities",
    response_model=CommunityListResponse,
    summary="List communities for a graph",
)
async def list_communities(
    graph_id: UUID,
    level: int | None = None,
    min_size: int = 2,
    limit: int = 50,
    offset: int = 0,
    include_summary: bool = True,
    user_id: str = Depends(get_current_user_id),
    analytics: GraphAnalyticsService = Depends(_get_analytics_service),
):
    """Return paginated list of communities, optionally filtered by level."""
    await _verify_graph_ownership(graph_id, user_id)
    result = await analytics.get_communities_list(
        graph_id=graph_id,
        level=level,
        min_size=min_size,
        limit=limit,
        offset=offset,
        include_summary=include_summary,
    )
    return CommunityListResponse(**result)


@router.get(
    "/graphs/{graph_id}/communities/{community_id}",
    response_model=CommunityDetailResponse,
    summary="Get a single community with its members",
)
async def get_community_detail(
    graph_id: UUID,
    community_id: str,
    user_id: str = Depends(get_current_user_id),
    analytics: GraphAnalyticsService = Depends(_get_analytics_service),
):
    """Return community detail including member entities and parent/child hierarchy."""
    await _verify_graph_ownership(graph_id, user_id)
    result = await analytics.get_community_detail(graph_id, community_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Community not found"
        )
    return CommunityDetailResponse(**result)


# ==================== TEMPORAL ENDPOINTS ====================


@router.get(
    "/graphs/{graph_id}/entities/at",
    summary="Point-in-time entity query",
    responses={
        403: {"description": "Graph belongs to another user"},
        404: {"description": "Graph not found"},
        422: {"description": "Invalid point_in_time format"},
    },
)
async def get_entities_at(
    graph_id: UUID,
    point_in_time: datetime,
    user_id: str = Depends(get_current_user_id),
):
    """
    Return all `__Entity__` nodes that were valid at the given instant.

    An entity is considered valid at `t` if:
    - `valid_from IS NULL OR valid_from <= t`
    - `valid_to IS NULL OR valid_to > t`

    Existing clients that do not pass `point_in_time` should use the standard
    graph entity endpoints instead.
    """
    await _verify_graph_ownership(graph_id, user_id)

    query = """
    MATCH (e:__Entity__ {graph_id: $graph_id})
    WHERE (e.valid_from IS NULL OR e.valid_from <= $pit)
      AND (e.valid_to IS NULL OR e.valid_to > $pit)
    RETURN elementId(e) AS element_id,
           e.name AS name,
           e.label AS label,
           e.valid_from AS valid_from,
           e.valid_to AS valid_to,
           e.transaction_time AS transaction_time
    ORDER BY e.name
    LIMIT 1000
    """
    try:
        records = await neo4j_client.execute_query(
            query, {"graph_id": str(graph_id), "pit": point_in_time.isoformat()}
        )
    except Exception as e:
        logger.error(f"Point-in-time entity query failed for graph {graph_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

    def _neo4j_dt(v) -> Any:
        if v is None:
            return None
        if hasattr(v, "to_native"):
            return v.to_native().isoformat()
        return str(v)

    return {
        "graph_id": str(graph_id),
        "point_in_time": point_in_time.isoformat(),
        "entities": [
            {
                "element_id": r["element_id"],
                "name": r["name"],
                "label": r["label"],
                "valid_from": _neo4j_dt(r.get("valid_from")),
                "valid_to": _neo4j_dt(r.get("valid_to")),
                "transaction_time": _neo4j_dt(r.get("transaction_time")),
            }
            for r in records
        ],
        "total": len(records),
    }


@router.get(
    "/graphs/{graph_id}/relationships/at",
    summary="Point-in-time relationship query",
    responses={
        403: {"description": "Graph belongs to another user"},
        404: {"description": "Graph not found"},
    },
)
async def get_relationships_at(
    graph_id: UUID,
    point_in_time: datetime,
    user_id: str = Depends(get_current_user_id),
):
    """
    Return all relationships between `__Entity__` nodes that were valid at the given instant.

    A relationship is valid at `t` if:
    - `r.valid_from IS NULL OR r.valid_from <= t`
    - `r.valid_to IS NULL OR r.valid_to > t`
    """
    await _verify_graph_ownership(graph_id, user_id)

    query = """
    MATCH (src:__Entity__ {graph_id: $graph_id})-[r]->(tgt:__Entity__ {graph_id: $graph_id})
    WHERE (r.valid_from IS NULL OR r.valid_from <= $pit)
      AND (r.valid_to IS NULL OR r.valid_to > $pit)
    RETURN src.name AS source_name,
           src.label AS source_label,
           type(r) AS relationship_type,
           tgt.name AS target_name,
           tgt.label AS target_label,
           r.valid_from AS valid_from,
           r.valid_to AS valid_to,
           r.confidence AS confidence
    ORDER BY src.name, type(r)
    LIMIT 2000
    """
    try:
        records = await neo4j_client.execute_query(
            query, {"graph_id": str(graph_id), "pit": point_in_time.isoformat()}
        )
    except Exception as e:
        logger.error(
            f"Point-in-time relationship query failed for graph {graph_id}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

    def _neo4j_dt(v) -> Any:
        if v is None:
            return None
        if hasattr(v, "to_native"):
            return v.to_native().isoformat()
        return str(v)

    return {
        "graph_id": str(graph_id),
        "point_in_time": point_in_time.isoformat(),
        "relationships": [
            {
                "source_name": r["source_name"],
                "source_label": r["source_label"],
                "relationship_type": r["relationship_type"],
                "target_name": r["target_name"],
                "target_label": r["target_label"],
                "valid_from": _neo4j_dt(r.get("valid_from")),
                "valid_to": _neo4j_dt(r.get("valid_to")),
                "confidence": r.get("confidence"),
            }
            for r in records
        ],
        "total": len(records),
    }


@router.patch(
    "/graphs/{graph_id}/entities/{entity_element_id}/temporal",
    summary="Update temporal bounds on an entity",
    responses={
        403: {"description": "Graph belongs to another user"},
        404: {"description": "Graph or entity not found"},
        422: {"description": "valid_from > valid_to"},
    },
)
async def update_entity_temporal_bounds(
    graph_id: UUID,
    entity_element_id: str,
    request: UpdateTemporalBoundsRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Update `valid_from` / `valid_to` on an existing entity node.

    - Setting `valid_to` to `null` clears the end date (entity is ongoing).
    - `valid_from > valid_to` is rejected (422).
    - `transaction_time` is NOT updated — it records when the fact was first ingested.
    """
    await _verify_graph_ownership(graph_id, user_id)

    # Build SET clause based on provided fields
    set_parts = []
    params: dict = {"graph_id": str(graph_id), "element_id": entity_element_id}

    if request.valid_from is not None:
        set_parts.append("e.valid_from = $valid_from")
        params["valid_from"] = request.valid_from.isoformat()
    if "valid_to" in request.model_fields_set:
        # Explicit null clears the field
        if request.valid_to is None:
            set_parts.append("e.valid_to = null")
        else:
            set_parts.append("e.valid_to = $valid_to")
            params["valid_to"] = request.valid_to.isoformat()

    if not set_parts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one of valid_from or valid_to must be provided.",
        )

    query = f"""
    MATCH (e:__Entity__ {{graph_id: $graph_id}})
    WHERE elementId(e) = $element_id
    SET {', '.join(set_parts)}
    RETURN elementId(e) AS element_id, e.name AS name, e.label AS label,
           e.valid_from AS valid_from, e.valid_to AS valid_to
    """
    try:
        records = await neo4j_client.execute_query(query, params)
    except Exception as e:
        logger.error(
            f"Temporal bounds update failed for entity {entity_element_id}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Entity not found in this graph",
        )

    def _neo4j_dt(v) -> Any:
        if v is None:
            return None
        if hasattr(v, "to_native"):
            return v.to_native().isoformat()
        return str(v)

    r = records[0]
    return {
        "element_id": r["element_id"],
        "name": r["name"],
        "label": r["label"],
        "valid_from": _neo4j_dt(r.get("valid_from")),
        "valid_to": _neo4j_dt(r.get("valid_to")),
    }


@router.get(
    "/graphs/{graph_id}/timeline",
    response_model=TimelineResponse,
    summary="Entity change timeline",
    responses={
        403: {"description": "Graph belongs to another user"},
        404: {"description": "Graph not found"},
    },
)
async def get_graph_timeline(
    graph_id: UUID,
    entity_id: str = None,
    user_id: str = Depends(get_current_user_id),
):
    """
    Return timeline events for a graph (or a single entity) in chronological order.

    Each event represents an entity creation or a relationship with temporal bounds.
    When `entity_id` (Neo4j elementId) is supplied, only events for that entity are returned.

    Results are ordered by `valid_from` ASC, then `transaction_time` ASC.
    """
    await _verify_graph_ownership(graph_id, user_id)

    entity_filter = "AND elementId(e) = $entity_id" if entity_id else ""

    # Entity creation events
    entity_query = f"""
    MATCH (e:__Entity__ {{graph_id: $graph_id}})
    WHERE true {entity_filter}
    RETURN 'entity_created' AS event_type,
           elementId(e) AS entity_id,
           e.name AS entity_name,
           e.label AS entity_label,
           null AS relationship_type,
           null AS related_entity_name,
           e.valid_from AS valid_from,
           e.valid_to AS valid_to,
           e.transaction_time AS transaction_time
    ORDER BY e.valid_from ASC, e.transaction_time ASC
    LIMIT 500
    """

    # Relationship events
    rel_filter = "AND elementId(src) = $entity_id" if entity_id else ""
    rel_query = f"""
    MATCH (src:__Entity__ {{graph_id: $graph_id}})-[r]->(tgt:__Entity__ {{graph_id: $graph_id}})
    WHERE (r.valid_from IS NOT NULL OR r.valid_to IS NOT NULL) {rel_filter}
    RETURN 'relationship' AS event_type,
           elementId(src) AS entity_id,
           src.name AS entity_name,
           src.label AS entity_label,
           type(r) AS relationship_type,
           tgt.name AS related_entity_name,
           r.valid_from AS valid_from,
           r.valid_to AS valid_to,
           r.transaction_time AS transaction_time
    ORDER BY r.valid_from ASC, r.transaction_time ASC
    LIMIT 500
    """

    params: dict = {"graph_id": str(graph_id)}
    if entity_id:
        params["entity_id"] = entity_id

    try:
        entity_records = await neo4j_client.execute_query(entity_query, params)
        rel_records = await neo4j_client.execute_query(rel_query, params)
    except Exception as e:
        logger.error(f"Timeline query failed for graph {graph_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

    def _neo4j_dt(v):
        if v is None:
            return None
        if hasattr(v, "to_native"):
            return v.to_native()
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                return None
        return v

    events: list = []
    for r in entity_records:
        events.append(
            TimelineEvent(
                event_type=r["event_type"],
                entity_id=r["entity_id"],
                entity_name=r.get("entity_name"),
                entity_label=r.get("entity_label"),
                relationship_type=r.get("relationship_type"),
                related_entity_name=r.get("related_entity_name"),
                valid_from=_neo4j_dt(r.get("valid_from")),
                valid_to=_neo4j_dt(r.get("valid_to")),
                transaction_time=_neo4j_dt(r.get("transaction_time")),
            )
        )
    for r in rel_records:
        events.append(
            TimelineEvent(
                event_type=r["event_type"],
                entity_id=r["entity_id"],
                entity_name=r.get("entity_name"),
                entity_label=r.get("entity_label"),
                relationship_type=r.get("relationship_type"),
                related_entity_name=r.get("related_entity_name"),
                valid_from=_neo4j_dt(r.get("valid_from")),
                valid_to=_neo4j_dt(r.get("valid_to")),
                transaction_time=_neo4j_dt(r.get("transaction_time")),
            )
        )

    # Sort all events chronologically
    events.sort(
        key=lambda e: (e.valid_from or datetime.min, e.transaction_time or datetime.min)
    )

    return TimelineResponse(
        graph_id=str(graph_id),
        entity_id=entity_id,
        events=events,
        total=len(events),
    )


# ==================== VERSIONING / SNAPSHOT ENDPOINTS ====================

_LARGE_GRAPH_ROLLBACK_THRESHOLD = (
    10_000  # nodes — above this, dispatch async Celery task
)


@router.post(
    "/graphs/{graph_id}/snapshots",
    response_model=VersionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a graph snapshot",
)
async def create_graph_snapshot(
    graph_id: UUID,
    request: VersionCreateRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Capture a named, zero-copy snapshot anchored to the current datetime.

    Returns snapshot metadata (~900 bytes written to Neo4j). No graph data is copied.
    """
    await _verify_graph_ownership(graph_id, user_id)
    try:
        version = await snapshot_service.create_snapshot(
            graph_id=str(graph_id),
            label=request.label,
            description=request.description,
            created_by=user_id,
        )
        return VersionResponse(**version)
    except Exception as exc:
        logger.error(f"Snapshot creation failed for graph {graph_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )


@router.get(
    "/graphs/{graph_id}/snapshots",
    response_model=VersionListResponse,
    summary="List all snapshots for a graph",
)
async def list_graph_snapshots(
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id),
):
    """Return all GraphVersion nodes for this graph, newest first."""
    await _verify_graph_ownership(graph_id, user_id)
    versions = await snapshot_service.list_snapshots(str(graph_id))
    return VersionListResponse(
        versions=[VersionResponse(**v) for v in versions],
        total=len(versions),
    )


@router.get(
    "/graphs/{graph_id}/snapshots/{snapshot_id}",
    response_model=VersionResponse,
    summary="Get a single snapshot",
)
async def get_graph_snapshot(
    graph_id: UUID,
    snapshot_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Return a specific snapshot by ID."""
    await _verify_graph_ownership(graph_id, user_id)
    version = await snapshot_service.get_snapshot(str(graph_id), snapshot_id)
    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Snapshot not found"
        )
    return VersionResponse(**version)


@router.delete(
    "/graphs/{graph_id}/snapshots/{snapshot_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a snapshot",
)
async def delete_graph_snapshot(
    graph_id: UUID,
    snapshot_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Permanently delete a snapshot node from Neo4j. Does not affect graph data."""
    await _verify_graph_ownership(graph_id, user_id)
    deleted = await snapshot_service.delete_snapshot(str(graph_id), snapshot_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Snapshot not found"
        )


@router.get(
    "/graphs/{graph_id}/snapshots/{snapshot_id}/diff",
    response_model=VersionDiffResponse,
    summary="Diff two graph snapshots",
)
async def diff_graph_snapshots(
    graph_id: UUID,
    snapshot_id: str,
    compare_to: str,
    offset: int = 0,
    limit: int = 100,
    user_id: str = Depends(get_current_user_id),
):
    """
    Return entities and relationships added/deleted between two snapshots.

    `compare_to` (required query param) is the ID of the second snapshot.
    The older snapshot is always treated as the 'from' side regardless of order.
    The response is paginated via `offset` and `limit` (max 500).
    """
    await _verify_graph_ownership(graph_id, user_id)
    if limit > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="limit must be ≤ 500"
        )
    try:
        diff = await snapshot_service.diff_snapshots(
            graph_id=str(graph_id),
            snapshot_id=snapshot_id,
            compare_to=compare_to,
            offset=offset,
            limit=limit,
        )
        return VersionDiffResponse(**diff)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from None
    except Exception as exc:
        logger.error(f"Diff failed for graph {graph_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )


@router.post(
    "/graphs/{graph_id}/snapshots/{snapshot_id}/rollback",
    summary="Rollback graph to a previous snapshot",
)
async def rollback_graph_snapshot(
    graph_id: UUID,
    snapshot_id: str,
    request: RollbackRequest,
    db: AsyncSession = Depends(get_database),
    user_id: str = Depends(get_current_user_id),
):
    """
    Roll back the graph to the state at the specified snapshot.

    - Graphs with ≤ 10K entities: synchronous rollback, returns `RollbackResponse`.
    - Graphs with > 10K entities: async Celery task, returns `AsyncRollbackResponse`
      with a `rollback_job_id` to poll via `GET /graphs/{graphId}/rollbacks/{rollbackId}`.

    `confirm: true` is required as a safety gate.
    `create_checkpoint: true` (default) auto-snapshots current state before rolling back.
    """
    await _verify_graph_ownership(graph_id, user_id)
    if not request.confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="confirm must be true to execute rollback",
        )

    # Count live entities to decide sync vs async path
    count_q = "MATCH (e:__Entity__ {graph_id: $graph_id}) WHERE e.invalidated_at IS NULL RETURN count(e) AS cnt"
    try:
        count_result = await neo4j_client.execute_query(
            count_q, {"graph_id": str(graph_id)}
        )
        entity_count = int((count_result or [{"cnt": 0}])[0]["cnt"])
    except Exception:
        entity_count = 0

    if entity_count > _LARGE_GRAPH_ROLLBACK_THRESHOLD:
        # Dispatch async Celery task — track via PostgreSQL
        from app.services.background_jobs import async_rollback_graph

        job_data = await rollback_service.create_rollback_job(
            db=db,
            graph_id=str(graph_id),
            version_id=snapshot_id,
            performed_by=user_id,
        )
        task = async_rollback_graph.delay(
            job_id=job_data["rollback_job_id"],
            graph_id=str(graph_id),
            version_id=snapshot_id,
            mode="full",
            performed_by=user_id,
            create_checkpoint=request.create_checkpoint,
            scope=None,
        )
        # Store Celery task ID
        import uuid as _uuid

        from sqlalchemy import update as sa_update

        from app.models.graph import GraphRollbackJob

        await db.execute(
            sa_update(GraphRollbackJob)
            .where(GraphRollbackJob.id == _uuid.UUID(job_data["rollback_job_id"]))
            .values(celery_task_id=task.id)
        )
        await db.commit()
        return AsyncRollbackResponse(
            rollback_job_id=job_data["rollback_job_id"],
            status="pending",
            message=f"Large graph ({entity_count} entities) — rollback dispatched async. Poll GET /graphs/{graph_id}/rollbacks/{job_data['rollback_job_id']}",
        )

    # Synchronous path for small/medium graphs
    try:
        result = await rollback_service.rollback(
            graph_id=str(graph_id),
            version_id=snapshot_id,
            performed_by=user_id,
            create_checkpoint=request.create_checkpoint,
        )
        return RollbackResponse(**result)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from None
    except Exception as exc:
        logger.error(
            f"Rollback failed for graph {graph_id} snapshot {snapshot_id}: {exc}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )


@router.get(
    "/graphs/{graph_id}/rollbacks/{rollback_job_id}",
    response_model=RollbackJobResponse,
    summary="Get rollback job status",
)
async def get_rollback_job_status(
    graph_id: UUID,
    rollback_job_id: str,
    db: AsyncSession = Depends(get_database),
    user_id: str = Depends(get_current_user_id),
):
    """Poll the status of an async rollback job dispatched for a large graph."""
    await _verify_graph_ownership(graph_id, user_id)
    job = await rollback_service.get_rollback_job(db, str(graph_id), rollback_job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Rollback job not found"
        )
    return RollbackJobResponse(**job)
