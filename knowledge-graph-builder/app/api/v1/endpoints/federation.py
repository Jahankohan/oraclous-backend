"""Cross-graph federation endpoints.

POST /api/v1/graphs/federate/query                     — federated entity search
POST /api/v1/graphs/federate/vector-search             — federated vector similarity search
POST /api/v1/graphs/{graph_id}/federation/candidates   — SAME_AS candidate pairs (TASK-016)
POST /api/v1/graphs/{graph_id}/federation/resolve      — trigger async SAME_AS resolution (TASK-017)
"""

from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi.security import HTTPAuthorizationCredentials

from app.api.dependencies import security, verify_graph_access
from app.core.config import settings
from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client
from app.schemas.federation_schemas import (
    FederatedQueryRequest,
    FederatedQueryResponse,
    FederatedVectorSearchRequest,
    FederatedVectorSearchResponse,
    FederationCandidateResult,
    FederationCandidatesRequest,
    FederationResolveRequest,
    FederationResolveResponse,
    SignalScores,
)
from app.services.auth_service import auth_service
from app.services.federation_service import FederationError, FederationService

logger = get_logger(__name__)

router = APIRouter(prefix="/graphs/federate", tags=["federation"])

# Second router for graph-scoped federation endpoints (/graphs/{graph_id}/federation/...)
graph_federation_router = APIRouter(prefix="/graphs", tags=["federation"])


def _get_federation_service() -> FederationService:
    if not neo4j_client.async_driver:
        raise HTTPException(status_code=503, detail="Neo4j async driver not available")
    return FederationService(
        async_driver=neo4j_client.async_driver,
        neo4j_database=settings.NEO4J_DATABASE,
    )


@router.post(
    "/query",
    response_model=FederatedQueryResponse,
    summary="Federated cross-graph entity search",
    responses={
        400: {"description": "Invalid graph_ids or graph not federatable"},
        403: {"description": "Access denied to one or more graphs"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def federated_query(
    request: FederatedQueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    service: FederationService = Depends(_get_federation_service),
) -> FederatedQueryResponse:
    """Search for entities matching *query* across multiple knowledge graphs.

    **Fail-closed:** all `graph_ids` must be owned by the caller AND have
    `federatable=true`. A single unauthorized or non-federatable graph returns
    an error rather than partial results.

    Max 10 graphs per request. Results are labeled with `source_graph_id`.
    When `deduplicate_entities=true`, same-name+type entities across graphs
    are surfaced as `cross_graph_links` with `SAME_AS` confidence scores.
    """
    user = await auth_service.verify_token(credentials.credentials)
    user_id = str(user["id"])

    try:
        result = await service.federated_query(
            user_id=user_id,
            graph_ids=request.graph_ids,
            search_term=request.query,
            options=request.options,
            principal=user,
        )
    except FederationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None
    except Exception as exc:
        logger.error("federated_query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Federation query failed") from None

    return FederatedQueryResponse(**result)


@router.post(
    "/vector-search",
    response_model=FederatedVectorSearchResponse,
    summary="Federated cross-graph vector similarity search",
    responses={
        400: {"description": "Invalid graph_ids or graph not federatable"},
        403: {"description": "Access denied to one or more graphs"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def federated_vector_search(
    request: FederatedVectorSearchRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    service: FederationService = Depends(_get_federation_service),
) -> FederatedVectorSearchResponse:
    """Run vector similarity search across multiple knowledge graphs.

    Uses the shared chunk-embedding index with `graph_id` post-filter.
    Over-fetches by `1.5 × top_k × len(graph_ids)` to compensate for
    recall loss from the post-filter step.

    Same permission rules as `/federate/query` — fail-closed.
    """
    user = await auth_service.verify_token(credentials.credentials)
    user_id = str(user["id"])

    try:
        result = await service.federated_vector_search(
            user_id=user_id,
            graph_ids=request.graph_ids,
            query_text=request.query_text,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            principal=user,
        )
    except FederationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None
    except Exception as exc:
        logger.error("federated_vector_search failed: %s", exc)
        raise HTTPException(
            status_code=500, detail="Federation vector search failed"
        ) from None

    return FederatedVectorSearchResponse(**result)


# ── Graph-scoped federation endpoints ─────────────────────────────────────────

_CANDIDATES_THRESHOLD = 0.60


@graph_federation_router.post(
    "/{graph_id}/federation/candidates",
    response_model=list[FederationCandidateResult],
    summary="Find SAME_AS candidate pairs for a graph",
    responses={
        400: {"description": "Invalid graph_ids, graphs not federatable, or duplicate IDs"},
        403: {"description": "Access denied to graph_id or any target_graph_id"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def federation_candidates(
    request: FederationCandidatesRequest,
    graph_id: str = Path(..., description="Source graph ID to find SAME_AS candidates for"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    service: FederationService = Depends(_get_federation_service),
) -> list[FederationCandidateResult]:
    """Return SAME_AS candidate entity pairs between *graph_id* and *target_graph_ids*.

    **Auth (fail-closed):** the caller must have read access to *all* graphs in
    `[graph_id] + target_graph_ids`.  A single inaccessible or non-federatable
    graph returns 403 rather than partial results.

    **Scoring:** each pair receives a combined score based on four signals —
    name match, type match, embedding similarity, and shared relation types.
    Only pairs with `score >= 0.60` are returned; pairs below the threshold
    are discarded server-side.

    Use the returned candidates to decide whether to call
    `POST /graphs/{graph_id}/federation/resolve` (TASK-017) to persist
    high-confidence pairs as SAME_AS links.
    """
    user = await auth_service.verify_token(credentials.credentials)
    user_id = str(user["id"])

    all_graph_ids = [graph_id] + request.target_graph_ids

    try:
        # Fail-closed: validate caller can access every graph in the request
        allowed = await service._validate_and_filter(
            user_id, all_graph_ids, principal=user
        )
        allowed_ids = {row["graph_id"] for row in allowed}
        missing = set(all_graph_ids) - allowed_ids
        if missing:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied — graphs not accessible: {', '.join(sorted(missing))}",
            )

        raw = await service.find_federation_candidates(
            graph_id=graph_id,
            target_graph_ids=request.target_graph_ids,
            entity_name=request.entity_name,
        )
    except FederationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("federation_candidates failed: %s", exc)
        raise HTTPException(
            status_code=500, detail="Federation candidates query failed"
        ) from None

    # Server-side threshold filter
    results = [
        FederationCandidateResult(
            entity_a=item["entity_a"],
            entity_b=item["entity_b"],
            score=item["score"],
            signals=SignalScores(**item["signals"]),
        )
        for item in raw
        if item["score"] >= _CANDIDATES_THRESHOLD
    ]
    return results


@graph_federation_router.post(
    "/{graph_id}/federation/resolve",
    response_model=FederationResolveResponse,
    summary="Trigger async SAME_AS entity resolution between two graphs",
    responses={
        403: {"description": "Access denied — read or write permission missing"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def resolve_federation(
    request: FederationResolveRequest,
    graph_id: str = Path(..., description="Source graph ID; caller must have write access"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    service: FederationService = Depends(_get_federation_service),
) -> FederationResolveResponse:
    """Dispatch a Celery task that scores all entity pairs between *graph_id* and
    *target_graph_id*, then creates SAME_AS links for pairs above
    *confidence_threshold*.

    **Auth (two-step, fail-closed):**

    1. Read access to both *graph_id* and *target_graph_id* is verified via
       `_validate_and_filter` — the same fail-closed gate used by
       `/federate/query`.  If either graph is inaccessible or non-federatable
       the request is rejected with 403.

    2. Write access to the **source** *graph_id* is verified via
       `verify_graph_access(..., "write", ...)`.  This prevents a user with
       read-only access from triggering SAME_AS link creation — a mutation
       operation — on graphs they do not own.

    The endpoint returns immediately with a Celery `task_id`; poll
    `GET /tasks/{task_id}` for status and the final link list.

    **Task result shape (available once task is complete):**
    ```json
    {
      "created_links": [{"entity_a_id": "...", "entity_b_id": "...", "confidence": 0.91}],
      "ambiguous_count": 3,
      "rejected_count": 7
    }
    ```
    """
    from app.services.auth_service import auth_service as _auth_service
    from app.tasks.federation_tasks import resolve_same_as_task

    user = await _auth_service.verify_token(credentials.credentials)
    user_id = str(user["id"])

    # Step 1: Read access check on both graphs (fail-closed: raises FederationError on denial)
    try:
        accessible = await service._validate_and_filter(
            user_id,
            [graph_id, request.target_graph_id],
            principal=user,
        )
    except FederationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None

    if len(accessible) < 2:
        raise HTTPException(
            status_code=403,
            detail="Insufficient access to one or more graphs",
        )

    # Step 2: Write access check on source graph (raises HTTP 403 if denied)
    await verify_graph_access(graph_id, "write", user_id)

    # Step 3: Dispatch Celery task
    task = resolve_same_as_task.delay(
        graph_id,
        request.target_graph_id,
        request.confidence_threshold,
    )
    logger.info(
        "resolve_federation dispatched: graph_id=%s target=%s threshold=%.2f task_id=%s user=%s",
        graph_id,
        request.target_graph_id,
        request.confidence_threshold,
        task.id,
        user_id,
    )
    return FederationResolveResponse(task_id=task.id, status="queued")
