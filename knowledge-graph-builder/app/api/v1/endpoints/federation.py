"""Cross-graph federation endpoints.

POST /api/v1/graphs/federate/query         — federated entity search
POST /api/v1/graphs/federate/vector-search — federated vector similarity search
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.api.dependencies import security
from app.core.neo4j_client import neo4j_client
from app.core.config import settings
from app.core.logging import get_logger
from app.services.auth_service import auth_service
from app.services.federation_service import FederationError, FederationService
from app.schemas.federation_schemas import (
    FederatedQueryRequest,
    FederatedQueryResponse,
    FederatedVectorSearchRequest,
    FederatedVectorSearchResponse,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/graphs/federate", tags=["federation"])


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
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    except Exception as exc:
        logger.error("federated_query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Federation query failed")

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
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    except Exception as exc:
        logger.error("federated_vector_search failed: %s", exc)
        raise HTTPException(status_code=500, detail="Federation vector search failed")

    return FederatedVectorSearchResponse(**result)
