"""GET /graphs/{graph_id}/entities — entity explorer endpoint (ORA-372)."""

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import get_current_user_id, verify_graph_access
from app.core.logging import get_logger
from app.schemas.entity_schemas import EntityItem, EntityListResponse
from app.services.analytics_service import GraphAnalyticsService

router = APIRouter()
logger = get_logger(__name__)


def _get_analytics_service() -> GraphAnalyticsService:
    return GraphAnalyticsService()


@router.get(
    "/graphs/{graph_id}/entities",
    response_model=EntityListResponse,
    summary="List entities in a graph",
)
async def list_entities(
    graph_id: UUID,
    q: Annotated[
        str | None, Query(description="Full-text search on name and aliases")
    ] = None,
    type_: Annotated[
        list[str] | None,
        Query(alias="type", description="Filter by entity type (multi-value)"),
    ] = None,
    community_id: Annotated[
        str | None, Query(description="Filter by community id")
    ] = None,
    sort: Annotated[
        Literal["confidence_desc", "name_asc", "degree_desc"],
        Query(description="Sort order: confidence_desc | name_asc | degree_desc"),
    ] = "confidence_desc",
    page: Annotated[int, Query(ge=1, description="Page number (1-based)")] = 1,
    page_size: Annotated[int, Query(ge=1, le=200, description="Items per page")] = 50,
    user_id: str = Depends(get_current_user_id),
    analytics: GraphAnalyticsService = Depends(_get_analytics_service),
) -> EntityListResponse:
    """Return a paginated, filterable list of entities for a knowledge graph.

    Supports full-text search on entity name and aliases, optional type and
    community filters, three sort variants, and page-based pagination.

    Auth: caller must have at least `read` access to the graph.
    """
    await verify_graph_access(str(graph_id), "read", user_id)

    try:
        result = await analytics.list_entities(
            graph_id=str(graph_id),
            q=q,
            types=type_,
            community_id=community_id,
            sort=sort,
            page=page,
            page_size=page_size,
        )
    except Exception as exc:
        logger.error(f"list_entities failed for graph {graph_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve entities",
        ) from exc

    return EntityListResponse(
        items=[EntityItem(**item) for item in result["items"]],
        total=result["total"],
        page=result["page"],
        page_size=result["page_size"],
    )
