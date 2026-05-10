"""
Communities listing endpoint (TASK-050).

Exposes Leiden communities computed during ingestion in the flat,
contract-stable shape the frontend's `Community` interface expects.

The richer per-community detail and detection-status routes still live in
`graphs.py` (`/graphs/{id}/communities/{community_id}`,
`/graphs/{id}/communities/status`, `/graphs/{id}/communities/detect`).
This module only owns the *list* route, which previously returned a
wrapped envelope (`CommunityListResponse`) and now returns the flat list
documented in TASK-050.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_current_user_id, verify_graph_access
from app.core.logging import get_logger
from app.schemas.community_schemas import Community
from app.services.analytics_service import GraphAnalyticsService

router = APIRouter()
logger = get_logger(__name__)


def _get_analytics_service() -> GraphAnalyticsService:
    return GraphAnalyticsService()


def _derive_label(community_id: str, summary: str | None) -> str:
    """Derive a short human-readable label for a community.

    Leiden communities don't have an explicit `label` property today, so
    we use the first sentence of the summary (truncated) and fall back to
    a synthetic `Community <short-id>` when no summary is available.
    """
    if summary:
        # Take up to the first period or 80 chars, whichever comes first
        cleaned = summary.strip()
        for terminator in (". ", "\n"):
            idx = cleaned.find(terminator)
            if idx > 0:
                cleaned = cleaned[:idx]
                break
        return cleaned[:80] if len(cleaned) > 80 else cleaned

    short = community_id.split("-")[0] if "-" in community_id else community_id[:8]
    return f"Community {short}"


@router.get(
    "/graphs/{graph_id}/communities",
    response_model=list[Community],
    summary="List Leiden communities for a graph",
    responses={
        403: {"description": "Caller lacks read access to the graph"},
    },
)
async def list_communities(
    graph_id: UUID,
    level: int | None = None,
    user_id: str = Depends(get_current_user_id),
    analytics: GraphAnalyticsService = Depends(_get_analytics_service),
) -> list[Community]:
    """
    Return Leiden communities computed during ingestion as a flat list,
    ordered by `level` then descending `size`.

    The shape is `[{community_id, level, label, size, summary?}]` to match
    the frontend's `Community` interface
    (`oraclous-visual-flow-main/src/lib/api.ts`). When no community data
    exists yet for the graph, the route returns an empty list — community
    detection is triggered separately via
    `POST /graphs/{graph_id}/communities/detect`.

    Requires `read`-level access via ReBAC.

    Optional `level` query param filters to a single hierarchy level.
    """
    # ReBAC check — read level required
    await verify_graph_access(str(graph_id), "read", user_id)

    try:
        # Pull all communities (no pagination — frontend wants the full list).
        # min_size=1 includes singleton communities; level=None returns all.
        result = await analytics.get_communities_list(
            graph_id=graph_id,
            level=level,
            min_size=1,
            limit=10_000,
            offset=0,
            include_summary=True,
        )
    except Exception as e:
        logger.error(f"Failed to list communities for graph {graph_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list communities",
        ) from None

    items = result.get("communities", []) or []
    return [
        Community(
            community_id=item["community_id"],
            level=item["level"],
            label=_derive_label(item["community_id"], item.get("summary")),
            size=item.get("entity_count", 0) or 0,
            summary=item.get("summary"),
        )
        for item in items
    ]
