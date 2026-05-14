"""
Communities listing + kind-discovery endpoints (TASK-050, STORY-4a).

Exposes communities computed during ingestion in the flat, contract-stable
shape the frontend's `Community` interface expects, and a discovery
endpoint that publishes the kind registry so the UI can stay metadata-driven.

The richer per-community detail and detection-status routes still live in
`graphs.py` (`/graphs/{id}/communities/{community_id}`,
`/graphs/{id}/communities/status`, `/graphs/{id}/communities/detect`); they
also accept `?kind=` to address any registered kind.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import get_current_user_id, verify_graph_access
from app.core.logging import get_logger
from app.schemas.community_kinds import (
    UnknownCommunityKindError,
    all_kinds,
    get_kind,
)
from app.schemas.community_schemas import Community, CommunityKindInfo
from app.services.analytics_service import GraphAnalyticsService

router = APIRouter()
logger = get_logger(__name__)


def _get_analytics_service() -> GraphAnalyticsService:
    return GraphAnalyticsService()


def _derive_label(community_id: str, summary: str | None) -> str:
    """Derive a short human-readable label for a community.

    Communities don't have an explicit `label` property today, so we use
    the first sentence of the summary (truncated) and fall back to a
    synthetic `Community <short-id>` when no summary is available.
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
    "/communities/kinds",
    response_model=list[CommunityKindInfo],
    summary="List registered community kinds",
)
async def list_community_kinds(
    _user_id: str = Depends(get_current_user_id),
) -> list[CommunityKindInfo]:
    """Return every community kind the platform knows how to handle.

    This is the source of truth for the frontend's kind picker. Each entry
    declares the underlying Neo4j label, member label, hierarchical-ness,
    and whether detection is supported (vs. read-only).

    Auth required so unauthenticated callers can't enumerate platform
    internals, but no graph-level ReBAC: the registry is the same for all
    graphs.
    """
    return [
        CommunityKindInfo(
            kind=spec.kind,
            display_name=spec.display_name,
            community_label=spec.community_label,
            member_label=spec.member_label,
            hierarchical=spec.hierarchical,
            detection_supported=spec.detector_task_name is not None,
        )
        for spec in all_kinds()
    ]


@router.get(
    "/graphs/{graph_id}/communities",
    response_model=list[Community],
    summary="List communities for a graph",
    responses={
        400: {"description": "Unknown community kind"},
        403: {"description": "Caller lacks read access to the graph"},
    },
)
async def list_communities(
    graph_id: UUID,
    level: int | None = None,
    kind: str = Query(
        default="entity",
        description=(
            "Which community kind to list. Must be one of the kinds returned "
            "by GET /communities/kinds. Defaults to ``entity`` to preserve "
            "behavior for clients that pre-date the registry."
        ),
    ),
    user_id: str = Depends(get_current_user_id),
    analytics: GraphAnalyticsService = Depends(_get_analytics_service),
) -> list[Community]:
    """
    Return communities computed for the graph as a flat list, ordered by
    `level` then descending `size` (hierarchical kinds) or by descending
    `size` alone (flat kinds).

    The shape is `[{community_id, kind, level, label, size, member_label,
    summary?}]`. When no community data exists yet for the kind, the route
    returns an empty list — entity-level detection is triggered separately
    via `POST /graphs/{graph_id}/communities/detect`; flat kinds (e.g.,
    chunk-Louvain) are read-only today.

    Requires `read`-level access via ReBAC.
    """
    try:
        spec = get_kind(kind)
    except UnknownCommunityKindError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

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
            kind=kind,
        )
    except UnknownCommunityKindError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
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
            kind=item.get("kind", kind),
            # Flat kinds have no hierarchy — surface level as 0 so the
            # response stays type-stable for clients with int-only level.
            level=item["level"] if item.get("level") is not None else 0,
            label=_derive_label(item["community_id"], item.get("summary")),
            size=item.get("entity_count", 0) or 0,
            member_label=spec.member_label,
            summary=item.get("summary"),
            summary_keywords=item.get("summary_keywords"),
            summary_excerpt=item.get("summary_excerpt"),
        )
        for item in items
    ]
