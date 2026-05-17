"""Cross-subgraph LINKED_TO API (TASK-204, ADR-021 §4 Mechanism A).

Exposes the directional, intra-organization ``LINKED_TO`` primitive over
subgraphs and over entities in different subgraphs. ``LINKED_TO`` is a new
edge type, distinct from federation's ``SAME_AS``.

Routes (all under the ``/api/v1`` prefix applied by ``main.py``):

  POST   /graphs/{source_graph_id}/linked-to
                                  — create a subgraph link (admin on source)
  GET    /graphs/{graph_id}/linked-to
                                  — list outbound subgraph links (read)
  DELETE /graphs/{source_graph_id}/linked-to/{target_graph_id}
                                  — delete a subgraph link (admin on source)
  POST   /graphs/{source_graph_id}/entities/{entity_id}/linked-to
                                  — create an entity link (admin on source)
  GET    /graphs/{graph_id}/entities/{entity_id}/linked-to
                                  — list outbound entity links (read)
  DELETE /graphs/{source_graph_id}/entities/{entity_id}/linked-to
                                  — delete an entity link (admin on source)

Access control:
  - Mutating routes require ``admin`` on the source graph
    (``verify_graph_access(source_graph_id, "admin", user_id)``).
  - Listing routes require ``read`` on the graph, then additionally apply the
    ADR-021 §4 ``min_role`` visibility filter against the caller's role on the
    source subgraph (done inside the service layer).

Invariant violations raised by the service as ``ValueError`` surface as 400.
This module is purely additive — it does not modify existing routers.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from neo4j import AsyncDriver

from app.api.dependencies import get_current_user_id, verify_graph_access
from app.core.dependencies import get_neo4j_async_driver
from app.schemas.linked_to_schemas import (
    EntityLinkCreate,
    EntityLinkDelete,
    EntityLinkResponse,
    GraphLinkCreate,
    GraphLinkResponse,
)
from app.services import linked_to_service

router = APIRouter()


# ── Subgraph-level links ────────────────────────────────────────────────────


@router.post(
    "/graphs/{source_graph_id}/linked-to",
    response_model=GraphLinkResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_graph_linked_to(
    source_graph_id: str,
    body: GraphLinkCreate,
    user_id: str = Depends(get_current_user_id),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> GraphLinkResponse:
    """Create a directional LINKED_TO edge from one subgraph to another.

    Requires ``admin`` on the source graph. Both graphs must belong to the
    same organization. A cross-org link, a self-link, or an invalid
    ``min_role`` all surface as 400.
    """
    await verify_graph_access(source_graph_id, "admin", user_id)

    try:
        link = await linked_to_service.create_graph_link(
            driver,
            source_graph_id=source_graph_id,
            target_graph_id=body.target_graph_id,
            min_role=body.min_role,
            created_by=user_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    return GraphLinkResponse(**link)


@router.get(
    "/graphs/{graph_id}/linked-to",
    response_model=list[GraphLinkResponse],
)
async def list_graph_linked_to(
    graph_id: str,
    user_id: str = Depends(get_current_user_id),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> list[GraphLinkResponse]:
    """List outbound subgraph LINKED_TO edges from *graph_id*.

    Requires ``read`` on the graph. Links are filtered to those the caller may
    see — i.e. whose ``min_role`` is at or below the caller's role on the
    source subgraph (ADR-021 §4).
    """
    await verify_graph_access(graph_id, "read", user_id)

    links = await linked_to_service.list_graph_links(driver, graph_id, user_id)
    return [GraphLinkResponse(**link) for link in links]


@router.delete(
    "/graphs/{source_graph_id}/linked-to/{target_graph_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_graph_linked_to(
    source_graph_id: str,
    target_graph_id: str,
    user_id: str = Depends(get_current_user_id),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> None:
    """Delete a subgraph LINKED_TO edge. Requires ``admin`` on the source.

    204 on success; 404 if there was no such link.
    """
    await verify_graph_access(source_graph_id, "admin", user_id)

    deleted = await linked_to_service.delete_graph_link(
        driver, source_graph_id, target_graph_id
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No such LINKED_TO link between these subgraphs",
        )


# ── Entity-level links ──────────────────────────────────────────────────────


@router.post(
    "/graphs/{source_graph_id}/entities/{entity_id}/linked-to",
    response_model=EntityLinkResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_entity_linked_to(
    source_graph_id: str,
    entity_id: str,
    body: EntityLinkCreate,
    user_id: str = Depends(get_current_user_id),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> EntityLinkResponse:
    """Create a directional LINKED_TO edge between two entities.

    Requires ``admin`` on the source graph. The two entities must live in two
    different subgraphs of the same organization, and both must exist. Any
    invariant violation surfaces as 400.
    """
    await verify_graph_access(source_graph_id, "admin", user_id)

    try:
        link = await linked_to_service.create_entity_link(
            driver,
            source_graph_id=source_graph_id,
            source_entity_id=entity_id,
            target_graph_id=body.target_graph_id,
            target_entity_id=body.target_entity_id,
            min_role=body.min_role,
            created_by=user_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    return EntityLinkResponse(**link)


@router.get(
    "/graphs/{graph_id}/entities/{entity_id}/linked-to",
    response_model=list[EntityLinkResponse],
)
async def list_entity_linked_to(
    graph_id: str,
    entity_id: str,
    user_id: str = Depends(get_current_user_id),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> list[EntityLinkResponse]:
    """List outbound entity LINKED_TO edges from a source entity.

    Requires ``read`` on the graph. Links are filtered to those the caller may
    see, against the caller's role on the source subgraph (ADR-021 §4).
    """
    await verify_graph_access(graph_id, "read", user_id)

    links = await linked_to_service.list_entity_links(
        driver, graph_id, entity_id, user_id
    )
    return [EntityLinkResponse(**link) for link in links]


@router.delete(
    "/graphs/{source_graph_id}/entities/{entity_id}/linked-to",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_entity_linked_to(
    source_graph_id: str,
    entity_id: str,
    body: EntityLinkDelete,
    user_id: str = Depends(get_current_user_id),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> None:
    """Delete an entity LINKED_TO edge. Requires ``admin`` on the source.

    The target graph/entity are named in the body to disambiguate which
    outbound link to remove. 204 on success; 404 if there was no such link.
    """
    await verify_graph_access(source_graph_id, "admin", user_id)

    deleted = await linked_to_service.delete_entity_link(
        driver,
        source_graph_id=source_graph_id,
        source_entity_id=entity_id,
        target_graph_id=body.target_graph_id,
        target_entity_id=body.target_entity_id,
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No such LINKED_TO link between these entities",
        )
