"""Pydantic schemas for the cross-subgraph LINKED_TO API (TASK-204).

A ``LINKED_TO`` edge is a directional, intra-organization link between two
subgraphs — or between two entities living in two different subgraphs of the
same organization. It is a brand-new edge type (ADR-021 §4, Mechanism A) and
is deliberately distinct from federation's ``SAME_AS`` (a different concern).

The link carries a ``min_role`` threshold: a principal may see and traverse
the link only if its ReBAC role on the **source** subgraph is at or above
``min_role``. ``min_role`` is one of the 5 ReBAC subgraph roles
(``owner|admin|editor|viewer|restricted_viewer``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# The 5 ReBAC subgraph roles, most→least privileged. ``min_role`` must be one.
RoleName = Literal["owner", "admin", "editor", "viewer", "restricted_viewer"]


# ── Request models ─────────────────────────────────────────────────────────


class GraphLinkCreate(BaseModel):
    """Request body to create a subgraph-level LINKED_TO edge.

    The source graph is taken from the route path; this body names the target
    graph and the minimum ReBAC role required to see/traverse the link.
    """

    target_graph_id: str = Field(..., min_length=1, max_length=255)
    min_role: RoleName


class EntityLinkCreate(BaseModel):
    """Request body to create an entity-level LINKED_TO edge.

    The source graph and source entity are taken from the route path; this
    body names the target graph, the target entity, and the ``min_role``.
    """

    target_graph_id: str = Field(..., min_length=1, max_length=255)
    target_entity_id: str = Field(..., min_length=1, max_length=255)
    min_role: RoleName


class EntityLinkDelete(BaseModel):
    """Request body to delete an entity-level LINKED_TO edge.

    The source graph and source entity are taken from the route path; this
    body names the target graph and target entity to disambiguate which
    outbound entity link to remove.
    """

    target_graph_id: str = Field(..., min_length=1, max_length=255)
    target_entity_id: str = Field(..., min_length=1, max_length=255)


# ── Response models ────────────────────────────────────────────────────────


class GraphLinkResponse(BaseModel):
    """A subgraph-level LINKED_TO edge."""

    source_graph_id: str
    target_graph_id: str
    min_role: str
    created_by: str
    created_at: str


class EntityLinkResponse(BaseModel):
    """An entity-level LINKED_TO edge."""

    source_graph_id: str
    source_entity_id: str
    target_graph_id: str
    target_entity_id: str
    min_role: str
    created_by: str
    created_at: str
