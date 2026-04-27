"""
Pydantic schemas for ReBAC permission management endpoints (ORA-48 / ORA-52).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── Role grant / revoke ────────────────────────────────────────────────────


class RoleGrantRequest(BaseModel):
    user_id: str = Field(..., description="User ID to grant the role to")
    role: str = Field(
        ...,
        description="Role name: owner | admin | editor | viewer | restricted_viewer",
    )
    email: str | None = Field(
        None, description="User email (for display; upserted on User node)"
    )
    expires_at: str | None = Field(
        None, description="ISO-8601 expiry datetime; null = permanent"
    )


class RoleRevokeRequest(BaseModel):
    role: str = Field(..., description="Role name to revoke")


class GraphMemberResponse(BaseModel):
    user_id: str
    email: str | None
    role: str
    granted_at: str | None
    expires_at: str | None


class GraphMembersResponse(BaseModel):
    graph_id: str
    members: list[GraphMemberResponse]


# ── SubGraph ───────────────────────────────────────────────────────────────


class SubGraphCreate(BaseModel):
    name: str = Field(
        ..., description="Unique name for this subgraph partition within the graph"
    )
    description: str | None = None


class SubGraphResponse(BaseModel):
    subgraph_id: str
    graph_id: str
    name: str
    description: str | None
    created_at: str | None


class SubGraphListResponse(BaseModel):
    graph_id: str
    subgraphs: list[SubGraphResponse]


# ── Access filter (used internally by retrieval layer) ─────────────────────


class UserAccessFilter(BaseModel):
    has_global_read: bool
    allowed_subgraph_ids: list[str]
