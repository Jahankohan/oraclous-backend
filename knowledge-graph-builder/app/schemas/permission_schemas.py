"""
Pydantic schemas for ReBAC permission management endpoints (ORA-48 / ORA-52).
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── Role grant / revoke ────────────────────────────────────────────────────

class RoleGrantRequest(BaseModel):
    user_id: str = Field(..., description="User ID to grant the role to")
    role: str = Field(..., description="Role name: owner | admin | editor | viewer | restricted_viewer")
    email: Optional[str] = Field(None, description="User email (for display; upserted on User node)")
    expires_at: Optional[str] = Field(None, description="ISO-8601 expiry datetime; null = permanent")


class RoleRevokeRequest(BaseModel):
    role: str = Field(..., description="Role name to revoke")


class GraphMemberResponse(BaseModel):
    user_id: str
    email: Optional[str]
    role: str
    granted_at: Optional[str]
    expires_at: Optional[str]


class GraphMembersResponse(BaseModel):
    graph_id: str
    members: list[GraphMemberResponse]


# ── SubGraph ───────────────────────────────────────────────────────────────

class SubGraphCreate(BaseModel):
    name: str = Field(..., description="Unique name for this subgraph partition within the graph")
    description: Optional[str] = None


class SubGraphResponse(BaseModel):
    subgraph_id: str
    graph_id: str
    name: str
    description: Optional[str]
    created_at: Optional[str]


class SubGraphListResponse(BaseModel):
    graph_id: str
    subgraphs: list[SubGraphResponse]


# ── Access filter (used internally by retrieval layer) ─────────────────────

class UserAccessFilter(BaseModel):
    has_global_read: bool
    allowed_subgraph_ids: list[str]
