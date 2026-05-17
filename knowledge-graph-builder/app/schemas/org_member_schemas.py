"""Pydantic schemas for the Organization Member API (TASK-203 Part 1).

An organization member is a :User connected to an :Organization by a
``BELONGS_TO`` edge carrying an ``org_role`` (one of ``owner|admin|member``,
ADR-021 §2). This is distinct from the 5 per-subgraph ReBAC roles
(``owner|admin|editor|viewer|restricted_viewer``) granted via ``HAS_ROLE``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ── Request models ─────────────────────────────────────────────────────────


class SubgraphGrantSpec(BaseModel):
    """A request to grant a member a ReBAC role on some of the org's subgraphs.

    ``graph_ids`` is either an explicit list of graph_id strings or the literal
    string ``"all"`` (every subgraph the organization owns).
    """

    role: str = Field(..., min_length=1, max_length=64)
    graph_ids: list[str] | Literal["all"]


class OrgMemberCreate(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255)
    org_role: str = Field(default="member", min_length=1, max_length=64)
    email: str | None = Field(default=None, max_length=320)
    subgraph_grants: SubgraphGrantSpec | None = None


# ── Response models ────────────────────────────────────────────────────────


class OrgMemberResponse(BaseModel):
    user_id: str
    email: str | None
    org_role: str
    since: str | None
    subgraph_grants: list[dict]
