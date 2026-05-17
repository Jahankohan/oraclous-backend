"""Pydantic schemas for the organization member-invitation API.

An invitation lets an org owner or admin invite someone by email to join the
organization as a member. The invitee receives a link with an opaque token;
once authenticated they accept it and are registered as an org member with the
invited ``org_role`` (one of ``owner|admin|member``, ADR-021 §2).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.org_member_schemas import SubgraphGrantSpec

# ── Request models ─────────────────────────────────────────────────────────


class OrgInvitationCreate(BaseModel):
    """Create an invitation for *email* to join the org as *org_role*.

    ``subgraph_grants`` optionally pre-selects the workspaces the invitee
    lands with (a ReBAC role + a list of graph_ids or ``"all"``). It is
    meaningful for ``member`` invites — owner/admin invitees get every
    subgraph automatically, so the field is ignored for them.
    """

    email: str = Field(..., min_length=3, max_length=320)
    org_role: str = Field(default="member", min_length=1, max_length=64)
    subgraph_grants: SubgraphGrantSpec | None = None


# ── Response models ────────────────────────────────────────────────────────


class OrgInvitationResponse(BaseModel):
    """An invitation record.

    ``invite_url`` and ``email_sent`` are populated only on the create
    response — they describe the link handed to the invitee and whether the
    invitation email was successfully delivered. They are ``None`` in list
    responses.
    """

    id: str
    org_id: str
    email: str
    org_role: str
    status: str
    invited_by_user_id: str
    accepted_by_user_id: str | None = None
    created_at: str
    expires_at: str
    accepted_at: str | None = None
    subgraph_grants: dict | None = None
    invite_url: str | None = None
    email_sent: bool | None = None


class OrgInvitationAcceptResponse(BaseModel):
    """Result of accepting an invitation — the now-joined membership."""

    org_id: str
    org_role: str
    user_id: str
