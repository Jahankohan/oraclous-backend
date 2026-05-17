"""Pydantic schemas for the Organization Agent registry API (TASK-203 Part 2).

An :Agent is org-aware: it carries an ``org_id`` and is connected to its owning
:Organization by a ``HAS_AGENT`` edge. It can be granted ``CAN_ACCESS`` to one
/ several / all of the org's subgraphs at a level of ``reader|writer|admin``
(the same level vocabulary used for AgentServiceAccount grants).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ── Request models ─────────────────────────────────────────────────────────


class AgentSubgraphGrantSpec(BaseModel):
    """A request to grant an agent ``CAN_ACCESS`` on some of the org's subgraphs.

    ``level`` is one of ``reader|writer|admin``. ``graph_ids`` is either an
    explicit list of graph_id strings or the literal string ``"all"`` (every
    subgraph the organization owns).
    """

    level: str = Field(..., min_length=1, max_length=64)
    graph_ids: list[str] | Literal["all"]


# ── Response models ────────────────────────────────────────────────────────


class AgentGrantResponse(BaseModel):
    """One ``CAN_ACCESS`` grant of an agent onto a subgraph."""

    graph_id: str
    level: str
    granted_at: str | None = None


class OrgAgentResponse(BaseModel):
    """An agent in the organization's agent registry."""

    agent_id: str
    org_id: str | None
    graph_id: str
    name: str
    description: str = ""
    # True when the agent has not been soft-deleted (deactivated_at IS NULL).
    active: bool = True
