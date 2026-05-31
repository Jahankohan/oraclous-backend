"""MCP request context — the principal seam.

A thin contextvar holding the bearer token for the in-flight MCP tool call.
TASK-229 provides the seam; TASK-231 wires it to per-call principal
re-validation and a bounded session TTL (ADR-023 D5).

The projection's dispatch reads the token here and forwards it as the
`Authorization` header into the REST surface — the MCP layer adds no parallel
auth path (ADR-023 D5): the same JWT / service-account key the REST API already
verifies is the only credential.
"""

from __future__ import annotations

import contextvars

_bearer_token: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mcp_bearer_token", default=None
)


def set_bearer_token(token: str | None) -> contextvars.Token:
    """Bind the bearer token for the current MCP tool call. Returns a reset token."""
    return _bearer_token.set(token)


def reset_bearer_token(token: contextvars.Token) -> None:
    """Restore the bearer-token contextvar to its prior state."""
    _bearer_token.reset(token)


def get_bearer_token() -> str | None:
    """The bearer token for the in-flight MCP tool call, or None if unbound."""
    return _bearer_token.get()
