"""Per-request bearer-token extraction for the MCP ASGI surface (TASK-231).

ADR-023 D5: the MCP layer adds no parallel auth path. Every MCP tool call
dispatches a fresh in-process HTTP request through the FastAPI app, whose
`get_current_user` dependency runs `auth_service.verify_token` on the forwarded
`Authorization` header. Re-validation is therefore *structural* — a token
revoked or expired mid-session is rejected on the next tool call, because the
next call re-verifies it.

This module supplies the missing half: it sources the token from the real MCP
HTTP request. `BearerTokenASGIMiddleware` wraps the MCP server's ASGI app, reads
the inbound `Authorization` header once per request, binds it on the
`context.py` contextvar for the duration of that request, and resets it in a
`finally`. The tool handler runs within the same request scope, so the
contextvar set here is the token the projection's dispatch forwards.

Session TTL — by design there is no separate session store:

  The MCP server is built with `stateless_http=True`, so there is *no*
  server-side session that could go stale (the retired substrate's TASK-091
  "stale session" failure cannot recur). The only credential is the caller's
  JWT / service-account token, and its `exp` claim *is* the session lifetime —
  enforced per tool call by `auth_service.verify_token`, which the dispatched
  REST request invokes every time. A bounded TTL is thus inherited from the
  token itself; building a parallel session-expiry mechanism here would be a
  redundant second source of truth.

Fail-closed: when no `Authorization` header is present the contextvar is left
unbound (None). The projection's `_require_token()` then raises, refusing the
tool call. A malformed header is forwarded verbatim — `verify_token` rejects it
downstream — so an invalid credential never silently succeeds.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.mcp.context import reset_bearer_token, set_bearer_token

logger = get_logger(__name__)

_AUTH_HEADER = b"authorization"
_BEARER_PREFIX = "bearer "


def _extract_bearer(scope: dict) -> str | None:
    """Pull the bearer token out of an ASGI `http` scope's headers.

    Returns the raw token string when an `Authorization: Bearer <token>` header
    is present, else None. A header that is present but not a well-formed
    bearer credential yields None — fail closed rather than bind junk.
    """
    for name, value in scope.get("headers", []):
        if name.lower() == _AUTH_HEADER:
            try:
                raw = value.decode("latin-1").strip()
            except (UnicodeDecodeError, AttributeError):
                return None
            if raw.lower().startswith(_BEARER_PREFIX):
                token = raw[len(_BEARER_PREFIX) :].strip()
                return token or None
            return None
    return None


class BearerTokenASGIMiddleware:
    """ASGI middleware that binds the inbound bearer token per MCP request.

    Wraps the MCP server's ASGI app. For every `http` request it binds the
    `Authorization` bearer token on the `context.py` contextvar so the tool
    handler — which runs inside this request — can read it, then resets the
    contextvar in a `finally`. Non-HTTP scopes (`lifespan`, `websocket`) pass
    through untouched.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        token = _extract_bearer(scope)
        # set_bearer_token(None) is a deliberate no-op-equivalent: it binds
        # None, leaving _require_token() to fail closed for an unauthenticated
        # request. Binding explicitly (rather than skipping) guarantees no
        # stale token from a prior request on the same task leaks through.
        reset_handle = set_bearer_token(token)
        try:
            await self.app(scope, receive, send)
        finally:
            reset_bearer_token(reset_handle)
