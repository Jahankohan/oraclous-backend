"""In-process dispatch from an MCP tool into the REST surface.

The projection (ADR-023 D3) is mechanical because an MCP tool does not
re-implement an endpoint — it re-issues the call *through the existing FastAPI
app*, in-process, over an ASGI transport (no socket is opened). The full REST
stack — routing, the auth dependency, `verify_graph_access`, request
validation, error handling — is reused verbatim. REST and MCP therefore cannot
drift, and no endpoint logic is duplicated; this is what keeps the ADR-023
"less custom code" promise honest.
"""

from __future__ import annotations

from typing import Any

import httpx

# ASGITransport never opens a socket — the host is cosmetic, only used to build
# a well-formed URL. No request ever leaves the process.
_BASE_URL = "http://mcp-projection.internal"
_TIMEOUT = httpx.Timeout(120.0)


def _auth_headers(bearer_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {bearer_token}"}


def _transport() -> httpx.ASGITransport:
    # Imported lazily: `app.main` pulls in the whole application; importing it at
    # module load would create a cycle once the MCP server is mounted into that
    # same app (TASK-232).
    from app.main import app

    return httpx.ASGITransport(app=app)


async def dispatch_json(
    method: str,
    path: str,
    *,
    bearer_token: str,
    query: dict[str, Any] | None = None,
    json_body: Any | None = None,
) -> httpx.Response:
    """Dispatch a plain request/response (or async-job submit/status) call."""
    async with httpx.AsyncClient(
        transport=_transport(), base_url=_BASE_URL, timeout=_TIMEOUT
    ) as client:
        return await client.request(
            method,
            path,
            params=query,
            json=json_body,
            headers=_auth_headers(bearer_token),
        )


async def dispatch_multipart(
    method: str,
    path: str,
    *,
    bearer_token: str,
    files: dict[str, Any],
    data: dict[str, Any] | None = None,
    query: dict[str, Any] | None = None,
) -> httpx.Response:
    """Dispatch a file-upload (multipart/form-data) call."""
    async with httpx.AsyncClient(
        transport=_transport(), base_url=_BASE_URL, timeout=_TIMEOUT
    ) as client:
        return await client.request(
            method,
            path,
            params=query,
            files=files,
            data=data,
            headers=_auth_headers(bearer_token),
        )


async def dispatch_stream(
    method: str,
    path: str,
    *,
    bearer_token: str,
    query: dict[str, Any] | None = None,
    json_body: Any | None = None,
) -> tuple[int, list[str]]:
    """Dispatch a streaming (SSE) call and collect the whole stream.

    Returns the HTTP status and the list of non-empty body lines. An MCP tool
    call yields exactly one result, so a stream is *collected* here, not
    forwarded — the streaming projection class assembles these lines into a
    single tool result.
    """
    lines: list[str] = []
    async with httpx.AsyncClient(
        transport=_transport(), base_url=_BASE_URL, timeout=_TIMEOUT
    ) as client:
        async with client.stream(
            method,
            path,
            params=query,
            json=json_body,
            headers=_auth_headers(bearer_token),
        ) as resp:
            async for line in resp.aiter_lines():
                if line.strip():
                    lines.append(line)
            return resp.status_code, lines
