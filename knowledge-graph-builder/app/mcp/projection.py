"""The per-I/O-class REST -> MCP projection mechanism (ADR-023 D3).

`project(spec)` turns one `CapabilitySpec` into one or more MCP `Tool`s. The
mechanism is uniform *within* each I/O class — there is exactly one builder per
class, not one hand-authored tool per endpoint:

  * PLAIN      — request/response  -> one tool
  * UPLOAD     — file upload       -> one tool (base64 content in, multipart out)
  * STREAMING  — SSE               -> one tool (the stream is collected)
  * ASYNC_JOB  — enqueue + poll    -> a pair: a submit tool and a status tool

Per-tool *typed* schemas (replacing the generic `body` object with the
endpoint's Pydantic model) are TASK-230; the mechanism and the four class
patterns are TASK-229.
"""

from __future__ import annotations

import base64
import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any

from mcp.server.fastmcp.tools.base import Tool

from app.core.logging import get_logger
from app.mcp.context import get_bearer_token
from app.mcp.dispatch import dispatch_json, dispatch_multipart, dispatch_stream
from app.mcp.registry import CapabilitySpec, IOClass

logger = get_logger(__name__)

# A param spec: (name, annotation, required).
ParamSpec = tuple[str, Any, bool]
Impl = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class MCPProjectionError(RuntimeError):
    """Raised when a tool call cannot be projected — e.g. no principal bound."""


def project(spec: CapabilitySpec) -> list[Tool]:
    """Project one capability into its MCP tool(s), by I/O class."""
    builder = _BUILDERS.get(spec.io_class)
    if builder is None:  # pragma: no cover - guarded by the IOClass enum
        raise MCPProjectionError(f"no projection builder for I/O class {spec.io_class}")
    return builder(spec)


# --- shared helpers ---------------------------------------------------------


def _require_token() -> str:
    """The bearer token for the in-flight call. Fail closed if none is bound."""
    token = get_bearer_token()
    if not token:
        raise MCPProjectionError(
            "no MCP principal on the call — a bearer token is required "
            "(per-call auth wiring lands in TASK-231)"
        )
    return token


def _fill_path(path: str, args: dict[str, Any], path_params: tuple[str, ...]) -> str:
    return path.format(**{p: args[p] for p in path_params})


def _parse_body(response: Any) -> Any:
    try:
        return response.json()
    except (ValueError, json.JSONDecodeError):
        return response.text


def _result(status: int, body: Any) -> dict[str, Any]:
    """Uniform tool result. A REST error becomes a structured error object so an
    MCP client sees the same detail a REST caller would — never a silent fail."""
    if status >= 400:
        return {"error": True, "http_status": status, "detail": body}
    return body if isinstance(body, dict) else {"result": body}


def _make_tool(
    name: str,
    description: str,
    param_specs: list[ParamSpec],
    impl: Impl,
) -> Tool:
    """Build one MCP Tool with a spec-driven signature.

    FastMCP derives both the published JSON schema and argument validation from
    the handler's signature, so the projection writes a real per-spec signature
    rather than an untyped `**kwargs` handler — ADR-023 D4 (no untyped tool
    schemas). The body's *value* type is still a generic object here; TASK-230
    replaces it with the endpoint's Pydantic model.
    """

    async def handler(**kwargs: Any) -> dict[str, Any]:
        return await impl(kwargs)

    parameters = [
        inspect.Parameter(
            pname,
            inspect.Parameter.KEYWORD_ONLY,
            annotation=annotation,
            default=(inspect.Parameter.empty if required else None),
        )
        for (pname, annotation, required) in param_specs
    ]
    handler.__signature__ = inspect.Signature(parameters, return_annotation=dict)
    handler.__annotations__ = {p: a for (p, a, _) in param_specs} | {"return": dict}
    return Tool.from_function(handler, name=name, description=description)


def _io_param_specs(spec: CapabilitySpec) -> list[ParamSpec]:
    """Path params (required str), query params (optional str), body (object)."""
    specs: list[ParamSpec] = [(p, str, True) for p in spec.path_params]
    specs += [(q, str, False) for q in spec.query_params]
    if spec.has_body:
        specs.append(("body", dict, True))
    return specs


def _collect_query(spec: CapabilitySpec, args: dict[str, Any]) -> dict[str, Any] | None:
    query = {q: args[q] for q in spec.query_params if args.get(q) is not None}
    return query or None


# --- PLAIN ------------------------------------------------------------------


def _project_plain(spec: CapabilitySpec) -> list[Tool]:
    async def impl(args: dict[str, Any]) -> dict[str, Any]:
        token = _require_token()
        path = _fill_path(spec.path, args, spec.path_params)
        resp = await dispatch_json(
            spec.method,
            path,
            bearer_token=token,
            query=_collect_query(spec, args),
            json_body=args.get("body") if spec.has_body else None,
        )
        return _result(resp.status_code, _parse_body(resp))

    return [_make_tool(spec.name, spec.description, _io_param_specs(spec), impl)]


# --- UPLOAD -----------------------------------------------------------------


def _project_upload(spec: CapabilitySpec) -> list[Tool]:
    param_specs: list[ParamSpec] = [(p, str, True) for p in spec.path_params]
    param_specs += [
        ("filename", str, True),
        ("content_base64", str, True),
        ("content_type", str, False),
        ("form_fields", dict, False),
    ]

    async def impl(args: dict[str, Any]) -> dict[str, Any]:
        token = _require_token()
        path = _fill_path(spec.path, args, spec.path_params)
        try:
            content = base64.b64decode(args["content_base64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise MCPProjectionError(
                f"content_base64 is not valid base64: {exc}"
            ) from exc
        files = {
            "file": (
                args["filename"],
                content,
                args.get("content_type") or "application/octet-stream",
            )
        }
        resp = await dispatch_multipart(
            spec.method,
            path,
            bearer_token=token,
            files=files,
            data=args.get("form_fields") or None,
        )
        return _result(resp.status_code, _parse_body(resp))

    return [_make_tool(spec.name, spec.description, param_specs, impl)]


# --- STREAMING --------------------------------------------------------------


def _parse_sse(lines: list[str]) -> list[Any]:
    """Parse collected SSE lines into events. Tolerant of `data:`-framed and of
    raw newline-delimited JSON; a non-JSON line is kept as a raw record."""
    events: list[Any] = []
    for line in lines:
        payload = line
        if payload.startswith("data:"):
            payload = payload[len("data:") :].strip()
        if not payload:
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            events.append({"raw": line})
    return events


def _project_streaming(spec: CapabilitySpec) -> list[Tool]:
    async def impl(args: dict[str, Any]) -> dict[str, Any]:
        token = _require_token()
        path = _fill_path(spec.path, args, spec.path_params)
        status, lines = await dispatch_stream(
            spec.method,
            path,
            bearer_token=token,
            query=_collect_query(spec, args),
            json_body=args.get("body") if spec.has_body else None,
        )
        events = _parse_sse(lines)
        if status >= 400:
            return {"error": True, "http_status": status, "events": events}
        # Assemble: concatenate every text-bearing chunk into one answer.
        text = "".join(
            str(e.get("text", ""))
            for e in events
            if isinstance(e, dict) and e.get("type") == "answer_chunk"
        )
        final = next(
            (e for e in events if isinstance(e, dict) and e.get("type") == "done"),
            None,
        )
        return {"events": events, "text": text, "final": final}

    return [_make_tool(spec.name, spec.description, _io_param_specs(spec), impl)]


# --- ASYNC_JOB --------------------------------------------------------------


def _project_async_job(spec: CapabilitySpec) -> list[Tool]:
    async def submit_impl(args: dict[str, Any]) -> dict[str, Any]:
        token = _require_token()
        path = _fill_path(spec.path, args, spec.path_params)
        resp = await dispatch_json(
            spec.method,
            path,
            bearer_token=token,
            query=_collect_query(spec, args),
            json_body=args.get("body") if spec.has_body else None,
        )
        body = _parse_body(resp)
        if resp.status_code >= 400:
            return {"error": True, "http_status": resp.status_code, "detail": body}
        is_obj = isinstance(body, dict)
        return {
            "job_id": body.get(spec.job_id_field) if is_obj else None,
            "status": body.get("status") if is_obj else None,
            "raw": body,
        }

    tools = [
        _make_tool(spec.name, spec.description, _io_param_specs(spec), submit_impl)
    ]

    if spec.status_name and spec.status_path:
        status_path = spec.status_path
        status_path_params = spec.status_path_params

        async def status_impl(args: dict[str, Any]) -> dict[str, Any]:
            token = _require_token()
            path = status_path.format(**{p: args[p] for p in status_path_params})
            resp = await dispatch_json(spec.status_method, path, bearer_token=token)
            return _result(resp.status_code, _parse_body(resp))

        tools.append(
            _make_tool(
                spec.status_name,
                spec.status_description,
                [(p, str, True) for p in spec.status_path_params],
                status_impl,
            )
        )

    return tools


_BUILDERS: dict[IOClass, Callable[[CapabilitySpec], list[Tool]]] = {
    IOClass.PLAIN: _project_plain,
    IOClass.UPLOAD: _project_upload,
    IOClass.STREAMING: _project_streaming,
    IOClass.ASYNC_JOB: _project_async_job,
}
