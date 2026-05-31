"""Typed-schema tests for the MCP projection — TASK-230 / STORY-035.

TASK-230 gives every projected MCP tool a fully typed input schema derived
from the REST endpoint's Pydantic models (ADR-023 D4 — no untyped `body: dict`,
a documented retirement failure). These tests assert that property *without a
live backend*: schema shape is checked statically, and argument validation is
exercised through `mcp.call_tool` (FastMCP rejects a malformed call before any
dispatch happens).

The end-to-end dispatch behaviour of the four I/O classes is covered by
`test_mcp_projection.py`; this module only adds the typed-schema guarantees.

Run inside the kg-builder container:

    docker compose exec -T knowledge-graph-builder \\
        python -m pytest tests/integration/test_mcp_typed_schemas.py -q
"""

from __future__ import annotations

import pytest

from app.mcp import build_mcp_server
from app.mcp.context import set_bearer_token
from app.mcp.registry import REGISTRY

# Building the server needs no app lifespan or auth patch — projection is a
# pure transform of the registry. (Dispatch would need them; validation does
# not, because FastMCP rejects a malformed call before the handler runs.)


@pytest.fixture(scope="module")
def mcp_server():
    return build_mcp_server()


def _resolve_ref(schema: dict, ref: str) -> dict:
    """Resolve a local `#/$defs/Name` JSON-schema reference within `schema`."""
    assert ref.startswith("#/$defs/"), f"unexpected ref form: {ref}"
    name = ref.split("/")[-1]
    defs = schema.get("$defs", {})
    assert name in defs, f"ref {ref} not found in $defs ({sorted(defs)})"
    return defs[name]


@pytest.mark.asyncio
async def test_every_body_property_is_a_typed_object(mcp_server):
    """For EVERY projected tool that has a `body` input, that property must
    resolve to a typed object — a `$ref` into `$defs` whose entry carries
    `properties` — never a bare `{"type": "object"}` with no fields.

    This is the hard ADR-023 D4 requirement: no untyped `body: dict` may remain.
    """
    tools = await mcp_server.list_tools()
    assert tools, "no tools projected"

    checked_bodies = 0
    for tool in tools:
        schema = tool.inputSchema
        props = schema.get("properties", {})
        if "body" not in props:
            continue
        checked_bodies += 1
        body = props["body"]

        # The body must be a $ref — not an inline untyped object.
        ref = body.get("$ref")
        assert ref, (
            f"{tool.name}: `body` is not a typed model reference "
            f"(got {body!r}) — untyped body is forbidden (ADR-023 D4)"
        )

        target = _resolve_ref(schema, ref)
        assert target.get("type") == "object", (
            f"{tool.name}: body model {ref} is not an object"
        )
        target_props = target.get("properties")
        assert target_props, (
            f"{tool.name}: body model {ref} has no `properties` — it is a bare "
            f"untyped object, which ADR-023 D4 forbids"
        )

    # The curated registry has body-carrying specs; the test must have run.
    assert checked_bodies > 0, "no tool with a `body` input was checked"


@pytest.mark.asyncio
async def test_no_tool_publishes_an_untyped_body(mcp_server):
    """Stronger restatement: scanning the whole surface, not one tool publishes
    a `body` of bare `{"type": "object"}` (the retired-substrate failure)."""
    tools = await mcp_server.list_tools()
    offenders = []
    for tool in tools:
        props = tool.inputSchema.get("properties", {})
        body = props.get("body")
        if body is None:
            continue
        if "$ref" not in body:
            # An inline schema is only acceptable if it is itself typed; a bare
            # object (`{"type": "object"}` with no properties) is the failure.
            if body.get("type") == "object" and not body.get("properties"):
                offenders.append(tool.name)
    assert not offenders, f"tools with an untyped body: {offenders}"


def test_every_body_carrying_spec_declares_a_model():
    """Registry-level guard: any spec with `has_body` names a `body_model`, so
    the projection can never silently fall back to an untyped `dict`."""
    missing = [s.name for s in REGISTRY if s.has_body and s.body_model is None]
    assert not missing, f"specs with a body but no body_model: {missing}"


@pytest.mark.asyncio
async def test_valid_typed_call_is_accepted(mcp_server):
    """A call whose `body` satisfies the typed schema passes FastMCP's argument
    validation and reaches the handler. `graph.create` is dispatched without a
    bound principal, so it fails *closed at the projection*, not at validation —
    proving the typed argument was accepted and the call advanced past schema
    checking."""
    set_bearer_token(None)
    with pytest.raises(Exception) as exc_info:
        await mcp_server.call_tool(
            "graph.create",
            {"body": {"name": "typed-schema-test", "description": "ok"}},
        )
    # Past validation: the failure is the missing-principal projection error,
    # not a schema/validation error.
    msg = str(exc_info.value).lower()
    assert "principal" in msg, (
        f"a valid typed call should fail only on the missing principal, got: {msg}"
    )


@pytest.mark.asyncio
async def test_malformed_call_is_rejected_before_dispatch(mcp_server):
    """A call missing a required typed field is rejected by FastMCP's argument
    validation — before the handler (and any dispatch) runs. `graph.create`'s
    body model requires `name`; omitting it must fail as a validation error,
    not as the missing-principal projection error."""
    set_bearer_token(None)
    with pytest.raises(Exception) as exc_info:
        await mcp_server.call_tool(
            "graph.create",
            {"body": {"description": "no name supplied"}},
        )
    msg = str(exc_info.value).lower()
    # Validation fires first: the error names the missing field, and crucially
    # is NOT the projection's missing-principal error (which would mean the
    # malformed body slipped through to the handler).
    assert "principal" not in msg, (
        f"malformed body reached the handler — validation did not fire: {msg}"
    )
    assert "name" in msg or "validation" in msg or "field required" in msg, (
        f"expected a validation error naming the missing field, got: {msg}"
    )


@pytest.mark.asyncio
async def test_wrong_type_call_is_rejected_before_dispatch(mcp_server):
    """A call whose typed field has the wrong type is rejected by validation
    before dispatch. `ingest.text`'s `IngestDataRequest` body is an object;
    passing a string for `body` must fail validation, not reach the handler."""
    set_bearer_token(None)
    with pytest.raises(Exception) as exc_info:
        await mcp_server.call_tool(
            "ingest.text",
            {
                "graph_id": "00000000-0000-0000-0000-000000000000",
                "body": "not-an-object",
            },
        )
    msg = str(exc_info.value).lower()
    assert "principal" not in msg, (
        f"wrong-typed body reached the handler — validation did not fire: {msg}"
    )


@pytest.mark.asyncio
async def test_descriptions_mention_the_result_type(mcp_server):
    """Each tool whose spec declares a `result_model` surfaces that type in its
    description (ADR-023 D4 — the result type is documented for the client)."""
    tools = {t.name: t for t in await mcp_server.list_tools()}
    for spec in REGISTRY:
        if spec.result_model is None:
            continue
        tool = tools.get(spec.name)
        assert tool is not None, f"spec {spec.name} not projected"
        assert spec.result_model.__name__ in tool.description, (
            f"{spec.name}: description does not mention result type "
            f"{spec.result_model.__name__}"
        )
