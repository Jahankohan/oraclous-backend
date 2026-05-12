"""ADR-007 1:1 invariant — every REST endpoint has a matching MCP tool.

This test is the verifiable contract of ADR-007 (§Validation): for every
HTTP route registered on `assessments.py` + `assessments_reads.py`, there
must be a corresponding MCP tool registered on the FastMCP server. No
tool without an endpoint; no endpoint without a tool.

The mapping is encoded explicitly here. If a new endpoint is added without
a corresponding tool (or vice-versa), this test fails — and the ADR-007
invariant is preserved by CI.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("ORACLOUS_API_KEY", "test-completeness-key")
os.environ.setdefault("ORACLOUS_BASE_URL", "http://localhost:8003")


# Explicit REST endpoint → MCP tool mapping. The endpoint signature is
# (method, path). Path-parameter names match the REST decorator exactly.
ENDPOINT_TO_TOOL: dict[tuple[str, str], str] = {
    # Writes (mirror app/api/v1/endpoints/assessments.py)
    ("POST", "/assessments/runs"): "assessment.create_run",
    (
        "PATCH",
        "/assessments/runs/{run_id}/module-runs/{module_run_id}",
    ): "assessment.update_module_run",
    (
        "POST",
        "/assessments/runs/{run_id}/findings:bulk",
    ): "assessment.record_finding_bulk",
    ("POST", "/assessments/runs/{run_id}/conflicts"): "assessment.record_conflict",
    (
        "POST",
        "/assessments/runs/{run_id}/unresolved-questions",
    ): "assessment.record_unresolved_question",
    (
        "POST",
        "/assessments/runs/{run_id}/deliverables",
    ): "assessment.persist_deliverable",
    (
        "POST",
        "/assessments/runs/{run_id}/deliverables:bulk-final",
    ): "assessment.persist_final_docs",
    ("POST", "/assessments/runs/{run_id}:finalize"): "assessment.finalize_run",
    ("POST", "/assessments/runs/{run_id}:heartbeat"): "assessment.heartbeat_run",
    (
        "POST",
        "/assessments/runs/{run_id}/module-runs/{module_run_id}:heartbeat",
    ): "assessment.heartbeat_module_run",
    ("POST", "/assessments/registry/{kind}"): "registry.persist_item",
    # Reads (mirror app/api/v1/endpoints/assessments_reads.py)
    ("GET", "/assessments/runs"): "assessment.list_runs",
    ("GET", "/assessments/runs/{run_id}"): "assessment.get_run",
    (
        "GET",
        "/assessments/runs/{run_id}/waves/{wave}",
    ): "assessment.get_wave_status",
    (
        "GET",
        "/assessments/runs/{run_id}/module-runs",
    ): "assessment.list_module_runs",
    ("GET", "/assessments/runs/{run_id}/findings"): "assessment.list_findings",
    ("GET", "/assessments/runs/{run_id}/conflicts"): "assessment.list_conflicts",
    (
        "GET",
        "/assessments/runs/{run_id}/unresolved-questions",
    ): "assessment.list_unresolved_questions",
    (
        "GET",
        "/assessments/runs/{run_id}/deliverables",
    ): "assessment.list_deliverables",
    (
        "GET",
        "/assessments/runs/{run_id}/deliverables/{deliverable_id}/content",
    ): "assessment.get_deliverable_content",
    (
        "GET",
        "/assessments/templates/{template_slug}/modules",
    ): "assessment.list_template_modules",
    ("GET", "/assessments/registry/{kind}"): "registry.list_items",
    ("GET", "/assessments/registry/{kind}/{slug}"): "registry.get_item",
    (
        "GET",
        "/assessments/registry/{kind}/{slug}/{version}/content",
    ): "registry.get_item_content",
    ("GET", "/assessments/findings:search"): "assessment.search_findings",
}


def _registered_tool_names() -> set[str]:
    """Return the set of names FastMCP has registered."""
    import app.mcp.server as srv

    return set(srv.mcp._tool_manager._tools.keys())


def _registered_endpoint_keys() -> set[tuple[str, str]]:
    """Return the set of (method, path) pairs from the assessment + reads routers."""
    from app.api.v1.endpoints import assessments, assessments_reads

    keys: set[tuple[str, str]] = set()
    for router in (assessments.router, assessments_reads.router):
        for route in router.routes:
            methods = getattr(route, "methods", None) or set()
            path = getattr(route, "path", "")
            for m in methods:
                if m in ("HEAD", "OPTIONS"):
                    continue
                keys.add((m, path))
    return keys


def test_endpoint_count_matches_mapping_count():
    """Sanity — our explicit mapping covers the same endpoints the router exposes."""
    router_keys = _registered_endpoint_keys()
    mapping_keys = set(ENDPOINT_TO_TOOL.keys())
    extra_in_router = router_keys - mapping_keys
    extra_in_mapping = mapping_keys - router_keys
    assert not extra_in_router, (
        "REST endpoints exist with no MCP tool mapping in this test "
        f"(new endpoint without ADR-007 wrapper?): {sorted(extra_in_router)}"
    )
    assert not extra_in_mapping, (
        "ENDPOINT_TO_TOOL references endpoints that no longer exist on the "
        f"REST router: {sorted(extra_in_mapping)}"
    )


def test_every_endpoint_has_a_registered_mcp_tool():
    """ADR-007 invariant — every REST endpoint must have its 1:1 MCP tool."""
    tools = _registered_tool_names()
    missing = [tool for tool in ENDPOINT_TO_TOOL.values() if tool not in tools]
    assert not missing, (
        "ADR-007 violated — REST endpoints exist with no corresponding MCP "
        f"tool registered: {missing}"
    )


def test_no_phantom_assessment_or_registry_tools():
    """Inverse direction — no `assessment.*` or `registry.*` tool without an endpoint."""
    tools = _registered_tool_names()
    namespaced = {t for t in tools if t.startswith(("assessment.", "registry."))}
    expected = set(ENDPOINT_TO_TOOL.values())
    extras = namespaced - expected
    assert not extras, (
        "MCP tools registered under assessment.* / registry.* without a "
        f"matching REST endpoint (1:1 invariant from ADR-007 broken): {sorted(extras)}"
    )


@pytest.mark.parametrize("name", sorted(set(ENDPOINT_TO_TOOL.values())))
def test_each_tool_callable_is_async(name):
    """Every assessment + registry tool must be an async function.

    The service layer is async, and the MCP transport awaits tool returns;
    a sync function would silently swallow ``await svc.<...>`` (returning a
    coroutine instead of the value). Catch that at test time.
    """
    import inspect

    import app.mcp.server as srv

    tool = srv.mcp._tool_manager._tools[name]
    fn = tool.fn if hasattr(tool, "fn") else tool
    assert inspect.iscoroutinefunction(fn), (
        f"MCP tool {name!r} must be an async function (service-layer "
        "tools all need to await async Cypher calls)."
    )
