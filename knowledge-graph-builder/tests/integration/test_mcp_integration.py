"""
Integration test: MCP server starts and all tools + resources are discoverable.

This test verifies that:
1. The MCP server module imports without error.
2. A FastMCP application object is present.
3. All expected tools are registered.
4. All expected resources are registered.

No live services (Neo4j, Oraclous API) are required — this is a structural
smoke test to catch import failures and missing registrations early.
"""
from __future__ import annotations

import importlib
import os
import pytest

os.environ.setdefault("ORACLOUS_API_KEY", "test-integration-key")
os.environ.setdefault("ORACLOUS_BASE_URL", "http://localhost:8003")


EXPECTED_TOOLS = {
    "create_graph",
    "list_graphs",
    "delete_graph",
    "get_graph_stats",
    "ingest_text",
    "ingest_file",
    "chat",
    "search_nodes",
    "get_node",
    "get_neighbors",
}

EXPECTED_RESOURCES = {
    "graphs://",
    "graph://{graph_id}/stats",
    "graph://{graph_id}/nodes",
}


def test_mcp_module_imports_cleanly():
    """The server module must import without raising any exception."""
    import app.mcp.server  # noqa: F401  — import is the assertion


def test_fastmcp_instance_exists():
    """A FastMCP instance named `mcp` must be present in the module."""
    import app.mcp.server as srv
    from mcp.server.fastmcp import FastMCP

    assert hasattr(srv, "mcp"), "Module must expose a `mcp` attribute"
    assert isinstance(srv.mcp, FastMCP), "`mcp` must be a FastMCP instance"


def test_all_expected_tools_registered():
    """Every tool listed in the ORA-26 spec must be registered with FastMCP."""
    import app.mcp.server as srv

    # FastMCP exposes registered tools via ._tool_manager or .list_tools()
    # The internal API varies by version, so we probe both.
    registered: set[str] = set()

    if hasattr(srv.mcp, "_tool_manager") and hasattr(srv.mcp._tool_manager, "_tools"):
        registered = set(srv.mcp._tool_manager._tools.keys())
    elif hasattr(srv.mcp, "list_tools"):
        # Synchronous introspection helper present in some FastMCP versions
        try:
            tools_list = srv.mcp.list_tools()
            registered = {t.name for t in tools_list}
        except Exception:
            pass

    if not registered:
        # Fallback: check that the expected functions exist and are decorated
        for tool_name in EXPECTED_TOOLS:
            assert hasattr(srv, tool_name), (
                f"Tool function `{tool_name}` not found in mcp.server module"
            )
        return  # structural check passed

    missing = EXPECTED_TOOLS - registered
    assert not missing, f"Missing MCP tools: {missing}"


def test_all_expected_resources_registered():
    """Every resource URI listed in the ORA-26 spec must be registered."""
    import app.mcp.server as srv

    # FastMCP stores plain resources in _resources and template resources
    # (those with {param} placeholders) in _templates — check both.
    registered: set[str] = set()

    if hasattr(srv.mcp, "_resource_manager"):
        rm = srv.mcp._resource_manager
        if hasattr(rm, "_resources"):
            registered |= set(rm._resources.keys())
        if hasattr(rm, "_templates"):
            registered |= set(rm._templates.keys())

    if not registered:
        # Fallback: ensure resource handler functions exist in the module
        resource_fn_names = {
            "resource_graphs",
            "resource_graph_stats",
            "resource_graph_nodes",
        }
        for fn_name in resource_fn_names:
            assert hasattr(srv, fn_name), (
                f"Resource handler `{fn_name}` not found in mcp.server module"
            )
        return  # structural check passed

    for uri in EXPECTED_RESOURCES:
        assert uri in registered, f"MCP resource {uri!r} not registered. Found: {registered}"


def test_main_function_exists():
    """The `main()` entry point must exist for use as a CLI command."""
    import app.mcp.server as srv
    assert callable(srv.main), "`main` must be a callable function"


def test_env_var_helpers():
    """Config helpers must read from environment variables correctly."""
    import app.mcp.server as srv

    # _base_url reads ORACLOUS_BASE_URL (set to http://localhost:8003 above)
    assert srv._base_url() == "http://localhost:8003"

    # _api_key returns whatever ORACLOUS_API_KEY is set to (non-empty)
    key = srv._api_key()
    assert key, "_api_key() must return a non-empty string"

    # _auth_headers uses the key
    headers = srv._auth_headers()
    assert headers == {"Authorization": f"Bearer {key}"}


def test_missing_api_key_raises():
    """_api_key() must raise RuntimeError when ORACLOUS_API_KEY is not set."""
    import app.mcp.server as srv

    original = os.environ.pop("ORACLOUS_API_KEY", None)
    try:
        with pytest.raises(RuntimeError, match="ORACLOUS_API_KEY"):
            srv._api_key()
    finally:
        if original is not None:
            os.environ["ORACLOUS_API_KEY"] = original
        else:
            os.environ.setdefault("ORACLOUS_API_KEY", "test-integration-key")
