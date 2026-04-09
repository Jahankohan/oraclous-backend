"""
Unit tests for ORA-218 — lifespan must initialize both async and sync Neo4j drivers.

Regression guard: connect_sync() must be called during startup so that
sync-driver-dependent endpoints are available immediately on container start
rather than returning 503 until the first Celery job fires.
"""

import ast
import inspect
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _ensure_stub(name):
    """Insert a stub module if not already present."""
    if name not in sys.modules:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    return sys.modules[name]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lifespan_calls_connect_sync(monkeypatch):
    """connect_sync() must be called during lifespan startup (ORA-218 regression)."""
    # Stub modules that are unavailable in unit-test context
    for mod_name in ["app.api.v1.router", "app.core.rate_limiter"]:
        stub = _ensure_stub(mod_name)
        if mod_name == "app.api.v1.router":
            stub.api_router = MagicMock()
        elif mod_name == "app.core.rate_limiter":
            stub.limiter = MagicMock()

    slowapi_stub = _ensure_stub("slowapi")
    if not hasattr(slowapi_stub, "RateLimitExceeded"):
        slowapi_stub.RateLimitExceeded = type("RateLimitExceeded", (Exception,), {})
    if not hasattr(slowapi_stub, "_rate_limit_exceeded_handler"):
        slowapi_stub._rate_limit_exceeded_handler = MagicMock()
    slowapi_errors = _ensure_stub("slowapi.errors")
    if not hasattr(slowapi_errors, "RateLimitExceeded"):
        slowapi_errors.RateLimitExceeded = slowapi_stub.RateLimitExceeded

    # Build mock neo4j_client
    mock_client = MagicMock()
    mock_client.connect = AsyncMock()
    mock_client.connect_sync = MagicMock()
    mock_client.disconnect = AsyncMock()
    mock_client.async_driver = MagicMock()

    # Service mocks
    mock_rebac = MagicMock()
    mock_rebac.initialize_schema = AsyncMock()
    mock_rebac.initialize_schema_full = AsyncMock()
    mock_rebac.seed_system_permissions = AsyncMock()
    mock_rebac.sync_existing_data = AsyncMock()

    mock_snapshot = MagicMock()
    mock_snapshot.ensure_indexes = AsyncMock()

    mock_sa_service = MagicMock()
    mock_sa_service.initialize_schema = AsyncMock()

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    # Force re-import of app.main so our patches take effect
    sys.modules.pop("app.main", None)

    with (
        patch("app.core.neo4j_client.neo4j_client", mock_client),
        patch("app.core.database.create_tables", new_callable=AsyncMock),
        patch("app.core.database.async_session_maker", return_value=mock_session),
        patch("app.core.telemetry.setup_telemetry"),
        patch("app.core.telemetry.shutdown_telemetry"),
        patch("app.core.telemetry.instrument_fastapi"),
    ):
        import importlib

        import app.main as main_module

        importlib.reload(main_module)
        lifespan_fn = main_module.lifespan
        app_obj = main_module.app

        with (
            patch(
                "app.services.rebac_service.rebac_service", mock_rebac, create=True
            ),
            patch(
                "app.core.database.async_session_maker", return_value=mock_session
            ),
            patch(
                "app.services.snapshot_service.snapshot_service",
                mock_snapshot,
                create=True,
            ),
            patch(
                "app.services.pipeline_service.ensure_fingerprint_indexes",
                AsyncMock(),
                create=True,
            ),
            patch(
                "app.services.code_parser_service.ensure_code_schema",
                AsyncMock(),
                create=True,
            ),
            patch(
                "app.services.service_account_service.service_account_service",
                mock_sa_service,
                create=True,
            ),
            patch.object(main_module, "neo4j_client", mock_client),
        ):
            async with lifespan_fn(app_obj):
                pass

    mock_client.connect.assert_awaited_once()
    mock_client.connect_sync.assert_called_once()


@pytest.mark.unit
def test_main_py_calls_connect_sync_after_connect():
    """Static AST guard: lifespan source must call connect_sync() after connect()."""
    import app.main as main_module

    source = inspect.getsource(main_module.lifespan)
    tree = ast.parse(source)

    # Collect (lineno, method_name) in source order
    calls = sorted(
        (node.lineno, node.func.attr)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    )
    call_names = [name for _, name in calls]

    assert "connect" in call_names, "lifespan must call neo4j_client.connect()"
    assert "connect_sync" in call_names, (
        "lifespan must call neo4j_client.connect_sync() — ORA-218 regression guard"
    )

    connect_idx = call_names.index("connect")
    connect_sync_idx = call_names.index("connect_sync")
    assert connect_idx < connect_sync_idx, (
        "connect_sync() must be called after connect()"
    )
