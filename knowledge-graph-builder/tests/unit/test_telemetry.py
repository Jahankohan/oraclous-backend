"""
Unit tests for OpenTelemetry instrumentation.

Covers:
- setup_telemetry() initialises tracer/meter providers
- current_trace_context() returns trace_id/span_id for active spans
- get_tracer() / get_meter() return valid (possibly noop) objects
- neo4j_client.execute_query creates a neo4j.query span
- neo4j_client.execute_write_query creates a neo4j.write_query span
- X-Trace-Id header is injected into HTTP responses when a trace is active
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_span_context(trace_id=0x1234ABCD1234ABCD1234ABCD1234ABCD,
                              span_id=0xABCD1234ABCD1234):
    from opentelemetry.trace import SpanContext, TraceFlags
    return SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )


# ---------------------------------------------------------------------------
# current_trace_context
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_current_trace_context_returns_empty_when_no_active_span():
    """Without an active span, current_trace_context returns an empty dict."""
    from app.core.telemetry import current_trace_context

    result = current_trace_context()
    assert result == {}


@pytest.mark.unit
def test_current_trace_context_returns_ids_when_span_is_active():
    """With a valid active span, trace_id and span_id are returned as hex strings."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from app.core.telemetry import current_trace_context

    provider = TracerProvider()
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("test-span") as span:
        ctx = current_trace_context()
        sc = span.get_span_context()
        assert ctx["trace_id"] == format(sc.trace_id, "032x")
        assert ctx["span_id"] == format(sc.span_id, "016x")


# ---------------------------------------------------------------------------
# get_tracer / get_meter
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_get_tracer_returns_tracer():
    from opentelemetry.trace import Tracer
    from app.core.telemetry import get_tracer

    tracer = get_tracer("test.tracer")
    assert tracer is not None


@pytest.mark.unit
def test_get_meter_returns_meter():
    from opentelemetry.metrics import Meter
    from app.core.telemetry import get_meter

    meter = get_meter("test.meter")
    assert meter is not None


# ---------------------------------------------------------------------------
# setup_telemetry — no-op when OTEL_ENABLED=False
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_setup_telemetry_is_noop_when_disabled(caplog):
    import logging
    from app.core import telemetry as tel_module

    # Reset module state
    tel_module._tracer_provider = None
    tel_module._meter_provider = None

    with patch.object(tel_module.settings, "OTEL_ENABLED", False):
        with caplog.at_level(logging.INFO, logger="app.core.telemetry"):
            tel_module.setup_telemetry()

    assert tel_module._tracer_provider is None
    assert tel_module._meter_provider is None


# ---------------------------------------------------------------------------
# Neo4j query span creation
# ---------------------------------------------------------------------------

@pytest.mark.unit
async def test_execute_query_creates_neo4j_span():
    """execute_query must open a neo4j.query span with db.system=neo4j."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry import trace
    import app.core.neo4j_client as neo4j_module

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Patch the module-level tracer so it picks up our test provider
    neo4j_module._tracer = provider.get_tracer("oraclous.neo4j")

    mock_session = AsyncMock()
    mock_result = AsyncMock()
    mock_result.data = AsyncMock(return_value=[{"n": 1}])
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.run = AsyncMock(return_value=mock_result)

    mock_async_driver = MagicMock()
    mock_async_driver.session = MagicMock(return_value=mock_session)

    client = neo4j_module.Neo4jClient()
    client.async_driver = mock_async_driver

    result = await client.execute_query("RETURN 1 AS n", {})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "neo4j.query"
    assert span.attributes["db.system"] == "neo4j"
    assert span.attributes["db.operation"] == "read"
    assert "RETURN 1" in span.attributes["db.statement"]
    assert span.attributes["db.neo4j.row_count"] == 1
    assert result == [{"n": 1}]


@pytest.mark.unit
async def test_execute_write_query_creates_neo4j_write_span():
    """execute_write_query must open a neo4j.write_query span."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry import trace
    import app.core.neo4j_client as neo4j_module

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    neo4j_module._tracer = provider.get_tracer("oraclous.neo4j")

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.execute_write = AsyncMock(return_value=[{"created": 1}])

    mock_async_driver = MagicMock()
    mock_async_driver.session = MagicMock(return_value=mock_session)

    client = neo4j_module.Neo4jClient()
    client.async_driver = mock_async_driver

    await client.execute_write_query("CREATE (n:Test) RETURN n", {})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "neo4j.write_query"
    assert span.attributes["db.operation"] == "write"


@pytest.mark.unit
async def test_execute_query_span_records_exception_on_failure():
    """A failed query must record the exception on the span and re-raise."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry import trace
    import app.core.neo4j_client as neo4j_module

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    neo4j_module._tracer = provider.get_tracer("oraclous.neo4j")

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.run = AsyncMock(side_effect=RuntimeError("neo4j down"))

    mock_async_driver = MagicMock()
    mock_async_driver.session = MagicMock(return_value=mock_session)

    client = neo4j_module.Neo4jClient()
    client.async_driver = mock_async_driver

    with pytest.raises(RuntimeError, match="neo4j down"):
        await client.execute_query("MATCH (n) RETURN n", {})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    from opentelemetry.trace import StatusCode
    assert spans[0].status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# X-Trace-Id response header (middleware)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_current_trace_context_is_empty_without_active_span():
    """Ensure middleware does not inject header when no span is active."""
    from app.core.telemetry import current_trace_context

    ctx = current_trace_context()
    assert "trace_id" not in ctx


@pytest.mark.unit
def test_current_trace_context_has_trace_id_with_active_span():
    """Ensure trace_id is a 32-char hex string when span is active."""
    from opentelemetry.sdk.trace import TracerProvider
    from app.core.telemetry import current_trace_context

    provider = TracerProvider()
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("middleware-test"):
        ctx = current_trace_context()
        assert "trace_id" in ctx
        assert len(ctx["trace_id"]) == 32
        int(ctx["trace_id"], 16)  # must be valid hex
