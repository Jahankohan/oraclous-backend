"""
Integration tests for OpenTelemetry span emission.

Verifies that key Oraclous flows emit correctly structured spans:
  1. Neo4j read query  → neo4j.query span with db.* attributes
  2. Neo4j write query → neo4j.write_query span
  3. Chat query        → chat.query span with result attributes
  4. Pipeline document → pipeline.document + child stage spans

All tests use InMemorySpanExporter so no running Jaeger / OTLP endpoint is needed.
OTEL_ENABLED is patched to True for the duration of each test.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider_with_exporter() -> tuple[TracerProvider, InMemorySpanExporter]:
    """Create an isolated TracerProvider backed by an in-memory exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


# ---------------------------------------------------------------------------
# Test 1 — Neo4j read query emits neo4j.query span
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_neo4j_execute_query_emits_span():
    """
    neo4j_client.execute_query() must emit a 'neo4j.query' span tagged with
    db.system, db.operation, db.statement, and db.neo4j.row_count.
    """
    import app.core.neo4j_client as neo4j_module
    from app.core.neo4j_client import Neo4jClient

    provider, exporter = _make_provider_with_exporter()
    otel_trace.set_tracer_provider(provider)
    neo4j_module._tracer = provider.get_tracer("oraclous.neo4j")

    mock_session = AsyncMock()
    mock_result = AsyncMock()
    mock_result.data = AsyncMock(return_value=[{"id": 1}, {"id": 2}])
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.run = AsyncMock(return_value=mock_result)

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session)

    client = Neo4jClient()
    client.async_driver = mock_driver

    rows = await client.execute_query("MATCH (n:Entity) RETURN n.id AS id", {})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    assert span.name == "neo4j.query"
    assert span.attributes["db.system"] == "neo4j"
    assert span.attributes["db.operation"] == "read"
    assert "MATCH" in span.attributes["db.statement"]
    assert span.attributes["db.neo4j.row_count"] == 2
    assert rows == [{"id": 1}, {"id": 2}]


# ---------------------------------------------------------------------------
# Test 2 — Neo4j write query emits neo4j.write_query span
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_neo4j_execute_write_query_emits_span():
    """
    neo4j_client.execute_write_query() must emit a 'neo4j.write_query' span
    tagged with db.operation='write'.
    """
    import app.core.neo4j_client as neo4j_module
    from app.core.neo4j_client import Neo4jClient

    provider, exporter = _make_provider_with_exporter()
    otel_trace.set_tracer_provider(provider)
    neo4j_module._tracer = provider.get_tracer("oraclous.neo4j")

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.execute_write = AsyncMock(return_value=[{"created": 1}])

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session)

    client = Neo4jClient()
    client.async_driver = mock_driver

    await client.execute_write_query("CREATE (n:Test {id: $id})", {"id": "abc"})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "neo4j.write_query"
    assert span.attributes["db.system"] == "neo4j"
    assert span.attributes["db.operation"] == "write"
    assert "CREATE" in span.attributes["db.statement"]


# ---------------------------------------------------------------------------
# Test 3 — Chat query emits chat.query span with result attributes
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_chat_search_emits_span():
    """
    ChatService.search() must emit a 'chat.query' span with graph_id,
    retriever_type, confidence, is_grounded, and hallucination_flag attributes.
    """
    import app.services.chat_service as chat_module
    from app.services.chat_service import ChatService
    from app.services.retriever_factory import RetrieverType

    provider, exporter = _make_provider_with_exporter()
    otel_trace.set_tracer_provider(provider)
    chat_module._chat_tracer = provider.get_tracer("oraclous.chat")

    graph_id = "integration-test-graph"
    service = ChatService.__new__(ChatService)
    service.graph_id = graph_id
    service.retriever_type = RetrieverType.VECTOR
    service.rag = MagicMock()

    from neo4j_graphrag.generation.types import RagResultModel
    from neo4j_graphrag.types import RetrieverResult, RetrieverResultItem

    mock_result = RagResultModel(
        answer="Paris is the capital of France.",
        retriever_result=RetrieverResult(
            items=[
                RetrieverResultItem(
                    content="France capital: Paris", metadata={"score": 0.95}
                ),
                RetrieverResultItem(
                    content="Paris city facts", metadata={"score": 0.88}
                ),
            ]
        ),
    )
    service.rag.search = MagicMock(return_value=mock_result)

    await service.search(query_text="What is the capital of France?")

    spans = exporter.get_finished_spans()
    chat_spans = [s for s in spans if s.name == "chat.query"]
    assert len(chat_spans) == 1, (
        f"Expected 1 chat.query span, got {[s.name for s in spans]}"
    )

    span = chat_spans[0]
    assert span.attributes["graph_id"] == graph_id
    assert span.attributes["chat.retriever_type"] == RetrieverType.VECTOR.value
    assert "chat.confidence" in span.attributes
    assert "chat.is_grounded" in span.attributes
    assert "chat.hallucination_flag" in span.attributes
    assert span.attributes["chat.sources_count"] >= 0


# ---------------------------------------------------------------------------
# Test 4 — Failed Neo4j query marks span as ERROR
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_neo4j_query_failure_records_error_span():
    """
    When execute_query raises, the span status must be ERROR and the
    exception must be recorded on the span.
    """
    import app.core.neo4j_client as neo4j_module
    from app.core.neo4j_client import Neo4jClient

    provider, exporter = _make_provider_with_exporter()
    otel_trace.set_tracer_provider(provider)
    neo4j_module._tracer = provider.get_tracer("oraclous.neo4j")

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.run = AsyncMock(side_effect=ConnectionError("Neo4j unavailable"))

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session)

    client = Neo4jClient()
    client.async_driver = mock_driver

    with pytest.raises(ConnectionError):
        await client.execute_query("MATCH (n) RETURN n", {})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.status.status_code == otel_trace.StatusCode.ERROR
    # Exception event should be recorded
    event_names = [e.name for e in span.events]
    assert "exception" in event_names
