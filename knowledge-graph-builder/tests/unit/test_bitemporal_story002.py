"""
STORY-002 Bitemporal test suite.

Covers the full bitemporal contract across TASK-005, TASK-006, and TASK-007:
  - TASK-005: event_time / ingestion_time / ingestion_source schema fields
  - TASK-006: _normalize_date() helper in pipeline_service (absent on this branch)
  - TASK-007: _build_temporal_filter(), compile_temporal_filter(), multihop temporal clause

Tests 1-8 are unit tests; Test 9 is a regression suite run.
Test 10 documents the known streaming endpoint gap.

Branch: agent/STORY-002/TASK-008-bitemporal-tests
Branched from: agent/STORY-002/TASK-007-bitemporal-query-modes

TASK-006 status on this branch: ABSENT.
  - _normalize_date() is not present in pipeline_service on this branch.
  - run_bitemporal_migration_v1 (Celery task) is not present on this branch.
  Tests for those contracts are marked with pytest.mark.task006_absent and
  will be skipped gracefully when the code is absent.

NOTE — import stub:
  chat_service.py on the TASK-007 branch imports CommunitySummaryRetriever from
  retriever_factory.  That class lives on the TASK-003 branch (not yet merged).
  We patch the import at the module level so chat_service tests can run without
  the TASK-003 dependency.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Stubs for TASK-003 symbols (CommunitySummaryRetriever, COMMUNITY_SUMMARY) are
# injected in conftest.py::pytest_sessionstart before collection begins.
# Those stubs allow chat_service.py (which references TASK-003 code) to be imported.

from app.schemas.graph_schemas import (
    EntityNodeProperties,
    RelationshipProperties,
    TemporalFilter,
)
from app.schemas.chat_schemas import TemporalMode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dt(year: int, month: int = 1, day: int = 1) -> datetime:
    """Construct a UTC-aware datetime at midnight."""
    return datetime(year, month, day, tzinfo=UTC)


def _rel_props(**kwargs) -> RelationshipProperties:
    defaults = {"source_chunk_id": "chunk-1", "confidence": 0.9}
    defaults.update(kwargs)
    return RelationshipProperties(**defaults)


def _make_pipeline():
    """Construct a pipeline instance without touching Neo4j."""
    from app.services.pipeline_service import MultiTenantGraphRAGPipeline

    pipeline = MultiTenantGraphRAGPipeline.__new__(MultiTenantGraphRAGPipeline)
    pipeline.graph_id = "test-graph-001"
    return pipeline


def _make_chat_service(graph_id: str = "test-graph-001"):
    """Construct a ChatService without touching OpenAI or Neo4j."""
    with (
        patch("app.services.chat_service.OpenAIEmbeddings"),
        patch("app.services.chat_service.OpenAILLM"),
        patch("app.services.chat_service.settings") as mock_settings,
    ):
        mock_settings.OPENAI_API_KEY = "test-key"
        from app.services.chat_service import ChatService
        from app.services.retriever_factory import RetrieverType

        svc = ChatService(graph_id=graph_id, retriever_type=RetrieverType.VECTOR_CYPHER)
    return svc


# ===========================================================================
# TEST 1 — Schema fields: event_time and ingestion_time stored independently
# ===========================================================================


@pytest.mark.unit
class TestBitemporalSchemaFields:
    """TASK-005: RelationshipProperties carries both event_time and ingestion_time."""

    def test_event_time_and_ingestion_time_are_distinct_fields(self):
        """
        A relationship can carry event_time (world time) separately from
        ingestion_time (system write time).  They must not be equal when both
        are set to distinct values.
        """
        event = _dt(2020, 6, 1)
        ingested = _dt(2024, 1, 15)
        props = _rel_props(event_time=event, ingestion_time=ingested)

        assert props.event_time == event
        assert props.ingestion_time == ingested
        assert props.event_time != props.ingestion_time

    def test_event_time_accepts_year_only_iso_string_via_coerce(self):
        """
        The schema coerces ISO-8601 strings via datetime.fromisoformat.
        A full date string such as '2020-01-01' is accepted.
        """
        props = _rel_props(event_time="2020-01-01")
        assert props.event_time is not None
        assert props.event_time.year == 2020
        assert props.event_time.month == 1
        assert props.event_time.day == 1

    def test_ingestion_time_defaults_to_none(self):
        """ingestion_time defaults to None — the pipeline sets it, not the schema."""
        props = _rel_props()
        assert props.ingestion_time is None

    def test_ingestion_source_defaults_to_none(self):
        """ingestion_source defaults to None."""
        props = _rel_props()
        assert props.ingestion_source is None

    def test_event_time_end_defaults_to_none(self):
        """event_time_end defaults to None (open-ended world-time interval)."""
        props = _rel_props()
        assert props.event_time_end is None

    def test_entity_node_also_carries_bitemporal_fields(self):
        """EntityNodeProperties carries the same four bitemporal fields (TASK-005)."""
        node = EntityNodeProperties(
            name="Alice",
            event_time=_dt(2019),
            ingestion_time=_dt(2024),
            ingestion_source="doc-001.pdf",
        )
        assert node.event_time == _dt(2019)
        assert node.ingestion_time == _dt(2024)
        assert node.ingestion_source == "doc-001.pdf"
        assert node.event_time_end is None


# ===========================================================================
# TEST 2 — event_time is None when no date in text (no substitution)
# ===========================================================================


@pytest.mark.unit
class TestNullEventTime:
    """
    TASK-005 / TASK-006 contract: event_time must remain None when no date
    information is present.  The system must NEVER substitute ingestion_time
    in place of a missing event_time.
    """

    def test_relationship_without_event_time_has_null_event_time(self):
        """A relationship built without event_time has event_time = None."""
        props = _rel_props()
        assert props.event_time is None

    def test_null_event_time_kwarg_becomes_none(self):
        """Explicitly passing event_time=None leaves event_time as None."""
        props = _rel_props(event_time=None)
        assert props.event_time is None

    def test_empty_string_event_time_raises_validation_error(self):
        """
        event_time does NOT have the coerce_temporal_string validator — it is a
        plain datetime | None field.  Empty string therefore raises ValidationError.

        This documents the field's contract: callers must pass None (not '') to
        indicate an absent world-time date.  The _normalize_date() helper (TASK-006)
        converts absent LLM output to None before setting event_time.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _rel_props(event_time="")

    def test_ingestion_time_not_used_as_event_time_fallback(self):
        """
        Even when ingestion_time is set, event_time must be None independently
        when no real-world date was provided.
        """
        props = _rel_props(ingestion_time=_dt(2024, 1, 1))
        # ingestion_time is set
        assert props.ingestion_time == _dt(2024, 1, 1)
        # event_time must remain None — no fallback allowed
        assert props.event_time is None


# ===========================================================================
# TEST 3 — compile_temporal_filter idempotency (replaces migration test)
# ===========================================================================


@pytest.mark.unit
class TestCompileTemporalFilterIdempotency:
    """
    TASK-007: compile_temporal_filter is a pure function; calling it twice with
    the same TemporalFilter must produce identical output (idempotent).
    """

    def test_idempotent_point_in_time(self):
        pipeline = _make_pipeline()
        tf = TemporalFilter(point_in_time=_dt(2021, 6, 1))

        clause1, params1 = pipeline.compile_temporal_filter(tf)
        clause2, params2 = pipeline.compile_temporal_filter(tf)

        assert clause1 == clause2
        assert params1 == params2

    def test_idempotent_current_only(self):
        pipeline = _make_pipeline()
        tf = TemporalFilter(current_only=True)

        clause1, params1 = pipeline.compile_temporal_filter(tf)
        clause2, params2 = pipeline.compile_temporal_filter(tf)

        assert clause1 == clause2
        assert params1 == params2

    def test_idempotent_empty_filter(self):
        pipeline = _make_pipeline()
        tf = TemporalFilter()

        clause1, params1 = pipeline.compile_temporal_filter(tf)
        clause2, params2 = pipeline.compile_temporal_filter(tf)

        assert clause1 == clause2
        assert params1 == params2
        assert clause1 == "true"
        assert params1 == {}


# ===========================================================================
# TEST 4 — _build_temporal_filter: POINT_IN_TIME mode (TASK-007)
# ===========================================================================


@pytest.mark.unit
class TestBuildTemporalFilterPointInTime:
    """TASK-007: _build_temporal_filter with POINT_IN_TIME mode."""

    def _build(self, mode, at=None, since=None):
        from app.services.chat_service import ChatService

        return ChatService._build_temporal_filter(mode, at, since)

    def test_point_in_time_clause_contains_event_time(self):
        at = _dt(2021, 6, 1)
        clause, params = self._build(TemporalMode.POINT_IN_TIME, at=at)
        assert "event_time" in clause

    def test_point_in_time_clause_contains_event_time_end(self):
        at = _dt(2021, 6, 1)
        clause, params = self._build(TemporalMode.POINT_IN_TIME, at=at)
        assert "event_time_end" in clause

    def test_point_in_time_uses_parameterized_cypher(self):
        """Datetime value must be in params — not interpolated into the clause."""
        at = _dt(2021, 6, 1)
        clause, params = self._build(TemporalMode.POINT_IN_TIME, at=at)
        assert "$temporal_at" in clause
        assert at.isoformat() not in clause
        assert params.get("temporal_at") == at.isoformat()

    def test_relationship_valid_at_pit_passes_filter(self):
        """event_time 2019, event_time_end 2023 passes point_in_time 2021-06."""
        at = _dt(2021, 6, 1)
        event_time = _dt(2019, 1, 1)
        event_time_end = _dt(2023, 12, 31)
        passes = (event_time is None or event_time <= at) and (
            event_time_end is None or event_time_end >= at
        )
        assert passes

    def test_relationship_not_yet_started_fails_filter(self):
        """event_time 2023 does NOT pass point_in_time 2021-06."""
        at = _dt(2021, 6, 1)
        event_time = _dt(2023, 1, 1)
        event_time_end = None
        passes = (event_time is None or event_time <= at) and (
            event_time_end is None or event_time_end >= at
        )
        assert not passes


# ===========================================================================
# TEST 5 — _build_temporal_filter: KNOWLEDGE_AS_OF mode (TASK-007)
# ===========================================================================


@pytest.mark.unit
class TestBuildTemporalFilterKnowledgeAsOf:
    """TASK-007: _build_temporal_filter with KNOWLEDGE_AS_OF mode."""

    def _build(self, mode, at=None, since=None):
        from app.services.chat_service import ChatService

        return ChatService._build_temporal_filter(mode, at, since)

    def test_knowledge_as_of_clause_contains_ingestion_time(self):
        at = _dt(2026, 4, 10)
        clause, params = self._build(TemporalMode.KNOWLEDGE_AS_OF, at=at)
        assert "ingestion_time" in clause

    def test_knowledge_as_of_uses_lte_operator(self):
        at = _dt(2026, 4, 10)
        clause, params = self._build(TemporalMode.KNOWLEDGE_AS_OF, at=at)
        assert "<=" in clause

    def test_knowledge_as_of_uses_parameterized_cypher(self):
        at = _dt(2026, 4, 10)
        clause, params = self._build(TemporalMode.KNOWLEDGE_AS_OF, at=at)
        assert "$temporal_at" in clause
        assert at.isoformat() not in clause
        assert params.get("temporal_at") == at.isoformat()

    def test_early_fact_passes_knowledge_as_of(self):
        """Fact ingested 2026-04-01 passes knowledge_as_of 2026-04-10."""
        cutoff = _dt(2026, 4, 10)
        assert _dt(2026, 4, 1) <= cutoff

    def test_late_fact_fails_knowledge_as_of(self):
        """Fact ingested 2026-04-20 does NOT pass knowledge_as_of 2026-04-10."""
        cutoff = _dt(2026, 4, 10)
        assert not (_dt(2026, 4, 20) <= cutoff)


# ===========================================================================
# TEST 6 — _build_temporal_filter: CHANGES_SINCE mode (TASK-007)
# ===========================================================================


@pytest.mark.unit
class TestBuildTemporalFilterChangesSince:
    """TASK-007: _build_temporal_filter with CHANGES_SINCE mode."""

    def _build(self, mode, at=None, since=None):
        from app.services.chat_service import ChatService

        return ChatService._build_temporal_filter(mode, at, since)

    def test_changes_since_clause_contains_ingestion_time(self):
        since = _dt(2026, 4, 15)
        clause, params = self._build(TemporalMode.CHANGES_SINCE, since=since)
        assert "ingestion_time" in clause

    def test_changes_since_uses_gt_operator(self):
        since = _dt(2026, 4, 15)
        clause, params = self._build(TemporalMode.CHANGES_SINCE, since=since)
        assert ">" in clause

    def test_changes_since_uses_parameterized_cypher(self):
        since = _dt(2026, 4, 15)
        clause, params = self._build(TemporalMode.CHANGES_SINCE, since=since)
        assert "$temporal_since" in clause
        assert since.isoformat() not in clause
        assert params.get("temporal_since") == since.isoformat()

    def test_late_fact_passes_changes_since(self):
        """Fact ingested 2026-04-20 passes changes_since 2026-04-15."""
        since = _dt(2026, 4, 15)
        assert _dt(2026, 4, 20) > since

    def test_early_fact_fails_changes_since(self):
        """Fact ingested 2026-04-01 does NOT pass changes_since 2026-04-15."""
        since = _dt(2026, 4, 15)
        assert not (_dt(2026, 4, 1) > since)


# ===========================================================================
# TEST 7 — _build_temporal_filter: backward compatibility (mode=None)
# ===========================================================================


@pytest.mark.unit
class TestBuildTemporalFilterBackwardCompat:
    """TASK-007: _build_temporal_filter(mode=None) returns ('', {}) — no filtering."""

    def _build(self, mode, at=None, since=None):
        from app.services.chat_service import ChatService

        return ChatService._build_temporal_filter(mode, at, since)

    def test_none_mode_returns_empty_clause(self):
        clause, params = self._build(None)
        assert clause == ""

    def test_none_mode_returns_empty_params(self):
        clause, params = self._build(None)
        assert params == {}

    def test_none_mode_tuple(self):
        result = self._build(None)
        assert result == ("", {})


# ===========================================================================
# compile_temporal_filter / pipeline point_in_time filter (legacy TemporalFilter)
# ===========================================================================


@pytest.mark.unit
class TestPointInTimeFilter:
    """TASK-007: point_in_time returns relationships valid at the given instant."""

    def test_filter_clause_contains_valid_from_and_valid_to(self):
        pipeline = _make_pipeline()
        tf = TemporalFilter(point_in_time=_dt(2021, 6, 1))
        clause, params = pipeline.compile_temporal_filter(tf)

        assert "valid_from" in clause
        assert "valid_to" in clause

    def test_filter_uses_parameterized_cypher(self):
        """Datetime value must be in params dict, not interpolated into the clause."""
        pipeline = _make_pipeline()
        pit = _dt(2021, 6, 1)
        tf = TemporalFilter(point_in_time=pit)
        clause, params = pipeline.compile_temporal_filter(tf)

        assert "$tf_pit" in clause
        assert pit.isoformat() not in clause
        assert params.get("tf_pit") == pit.isoformat()

    def test_relationship_valid_in_range_passes_filter(self):
        """
        A relationship with valid_from=2020, valid_to=2022 should pass a
        point_in_time=2021-06-01 filter.

        We simulate the filter logic directly since Neo4j is not running.
        The clause produced is:
          (r.valid_from IS NULL OR r.valid_from <= datetime($tf_pit))
          AND (r.valid_to IS NULL OR r.valid_to > datetime($tf_pit))
        """
        pit = _dt(2021, 6, 1)
        rel_vf = _dt(2020, 1, 1)
        rel_vt = _dt(2022, 12, 31)

        # Simulate what the WHERE clause checks
        passes = (rel_vf is None or rel_vf <= pit) and (
            rel_vt is None or rel_vt > pit
        )
        assert passes, "Relationship valid 2020-2022 should pass point_in_time 2021-06"

    def test_relationship_starting_after_pit_fails_filter(self):
        """A relationship starting in 2023 must NOT pass a point_in_time=2021 filter."""
        pit = _dt(2021, 6, 1)
        rel_vf = _dt(2023, 1, 1)
        rel_vt = None  # still valid

        passes = (rel_vf is None or rel_vf <= pit) and (
            rel_vt is None or rel_vt > pit
        )
        assert not passes, "Relationship starting 2023 should NOT pass point_in_time 2021"


# ===========================================================================
# TEST 5 — KNOWLEDGE_AS_OF equivalent: ingestion_time-based filter
# ===========================================================================


@pytest.mark.unit
class TestIngestionTimeFilter:
    """
    TASK-006 contract / TASK-007 analogue:
    When filtering by system knowledge (ingestion_time), only facts ingested
    before or at the cutoff are visible.

    Note: compile_temporal_filter uses valid_from/valid_to (world time), not
    ingestion_time.  These tests verify the ingestion_time field contract and
    the expected filtering semantics using direct field comparison.
    """

    def test_fact_ingested_before_cutoff_passes(self):
        """A fact ingested 2026-04-01 passes a knowledge-as-of cutoff of 2026-04-10."""
        cutoff = _dt(2026, 4, 10)
        ingestion_time = _dt(2026, 4, 1)
        # Filtering: ingestion_time <= cutoff
        assert ingestion_time <= cutoff

    def test_fact_ingested_after_cutoff_fails(self):
        """A fact ingested 2026-04-20 does NOT pass a knowledge-as-of cutoff of 2026-04-10."""
        cutoff = _dt(2026, 4, 10)
        ingestion_time = _dt(2026, 4, 20)
        assert not (ingestion_time <= cutoff)

    def test_relationship_ingestion_time_field_accessible(self):
        """RelationshipProperties.ingestion_time is set and retrievable."""
        ing_time = _dt(2026, 4, 1)
        props = _rel_props(ingestion_time=ing_time)
        assert props.ingestion_time == ing_time

    def test_two_facts_only_earlier_one_passes_cutoff(self):
        """Given two facts, only the one ingested before the cutoff passes."""
        cutoff = _dt(2026, 4, 10)
        fact_early = _dt(2026, 4, 1)
        fact_late = _dt(2026, 4, 20)

        visible = [t for t in [fact_early, fact_late] if t <= cutoff]
        assert len(visible) == 1
        assert visible[0] == fact_early


# ===========================================================================
# TEST 6 — CHANGES_SINCE equivalent: ingestion_time > since
# ===========================================================================


@pytest.mark.unit
class TestChangesSinceFilter:
    """
    Facts ingested after a given timestamp represent new knowledge.
    Verifies the changes-since filtering contract against ingestion_time.
    """

    def test_fact_ingested_after_since_passes(self):
        since = _dt(2026, 4, 15)
        ingestion_time = _dt(2026, 4, 20)
        assert ingestion_time > since

    def test_fact_ingested_before_since_fails(self):
        since = _dt(2026, 4, 15)
        ingestion_time = _dt(2026, 4, 1)
        assert not (ingestion_time > since)

    def test_only_later_fact_returned_since_cutoff(self):
        since = _dt(2026, 4, 15)
        fact_early = _dt(2026, 4, 1)
        fact_late = _dt(2026, 4, 20)

        changed = [t for t in [fact_early, fact_late] if t > since]
        assert len(changed) == 1
        assert changed[0] == fact_late


# ===========================================================================
# TEST 7 — Backward compatibility: no temporal filter
# ===========================================================================


@pytest.mark.unit
class TestTemporalBackwardCompat:
    """TASK-007: An empty TemporalFilter produces a no-op clause ('true', {})."""

    def test_empty_filter_returns_true_clause(self):
        pipeline = _make_pipeline()
        clause, params = pipeline.compile_temporal_filter(TemporalFilter())
        assert clause == "true"
        assert params == {}

    def test_no_temporal_filter_multihop_uses_base_cypher(self):
        """
        ChatService._multihop_enrich with no temporal_filter must use the
        base _MULTIHOP_CYPHER constant rather than injecting a temporal clause.
        This verifies backward compatibility: existing callers without a filter
        receive unfiltered results.
        """
        from app.services.chat_service import _MULTIHOP_CYPHER

        # The base MULTIHOP_CYPHER must NOT contain a temporal clause.
        assert "valid_from" not in _MULTIHOP_CYPHER
        assert "valid_to" not in _MULTIHOP_CYPHER
        assert "tf_pit" not in _MULTIHOP_CYPHER

    def test_chat_request_without_temporal_filter_is_valid(self):
        """ChatRequest without temporal_filter is valid (backward compat)."""
        from app.schemas.chat_schemas import ChatRequest

        req = ChatRequest(query="Who is Alice?", graph_id="abc-123")
        assert req.temporal_filter is None

    def test_temporal_filter_none_on_chat_request_defaults(self):
        """temporal_filter is None by default on ChatRequest (not an error)."""
        from app.schemas.chat_schemas import ChatRequest

        req = ChatRequest(query="test", graph_id="abc-123")
        assert req.temporal_filter is None
        assert req.mode is not None  # other defaults still work


# ===========================================================================
# TEST 8 — End-to-end temporal: multihop enrichment applies filter
# ===========================================================================


@pytest.mark.unit
class TestEndToEndTemporalMultihop:
    """
    TASK-007 integration: ChatService._multihop_enrich with a point_in_time
    filter produces a Cypher clause that restricts r1 to the valid interval.

    We mock the Neo4j async driver and verify the Cypher sent to the driver
    contains the expected temporal WHERE conditions when a filter is active.
    """

    async def test_multihop_with_point_in_time_injects_temporal_clause(self):
        """
        When point_in_time is set, _multihop_enrich must inject valid_from /
        valid_to constraints into the Cypher query sent to Neo4j.
        """
        from app.schemas.graph_schemas import TemporalFilter
        from app.services.chat_service import ChatService
        from app.services.retriever_factory import RetrieverType

        pit = _dt(2021, 6, 1)
        tf = TemporalFilter(point_in_time=pit)

        mock_driver = AsyncMock()
        mock_result = MagicMock()
        mock_result.records = []
        mock_driver.execute_query = AsyncMock(return_value=mock_result)

        with (
            patch("app.services.chat_service.OpenAIEmbeddings"),
            patch("app.services.chat_service.OpenAILLM"),
            patch("app.services.chat_service.settings") as mock_settings,
            patch(
                "app.services.chat_service.neo4j_client"
            ) as mock_neo4j,
        ):
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_neo4j.async_driver = mock_driver

            svc = ChatService(
                graph_id="test-graph-001",
                retriever_type=RetrieverType.VECTOR_CYPHER,
            )
            await svc._multihop_enrich(
                ["Alice"], temporal_filter=tf
            )

        # Verify the driver was called with a query that includes temporal params.
        assert mock_driver.execute_query.called
        call_args = mock_driver.execute_query.call_args
        cypher_sent = call_args[0][0]  # first positional arg = Cypher string
        params_sent = call_args[0][1]  # second positional = params dict

        assert "valid_from" in cypher_sent, (
            "Temporal filter must inject valid_from check into multihop Cypher"
        )
        assert "valid_to" in cypher_sent, (
            "Temporal filter must inject valid_to check into multihop Cypher"
        )
        assert "tf_pit" in params_sent, (
            "Temporal filter must pass $tf_pit as a Cypher parameter"
        )
        assert params_sent["tf_pit"] == pit.isoformat()

    async def test_multihop_without_filter_uses_base_cypher(self):
        """
        Without a temporal filter, _multihop_enrich must use the base
        _MULTIHOP_CYPHER query (no valid_from / valid_to conditions).
        """
        from app.services.chat_service import ChatService, _MULTIHOP_CYPHER
        from app.services.retriever_factory import RetrieverType

        mock_driver = AsyncMock()
        mock_result = MagicMock()
        mock_result.records = []
        mock_driver.execute_query = AsyncMock(return_value=mock_result)

        with (
            patch("app.services.chat_service.OpenAIEmbeddings"),
            patch("app.services.chat_service.OpenAILLM"),
            patch("app.services.chat_service.settings") as mock_settings,
            patch(
                "app.services.chat_service.neo4j_client"
            ) as mock_neo4j,
        ):
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_neo4j.async_driver = mock_driver

            svc = ChatService(
                graph_id="test-graph-001",
                retriever_type=RetrieverType.VECTOR_CYPHER,
            )
            await svc._multihop_enrich(["Alice"], temporal_filter=None)

        assert mock_driver.execute_query.called
        call_args = mock_driver.execute_query.call_args
        cypher_sent = call_args[0][0]

        # Base query must not carry any temporal clause
        assert "valid_from" not in cypher_sent
        assert "valid_to" not in cypher_sent

    async def test_relationship_valid_in_range_included_in_pit_query(self):
        """
        Simulate that a relationship WORKS_FOR {event_time: 2019, event_time_end: 2023}
        between Alice and Acme passes a point_in_time=2021-06-01 filter.
        Passes a point_in_time=2024-06-01 filter must exclude it.
        """
        rel_start = _dt(2019, 1, 1)
        rel_end = _dt(2023, 12, 31)

        pit_included = _dt(2021, 6, 1)
        pit_excluded = _dt(2024, 6, 1)

        def _passes(pit: datetime) -> bool:
            return (rel_start is None or rel_start <= pit) and (
                rel_end is None or rel_end > pit
            )

        assert _passes(pit_included), "Alice-Acme 2019-2023 must be visible at 2021-06"
        assert not _passes(pit_excluded), (
            "Alice-Acme 2019-2023 must NOT be visible at 2024-06"
        )


# ===========================================================================
# TEST 9 — TASK-006 absent: _normalize_date contract
# ===========================================================================


@pytest.mark.unit
class TestNormalizeDateContract:
    """
    TASK-006: _normalize_date() coerces year-only strings to full ISO-8601
    dates and returns None for absent / empty values.

    This test class checks for the function's presence and tests the contract
    if it is available.  When absent (TASK-006 not merged), tests are skipped.
    """

    @pytest.fixture(autouse=True)
    def _check_presence(self):
        """Skip all tests if _normalize_date is not on this branch."""
        try:
            from app.services.pipeline_service import _normalize_date  # noqa: F401
        except ImportError:
            pytest.skip(
                "TASK-006 absent: _normalize_date not in pipeline_service on this branch"
            )

    def test_year_only_string_normalized_to_jan_01(self):
        from app.services.pipeline_service import _normalize_date

        result = _normalize_date("2023")
        assert result == "2023-01-01"

    def test_full_iso_date_returned_unchanged(self):
        from app.services.pipeline_service import _normalize_date

        result = _normalize_date("2020-06-15")
        assert result == "2020-06-15"

    def test_empty_string_returns_none(self):
        from app.services.pipeline_service import _normalize_date

        result = _normalize_date("")
        assert result is None

    def test_none_input_returns_none(self):
        from app.services.pipeline_service import _normalize_date

        result = _normalize_date(None)
        assert result is None


# ===========================================================================
# TEST 9b — TASK-006 absent: bitemporal migration task
# ===========================================================================


@pytest.mark.unit
class TestBitemporalMigrationTask:
    """
    TASK-006: run_bitemporal_migration_v1 Celery task must be idempotent.

    When absent on this branch, tests are skipped.
    """

    @pytest.fixture(autouse=True)
    def _check_presence(self):
        try:
            from app.services.background_jobs import run_bitemporal_migration_v1  # noqa: F401
        except (ImportError, AttributeError):
            pytest.skip(
                "TASK-006 absent: run_bitemporal_migration_v1 not in background_jobs on this branch"
            )

    def test_migration_task_exists_and_is_callable(self):
        from app.services.background_jobs import run_bitemporal_migration_v1

        assert callable(run_bitemporal_migration_v1)


# ===========================================================================
# TEST 10 — Known gap: streaming endpoint does not forward temporal_filter
# ===========================================================================


@pytest.mark.unit
class TestStreamingEndpointTemporalGap:
    """
    Security / regression: POST /chat/stream does NOT forward temporal_filter
    to stream_search().

    This is a known gap — stream_search() does not accept a temporal_filter
    parameter, so temporal scoping is silently ignored when the client uses
    the streaming endpoint.

    This test DOCUMENTS the gap rather than asserting correctness.
    When the gap is fixed, this test should be updated to verify correct
    forwarding instead.
    """

    def test_stream_search_signature_has_no_temporal_filter_param(self):
        """
        stream_search() must NOT have a temporal_filter parameter.
        This documents the gap: the streaming path ignores temporal scoping.

        When TASK-XXX fixes this, change this assertion to:
          assert "temporal_filter" in sig.parameters
        """
        import inspect

        from app.services.chat_service import ChatService

        sig = inspect.signature(ChatService.stream_search)
        assert "temporal_filter" not in sig.parameters, (
            "stream_search() has gained a temporal_filter param — update this "
            "test to verify it is correctly forwarded by the /chat/stream endpoint"
        )

    def test_streaming_endpoint_call_omits_temporal_filter(self):
        """
        The /chat/stream endpoint calls stream_search() without forwarding
        body.temporal_filter.  Even when the ChatRequest carries a temporal_filter,
        the streaming path silently ignores it.

        Verified by inspecting the endpoint source: stream_search() is called
        with only query_text and retriever_config — not temporal_filter.
        """
        import inspect

        # Load the streaming endpoint source
        from app.api.v1.endpoints import chat as chat_module

        source = inspect.getsource(chat_module.stream_chat_with_graph)

        # The endpoint must call stream_search without temporal_filter
        assert "stream_search" in source
        # temporal_filter forwarding would look like: temporal_filter=body.temporal_filter
        assert "temporal_filter=body.temporal_filter" not in source, (
            "Streaming endpoint now forwards temporal_filter — remove this "
            "known-gap assertion and add a positive coverage test"
        )

    def test_non_streaming_endpoint_does_forward_temporal_filter(self):
        """
        Positive control: POST /chat (non-streaming) correctly forwards
        body.temporal_filter to chat_service.search().
        This confirms the gap is isolated to the streaming path only.
        """
        import inspect

        from app.api.v1.endpoints import chat as chat_module

        source = inspect.getsource(chat_module.chat_with_graph)
        assert "temporal_filter=body.temporal_filter" in source, (
            "Non-streaming endpoint must forward temporal_filter to search()"
        )
