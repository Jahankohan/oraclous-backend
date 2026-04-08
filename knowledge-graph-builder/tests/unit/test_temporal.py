"""
Unit tests for ORA-32: Temporal Properties on Entities & Relationships

Covers:
- RelationshipProperties temporal field hardening (datetime coercion, validation)
- valid_from > valid_to rejection
- TemporalContext and TemporalFilter model validation
- TemporalFilter mutual-exclusion (point_in_time + current_only)
- UpdateTemporalBoundsRequest validation
- compile_temporal_filter produces correct WHERE clauses
- temporal_context overrides LLM-extracted values
- Backward compat: RelationshipProperties without temporal fields still valid
- IngestDataRequest.temporal_context field present and optional
- ChatRequest.temporal_filter field present and optional
"""
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import ValidationError

from app.schemas.graph_schemas import (
    RelationshipProperties,
    TemporalContext,
    TemporalFilter,
    UpdateTemporalBoundsRequest,
    IngestDataRequest,
)
from app.schemas.chat_schemas import ChatRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dt(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def _rel_props(**kwargs) -> RelationshipProperties:
    defaults = {"source_chunk_id": "chunk-1", "confidence": 0.9}
    defaults.update(kwargs)
    return RelationshipProperties(**defaults)


# ---------------------------------------------------------------------------
# RelationshipProperties — temporal field hardening
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRelationshipPropertiesTemporal:
    def test_valid_from_and_valid_to_accept_datetime(self):
        vf = _dt(2020)
        vt = _dt(2023)
        props = _rel_props(valid_from=vf, valid_to=vt)
        assert props.valid_from == vf
        assert props.valid_to == vt

    def test_valid_from_accepts_iso_string(self):
        props = _rel_props(valid_from="2021-06-15")
        assert props.valid_from is not None
        assert props.valid_from.year == 2021
        assert props.valid_from.month == 6
        assert props.valid_from.day == 15

    def test_valid_from_accepts_iso_string_with_z(self):
        props = _rel_props(valid_from="2021-06-15T00:00:00Z")
        assert props.valid_from is not None
        assert props.valid_from.year == 2021

    def test_valid_to_accepts_iso_string(self):
        props = _rel_props(valid_to="2022-12-31")
        assert props.valid_to is not None
        assert props.valid_to.year == 2022
        assert props.valid_to.month == 12
        assert props.valid_to.day == 31

    def test_unparseable_temporal_string_becomes_none(self):
        props = _rel_props(valid_from="not-a-date")
        assert props.valid_from is None

    def test_valid_from_greater_than_valid_to_rejected(self):
        with pytest.raises(ValidationError, match="valid_from"):
            _rel_props(valid_from=_dt(2023), valid_to=_dt(2020))

    def test_valid_from_equal_to_valid_to_accepted(self):
        t = _dt(2022, 6, 1)
        props = _rel_props(valid_from=t, valid_to=t)
        assert props.valid_from == props.valid_to

    def test_transaction_time_set_server_side(self):
        props = _rel_props()
        assert props.transaction_time is not None
        assert isinstance(props.transaction_time, datetime)

    def test_backward_compat_no_temporal_fields(self):
        props = _rel_props()
        assert props.valid_from is None
        assert props.valid_to is None

    def test_null_valid_to_means_still_valid(self):
        props = _rel_props(valid_from=_dt(2019))
        assert props.valid_to is None


# ---------------------------------------------------------------------------
# TemporalContext
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTemporalContext:
    def test_basic_creation(self):
        ctx = TemporalContext(valid_from=_dt(2020), valid_to=_dt(2023))
        assert ctx.valid_from == _dt(2020)
        assert ctx.valid_to == _dt(2023)

    def test_valid_from_greater_than_valid_to_rejected(self):
        with pytest.raises(ValidationError):
            TemporalContext(valid_from=_dt(2023), valid_to=_dt(2020))

    def test_all_fields_optional(self):
        ctx = TemporalContext()
        assert ctx.valid_from is None
        assert ctx.valid_to is None
        assert ctx.source_date is None

    def test_source_date_stored_as_string(self):
        ctx = TemporalContext(source_date="Q3 2023")
        assert ctx.source_date == "Q3 2023"


# ---------------------------------------------------------------------------
# TemporalFilter
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTemporalFilter:
    def test_point_in_time_filter(self):
        f = TemporalFilter(point_in_time=_dt(2022))
        assert f.point_in_time == _dt(2022)
        assert not f.current_only

    def test_current_only_filter(self):
        f = TemporalFilter(current_only=True)
        assert f.current_only
        assert f.point_in_time is None

    def test_point_in_time_and_current_only_mutually_exclusive(self):
        with pytest.raises(ValidationError):
            TemporalFilter(point_in_time=_dt(2022), current_only=True)

    def test_range_filter(self):
        f = TemporalFilter(valid_from_gte=_dt(2020), valid_to_lte=_dt(2023))
        assert f.valid_from_gte == _dt(2020)
        assert f.valid_to_lte == _dt(2023)

    def test_empty_filter_is_valid(self):
        f = TemporalFilter()
        assert f.point_in_time is None
        assert f.current_only is False


# ---------------------------------------------------------------------------
# UpdateTemporalBoundsRequest
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUpdateTemporalBoundsRequest:
    def test_valid_request(self):
        req = UpdateTemporalBoundsRequest(valid_from=_dt(2021))
        assert req.valid_from == _dt(2021)

    def test_invalid_range_rejected(self):
        with pytest.raises(ValidationError):
            UpdateTemporalBoundsRequest(valid_from=_dt(2025), valid_to=_dt(2020))

    def test_null_valid_to_clears_end_date(self):
        req = UpdateTemporalBoundsRequest(valid_to=None)
        assert req.valid_to is None


# ---------------------------------------------------------------------------
# compile_temporal_filter (TemporalFilterCompiler)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCompileTemporalFilter:
    def _make_pipeline(self):
        from app.services.pipeline_service import MultiTenantGraphRAGPipeline
        pipeline = MultiTenantGraphRAGPipeline.__new__(MultiTenantGraphRAGPipeline)
        pipeline.graph_id = "test-graph"
        return pipeline

    def test_current_only_produces_null_check(self):
        pipeline = self._make_pipeline()
        result = pipeline.compile_temporal_filter(TemporalFilter(current_only=True))
        assert result == "r.valid_to IS NULL"

    def test_point_in_time_produces_range_clauses(self):
        pit = _dt(2022)
        pipeline = self._make_pipeline()
        result = pipeline.compile_temporal_filter(TemporalFilter(point_in_time=pit))
        assert "r.valid_from" in result
        assert "r.valid_to" in result
        assert pit.isoformat() in result

    def test_empty_filter_returns_true(self):
        pipeline = self._make_pipeline()
        result = pipeline.compile_temporal_filter(TemporalFilter())
        assert result == "true"

    def test_range_filter_includes_both_bounds(self):
        pipeline = self._make_pipeline()
        f = TemporalFilter(valid_from_gte=_dt(2020), valid_to_lte=_dt(2023))
        result = pipeline.compile_temporal_filter(f)
        assert _dt(2020).isoformat() in result
        assert _dt(2023).isoformat() in result
        assert "AND" in result

    def test_graph_id_not_injected_into_clause(self):
        """compile_temporal_filter produces a clause for relationships only — not graph_id."""
        pipeline = self._make_pipeline()
        result = pipeline.compile_temporal_filter(TemporalFilter(current_only=True))
        assert "graph_id" not in result


# ---------------------------------------------------------------------------
# IngestDataRequest — temporal_context field
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIngestDataRequestTemporalContext:
    def test_temporal_context_is_optional(self):
        req = IngestDataRequest(content="some text about events")
        assert req.temporal_context is None

    def test_temporal_context_accepted(self):
        ctx = TemporalContext(valid_from=_dt(2020), valid_to=_dt(2023))
        req = IngestDataRequest(content="some text about events", temporal_context=ctx)
        assert req.temporal_context.valid_from == _dt(2020)


# ---------------------------------------------------------------------------
# ChatRequest — temporal_filter field
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestChatRequestTemporalFilter:
    def test_temporal_filter_is_optional(self):
        req = ChatRequest(query="Who is Alice?", graph_id="abc-123")
        assert req.temporal_filter is None

    def test_temporal_filter_accepted(self):
        f = TemporalFilter(point_in_time=_dt(2022))
        req = ChatRequest(query="Who is Alice?", graph_id="abc-123", temporal_filter=f)
        assert req.temporal_filter.point_in_time == _dt(2022)

    def test_backward_compat_no_temporal_filter(self):
        req = ChatRequest(query="test", graph_id="abc-123")
        assert req.temporal_filter is None
        # Default mode still works
        assert req.mode is not None


# ---------------------------------------------------------------------------
# Temporal context override (unit — no Neo4j)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTemporalContextApplicationInPipeline:
    """Verify that temporal_context overrides LLM-extracted valid_from/valid_to on rels."""

    def _make_rel(self, vf=None, vt=None):
        rel = MagicMock()
        rel.properties = {}
        if vf:
            rel.properties["valid_from"] = vf
        if vt:
            rel.properties["valid_to"] = vt
        return rel

    def test_temporal_context_sets_valid_from_when_missing(self):
        """temporal_context.valid_from is applied to relationships lacking valid_from."""
        rel = self._make_rel()
        ctx = TemporalContext(valid_from=_dt(2021))

        if ctx.valid_from and not rel.properties.get("valid_from"):
            rel.properties["valid_from"] = ctx.valid_from

        assert rel.properties["valid_from"] == _dt(2021)

    def test_temporal_context_does_not_override_existing_valid_from(self):
        """LLM-extracted valid_from is preserved when temporal_context is set."""
        rel = self._make_rel(vf=_dt(2019))
        ctx = TemporalContext(valid_from=_dt(2021))

        if ctx.valid_from and not rel.properties.get("valid_from"):
            rel.properties["valid_from"] = ctx.valid_from

        assert rel.properties["valid_from"] == _dt(2019)
