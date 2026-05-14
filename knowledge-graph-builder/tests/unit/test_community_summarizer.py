"""Unit tests for CommunitySummarizer (STORY-4b).

Verifies the chunk-community summary path:
- singletons (size=1) are skipped without any LLM call
- existing summaries are skipped unless force_rebuild=True
- successful summarisation writes the 3-field shape to Neo4j
- JSON-parse failure produces a partial write (summary only)
- unsupported kinds raise the right exceptions
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.community_kinds import UnknownCommunityKindError
from app.services.community_summarizer import (
    CommunitySummarizer,
    SummarizeReport,
)


def _mock_neo4j_for_chunk_list(rows: list[dict]) -> MagicMock:
    """Mock neo4j_client.execute_query so the first call returns the
    community list, and subsequent calls return values per-side-effect."""
    mock = MagicMock()
    mock.execute_query = AsyncMock(return_value=rows)
    return mock


@pytest.fixture
def fake_openai_client():
    """OpenAI client that returns one canned response per invocation."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


def _canned_llm_json_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps(payload)
    return resp


class TestSupportedKinds:
    @pytest.mark.unit
    def test_chunk_supported(self):
        assert "chunk" in CommunitySummarizer.supported_kinds()

    @pytest.mark.unit
    def test_entity_not_supported_here(self):
        """Entity-Leiden is owned by the Celery detector, not this class."""
        assert "entity" not in CommunitySummarizer.supported_kinds()


class TestSummarizeAllDispatch:
    @pytest.mark.unit
    async def test_unknown_kind_raises(self, fake_openai_client):
        summ = CommunitySummarizer(openai_client=fake_openai_client)
        with pytest.raises(UnknownCommunityKindError):
            await summ.summarize_all("g1", "bogus")

    @pytest.mark.unit
    async def test_unsupported_kind_raises_value_error(self, fake_openai_client):
        """Registered but not supported (entity) raises ValueError."""
        summ = CommunitySummarizer(openai_client=fake_openai_client)
        with pytest.raises(ValueError, match="does not handle"):
            await summ.summarize_all("g1", "entity")


class TestChunkSingletonSkip:
    @pytest.mark.unit
    async def test_singletons_skipped_without_llm_call(self, fake_openai_client):
        """All singletons → 0 LLM calls."""
        rows = [
            {"community_id": f"cc_{i}", "size": 1, "summary": None} for i in range(5)
        ]
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=rows)

        with patch("app.services.community_summarizer.neo4j_client", mock_client):
            summ = CommunitySummarizer(openai_client=fake_openai_client)
            report = await summ.summarize_all("g1", "chunk")

        assert report.total == 5
        assert report.skipped_singleton == 5
        assert report.summarized == 0
        assert report.failed == 0
        # Crucial: no LLM call for singletons
        fake_openai_client.chat.completions.create.assert_not_called()


class TestChunkSummarize:
    @pytest.mark.unit
    async def test_summarizes_real_clusters_via_llm(self, fake_openai_client):
        """Non-singletons get summarised; the 3-field write shape is correct."""
        rows_listing = [
            {"community_id": "cc_real", "size": 5, "summary": None},
        ]
        rows_chunks = [
            {"text": "Chunk one text", "id": "ck-1"},
            {"text": "Chunk two text", "id": "ck-2"},
        ]
        # neo4j_client.execute_query is called for: (1) list communities,
        # (2) sample chunks, (3) MERGE summary back
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(
            side_effect=[rows_listing, rows_chunks, None]
        )

        fake_openai_client.chat.completions.create.return_value = (
            _canned_llm_json_response(
                {
                    "summary": "Cluster about X and Y.",
                    "keywords": ["X", "Y", "Z"],
                    "representative_excerpt": "An excerpt.",
                }
            )
        )

        with patch("app.services.community_summarizer.neo4j_client", mock_client):
            summ = CommunitySummarizer(openai_client=fake_openai_client)
            report = await summ.summarize_all("g1", "chunk")

        assert report.total == 1
        assert report.summarized == 1
        assert report.skipped_singleton == 0
        assert report.failed == 0

        # Verify the WRITE call carried the right fields
        write_call = mock_client.execute_query.await_args_list[-1]
        params = write_call.args[1]
        assert params["community_id"] == "cc_real"
        assert params["graph_id"] == "g1"
        assert params["summary"] == "Cluster about X and Y."
        assert json.loads(params["keywords"]) == ["X", "Y", "Z"]
        assert params["excerpt"] == "An excerpt."
        assert params["member_count"] == 5
        assert params["sampled"] == 2

    @pytest.mark.unit
    async def test_skips_existing_summaries_when_not_force(self, fake_openai_client):
        rows_listing = [
            {
                "community_id": "cc_done",
                "size": 5,
                "summary": "Already summarised",
            },
        ]
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=rows_listing)

        with patch("app.services.community_summarizer.neo4j_client", mock_client):
            summ = CommunitySummarizer(openai_client=fake_openai_client)
            report = await summ.summarize_all("g1", "chunk")

        assert report.skipped_existing == 1
        assert report.summarized == 0
        fake_openai_client.chat.completions.create.assert_not_called()

    @pytest.mark.unit
    async def test_force_rebuild_overwrites_existing(self, fake_openai_client):
        rows_listing = [
            {
                "community_id": "cc_done",
                "size": 5,
                "summary": "Already summarised",
            },
        ]
        rows_chunks = [{"text": "fresh chunk", "id": "ck-1"}]
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(
            side_effect=[rows_listing, rows_chunks, None]
        )
        fake_openai_client.chat.completions.create.return_value = (
            _canned_llm_json_response(
                {
                    "summary": "Rewritten summary",
                    "keywords": ["new"],
                    "representative_excerpt": "fresh",
                }
            )
        )

        with patch("app.services.community_summarizer.neo4j_client", mock_client):
            summ = CommunitySummarizer(openai_client=fake_openai_client)
            report = await summ.summarize_all("g1", "chunk", force_rebuild=True)

        assert report.summarized == 1
        assert report.skipped_existing == 0

    @pytest.mark.unit
    async def test_llm_failure_counted_in_failed(self, fake_openai_client):
        rows_listing = [
            {"community_id": "cc_x", "size": 5, "summary": None},
        ]
        rows_chunks = [{"text": "chunk text", "id": "ck-1"}]
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(side_effect=[rows_listing, rows_chunks])

        # Raise from the LLM call
        fake_openai_client.chat.completions.create.side_effect = RuntimeError("boom")

        with patch("app.services.community_summarizer.neo4j_client", mock_client):
            summ = CommunitySummarizer(openai_client=fake_openai_client)
            report = await summ.summarize_all("g1", "chunk")

        assert report.failed == 1
        assert report.summarized == 0


class TestJsonParseFallback:
    @pytest.mark.unit
    def test_safe_parse_handles_valid_json(self):
        out = CommunitySummarizer._safe_parse_chunk_response(
            json.dumps(
                {
                    "summary": "OK",
                    "keywords": ["a", "b"],
                    "representative_excerpt": "x",
                }
            )
        )
        assert out == {"summary": "OK", "keywords": ["a", "b"], "excerpt": "x"}

    @pytest.mark.unit
    def test_safe_parse_falls_back_to_plain_text(self):
        """Non-JSON output → keep as summary, empty keywords/excerpt."""
        out = CommunitySummarizer._safe_parse_chunk_response(
            "not json at all just prose"
        )
        assert out["summary"] == "not json at all just prose"
        assert out["keywords"] == []
        assert out["excerpt"] == ""

    @pytest.mark.unit
    def test_safe_parse_trims_long_fields(self):
        out = CommunitySummarizer._safe_parse_chunk_response(
            json.dumps(
                {
                    "summary": "x" * 5000,
                    "keywords": ["k" * 200],
                    "representative_excerpt": "y" * 1000,
                }
            )
        )
        assert len(out["summary"]) == 2000
        assert len(out["keywords"][0]) == 120
        assert len(out["excerpt"]) == 600

    @pytest.mark.unit
    def test_safe_parse_drops_non_list_keywords(self):
        out = CommunitySummarizer._safe_parse_chunk_response(
            json.dumps({"summary": "x", "keywords": "not-a-list"})
        )
        assert out["keywords"] == []


class TestReportShape:
    @pytest.mark.unit
    def test_to_dict_includes_all_fields(self):
        r = SummarizeReport(kind="chunk", total=10, summarized=2, skipped_singleton=8)
        d = r.to_dict()
        for k in (
            "kind",
            "total",
            "summarized",
            "skipped_existing",
            "skipped_singleton",
            "failed",
            "per_level",
        ):
            assert k in d
