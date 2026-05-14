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
            # STORY-4c embedding fields
            "embedded",
            "skipped_existing_embeddings",
            "failed_embeddings",
        ):
            assert k in d


# ── STORY-4c — embedding tests ────────────────────────────────────────────────


class TestEmbedSummaries:
    @pytest.mark.unit
    async def test_writes_to_kind_specific_property(self, fake_openai_client):
        """chunk → cc.summary_embedding; entity → c.embedding. The Cypher
        SET clause must use the registry's ``embedding_property``, not a
        hardcoded name."""
        rows_listing = [
            {"community_id": "cc_1", "summary": "A summary about X."},
        ]
        mock_client = MagicMock()
        # list → count skipped → write
        mock_client.execute_query = AsyncMock(
            side_effect=[rows_listing, [{"cnt": 5}], None]
        )

        with (
            patch("app.services.community_summarizer.neo4j_client", mock_client),
            patch("neo4j_graphrag.embeddings.OpenAIEmbeddings") as MockEmbedder,
        ):
            MockEmbedder.return_value.embed_query = MagicMock(return_value=[0.1] * 3072)
            summ = CommunitySummarizer(openai_client=fake_openai_client)
            report = await summ.embed_summaries("g1", "chunk")

        assert report.embedded == 1
        assert report.skipped_existing == 5
        write_call = mock_client.execute_query.await_args_list[-1]
        cypher = write_call.args[0]
        assert "summary_embedding" in cypher
        params = write_call.args[1]
        assert params["community_id"] == "cc_1"
        assert len(params["embedding"]) == 3072

    @pytest.mark.unit
    async def test_force_rebuild_pulls_all_with_summary(self, fake_openai_client):
        rows_listing = [
            {"community_id": "cc_x", "summary": "y"},
            {"community_id": "cc_z", "summary": "w"},
        ]
        mock_client = MagicMock()
        # No skip-counting call when force=True; one list + two writes
        mock_client.execute_query = AsyncMock(side_effect=[rows_listing, None, None])

        with (
            patch("app.services.community_summarizer.neo4j_client", mock_client),
            patch("neo4j_graphrag.embeddings.OpenAIEmbeddings") as MockEmbedder,
        ):
            MockEmbedder.return_value.embed_query = MagicMock(return_value=[0.0] * 3072)
            summ = CommunitySummarizer(openai_client=fake_openai_client)
            report = await summ.embed_summaries("g1", "chunk", force_rebuild=True)

        assert report.embedded == 2

    @pytest.mark.unit
    async def test_embedding_failure_counted_in_failed(self, fake_openai_client):
        rows_listing = [{"community_id": "cc_y", "summary": "summary text"}]
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(side_effect=[rows_listing, [{"cnt": 0}]])
        with (
            patch("app.services.community_summarizer.neo4j_client", mock_client),
            patch("neo4j_graphrag.embeddings.OpenAIEmbeddings") as MockEmbedder,
        ):
            MockEmbedder.return_value.embed_query = MagicMock(
                side_effect=RuntimeError("boom")
            )
            summ = CommunitySummarizer(openai_client=fake_openai_client)
            report = await summ.embed_summaries("g1", "chunk")

        assert report.failed == 1
        assert report.embedded == 0


class TestEnsureCommunityVectorIndexes:
    @pytest.mark.unit
    async def test_creates_one_index_per_registered_kind(self):
        """For each entry in COMMUNITY_KINDS, exactly one
        CREATE VECTOR INDEX query runs."""
        from app.services.community_summarizer import (
            ensure_community_vector_indexes,
        )

        mock_client = MagicMock()
        mock_client.execute_write_query = AsyncMock(return_value=None)
        with patch("app.services.community_summarizer.neo4j_client", mock_client):
            await ensure_community_vector_indexes()

        # One call per registered kind (entity + chunk = 2)
        assert mock_client.execute_write_query.await_count == 2
        for call in mock_client.execute_write_query.await_args_list:
            cypher = call.args[0]
            assert "CREATE VECTOR INDEX" in cypher
            assert "IF NOT EXISTS" in cypher
            assert "3072" in cypher
            assert "cosine" in cypher.lower()

    @pytest.mark.unit
    async def test_continues_on_individual_index_failure(self):
        """A single index-creation failure shouldn't crash startup."""
        from app.services.community_summarizer import (
            ensure_community_vector_indexes,
        )

        mock_client = MagicMock()
        mock_client.execute_write_query = AsyncMock(
            side_effect=[RuntimeError("first fails"), None]
        )
        with patch("app.services.community_summarizer.neo4j_client", mock_client):
            await ensure_community_vector_indexes()  # should not raise

        assert mock_client.execute_write_query.await_count == 2
