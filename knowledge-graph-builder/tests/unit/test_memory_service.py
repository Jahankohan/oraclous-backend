"""
Unit tests for memory_service.py.

All Neo4j calls are mocked — no real database required.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.memory import (
    MemoryCreate,
    MemoryScope,
    MemorySource,
    MemoryType,
    MemoryUpdate,
)
from app.services.memory_service import (
    MemoryService,
    _content_hash,
    compute_importance,
)

# ============================================================
# Helpers
# ============================================================


def _now() -> datetime:
    return datetime.now(UTC)


def _make_service() -> MemoryService:
    return MemoryService()


# ============================================================
# compute_importance
# ============================================================


class TestComputeImportance:
    @pytest.mark.unit
    def test_fresh_memory_returns_base_importance(self):
        now = _now()
        score = compute_importance(
            base_importance=0.8,
            memory_type=MemoryType.SEMANTIC,
            last_accessed_at=now,
            access_count=0,
            now=now,
        )
        assert abs(score - 0.8) < 0.001

    @pytest.mark.unit
    def test_episodic_decays_faster_than_semantic(self):
        now = _now()
        past = now - timedelta(days=30)
        episodic = compute_importance(0.8, MemoryType.EPISODIC, past, 0, now)
        semantic = compute_importance(0.8, MemoryType.SEMANTIC, past, 0, now)
        assert episodic < semantic

    @pytest.mark.unit
    def test_access_boost_increases_score(self):
        now = _now()
        past = now - timedelta(days=10)
        no_access = compute_importance(0.5, MemoryType.SEMANTIC, past, 0, now)
        with_access = compute_importance(0.5, MemoryType.SEMANTIC, past, 50, now)
        assert with_access > no_access

    @pytest.mark.unit
    def test_score_capped_at_one(self):
        now = _now()
        score = compute_importance(1.0, MemoryType.SEMANTIC, now, 1000, now)
        assert score <= 1.0

    @pytest.mark.unit
    def test_score_never_negative(self):
        now = _now()
        far_past = now - timedelta(days=3650)
        score = compute_importance(0.1, MemoryType.EPISODIC, far_past, 0, now)
        assert score >= 0.0

    @pytest.mark.unit
    def test_naive_datetime_handled(self):
        now = _now()
        naive_past = datetime(2025, 1, 1, 0, 0, 0)  # no tzinfo
        score = compute_importance(0.8, MemoryType.SEMANTIC, naive_past, 0, now)
        assert 0.0 <= score <= 1.0


# ============================================================
# _content_hash
# ============================================================


class TestContentHash:
    @pytest.mark.unit
    def test_same_content_same_hash(self):
        assert _content_hash("Hello World") == _content_hash("Hello World")

    @pytest.mark.unit
    def test_normalisation_collapses_whitespace(self):
        assert _content_hash("hello   world") == _content_hash("hello world")

    @pytest.mark.unit
    def test_case_insensitive(self):
        assert _content_hash("HELLO") == _content_hash("hello")

    @pytest.mark.unit
    def test_different_content_different_hash(self):
        assert _content_hash("foo") != _content_hash("bar")


# ============================================================
# MemoryService.store_memory
# ============================================================


class TestStoreMemory:
    @pytest.mark.unit
    async def test_duplicate_returns_existing(self):
        svc = _make_service()
        existing = {"memory_id": "existing-id", "importance_score": 0.9}

        with patch.object(
            svc, "_find_by_content_hash", new=AsyncMock(return_value=existing)
        ):
            req = MemoryCreate(type=MemoryType.SEMANTIC, content="Reza is CEO")
            result = await svc.store_memory("graph-1", req)

        assert result.memory_id == "existing-id"
        assert result.importance_score == 0.9
        assert result.contradictions_detected == []

    @pytest.mark.unit
    async def test_new_memory_creates_node(self):
        svc = _make_service()

        with (
            patch.object(
                svc, "_find_by_content_hash", new=AsyncMock(return_value=None)
            ),
            patch("app.services.memory_service.neo4j_client") as mock_client,
            patch.object(
                svc, "_detect_and_record_contradictions", new=AsyncMock(return_value=[])
            ),
            patch.object(svc, "_link_to_entity", new=AsyncMock(return_value=None)),
        ):
            mock_client.execute_write_query = AsyncMock(return_value=[])

            req = MemoryCreate(
                type=MemoryType.SEMANTIC,
                content="Reza is CEO of DeAgenticAI",
                subject="Reza",
                predicate="IS_CEO_OF",
                object="DeAgenticAI",
                confidence=0.95,
            )
            result = await svc.store_memory("graph-1", req)

        assert result.memory_id  # non-empty UUID
        assert result.importance_score == pytest.approx(
            0.8, abs=0.01
        )  # high confidence

    @pytest.mark.unit
    async def test_user_feedback_gets_max_importance(self):
        svc = _make_service()

        with (
            patch.object(
                svc, "_find_by_content_hash", new=AsyncMock(return_value=None)
            ),
            patch("app.services.memory_service.neo4j_client") as mock_client,
            patch.object(
                svc, "_detect_and_record_contradictions", new=AsyncMock(return_value=[])
            ),
            patch.object(svc, "_link_to_entity", new=AsyncMock(return_value=None)),
        ):
            mock_client.execute_write_query = AsyncMock(return_value=[])

            req = MemoryCreate(
                type=MemoryType.PROCEDURAL,
                content="User prefers bullet points",
                source=MemorySource.USER_FEEDBACK,
            )
            result = await svc.store_memory("graph-1", req)

        assert result.importance_score == pytest.approx(1.0)

    @pytest.mark.unit
    async def test_contradiction_detection_called_for_semantic(self):
        svc = _make_service()
        mock_contradictions = [
            MagicMock(
                conflict_memory_id="old-id", content="Bob is CEO", resolution="new_wins"
            )
        ]

        with (
            patch.object(
                svc, "_find_by_content_hash", new=AsyncMock(return_value=None)
            ),
            patch("app.services.memory_service.neo4j_client") as mock_client,
            patch.object(
                svc,
                "_detect_and_record_contradictions",
                new=AsyncMock(return_value=mock_contradictions),
            ),
            patch.object(svc, "_link_to_entity", new=AsyncMock(return_value=None)),
        ):
            mock_client.execute_write_query = AsyncMock(return_value=[])

            req = MemoryCreate(
                type=MemoryType.SEMANTIC,
                content="Alice is CEO",
                subject="Alice",
                predicate="IS_CEO_OF",
                object="Acme",
            )
            result = await svc.store_memory("graph-1", req)

        assert len(result.contradictions_detected) == 1


# ============================================================
# MemoryService.delete_memory
# ============================================================


class TestDeleteMemory:
    @pytest.mark.unit
    async def test_soft_delete_sets_valid_to(self):
        svc = _make_service()
        with patch("app.services.memory_service.neo4j_client") as mock_client:
            mock_client.execute_write_query = AsyncMock(return_value=[])
            await svc.delete_memory("graph-1", "mem-1", hard=False)
            call_args = mock_client.execute_write_query.call_args[0]
            assert "valid_to" in call_args[0]
            assert "DETACH DELETE" not in call_args[0]

    @pytest.mark.unit
    async def test_hard_delete_detach_deletes(self):
        svc = _make_service()
        with patch("app.services.memory_service.neo4j_client") as mock_client:
            mock_client.execute_write_query = AsyncMock(return_value=[])
            await svc.delete_memory("graph-1", "mem-1", hard=True)
            call_args = mock_client.execute_write_query.call_args[0]
            assert "DETACH DELETE" in call_args[0]


# ============================================================
# MemoryService.update_memory
# ============================================================


class TestUpdateMemory:
    @pytest.mark.unit
    async def test_update_raises_on_missing_memory(self):
        svc = _make_service()
        with patch("app.services.memory_service.neo4j_client") as mock_client:
            mock_client.execute_query = AsyncMock(return_value=[])
            with pytest.raises(ValueError, match="not found"):
                await svc.update_memory(
                    "graph-1", "nonexistent", MemoryUpdate(content="new")
                )

    @pytest.mark.unit
    async def test_update_creates_supersedes_link(self):
        svc = _make_service()
        old_record = {
            "content": "old content",
            "memory_type": "semantic",
            "confidence": 0.8,
            "scope": "agent",
            "agent_id": "",
            "session_id": "",
            "source": "agent",
            "base_importance": 0.8,
        }
        with patch("app.services.memory_service.neo4j_client") as mock_client:
            mock_client.execute_query = AsyncMock(return_value=[old_record])
            mock_client.execute_write_query = AsyncMock(return_value=[])
            result = await svc.update_memory(
                "graph-1", "old-id", MemoryUpdate(content="new content")
            )

        assert result.old_memory_id == "old-id"
        assert result.new_memory_id  # non-empty
        assert result.old_memory_id != result.new_memory_id


# ============================================================
# MemoryService.consolidate
# ============================================================


class TestConsolidate:
    @pytest.mark.unit
    async def test_empty_graph_returns_zero_merged(self):
        svc = _make_service()
        with patch("app.services.memory_service.neo4j_client") as mock_client:
            mock_client.execute_query = AsyncMock(return_value=[])
            result = await svc.consolidate("graph-1")
        assert result["merged"] == 0

    @pytest.mark.unit
    async def test_single_memory_returns_zero_merged(self):
        svc = _make_service()
        with patch("app.services.memory_service.neo4j_client") as mock_client:
            mock_client.execute_query = AsyncMock(
                return_value=[
                    {
                        "memory_id": "m1",
                        "content": "only memory",
                        "importance_score": 0.8,
                        "confidence": 0.9,
                        "base_importance": 0.8,
                    }
                ]
            )
            result = await svc.consolidate("graph-1")
        assert result["merged"] == 0

    @pytest.mark.unit
    async def test_duplicate_memories_merged(self):
        svc = _make_service()
        content = "Reza is CEO"
        dupes = [
            {
                "memory_id": "m1",
                "content": content,
                "importance_score": 0.9,
                "confidence": 0.9,
                "base_importance": 0.9,
            },
            {
                "memory_id": "m2",
                "content": content,
                "importance_score": 0.7,
                "confidence": 0.7,
                "base_importance": 0.7,
            },
        ]
        with patch("app.services.memory_service.neo4j_client") as mock_client:
            mock_client.execute_query = AsyncMock(return_value=dupes)
            mock_client.execute_write_query = AsyncMock(return_value=[])
            result = await svc.consolidate("graph-1")
        assert result["merged"] == 1


# ============================================================
# MemoryRetriever
# ============================================================


class TestMemoryRetriever:
    @pytest.mark.unit
    async def test_search_returns_formatted_results(self):
        from app.schemas.memory import MemorySearchResponse, MemorySearchResult
        from app.schemas.retriever_schemas import MemoryRetrieverConfig
        from app.services.retriever_factory import MemoryRetriever

        config = MemoryRetrieverConfig(query="test", top_k=5)
        retriever = MemoryRetriever(graph_id="graph-1", config=config)

        mock_result = MemorySearchResult(
            memory_id="m1",
            type=MemoryType.SEMANTIC,
            content="Reza is CEO",
            importance_score=0.9,
            relevance_score=0.85,
            confidence=0.95,
            valid_from=None,
            valid_to=None,
            scope=MemoryScope.ORGANIZATION,
        )
        MemorySearchResponse(memories=[mock_result], total=1)

        with patch(
            "app.services.retriever_factory.MemoryRetriever.search",
            new=AsyncMock(
                return_value=[
                    {
                        "content": "Reza is CEO",
                        "score": 0.85,
                        "memory_id": "m1",
                        "type": "semantic",
                        "importance_score": 0.9,
                    }
                ]
            ),
        ):
            results = await retriever.search("CEO query")

        assert len(results) == 1
        assert results[0]["memory_id"] == "m1"
