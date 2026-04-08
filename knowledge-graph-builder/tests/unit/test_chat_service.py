"""
Unit tests for ChatService.

Tests graph-grounded chat with hallucination prevention, source citation,
no-data handling, retriever selection, entity detection, and streaming —
all external deps mocked.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.chat_service import (
    ChatService,
    GroundedSearchResult,
    STRICT_GROUNDING_PROMPT,
    _INSUFFICIENT_PREFIX,
)
from app.services.retriever_factory import RetrieverType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_retriever_item(content: str = "Test content", score: float = 0.9, **metadata):
    item = MagicMock()
    item.content = content
    item.score = score
    item.metadata = {"id": "node-1", "labels": ["Entity"], **metadata}
    return item


def _make_rag_result(answer: str, items=None):
    result = MagicMock()
    result.answer = answer
    retriever_result = MagicMock()
    retriever_result.items = items or []
    result.retriever_result = retriever_result
    return result


# ---------------------------------------------------------------------------
# Tests: ChatService initialisation
# ---------------------------------------------------------------------------

class TestChatServiceInit:
    @pytest.mark.unit
    @pytest.mark.chat
    def test_init_vector_retriever_type(self):
        with patch("app.services.chat_service.OpenAIEmbeddings"), \
             patch("app.services.chat_service.OpenAILLM"), \
             patch("app.services.chat_service.settings") as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            svc = ChatService(graph_id="g1", retriever_type=RetrieverType.VECTOR)
            assert svc.graph_id == "g1"
            assert svc.retriever_type == RetrieverType.VECTOR
            assert svc.retriever is None
            assert svc.rag is None

    @pytest.mark.unit
    @pytest.mark.chat
    def test_init_default_retriever_is_vector_cypher(self):
        with patch("app.services.chat_service.OpenAIEmbeddings"), \
             patch("app.services.chat_service.OpenAILLM"), \
             patch("app.services.chat_service.settings") as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            svc = ChatService(graph_id="g1")
            assert svc.retriever_type == RetrieverType.VECTOR_CYPHER

    @pytest.mark.unit
    @pytest.mark.chat
    def test_init_unsupported_retriever_raises(self):
        with patch("app.services.chat_service.OpenAIEmbeddings"), \
             patch("app.services.chat_service.OpenAILLM"), \
             patch("app.services.chat_service.settings") as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            # get_default_retriever_config raises KeyError for unknown types
            # before the ValueError guard is reached
            with pytest.raises(KeyError):
                ChatService(graph_id="g1", retriever_type="INVALID")  # type: ignore


# ---------------------------------------------------------------------------
# Tests: ChatService.initialize
# ---------------------------------------------------------------------------

class TestChatServiceInitialize:
    @pytest.mark.unit
    @pytest.mark.chat
    async def test_initialize_sets_rag(self):
        with patch("app.services.chat_service.OpenAIEmbeddings"), \
             patch("app.services.chat_service.OpenAILLM"), \
             patch("app.services.chat_service.settings") as mock_settings, \
             patch("app.services.chat_service.retriever_factory") as mock_factory, \
             patch("app.services.chat_service.GraphRAG") as mock_rag_class:
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_factory.create_retriever = AsyncMock(return_value=MagicMock())
            mock_rag_class.return_value = MagicMock()

            svc = ChatService(graph_id="g1", retriever_type=RetrieverType.VECTOR)
            await svc.initialize()

            assert svc.retriever is not None
            assert svc.rag is not None
            mock_rag_class.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.chat
    async def test_initialize_raises_when_retriever_creation_fails(self):
        with patch("app.services.chat_service.OpenAIEmbeddings"), \
             patch("app.services.chat_service.OpenAILLM"), \
             patch("app.services.chat_service.settings") as mock_settings, \
             patch("app.services.chat_service.retriever_factory") as mock_factory:
            mock_settings.OPENAI_API_KEY = "test-key"
            # Both primary and fallback fail
            mock_factory.create_retriever = AsyncMock(side_effect=Exception("Neo4j down"))

            svc = ChatService(graph_id="g1", retriever_type=RetrieverType.HYBRID)
            with pytest.raises(RuntimeError, match="Failed to create any retriever"):
                await svc.initialize()

    @pytest.mark.unit
    @pytest.mark.chat
    async def test_initialize_falls_back_to_vector_on_error(self):
        call_count = 0

        async def mock_create_retriever(retriever_config, graph_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("hybrid failed")
            return MagicMock()

        with patch("app.services.chat_service.OpenAIEmbeddings"), \
             patch("app.services.chat_service.OpenAILLM"), \
             patch("app.services.chat_service.settings") as mock_settings, \
             patch("app.services.chat_service.retriever_factory") as mock_factory, \
             patch("app.services.chat_service.GraphRAG"):
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_factory.create_retriever = AsyncMock(side_effect=mock_create_retriever)

            svc = ChatService(graph_id="g1", retriever_type=RetrieverType.HYBRID)
            await svc.initialize()

            # Should have fallen back to VECTOR
            assert svc.retriever_type == RetrieverType.VECTOR
            assert call_count == 2

    @pytest.mark.unit
    @pytest.mark.chat
    async def test_initialize_sets_up_fulltext_index_for_hybrid(self):
        with patch("app.services.chat_service.OpenAIEmbeddings"), \
             patch("app.services.chat_service.OpenAILLM"), \
             patch("app.services.chat_service.settings") as mock_settings, \
             patch("app.services.chat_service.retriever_factory") as mock_factory, \
             patch("app.services.chat_service.fulltext_index_manager") as mock_fti, \
             patch("app.services.chat_service.GraphRAG"):
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_factory.create_retriever = AsyncMock(return_value=MagicMock())
            mock_fti.setup_default_indexes = AsyncMock()

            svc = ChatService(graph_id="g1", retriever_type=RetrieverType.HYBRID)
            await svc.initialize()

            mock_fti.setup_default_indexes.assert_called_once_with("g1")


# ---------------------------------------------------------------------------
# Tests: ChatService.search
# ---------------------------------------------------------------------------

class TestChatServiceSearch:
    def _build_service(self):
        with patch("app.services.chat_service.OpenAIEmbeddings"), \
             patch("app.services.chat_service.OpenAILLM"), \
             patch("app.services.chat_service.settings") as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            svc = ChatService(graph_id="g1", retriever_type=RetrieverType.VECTOR)
        return svc

    @pytest.mark.unit
    @pytest.mark.chat
    async def test_search_returns_grounded_result(self):
        svc = self._build_service()
        items = [_make_retriever_item(content="TechNova is a tech company.", score=0.95)]
        mock_rag_result = _make_rag_result("TechNova is a tech company.", items)

        mock_rag = MagicMock()
        mock_rag.search.return_value = mock_rag_result
        svc.rag = mock_rag

        result = await svc.search("Tell me about TechNova")

        assert isinstance(result, GroundedSearchResult)
        assert result.is_grounded is True
        assert result.confidence > 0.0
        assert len(result.sources) == 1
        assert result.retriever_used == RetrieverType.VECTOR.value

    @pytest.mark.unit
    @pytest.mark.chat
    async def test_search_returns_insufficient_data_when_no_items(self):
        svc = self._build_service()
        mock_rag_result = _make_rag_result("Some answer", items=[])
        mock_rag = MagicMock()
        mock_rag.search.return_value = mock_rag_result
        svc.rag = mock_rag

        result = await svc.search("Who is John?")

        assert result.is_grounded is False
        assert result.confidence == 0.0
        assert "does not contain sufficient data" in result.answer
        assert result.sources == []

    @pytest.mark.unit
    @pytest.mark.chat
    async def test_search_detects_insufficient_prefix_from_llm(self):
        svc = self._build_service()
        items = [_make_retriever_item(score=0.8)]
        insufficient_answer = f"{_INSUFFICIENT_PREFIX} No relevant data found"
        mock_rag_result = _make_rag_result(insufficient_answer, items)
        mock_rag = MagicMock()
        mock_rag.search.return_value = mock_rag_result
        svc.rag = mock_rag

        result = await svc.search("Unknown question")

        assert result.is_grounded is False
        assert "does not contain sufficient data" in result.answer
        # Confidence should be reduced
        assert result.confidence < 0.5

    @pytest.mark.unit
    @pytest.mark.chat
    async def test_search_uses_strict_grounding_prompt(self):
        svc = self._build_service()
        items = [_make_retriever_item(score=0.85)]
        mock_rag_result = _make_rag_result("Some grounded answer", items)
        mock_rag = MagicMock()
        mock_rag.search.return_value = mock_rag_result
        svc.rag = mock_rag

        await svc.search("A question")

        call_kwargs = mock_rag.search.call_args[1]
        assert call_kwargs.get("prompt_template") == STRICT_GROUNDING_PROMPT

    @pytest.mark.unit
    @pytest.mark.chat
    async def test_search_always_requests_context(self):
        """search() must always pass return_context=True internally to inspect items."""
        svc = self._build_service()
        items = [_make_retriever_item(score=0.85)]
        mock_rag_result = _make_rag_result("Answer", items)
        mock_rag = MagicMock()
        mock_rag.search.return_value = mock_rag_result
        svc.rag = mock_rag

        await svc.search("Q", return_context=False)

        call_kwargs = mock_rag.search.call_args[1]
        assert call_kwargs.get("return_context") is True

    @pytest.mark.unit
    @pytest.mark.chat
    async def test_search_returns_context_only_when_requested(self):
        svc = self._build_service()
        items = [_make_retriever_item(score=0.9)]
        mock_rag_result = _make_rag_result("Answer", items)
        mock_rag = MagicMock()
        mock_rag.search.return_value = mock_rag_result
        svc.rag = mock_rag

        result_with = await svc.search("Q", return_context=True)
        assert result_with.retriever_result is not None

        result_without = await svc.search("Q", return_context=False)
        assert result_without.retriever_result is None

    @pytest.mark.unit
    @pytest.mark.chat
    async def test_search_initializes_rag_if_not_set(self):
        """If rag is None, search() should call initialize() first."""
        svc = self._build_service()
        svc.rag = None

        items = [_make_retriever_item(score=0.9)]
        mock_rag_result = _make_rag_result("Answer", items)
        mock_rag = MagicMock()
        mock_rag.search.return_value = mock_rag_result

        async def mock_initialize():
            svc.rag = mock_rag

        svc.initialize = AsyncMock(side_effect=mock_initialize)

        result = await svc.search("Q")

        svc.initialize.assert_called_once()
        assert result.answer == "Answer"

    @pytest.mark.unit
    @pytest.mark.chat
    async def test_search_handles_exception_gracefully(self):
        svc = self._build_service()
        mock_rag = MagicMock()
        mock_rag.search.side_effect = Exception("Neo4j connection lost")
        svc.rag = mock_rag

        result = await svc.search("Q")

        assert result.is_grounded is False
        assert result.confidence == 0.0
        assert "error" in result.answer.lower()

    @pytest.mark.unit
    @pytest.mark.chat
    async def test_search_sorts_items_by_score_descending(self):
        """Items must be sorted highest-score-first before extraction."""
        svc = self._build_service()
        low = _make_retriever_item(score=0.3)
        high = _make_retriever_item(score=0.9)
        items = [low, high]  # intentionally unsorted
        mock_rag_result = _make_rag_result("Answer", items)
        mock_rag = MagicMock()
        mock_rag.search.return_value = mock_rag_result
        svc.rag = mock_rag

        result = await svc.search("Q")

        # Confidence is mean of top-3; with two items [0.9, 0.3], should be 0.6
        assert abs(result.confidence - 0.6) < 0.01


# ---------------------------------------------------------------------------
# Tests: ChatService._extract_sources
# ---------------------------------------------------------------------------

class TestExtractSources:
    @pytest.mark.unit
    @pytest.mark.chat
    def test_extract_sources_happy_path(self):
        item = MagicMock()
        item.content = "Some content about entity X"
        item.score = 0.88
        item.metadata = {"id": "n1", "labels": ["Person"], "name": "Alice"}

        sources = ChatService._extract_sources([item])

        assert len(sources) == 1
        s = sources[0]
        assert s["node_id"] == "n1"
        assert s["node_labels"] == ["Person"]
        assert "Alice" in str(s["properties"])
        # id/labels/score/embedding must NOT appear in properties
        assert "id" not in s["properties"]
        assert "labels" not in s["properties"]

    @pytest.mark.unit
    @pytest.mark.chat
    def test_extract_sources_truncates_content_at_500(self):
        item = MagicMock()
        item.content = "x" * 600
        item.score = 0.5
        item.metadata = {}

        sources = ChatService._extract_sources([item])
        assert len(sources[0]["content"]) == 500

    @pytest.mark.unit
    @pytest.mark.chat
    def test_extract_sources_empty_list(self):
        assert ChatService._extract_sources([]) == []

    @pytest.mark.unit
    @pytest.mark.chat
    def test_extract_sources_uses_elementId_fallback(self):
        item = MagicMock()
        item.content = "content"
        item.score = 0.7
        item.metadata = {"elementId": "elem-99"}

        sources = ChatService._extract_sources([item])
        assert sources[0]["node_id"] == "elem-99"

    @pytest.mark.unit
    @pytest.mark.chat
    def test_extract_sources_excludes_embedding_from_properties(self):
        item = MagicMock()
        item.content = "content"
        item.score = 0.7
        item.metadata = {"id": "n1", "embedding": [0.1, 0.2], "name": "Bob"}

        sources = ChatService._extract_sources([item])
        assert "embedding" not in sources[0]["properties"]
        assert "name" in sources[0]["properties"]


# ---------------------------------------------------------------------------
# Tests: ChatService._calculate_confidence
# ---------------------------------------------------------------------------

class TestCalculateConfidence:
    @pytest.mark.unit
    @pytest.mark.chat
    def test_confidence_mean_of_top_3(self):
        items = [
            _make_retriever_item(score=0.9),
            _make_retriever_item(score=0.6),
            _make_retriever_item(score=0.3),
            _make_retriever_item(score=0.1),  # 4th — should be ignored
        ]
        confidence = ChatService._calculate_confidence(items)
        expected = (0.9 + 0.6 + 0.3) / 3
        assert abs(confidence - expected) < 0.001

    @pytest.mark.unit
    @pytest.mark.chat
    def test_confidence_clamps_to_1(self):
        items = [_make_retriever_item(score=1.5)]
        assert ChatService._calculate_confidence(items) == 1.0

    @pytest.mark.unit
    @pytest.mark.chat
    def test_confidence_clamps_to_0(self):
        items = [_make_retriever_item(score=-0.5)]
        assert ChatService._calculate_confidence(items) == 0.0

    @pytest.mark.unit
    @pytest.mark.chat
    def test_confidence_fallback_when_no_scores(self):
        item = MagicMock()
        item.score = None
        item.metadata = {}
        confidence = ChatService._calculate_confidence([item])
        assert confidence == 0.5

    @pytest.mark.unit
    @pytest.mark.chat
    def test_confidence_empty_list(self):
        # No items means no scores → fallback 0.5 (but we guard for this in search)
        confidence = ChatService._calculate_confidence([])
        assert confidence == 0.5

    @pytest.mark.unit
    @pytest.mark.chat
    def test_confidence_skips_non_numeric_scores(self):
        item = MagicMock()
        item.score = "not-a-number"
        item.metadata = {}
        # Should fall back gracefully
        confidence = ChatService._calculate_confidence([item])
        assert confidence == 0.5


# ---------------------------------------------------------------------------
# Tests: Multi-tenant isolation
# ---------------------------------------------------------------------------

class TestChatServiceMultiTenantIsolation:
    @pytest.mark.unit
    @pytest.mark.chat
    async def test_graph_id_passed_to_retriever_factory(self):
        """Retriever factory must always receive the graph_id for tenant isolation."""
        with patch("app.services.chat_service.OpenAIEmbeddings"), \
             patch("app.services.chat_service.OpenAILLM"), \
             patch("app.services.chat_service.settings") as mock_settings, \
             patch("app.services.chat_service.retriever_factory") as mock_factory, \
             patch("app.services.chat_service.GraphRAG"):
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_factory.create_retriever = AsyncMock(return_value=MagicMock())

            svc = ChatService(graph_id="tenant-xyz", retriever_type=RetrieverType.VECTOR)
            await svc.initialize()

            call_kwargs = mock_factory.create_retriever.call_args[1]
            assert call_kwargs.get("graph_id") == "tenant-xyz"

    @pytest.mark.unit
    @pytest.mark.chat
    async def test_different_graph_ids_use_separate_services(self):
        """Two ChatService instances with different graph IDs are fully independent."""
        with patch("app.services.chat_service.OpenAIEmbeddings"), \
             patch("app.services.chat_service.OpenAILLM"), \
             patch("app.services.chat_service.settings") as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            svc1 = ChatService(graph_id="tenant-A", retriever_type=RetrieverType.VECTOR)
            svc2 = ChatService(graph_id="tenant-B", retriever_type=RetrieverType.VECTOR)

        assert svc1.graph_id != svc2.graph_id
        assert svc1.rag is None
        assert svc2.rag is None


# ---------------------------------------------------------------------------
# Tests: ChatService.auto_select_retriever_type
# ---------------------------------------------------------------------------

class TestAutoSelectRetrieverType:
    @pytest.mark.unit
    @pytest.mark.chat
    def test_cypher_keywords_return_text2cypher(self):
        assert ChatService.auto_select_retriever_type("MATCH (n) RETURN n") == RetrieverType.TEXT2CYPHER
        assert ChatService.auto_select_retriever_type("give me a cypher query") == RetrieverType.TEXT2CYPHER
        assert ChatService.auto_select_retriever_type("path between A and B") == RetrieverType.TEXT2CYPHER

    @pytest.mark.unit
    @pytest.mark.chat
    def test_analytic_keywords_return_hybrid(self):
        assert ChatService.auto_select_retriever_type("list all companies") == RetrieverType.HYBRID
        assert ChatService.auto_select_retriever_type("how many employees are there?") == RetrieverType.HYBRID
        assert ChatService.auto_select_retriever_type("find all partnerships") == RetrieverType.HYBRID

    @pytest.mark.unit
    @pytest.mark.chat
    def test_relationship_keywords_return_vector_cypher(self):
        assert ChatService.auto_select_retriever_type("who is related to TechNova?") == RetrieverType.VECTOR_CYPHER
        assert ChatService.auto_select_retriever_type("show partnerships between companies") == RetrieverType.VECTOR_CYPHER

    @pytest.mark.unit
    @pytest.mark.chat
    def test_default_returns_vector_cypher(self):
        assert ChatService.auto_select_retriever_type("Tell me about TechNova") == RetrieverType.VECTOR_CYPHER
        assert ChatService.auto_select_retriever_type("What is the company strategy?") == RetrieverType.VECTOR_CYPHER

    @pytest.mark.unit
    @pytest.mark.chat
    def test_cypher_takes_priority_over_analytic(self):
        # "list all" (analytic) + "cypher" keyword — cypher wins
        result = ChatService.auto_select_retriever_type("list all nodes using cypher query")
        assert result == RetrieverType.TEXT2CYPHER


# ---------------------------------------------------------------------------
# Tests: ChatService.detect_entity_candidates
# ---------------------------------------------------------------------------

class TestDetectEntityCandidates:
    @pytest.mark.unit
    @pytest.mark.chat
    def test_extracts_capitalised_single_token(self):
        candidates = ChatService.detect_entity_candidates("Tell me about TechNova")
        assert "TechNova" in candidates

    @pytest.mark.unit
    @pytest.mark.chat
    def test_merges_consecutive_capital_tokens(self):
        candidates = ChatService.detect_entity_candidates("What does TechNova Corp do?")
        assert "TechNova Corp" in candidates

    @pytest.mark.unit
    @pytest.mark.chat
    def test_ignores_stopwords(self):
        candidates = ChatService.detect_entity_candidates("What is the company?")
        # "What" is in stopwords and has length ≤ 4; should produce nothing
        assert candidates == []

    @pytest.mark.unit
    @pytest.mark.chat
    def test_ignores_short_tokens(self):
        candidates = ChatService.detect_entity_candidates("Is AI bad?")
        # "AI" has length 2 — filtered out
        assert "AI" not in candidates

    @pytest.mark.unit
    @pytest.mark.chat
    def test_deduplicates_candidates(self):
        candidates = ChatService.detect_entity_candidates("TechNova Corp and TechNova Corp again")
        assert candidates.count("TechNova Corp") == 1

    @pytest.mark.unit
    @pytest.mark.chat
    def test_empty_query_returns_empty(self):
        assert ChatService.detect_entity_candidates("") == []

    @pytest.mark.unit
    @pytest.mark.chat
    def test_multiple_entities(self):
        candidates = ChatService.detect_entity_candidates(
            "What is the relationship between Acme Inc and GlobalBank Corp?"
        )
        assert any("Acme" in c for c in candidates)
        assert any("GlobalBank" in c for c in candidates)


# ---------------------------------------------------------------------------
# Tests: ChatService.stream_search
# ---------------------------------------------------------------------------

class TestStreamSearch:
    def _build_service(self):
        with patch("app.services.chat_service.OpenAIEmbeddings"), \
             patch("app.services.chat_service.OpenAILLM"), \
             patch("app.services.chat_service.settings") as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            svc = ChatService(graph_id="g1", retriever_type=RetrieverType.VECTOR)
        return svc

    @pytest.mark.unit
    @pytest.mark.chat
    async def test_stream_yields_sources_then_answer_then_done(self):
        svc = self._build_service()
        items = [_make_retriever_item(content="Node content", score=0.9)]
        mock_rag_result = _make_rag_result("This is the answer.", items)
        mock_rag = MagicMock()
        mock_rag.search.return_value = mock_rag_result
        svc.rag = mock_rag

        chunks = [c async for c in svc.stream_search("Q")]

        types = []
        for chunk in chunks:
            data = chunk.removeprefix("data: ").strip()
            types.append(json.loads(data)["type"])

        assert types[0] == "source"
        assert "answer_chunk" in types
        assert types[-1] == "done"

    @pytest.mark.unit
    @pytest.mark.chat
    async def test_stream_no_context_returns_no_data_answer(self):
        svc = self._build_service()
        mock_rag_result = _make_rag_result("Some answer", items=[])
        mock_rag = MagicMock()
        mock_rag.search.return_value = mock_rag_result
        svc.rag = mock_rag

        chunks = [c async for c in svc.stream_search("Q")]

        answer_chunks = [
            json.loads(c.removeprefix("data: ").strip())
            for c in chunks
            if '"answer_chunk"' in c
        ]
        assert any("does not contain sufficient data" in ac["text"] for ac in answer_chunks)

        done_events = [
            json.loads(c.removeprefix("data: ").strip())
            for c in chunks
            if '"done"' in c
        ]
        assert done_events[-1]["is_grounded"] is False
        assert done_events[-1]["confidence"] == 0.0

    @pytest.mark.unit
    @pytest.mark.chat
    async def test_stream_done_event_has_retriever_used(self):
        svc = self._build_service()
        items = [_make_retriever_item(score=0.8)]
        mock_rag_result = _make_rag_result("Answer text here.", items)
        mock_rag = MagicMock()
        mock_rag.search.return_value = mock_rag_result
        svc.rag = mock_rag

        chunks = [c async for c in svc.stream_search("Q")]
        done = json.loads(chunks[-1].removeprefix("data: ").strip())

        assert "retriever_used" in done
        assert done["retriever_used"] == RetrieverType.VECTOR.value

    @pytest.mark.unit
    @pytest.mark.chat
    async def test_stream_uses_strict_grounding_prompt(self):
        svc = self._build_service()
        items = [_make_retriever_item(score=0.8)]
        mock_rag_result = _make_rag_result("Answer", items)
        mock_rag = MagicMock()
        mock_rag.search.return_value = mock_rag_result
        svc.rag = mock_rag

        _ = [c async for c in svc.stream_search("Q")]

        call_kwargs = mock_rag.search.call_args[1]
        assert call_kwargs.get("prompt_template") == STRICT_GROUNDING_PROMPT

    @pytest.mark.unit
    @pytest.mark.chat
    async def test_stream_emits_error_event_on_exception(self):
        svc = self._build_service()
        mock_rag = MagicMock()
        mock_rag.search.side_effect = Exception("stream failure")
        svc.rag = mock_rag

        chunks = [c async for c in svc.stream_search("Q")]

        assert len(chunks) == 1
        event = json.loads(chunks[0].removeprefix("data: ").strip())
        assert event["type"] == "error"
        assert "stream failure" in event["message"]


# ---------------------------------------------------------------------------
# Tests: _multihop_enrich temporal parameter safety — ORA-138
# ---------------------------------------------------------------------------

class TestMultihopTemporalParams:
    """
    ORA-138: _multihop_enrich must pass temporal values as Cypher parameters,
    not as f-string-interpolated datetime literals.
    """

    def _build_service(self):
        return ChatService(graph_id="g-test")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_point_in_time_uses_parameter_not_fstring(self):
        """$tf_pit must appear as a query parameter, not baked into the Cypher string."""
        from datetime import datetime, timezone
        from app.schemas.graph_schemas import TemporalFilter

        svc = self._build_service()
        pit = datetime(2015, 6, 1, tzinfo=timezone.utc)
        tf = TemporalFilter(point_in_time=pit)

        mock_result = MagicMock()
        mock_result.records = []
        mock_driver = AsyncMock()
        mock_driver.execute_query = AsyncMock(return_value=mock_result)
        mock_client = MagicMock()
        mock_client.async_driver = mock_driver

        with patch("app.services.chat_service.neo4j_client", mock_client):
            await svc._multihop_enrich(["Alice"], temporal_filter=tf)

        assert mock_driver.execute_query.called
        call_kwargs = mock_driver.execute_query.call_args
        query = call_kwargs[0][0]
        params = call_kwargs[0][1] if len(call_kwargs[0]) > 1 else call_kwargs[1]

        # The ISO string must NOT be hard-coded into the query
        assert pit.isoformat() not in query, (
            "point_in_time value must be passed as $tf_pit parameter, "
            "not interpolated into the Cypher string"
        )
        # The parameter must be present in the params dict
        assert "tf_pit" in params, (
            "$tf_pit must be in the query parameters dict"
        )
        assert params["tf_pit"] == pit.isoformat()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_current_only_adds_valid_to_null_clause(self):
        """current_only filter must add valid_to IS NULL to the Cypher."""
        from app.schemas.graph_schemas import TemporalFilter

        svc = self._build_service()
        tf = TemporalFilter(current_only=True)

        mock_result = MagicMock()
        mock_result.records = []
        mock_driver = AsyncMock()
        mock_driver.execute_query = AsyncMock(return_value=mock_result)
        mock_client = MagicMock()
        mock_client.async_driver = mock_driver

        with patch("app.services.chat_service.neo4j_client", mock_client):
            await svc._multihop_enrich(["Alice"], temporal_filter=tf)

        assert mock_driver.execute_query.called
        call_kwargs = mock_driver.execute_query.call_args
        query = call_kwargs[0][0]
        assert "r1.valid_to IS NULL" in query

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_temporal_filter_uses_default_cypher(self):
        """Without temporal_filter, the static _MULTIHOP_CYPHER must be used."""
        from app.services.chat_service import _MULTIHOP_CYPHER

        svc = self._build_service()

        mock_result = MagicMock()
        mock_result.records = []
        mock_driver = AsyncMock()
        mock_driver.execute_query = AsyncMock(return_value=mock_result)
        mock_client = MagicMock()
        mock_client.async_driver = mock_driver

        with patch("app.services.chat_service.neo4j_client", mock_client):
            await svc._multihop_enrich(["Alice"], temporal_filter=None)

        assert mock_driver.execute_query.called
        call_kwargs = mock_driver.execute_query.call_args
        query = call_kwargs[0][0]
        assert query == _MULTIHOP_CYPHER
