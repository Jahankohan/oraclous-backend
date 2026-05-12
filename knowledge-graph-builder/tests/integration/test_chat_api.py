"""
Integration tests for the Chat API.

Verifies the full request→service→response pipeline against the actual
ChatService implementation (mocking only Neo4j GraphRAG internals).

Correct endpoint: POST /api/v1/api/v1/chat  (main prefix /api/v1 + router prefix /api/v1)
Response schema fields: answer, query, graph_id, success, mode, retriever_type,
                        is_grounded, confidence, sources, context
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_retriever_item(content="Graph node content", score=0.9, node_id="n-1"):
    item = MagicMock()
    item.content = content
    item.score = score
    item.metadata = {"id": node_id, "labels": ["__Entity__"], "name": "TechNova Corp"}
    return item


def _make_rag_result(answer, items=None):
    result = MagicMock()
    result.answer = answer
    retriever_result = MagicMock()
    retriever_result.items = items or []
    result.retriever_result = retriever_result
    return result


def _patch_chat_service(
    answer="TechNova Corp is a tech company.", items=None, init_raises=None
):
    """Context manager that patches ChatService.initialize and rag.search."""
    items = items if items is not None else [_make_retriever_item()]

    class _ctx:
        def __enter__(self):
            self._p_emb = patch("app.services.chat_service.OpenAIEmbeddings")
            self._p_llm = patch("app.services.chat_service.OpenAILLM")
            self._p_cfg = patch("app.services.chat_service.settings")
            self._p_fac = patch("app.services.chat_service.retriever_factory")
            self._p_rag = patch("app.services.chat_service.GraphRAG")

            self._p_emb.start()
            self._p_llm.start()
            mock_cfg = self._p_cfg.start()
            mock_fac = self._p_fac.start()
            mock_rag_cls = self._p_rag.start()

            mock_cfg.OPENAI_API_KEY = "test-key"
            if init_raises:
                mock_fac.create_retriever = AsyncMock(side_effect=init_raises)
            else:
                mock_fac.create_retriever = AsyncMock(return_value=MagicMock())
                rag_instance = MagicMock()
                rag_instance.search.return_value = _make_rag_result(answer, items)
                mock_rag_cls.return_value = rag_instance

            return self

        def __exit__(self, *_):
            self._p_emb.stop()
            self._p_llm.stop()
            self._p_cfg.stop()
            self._p_fac.stop()
            self._p_rag.stop()

    return _ctx()


# ---------------------------------------------------------------------------
# Auth + ownership bypass
# ---------------------------------------------------------------------------

FAKE_USER = {"id": "test-user-id", "email": "test@example.com"}


def _patch_auth_and_ownership(user_id=None):
    """Context manager that mocks auth_service.verify_token and GraphNodeService."""
    uid = user_id or FAKE_USER["id"]

    class _ctx:
        def __enter__(self):
            self._p_auth = patch("app.api.v1.endpoints.chat.auth_service")
            self._p_gs = patch("app.api.v1.endpoints.chat.GraphNodeService")
            mock_auth = self._p_auth.start()
            mock_auth.verify_token = AsyncMock(return_value={"id": uid})
            mock_gs_cls = self._p_gs.start()
            mock_gs_cls.return_value.get_graph.return_value = {"user_id": uid}
            return self

        def __exit__(self, *_):
            self._p_auth.stop()
            self._p_gs.stop()

    return _ctx()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChatAPIIntegration:
    @pytest.mark.integration
    @pytest.mark.api
    async def test_chat_returns_grounded_answer(self, async_client):
        """POST /chat with graph data → answer is grounded, sources returned."""
        with (
            _patch_chat_service(
                answer="TechNova Corp is a leading tech firm.",
                items=[
                    _make_retriever_item(
                        content="TechNova Corp node content", score=0.95
                    )
                ],
            ),
            _patch_auth_and_ownership(),
        ):
            response = await async_client.post(
                "/api/v1/api/v1/chat",
                json={"query": "Tell me about TechNova", "graph_id": "test-graph-123"},
                headers={"Authorization": "Bearer fake-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["is_grounded"] is True
        assert isinstance(data["confidence"], float)
        assert data["confidence"] > 0.0
        assert data["retriever_type"] is not None
        assert "TechNova" in data["answer"]

    @pytest.mark.integration
    @pytest.mark.api
    async def test_chat_returns_no_data_response_for_empty_graph(self, async_client):
        """POST /chat with empty graph → structured no-data response, not a hallucination."""
        with _patch_chat_service(answer="", items=[]), _patch_auth_and_ownership():
            response = await async_client.post(
                "/api/v1/api/v1/chat",
                json={"query": "Who is the CEO?", "graph_id": "empty-graph"},
                headers={"Authorization": "Bearer fake-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["is_grounded"] is False
        assert data["confidence"] == 0.0
        assert "does not contain sufficient data" in data["answer"]
        assert data["sources"] == [] or data["sources"] is None

    @pytest.mark.integration
    @pytest.mark.api
    async def test_chat_sources_list_returned_when_include_sources_true(
        self, async_client
    ):
        """Sources list is populated when include_sources=True (default)."""
        with (
            _patch_chat_service(
                items=[_make_retriever_item(node_id="node-abc", score=0.85)],
            ),
            _patch_auth_and_ownership(),
        ):
            response = await async_client.post(
                "/api/v1/api/v1/chat",
                json={
                    "query": "What is TechNova?",
                    "graph_id": "g1",
                    "include_sources": True,
                },
                headers={"Authorization": "Bearer fake-token"},
            )

        data = response.json()
        assert data["success"] is True
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) == 1
        assert data["sources"][0]["node_id"] == "node-abc"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_chat_sources_omitted_when_include_sources_false(self, async_client):
        with _patch_chat_service(), _patch_auth_and_ownership():
            response = await async_client.post(
                "/api/v1/api/v1/chat",
                json={
                    "query": "What is TechNova?",
                    "graph_id": "g1",
                    "include_sources": False,
                },
                headers={"Authorization": "Bearer fake-token"},
            )

        data = response.json()
        assert data["sources"] is None

    @pytest.mark.integration
    @pytest.mark.api
    async def test_chat_context_returned_when_return_context_true(self, async_client):
        with _patch_chat_service(), _patch_auth_and_ownership():
            response = await async_client.post(
                "/api/v1/api/v1/chat",
                json={
                    "query": "What is TechNova?",
                    "graph_id": "g1",
                    "return_context": True,
                },
                headers={"Authorization": "Bearer fake-token"},
            )

        data = response.json()
        assert data["context"] is not None
        assert data["context"]["retriever_type"] is not None
        assert isinstance(data["context"]["total_results"], int)

    @pytest.mark.integration
    @pytest.mark.api
    async def test_chat_all_modes_accepted(self, async_client):
        """All five chat modes must be accepted without 422."""
        modes = ["simple", "enhanced", "hybrid", "hybrid_plus", "natural"]
        for mode in modes:
            with _patch_chat_service(), _patch_auth_and_ownership():
                response = await async_client.post(
                    "/api/v1/api/v1/chat",
                    json={"query": "Q", "graph_id": "g1", "mode": mode},
                    headers={"Authorization": "Bearer fake-token"},
                )
            assert response.status_code == 200, f"Mode {mode} failed: {response.text}"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_chat_invalid_mode_returns_422(self, async_client):
        response = await async_client.post(
            "/api/v1/api/v1/chat",
            json={"query": "Q", "graph_id": "g1", "mode": "invalid_mode"},
            headers={"Authorization": "Bearer fake-token"},
        )
        assert response.status_code == 422

    @pytest.mark.integration
    @pytest.mark.api
    async def test_chat_missing_graph_id_returns_422(self, async_client):
        response = await async_client.post(
            "/api/v1/api/v1/chat",
            json={"query": "Q"},
            headers={"Authorization": "Bearer fake-token"},
        )
        assert response.status_code == 422

    @pytest.mark.integration
    @pytest.mark.api
    async def test_chat_missing_query_returns_422(self, async_client):
        response = await async_client.post(
            "/api/v1/api/v1/chat",
            json={"graph_id": "g1"},
            headers={"Authorization": "Bearer fake-token"},
        )
        assert response.status_code == 422

    @pytest.mark.integration
    @pytest.mark.api
    async def test_chat_service_failure_returns_500(self, async_client):
        with (
            _patch_chat_service(init_raises=Exception("Neo4j down")),
            _patch_auth_and_ownership(),
        ):
            response = await async_client.post(
                "/api/v1/api/v1/chat",
                json={"query": "Q", "graph_id": "g1"},
                headers={"Authorization": "Bearer fake-token"},
            )
        assert response.status_code == 500

    @pytest.mark.integration
    @pytest.mark.api
    async def test_get_modes_returns_all_five(self, async_client):
        with _patch_auth_and_ownership():
            response = await async_client.get(
                "/api/v1/api/v1/modes",
                headers={"Authorization": "Bearer fake-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "modes" in data
        assert len(data["modes"]) == 5
        mode_values = [m["mode"] for m in data["modes"]]
        for expected in ["simple", "enhanced", "hybrid", "hybrid_plus", "natural"]:
            assert expected in mode_values
        assert data["default_mode"] == "enhanced"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_chat_response_contains_required_fields(self, async_client):
        """All required ChatResponse fields must be present."""
        with _patch_chat_service(), _patch_auth_and_ownership():
            response = await async_client.post(
                "/api/v1/api/v1/chat",
                json={"query": "Q", "graph_id": "g1"},
                headers={"Authorization": "Bearer fake-token"},
            )

        data = response.json()
        for field in [
            "answer",
            "query",
            "graph_id",
            "success",
            "mode",
            "retriever_type",
            "is_grounded",
            "timestamp",
        ]:
            assert field in data, f"Missing required field: {field}"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_chat_stream_endpoint_returns_event_stream(self, async_client):
        """POST /chat/stream must return text/event-stream content type."""
        with _patch_chat_service(), _patch_auth_and_ownership():
            response = await async_client.post(
                "/api/v1/api/v1/chat/stream",
                json={"query": "Q", "graph_id": "g1"},
                headers={"Authorization": "Bearer fake-token"},
            )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

    @pytest.mark.integration
    @pytest.mark.api
    async def test_chat_stream_events_are_valid_sse(self, async_client):
        """Each line from /chat/stream must be valid SSE data: JSON."""
        with (
            _patch_chat_service(
                answer="TechNova is a tech firm.",
                items=[_make_retriever_item()],
            ),
            _patch_auth_and_ownership(),
        ):
            response = await async_client.post(
                "/api/v1/api/v1/chat/stream",
                json={"query": "Tell me about TechNova", "graph_id": "g1"},
                headers={"Authorization": "Bearer fake-token"},
            )

        assert response.status_code == 200
        lines = [ln for ln in response.text.strip().split("\n\n") if ln.strip()]
        event_types = set()
        for line in lines:
            line = line.strip()
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                event_types.add(payload["type"])

        assert "done" in event_types
