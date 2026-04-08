"""
Chat Hallucination Tests — Suite 4.

Verifies that the chat endpoint:
  1. Returns factually correct answers grounded in graph data (known facts test)
  2. Does NOT hallucinate answers when the graph has no relevant data
  3. Returns structured no-data response rather than invented information
  4. Cites the correct source nodes (provenance)
  5. Does not fabricate entities not present in the graph
  6. Confidence score reflects retrieval quality (0.0 when no context found)
  7. is_grounded flag is False when no context was retrieved
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRAPH_ID = "hallucination-test-graph-001"
FAKE_USER = {"id": "test-user-hall-001", "email": "qa@example.com"}


# ---------------------------------------------------------------------------
# Helpers (mirrored from test_chat_api.py for self-containment)
# ---------------------------------------------------------------------------


def _retriever_item(
    content: str, score: float = 0.9, node_id: str = "n1", name: str = "Entity"
):
    item = MagicMock()
    item.content = content
    item.score = score
    item.metadata = {"id": node_id, "labels": ["__Entity__"], "name": name}
    return item


def _rag_result(answer: str, items=None):
    result = MagicMock()
    result.answer = answer
    ret = MagicMock()
    ret.items = items if items is not None else []
    result.retriever_result = ret
    return result


class _ChatPatch:
    """Context manager: patches the full chat service stack."""

    def __init__(self, answer: str = "", items=None, retriever_error=None):
        self.answer = answer
        self.items = items if items is not None else []
        self.retriever_error = retriever_error
        self._patches = []

    def __enter__(self):
        targets = [
            "app.services.chat_service.OpenAIEmbeddings",
            "app.services.chat_service.OpenAILLM",
            "app.services.chat_service.settings",
            "app.services.chat_service.retriever_factory",
            "app.services.chat_service.GraphRAG",
        ]
        mocks = [patch(t) for t in targets]
        self._patches = mocks
        started = [p.start() for p in mocks]

        mock_emb, mock_llm, mock_cfg, mock_fac, mock_rag_cls = started
        mock_cfg.OPENAI_API_KEY = "test-key"

        if self.retriever_error:
            mock_fac.create_retriever = AsyncMock(side_effect=self.retriever_error)
        else:
            mock_fac.create_retriever = AsyncMock(return_value=MagicMock())
            rag_instance = MagicMock()
            rag_instance.search.return_value = _rag_result(self.answer, self.items)
            mock_rag_cls.return_value = rag_instance

        return self

    def __exit__(self, *_):
        for p in self._patches:
            p.stop()


class _auth_patch:
    """
    Context manager: overrides get_current_user_id dependency and mocks
    GraphNodeService.get_graph so the ORA-21 ownership check passes.
    Usage: auth = _auth_patch(); ... ; auth.stop()
    """

    def __init__(self, graph_id: str = GRAPH_ID):
        self._graph_id = graph_id
        self._gs_patch = None
        self.start()  # auto-start to match old usage: auth = _auth_patch()

    def start(self):
        uid = FAKE_USER["id"]
        self._auth_svc_patch = patch("app.api.v1.endpoints.chat.auth_service")
        mock_auth = self._auth_svc_patch.start()
        mock_auth.verify_token = AsyncMock(return_value={"id": uid})
        self._gs_patch = patch("app.api.v1.endpoints.chat.GraphNodeService")
        mock_cls = self._gs_patch.start()
        mock_cls.return_value.get_graph.return_value = {
            "user_id": uid,
            "graph_id": self._graph_id,
        }
        return self

    def stop(self):
        if hasattr(self, "_auth_svc_patch") and self._auth_svc_patch:
            self._auth_svc_patch.stop()
        if self._gs_patch:
            self._gs_patch.stop()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()


def _headers():
    return {"Authorization": "Bearer fake-token"}


# ---------------------------------------------------------------------------
# Suite 4a — Known-facts correctness
# ---------------------------------------------------------------------------


class TestKnownFactsCorrectness:

    @pytest.mark.integration
    @pytest.mark.api
    async def test_answer_contains_known_graph_fact(self, async_client):
        """
        ANTI-HALLUCINATION: When graph contains "Alice Smith is CEO of TechNova",
        asking "Who is the CEO of TechNova?" must return Alice Smith.
        """
        graph_content = "Alice Smith is the CEO of TechNova Corp since 2018."
        expected_answer = "Alice Smith is the CEO of TechNova Corp."

        auth = _auth_patch()
        try:
            with _ChatPatch(
                answer=expected_answer,
                items=[
                    _retriever_item(
                        content=graph_content,
                        node_id="person-alice",
                        name="Alice Smith",
                    )
                ],
            ):
                response = await async_client.post(
                    "/api/v1/api/v1/chat",
                    json={
                        "query": "Who is the CEO of TechNova Corp?",
                        "graph_id": GRAPH_ID,
                    },
                    headers=_headers(),
                )
        finally:
            auth.stop()

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["is_grounded"] is True
        assert (
            "Alice Smith" in data["answer"]
        ), "Known fact 'Alice Smith is CEO' was not reflected in the answer"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_answer_cites_correct_source_nodes(self, async_client):
        """
        PROVENANCE: Answer must include source node references that exist in the graph.
        """
        auth = _auth_patch()
        try:
            with _ChatPatch(
                answer="TechNova Corp is headquartered in San Francisco.",
                items=[
                    _retriever_item(
                        content="TechNova Corp, Location: San Francisco",
                        node_id="company-technova",
                        name="TechNova Corp",
                        score=0.95,
                    )
                ],
            ):
                response = await async_client.post(
                    "/api/v1/api/v1/chat",
                    json={
                        "query": "Where is TechNova headquartered?",
                        "graph_id": GRAPH_ID,
                        "include_sources": True,
                    },
                    headers=_headers(),
                )
        finally:
            auth.stop()

        assert response.status_code == 200
        data = response.json()
        assert data["is_grounded"] is True
        assert data["sources"] is not None
        assert len(data["sources"]) > 0
        # The source node ID must match what was in the graph
        source_ids = [s["node_id"] for s in data["sources"]]
        assert (
            "company-technova" in source_ids
        ), f"Expected source node 'company-technova' in {source_ids}"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_confidence_reflects_retrieval_quality(self, async_client):
        """
        ANTI-HALLUCINATION: High retrieval score → confidence > 0.5.
        """
        auth = _auth_patch()
        try:
            with _ChatPatch(
                answer="Alice Smith founded TechNova.",
                items=[
                    _retriever_item(content="Alice Smith founded TechNova", score=0.95),
                    _retriever_item(
                        content="TechNova was established in 2015", score=0.88
                    ),
                ],
            ):
                response = await async_client.post(
                    "/api/v1/api/v1/chat",
                    json={"query": "Who founded TechNova?", "graph_id": GRAPH_ID},
                    headers=_headers(),
                )
        finally:
            auth.stop()

        assert response.status_code == 200
        data = response.json()
        assert (
            data["confidence"] > 0.5
        ), f"Confidence {data['confidence']} too low for high-score retrieval"


# ---------------------------------------------------------------------------
# Suite 4b — No-data / empty graph scenarios
# ---------------------------------------------------------------------------


class TestNoDataResponses:

    @pytest.mark.integration
    @pytest.mark.api
    async def test_empty_graph_returns_no_data_response_not_hallucination(
        self, async_client
    ):
        """
        ANTI-HALLUCINATION: Empty graph → structured no-data response.
        Must NOT invent an answer (is_grounded=False, confidence=0.0).
        """
        auth = _auth_patch()
        try:
            with _ChatPatch(answer="", items=[]):
                response = await async_client.post(
                    "/api/v1/api/v1/chat",
                    json={
                        "query": "Who is the CFO of Acme Corp?",
                        "graph_id": "empty-graph-id",
                    },
                    headers=_headers(),
                )
        finally:
            auth.stop()

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert (
            data["is_grounded"] is False
        ), "is_grounded must be False when no context was retrieved"
        assert (
            data["confidence"] == 0.0
        ), f"Confidence must be 0.0 for empty retrieval, got {data['confidence']}"
        # The response should clearly indicate lack of data (not fabricate an answer)
        answer_lower = data["answer"].lower()
        assert any(
            phrase in answer_lower
            for phrase in [
                "does not contain",
                "no information",
                "not found",
                "don't have",
                "unable to find",
                "no data",
                "cannot find",
                "not available",
            ]
        ), f"Expected a no-data indicator in the answer, got: '{data['answer']}'"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_no_sources_returned_for_empty_graph(self, async_client):
        """ANTI-HALLUCINATION: Empty graph → empty/null sources list."""
        auth = _auth_patch()
        try:
            with _ChatPatch(answer="", items=[]):
                response = await async_client.post(
                    "/api/v1/api/v1/chat",
                    json={
                        "query": "What is the revenue of Phantom Corp?",
                        "graph_id": "empty-graph-id",
                        "include_sources": True,
                    },
                    headers=_headers(),
                )
        finally:
            auth.stop()

        assert response.status_code == 200
        data = response.json()
        sources = data.get("sources") or []
        assert (
            len(sources) == 0
        ), f"Expected 0 sources for empty graph, got {len(sources)}: {sources}"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_out_of_scope_question_does_not_hallucinate(self, async_client):
        """
        ANTI-HALLUCINATION: Question about a topic not in the graph must not
        hallucinate. Graph contains TechNova data; query is about an unrelated
        entity (Phantom Corp) not present.
        """
        # Graph has TechNova entities, no Phantom Corp data → retrieval returns nothing relevant
        auth = _auth_patch()
        try:
            with _ChatPatch(answer="", items=[]):
                response = await async_client.post(
                    "/api/v1/api/v1/chat",
                    json={
                        "query": "What is the annual revenue of Phantom Corp in fiscal year 2024?",
                        "graph_id": GRAPH_ID,
                    },
                    headers=_headers(),
                )
        finally:
            auth.stop()

        assert response.status_code == 200
        data = response.json()
        # Must not claim to know Phantom Corp's revenue
        assert (
            "Phantom Corp" not in data["answer"] or data["is_grounded"] is False
        ), "System appears to have hallucinated Phantom Corp data"
        assert data["confidence"] == 0.0 or data["is_grounded"] is False

    @pytest.mark.integration
    @pytest.mark.api
    async def test_fabricated_entities_not_in_answer(self, async_client):
        """
        ANTI-HALLUCINATION: The answer must only name entities present in the
        retrieved context, not entities invented by the LLM.
        """
        real_entity = "TechNova Corp"
        fabricated_entity = "InvenTech Solutions"  # not in graph/retrieval

        auth = _auth_patch()
        try:
            with _ChatPatch(
                answer=f"{real_entity} is a leading AI company.",
                items=[
                    _retriever_item(
                        content=f"{real_entity} develops AI solutions",
                        node_id="company-technova",
                        name=real_entity,
                    )
                ],
            ):
                response = await async_client.post(
                    "/api/v1/api/v1/chat",
                    json={
                        "query": "Tell me about AI companies in the graph",
                        "graph_id": GRAPH_ID,
                    },
                    headers=_headers(),
                )
        finally:
            auth.stop()

        assert response.status_code == 200
        data = response.json()
        # The fabricated entity should not appear in the answer
        assert (
            fabricated_entity not in data["answer"]
        ), f"Fabricated entity '{fabricated_entity}' appeared in grounded answer"
        # The real entity from the graph should be there
        assert real_entity in data["answer"]


# ---------------------------------------------------------------------------
# Suite 4c — Grounding and provenance integrity
# ---------------------------------------------------------------------------


class TestGroundingIntegrity:

    @pytest.mark.integration
    @pytest.mark.api
    async def test_is_grounded_false_when_zero_sources_retrieved(self, async_client):
        """is_grounded must be False when retriever returns zero items."""
        auth = _auth_patch()
        try:
            with _ChatPatch(answer="Some invented text", items=[]):
                response = await async_client.post(
                    "/api/v1/api/v1/chat",
                    json={"query": "Random question", "graph_id": GRAPH_ID},
                    headers=_headers(),
                )
        finally:
            auth.stop()

        assert response.status_code == 200
        data = response.json()
        assert data["is_grounded"] is False
        assert data["confidence"] == 0.0

    @pytest.mark.integration
    @pytest.mark.api
    async def test_is_grounded_true_when_sources_retrieved(self, async_client):
        """is_grounded must be True when retriever returns relevant items."""
        auth = _auth_patch()
        try:
            with _ChatPatch(
                answer="Alice Smith is the CEO.",
                items=[_retriever_item("Alice Smith - CEO", score=0.9)],
            ):
                response = await async_client.post(
                    "/api/v1/api/v1/chat",
                    json={"query": "Who is the CEO?", "graph_id": GRAPH_ID},
                    headers=_headers(),
                )
        finally:
            auth.stop()

        assert response.status_code == 200
        data = response.json()
        assert data["is_grounded"] is True
        assert data["confidence"] > 0.0

    @pytest.mark.integration
    @pytest.mark.api
    async def test_response_graph_id_matches_request(self, async_client):
        """Response graph_id must match the graph_id in the request (no substitution)."""
        target_graph_id = "specific-graph-id-abc123"

        auth = _auth_patch()
        try:
            with _ChatPatch(
                answer="Answer from the correct graph.",
                items=[_retriever_item("Some content")],
            ):
                response = await async_client.post(
                    "/api/v1/api/v1/chat",
                    json={"query": "Test query", "graph_id": target_graph_id},
                    headers=_headers(),
                )
        finally:
            auth.stop()

        assert response.status_code == 200
        data = response.json()
        assert (
            data["graph_id"] == target_graph_id
        ), f"Response graph_id '{data['graph_id']}' does not match request '{target_graph_id}'"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_context_retrieval_count_matches_sources(self, async_client):
        """
        PROVENANCE: context.total_results must equal the number of retrieved items,
        and sources count must be consistent.
        """
        n_items = 3
        items = [
            _retriever_item(
                f"Entity content {i}", node_id=f"node-{i}", score=0.9 - i * 0.05
            )
            for i in range(n_items)
        ]

        auth = _auth_patch()
        try:
            with _ChatPatch(
                answer="Answer based on 3 sources.",
                items=items,
            ):
                response = await async_client.post(
                    "/api/v1/api/v1/chat",
                    json={
                        "query": "Summary query",
                        "graph_id": GRAPH_ID,
                        "include_sources": True,
                        "return_context": True,
                    },
                    headers=_headers(),
                )
        finally:
            auth.stop()

        assert response.status_code == 200
        data = response.json()
        assert data["is_grounded"] is True
        assert (
            len(data["sources"]) == n_items
        ), f"Expected {n_items} sources, got {len(data['sources'])}"
        assert (
            data["context"]["total_results"] == n_items
        ), f"context.total_results should be {n_items}, got {data['context']['total_results']}"


# ---------------------------------------------------------------------------
# Suite 4d — Error recovery (no hallucination on failure)
# ---------------------------------------------------------------------------


class TestErrorRecoveryNoHallucination:

    @pytest.mark.integration
    @pytest.mark.api
    async def test_service_error_returns_500_not_fabricated_answer(self, async_client):
        """
        ANTI-HALLUCINATION: If retrieval fails (e.g. Neo4j down), the endpoint
        must return 500 — never a fabricated answer passed off as grounded.
        """
        auth = _auth_patch()
        try:
            with _ChatPatch(retriever_error=Exception("Neo4j connection refused")):
                response = await async_client.post(
                    "/api/v1/api/v1/chat",
                    json={"query": "What are the key entities?", "graph_id": GRAPH_ID},
                    headers=_headers(),
                )
        finally:
            auth.stop()

        assert response.status_code == 500, (
            f"Expected 500 on retrieval failure, got {response.status_code} — "
            "never return fabricated content when retrieval fails"
        )
        # The error response must not claim to be grounded
        if response.headers.get("content-type", "").startswith("application/json"):
            data = response.json()
            # Should be an error detail, not a chat answer
            assert "detail" in data or "error" in data
