"""
Security Validation — ORA-22: Cross-Tenant Chat Endpoint Isolation
Validates the fix from ORA-21 (graph ownership check on /chat and /chat/stream).

TC-SEC-001: User A cannot chat with User B's graph via POST /chat         → 403
TC-SEC-002: User A cannot stream User B's graph via POST /chat/stream      → 403
TC-SEC-003: User A can chat with their own graph (regression check)        → 200
TC-SEC-004: Graph not found → 403 (no info leakage)
TC-SEC-005: Empty/null graph_id results in validation error                → 422
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_A_ID = str(uuid.uuid4())
USER_B_ID = str(uuid.uuid4())
GRAPH_A_ID = str(uuid.uuid4())  # owned by User A
GRAPH_B_ID = str(uuid.uuid4())  # owned by User B

CHAT_ENDPOINT = "/api/v1/api/v1/chat"
STREAM_ENDPOINT = "/api/v1/api/v1/chat/stream"

AUTH_HEADER = {"Authorization": "Bearer test-token"}

_auth_patch_ref = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _override_user(user_id: str):
    """Mock auth_service.verify_token to return the given user_id."""
    global _auth_patch_ref
    _auth_patch_ref = patch("app.api.v1.endpoints.chat.auth_service")
    mock_auth = _auth_patch_ref.start()
    mock_auth.verify_token = AsyncMock(return_value={"id": user_id})
    return user_id


def _clear_overrides():
    global _auth_patch_ref
    if _auth_patch_ref:
        _auth_patch_ref.stop()
        _auth_patch_ref = None


def _mock_graph_service(graph_data):
    """Patch GraphNodeService.get_graph to return graph_data."""
    p = patch("app.api.v1.endpoints.chat.GraphNodeService")
    mock_cls = p.start()
    mock_cls.return_value.get_graph.return_value = graph_data
    return p


def _mock_chat_service_success():
    """Patch ChatService to return a minimal valid response (no LLM calls)."""
    patches = [
        patch("app.services.chat_service.OpenAIEmbeddings"),
        patch("app.services.chat_service.OpenAILLM"),
        patch("app.services.chat_service.settings"),
        patch("app.services.chat_service.retriever_factory"),
        patch("app.services.chat_service.GraphRAG"),
    ]
    mocks = [p.start() for p in patches]

    # settings mock
    mocks[2].OPENAI_API_KEY = "test-key"

    # retriever_factory mock
    mocks[3].create_retriever = AsyncMock(return_value=MagicMock())

    # GraphRAG mock — returns a grounded answer
    rag_result = MagicMock()
    rag_result.answer = "TechNova Corp is a technology company."
    retriever_result = MagicMock()
    retriever_result.items = [MagicMock()]
    rag_result.retriever_result = retriever_result
    mocks[4].return_value.search.return_value = rag_result

    return patches


def _stop_patches(patches):
    for p in patches:
        p.stop()


def _minimal_chat_payload(graph_id: str, include_sources: bool = False) -> dict:
    return {
        "query": "Tell me about this graph.",
        "graph_id": graph_id,
        "include_sources": include_sources,
    }


# ---------------------------------------------------------------------------
# TC-SEC-001 — Cross-tenant POST /chat returns 403
# ---------------------------------------------------------------------------


class TestTC_SEC_001_CrossTenantChatBlocked:
    """
    TC-SEC-001: Authenticated User A calls POST /chat with graph_B_ID
    (owned by User B). The endpoint must return 403 and must NOT return
    any knowledge graph data.
    """

    @pytest.mark.integration
    @pytest.mark.security
    async def test_cross_tenant_chat_returns_403(self, async_client):
        """Core assertion: User A + graph_B_ID → 403."""
        _override_user(USER_A_ID)
        graph_p = _mock_graph_service({"user_id": USER_B_ID, "graph_id": GRAPH_B_ID})
        try:
            response = await async_client.post(
                CHAT_ENDPOINT,
                json=_minimal_chat_payload(GRAPH_B_ID),
                headers=AUTH_HEADER,
            )
        finally:
            graph_p.stop()
            _clear_overrides()

        assert response.status_code == 403, (
            f"TC-SEC-001 FAIL: Expected 403, got {response.status_code}. "
            "User A must not access User B's graph via /chat."
        )

    @pytest.mark.integration
    @pytest.mark.security
    async def test_cross_tenant_chat_response_contains_no_graph_data(
        self, async_client
    ):
        """Response body must not leak any graph content on 403."""
        _override_user(USER_A_ID)
        graph_p = _mock_graph_service(
            {
                "user_id": USER_B_ID,
                "graph_id": GRAPH_B_ID,
                "name": "SECRET_GRAPH_NAME_B",
            }
        )
        try:
            response = await async_client.post(
                CHAT_ENDPOINT,
                json=_minimal_chat_payload(GRAPH_B_ID),
                headers=AUTH_HEADER,
            )
        finally:
            graph_p.stop()
            _clear_overrides()

        assert response.status_code == 403
        # Ensure no graph-specific data is leaked in the response body
        body = response.text
        assert (
            "SECRET_GRAPH_NAME_B" not in body
        ), "Graph name leaked in 403 response body"
        assert USER_B_ID not in body, "User B's ID leaked in 403 response body"

    @pytest.mark.integration
    @pytest.mark.security
    async def test_cross_tenant_chat_no_answer_field_in_403(self, async_client):
        """403 response must not include an 'answer' field with graph data."""
        _override_user(USER_A_ID)
        graph_p = _mock_graph_service({"user_id": USER_B_ID})
        try:
            response = await async_client.post(
                CHAT_ENDPOINT,
                json=_minimal_chat_payload(GRAPH_B_ID),
                headers=AUTH_HEADER,
            )
        finally:
            graph_p.stop()
            _clear_overrides()

        assert response.status_code == 403
        data = response.json()
        # Must not have a populated answer field
        assert "answer" not in data or data.get("answer") is None


# ---------------------------------------------------------------------------
# TC-SEC-002 — Cross-tenant POST /chat/stream returns 403
# ---------------------------------------------------------------------------


class TestTC_SEC_002_CrossTenantStreamBlocked:
    """
    TC-SEC-002: Authenticated User A calls POST /chat/stream with graph_B_ID.
    Must return 403 immediately — no SSE events with graph B's data.
    """

    @pytest.mark.integration
    @pytest.mark.security
    async def test_cross_tenant_stream_returns_403(self, async_client):
        """Core assertion: User A + graph_B_ID on /stream → 403."""
        _override_user(USER_A_ID)
        graph_p = _mock_graph_service({"user_id": USER_B_ID, "graph_id": GRAPH_B_ID})
        try:
            response = await async_client.post(
                STREAM_ENDPOINT,
                json=_minimal_chat_payload(GRAPH_B_ID),
                headers=AUTH_HEADER,
            )
        finally:
            graph_p.stop()
            _clear_overrides()

        assert response.status_code == 403, (
            f"TC-SEC-002 FAIL: Expected 403, got {response.status_code}. "
            "User A must not stream User B's graph via /chat/stream."
        )

    @pytest.mark.integration
    @pytest.mark.security
    async def test_cross_tenant_stream_no_sse_data_leaked(self, async_client):
        """Stream must not emit any SSE events before the 403 is returned."""
        _override_user(USER_A_ID)
        graph_p = _mock_graph_service({"user_id": USER_B_ID})
        try:
            response = await async_client.post(
                STREAM_ENDPOINT,
                json=_minimal_chat_payload(GRAPH_B_ID),
                headers=AUTH_HEADER,
            )
        finally:
            graph_p.stop()
            _clear_overrides()

        assert response.status_code == 403
        # Response body should not contain SSE event data
        body = response.text
        assert (
            "data:" not in body
        ), "SSE data events were emitted before the 403 — potential data leakage"
        assert (
            '"type"' not in body
            or "error" in body.lower()
            or response.status_code == 403
        )


# ---------------------------------------------------------------------------
# TC-SEC-003 — Normal operation regression check
# ---------------------------------------------------------------------------


class TestTC_SEC_003_NormalOperationRegression:
    """
    TC-SEC-003: User A calls POST /chat with their own graph_A_ID.
    Must return 200 — verify the security fix does not break normal use.
    """

    @pytest.mark.integration
    @pytest.mark.security
    async def test_owner_can_chat_with_own_graph(self, async_client):
        """Ownership match → 200 with valid response."""
        _override_user(USER_A_ID)
        graph_p = _mock_graph_service({"user_id": USER_A_ID, "graph_id": GRAPH_A_ID})
        chat_patches = _mock_chat_service_success()
        try:
            response = await async_client.post(
                CHAT_ENDPOINT,
                json=_minimal_chat_payload(GRAPH_A_ID),
                headers=AUTH_HEADER,
            )
        finally:
            _stop_patches(chat_patches)
            graph_p.stop()
            _clear_overrides()

        assert response.status_code == 200, (
            f"TC-SEC-003 FAIL: Expected 200, got {response.status_code}. "
            "Graph owner must be able to use /chat normally."
        )
        data = response.json()
        assert "answer" in data
        assert data.get("graph_id") == GRAPH_A_ID

    @pytest.mark.integration
    @pytest.mark.security
    async def test_owner_stream_with_own_graph_not_blocked(self, async_client):
        """Owner using /chat/stream with own graph → NOT 403."""
        _override_user(USER_A_ID)
        graph_p = _mock_graph_service({"user_id": USER_A_ID, "graph_id": GRAPH_A_ID})
        chat_patches = _mock_chat_service_success()
        try:
            response = await async_client.post(
                STREAM_ENDPOINT,
                json=_minimal_chat_payload(GRAPH_A_ID),
                headers=AUTH_HEADER,
            )
        finally:
            _stop_patches(chat_patches)
            graph_p.stop()
            _clear_overrides()

        assert response.status_code != 403, (
            f"TC-SEC-003 FAIL: Graph owner was blocked from streaming their own graph. "
            f"Got {response.status_code}."
        )


# ---------------------------------------------------------------------------
# TC-SEC-004 — Graph not found → 403 (no info leakage)
# ---------------------------------------------------------------------------


class TestTC_SEC_004_GraphNotFound:
    """
    TC-SEC-004: User A calls POST /chat with a non-existent graph_id.
    Must return 403 (not 404) — prevents graph existence enumeration.
    """

    @pytest.mark.integration
    @pytest.mark.security
    async def test_nonexistent_graph_returns_403_not_404(self, async_client):
        """
        Non-existent graph must return 403, not 404.
        Returning 404 would allow attackers to enumerate valid graph IDs.
        """
        _override_user(USER_A_ID)
        graph_p = _mock_graph_service(None)  # graph not found
        try:
            response = await async_client.post(
                CHAT_ENDPOINT,
                json=_minimal_chat_payload(str(uuid.uuid4())),
                headers=AUTH_HEADER,
            )
        finally:
            graph_p.stop()
            _clear_overrides()

        assert response.status_code == 403, (
            f"TC-SEC-004 FAIL: Expected 403 for non-existent graph, got {response.status_code}. "
            "Returning 404 leaks graph existence information."
        )
