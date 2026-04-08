"""
Unit tests for graph ownership enforcement on chat endpoints.

Security invariant: No authenticated user may access another tenant's
knowledge graph via the chat API. These tests verify that /chat and
/chat/stream return 403 when the requesting user does not own the graph.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

OWNER_USER_ID = "owner-user-abc"
OTHER_USER_ID = "intruder-user-xyz"

AUTH_HEADER = {"Authorization": "Bearer fake-token"}
CHAT_PAYLOAD = {"query": "Who is the CEO?", "graph_id": "some-graph-id"}


def _mock_auth(user_id: str):
    """Return a started patch that makes auth_service.verify_token return the given user_id."""
    p = patch("app.api.v1.endpoints.chat.auth_service")
    mock_auth = p.start()
    mock_auth.verify_token = AsyncMock(return_value={"id": user_id})
    return p


# ---------------------------------------------------------------------------
# POST /chat ownership checks
# ---------------------------------------------------------------------------

class TestChatOwnership:

    @pytest.mark.unit
    async def test_chat_returns_403_when_graph_not_found(self, async_client):
        """Returns 403 when graph_id does not exist (not 404 — avoid enumeration)."""
        p = _mock_auth(OWNER_USER_ID)
        try:
            with patch("app.api.v1.endpoints.chat.GraphNodeService") as mock_gs_cls:
                mock_gs_cls.return_value.get_graph.return_value = None
                response = await async_client.post(
                    "/api/v1/api/v1/chat",
                    json=CHAT_PAYLOAD,
                    headers=AUTH_HEADER,
                )
        finally:
            p.stop()

        assert response.status_code == 403
        assert response.json()["detail"] == "Access denied"

    @pytest.mark.unit
    async def test_chat_returns_403_when_graph_owned_by_other_user(self, async_client):
        """Returns 403 when authenticated user does not own the requested graph."""
        p = _mock_auth(OTHER_USER_ID)
        try:
            with patch("app.api.v1.endpoints.chat.GraphNodeService") as mock_gs_cls:
                mock_gs_cls.return_value.get_graph.return_value = {"user_id": OWNER_USER_ID}
                response = await async_client.post(
                    "/api/v1/api/v1/chat",
                    json=CHAT_PAYLOAD,
                    headers=AUTH_HEADER,
                )
        finally:
            p.stop()

        assert response.status_code == 403
        assert response.json()["detail"] == "Access denied"

    @pytest.mark.unit
    async def test_chat_proceeds_when_ownership_verified(self, async_client):
        """Ownership check passes and endpoint logic executes when user owns the graph."""
        p = _mock_auth(OWNER_USER_ID)
        try:
            with patch("app.api.v1.endpoints.chat.GraphNodeService") as mock_gs_cls, \
                 patch("app.services.chat_service.OpenAIEmbeddings"), \
                 patch("app.services.chat_service.OpenAILLM"), \
                 patch("app.services.chat_service.settings") as mock_cfg, \
                 patch("app.services.chat_service.retriever_factory") as mock_fac, \
                 patch("app.services.chat_service.GraphRAG") as mock_rag_cls:
                mock_gs_cls.return_value.get_graph.return_value = {"user_id": OWNER_USER_ID}
                mock_cfg.OPENAI_API_KEY = "test-key"
                mock_fac.create_retriever.return_value = MagicMock()
                rag_instance = MagicMock()
                result = MagicMock()
                result.answer = "The CEO is Alice."
                result.retriever_result = MagicMock()
                result.retriever_result.items = []
                rag_instance.search.return_value = result
                mock_rag_cls.return_value = rag_instance

                response = await async_client.post(
                    "/api/v1/api/v1/chat",
                    json=CHAT_PAYLOAD,
                    headers=AUTH_HEADER,
                )
        finally:
            p.stop()

        # Ownership check passed — endpoint ran (200 or 500 from downstream mocks, not 403)
        assert response.status_code != 403

    @pytest.mark.unit
    async def test_chat_ownership_check_uses_correct_graph_id(self, async_client):
        """GraphNodeService.get_graph is called with the graph_id from the request body."""
        p = _mock_auth(OWNER_USER_ID)
        try:
            with patch("app.api.v1.endpoints.chat.GraphNodeService") as mock_gs_cls:
                mock_instance = mock_gs_cls.return_value
                mock_instance.get_graph.return_value = None
                await async_client.post(
                    "/api/v1/api/v1/chat",
                    json={"query": "Q", "graph_id": "specific-graph-id"},
                    headers=AUTH_HEADER,
                )
                mock_instance.get_graph.assert_called_once_with("specific-graph-id")
        finally:
            p.stop()


# ---------------------------------------------------------------------------
# POST /chat/stream ownership checks
# ---------------------------------------------------------------------------

class TestChatStreamOwnership:

    @pytest.mark.unit
    async def test_stream_returns_403_when_graph_not_found(self, async_client):
        """POST /chat/stream returns 403 when graph_id does not exist."""
        p = _mock_auth(OWNER_USER_ID)
        try:
            with patch("app.api.v1.endpoints.chat.GraphNodeService") as mock_gs_cls:
                mock_gs_cls.return_value.get_graph.return_value = None
                response = await async_client.post(
                    "/api/v1/api/v1/chat/stream",
                    json=CHAT_PAYLOAD,
                    headers=AUTH_HEADER,
                )
        finally:
            p.stop()

        assert response.status_code == 403
        assert response.json()["detail"] == "Access denied"

    @pytest.mark.unit
    async def test_stream_returns_403_when_graph_owned_by_other_user(self, async_client):
        """POST /chat/stream returns 403 when authenticated user does not own the graph."""
        p = _mock_auth(OTHER_USER_ID)
        try:
            with patch("app.api.v1.endpoints.chat.GraphNodeService") as mock_gs_cls:
                mock_gs_cls.return_value.get_graph.return_value = {"user_id": OWNER_USER_ID}
                response = await async_client.post(
                    "/api/v1/api/v1/chat/stream",
                    json=CHAT_PAYLOAD,
                    headers=AUTH_HEADER,
                )
        finally:
            p.stop()

        assert response.status_code == 403
        assert response.json()["detail"] == "Access denied"

    @pytest.mark.unit
    async def test_stream_proceeds_when_ownership_verified(self, async_client):
        """Ownership check passes and streaming pipeline executes for graph owner."""
        p = _mock_auth(OWNER_USER_ID)
        try:
            with patch("app.api.v1.endpoints.chat.GraphNodeService") as mock_gs_cls, \
                 patch("app.services.chat_service.OpenAIEmbeddings"), \
                 patch("app.services.chat_service.OpenAILLM"), \
                 patch("app.services.chat_service.settings") as mock_cfg, \
                 patch("app.services.chat_service.retriever_factory") as mock_fac, \
                 patch("app.services.chat_service.GraphRAG") as mock_rag_cls:
                mock_gs_cls.return_value.get_graph.return_value = {"user_id": OWNER_USER_ID}
                mock_cfg.OPENAI_API_KEY = "test-key"
                mock_fac.create_retriever.return_value = MagicMock()
                rag_instance = MagicMock()
                result = MagicMock()
                result.answer = "Alice is CEO."
                result.retriever_result = MagicMock()
                result.retriever_result.items = []
                rag_instance.search.return_value = result
                mock_rag_cls.return_value = rag_instance

                response = await async_client.post(
                    "/api/v1/api/v1/chat/stream",
                    json=CHAT_PAYLOAD,
                    headers=AUTH_HEADER,
                )
        finally:
            p.stop()

        assert response.status_code != 403
