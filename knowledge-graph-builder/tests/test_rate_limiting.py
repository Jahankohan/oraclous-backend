"""
Integration tests for API rate limiting (ORA-104).

Verifies that rate-limited endpoints return HTTP 429 when the per-minute
limit is exceeded.  Tests use slowapi's built-in test-mode override so
they run without a live Neo4j instance.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.core.limiter import limiter


@pytest.fixture(autouse=True)
def reset_limiter():
    """Reset slowapi limiter storage between tests."""
    limiter._storage.reset()
    yield
    limiter._storage.reset()


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _hit_endpoint(client: TestClient, method: str, url: str, n: int, **kwargs):
    """Call an endpoint n times and return list of status codes."""
    statuses = []
    for _ in range(n):
        resp = getattr(client, method)(url, **kwargs)
        statuses.append(resp.status_code)
    return statuses


# ---------------------------------------------------------------------------
# Chat endpoint  — limit: 30/minute
# ---------------------------------------------------------------------------

class TestChatRateLimit:
    CHAT_PAYLOAD = {
        "message": "hello",
        "mode": "vector",
        "file_names": [],
        "session_id": "test-session",
    }

    @patch("app.routers.chat.EnhancedChatService")
    @patch("app.routers.chat.get_neo4j_client")
    def test_chat_bot_allows_within_limit(self, mock_neo4j, mock_service_cls, client):
        mock_service = AsyncMock()
        mock_service.chat.return_value = MagicMock(
            model_dump=lambda: {"response": "ok", "session_id": "s"}
        )
        mock_service_cls.return_value = mock_service

        statuses = _hit_endpoint(client, "post", "/api/v1/chat_bot", 5, json=self.CHAT_PAYLOAD)
        assert all(s != 429 for s in statuses), f"Unexpected 429 within limit: {statuses}"

    @patch("app.routers.chat.EnhancedChatService")
    @patch("app.routers.chat.get_neo4j_client")
    def test_chat_bot_returns_429_when_limit_exceeded(self, mock_neo4j, mock_service_cls, client):
        mock_service = AsyncMock()
        mock_service.chat.return_value = MagicMock(
            model_dump=lambda: {"response": "ok", "session_id": "s"}
        )
        mock_service_cls.return_value = mock_service

        # Exhaust the 30/minute limit
        statuses = _hit_endpoint(client, "post", "/api/v1/chat_bot", 31, json=self.CHAT_PAYLOAD)
        assert 429 in statuses, "Expected 429 after exceeding 30/minute chat limit"
        assert statuses[-1] == 429


# ---------------------------------------------------------------------------
# Ingest endpoints  — limit: 10/minute
# ---------------------------------------------------------------------------

class TestIngestRateLimit:
    @patch("app.routers.documents.AdvancedGraphIntegrationService")
    @patch("app.routers.documents.get_neo4j_client")
    def test_upload_returns_429_when_limit_exceeded(self, mock_neo4j, mock_svc_cls, client):
        mock_svc = AsyncMock()
        mock_svc.scan_sources.return_value = []
        mock_svc_cls.return_value = mock_svc

        statuses = _hit_endpoint(
            client, "post", "/api/v1/upload", 11,
            files={"files": ("test.txt", b"data", "text/plain")},
        )
        assert 429 in statuses, "Expected 429 after exceeding 10/minute upload limit"

    @patch("app.routers.documents.DocumentService")
    @patch("app.routers.documents.get_neo4j_client")
    def test_url_scan_returns_429_when_limit_exceeded(self, mock_neo4j, mock_svc_cls, client):
        mock_svc = AsyncMock()
        mock_svc.scan_sources.return_value = []
        mock_svc_cls.return_value = mock_svc

        payload = {"source_type": "web", "url": "https://example.com"}
        statuses = _hit_endpoint(client, "post", "/api/v1/url/scan", 11, json=payload)
        assert 429 in statuses, "Expected 429 after exceeding 10/minute url/scan limit"

    @patch("app.routers.documents.ExtractionService")
    @patch("app.routers.documents.get_neo4j_client")
    def test_extract_returns_429_when_limit_exceeded(self, mock_neo4j, mock_svc_cls, client):
        mock_svc = AsyncMock()
        mock_svc_cls.return_value = mock_svc

        payload = {"file_names": ["doc.pdf"], "model": "openai_gpt_4o"}
        statuses = _hit_endpoint(client, "post", "/api/v1/extract", 11, json=payload)
        assert 429 in statuses, "Expected 429 after exceeding 10/minute extract limit"


# ---------------------------------------------------------------------------
# Connection endpoint  — limit: 10/minute
# ---------------------------------------------------------------------------

class TestConnectionRateLimit:
    CONNECT_PAYLOAD = {
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "password",
        "database": "neo4j",
    }

    @patch("app.routers.infrastructure.Neo4jClient")
    @patch("app.routers.infrastructure.get_neo4j_client")
    def test_connect_returns_429_when_limit_exceeded(self, mock_neo4j, mock_client_cls, client):
        mock_test_client = MagicMock()
        mock_test_client.get_schema.return_value = {}
        mock_test_client.check_vector_index_dimensions.return_value = 384
        mock_client_cls.return_value = mock_test_client

        statuses = _hit_endpoint(
            client, "post", "/api/v1/connect", 11, json=self.CONNECT_PAYLOAD
        )
        assert 429 in statuses, "Expected 429 after exceeding 10/minute connect limit"
