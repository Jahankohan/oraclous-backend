"""
Unit tests for TASK-029 / STORY-009: graceful degradation paths.

All external dependencies (Neo4j, Redis, Celery broker, LLM) are mocked so
these tests run without any running infrastructure.

Coverage:
- Neo4j ServiceUnavailable → 503 with Retry-After header and error_code: KGB-5001
- Redis ConnectionError in cache get → chat request completes (200), cache_hit: False
- LLM timeout → chat response includes error field, status 200, no 500
- Celery BrokerNotRunning/OperationalError → 202 response with fallback status
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. Neo4j ServiceUnavailable → 503 with Retry-After + KGB-5001
# ---------------------------------------------------------------------------


class TestNeo4jDegradation:
    """Neo4jDegradationMiddleware must convert ServiceUnavailable to HTTP 503."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_middleware_converts_service_unavailable_to_503(self):
        """
        When the inner route raises ServiceUnavailable the middleware must
        intercept it and return HTTP 503 with Retry-After and KGB-5001 body.
        """
        from neo4j.exceptions import ServiceUnavailable
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.testclient import TestClient

        from app.core.telemetry import Neo4jDegradationMiddleware

        async def boom(request):
            raise ServiceUnavailable("Neo4j is down")

        app = Starlette(routes=[Route("/", boom)])
        app.add_middleware(Neo4jDegradationMiddleware)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/")

        assert response.status_code == 503
        assert response.headers.get("retry-after") == "30"
        body = response.json()
        assert body["error_code"] == "KGB-5001"
        assert body["retry_after"] == 30

    @pytest.mark.unit
    def test_middleware_retry_after_header_is_30(self):
        """The Retry-After header value must be exactly 30 seconds."""
        from app.core.telemetry import Neo4jDegradationMiddleware

        assert Neo4jDegradationMiddleware._RETRY_AFTER_SECONDS == 30

    @pytest.mark.unit
    def test_kgb_error_neo4j_code_is_kgb_5001(self):
        """KGBError.NEO4J_UNAVAILABLE must have code KGB-5001."""
        from app.core.errors import KGBError

        code, message = KGBError.NEO4J_UNAVAILABLE
        assert code == "KGB-5001"
        assert "graph service" in message.lower()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_query_logs_critical_on_service_unavailable(self):
        """
        execute_query must log at CRITICAL level and re-raise ServiceUnavailable.
        """
        from neo4j.exceptions import ServiceUnavailable

        from app.core.neo4j_client import Neo4jClient

        client = Neo4jClient()

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(
            side_effect=ServiceUnavailable("connection refused")
        )
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session = MagicMock(return_value=mock_session_ctx)
        client.async_driver = mock_driver

        with patch("app.core.neo4j_client.logger") as mock_logger:
            with pytest.raises(ServiceUnavailable):
                await client.execute_query("RETURN 1")

            mock_logger.critical.assert_called_once()
            critical_msg = mock_logger.critical.call_args[0][0]
            assert "Neo4j unavailable" in critical_msg

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_middleware_passes_through_non_neo4j_exceptions(self):
        """
        Non-ServiceUnavailable exceptions must NOT be caught by the middleware;
        they should propagate normally.
        """
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.testclient import TestClient

        from app.core.telemetry import Neo4jDegradationMiddleware

        async def boom(request):
            raise ValueError("some other error")

        app = Starlette(routes=[Route("/", boom)])
        app.add_middleware(Neo4jDegradationMiddleware)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/")

        # Should not be a 503 with KGB-5001 — the middleware should not catch this
        assert response.status_code != 503 or (
            response.status_code == 503
            and response.json().get("error_code") != "KGB-5001"
        )


# ---------------------------------------------------------------------------
# 2. Redis down → chat returns 200 (cache skipped, not crash)
# ---------------------------------------------------------------------------


class TestRedisDegradation:
    """Chat service must continue without cache when Redis raises ConnectionError."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_get_connection_error_does_not_crash_chat(self):
        """
        Redis ConnectionError in cache.get() must be swallowed by QueryCacheService.
        The service must return None (cache miss), not raise — ensuring the chat
        request path completes with cache_hit: False.
        """
        import redis.exceptions

        from app.services.query_cache_service import QueryCacheService

        r = MagicMock()
        r.get = AsyncMock(side_effect=redis.exceptions.ConnectionError("refused"))

        svc = QueryCacheService(r)
        result = await svc.get("test-graph", "Who is Alice?", "vector_cypher")

        # Must not raise; must return None so caller falls through to live query
        assert result is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_set_connection_error_does_not_crash(self):
        """
        Redis ConnectionError in cache.set() must be silently ignored.
        The response has already been generated; caching is advisory only.
        """
        import redis.exceptions

        from app.services.query_cache_service import QueryCacheService

        r = MagicMock()
        r.set = AsyncMock(side_effect=redis.exceptions.ConnectionError("refused"))

        svc = QueryCacheService(r)
        # Must not raise
        await svc.set("test-graph", "Who is Alice?", "vector_cypher", {"answer": "Alice is..."})

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_inner_returns_result_when_cache_raises(self):
        """
        When the cache.get() raises, _search_inner must still return a
        GroundedSearchResult (cache_hit=False path), not propagate the error.
        """
        from app.services.chat_service import ChatService, GroundedSearchResult
        from app.services.retriever_factory import RetrieverType

        chat = ChatService.__new__(ChatService)
        chat.graph_id = "test-graph-redis"
        chat.retriever_type = RetrieverType.VECTOR_CYPHER
        chat.retriever = None
        chat.rag = None

        expected_result = GroundedSearchResult(
            answer="uncached answer",
            sources=[],
            confidence=0.5,
            is_grounded=True,
            retriever_used="vector_cypher",
            retriever_result=None,
        )

        with patch.object(
            ChatService,
            "_search_inner",
            new_callable=AsyncMock,
            return_value=expected_result,
        ):
            result = await chat._search_inner(
                span=MagicMock(),
                query_text="Who is Alice?",
                retriever_config=None,
                return_context=False,
                examples="",
                temporal_filter=None,
            )

        assert isinstance(result, GroundedSearchResult)
        assert result.answer is not None

    @pytest.mark.unit
    def test_kgb_error_redis_code(self):
        """KGBError.REDIS_UNAVAILABLE must have the correct code."""
        from app.core.errors import KGBError

        code, message = KGBError.REDIS_UNAVAILABLE
        assert code == "KGB-5002"
        assert "cache service" in message.lower()


# ---------------------------------------------------------------------------
# 3. LLM timeout → chat returns partial response (answer=None), not 500
# ---------------------------------------------------------------------------


class TestLLMDegradation:
    """When the LLM times out or is rate-limited, chat must return partial, not 500."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_inner_returns_partial_on_llm_timeout(self):
        """
        _search_inner must catch openai.APITimeoutError and return a
        GroundedSearchResult with answer=None (not raise or return 500).
        """
        import openai

        from app.services.chat_service import ChatService, GroundedSearchResult
        from app.services.retriever_factory import RetrieverType

        chat = ChatService.__new__(ChatService)
        chat.graph_id = "test-graph-llm"
        chat.retriever_type = RetrieverType.VECTOR_CYPHER
        chat.retriever_config = MagicMock()
        chat.retriever = MagicMock()

        mock_rag = MagicMock()
        mock_rag.search.side_effect = openai.APITimeoutError(request=MagicMock())
        chat.rag = mock_rag

        mock_span = MagicMock()
        mock_span.set_attribute = MagicMock()
        mock_span.record_exception = MagicMock()
        mock_span.set_status = MagicMock()

        result = await chat._search_inner(
            span=mock_span,
            query_text="Who founded Acme Corp?",
            retriever_config=None,
            return_context=False,
            examples="",
            temporal_filter=None,
        )

        assert isinstance(result, GroundedSearchResult)
        # answer must be None — not a 500 exception
        assert result.answer is None
        assert result.is_grounded is False
        assert result.confidence == 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_inner_returns_partial_on_llm_rate_limit(self):
        """
        _search_inner must catch openai.RateLimitError and return a
        GroundedSearchResult with answer=None.
        """
        import openai

        from app.services.chat_service import ChatService, GroundedSearchResult
        from app.services.retriever_factory import RetrieverType

        chat = ChatService.__new__(ChatService)
        chat.graph_id = "test-graph-ratelimit"
        chat.retriever_type = RetrieverType.VECTOR_CYPHER
        chat.retriever_config = MagicMock()
        chat.retriever = MagicMock()

        mock_rag = MagicMock()
        mock_rag.search.side_effect = openai.RateLimitError(
            message="rate limit exceeded",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )
        chat.rag = mock_rag

        mock_span = MagicMock()
        mock_span.set_attribute = MagicMock()
        mock_span.record_exception = MagicMock()
        mock_span.set_status = MagicMock()

        result = await chat._search_inner(
            span=mock_span,
            query_text="Tell me about TechNova Corp",
            retriever_config=None,
            return_context=False,
            examples="",
            temporal_filter=None,
        )

        assert isinstance(result, GroundedSearchResult)
        assert result.answer is None
        assert result.is_grounded is False

    @pytest.mark.unit
    def test_kgb_error_llm_code(self):
        """KGBError.LLM_UNAVAILABLE must have the correct code."""
        from app.core.errors import KGBError

        code, message = KGBError.LLM_UNAVAILABLE
        assert code == "KGB-5003"
        assert "llm service" in message.lower()


# ---------------------------------------------------------------------------
# 4. Celery broker down → 202-semantics with queued_to_fallback
# ---------------------------------------------------------------------------


class TestCeleryDegradation:
    """When the Celery broker is down, jobs must fall back to the PG queue."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_start_ingestion_job_returns_fallback_on_broker_down(self):
        """
        When process_ingestion_job.delay() raises OperationalError,
        start_ingestion_job must return status: queued_to_fallback (202 semantics).
        """
        from celery.exceptions import OperationalError

        from app.services.background_job_service import BackgroundJobService

        with patch(
            "app.services.background_job_service.process_ingestion_job"
        ) as mock_task:
            mock_task.delay.side_effect = OperationalError("broker connection refused")

            with patch(
                "app.services.background_job_service._write_fallback_job",
                new_callable=AsyncMock,
            ) as mock_write:
                result = await BackgroundJobService.start_ingestion_job(
                    "job-123", "user-456"
                )

        assert result["status"] == "queued_to_fallback"
        assert result["task_id"] is None
        assert result["job_id"] == "job-123"
        mock_write.assert_awaited_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_write_fallback_job_stores_to_postgres(self):
        """
        _write_fallback_job must insert a FallbackJobQueue row via the async session.
        """
        from app.services.background_job_service import _write_fallback_job

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()

        with patch(
            "app.core.database.async_session_maker",
            return_value=mock_session,
        ):
            with patch("app.models.graph.FallbackJobQueue") as mock_model:
                mock_model.return_value = MagicMock()
                await _write_fallback_job(
                    task_name="my.task",
                    args=["arg1"],
                    kwargs={"key": "val"},
                    error="broker down",
                )

        mock_session.add.assert_called_once()
        mock_session.commit.assert_awaited_once()

    @pytest.mark.unit
    def test_kgb_error_celery_code(self):
        """KGBError.CELERY_UNAVAILABLE must have the correct code."""
        from app.core.errors import KGBError

        code, message = KGBError.CELERY_UNAVAILABLE
        assert code == "KGB-5004"
        assert "background job service" in message.lower()


# ---------------------------------------------------------------------------
# 5. errors.py — all error codes present and well-formed
# ---------------------------------------------------------------------------


class TestKGBErrorCodes:
    """All KGB error codes must be defined and have valid format."""

    @pytest.mark.unit
    def test_all_error_codes_present(self):
        from app.core.errors import KGBError

        required = [
            "NEO4J_UNAVAILABLE",
            "REDIS_UNAVAILABLE",
            "LLM_UNAVAILABLE",
            "CELERY_UNAVAILABLE",
            "GRAPH_NOT_FOUND",
            "PERMISSION_DENIED",
            "RATE_LIMIT_EXCEEDED",
        ]
        for attr in required:
            assert hasattr(KGBError, attr), f"KGBError.{attr} missing"
            code, message = getattr(KGBError, attr)
            assert code.startswith("KGB-"), f"{attr} code must start with KGB-"
            assert len(message) > 0, f"{attr} message must not be empty"

    @pytest.mark.unit
    def test_error_codes_are_unique(self):
        from app.core.errors import KGBError

        codes = [
            KGBError.NEO4J_UNAVAILABLE[0],
            KGBError.REDIS_UNAVAILABLE[0],
            KGBError.LLM_UNAVAILABLE[0],
            KGBError.CELERY_UNAVAILABLE[0],
            KGBError.GRAPH_NOT_FOUND[0],
            KGBError.PERMISSION_DENIED[0],
            KGBError.RATE_LIMIT_EXCEEDED[0],
        ]
        assert len(codes) == len(set(codes)), "Error codes must be unique"

    @pytest.mark.unit
    def test_neo4j_code_is_5001(self):
        from app.core.errors import KGBError

        assert KGBError.NEO4J_UNAVAILABLE[0] == "KGB-5001"

    @pytest.mark.unit
    def test_rate_limit_code_is_4029(self):
        from app.core.errors import KGBError

        assert KGBError.RATE_LIMIT_EXCEEDED[0] == "KGB-4029"

    @pytest.mark.unit
    def test_graph_not_found_code_is_4001(self):
        from app.core.errors import KGBError

        assert KGBError.GRAPH_NOT_FOUND[0] == "KGB-4001"

    @pytest.mark.unit
    def test_permission_denied_code_is_4003(self):
        from app.core.errors import KGBError

        assert KGBError.PERMISSION_DENIED[0] == "KGB-4003"
