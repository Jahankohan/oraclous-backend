"""Unit tests for audit_service.log_public_call (STORY-022 / TASK-044)."""

import hashlib
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from app.services.audit_service import log_public_call


def _mock_driver():
    driver = MagicMock()
    driver.execute_query = AsyncMock(return_value=MagicMock())
    return driver


class TestLogPublicCall:
    async def test_writes_audit_event_node(self):
        driver = _mock_driver()
        await log_public_call(driver, "g1", "a1", "ab12", "hello", "world")
        driver.execute_query.assert_called_once()

    async def test_stores_sha256_input_hash_not_plaintext(self):
        driver = _mock_driver()
        input_text = "secret user question"
        await log_public_call(driver, "g1", "a1", "ab12", input_text, "response")

        _, params = driver.execute_query.call_args.args
        assert params["input_hash"] == hashlib.sha256(input_text.encode()).hexdigest()
        assert input_text not in params.values()

    async def test_stores_sha256_response_hash_not_plaintext(self):
        driver = _mock_driver()
        response_text = "sensitive agent response"
        await log_public_call(driver, "g1", "a1", "ab12", "question", response_text)

        _, params = driver.execute_query.call_args.args
        assert params["response_hash"] == hashlib.sha256(response_text.encode()).hexdigest()
        assert response_text not in params.values()

    async def test_stores_key_last4_not_full_key(self):
        driver = _mock_driver()
        await log_public_call(driver, "g1", "a1", "ab12", "q", "r")

        _, params = driver.execute_query.call_args.args
        assert params["key_last4"] == "ab12"
        # No full key stored; last4 is only 4 chars
        assert len(params["key_last4"]) == 4

    async def test_includes_graph_id_for_tenant_scoping(self):
        driver = _mock_driver()
        await log_public_call(driver, "graph-tenant-x", "a1", "ab12", "q", "r")

        _, params = driver.execute_query.call_args.args
        assert params["graph_id"] == "graph-tenant-x"

    async def test_includes_agent_id(self):
        driver = _mock_driver()
        await log_public_call(driver, "g1", "agent-99", "ab12", "q", "r")

        _, params = driver.execute_query.call_args.args
        assert params["agent_id"] == "agent-99"

    async def test_cypher_creates_audit_event_node(self):
        driver = _mock_driver()
        await log_public_call(driver, "g1", "a1", "ab12", "q", "r")

        query, _ = driver.execute_query.call_args.args
        assert "AuditEvent" in query
        assert "CREATE" in query

    async def test_does_not_raise_on_driver_error(self):
        driver = MagicMock()
        driver.execute_query = AsyncMock(side_effect=Exception("Neo4j down"))

        # Should not propagate the exception
        await log_public_call(driver, "g1", "a1", "ab12", "q", "r")

    async def test_each_call_generates_unique_event_id(self):
        driver = _mock_driver()
        await log_public_call(driver, "g1", "a1", "ab12", "q", "r")
        params1 = driver.execute_query.call_args.args[1]

        driver.execute_query.reset_mock()
        await log_public_call(driver, "g1", "a1", "ab12", "q", "r")
        params2 = driver.execute_query.call_args.args[1]

        assert params1["event_id"] != params2["event_id"]

    async def test_hash_uses_sha256_not_md5(self):
        driver = _mock_driver()
        input_text = "test input"
        await log_public_call(driver, "g1", "a1", "ab12", input_text, "r")

        _, params = driver.execute_query.call_args.args
        # SHA-256 hex digest is 64 chars; MD5 is 32 chars
        assert len(params["input_hash"]) == 64
        assert len(params["response_hash"]) == 64
