"""Unit tests for IntegrationKeyService (STORY-022 / TASK-044)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.integration_key_service import IntegrationKeyService


def _mock_driver(records=None):
    driver = MagicMock()
    result = MagicMock()
    result.records = records or []
    driver.execute_query = AsyncMock(return_value=result)
    return driver


class TestKeyUtilities:
    def test_generate_key_starts_with_oak_prefix(self):
        key = IntegrationKeyService.generate_key()
        assert key.startswith("oak-")

    def test_generate_key_has_sufficient_length(self):
        key = IntegrationKeyService.generate_key()
        assert len(key) > 36  # "oak-" + 43 base64url chars from 32 bytes

    def test_generate_key_is_unique(self):
        keys = {IntegrationKeyService.generate_key() for _ in range(100)}
        assert len(keys) == 100

    def test_hash_and_verify_round_trip(self):
        key = IntegrationKeyService.generate_key()
        key_hash = IntegrationKeyService.hash_key(key)
        assert IntegrationKeyService.verify_key(key, key_hash) is True

    def test_verify_wrong_key_returns_false(self):
        key = IntegrationKeyService.generate_key()
        key_hash = IntegrationKeyService.hash_key(key)
        assert IntegrationKeyService.verify_key("oak-wrongkey", key_hash) is False

    def test_hash_is_deterministic(self):
        key = "oak-testkey"
        assert IntegrationKeyService.hash_key(key) == IntegrationKeyService.hash_key(key)

    def test_hash_never_stores_plaintext(self):
        key = "oak-supersecret"
        key_hash = IntegrationKeyService.hash_key(key)
        assert "supersecret" not in key_hash


class TestPublishAgent:
    async def test_writes_key_hash_not_plaintext(self):
        driver = _mock_driver()
        # Make the first execute_query (publish) return a record, second (check) returns nothing
        result_with_record = MagicMock()
        result_with_record.records = [MagicMock()]
        driver.execute_query = AsyncMock(return_value=result_with_record)

        svc = IntegrationKeyService(driver)
        key, slug = await svc.publish_agent(
            agent_id="a1", graph_id="g1", org_id="org1", user_id="u1",
            slug="test-slug", cors_origins=[], rate_limit_rpm=60,
        )

        assert key.startswith("oak-")
        query, params = driver.execute_query.call_args.args
        assert "key_hash" in query
        assert params.get("key_hash") == IntegrationKeyService.hash_key(key)
        assert "key" not in [k for k in params if k not in ("key_hash", "key_last4")]

    async def test_raises_on_slug_conflict(self):
        driver = MagicMock()
        # First call (publish attempt) returns empty — slug conflict
        empty = MagicMock()
        empty.records = []
        # Second call (conflict check) returns a record — slug taken
        conflict = MagicMock()
        conflict.records = [MagicMock()]
        driver.execute_query = AsyncMock(side_effect=[empty, conflict])

        svc = IntegrationKeyService(driver)
        with pytest.raises(ValueError, match="already taken"):
            await svc.publish_agent(
                agent_id="a1", graph_id="g1", org_id="org1", user_id="u1",
                slug="taken-slug", cors_origins=[],
            )


class TestUnpublishAgent:
    async def test_returns_true_when_found(self):
        driver = _mock_driver(records=[MagicMock()])
        svc = IntegrationKeyService(driver)
        result = await svc.unpublish_agent("a1", "g1")
        assert result is True
        query, params = driver.execute_query.call_args.args
        assert "unpublished_at" in query
        assert params["agent_id"] == "a1"

    async def test_returns_false_when_not_found(self):
        driver = _mock_driver(records=[])
        svc = IntegrationKeyService(driver)
        result = await svc.unpublish_agent("a1", "g1")
        assert result is False


class TestRotateKey:
    async def test_new_key_updates_hash_and_last4(self):
        driver = _mock_driver(records=[MagicMock()])
        svc = IntegrationKeyService(driver)
        new_key = await svc.rotate_key("a1", "g1")

        assert new_key.startswith("oak-")
        query, params = driver.execute_query.call_args.args
        assert params["key_hash"] == IntegrationKeyService.hash_key(new_key)
        assert params["key_last4"] == new_key[-4:]

    async def test_raises_when_agent_not_published(self):
        driver = _mock_driver(records=[])
        svc = IntegrationKeyService(driver)
        with pytest.raises(ValueError, match="No active published agent"):
            await svc.rotate_key("a1", "g1")


class TestValidateKey:
    async def test_returns_dict_on_correct_key(self):
        key = IntegrationKeyService.generate_key()
        key_hash = IntegrationKeyService.hash_key(key)

        published = {
            "agent_id": "a1", "graph_id": "g1", "slug": "test-slug",
            "key_hash": key_hash, "key_last4": key[-4:],
            "cors_origins": [], "rate_limit_rpm": 60,
            "egress_url": None, "published_at": 1000, "unpublished_at": None,
        }

        driver = MagicMock()
        result = MagicMock()
        rec = MagicMock()
        rec.__getitem__ = lambda self, k: published if k == "p" else None
        result.records = [rec]
        driver.execute_query = AsyncMock(return_value=result)

        svc = IntegrationKeyService(driver)
        svc.get_published = AsyncMock(return_value=published)
        found = await svc.validate_key("test-slug", key)
        assert found is not None
        assert found["agent_id"] == "a1"

    async def test_returns_none_on_wrong_key(self):
        key = IntegrationKeyService.generate_key()
        key_hash = IntegrationKeyService.hash_key(key)
        published = {"key_hash": key_hash, "key_last4": key[-4:]}

        svc = IntegrationKeyService(MagicMock())
        svc.get_published = AsyncMock(return_value=published)
        result = await svc.validate_key("test-slug", "oak-wrongkey")
        assert result is None

    async def test_returns_none_when_slug_not_found(self):
        svc = IntegrationKeyService(MagicMock())
        svc.get_published = AsyncMock(return_value=None)
        result = await svc.validate_key("no-such-slug", "oak-anykey")
        assert result is None
