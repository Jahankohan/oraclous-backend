"""Unit tests for the /service-token rate limiting logic.

Tests cover:
- enforce_key_prefix_rate_limit: blocks on the 11th request within a window
- enforce_key_prefix_rate_limit: allows requests when under the limit
- enforce_key_prefix_rate_limit: fails open when Redis is unavailable
- enforce_key_prefix_rate_limit: skips check when api_key is missing/empty
- key_prefix extracted correctly (first 12 chars)
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException


def _make_request(api_key: str = "osk_AbCdEfGh1234XYZ", redis_client=None, missing_redis=False):
    """Build a mock Starlette Request with the given api_key body and Redis state."""
    body = json.dumps({"api_key": api_key}).encode()

    request = MagicMock()
    request.body = AsyncMock(return_value=body)

    app_state = MagicMock()
    if missing_redis:
        app_state.redis = None
    else:
        app_state.redis = redis_client or AsyncMock()

    request.app.state = app_state
    return request


@pytest.mark.asyncio
@pytest.mark.unit
async def test_allows_requests_under_limit():
    """Requests below the threshold (count <= 10) are allowed."""
    from app.core.rate_limiter import enforce_key_prefix_rate_limit

    redis_mock = AsyncMock()
    redis_mock.incr = AsyncMock(return_value=5)  # 5th request — under limit
    redis_mock.expire = AsyncMock()

    request = _make_request(redis_client=redis_mock)
    # Should complete without raising
    await enforce_key_prefix_rate_limit(request)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_blocks_on_eleventh_request():
    """The 11th request for the same key_prefix within the window returns 429."""
    from app.core.rate_limiter import enforce_key_prefix_rate_limit

    redis_mock = AsyncMock()
    redis_mock.incr = AsyncMock(return_value=11)  # Over limit
    redis_mock.expire = AsyncMock()
    redis_mock.ttl = AsyncMock(return_value=45)

    request = _make_request(redis_client=redis_mock)
    with pytest.raises(HTTPException) as exc_info:
        await enforce_key_prefix_rate_limit(request)

    assert exc_info.value.status_code == 429
    assert "Retry-After" in exc_info.value.headers
    assert exc_info.value.headers["Retry-After"] == "45"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_retry_after_minimum_is_one_second():
    """Retry-After header is at least 1 even when TTL returns 0 or negative."""
    from app.core.rate_limiter import enforce_key_prefix_rate_limit

    redis_mock = AsyncMock()
    redis_mock.incr = AsyncMock(return_value=15)
    redis_mock.expire = AsyncMock()
    redis_mock.ttl = AsyncMock(return_value=0)

    request = _make_request(redis_client=redis_mock)
    with pytest.raises(HTTPException) as exc_info:
        await enforce_key_prefix_rate_limit(request)

    assert exc_info.value.headers["Retry-After"] == "1"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fails_open_when_redis_unavailable():
    """When Redis is None (not configured), the check is skipped — fail open."""
    from app.core.rate_limiter import enforce_key_prefix_rate_limit

    request = _make_request(missing_redis=True)
    # Should NOT raise even without Redis
    await enforce_key_prefix_rate_limit(request)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fails_open_on_redis_error():
    """If Redis raises an unexpected error, the request is allowed through."""
    from app.core.rate_limiter import enforce_key_prefix_rate_limit

    redis_mock = AsyncMock()
    redis_mock.incr = AsyncMock(side_effect=ConnectionError("Redis down"))

    request = _make_request(redis_client=redis_mock)
    # Should NOT raise — fail open behavior
    await enforce_key_prefix_rate_limit(request)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_skips_check_for_empty_api_key():
    """If api_key is missing or empty, the prefix check is skipped entirely."""
    from app.core.rate_limiter import enforce_key_prefix_rate_limit

    redis_mock = AsyncMock()
    redis_mock.incr = AsyncMock()

    request = _make_request(api_key="", redis_client=redis_mock)
    await enforce_key_prefix_rate_limit(request)

    # Redis should not have been touched
    redis_mock.incr.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_key_prefix_is_first_12_chars():
    """Redis key uses exactly the first 12 chars of the api_key as the prefix."""
    from app.core.rate_limiter import enforce_key_prefix_rate_limit

    redis_mock = AsyncMock()
    redis_mock.incr = AsyncMock(return_value=1)
    redis_mock.expire = AsyncMock()

    api_key = "osk_AbCdEfGh1234_extra_suffix"
    request = _make_request(api_key=api_key, redis_client=redis_mock)
    await enforce_key_prefix_rate_limit(request)

    expected_prefix = api_key[:12]  # "osk_AbCdEfGh"
    redis_mock.incr.assert_awaited_once_with(f"rl:pfx:{expected_prefix}")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_expiry_set_only_on_first_request():
    """expire() is called only when incr() returns 1 (first request in window)."""
    from app.core.rate_limiter import enforce_key_prefix_rate_limit

    redis_mock = AsyncMock()
    redis_mock.expire = AsyncMock()

    # Second request in window — expire should NOT be set again
    redis_mock.incr = AsyncMock(return_value=2)
    request = _make_request(redis_client=redis_mock)
    await enforce_key_prefix_rate_limit(request)
    redis_mock.expire.assert_not_awaited()

    # First request — expire SHOULD be set
    redis_mock.incr = AsyncMock(return_value=1)
    request = _make_request(redis_client=redis_mock)
    await enforce_key_prefix_rate_limit(request)
    redis_mock.expire.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_skips_check_on_malformed_body():
    """Malformed JSON body does not raise — treated as empty key, skipped."""
    from app.core.rate_limiter import enforce_key_prefix_rate_limit

    redis_mock = AsyncMock()
    redis_mock.incr = AsyncMock()

    request = MagicMock()
    request.body = AsyncMock(return_value=b"not valid json")
    request.app.state.redis = redis_mock

    await enforce_key_prefix_rate_limit(request)
    redis_mock.incr.assert_not_called()
