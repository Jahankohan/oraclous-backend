"""Unit tests for the /service-token rate limiting logic.

Tests cover:
- enforce_key_prefix_rate_limit: blocks on the 11th request within a window
- enforce_key_prefix_rate_limit: allows requests when under the limit
- enforce_key_prefix_rate_limit: fails open when Redis is unavailable
- enforce_key_prefix_rate_limit: skips check when api_key is missing/empty
- key_prefix extracted correctly (first 12 chars)
- INCR+EXPIRE are executed atomically via pipeline (race-condition fix)
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import HTTPException


def _make_pipeline_request(
    incr_result: int,
    ttl_result: int = 45,
    api_key: str = "osk_AbCdEfGh1234XYZ",
    pipeline_error=None,
):
    """Build a mock Starlette Request whose Redis client uses a pipeline.

    Returns (request, pipe_mock, redis_mock) so callers can assert on
    individual pipeline calls.
    """
    body = json.dumps({"api_key": api_key}).encode()

    request = MagicMock()
    request.body = AsyncMock(return_value=body)

    pipe_mock = AsyncMock()
    pipe_mock.incr = AsyncMock(return_value=None)   # queued — no direct result
    pipe_mock.expire = AsyncMock(return_value=None)  # queued — no direct result
    if pipeline_error:
        pipe_mock.execute = AsyncMock(side_effect=pipeline_error)
    else:
        pipe_mock.execute = AsyncMock(return_value=[incr_result, True])

    pipeline_ctx = MagicMock()
    pipeline_ctx.__aenter__ = AsyncMock(return_value=pipe_mock)
    pipeline_ctx.__aexit__ = AsyncMock(return_value=None)

    redis_mock = AsyncMock()
    redis_mock.pipeline = MagicMock(return_value=pipeline_ctx)
    redis_mock.ttl = AsyncMock(return_value=ttl_result)

    app_state = MagicMock()
    app_state.redis = redis_mock
    request.app.state = app_state

    return request, pipe_mock, redis_mock


def _make_no_redis_request(api_key: str = "osk_AbCdEfGh1234XYZ"):
    """Build a mock Request with no Redis client attached."""
    body = json.dumps({"api_key": api_key}).encode()
    request = MagicMock()
    request.body = AsyncMock(return_value=body)
    app_state = MagicMock()
    app_state.redis = None
    request.app.state = app_state
    return request


@pytest.mark.asyncio
@pytest.mark.unit
async def test_allows_requests_under_limit():
    """Requests below the threshold (count <= 10) are allowed."""
    from app.core.rate_limiter import enforce_key_prefix_rate_limit

    request, pipe_mock, _ = _make_pipeline_request(incr_result=5)
    # Should complete without raising
    await enforce_key_prefix_rate_limit(request)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_blocks_on_eleventh_request():
    """The 11th request for the same key_prefix within the window returns 429."""
    from app.core.rate_limiter import enforce_key_prefix_rate_limit

    request, _, _ = _make_pipeline_request(incr_result=11, ttl_result=45)
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

    request, _, _ = _make_pipeline_request(incr_result=15, ttl_result=0)
    with pytest.raises(HTTPException) as exc_info:
        await enforce_key_prefix_rate_limit(request)

    assert exc_info.value.headers["Retry-After"] == "1"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fails_open_when_redis_unavailable():
    """When Redis is None (not configured), the check is skipped — fail open."""
    from app.core.rate_limiter import enforce_key_prefix_rate_limit

    request = _make_no_redis_request()
    # Should NOT raise even without Redis
    await enforce_key_prefix_rate_limit(request)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fails_open_on_redis_error():
    """If the pipeline raises an unexpected error, the request is allowed through."""
    from app.core.rate_limiter import enforce_key_prefix_rate_limit

    request, _, _ = _make_pipeline_request(
        incr_result=1,
        pipeline_error=ConnectionError("Redis down"),
    )
    # Should NOT raise — fail open behavior
    await enforce_key_prefix_rate_limit(request)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_skips_check_for_empty_api_key():
    """If api_key is missing or empty, the prefix check is skipped entirely."""
    from app.core.rate_limiter import enforce_key_prefix_rate_limit

    request, pipe_mock, redis_mock = _make_pipeline_request(
        incr_result=1, api_key=""
    )
    await enforce_key_prefix_rate_limit(request)

    # Pipeline should not have been touched
    redis_mock.pipeline.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_key_prefix_is_first_12_chars():
    """Redis pipeline uses exactly the first 12 chars of the api_key as the key."""
    from app.core.rate_limiter import enforce_key_prefix_rate_limit

    api_key = "osk_AbCdEfGh1234_extra_suffix"
    request, pipe_mock, _ = _make_pipeline_request(
        incr_result=1, api_key=api_key
    )
    await enforce_key_prefix_rate_limit(request)

    expected_prefix = api_key[:12]  # "osk_AbCdEfGh"
    pipe_mock.incr.assert_awaited_once_with(f"rl:pfx:{expected_prefix}")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_incr_and_expire_both_queued_in_pipeline():
    """INCR and EXPIRE are both queued in the same pipeline (atomic execution)."""
    from app.core.rate_limiter import enforce_key_prefix_rate_limit

    request, pipe_mock, _ = _make_pipeline_request(incr_result=1)
    await enforce_key_prefix_rate_limit(request)

    # Both commands must be queued and execute() must be called exactly once
    pipe_mock.incr.assert_awaited_once()
    pipe_mock.expire.assert_awaited_once()
    pipe_mock.execute.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_skips_check_on_malformed_body():
    """Malformed JSON body does not raise — treated as empty key, skipped."""
    from app.core.rate_limiter import enforce_key_prefix_rate_limit

    request = MagicMock()
    request.body = AsyncMock(return_value=b"not valid json")
    redis_mock = AsyncMock()
    request.app.state.redis = redis_mock

    await enforce_key_prefix_rate_limit(request)
    redis_mock.pipeline.assert_not_called()
