"""Unit tests for SA security audit log (ORA-316).

6 required cases:
1. audit write called on SA create
2. audit write called on key rotate
3. audit write called on SA revoke
4. raw key never in audit params
5. tenant isolation enforced in Cypher
6. audit errors propagate (not swallowed)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.audit_service import log_sa_security_event
from app.services.service_account_service import ServiceAccountService

# ── Constants ──────────────────────────────────────────────────────────────

SA_ID = "sa-uuid-1234"
TENANT_ID = "tenant-uuid-5678"
GRAPH_ID = "graph-uuid-abcd"
USER_ID = "user-uuid-efgh"
KEY_PREFIX = "osk_abc1"
RAW_KEY = "osk_abc1_supersecretrawkey"
BCRYPT_HASH = "$2b$12$thisisabcrypthash"


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_session(single_return=None, data_return=None):
    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value=single_return)
    mock_result.data = AsyncMock(return_value=data_return or [])

    session = AsyncMock()
    session.run = AsyncMock(return_value=mock_result)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


def _make_driver(single_return=None, data_return=None):
    session = _make_session(single_return=single_return, data_return=data_return)
    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    return driver, session


# ── Test 1: audit write is called on SA create ─────────────────────────────


@pytest.mark.asyncio
async def test_audit_called_on_create():
    """log_sa_security_event must be invoked with event_type=service_account.created."""
    service = ServiceAccountService()
    driver, session = _make_driver()

    key_data = {"key_prefix": KEY_PREFIX, "api_key": RAW_KEY}

    with (
        patch.object(service, "_create_auth_key", return_value=key_data),
        patch(
            "app.services.service_account_service.log_sa_security_event",
            new_callable=AsyncMock,
        ) as mock_audit,
    ):
        await service.create_service_account(
            driver=driver,
            tenant_id=TENANT_ID,
            graph_id=GRAPH_ID,
            created_by_user_id=USER_ID,
            name="test-sa",
        )

    mock_audit.assert_called_once()
    call_kwargs = mock_audit.call_args.kwargs
    assert call_kwargs["event_type"] == "service_account.created"
    assert call_kwargs["sa_id"] is not None
    assert call_kwargs["tenant_id"] == TENANT_ID


# ── Test 2: audit write is called on key rotate ────────────────────────────


@pytest.mark.asyncio
async def test_audit_called_on_rotate():
    """log_sa_security_event must be invoked with event_type=service_account.key_rotated."""
    service = ServiceAccountService()

    sa_record = {
        "service_account_id": SA_ID,
        "name": "test-sa",
        "description": "",
        "home_graph_id": GRAPH_ID,
        "tenant_id": TENANT_ID,
        "status": "active",
        "key_prefix": KEY_PREFIX,
        "created_at": "2026-01-01T00:00:00+00:00",
        "last_used_at": None,
    }
    driver, session = _make_driver(single_return=sa_record)
    key_data = {"key_prefix": "osk_new1", "api_key": RAW_KEY}

    with (
        patch.object(service, "get_service_account", return_value=sa_record),
        patch.object(service, "_revoke_auth_keys", new_callable=AsyncMock),
        patch.object(service, "_create_auth_key", return_value=key_data),
        patch(
            "app.services.service_account_service.log_sa_security_event",
            new_callable=AsyncMock,
        ) as mock_audit,
    ):
        await service.rotate_key(
            driver=driver,
            sa_id=SA_ID,
            tenant_id=TENANT_ID,
            created_by_user_id=USER_ID,
        )

    mock_audit.assert_called_once()
    call_kwargs = mock_audit.call_args.kwargs
    assert call_kwargs["event_type"] == "service_account.key_rotated"
    assert call_kwargs["key_prefix"] == "osk_new1"


# ── Test 3: audit write is called on SA revoke ─────────────────────────────


@pytest.mark.asyncio
async def test_audit_called_on_revoke():
    """log_sa_security_event must be invoked with event_type=service_account.revoked."""
    service = ServiceAccountService()

    revoke_record = {"sa_id": SA_ID, "home_graph_id": GRAPH_ID}
    driver, session = _make_driver(single_return=revoke_record)

    with (
        patch.object(service, "_revoke_auth_keys", new_callable=AsyncMock),
        patch(
            "app.services.service_account_service.log_sa_security_event",
            new_callable=AsyncMock,
        ) as mock_audit,
    ):
        result = await service.revoke_service_account(
            driver=driver,
            sa_id=SA_ID,
            tenant_id=TENANT_ID,
            actor_user_id=USER_ID,
        )

    assert result is True
    mock_audit.assert_called_once()
    call_kwargs = mock_audit.call_args.kwargs
    assert call_kwargs["event_type"] == "service_account.revoked"


# ── Test 4: raw API key / bcrypt hash never in audit params ───────────────


@pytest.mark.asyncio
async def test_no_raw_key_in_audit_params():
    """The audit call must never receive the raw api_key or a bcrypt hash as key_prefix."""
    service = ServiceAccountService()
    driver, session = _make_driver()

    key_data = {"key_prefix": KEY_PREFIX, "api_key": RAW_KEY}

    with (
        patch.object(service, "_create_auth_key", return_value=key_data),
        patch(
            "app.services.service_account_service.log_sa_security_event",
            new_callable=AsyncMock,
        ) as mock_audit,
    ):
        await service.create_service_account(
            driver=driver,
            tenant_id=TENANT_ID,
            graph_id=GRAPH_ID,
            created_by_user_id=USER_ID,
            name="test-sa",
        )

    call_kwargs = mock_audit.call_args.kwargs
    # key_prefix must not be the full raw API key or a bcrypt hash
    assert call_kwargs.get("key_prefix") != RAW_KEY
    assert call_kwargs.get("key_prefix") != BCRYPT_HASH
    # key_prefix may be None or the short prefix string
    kp = call_kwargs.get("key_prefix")
    assert kp is None or kp == KEY_PREFIX


# ── Test 5: tenant isolation in Cypher ────────────────────────────────────


@pytest.mark.asyncio
async def test_tenant_isolation_in_cypher():
    """log_sa_security_event Cypher must include tenant_id in the WHERE clause."""
    session = AsyncMock()
    session.run = AsyncMock(return_value=AsyncMock())

    await log_sa_security_event(
        session=session,
        event_type="service_account.created",
        sa_id=SA_ID,
        actor_user_id=USER_ID,
        home_graph_id=GRAPH_ID,
        tenant_id=TENANT_ID,
        key_prefix=KEY_PREFIX,
    )

    session.run.assert_called_once()
    call_args = session.run.call_args
    cypher = call_args.args[0] if call_args.args else call_args[0][0]
    params = call_args.args[1] if len(call_args.args) > 1 else call_args[0][1]

    # Cypher must reference tenant_id to enforce isolation
    assert "tenant_id" in cypher
    assert params["tenant_id"] == TENANT_ID


# ── Test 6: audit errors propagate (not swallowed) ────────────────────────


@pytest.mark.asyncio
async def test_audit_errors_propagate():
    """Exceptions in log_sa_security_event must not be caught — they must bubble up."""
    session = AsyncMock()
    session.run = AsyncMock(side_effect=RuntimeError("Neo4j write failed"))

    with pytest.raises(RuntimeError, match="Neo4j write failed"):
        await log_sa_security_event(
            session=session,
            event_type="service_account.created",
            sa_id=SA_ID,
            actor_user_id=USER_ID,
            home_graph_id=GRAPH_ID,
            tenant_id=TENANT_ID,
        )
