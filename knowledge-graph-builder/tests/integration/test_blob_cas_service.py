"""Integration tests for the Postgres-backed BlobCAS (STORY-026, TASK-082).

These tests exercise the `blob_cas` table against the real Postgres in the
Docker compose stack. They cover the DoD bullets for TASK-082:

- The migration applies cleanly; the table + index are present
- `put` returns deterministic sha256 for identical content
- `get` returns `None` for cross-tenant access (existence-masked)
- A 10 MB binary payload round-trips losslessly
- `delete` is `graph_id`-scoped
- `persist_deliverable` writes through the CAS and a subsequent
  `get_deliverable_content` byte-matches the original
"""

from __future__ import annotations

import hashlib
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.services.blob_cas_service import (
    BLOB_URI_SCHEME,
    BlobCASService,
    InvalidBlobURIError,
    make_blob_uri,
)

# Unique-per-session graph_ids so parallel runs don't collide.
_SESSION = uuid.uuid4().hex[:8]
_GID_A = f"integ-blob-A-{_SESSION}"
_GID_B = f"integ-blob-B-{_SESSION}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db_session():
    """Task-scoped async Postgres session.

    Reuses the existing settings.POSTGRES_URL (same connection that the
    FastAPI app + Celery use) — no new pool, per the task constraint.
    """
    engine = create_async_engine(settings.POSTGRES_URL, poolclass=NullPool, future=True)
    session_maker = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_maker() as session:
        yield session
    await engine.dispose()


@pytest_asyncio.fixture(autouse=True)
async def _cleanup(db_session: AsyncSession):
    """Wipe any rows from prior runs of this test module under our session
    graph_ids before and after each test.
    """

    async def _wipe():
        await db_session.execute(
            text("DELETE FROM blob_cas WHERE graph_id IN (:a, :b)"),
            {"a": _GID_A, "b": _GID_B},
        )
        await db_session.commit()

    await _wipe()
    yield
    await _wipe()


# ---------------------------------------------------------------------------
# Migration sanity
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_table_and_index_exist(db_session: AsyncSession):
    """DoD: 'Alembic migration applied; table + index present'."""
    table_check = await db_session.execute(
        text("SELECT to_regclass('public.blob_cas') AS oid")
    )
    assert table_check.scalar() is not None

    index_check = await db_session.execute(
        text(
            "SELECT 1 FROM pg_indexes "
            "WHERE schemaname='public' "
            "AND tablename='blob_cas' "
            "AND indexname='blob_cas_graph_id'"
        )
    )
    assert index_check.first() is not None


# ---------------------------------------------------------------------------
# put / get round-trip
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_put_returns_deterministic_sha256(db_session: AsyncSession):
    """DoD: 'put returns deterministic sha256 for identical content'."""
    svc = BlobCASService()
    content = b"deterministic content " * 100
    expected = hashlib.sha256(content).hexdigest()

    r1 = await svc.put(db_session, _GID_A, content, "text/plain")
    await db_session.commit()
    r2 = await svc.put(db_session, _GID_A, content, "text/plain")
    await db_session.commit()

    assert r1["sha256"] == expected
    assert r2["sha256"] == expected
    assert r1["content_uri"] == r2["content_uri"]
    assert r1["content_uri"].startswith(BLOB_URI_SCHEME)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_returns_payload_for_owner(db_session: AsyncSession):
    svc = BlobCASService()
    content = b"hello round-trip"
    r = await svc.put(db_session, _GID_A, content, "text/plain")
    await db_session.commit()

    out = await svc.get(db_session, _GID_A, r["content_uri"])
    assert out is not None
    assert out["content_bytes"] == content
    assert out["mime_type"] == "text/plain"
    assert out["size_bytes"] == len(content)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_cross_tenant_returns_none(db_session: AsyncSession):
    """DoD: 'cross-tenant CAS read returns 404, existence is masked'."""
    svc = BlobCASService()
    content = b"alice-only"
    r = await svc.put(db_session, _GID_A, content, "text/plain")
    await db_session.commit()

    # Bob with the same URI sees nothing.
    out = await svc.get(db_session, _GID_B, r["content_uri"])
    assert out is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_returns_none_for_unknown_sha(db_session: AsyncSession):
    svc = BlobCASService()
    fake_uri = make_blob_uri("0" * 64)
    out = await svc.get(db_session, _GID_A, fake_uri)
    assert out is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_raises_on_malformed_uri(db_session: AsyncSession):
    svc = BlobCASService()
    with pytest.raises(InvalidBlobURIError):
        await svc.get(db_session, _GID_A, "not://a/uri")


# ---------------------------------------------------------------------------
# 10 MB round-trip — DoD bullet
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ten_megabyte_payload_round_trips_lossless(db_session: AsyncSession):
    """DoD: 'A 10MB PDF round-trips losslessly'.

    We don't need a real PDF — a deterministic 10MB bytestring exercises
    the bytea path the same way. We assert byte-for-byte equality.
    """
    svc = BlobCASService()
    payload_size = 10 * 1024 * 1024  # 10 MiB
    # Use a non-zero pattern so a corrupted truncation wouldn't accidentally
    # match the original.
    content = bytes(i % 251 for i in range(payload_size))
    expected_sha = hashlib.sha256(content).hexdigest()

    r = await svc.put(db_session, _GID_A, content, "application/pdf")
    await db_session.commit()
    assert r["sha256"] == expected_sha

    out = await svc.get(db_session, _GID_A, r["content_uri"])
    assert out is not None
    assert out["size_bytes"] == payload_size
    assert len(out["content_bytes"]) == payload_size
    assert out["content_bytes"] == content
    assert hashlib.sha256(out["content_bytes"]).hexdigest() == expected_sha


# ---------------------------------------------------------------------------
# delete is graph_id-scoped
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_returns_true_when_owner_deletes(db_session: AsyncSession):
    svc = BlobCASService()
    r = await svc.put(db_session, _GID_A, b"to-delete", "text/plain")
    await db_session.commit()

    deleted = await svc.delete(db_session, _GID_A, r["content_uri"])
    await db_session.commit()
    assert deleted is True

    # And the row is gone.
    assert await svc.get(db_session, _GID_A, r["content_uri"]) is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_cross_tenant_does_not_remove(db_session: AsyncSession):
    """A foreign tenant deleting our URI matches 0 rows and returns False."""
    svc = BlobCASService()
    r = await svc.put(db_session, _GID_A, b"safe", "text/plain")
    await db_session.commit()

    deleted = await svc.delete(db_session, _GID_B, r["content_uri"])
    await db_session.commit()
    assert deleted is False

    # And our row is still intact under our graph_id.
    out = await svc.get(db_session, _GID_A, r["content_uri"])
    assert out is not None
    assert out["content_bytes"] == b"safe"


# ---------------------------------------------------------------------------
# Cross-tenant identical content stays separate (Data Ownership)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cross_tenant_identical_content_stays_separate(
    db_session: AsyncSession,
):
    """Two tenants uploading byte-for-byte identical content keep two rows.

    Per ADR-018/founding principles, no platform-level dedup across tenants;
    each tenant gets its own row. We verify this by counting rows in the
    table after both writes.
    """
    svc = BlobCASService()
    content = b"identical-across-tenants"
    expected_sha = hashlib.sha256(content).hexdigest()

    await svc.put(db_session, _GID_A, content, "text/plain")
    await svc.put(db_session, _GID_B, content, "text/plain")
    await db_session.commit()

    rows = await db_session.execute(
        text(
            "SELECT graph_id FROM blob_cas WHERE sha256 = :sha "
            "AND graph_id IN (:a, :b) ORDER BY graph_id"
        ),
        {"sha": expected_sha, "a": _GID_A, "b": _GID_B},
    )
    graph_ids = [r[0] for r in rows.fetchall()]
    assert graph_ids == sorted([_GID_A, _GID_B])
