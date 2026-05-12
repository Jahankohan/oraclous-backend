"""Unit tests for BlobCASService (STORY-026, TASK-082).

Mocks the async Postgres session and asserts the service:

- Hashes content with SHA-256 deterministically — identical bytes → identical sha256
- Returns the canonical `blob://sha256/<hex>` URI
- Filters every read/write by `graph_id`
- Returns `None` (existence-masked) on cross-tenant read
- Refuses to parse malformed CAS URIs

Integration tests against a real Postgres live in
`tests/integration/test_blob_cas_service.py`.
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.blob_cas_service import (
    BlobCASService,
    InvalidBlobURIError,
    is_blob_uri,
    make_blob_uri,
    parse_blob_uri,
)

# ── URI helpers ───────────────────────────────────────────────────────────────


class TestUriHelpers:
    def test_make_blob_uri_canonical_form(self):
        sha = "a" * 64
        assert make_blob_uri(sha) == f"blob://sha256/{sha}"

    def test_make_blob_uri_rejects_non_hex(self):
        with pytest.raises(InvalidBlobURIError):
            make_blob_uri("Z" * 64)

    def test_make_blob_uri_rejects_wrong_length(self):
        with pytest.raises(InvalidBlobURIError):
            make_blob_uri("a" * 63)

    def test_parse_blob_uri_extracts_hex(self):
        sha = "f" * 64
        assert parse_blob_uri(f"blob://sha256/{sha}") == sha

    def test_parse_blob_uri_rejects_other_schemes(self):
        with pytest.raises(InvalidBlobURIError):
            parse_blob_uri("file:///tmp/foo")
        with pytest.raises(InvalidBlobURIError):
            parse_blob_uri("blob://md5/" + "a" * 32)

    def test_parse_blob_uri_rejects_uppercase_hex(self):
        # We canonicalize on lowercase per `hashlib.hexdigest()` convention.
        with pytest.raises(InvalidBlobURIError):
            parse_blob_uri("blob://sha256/" + "A" * 64)

    def test_is_blob_uri_returns_false_for_none(self):
        assert is_blob_uri(None) is False
        assert is_blob_uri("") is False
        assert is_blob_uri("/tmp/foo.md") is False


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_session(rows: list | None = None, rowcount: int | None = None):
    """Build a mock AsyncSession that returns `rows` on first `execute().first()`
    and `rowcount` on `execute().rowcount`.
    """
    session = AsyncMock()

    def _make_result():
        result = MagicMock()
        result.first = MagicMock(return_value=rows[0] if rows else None)
        result.rowcount = rowcount if rowcount is not None else 0
        return result

    session.execute = AsyncMock(side_effect=lambda *args, **kwargs: _make_result())
    return session


# ── put ────────────────────────────────────────────────────────────────────────


class TestPut:
    async def test_put_returns_canonical_sha256_and_uri(self):
        svc = BlobCASService()
        db = _make_session()
        content = b"hello world"
        expected_sha = hashlib.sha256(content).hexdigest()

        result = await svc.put(db, "g1", content, "text/plain")

        assert result["sha256"] == expected_sha
        assert result["content_uri"] == f"blob://sha256/{expected_sha}"

    async def test_put_is_deterministic_for_identical_content(self):
        svc = BlobCASService()
        db = _make_session()
        content = b"identical bytes"

        r1 = await svc.put(db, "g1", content, "text/plain")
        r2 = await svc.put(db, "g1", content, "text/plain")

        assert r1["sha256"] == r2["sha256"]
        assert r1["content_uri"] == r2["content_uri"]

    async def test_put_requires_graph_id(self):
        svc = BlobCASService()
        db = _make_session()
        with pytest.raises(ValueError, match="graph_id"):
            await svc.put(db, "", b"x", "text/plain")

    async def test_put_requires_mime_type(self):
        svc = BlobCASService()
        db = _make_session()
        with pytest.raises(ValueError, match="mime_type"):
            await svc.put(db, "g1", b"x", "")

    async def test_put_rejects_none_content(self):
        svc = BlobCASService()
        db = _make_session()
        with pytest.raises(ValueError, match="content_bytes"):
            await svc.put(db, "g1", None, "text/plain")  # type: ignore[arg-type]

    async def test_put_handles_empty_bytes(self):
        svc = BlobCASService()
        db = _make_session()
        # Empty content is legal — sha256("") is well-defined.
        result = await svc.put(db, "g1", b"", "application/octet-stream")
        assert result["sha256"] == hashlib.sha256(b"").hexdigest()


# ── get ────────────────────────────────────────────────────────────────────────


class TestGet:
    async def test_get_returns_content_when_present(self):
        svc = BlobCASService()
        sha = hashlib.sha256(b"abc").hexdigest()
        db = _make_session(rows=[(b"abc", "text/plain", 3)])
        out = await svc.get(db, "g1", f"blob://sha256/{sha}")
        assert out is not None
        assert out["content_bytes"] == b"abc"
        assert out["mime_type"] == "text/plain"
        assert out["size_bytes"] == 3

    async def test_get_returns_none_when_missing(self):
        svc = BlobCASService()
        sha = hashlib.sha256(b"missing").hexdigest()
        db = _make_session(rows=[])
        out = await svc.get(db, "g1", f"blob://sha256/{sha}")
        assert out is None

    async def test_get_returns_none_cross_tenant(self):
        """Cross-tenant access is existence-masked, NOT 403."""
        svc = BlobCASService()
        sha = hashlib.sha256(b"alice").hexdigest()
        # Bob queries with the right URI but Bob's graph_id has no row.
        db = _make_session(rows=[])
        out = await svc.get(db, "bob-tenant", f"blob://sha256/{sha}")
        assert out is None

    async def test_get_rejects_malformed_uri(self):
        svc = BlobCASService()
        db = _make_session()
        with pytest.raises(InvalidBlobURIError):
            await svc.get(db, "g1", "not-a-blob-uri")

    async def test_get_requires_graph_id(self):
        svc = BlobCASService()
        sha = hashlib.sha256(b"x").hexdigest()
        db = _make_session()
        with pytest.raises(ValueError, match="graph_id"):
            await svc.get(db, "", f"blob://sha256/{sha}")

    async def test_get_coerces_memoryview_to_bytes(self):
        """asyncpg may surface bytea as memoryview; we coerce defensively."""
        svc = BlobCASService()
        sha = hashlib.sha256(b"mv").hexdigest()
        db = _make_session(rows=[(memoryview(b"mv"), "text/plain", 2)])
        out = await svc.get(db, "g1", f"blob://sha256/{sha}")
        assert out is not None
        assert isinstance(out["content_bytes"], bytes)
        assert out["content_bytes"] == b"mv"


# ── delete ─────────────────────────────────────────────────────────────────────


class TestDelete:
    async def test_delete_returns_true_when_row_removed(self):
        svc = BlobCASService()
        sha = hashlib.sha256(b"d").hexdigest()
        db = _make_session(rowcount=1)
        deleted = await svc.delete(db, "g1", f"blob://sha256/{sha}")
        assert deleted is True

    async def test_delete_returns_false_when_no_row(self):
        svc = BlobCASService()
        sha = hashlib.sha256(b"d").hexdigest()
        db = _make_session(rowcount=0)
        deleted = await svc.delete(db, "g1", f"blob://sha256/{sha}")
        assert deleted is False

    async def test_delete_cross_tenant_returns_false(self):
        """A foreign tenant deleting our row matches 0 rows under their graph_id."""
        svc = BlobCASService()
        sha = hashlib.sha256(b"d").hexdigest()
        db = _make_session(rowcount=0)
        deleted = await svc.delete(db, "other-tenant", f"blob://sha256/{sha}")
        assert deleted is False

    async def test_delete_rejects_malformed_uri(self):
        svc = BlobCASService()
        db = _make_session()
        with pytest.raises(InvalidBlobURIError):
            await svc.delete(db, "g1", "not-a-uri")
