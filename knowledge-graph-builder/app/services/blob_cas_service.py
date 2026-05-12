"""Postgres-backed content-addressable store for deliverable bytes (TASK-082).

`BlobCASService` is the application-layer interface to the `blob_cas` table.
:Deliverable nodes in Neo4j carry a small metadata footprint plus a
`content_uri` of the form `blob://sha256/<hex>` that this service resolves to
raw bytes + mime type.

Tenancy
-------

Every public method takes a `graph_id` and filters every SQL statement by it.
A `get()` for content owned by another tenant returns `None` (existence is
masked, not exposed via 403), which the REST layer translates to HTTP 404.
This matches the fail-closed security posture in
`oraclous-data-studio/CLAUDE.md` and the explicit DoD bullet on TASK-082.

Why bytea (not S3)
------------------

The Self-Hosted + Data-Ownership founding principles forbid egress to an
Oraclous-operated SaaS. Customers ship the platform inside their own
infrastructure; sticking with Postgres bytea means the same Postgres pool
that already holds tenant metadata holds the deliverable bytes — one DB to
back up, one to restore. The 5 final `/docify` docs are at most low single-
digit MB each; bytea is comfortable up to that range. If a future use case
needs multi-GB blobs we revisit and plug in a pluggable object-store
adapter behind the same `put`/`get`/`delete` surface.

Connection pool
---------------

We reuse the existing async engine from `app.core.database` (the same pool
the rest of the FastAPI app shares). No new pool, no new ORM pattern.
Callers pass an `AsyncSession` from the existing `get_db` dependency; the
service does not manage the session lifecycle (commit / rollback / close is
the caller's responsibility — matches `connector_service.py`).
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.logging import get_logger
from app.models.graph import BlobCAS

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)


# =============================================================================
# Module-level constants
# =============================================================================

#: The URI scheme the CAS issues. :Deliverable.content_uri values produced by
#: `BlobCASService.put()` start with this prefix.
BLOB_URI_SCHEME = "blob://sha256/"

#: Regex that matches a valid CAS URI and captures the sha256 hex.
_BLOB_URI_RE = re.compile(r"^blob://sha256/([0-9a-f]{64})$")


# =============================================================================
# Typed errors
# =============================================================================


class InvalidBlobURIError(ValueError):
    """Raised when a `content_uri` is malformed or not a CAS URI.

    The REST layer maps this to HTTP 400. Callers that legitimately mix
    blob CAS URIs with foreign URIs (e.g. a filesystem path produced by
    SPRINT-001) should inspect the prefix with `is_blob_uri()` rather than
    rely on the exception.
    """


# =============================================================================
# Helpers
# =============================================================================


def is_blob_uri(uri: str | None) -> bool:
    """Return True if `uri` is shaped like `blob://sha256/<64-hex>`."""
    return bool(uri and _BLOB_URI_RE.match(uri))


def parse_blob_uri(uri: str) -> str:
    """Extract the sha256 hex from a CAS URI; raise on malformed input."""
    m = _BLOB_URI_RE.match(uri or "")
    if not m:
        raise InvalidBlobURIError(
            f"Not a valid CAS URI: {uri!r} (expected 'blob://sha256/<64-hex>')"
        )
    return m.group(1)


def make_blob_uri(sha256: str) -> str:
    """Format a sha256 hex into the canonical `blob://sha256/<hex>` URI."""
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise InvalidBlobURIError(f"sha256 must be 64 hex chars; got {sha256!r}")
    return f"{BLOB_URI_SCHEME}{sha256}"


# =============================================================================
# Result types — kept as plain TypedDict-shaped dicts for callers that already
# speak Pydantic models elsewhere; the service surface is intentionally narrow.
# =============================================================================


class BlobPutResult(dict):
    """Return type for `BlobCASService.put`.

    Plain `dict` subclass for ergonomic key access (`r["sha256"]`) and to
    avoid pulling another schema module into the service layer.
    """

    sha256: str
    content_uri: str


class BlobGetResult(dict):
    """Return type for `BlobCASService.get`."""

    content_bytes: bytes
    mime_type: str
    size_bytes: int


# =============================================================================
# Service
# =============================================================================


class BlobCASService:
    """CRUD on the content-addressable blob store.

    The service is stateless apart from the injected `AsyncSession` factory
    per call. Constructed once per process; production callers wire it as a
    FastAPI Depends() singleton.
    """

    @staticmethod
    def _digest(content_bytes: bytes) -> str:
        return hashlib.sha256(content_bytes).hexdigest()

    async def put(
        self,
        db: AsyncSession,
        graph_id: str,
        content_bytes: bytes,
        mime_type: str,
    ) -> BlobPutResult:
        """Write `content_bytes` into the CAS under `graph_id`.

        Idempotent: re-uploading identical bytes under the same tenant
        returns the same `sha256` and a no-op write (Postgres ON CONFLICT
        DO NOTHING; the existing row's mime_type / size_bytes / content
        are left untouched on collision).

        Returns the canonical CAS URI the caller should write onto the
        `:Deliverable.content_uri` property.
        """
        if not graph_id:
            raise ValueError("graph_id is required")
        if content_bytes is None:
            raise ValueError("content_bytes is required (use empty b'' for empty)")
        if not mime_type:
            raise ValueError("mime_type is required")

        sha256 = self._digest(content_bytes)
        size_bytes = len(content_bytes)

        stmt = (
            pg_insert(BlobCAS)
            .values(
                graph_id=graph_id,
                sha256=sha256,
                mime_type=mime_type,
                size_bytes=size_bytes,
                content=content_bytes,
            )
            .on_conflict_do_nothing(index_elements=["graph_id", "sha256"])
        )
        await db.execute(stmt)
        # The caller commits — matches the rest of the service layer.

        logger.info(
            "blob_cas.put graph_id=%s sha256=%s size_bytes=%d mime=%s",
            graph_id,
            sha256,
            size_bytes,
            mime_type,
        )
        return BlobPutResult(sha256=sha256, content_uri=make_blob_uri(sha256))

    async def get(
        self,
        db: AsyncSession,
        graph_id: str,
        content_uri: str,
    ) -> BlobGetResult | None:
        """Resolve a CAS URI to its bytes, scoped to `graph_id`.

        Returns `None` when:
          - the URI does not match any row in this tenant's namespace
          - the underlying sha256 exists but belongs to another tenant
            (existence is masked — see module docstring)

        Raises `InvalidBlobURIError` only when the URI itself is malformed.
        Callers map a `None` result to HTTP 404.
        """
        if not graph_id:
            raise ValueError("graph_id is required")
        sha256 = parse_blob_uri(content_uri)

        stmt = select(BlobCAS.content, BlobCAS.mime_type, BlobCAS.size_bytes).where(
            BlobCAS.graph_id == graph_id,
            BlobCAS.sha256 == sha256,
        )
        result = await db.execute(stmt)
        row = result.first()
        if row is None:
            return None
        content_bytes, mime_type, size_bytes = row
        # Defensive: bytea round-trips as `bytes` from asyncpg; coerce
        # memoryview just in case a future driver swap surfaces it.
        if isinstance(content_bytes, memoryview):
            content_bytes = bytes(content_bytes)
        return BlobGetResult(
            content_bytes=content_bytes,
            mime_type=mime_type,
            size_bytes=int(size_bytes),
        )

    async def delete(
        self,
        db: AsyncSession,
        graph_id: str,
        content_uri: str,
    ) -> bool:
        """Hard-delete a CAS row. Admin-only (gated at the endpoint layer).

        Returns True iff a row was deleted in this tenant's namespace.
        Cross-tenant deletes return False (existence-masking applies the
        same way as `get`). Callers commit.
        """
        if not graph_id:
            raise ValueError("graph_id is required")
        sha256 = parse_blob_uri(content_uri)

        stmt = delete(BlobCAS).where(
            BlobCAS.graph_id == graph_id,
            BlobCAS.sha256 == sha256,
        )
        result = await db.execute(stmt)
        deleted = (result.rowcount or 0) > 0
        if deleted:
            logger.info("blob_cas.delete graph_id=%s sha256=%s", graph_id, sha256)
        return deleted
