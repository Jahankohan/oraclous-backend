"""
Integration tests for multimodal ingest endpoints (ORA-105).

Covers the FastAPI HTTP layer only — Celery tasks are mocked so no
worker infrastructure is required.

Endpoints under test:
  POST /graphs/{graph_id}/ingest/document
  POST /graphs/{graph_id}/ingest/image

Test scenarios (per acceptance criteria):
  1. Successful PDF upload    → 201, job_id in response
  2. Successful image upload  → 201, job_id in response
  3. Oversized document file  → 413-equivalent (400 with size message)
  4. Invalid MIME type        → 400 (unsupported type)
  5. Missing MULTIMODAL_UPLOAD_DIR → 500 with error detail
  6. Neo4j unavailable        → 503
  7. Cross-tenant access      → 403
"""

from __future__ import annotations

import io
import os
import tempfile
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from fastapi import Depends, HTTPException, status
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_USER_ID = "multimodal-test-user-00001"
TEST_GRAPH_ID = str(uuid4())
OTHER_GRAPH_ID = str(uuid4())  # owned by a different user

_1_BYTE_PDF = b"%PDF-1.4 1 0 obj<</Type/Catalog>>endobj"  # minimal valid-ish PDF header
_1_BYTE_PNG = (
    b"\x89PNG\r\n\x1a\n"  # PNG magic
    + b"\x00\x00\x00\rIHDR"  # IHDR chunk
    + b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"  # 1x1 RGB
    + b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
    + b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_client(user_id: str = TEST_USER_ID, graph_owner_id: str = TEST_USER_ID):
    """
    Return an async HTTP client with mocked auth, mocked graph access,
    and an in-memory database session.

    `graph_owner_id` is the user that "owns" the graph returned by Neo4j.
    When `graph_owner_id != user_id` the verify_graph_write_access mock raises 403.
    """
    from app.api.dependencies import (
        get_current_user_id,
        get_database,
        verify_graph_write_access,
    )
    from app.main import app

    async def _mock_user_id() -> str:
        return user_id

    async def _mock_verify(
        graph_id: str,
        uid: str = Depends(get_current_user_id),
    ) -> str:
        if uid != graph_owner_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
            )
        return graph_id

    mock_session = MagicMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock(
        side_effect=lambda obj: setattr(obj, "id", uuid4()) or None
    )

    async def _mock_db():
        yield mock_session

    app.dependency_overrides[get_current_user_id] = _mock_user_id
    app.dependency_overrides[verify_graph_write_access] = _mock_verify
    app.dependency_overrides[get_database] = _mock_db

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Standard client: TEST_USER_ID owns TEST_GRAPH_ID."""
    from app.main import app

    async with _make_client() as c:
        yield c

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def cross_tenant_client() -> AsyncGenerator[AsyncClient, None]:
    """Client where user does NOT own the target graph → 403."""
    from app.main import app

    # graph_owner_id is different from TEST_USER_ID
    async with _make_client(
        user_id=TEST_USER_ID, graph_owner_id="other-user-99999"
    ) as c:
        yield c

    app.dependency_overrides.clear()


def _mock_neo4j_graph(graph_id: str = TEST_GRAPH_ID, user_id: str = TEST_USER_ID):
    """Return a mock GraphNodeService that claims the graph exists and is owned by user_id."""
    svc = MagicMock()
    svc.get_graph.return_value = {"id": graph_id, "user_id": user_id}
    return svc


def _mock_job_queued():
    """Return a mock background_job_service where queueing always succeeds.

    `start_ingestion_job` / `start_image_ingestion_job` are async — the
    multimodal endpoints `await` them, so the mocks must be `AsyncMock`.
    """
    svc = MagicMock()
    svc.start_ingestion_job = AsyncMock(
        return_value={"status": "queued", "job_id": str(uuid4())}
    )
    svc.start_image_ingestion_job = AsyncMock(
        return_value={"status": "queued", "job_id": str(uuid4())}
    )
    return svc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pdf_upload(size: int = len(_1_BYTE_PDF), filename: str = "test.pdf") -> tuple:
    data = (
        _1_BYTE_PDF
        if size <= len(_1_BYTE_PDF)
        else _1_BYTE_PDF + b" " * (size - len(_1_BYTE_PDF))
    )
    return ("file", (filename, io.BytesIO(data), "application/pdf"))


def _png_upload(size: int = len(_1_BYTE_PNG), filename: str = "test.png") -> tuple:
    data = (
        _1_BYTE_PNG
        if size <= len(_1_BYTE_PNG)
        else _1_BYTE_PNG + b"\x00" * (size - len(_1_BYTE_PNG))
    )
    return ("file", (filename, io.BytesIO(data), "image/png"))


# ---------------------------------------------------------------------------
# Scenario 1 — Successful PDF upload
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_document_pdf_success(client: AsyncClient) -> None:
    """POST /graphs/{id}/ingest/document with a valid PDF → 201 + job_id."""
    graph_svc = _mock_neo4j_graph()
    job_svc = _mock_job_queued()

    with (
        patch(
            "app.api.v1.endpoints.multimodal.GraphNodeService", return_value=graph_svc
        ),
        patch("app.api.v1.endpoints.multimodal.background_job_service", job_svc),
        patch("app.api.v1.endpoints.multimodal.neo4j_client") as mock_neo4j,
    ):
        mock_neo4j.sync_driver = MagicMock()  # driver is available

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"MULTIMODAL_UPLOAD_DIR": tmpdir}):
                # Re-patch the module-level _UPLOAD_ROOT
                with patch(
                    "app.api.v1.endpoints.multimodal._UPLOAD_ROOT",
                    __import__("pathlib").Path(tmpdir),
                ):
                    resp = await client.post(
                        f"/api/v1/graphs/{TEST_GRAPH_ID}/ingest/document",
                        files=[_pdf_upload()],
                        data={"context": "test context", "extractor": "auto"},
                    )

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert "job_id" in body
    UUID(body["job_id"])  # must be a valid UUID
    assert body["source_type"] == "pdf"
    assert body["status"] == "pending"
    assert body["graph_id"] == TEST_GRAPH_ID


# ---------------------------------------------------------------------------
# Scenario 2 — Successful image upload
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_image_png_success(client: AsyncClient) -> None:
    """POST /graphs/{id}/ingest/image with a valid PNG → 201 + job_id."""
    graph_svc = _mock_neo4j_graph()
    job_svc = _mock_job_queued()

    with (
        patch(
            "app.api.v1.endpoints.multimodal.GraphNodeService", return_value=graph_svc
        ),
        patch("app.api.v1.endpoints.multimodal.background_job_service", job_svc),
        patch("app.api.v1.endpoints.multimodal.neo4j_client") as mock_neo4j,
    ):
        mock_neo4j.sync_driver = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "app.api.v1.endpoints.multimodal._UPLOAD_ROOT",
                __import__("pathlib").Path(tmpdir),
            ):
                resp = await client.post(
                    f"/api/v1/graphs/{TEST_GRAPH_ID}/ingest/image",
                    files=[_png_upload()],
                    data={
                        "context": "AWS architecture diagram",
                        "vision_model": "claude",
                    },
                )

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert "job_id" in body
    UUID(body["job_id"])
    assert body["source_type"] == "image"
    assert body["status"] == "pending"
    assert body["graph_id"] == TEST_GRAPH_ID


# ---------------------------------------------------------------------------
# Scenario 3 — Oversized files rejected
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_document_too_large(client: AsyncClient) -> None:
    """PDF > 50 MB → 400 with size detail."""
    _50MB_PLUS_1 = 50 * 1024 * 1024 + 1
    oversized_data = b"a" * _50MB_PLUS_1

    graph_svc = _mock_neo4j_graph()

    with (
        patch(
            "app.api.v1.endpoints.multimodal.GraphNodeService", return_value=graph_svc
        ),
        patch("app.api.v1.endpoints.multimodal.neo4j_client") as mock_neo4j,
    ):
        mock_neo4j.sync_driver = MagicMock()

        resp = await client.post(
            f"/api/v1/graphs/{TEST_GRAPH_ID}/ingest/document",
            files=[
                ("file", ("big.pdf", io.BytesIO(oversized_data), "application/pdf"))
            ],
            data={"extractor": "auto"},
        )

    assert resp.status_code == 400, resp.text
    assert "50 MB" in resp.json().get("detail", "")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_image_too_large(client: AsyncClient) -> None:
    """PNG > 20 MB → 400 with size detail."""
    _20MB_PLUS_1 = 20 * 1024 * 1024 + 1
    oversized_data = b"\x89PNG" + b"\x00" * (_20MB_PLUS_1 - 4)

    graph_svc = _mock_neo4j_graph()

    with (
        patch(
            "app.api.v1.endpoints.multimodal.GraphNodeService", return_value=graph_svc
        ),
        patch("app.api.v1.endpoints.multimodal.neo4j_client") as mock_neo4j,
    ):
        mock_neo4j.sync_driver = MagicMock()

        resp = await client.post(
            f"/api/v1/graphs/{TEST_GRAPH_ID}/ingest/image",
            files=[("file", ("big.png", io.BytesIO(oversized_data), "image/png"))],
            data={"vision_model": "claude"},
        )

    assert resp.status_code == 400, resp.text
    assert "20 MB" in resp.json().get("detail", "")


# ---------------------------------------------------------------------------
# Scenario 4 — Invalid MIME type rejected
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_document_invalid_mime(client: AsyncClient) -> None:
    """Uploading an audio file as a document → 400 unsupported type."""
    graph_svc = _mock_neo4j_graph()

    with (
        patch(
            "app.api.v1.endpoints.multimodal.GraphNodeService", return_value=graph_svc
        ),
        patch("app.api.v1.endpoints.multimodal.neo4j_client") as mock_neo4j,
    ):
        mock_neo4j.sync_driver = MagicMock()

        resp = await client.post(
            f"/api/v1/graphs/{TEST_GRAPH_ID}/ingest/document",
            files=[("file", ("audio.mp3", io.BytesIO(b"ID3"), "audio/mpeg"))],
            data={"extractor": "auto"},
        )

    assert resp.status_code == 400, resp.text
    detail = resp.json().get("detail", "")
    assert "Unsupported" in detail or "unsupported" in detail


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_image_invalid_mime(client: AsyncClient) -> None:
    """Uploading a PDF as an image → 400 unsupported type."""
    graph_svc = _mock_neo4j_graph()

    with (
        patch(
            "app.api.v1.endpoints.multimodal.GraphNodeService", return_value=graph_svc
        ),
        patch("app.api.v1.endpoints.multimodal.neo4j_client") as mock_neo4j,
    ):
        mock_neo4j.sync_driver = MagicMock()

        resp = await client.post(
            f"/api/v1/graphs/{TEST_GRAPH_ID}/ingest/image",
            files=[("file", ("doc.pdf", io.BytesIO(_1_BYTE_PDF), "application/pdf"))],
            data={"vision_model": "claude"},
        )

    assert resp.status_code == 400, resp.text
    detail = resp.json().get("detail", "")
    assert "Unsupported" in detail or "unsupported" in detail


# ---------------------------------------------------------------------------
# Scenario 5 — Missing MULTIMODAL_UPLOAD_DIR → graceful 500
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_document_missing_upload_dir(client: AsyncClient) -> None:
    """If the upload directory cannot be created (e.g. bad config) → 500 with detail."""
    graph_svc = _mock_neo4j_graph()
    job_svc = _mock_job_queued()

    with (
        patch(
            "app.api.v1.endpoints.multimodal.GraphNodeService", return_value=graph_svc
        ),
        patch("app.api.v1.endpoints.multimodal.background_job_service", job_svc),
        patch("app.api.v1.endpoints.multimodal.neo4j_client") as mock_neo4j,
        # Point _UPLOAD_ROOT at a path that cannot be created
        patch(
            "app.api.v1.endpoints.multimodal._UPLOAD_ROOT",
            __import__("pathlib").Path("/proc/oraclous_forbidden_upload_dir"),
        ),
    ):
        mock_neo4j.sync_driver = MagicMock()

        resp = await client.post(
            f"/api/v1/graphs/{TEST_GRAPH_ID}/ingest/document",
            files=[_pdf_upload()],
            data={"extractor": "auto"},
        )

    # The endpoint should not crash with a raw 500 traceback but return a
    # structured error response.
    assert resp.status_code == 500
    body = resp.json()
    # The response must contain a "detail" key (not a bare traceback dump)
    assert "detail" in body or "error" in body


# ---------------------------------------------------------------------------
# Scenario 6 — Neo4j unavailable → 503
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_document_neo4j_unavailable(client: AsyncClient) -> None:
    """If Neo4j sync_driver is None → 503."""
    with patch("app.api.v1.endpoints.multimodal.neo4j_client") as mock_neo4j:
        mock_neo4j.sync_driver = None  # simulate connection not ready

        resp = await client.post(
            f"/api/v1/graphs/{TEST_GRAPH_ID}/ingest/document",
            files=[_pdf_upload()],
            data={"extractor": "auto"},
        )

    assert resp.status_code == 503, resp.text
    assert "Neo4j" in resp.json().get("detail", "")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_image_neo4j_unavailable(client: AsyncClient) -> None:
    """If Neo4j sync_driver is None → 503."""
    with patch("app.api.v1.endpoints.multimodal.neo4j_client") as mock_neo4j:
        mock_neo4j.sync_driver = None

        resp = await client.post(
            f"/api/v1/graphs/{TEST_GRAPH_ID}/ingest/image",
            files=[_png_upload()],
            data={"vision_model": "claude"},
        )

    assert resp.status_code == 503, resp.text
    assert "Neo4j" in resp.json().get("detail", "")


# ---------------------------------------------------------------------------
# Scenario 7 — Cross-tenant access → 403
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_document_cross_tenant_denied(
    cross_tenant_client: AsyncClient,
) -> None:
    """User trying to ingest into a graph they don't own → 403."""
    resp = await cross_tenant_client.post(
        f"/api/v1/graphs/{OTHER_GRAPH_ID}/ingest/document",
        files=[_pdf_upload()],
        data={"extractor": "auto"},
    )
    assert resp.status_code == 403, resp.text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_image_cross_tenant_denied(
    cross_tenant_client: AsyncClient,
) -> None:
    """User trying to ingest image into a graph they don't own → 403."""
    resp = await cross_tenant_client.post(
        f"/api/v1/graphs/{OTHER_GRAPH_ID}/ingest/image",
        files=[_png_upload()],
        data={"vision_model": "claude"},
    )
    assert resp.status_code == 403, resp.text
