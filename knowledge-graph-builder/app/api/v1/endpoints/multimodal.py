"""
Multi-modal ingestion endpoints.

POST /graphs/{graph_id}/ingest/document  — PDF / DOCX file upload
POST /graphs/{graph_id}/ingest/image     — PNG / JPG / WEBP image upload

Both endpoints follow the same pattern as the existing text ingest endpoint:
1. Validate auth + graph access (ReBAC write check)
2. Save the uploaded file to a temp directory scoped to graph_id/job_id
3. Create an IngestionJob row in PostgreSQL (source_content = file path)
4. Queue the appropriate Celery task and return the job id for polling
"""

import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user_id, get_database, verify_graph_write_access
from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client
from app.models.graph import IngestionJob
from app.schemas.multimodal import MultiModalJobResponse, PDFExtractor, VisionModel
from app.services.background_job_service import background_job_service
from app.services.graph_node_service import GraphNodeService

router = APIRouter()
logger = get_logger(__name__)

# Root directory for uploaded files.  Each upload gets:
#   {_UPLOAD_ROOT}/{graph_id}/{job_id}/{original_filename}
_UPLOAD_ROOT = Path(os.environ.get("MULTIMODAL_UPLOAD_DIR", "/tmp/oraclous_uploads"))

_ALLOWED_DOCUMENT_MIME = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
}
_ALLOWED_IMAGE_MIME = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
}
_MAX_DOCUMENT_BYTES = 50 * 1024 * 1024  # 50 MB
_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB


# ── Helpers ───────────────────────────────────────────────────────────────────


def _save_upload(graph_id: str, job_id: str, upload: UploadFile, data: bytes) -> str:
    """Write uploaded bytes to an isolated temp path and return the absolute path."""
    dest_dir = _UPLOAD_ROOT / graph_id / job_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / (upload.filename or "upload")
    dest_path.write_bytes(data)
    return str(dest_path)


async def _verify_graph(graph_id: UUID) -> None:
    """Raise 503/404 if Neo4j is unavailable or the graph does not exist."""
    if not neo4j_client.sync_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j connection not available",
        )
    graph_service = GraphNodeService(neo4j_client.sync_driver)
    if not graph_service.get_graph(str(graph_id)):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Graph not found"
        )


# ── Document endpoint ─────────────────────────────────────────────────────────


@router.post(
    "/graphs/{graph_id}/ingest/document",
    response_model=MultiModalJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a PDF or DOCX document into a graph",
    responses={
        400: {"description": "Unsupported file type or file too large"},
        403: {"description": "Access denied"},
        404: {"description": "Graph not found"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def ingest_document(
    graph_id: UUID,
    file: UploadFile = File(..., description="PDF or DOCX file (max 50 MB)"),
    context: str = Form("", description="Domain hint for extraction (optional)"),
    extractor: PDFExtractor = Form(
        PDFExtractor.AUTO, description="PDF extraction backend"
    ),
    user_id: str = Depends(get_current_user_id),
    _access: str = Depends(verify_graph_write_access),
    db: AsyncSession = Depends(get_database),
):
    """
    Upload a PDF or DOCX document for entity/relationship extraction.

    The file is saved to a local temp directory, then processed asynchronously
    by a Celery worker.  Poll `GET /graphs/{id}/jobs/{job_id}` for status.
    """
    await _verify_graph(graph_id)

    # ── Read and validate ─────────────────────────────────────────────────────
    data = await file.read()
    if len(data) > _MAX_DOCUMENT_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds 50 MB limit ({len(data) // (1024*1024)} MB received)",
        )

    content_type = file.content_type or ""
    filename = file.filename or "document"
    ext = Path(filename).suffix.lower()
    if content_type not in _ALLOWED_DOCUMENT_MIME and ext not in {
        ".pdf",
        ".docx",
        ".doc",
    }:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported document type '{content_type}'. "
                "Supported types: PDF, DOCX"
            ),
        )

    source_type = (
        "pdf" if (content_type == "application/pdf" or ext == ".pdf") else "docx"
    )

    # ── Persist job record ────────────────────────────────────────────────────
    job_id = uuid4()
    try:
        file_path = _save_upload(str(graph_id), str(job_id), file, data)
    except OSError as exc:
        logger.error(f"Failed to save upload for graph {graph_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file — check MULTIMODAL_UPLOAD_DIR configuration",
        )

    # Encode context in effective_instructions so the worker can read it
    effective_instructions = (
        {"context": context, "extractor": extractor.value}
        if context
        else {"extractor": extractor.value}
    )

    job = IngestionJob(
        id=job_id,
        graph_id=graph_id,
        source_type=source_type,
        source_content=file_path,  # worker reads from this path
        status="pending",
        ingest_mode="incremental",
        effective_instructions=effective_instructions,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # ── Queue Celery task (reuses existing text pipeline after extraction) ────
    job_result = background_job_service.start_ingestion_job(str(job.id), user_id)
    if job_result["status"] == "failed":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start document ingestion job",
        )

    logger.info(
        f"Queued document ingestion job {job.id} (type={source_type}) for graph {graph_id}"
    )

    return MultiModalJobResponse(
        job_id=job.id,  # type: ignore[arg-type]
        graph_id=job.graph_id,  # type: ignore[arg-type]
        status=job.status,  # type: ignore[arg-type]
        source_type=source_type,
        filename=filename,
        created_at=job.created_at or datetime.now(UTC),  # type: ignore[arg-type]
    )


# ── Image endpoint ────────────────────────────────────────────────────────────


@router.post(
    "/graphs/{graph_id}/ingest/image",
    response_model=MultiModalJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest an image into a graph via vision extraction",
    responses={
        400: {"description": "Unsupported image type or file too large"},
        403: {"description": "Access denied"},
        404: {"description": "Graph not found"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def ingest_image(
    graph_id: UUID,
    file: UploadFile = File(..., description="PNG, JPG, or WEBP image (max 20 MB)"),
    context: str = Form("", description="Domain hint, e.g. 'AWS architecture diagram'"),
    vision_model: VisionModel = Form(VisionModel.CLAUDE, description="Vision model"),
    user_id: str = Depends(get_current_user_id),
    _access: str = Depends(verify_graph_write_access),
    db: AsyncSession = Depends(get_database),
):
    """
    Upload an image for entity/relationship extraction using a vision model.

    Claude 3.5 Sonnet is used by default.  Supply `vision_model=gpt4o` to use
    GPT-4o instead.  The job runs asynchronously; poll `GET /graphs/{id}/jobs/{job_id}`.
    """
    await _verify_graph(graph_id)

    data = await file.read()
    if len(data) > _MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image exceeds 20 MB limit ({len(data) // (1024*1024)} MB received)",
        )

    content_type = file.content_type or ""
    filename = file.filename or "image"
    ext = Path(filename).suffix.lower()
    if content_type not in _ALLOWED_IMAGE_MIME and ext not in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
    }:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported image type '{content_type}'. "
                "Supported types: PNG, JPG, WEBP, GIF"
            ),
        )

    job_id = uuid4()
    try:
        file_path = _save_upload(str(graph_id), str(job_id), file, data)
    except OSError as exc:
        logger.error(f"Failed to save image upload for graph {graph_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file — check MULTIMODAL_UPLOAD_DIR configuration",
        )

    effective_instructions = {
        "context": context,
        "vision_model": vision_model.value,
        "modal_type": "image",
    }

    job = IngestionJob(
        id=job_id,
        graph_id=graph_id,
        source_type="image",
        source_content=file_path,
        status="pending",
        ingest_mode="incremental",
        effective_instructions=effective_instructions,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Image jobs use a dedicated Celery task (vision → text → pipeline)
    job_result = background_job_service.start_image_ingestion_job(str(job.id), user_id)
    if job_result["status"] == "failed":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start image ingestion job",
        )

    logger.info(
        f"Queued image ingestion job {job.id} (model={vision_model}) for graph {graph_id}"
    )

    return MultiModalJobResponse(
        job_id=job.id,  # type: ignore[arg-type]
        graph_id=job.graph_id,  # type: ignore[arg-type]
        status=job.status,  # type: ignore[arg-type]
        source_type="image",
        filename=filename,
        created_at=job.created_at or datetime.now(UTC),  # type: ignore[arg-type]
    )
