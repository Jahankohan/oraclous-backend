"""
Pydantic schemas for multi-modal ingestion endpoints.

Covers document (PDF, DOCX) and image (PNG, JPG, WEBP) upload requests
and the corresponding job responses.
"""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel


class VisionModel(StrEnum):
    """Vision model to use for image entity extraction."""

    CLAUDE = "claude"
    GPT4O = "gpt4o"


class PDFExtractor(StrEnum):
    """PDF extraction backend."""

    AUTO = "auto"  # PyMuPDF for standard PDFs, falls back to basic text
    PYMUPDF = "pymupdf"


class MultiModalJobResponse(BaseModel):
    """Response returned immediately after queuing a multi-modal ingestion job."""

    job_id: UUID
    graph_id: UUID
    status: str
    source_type: str
    filename: str
    created_at: datetime

    model_config = {"from_attributes": True}
