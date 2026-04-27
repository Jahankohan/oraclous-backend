"""
Document Processing Service

Handles different file types for ingestion:
- Raw text
- PDF files (with diagram-mode image extraction for flagged pages)
- DOC/DOCX files
- CSV / TSV files
- JSON / JSONL files
- Markdown files
- Future: URLs
"""

from typing import Any

from fastapi import HTTPException, status

from app.core.logging import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """
    Process different document types for GraphRAG ingestion.

    Supported formats:
    - text     : Raw text content
    - pdf      : PDF files (via pdf_extractor; diagram pages routed to VisionExtractor)
    - doc/docx : Word documents
    - csv/tsv  : CSV and TSV files (via csv_extractor)
    - json/jsonl: JSON and JSONL files (via json_extractor)
    - md/markdown: Markdown files (via md_extractor)
    - url      : Web content (not yet implemented)
    """

    @staticmethod
    def process_document(
        content: str, source_type: str = "text", metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Process document content based on source type.

        Args:
            content: Document content (raw text) or absolute file path on disk.
            source_type: One of "text", "pdf", "doc", "docx", "csv", "tsv",
                         "json", "jsonl", "md", "markdown", "url".
            metadata: Additional metadata for the document.

        Returns:
            Processed document with text content and metadata (+ ``structured``
            key for CSV/JSON/MD documents).
        """
        if source_type == "text":
            return DocumentProcessor._process_text(content, metadata)
        elif source_type == "pdf":
            return DocumentProcessor._process_pdf(content, metadata)
        elif source_type in {"doc", "docx"}:
            return DocumentProcessor._process_word(content, metadata)
        elif source_type in {"csv", "tsv"}:
            return DocumentProcessor._process_csv(content, metadata)
        elif source_type in {"json", "jsonl"}:
            return DocumentProcessor._process_json(content, metadata)
        elif source_type in {"md", "markdown"}:
            return DocumentProcessor._process_markdown(content, metadata)
        elif source_type == "url":
            return DocumentProcessor._process_url(content, metadata)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported source type: {source_type}. "
                    "Supported types: text, pdf, doc, docx, csv, tsv, "
                    "json, jsonl, md, markdown, url"
                ),
            )

    # ── Core format handlers ──────────────────────────────────────────────────

    @staticmethod
    def _process_text(
        content: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Process raw text content."""
        if not content or len(content.strip()) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text content must be at least 10 characters long",
            )

        processed_metadata = metadata or {}
        processed_metadata.update(
            {
                "content_length": len(content),
                "content_type": "text/plain",
                "processing_method": "raw_text",
            }
        )

        return {
            "text": content.strip(),
            "metadata": processed_metadata,
            "success": True,
        }

    @staticmethod
    def _process_pdf(
        content: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Process a PDF file.

        Image paths returned by pdf_extractor that carry ``likely_diagram: True``
        in their per-page metadata are forwarded to VisionExtractor in diagram
        mode so they produce structured nodes/edges rather than plain text.

        Args:
            content: Absolute path to the PDF file on disk.
            metadata: Additional metadata dict.
        """
        from app.services.pdf_extractor import extract_pdf

        try:
            result = extract_pdf(content)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"PDF extraction failed: {exc}",
            )

        if not result["text"].strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No text could be extracted from this PDF.",
            )

        processed_metadata = metadata or {}
        processed_metadata.update(result["metadata"])
        processed_metadata["content_length"] = len(result["text"])

        # Route diagram-flagged pages through VisionExtractor in diagram mode
        image_paths = result.get("image_paths", [])
        diagram_results: list[dict[str, Any]] = []
        page_metadata: dict[str, Any] = result.get("metadata", {})

        if image_paths and page_metadata.get("likely_diagram"):
            from app.services.vision_extractor import vision_extractor

            for img_path in image_paths:
                try:
                    diagram = vision_extractor.extract(
                        img_path,
                        metadata={"likely_diagram": True},
                    )
                    diagram_results.append({"image_path": img_path, "diagram": diagram})
                except Exception as exc:
                    logger.warning(f"Diagram extraction failed for {img_path}: {exc}")

        response: dict[str, Any] = {
            "text": result["text"],
            "metadata": processed_metadata,
            "image_paths": image_paths,
            "success": True,
        }
        if diagram_results:
            response["diagram_extractions"] = diagram_results

        return response

    @staticmethod
    def _process_word(
        content: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Process a DOCX file.

        Args:
            content: Absolute path to the DOCX file on disk.
            metadata: Additional metadata dict.
        """
        from app.services.pdf_extractor import extract_docx

        try:
            result = extract_docx(content)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"DOCX extraction failed: {exc}",
            )

        if not result["text"].strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No text could be extracted from this DOCX file.",
            )

        processed_metadata = metadata or {}
        processed_metadata.update(result["metadata"])
        processed_metadata["content_length"] = len(result["text"])

        return {
            "text": result["text"],
            "metadata": processed_metadata,
            "image_paths": [],
            "success": True,
        }

    # ── Structured format handlers (TASK-025) ─────────────────────────────────

    @staticmethod
    def _process_csv(
        content: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Process a CSV or TSV file.

        Args:
            content: Absolute path to the CSV/TSV file on disk.
            metadata: Additional metadata dict.
        """
        from app.services.csv_extractor import extract_csv

        try:
            result = extract_csv(content)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"CSV extraction failed: {exc}",
            )

        processed_metadata = metadata or {}
        processed_metadata.update(
            {
                "content_type": "text/csv",
                "processing_method": "csv_extractor",
                "row_count": result["row_count"],
                "column_count": len(result["columns"]),
            }
        )

        return {
            "text": "",
            "structured": result,
            "metadata": processed_metadata,
            "success": True,
        }

    @staticmethod
    def _process_json(
        content: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Process a JSON or JSONL file.

        Args:
            content: Absolute path to the JSON/JSONL file on disk.
            metadata: Additional metadata dict.
        """
        from app.services.json_extractor import extract_json

        try:
            result = extract_json(content)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"JSON extraction failed: {exc}",
            )

        processed_metadata = metadata or {}
        processed_metadata.update(
            {
                "content_type": "application/json",
                "processing_method": "json_extractor",
                "record_count": result["record_count"],
            }
        )

        return {
            "text": "",
            "structured": result,
            "metadata": processed_metadata,
            "success": True,
        }

    @staticmethod
    def _process_markdown(
        content: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Process a Markdown file.

        Args:
            content: Absolute path to the Markdown file on disk.
            metadata: Additional metadata dict.
        """
        from app.services.md_extractor import extract_markdown

        try:
            result = extract_markdown(content)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Markdown extraction failed: {exc}",
            )

        processed_metadata = metadata or {}
        processed_metadata.update(
            {
                "content_type": "text/markdown",
                "processing_method": "md_extractor",
                "section_count": len(result["sections"]),
                "title": result["title"],
            }
        )

        return {
            "text": "",
            "structured": result,
            "metadata": processed_metadata,
            "success": True,
        }

    @staticmethod
    def _process_url(
        content: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Process URL content by fetching and extracting text.

        TODO: Implement web scraping using:
        - requests + BeautifulSoup
        - newspaper3k for articles
        - readability-lxml for clean text extraction
        """
        logger.warning("URL processing not yet implemented")

        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="URL processing is not yet implemented. Coming soon!",
        )

    # ── Utility methods ───────────────────────────────────────────────────────

    @staticmethod
    def get_supported_types() -> list[str]:
        """Return list of supported document types."""
        return ["text", "pdf", "doc", "docx", "csv", "tsv", "json", "jsonl", "md", "markdown"]

    @staticmethod
    def validate_source_type(source_type: str) -> bool:
        """Validate if source type is supported."""
        return source_type in DocumentProcessor.get_supported_types()


# Global instance
document_processor = DocumentProcessor()
