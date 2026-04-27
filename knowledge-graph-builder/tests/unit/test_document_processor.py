"""
Unit tests for DocumentProcessor.

Tests text processing, validation, unsupported type handling,
and edge cases — no external deps required.
"""

import pytest
from fastapi import HTTPException

from app.services.document_processor import DocumentProcessor, document_processor

# ---------------------------------------------------------------------------
# Tests: process_document routing
# ---------------------------------------------------------------------------


class TestProcessDocumentRouting:
    @pytest.mark.unit
    def test_text_type_routes_to_process_text(self):
        result = DocumentProcessor.process_document(
            "Hello world, this is test content.", source_type="text"
        )
        assert result["success"] is True
        assert "text" in result

    @pytest.mark.unit
    def test_pdf_type_raises_422_for_bad_path(self):
        # PDF processing is now implemented; a non-existent path raises 422
        with pytest.raises(HTTPException) as exc_info:
            DocumentProcessor.process_document(
                "/nonexistent/path/file.pdf", source_type="pdf"
            )
        assert exc_info.value.status_code == 422

    @pytest.mark.unit
    def test_doc_type_raises_422_for_bad_path(self):
        with pytest.raises(HTTPException) as exc_info:
            DocumentProcessor.process_document(
                "/nonexistent/path/file.doc", source_type="doc"
            )
        assert exc_info.value.status_code == 422

    @pytest.mark.unit
    def test_docx_type_raises_422_for_bad_path(self):
        with pytest.raises(HTTPException) as exc_info:
            DocumentProcessor.process_document(
                "/nonexistent/path/file.docx", source_type="docx"
            )
        assert exc_info.value.status_code == 422

    @pytest.mark.unit
    def test_url_type_raises_not_implemented(self):
        with pytest.raises(HTTPException) as exc_info:
            DocumentProcessor.process_document("https://example.com", source_type="url")
        assert exc_info.value.status_code == 501

    @pytest.mark.unit
    def test_unsupported_type_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            DocumentProcessor.process_document("content", source_type="xml")
        assert exc_info.value.status_code == 400
        assert "xml" in exc_info.value.detail

    @pytest.mark.unit
    def test_default_source_type_is_text(self):
        result = DocumentProcessor.process_document(
            "Long enough content for processing."
        )
        assert result["success"] is True


# ---------------------------------------------------------------------------
# Tests: _process_text
# ---------------------------------------------------------------------------


class TestProcessText:
    @pytest.mark.unit
    def test_returns_stripped_text(self):
        result = DocumentProcessor._process_text("  Hello world text.  ")
        assert result["text"] == "Hello world text."

    @pytest.mark.unit
    def test_adds_content_length_to_metadata(self):
        content = "This is some test content here."
        result = DocumentProcessor._process_text(content)
        assert result["metadata"]["content_length"] == len(content)

    @pytest.mark.unit
    def test_adds_content_type_to_metadata(self):
        result = DocumentProcessor._process_text("Some content here.")
        assert result["metadata"]["content_type"] == "text/plain"

    @pytest.mark.unit
    def test_adds_processing_method_to_metadata(self):
        result = DocumentProcessor._process_text("Some content here.")
        assert result["metadata"]["processing_method"] == "raw_text"

    @pytest.mark.unit
    def test_merges_provided_metadata(self):
        result = DocumentProcessor._process_text(
            "Some content here.",
            metadata={"job_id": "job-123", "graph_id": "graph-abc"},
        )
        assert result["metadata"]["job_id"] == "job-123"
        assert result["metadata"]["graph_id"] == "graph-abc"
        assert result["metadata"]["content_type"] == "text/plain"

    @pytest.mark.unit
    def test_short_content_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            DocumentProcessor._process_text("Short")
        assert exc_info.value.status_code == 400

    @pytest.mark.unit
    def test_empty_content_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            DocumentProcessor._process_text("")
        assert exc_info.value.status_code == 400

    @pytest.mark.unit
    def test_whitespace_only_content_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            DocumentProcessor._process_text("         ")
        assert exc_info.value.status_code == 400

    @pytest.mark.unit
    def test_exactly_10_chars_passes(self):
        # 10 chars exactly — minimum boundary
        result = DocumentProcessor._process_text("1234567890")
        assert result["success"] is True

    @pytest.mark.unit
    def test_success_flag_is_true(self):
        result = DocumentProcessor._process_text("Hello world content here.")
        assert result["success"] is True

    @pytest.mark.unit
    def test_none_metadata_defaults_to_empty_dict(self):
        result = DocumentProcessor._process_text("Some content here.", metadata=None)
        assert "content_type" in result["metadata"]


# ---------------------------------------------------------------------------
# Tests: get_supported_types and validate_source_type
# ---------------------------------------------------------------------------


class TestSupportedTypes:
    @pytest.mark.unit
    def test_get_supported_types_returns_list(self):
        types = DocumentProcessor.get_supported_types()
        assert isinstance(types, list)

    @pytest.mark.unit
    def test_text_is_supported(self):
        assert "text" in DocumentProcessor.get_supported_types()

    @pytest.mark.unit
    def test_validate_text_source_type(self):
        assert DocumentProcessor.validate_source_type("text") is True

    @pytest.mark.unit
    def test_pdf_is_supported(self):
        assert "pdf" in DocumentProcessor.get_supported_types()

    @pytest.mark.unit
    def test_validate_pdf_returns_true(self):
        # pdf is now implemented
        assert DocumentProcessor.validate_source_type("pdf") is True

    @pytest.mark.unit
    def test_validate_unknown_type_returns_false(self):
        assert DocumentProcessor.validate_source_type("unknown") is False

    @pytest.mark.unit
    def test_validate_empty_string_returns_false(self):
        assert DocumentProcessor.validate_source_type("") is False


# ---------------------------------------------------------------------------
# Tests: global instance
# ---------------------------------------------------------------------------


class TestGlobalInstance:
    @pytest.mark.unit
    def test_global_document_processor_exists(self):
        assert document_processor is not None
        assert isinstance(document_processor, DocumentProcessor)

    @pytest.mark.unit
    def test_global_instance_works(self):
        result = document_processor.process_document(
            "Global instance test content.", source_type="text"
        )
        assert result["success"] is True
