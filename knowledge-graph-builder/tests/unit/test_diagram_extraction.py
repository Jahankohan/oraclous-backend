"""
Unit tests for VisionExtractor diagram mode (TASK-025).

Covers:
- Diagram mode triggers when metadata has likely_diagram: True
- Diagram mode triggers from filename heuristics (arch, diagram, flow, uml, schema, model)
- Diagram mode does NOT trigger for regular images
- _is_diagram_mode() logic
- extract() routes to diagram mode vs standard mode
- Diagram extraction returns dict with nodes/edges keys
- Non-diagram path returns entities/relationships dict
"""

import json
import tempfile
import os
from unittest.mock import MagicMock, patch

import pytest

from app.services.vision_extractor import VisionExtractor, _is_diagram_mode


# ─── _is_diagram_mode logic ───────────────────────────────────────────────────


class TestIsDiagramMode:
    @pytest.mark.unit
    def test_likely_diagram_true_in_metadata(self):
        assert _is_diagram_mode("/tmp/random.png", {"likely_diagram": True}) is True

    @pytest.mark.unit
    def test_likely_diagram_false_in_metadata(self):
        assert _is_diagram_mode("/tmp/random.png", {"likely_diagram": False}) is False

    @pytest.mark.unit
    def test_no_metadata_no_keyword_is_false(self):
        assert _is_diagram_mode("/tmp/photo.png", None) is False

    @pytest.mark.unit
    def test_empty_metadata_no_keyword_is_false(self):
        assert _is_diagram_mode("/tmp/photo.png", {}) is False

    @pytest.mark.unit
    def test_filename_with_arch_keyword(self):
        assert _is_diagram_mode("/tmp/system_arch.png", None) is True

    @pytest.mark.unit
    def test_filename_with_diagram_keyword(self):
        assert _is_diagram_mode("/tmp/network_diagram.png", None) is True

    @pytest.mark.unit
    def test_filename_with_flow_keyword(self):
        assert _is_diagram_mode("/tmp/auth_flow.png", None) is True

    @pytest.mark.unit
    def test_filename_with_uml_keyword(self):
        assert _is_diagram_mode("/tmp/class_uml.jpg", None) is True

    @pytest.mark.unit
    def test_filename_with_schema_keyword(self):
        assert _is_diagram_mode("/tmp/db_schema.png", None) is True

    @pytest.mark.unit
    def test_filename_with_model_keyword(self):
        assert _is_diagram_mode("/tmp/data_model.svg", None) is True

    @pytest.mark.unit
    def test_filename_with_no_keyword_is_false(self):
        assert _is_diagram_mode("/tmp/screenshot.png", None) is False

    @pytest.mark.unit
    def test_non_image_extension_with_keyword_is_false(self):
        # .txt extension is not in _DIAGRAM_IMAGE_EXTENSIONS
        assert _is_diagram_mode("/tmp/arch_notes.txt", None) is False

    @pytest.mark.unit
    def test_likely_diagram_true_overrides_filename(self):
        # Even a non-keyword filename triggers if likely_diagram is set
        assert _is_diagram_mode("/tmp/photo.png", {"likely_diagram": True}) is True

    @pytest.mark.unit
    def test_svg_extension_with_keyword(self):
        assert _is_diagram_mode("/tmp/component_diagram.svg", None) is True


# ─── VisionExtractor.extract() routing ───────────────────────────────────────


class TestVisionExtractorExtractRouting:
    """Tests that extract() routes to diagram or standard mode correctly."""

    def _make_png(self) -> str:
        """Create a minimal PNG temp file."""
        fd, path = tempfile.mkstemp(suffix=".png")
        with os.fdopen(fd, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
        return path

    @pytest.mark.unit
    def test_extract_with_likely_diagram_returns_nodes_edges(self):
        """When likely_diagram is True, extract() returns diagram structure."""
        extractor = VisionExtractor()
        path = self._make_png()

        diagram_response = json.dumps({
            "nodes": [{"id": "svc1", "label": "Service A", "type": "service"}],
            "edges": [{"from": "svc1", "to": "db1", "label": "reads", "type": "data_flow"}],
            "diagram_type": "architecture",
            "description": "A simple service architecture.",
        })

        try:
            with (
                patch("app.services.vision_extractor.settings") as mock_settings,
                patch("anthropic.Anthropic") as mock_anthropic_cls,
            ):
                mock_settings.ANTHROPIC_API_KEY = "test-key"
                mock_client = MagicMock()
                mock_anthropic_cls.return_value = mock_client
                mock_client.messages.create.return_value = MagicMock(
                    content=[MagicMock(text=diagram_response)]
                )

                result = extractor.extract(path, metadata={"likely_diagram": True})
        finally:
            os.unlink(path)

        assert "nodes" in result
        assert "edges" in result
        assert "diagram_type" in result
        assert result["diagram_type"] == "architecture"
        assert len(result["nodes"]) == 1

    @pytest.mark.unit
    def test_extract_without_diagram_flag_returns_entities_relationships(self):
        """Non-diagram image returns standard entities/relationships."""
        extractor = VisionExtractor()
        path = self._make_png()

        standard_response = json.dumps({
            "entities": [{"name": "Alice", "type": "Person", "description": ""}],
            "relationships": [],
        })

        try:
            with (
                patch("app.services.vision_extractor.settings") as mock_settings,
                patch("anthropic.Anthropic") as mock_anthropic_cls,
            ):
                mock_settings.ANTHROPIC_API_KEY = "test-key"
                mock_client = MagicMock()
                mock_anthropic_cls.return_value = mock_client
                mock_client.messages.create.return_value = MagicMock(
                    content=[MagicMock(text=standard_response)]
                )

                result = extractor.extract(path, metadata={"likely_diagram": False})
        finally:
            os.unlink(path)

        assert "entities" in result
        assert "relationships" in result
        assert "nodes" not in result

    @pytest.mark.unit
    def test_extract_filename_heuristic_triggers_diagram_mode(self):
        """PNG named *_arch.png should trigger diagram mode."""
        extractor = VisionExtractor()
        fd, path = tempfile.mkstemp(prefix="system_arch_", suffix=".png")
        with os.fdopen(fd, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        diagram_response = json.dumps({
            "nodes": [],
            "edges": [],
            "diagram_type": "other",
            "description": "Empty diagram.",
        })

        try:
            with (
                patch("app.services.vision_extractor.settings") as mock_settings,
                patch("anthropic.Anthropic") as mock_anthropic_cls,
            ):
                mock_settings.ANTHROPIC_API_KEY = "test-key"
                mock_client = MagicMock()
                mock_anthropic_cls.return_value = mock_client
                mock_client.messages.create.return_value = MagicMock(
                    content=[MagicMock(text=diagram_response)]
                )

                result = extractor.extract(path, metadata=None)
        finally:
            os.unlink(path)

        assert "nodes" in result
        assert "edges" in result

    @pytest.mark.unit
    def test_diagram_extraction_with_bad_json_returns_empty_diagram(self):
        """If LLM returns non-JSON, extract() returns empty nodes/edges gracefully."""
        extractor = VisionExtractor()
        fd, path = tempfile.mkstemp(prefix="arch_", suffix=".png")
        with os.fdopen(fd, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        try:
            with (
                patch("app.services.vision_extractor.settings") as mock_settings,
                patch("anthropic.Anthropic") as mock_anthropic_cls,
            ):
                mock_settings.ANTHROPIC_API_KEY = "test-key"
                mock_client = MagicMock()
                mock_anthropic_cls.return_value = mock_client
                mock_client.messages.create.return_value = MagicMock(
                    content=[MagicMock(text="I cannot extract a diagram from this.")]
                )

                result = extractor.extract(path, metadata={"likely_diagram": True})
        finally:
            os.unlink(path)

        assert result["nodes"] == []
        assert result["edges"] == []
        assert result["diagram_type"] == "other"

    @pytest.mark.unit
    def test_extract_no_metadata_no_keyword_uses_standard_mode(self):
        """A generic image filename without any keywords uses standard entity mode."""
        extractor = VisionExtractor()
        fd, path = tempfile.mkstemp(prefix="photo_", suffix=".png")
        with os.fdopen(fd, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        standard_response = json.dumps({
            "entities": [],
            "relationships": [],
        })

        try:
            with (
                patch("app.services.vision_extractor.settings") as mock_settings,
                patch("anthropic.Anthropic") as mock_anthropic_cls,
            ):
                mock_settings.ANTHROPIC_API_KEY = "test-key"
                mock_client = MagicMock()
                mock_anthropic_cls.return_value = mock_client
                mock_client.messages.create.return_value = MagicMock(
                    content=[MagicMock(text=standard_response)]
                )

                result = extractor.extract(path, metadata=None)
        finally:
            os.unlink(path)

        assert "entities" in result
        assert "relationships" in result
