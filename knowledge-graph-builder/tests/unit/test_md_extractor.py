"""
Unit tests for md_extractor.py

Covers:
- Heading hierarchy extraction
- Title detection (first H1 or filename)
- Section content extraction
- Parent-child hierarchy building
- Nested headings
"""

import os
import tempfile

import pytest

from app.services.md_extractor import extract_markdown, extract_markdown_from_text


def _write_md(content: str) -> str:
    """Write content to a temporary .md file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".md")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


# ─── Title detection ─────────────────────────────────────────────────────────


class TestTitleDetection:
    @pytest.mark.unit
    def test_first_h1_is_title(self):
        path = _write_md("# My Document\n\nSome content.\n\n## Section A\n\nText.")
        try:
            result = extract_markdown(path)
        finally:
            os.unlink(path)
        assert result["title"] == "My Document"

    @pytest.mark.unit
    def test_filename_used_when_no_h1(self):
        path = _write_md("## Only an H2\n\nContent here.\n")
        try:
            result = extract_markdown(path)
        finally:
            os.unlink(path)
        # Title falls back to the file's stem (no .md extension)
        assert result["title"] != ""
        assert ".md" not in result["title"]

    @pytest.mark.unit
    def test_first_h1_wins_over_later_h1(self):
        path = _write_md("# First\n\n# Second\n")
        try:
            result = extract_markdown(path)
        finally:
            os.unlink(path)
        assert result["title"] == "First"


class TestExtractMarkdownFromText:
    """The from-text variant — used by document_processor when the caller
    already has the markdown content in memory (no file path)."""

    @pytest.mark.unit
    def test_accepts_raw_text_no_path(self):
        text = "# In-Memory Doc\n\nBody.\n\n## A subsection\n\nMore body.\n"
        result = extract_markdown_from_text(text)
        assert result["title"] == "In-Memory Doc"
        assert len(result["sections"]) == 2
        assert result["sections"][0]["heading"] == "In-Memory Doc"
        assert result["sections"][1]["heading"] == "A subsection"

    @pytest.mark.unit
    def test_uses_fallback_title_when_no_h1(self):
        text = "## Only an H2\n\nContent.\n"
        result = extract_markdown_from_text(text, fallback_title="my-doc")
        assert result["title"] == "my-doc"

    @pytest.mark.unit
    def test_fallback_empty_when_unspecified_and_no_h1(self):
        text = "## Only an H2\n\nContent.\n"
        result = extract_markdown_from_text(text)
        assert result["title"] == ""

    @pytest.mark.unit
    def test_does_not_open_a_file(self, tmp_path):
        # Regression: extract_markdown_from_text must NOT try to open content
        # as a path. Earlier bug treated upload content as a filename and
        # raised OSError [Errno 36] File name too long.
        text = "# Long Heading That Would Be A Bad Path " * 50 + "\n\nbody"
        # Should not raise
        result = extract_markdown_from_text(text)
        assert result["sections"][0]["heading"].startswith("Long Heading")


# ─── Sections extraction ─────────────────────────────────────────────────────


class TestSectionsExtraction:
    @pytest.mark.unit
    def test_sections_list_populated(self):
        md = "# Title\n\nIntro.\n\n## Chapter 1\n\nChapter content.\n\n## Chapter 2\n\nMore.\n"
        path = _write_md(md)
        try:
            result = extract_markdown(path)
        finally:
            os.unlink(path)
        assert len(result["sections"]) == 3

    @pytest.mark.unit
    def test_section_level_correct(self):
        md = "# H1\n\n## H2\n\n### H3\n"
        path = _write_md(md)
        try:
            result = extract_markdown(path)
        finally:
            os.unlink(path)
        levels = [s["level"] for s in result["sections"]]
        assert levels == [1, 2, 3]

    @pytest.mark.unit
    def test_section_heading_correct(self):
        md = "# Introduction\n\nContent.\n"
        path = _write_md(md)
        try:
            result = extract_markdown(path)
        finally:
            os.unlink(path)
        assert result["sections"][0]["heading"] == "Introduction"

    @pytest.mark.unit
    def test_section_content_stripped(self):
        md = "# Title\n\nParagraph one.\n\nParagraph two.\n\n## Next\n\nFoo.\n"
        path = _write_md(md)
        try:
            result = extract_markdown(path)
        finally:
            os.unlink(path)
        # Content under Title should contain the paragraphs
        assert "Paragraph one" in result["sections"][0]["content"]

    @pytest.mark.unit
    def test_empty_document_returns_empty_sections(self):
        path = _write_md("")
        try:
            result = extract_markdown(path)
        finally:
            os.unlink(path)
        assert result["sections"] == []
        assert result["hierarchy"] == []

    @pytest.mark.unit
    def test_no_headings_returns_empty_sections(self):
        path = _write_md("Just plain text with no headings.\n")
        try:
            result = extract_markdown(path)
        finally:
            os.unlink(path)
        assert result["sections"] == []


# ─── Hierarchy building ───────────────────────────────────────────────────────


class TestHierarchyBuilding:
    @pytest.mark.unit
    def test_top_level_sections_have_no_parent(self):
        md = "# Alpha\n\n# Beta\n"
        path = _write_md(md)
        try:
            result = extract_markdown(path)
        finally:
            os.unlink(path)
        for node in result["hierarchy"]:
            assert node["parent"] is None

    @pytest.mark.unit
    def test_h2_parent_is_h1(self):
        md = "# Root\n\n## Child\n"
        path = _write_md(md)
        try:
            result = extract_markdown(path)
        finally:
            os.unlink(path)
        hierarchy = result["hierarchy"]
        child = next(n for n in hierarchy if n["heading"] == "Child")
        assert child["parent"] == "Root"

    @pytest.mark.unit
    def test_h3_parent_is_h2(self):
        md = "# Top\n\n## Middle\n\n### Leaf\n"
        path = _write_md(md)
        try:
            result = extract_markdown(path)
        finally:
            os.unlink(path)
        hierarchy = result["hierarchy"]
        leaf = next(n for n in hierarchy if n["heading"] == "Leaf")
        assert leaf["parent"] == "Middle"

    @pytest.mark.unit
    def test_children_populated(self):
        md = "# Root\n\n## Child A\n\n## Child B\n"
        path = _write_md(md)
        try:
            result = extract_markdown(path)
        finally:
            os.unlink(path)
        root = next(n for n in result["hierarchy"] if n["heading"] == "Root")
        assert "Child A" in root["children"]
        assert "Child B" in root["children"]

    @pytest.mark.unit
    def test_deep_nesting(self):
        md = "# L1\n\n## L2\n\n### L3\n\n#### L4\n"
        path = _write_md(md)
        try:
            result = extract_markdown(path)
        finally:
            os.unlink(path)
        hierarchy = result["hierarchy"]
        assert len(hierarchy) == 4
        l4 = next(n for n in hierarchy if n["heading"] == "L4")
        assert l4["parent"] == "L3"

    @pytest.mark.unit
    def test_sibling_sections_at_h2_same_parent(self):
        md = "# Parent\n\n## A\n\n## B\n\n## C\n"
        path = _write_md(md)
        try:
            result = extract_markdown(path)
        finally:
            os.unlink(path)
        parent_node = next(n for n in result["hierarchy"] if n["heading"] == "Parent")
        assert set(parent_node["children"]) == {"A", "B", "C"}

    @pytest.mark.unit
    def test_hierarchy_all_nodes_present(self):
        md = "# One\n\n## Two\n\n### Three\n"
        path = _write_md(md)
        try:
            result = extract_markdown(path)
        finally:
            os.unlink(path)
        headings = {n["heading"] for n in result["hierarchy"]}
        assert headings == {"One", "Two", "Three"}
