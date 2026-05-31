"""Markdown primitive — wraps `app.services.md_extractor`.

TASK-222 / STORY-034 / ADR-022.

Deterministically decomposes a Markdown document into a
`StructuralRepresentation`:

  * one `DOCUMENT` unit for the document as a whole;
  * one `CHUNK` unit per heading section, with `parent_id` containment that
    mirrors the Markdown heading hierarchy — a deeper heading is parented to
    the nearest shallower heading above it, and a top-level section is
    parented to the document.

`source_type` is `"text"` (recipe-spec §3 — Markdown is a text source).

The adapter does NOT modify `md_extractor`. It calls `extract_markdown_from_text`
(string in) or `extract_markdown` (path in) and translates the flat `sections`
list plus the `hierarchy` tree. In SAMPLE mode each chunk's body is truncated to
a bounded preview; in FULL mode the full section content is emitted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.recipes.primitives.interface import (
    ExtractionMode,
    StructuralRepresentation,
    StructuralUnit,
    UnitKind,
)
from app.services.md_extractor import extract_markdown_from_text

# Bounded length of a section body preview in SAMPLE mode.
_SAMPLE_PREVIEW_CHARS = 280


class MarkdownPrimitive:
    """Adapter turning a Markdown document into a `StructuralRepresentation`."""

    source_type: str = "text"

    def decompose(self, source: Any, mode: ExtractionMode) -> StructuralRepresentation:
        """Decompose a Markdown document into structural units.

        Args:
            source: Either the raw markdown content as a `str`, or a path
                (`Path` / path-like) to a Markdown file on disk.
            mode: SAMPLE truncates chunk bodies to a bounded preview; FULL
                emits the complete section content.
        """
        text, fallback_title = self._resolve_source(source)
        extracted = extract_markdown_from_text(text, fallback_title=fallback_title)
        title: str = extracted["title"]
        sections: list[dict[str, Any]] = extracted["sections"]
        hierarchy: list[dict[str, Any]] = extracted["hierarchy"]

        doc_id = "document"
        units: list[StructuralUnit] = [
            StructuralUnit(
                kind=UnitKind.DOCUMENT,
                unit_id=doc_id,
                name=title,
                metadata={"section_count": len(sections)},
            )
        ]

        # The hierarchy list is parallel to the sections list (both built by
        # md_extractor in document order). Walk them together and resolve each
        # section's parent heading to a chunk unit_id.
        chunk_id_by_index: dict[int, str] = {}
        # Track, per heading text, the most recent chunk unit_id — duplicate
        # headings resolve to the latest, matching md_extractor's own
        # heading_to_node behaviour.
        latest_chunk_for_heading: dict[str, str] = {}

        for idx in range(len(sections)):
            chunk_id_by_index[idx] = f"chunk:{idx}"

        for idx, (section, node) in enumerate(zip(sections, hierarchy, strict=False)):
            chunk_id = chunk_id_by_index[idx]
            parent_heading = node.get("parent")
            parent_id = (
                latest_chunk_for_heading.get(parent_heading, doc_id)
                if parent_heading is not None
                else doc_id
            )

            content: str = section["content"]
            if mode == ExtractionMode.SAMPLE:
                body = content[:_SAMPLE_PREVIEW_CHARS]
            else:
                body = content

            units.append(
                StructuralUnit(
                    kind=UnitKind.CHUNK,
                    unit_id=chunk_id,
                    name=section["heading"],
                    role="free_text",
                    parent_id=parent_id,
                    sample_values=[body] if body else [],
                    metadata={"heading_level": section["level"]},
                )
            )
            latest_chunk_for_heading[section["heading"]] = chunk_id

        return StructuralRepresentation(
            source_type=self.source_type,
            shape_signature=self._shape_signature(title, sections),
            mode=mode,
            units=units,
        )

    @staticmethod
    def _resolve_source(source: Any) -> tuple[str, str]:
        """Return `(markdown_text, fallback_title)` for str or path input."""
        if isinstance(source, Path):
            return (
                source.read_text(encoding="utf-8", errors="replace"),
                source.stem,
            )
        # A plain string is treated as markdown content. If it looks like a
        # path to an existing file, read that file instead.
        if isinstance(source, str):
            candidate = Path(source)
            try:
                if "\n" not in source and candidate.is_file():
                    return (
                        candidate.read_text(encoding="utf-8", errors="replace"),
                        candidate.stem,
                    )
            except OSError:
                pass
            return source, ""
        # Anything else path-like.
        candidate = Path(str(source))
        return (
            candidate.read_text(encoding="utf-8", errors="replace"),
            candidate.stem,
        )

    @staticmethod
    def _shape_signature(title: str, sections: list[dict[str, Any]]) -> str:
        """Deterministic descriptor — heading levels in document order.

        Captures the document's outline shape (recipe-spec §4 lookup key);
        section bodies are excluded so two documents with the same heading
        skeleton match.
        """
        levels = "".join(str(s["level"]) for s in sections)
        return f"text(md:{len(sections)}sections:{levels})"
