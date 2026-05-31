"""Document primitive — PDF / DOCX / plain-text → structural units.

TASK-222 / STORY-034 / ADR-022.

Wraps the deterministic text extractors `pdf_extractor.extract_pdf` /
`extract_docx` (and a plain-text read). A document becomes one ``DOCUMENT`` unit
plus ``CHUNK`` units — one per ``\\n\\n``-delimited segment, which is the
natural grain the extractors already produce (pages for PDF; paragraphs and
tables for DOCX). Deterministic — no LLM.

`source_type` is ``"text"`` — the same family as the markdown primitive; a
dispatcher selects the right primitive by file type, not by `source_type`.
"""

import os
from typing import Any

from app.recipes.primitives.interface import (
    ExtractionMode,
    StructuralRepresentation,
    StructuralUnit,
    UnitKind,
)
from app.services.pdf_extractor import extract_docx, extract_pdf

_SAMPLE_PREVIEW_CHARS = 280


def _document_unit_id(name: str) -> str:
    return f"document:{name}"


class DocumentPrimitive:
    """Adapter turning a PDF / DOCX / plain-text file into a representation."""

    source_type: str = "text"

    def decompose(self, source: Any, mode: ExtractionMode) -> StructuralRepresentation:
        """Decompose a document file into a ``DOCUMENT`` unit + ``CHUNK`` units.

        Args:
            source: An absolute path to a ``.pdf``, ``.docx`` / ``.doc``, or
                plain-text (``.txt`` and the like) file.
            mode: In ``SAMPLE`` mode each chunk's text is truncated to a
                preview; in ``FULL`` mode the whole text is kept. The set of
                chunks (the structure) is the same in both modes.
        """
        path = str(source)
        ext = os.path.splitext(path)[1].lower()
        name = os.path.basename(path)

        if ext == ".pdf":
            extracted = extract_pdf(path)
        elif ext in (".docx", ".doc"):
            extracted = extract_docx(path)
        else:
            with open(path, encoding="utf-8", errors="replace") as handle:
                extracted = {
                    "text": handle.read(),
                    "metadata": {"content_type": "text/plain"},
                }

        text: str = extracted.get("text", "") or ""
        doc_metadata: dict[str, Any] = dict(extracted.get("metadata", {}))

        doc_id = _document_unit_id(name)
        units: list[StructuralUnit] = [
            StructuralUnit(
                kind=UnitKind.DOCUMENT,
                unit_id=doc_id,
                name=name,
                metadata=doc_metadata,
            )
        ]

        segments = [seg.strip() for seg in text.split("\n\n") if seg.strip()]
        for idx, segment in enumerate(segments):
            body = (
                segment[:_SAMPLE_PREVIEW_CHARS]
                if mode == ExtractionMode.SAMPLE
                else segment
            )
            units.append(
                StructuralUnit(
                    kind=UnitKind.CHUNK,
                    unit_id=f"chunk:{idx}",
                    parent_id=doc_id,
                    role="free_text",
                    sample_values=[body],
                    metadata={"ordinal": idx},
                )
            )

        return StructuralRepresentation(
            source_type=self.source_type,
            shape_signature="text-document",
            mode=mode,
            units=units,
        )
