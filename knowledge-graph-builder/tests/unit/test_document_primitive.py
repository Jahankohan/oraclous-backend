"""Hermetic unit tests for the document primitive (TASK-222, STORY-034).

Builds real PDF / DOCX / plain-text fixtures in temp files and runs them through
`DocumentPrimitive`. No DB, no Docker — the extraction libraries (PyMuPDF,
python-docx) run in-process.
"""

import pytest

from app.recipes.primitives import DocumentPrimitive
from app.recipes.primitives.interface import ExtractionMode, UnitKind

_LONG_A = "Alpha paragraph. " * 12  # > 100 chars, avoids OCR fallback in PDF
_LONG_B = "Beta paragraph. " * 12


def _make_pdf(path: str) -> None:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for body in (_LONG_A, _LONG_B):
        page = doc.new_page()
        page.insert_text((72, 72), body)
    doc.save(path)
    doc.close()


def _make_docx(path: str) -> None:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph(_LONG_A)
    document.add_paragraph(_LONG_B)
    document.save(path)


def test_docx_decomposes_to_document_plus_chunks(tmp_path):
    path = str(tmp_path / "sample.docx")
    _make_docx(path)

    rep = DocumentPrimitive().decompose(path, ExtractionMode.FULL)

    assert rep.source_type == "text"
    docs = [u for u in rep.units if u.kind is UnitKind.DOCUMENT]
    chunks = [u for u in rep.units if u.kind is UnitKind.CHUNK]
    assert len(docs) == 1 and docs[0].name == "sample.docx"
    assert len(chunks) == 2
    assert all(c.parent_id == docs[0].unit_id for c in chunks)
    assert all(c.role == "free_text" for c in chunks)


def test_pdf_decomposes_to_document_plus_chunks(tmp_path):
    path = str(tmp_path / "sample.pdf")
    _make_pdf(path)

    rep = DocumentPrimitive().decompose(path, ExtractionMode.FULL)

    docs = [u for u in rep.units if u.kind is UnitKind.DOCUMENT]
    chunks = [u for u in rep.units if u.kind is UnitKind.CHUNK]
    assert len(docs) == 1
    assert len(chunks) >= 1  # one per page; PyMuPDF joins pages on \n\n
    assert all(c.parent_id == docs[0].unit_id for c in chunks)


def test_plain_text_is_chunked_on_blank_lines(tmp_path):
    path = str(tmp_path / "notes.txt")
    (tmp_path / "notes.txt").write_text(f"{_LONG_A}\n\n{_LONG_B}", encoding="utf-8")

    rep = DocumentPrimitive().decompose(path, ExtractionMode.FULL)

    chunks = [u for u in rep.units if u.kind is UnitKind.CHUNK]
    assert len(chunks) == 2
    assert chunks[0].metadata["ordinal"] == 0


def test_sample_mode_truncates_chunk_text_but_keeps_structure(tmp_path):
    path = str(tmp_path / "notes.txt")
    long_segment = "x" * 1000
    (tmp_path / "notes.txt").write_text(
        f"{long_segment}\n\n{long_segment}", encoding="utf-8"
    )

    full = DocumentPrimitive().decompose(path, ExtractionMode.FULL)
    sample = DocumentPrimitive().decompose(path, ExtractionMode.SAMPLE)

    full_chunks = [u for u in full.units if u.kind is UnitKind.CHUNK]
    sample_chunks = [u for u in sample.units if u.kind is UnitKind.CHUNK]
    # same structure (same chunk count)...
    assert len(full_chunks) == len(sample_chunks) == 2
    # ...but sample chunk text is bounded, full is not
    assert len(sample_chunks[0].sample_values[0]) < len(full_chunks[0].sample_values[0])


def test_shape_signature_is_stable_across_documents(tmp_path):
    p1, p2 = str(tmp_path / "a.txt"), str(tmp_path / "b.txt")
    (tmp_path / "a.txt").write_text(f"{_LONG_A}\n\n{_LONG_B}", encoding="utf-8")
    (tmp_path / "b.txt").write_text("different content entirely", encoding="utf-8")

    sig1 = DocumentPrimitive().decompose(p1, ExtractionMode.FULL).shape_signature
    sig2 = DocumentPrimitive().decompose(p2, ExtractionMode.FULL).shape_signature
    # all text documents share one shape -> one recipe matches them (recipe-spec §4)
    assert sig1 == sig2 == "text-document"
