"""
Unit tests for OCR fallback in pdf_extractor.extract_pdf().

All PyMuPDF, pytesseract, and filesystem calls are mocked so no real PDFs or
Tesseract installation are required.
"""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — build minimal fitz / pytesseract mocks
# ---------------------------------------------------------------------------


def _make_page(text: str, images: list | None = None):
    """Return a mock PyMuPDF page object."""
    page = MagicMock()
    page.get_text.return_value = text
    page.get_images.return_value = images if images is not None else []
    # get_pixmap returns a minimal pixmap-like object
    pixmap = MagicMock()
    pixmap.width = 800
    pixmap.height = 600
    pixmap.samples = b"\x00" * (800 * 600 * 3)
    page.get_pixmap.return_value = pixmap
    return page


def _make_doc(pages: list, page_count: int | None = None):
    """Return a mock fitz.Document."""
    doc = MagicMock()
    doc.page_count = page_count if page_count is not None else len(pages)
    doc.__iter__ = MagicMock(return_value=iter(pages))
    doc.extract_image.return_value = {"width": 0, "height": 0, "image": b""}
    return doc


# ---------------------------------------------------------------------------
# Patches applied to every test in this module
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_filesystem(tmp_path, monkeypatch):
    """Redirect img_dir creation to a temp directory so nothing hits disk."""
    import app.services.pdf_extractor as mod

    orig_path = mod.Path

    def fake_path(p):
        np = orig_path(tmp_path) / "extracted_images"
        np.mkdir(parents=True, exist_ok=True)

        class _FakePath:
            def __truediv__(self, other):
                return tmp_path / other

            def mkdir(self, **kw):
                pass

            @property
            def parent(self):
                return _FakePath()

        return _FakePath()

    # We only need mkdir to not fail; easiest is to patch Path.mkdir directly.
    monkeypatch.setattr(mod.Path, "mkdir", lambda self, **kw: None)


# ---------------------------------------------------------------------------
# Test 1 — short text → OCR called, ocr_used: True in metadata
# ---------------------------------------------------------------------------


def test_short_page_triggers_ocr():
    """When PyMuPDF returns <100 chars, pytesseract.image_to_string is called."""
    short_text = "A" * 10  # well below threshold
    page = _make_page(short_text)
    doc = _make_doc([page])

    ocr_result = "OCR extracted text from scanned page.\n"

    with (
        patch("fitz.open", return_value=doc),
        patch("app.services.pdf_extractor._OCR_AVAILABLE", True),
        patch("app.services.pdf_extractor.pytesseract") as mock_tess,
        patch("app.services.pdf_extractor._page_to_pil", return_value=MagicMock()),
        patch("pdfplumber.open", side_effect=ImportError),
    ):
        mock_tess.image_to_string.return_value = ocr_result

        from app.services.pdf_extractor import extract_pdf

        result = extract_pdf("/fake/doc.pdf")

    mock_tess.image_to_string.assert_called_once()
    assert result["metadata"]["ocr_used"] is True
    assert result["pages"][0]["ocr_used"] is True
    assert ocr_result.strip() in result["text"]


# ---------------------------------------------------------------------------
# Test 2 — long text → OCR NOT called
# ---------------------------------------------------------------------------


def test_long_page_skips_ocr():
    """When PyMuPDF returns ≥100 chars, pytesseract must not be called."""
    long_text = "B" * 200  # above threshold
    page = _make_page(long_text)
    doc = _make_doc([page])

    with (
        patch("fitz.open", return_value=doc),
        patch("app.services.pdf_extractor._OCR_AVAILABLE", True),
        patch("app.services.pdf_extractor.pytesseract") as mock_tess,
        patch("app.services.pdf_extractor._page_to_pil", return_value=MagicMock()),
        patch("pdfplumber.open", side_effect=ImportError),
    ):
        from app.services.pdf_extractor import extract_pdf

        result = extract_pdf("/fake/doc.pdf")

    mock_tess.image_to_string.assert_not_called()
    assert result["metadata"]["ocr_used"] is False
    assert result["pages"][0]["ocr_used"] is False


# ---------------------------------------------------------------------------
# Test 3 — short text + embedded images → likely_diagram: True
# ---------------------------------------------------------------------------


def test_short_text_with_images_flags_likely_diagram():
    """
    A page with <50 chars after OCR AND embedded images must have
    likely_diagram: True in its per-page metadata.
    """
    # OCR returns very short text so total is still < 50 chars
    short_page_text = ""
    fake_image_ref = (1, 0, 0, 0, 0, 0, 0, 0)
    page = _make_page(short_page_text, images=[fake_image_ref])
    doc = _make_doc([page])

    with (
        patch("fitz.open", return_value=doc),
        patch("app.services.pdf_extractor._OCR_AVAILABLE", True),
        patch("app.services.pdf_extractor.pytesseract") as mock_tess,
        patch("app.services.pdf_extractor._page_to_pil", return_value=MagicMock()),
        patch("pdfplumber.open", side_effect=ImportError),
    ):
        # OCR also returns very short text → still below diagram threshold
        mock_tess.image_to_string.return_value = "Fig"

        from app.services.pdf_extractor import extract_pdf

        result = extract_pdf("/fake/doc.pdf")

    assert result["pages"][0]["likely_diagram"] is True


# ---------------------------------------------------------------------------
# Test 4 — long text page with images → NOT flagged as likely_diagram
# ---------------------------------------------------------------------------


def test_long_text_with_images_not_flagged_as_diagram():
    """A page with ≥100 chars is never flagged as a diagram, even with images."""
    long_text = "C" * 200
    fake_image_ref = (1, 0, 0, 0, 0, 0, 0, 0)
    page = _make_page(long_text, images=[fake_image_ref])
    doc = _make_doc([page])

    with (
        patch("fitz.open", return_value=doc),
        patch("app.services.pdf_extractor._OCR_AVAILABLE", True),
        patch("app.services.pdf_extractor.pytesseract") as mock_tess,
        patch("app.services.pdf_extractor._page_to_pil", return_value=MagicMock()),
        patch("pdfplumber.open", side_effect=ImportError),
    ):
        from app.services.pdf_extractor import extract_pdf

        result = extract_pdf("/fake/doc.pdf")

    mock_tess.image_to_string.assert_not_called()
    assert result["pages"][0]["likely_diagram"] is False


# ---------------------------------------------------------------------------
# Test 5 — pytesseract ImportError → graceful skip, no exception raised
# ---------------------------------------------------------------------------


def test_ocr_unavailable_skips_gracefully():
    """
    When _OCR_AVAILABLE is False (pytesseract not importable), extract_pdf must
    complete without raising an exception and must NOT set ocr_used: True.
    """
    short_text = "X" * 5
    page = _make_page(short_text)
    doc = _make_doc([page])

    with (
        patch("fitz.open", return_value=doc),
        patch("app.services.pdf_extractor._OCR_AVAILABLE", False),
        patch("pdfplumber.open", side_effect=ImportError),
    ):
        from app.services.pdf_extractor import extract_pdf

        # Must not raise
        result = extract_pdf("/fake/doc.pdf")

    assert result["metadata"]["ocr_used"] is False
    assert result["pages"][0]["ocr_used"] is False


# ---------------------------------------------------------------------------
# Test 6 — OCR exception on a page → graceful skip, processing continues
# ---------------------------------------------------------------------------


def test_ocr_exception_on_page_is_swallowed():
    """
    If pytesseract.image_to_string raises, the page is skipped gracefully and
    the rest of the document is still returned.
    """
    pages = [_make_page("short"), _make_page("D" * 200)]
    doc = _make_doc(pages)

    with (
        patch("fitz.open", return_value=doc),
        patch("app.services.pdf_extractor._OCR_AVAILABLE", True),
        patch("app.services.pdf_extractor.pytesseract") as mock_tess,
        patch("app.services.pdf_extractor._page_to_pil", return_value=MagicMock()),
        patch("pdfplumber.open", side_effect=ImportError),
    ):
        mock_tess.image_to_string.side_effect = RuntimeError("tesseract crashed")

        from app.services.pdf_extractor import extract_pdf

        result = extract_pdf("/fake/doc.pdf")

    # The second page (long text) must still appear in the output
    assert "D" * 200 in result["text"]
    # OCR was attempted (and failed) on page 1, ocr_used_any remains False
    assert result["metadata"]["ocr_used"] is False


# ---------------------------------------------------------------------------
# Test 7 — pages key present in result and contains correct structure
# ---------------------------------------------------------------------------


def test_result_includes_pages_metadata():
    """extract_pdf always returns a 'pages' list with per-page dicts."""
    pages = [_make_page("Hello world " * 10), _make_page("Z" * 5)]
    doc = _make_doc(pages)

    with (
        patch("fitz.open", return_value=doc),
        patch("app.services.pdf_extractor._OCR_AVAILABLE", True),
        patch("app.services.pdf_extractor.pytesseract") as mock_tess,
        patch("app.services.pdf_extractor._page_to_pil", return_value=MagicMock()),
        patch("pdfplumber.open", side_effect=ImportError),
    ):
        mock_tess.image_to_string.return_value = "ocr text\n"

        from app.services.pdf_extractor import extract_pdf

        result = extract_pdf("/fake/doc.pdf")

    assert "pages" in result
    assert len(result["pages"]) == 2
    for meta in result["pages"]:
        assert "page_number" in meta
        assert "ocr_used" in meta
        assert "likely_diagram" in meta
