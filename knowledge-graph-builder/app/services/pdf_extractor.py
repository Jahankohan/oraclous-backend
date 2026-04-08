"""
PDF and DOCX extraction service.

Extracts text content (plus embedded images) from PDF and DOCX files.

Libraries used:
- PyMuPDF (fitz)   — primary PDF parser; text + image extraction
- pdfplumber       — table extraction (optional, gracefully skipped if absent)
- python-docx      — DOCX text and table extraction
"""

from pathlib import Path
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

# Minimum image dimensions to bother extracting (avoids tiny decorative images).
_MIN_IMAGE_PX = 50


def extract_pdf(file_path: str) -> dict[str, Any]:
    """
    Extract text and embedded images from a PDF file.

    Args:
        file_path: Absolute path to the PDF.

    Returns:
        {
            "text": str,            # full text across all pages
            "page_count": int,
            "has_tables": bool,
            "image_paths": list[str],  # PNG files written alongside the source
            "metadata": dict,
        }
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError(
            "PyMuPDF is required for PDF extraction. "
            "Install it with: pip install pymupdf"
        )

    doc = fitz.open(file_path)
    page_count = doc.page_count
    text_parts: list[str] = []
    image_paths: list[str] = []
    has_tables = False

    img_dir = Path(file_path).parent / "extracted_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-page text + image extraction ─────────────────────────────────────
    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")
        if page_text.strip():
            text_parts.append(f"[Page {page_num + 1}]\n{page_text.strip()}")

        for img_index, img_ref in enumerate(page.get_images(full=True)):
            xref = img_ref[0]
            base_image = doc.extract_image(xref)
            if (
                base_image.get("width", 0) < _MIN_IMAGE_PX
                or base_image.get("height", 0) < _MIN_IMAGE_PX
            ):
                continue  # skip decorative/tiny images

            img_path = img_dir / f"page{page_num + 1}_img{img_index}.png"
            with open(img_path, "wb") as fh:
                fh.write(base_image["image"])
            image_paths.append(str(img_path))

    doc.close()

    # ── Optional table extraction via pdfplumber ──────────────────────────────
    try:
        import pdfplumber

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    has_tables = True
                    rows = [
                        "\t".join(str(cell) if cell else "" for cell in row)
                        for row in table
                        if row
                    ]
                    text_parts.append("[Table]\n" + "\n".join(rows))
    except ImportError:
        logger.debug("pdfplumber not available — table extraction skipped")
    except Exception as exc:
        logger.warning(f"Table extraction failed: {exc}")

    return {
        "text": "\n\n".join(text_parts),
        "page_count": page_count,
        "has_tables": has_tables,
        "image_paths": image_paths,
        "metadata": {
            "page_count": page_count,
            "content_type": "application/pdf",
            "processing_method": "pymupdf",
            "has_embedded_images": len(image_paths) > 0,
        },
    }


def extract_docx(file_path: str) -> dict[str, Any]:
    """
    Extract text and tables from a DOCX file using python-docx.

    Args:
        file_path: Absolute path to the DOCX file.

    Returns:
        {
            "text": str,
            "metadata": dict,
        }
    """
    try:
        import docx
    except ImportError:
        raise RuntimeError(
            "python-docx is required for DOCX extraction. "
            "Install it with: pip install python-docx"
        )

    doc = docx.Document(file_path)
    text_parts: list[str] = []

    for para in doc.paragraphs:
        stripped = para.text.strip()
        if stripped:
            text_parts.append(stripped)

    for table in doc.tables:
        rows = [
            "\t".join(cell.text.strip() for cell in row.cells) for row in table.rows
        ]
        non_empty = [r for r in rows if r.strip()]
        if non_empty:
            text_parts.append("[Table]\n" + "\n".join(non_empty))

    return {
        "text": "\n\n".join(text_parts),
        "metadata": {
            "content_type": (
                "application/vnd.openxmlformats-officedocument"
                ".wordprocessingml.document"
            ),
            "processing_method": "python-docx",
        },
    }
