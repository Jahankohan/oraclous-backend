"""Deterministic, per-data-type primitives for concern-driven ingestion (ADR-022).

A primitive mechanically decomposes a data source into a normalized
`StructuralRepresentation`. See `interface.py` for the contract and TASK-222.

The concrete adapters wrap the existing per-type extractors and conform to the
`Primitive` protocol:

  * `CsvPrimitive`        — CSV / TSV files          (`source_type = "csv"`)
  * `JsonPrimitive`       — JSON / JSONL files       (`source_type = "json"`)
  * `RelationalPrimitive` — relational SchemaSnapshot (`source_type = "relational"`)
  * `CodePrimitive`       — parsed source files      (`source_type = "code"`)
  * `MarkdownPrimitive`   — Markdown documents       (`source_type = "text"`)
  * `DocumentPrimitive`   — PDF / DOCX / plain text   (`source_type = "text"`)

TODO(STORY-034 — vision): a vision/diagram primitive is deliberately NOT
implemented. `vision_extractor.py` depends on an OCR/vision model, so it cannot
meet the deterministic, no-LLM primitive contract. Diagram→graph stays its own
path; revisit only if it must join the recipe pipeline. (Reza: not required for
now — flagged here per request.)
"""

from app.recipes.primitives.code_primitive import CodePrimitive
from app.recipes.primitives.csv_primitive import CsvPrimitive
from app.recipes.primitives.document_primitive import DocumentPrimitive
from app.recipes.primitives.interface import (
    ExtractionMode,
    Primitive,
    StructuralRepresentation,
    StructuralUnit,
    UnitKind,
)
from app.recipes.primitives.json_primitive import JsonPrimitive
from app.recipes.primitives.markdown_primitive import MarkdownPrimitive
from app.recipes.primitives.relational_primitive import RelationalPrimitive

__all__ = [
    "ExtractionMode",
    "Primitive",
    "StructuralRepresentation",
    "StructuralUnit",
    "UnitKind",
    "CsvPrimitive",
    "JsonPrimitive",
    "RelationalPrimitive",
    "CodePrimitive",
    "MarkdownPrimitive",
    "DocumentPrimitive",
]
