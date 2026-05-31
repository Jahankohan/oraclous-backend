"""JSON/JSONL primitive — wraps `app.services.json_extractor`.

TASK-222 / STORY-034 / ADR-022.

Deterministically decomposes a JSON object, JSON array, or JSONL file into a
`StructuralRepresentation`:

  * one `SOURCE` unit for the file as a whole;
  * one `RECORD` unit per top-level record (a single object → one record;
    an array / JSONL → one per element), with `parent_id` at the source;
  * one `FIELD` unit per key of the merged record schema, with `parent_id`
    pointing at the source (the field is shared across records of the same
    shape — the recipe engine projects it per record).

The adapter does NOT modify `json_extractor`. It calls `extract_json` for the
schema, record count, and sample records in both modes; in FULL mode it
re-reads the file once so every record becomes a `RECORD` unit (the extractor
caps its own `sample_records` at 5).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.recipes.primitives.interface import (
    ExtractionMode,
    StructuralRepresentation,
    StructuralUnit,
    UnitKind,
)
from app.services.json_extractor import _load_jsonl, extract_json

_SAMPLE_LIMIT = 5


class JsonPrimitive:
    """Adapter turning a JSON/JSONL file into a `StructuralRepresentation`."""

    source_type: str = "json"

    def decompose(self, source: Any, mode: ExtractionMode) -> StructuralRepresentation:
        """Decompose a JSON/JSONL file path into structural units.

        Args:
            source: Absolute path to the JSON or JSONL file (str or Path).
            mode: SAMPLE emits bounded records; FULL emits every record.
        """
        file_path = str(source)
        extracted = extract_json(file_path)
        schema: Any = extracted["schema"]
        record_count: int = extracted["record_count"]
        sample_records: list[Any] = extracted["sample_records"]

        source_id = "source"
        units: list[StructuralUnit] = [
            StructuralUnit(
                kind=UnitKind.SOURCE,
                unit_id=source_id,
                name=Path(file_path).name,
                metadata={"record_count": record_count},
            )
        ]

        # FIELD units — derived from the merged record schema. For a record
        # array/JSONL the inferred schema is {"type": "array", "items": {...}};
        # for a single object it is the object schema itself.
        field_schema = self._record_field_schema(schema)
        for key, ftype in field_schema.items():
            units.append(
                StructuralUnit(
                    kind=UnitKind.FIELD,
                    unit_id=f"field:{key}",
                    name=key,
                    data_type=self._type_label(ftype),
                    role="free_text" if self._type_label(ftype) == "string" else None,
                    parent_id=source_id,
                    sample_values=self._field_samples(sample_records, key),
                )
            )

        # RECORD units.
        records: list[Any]
        if mode == ExtractionMode.FULL:
            records = self._read_all_records(file_path)
        else:
            records = sample_records[:_SAMPLE_LIMIT]

        for idx, record in enumerate(records):
            units.append(
                StructuralUnit(
                    kind=UnitKind.RECORD,
                    unit_id=f"record:{idx}",
                    name=f"record {idx}",
                    parent_id=source_id,
                    sample_values=[record],
                )
            )

        return StructuralRepresentation(
            source_type=self.source_type,
            shape_signature=self._shape_signature(field_schema),
            mode=mode,
            units=units,
        )

    @staticmethod
    def _record_field_schema(schema: Any) -> dict[str, Any]:
        """Return the per-record field schema as a flat {key: type} dict.

        Handles the two inferred-schema shapes `json_extractor` produces:
        an array schema `{"type": "array", "items": {...}}` and a bare object
        schema `{"key": type, ...}`.
        """
        if isinstance(schema, dict) and schema.get("type") == "array":
            items = schema.get("items")
            return items if isinstance(items, dict) else {}
        if isinstance(schema, dict):
            return schema
        return {}

    @staticmethod
    def _type_label(ftype: Any) -> str:
        """Coarse type label for a field — collapses nested schemas."""
        if isinstance(ftype, str):
            return ftype
        if isinstance(ftype, dict) and ftype.get("type") == "array":
            return "array"
        if isinstance(ftype, dict):
            return "object"
        return "unknown"

    @staticmethod
    def _field_samples(records: list[Any], key: str) -> list[Any]:
        """Bounded example values for a field, drawn from sample records."""
        values: list[Any] = []
        for rec in records[:_SAMPLE_LIMIT]:
            if isinstance(rec, dict) and key in rec:
                values.append(rec[key])
        return values

    @staticmethod
    def _read_all_records(file_path: str) -> list[Any]:
        """Read every top-level record from the JSON/JSONL file."""
        ext = Path(file_path).suffix.lower()
        if ext == ".jsonl":
            return _load_jsonl(file_path)

        with open(file_path, encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            records: list[Any] = []
            for line in raw.splitlines():
                stripped = line.strip()
                if stripped:
                    try:
                        records.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        pass
            return records

        if isinstance(data, list):
            return data
        return [data]

    @staticmethod
    def _shape_signature(field_schema: dict[str, Any]) -> str:
        """Deterministic shape descriptor — sorted field keys + coarse types."""
        parts = [
            f"{key}:{JsonPrimitive._type_label(field_schema[key])}"
            for key in sorted(field_schema)
        ]
        return "json(" + ",".join(parts) + ")"
