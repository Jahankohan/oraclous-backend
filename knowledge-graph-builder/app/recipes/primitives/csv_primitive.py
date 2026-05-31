"""CSV/TSV primitive — wraps `app.services.csv_extractor`.

TASK-222 / STORY-034 / ADR-022.

Deterministically decomposes a CSV or TSV file into a
`StructuralRepresentation`:

  * one `SOURCE` unit for the file as a whole;
  * one `COLUMN` unit per column, carrying the inferred `data_type` and a
    coarse `role` hint (`free_text` for string columns);
  * one `RECORD` unit per data row, with `parent_id` pointing at the source.

The adapter does NOT modify `csv_extractor`. It calls `extract_csv` for the
schema/columns/sample-rows in both modes, and — only in FULL mode — re-reads
the file once to emit a `RECORD` unit for every row (the extractor caps its
own `sample_rows` at 5).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from app.recipes.primitives.interface import (
    ExtractionMode,
    StructuralRepresentation,
    StructuralUnit,
    UnitKind,
)
from app.services.csv_extractor import _detect_delimiter, extract_csv

# Bounded number of example values per column / records in SAMPLE mode.
_SAMPLE_LIMIT = 5


class CsvPrimitive:
    """Adapter turning a CSV/TSV file into a `StructuralRepresentation`."""

    source_type: str = "csv"

    def decompose(self, source: Any, mode: ExtractionMode) -> StructuralRepresentation:
        """Decompose a CSV/TSV file path into structural units.

        Args:
            source: Absolute path to the CSV or TSV file (str or Path).
            mode: SAMPLE emits bounded example values; FULL emits every row.
        """
        file_path = str(source)
        extracted = extract_csv(file_path)
        columns: list[str] = extracted["columns"]
        schema: dict[str, str] = extracted["schema"]
        sample_rows: list[dict[str, str]] = extracted["sample_rows"]
        row_count: int = extracted["row_count"]

        source_id = "source"
        units: list[StructuralUnit] = [
            StructuralUnit(
                kind=UnitKind.SOURCE,
                unit_id=source_id,
                name=Path(file_path).name,
                metadata={
                    "row_count": row_count,
                    "column_count": len(columns),
                },
            )
        ]

        # One COLUMN unit per column.
        for col in columns:
            data_type = schema.get(col, "str")
            col_samples = [
                row.get(col, "") for row in sample_rows[:_SAMPLE_LIMIT] if col in row
            ]
            units.append(
                StructuralUnit(
                    kind=UnitKind.COLUMN,
                    unit_id=f"column:{col}",
                    name=col,
                    data_type=data_type,
                    # A string column is free text by default; recipes refine.
                    role="free_text" if data_type == "str" else None,
                    parent_id=source_id,
                    sample_values=col_samples,
                )
            )

        # RECORD units. In SAMPLE mode use the extractor's bounded sample_rows;
        # in FULL mode re-read the file to emit every row.
        record_rows: list[dict[str, str]]
        if mode == ExtractionMode.FULL:
            record_rows = self._read_all_rows(file_path)
        else:
            record_rows = sample_rows[:_SAMPLE_LIMIT]

        for idx, row in enumerate(record_rows):
            units.append(
                StructuralUnit(
                    kind=UnitKind.RECORD,
                    unit_id=f"record:{idx}",
                    name=f"row {idx}",
                    parent_id=source_id,
                    sample_values=[dict(row)],
                )
            )

        signature = self._shape_signature(columns, schema)
        return StructuralRepresentation(
            source_type=self.source_type,
            shape_signature=signature,
            mode=mode,
            units=units,
        )

    @staticmethod
    def _read_all_rows(file_path: str) -> list[dict[str, str]]:
        """Read every data row from the CSV/TSV file as ordered dicts."""
        delimiter = _detect_delimiter(file_path)
        rows: list[dict[str, str]] = []
        with open(file_path, newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            for row in reader:
                rows.append(dict(row))
        return rows

    @staticmethod
    def _shape_signature(columns: list[str], schema: dict[str, str]) -> str:
        """Deterministic shape descriptor — column names + inferred types.

        Order-preserving so two CSVs with the same header and inferred schema
        produce the same signature (recipe-spec §4 lookup key).
        """
        parts = [f"{col}:{schema.get(col, 'str')}" for col in columns]
        return "csv(" + ",".join(parts) + ")"
