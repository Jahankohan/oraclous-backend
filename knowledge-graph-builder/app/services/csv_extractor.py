"""
CSV/TSV structured extraction service.

Extracts schema, sample rows, and row count from CSV and TSV files using
Python stdlib only (csv, io). No pandas or external dependencies.
"""

import csv
from datetime import date
from pathlib import Path
from typing import Any


def _infer_type(values: list[str]) -> str:
    """
    Infer the column type from a list of sample string values.

    Precedence: bool → int → float → date → str
    Empty strings are skipped; if all values are empty, returns "str".
    """
    non_empty = [v.strip() for v in values if v.strip()]
    if not non_empty:
        return "str"

    def is_bool(v: str) -> bool:
        return v.lower() in {"true", "false", "yes", "no", "1", "0"}

    def is_int(v: str) -> bool:
        try:
            int(v)
            return True
        except ValueError:
            return False

    def is_float(v: str) -> bool:
        try:
            float(v)
            return True
        except ValueError:
            return False

    def is_date(v: str) -> bool:
        # Accept YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY (most common formats)
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"):
            try:
                (
                    date.fromisoformat(v)
                    if fmt == "%Y-%m-%d"
                    else __import__("datetime").datetime.strptime(v, fmt)
                )
                return True
            except ValueError:
                continue
        return False

    if all(is_bool(v) for v in non_empty):
        return "bool"
    if all(is_int(v) for v in non_empty):
        return "int"
    if all(is_float(v) for v in non_empty):
        return "float"
    if all(is_date(v) for v in non_empty):
        return "date"
    return "str"


def _detect_delimiter(file_path: str) -> str:
    """
    Auto-detect CSV delimiter using csv.Sniffer.

    Falls back to comma if detection fails or the file extension is .csv.
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".tsv":
        return "\t"

    with open(file_path, newline="", encoding="utf-8", errors="replace") as fh:
        sample = fh.read(4096)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        return dialect.delimiter
    except csv.Error:
        return ","


def extract_csv(file_path: str) -> dict[str, Any]:
    """
    Extract schema, sample rows, and row count from a CSV or TSV file.

    Args:
        file_path: Absolute path to the CSV or TSV file.

    Returns:
        {
            "columns": ["col1", "col2", ...],
            "row_count": int,
            "sample_rows": [first 5 rows as list of dicts],
            "schema": {"col_name": "str|int|float|bool|date"}
        }
    """
    delimiter = _detect_delimiter(file_path)

    columns: list[str] = []
    sample_rows: list[dict[str, str]] = []
    row_count = 0

    # Accumulate up to 100 rows per column for type inference
    type_samples: dict[str, list[str]] = {}

    with open(file_path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)

        # DictReader exposes fieldnames after the first row is read
        for row in reader:
            if row_count == 0:
                columns = list(reader.fieldnames or [])
                type_samples = {col: [] for col in columns}

            row_count += 1

            if row_count <= 5:
                sample_rows.append(dict(row))

            if row_count <= 100:
                for col in columns:
                    val = row.get(col, "") or ""
                    type_samples[col].append(val)

    schema: dict[str, str] = {col: _infer_type(type_samples[col]) for col in columns}

    return {
        "columns": columns,
        "row_count": row_count,
        "sample_rows": sample_rows,
        "schema": schema,
    }
