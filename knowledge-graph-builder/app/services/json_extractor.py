"""
JSON/JSONL structured extraction service.

Handles single JSON objects, JSON arrays, and JSONL (newline-delimited JSON).
Schema inference recurses into nested objects and detects homogeneous arrays.

Uses Python stdlib only (json, pathlib).
"""

import json
from pathlib import Path
from typing import Any


def _type_name(value: Any) -> str:
    """Return a simple type name for a JSON value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _infer_schema(obj: Any) -> Any:
    """
    Recursively infer a JSON schema from a value.

    - Primitives → type name string
    - Dicts → {"key": inferred_schema(value), ...}
    - Arrays → {"type": "array", "items": merged_schema or type_name}
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return _type_name(obj)

    if isinstance(obj, dict):
        return {k: _infer_schema(v) for k, v in obj.items()}

    if isinstance(obj, list):
        if not obj:
            return {"type": "array", "items": "unknown"}

        # Collect schemas from up to the first 10 items
        item_schemas = [_infer_schema(item) for item in obj[:10]]

        # If all items are dicts, merge their keys
        if all(isinstance(s, dict) for s in item_schemas):
            merged: dict[str, Any] = {}
            for schema in item_schemas:
                for key, val in schema.items():  # type: ignore[union-attr]
                    if key not in merged:
                        merged[key] = val
                    # If schemas differ, keep the first one seen
            return {"type": "array", "items": merged}

        # If all items have the same primitive type, use that
        unique = set(s if isinstance(s, str) else str(s) for s in item_schemas)
        if len(unique) == 1:
            return {"type": "array", "items": item_schemas[0]}

        return {"type": "array", "items": "mixed"}

    return "unknown"


def _merge_schemas(a: Any, b: Any) -> Any:
    """Merge two inferred schemas, preferring the first on conflict."""
    if isinstance(a, dict) and isinstance(b, dict):
        merged = dict(a)
        for key, val in b.items():
            if key not in merged:
                merged[key] = val
            else:
                merged[key] = _merge_schemas(merged[key], val)
        return merged
    if a == b:
        return a
    if a == "unknown":
        return b
    if b == "unknown":
        return a
    return a  # keep first on type conflict


def _load_jsonl(file_path: str) -> list[Any]:
    """Load all records from a JSONL file (one JSON value per line)."""
    records: list[Any] = []
    with open(file_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def extract_json(file_path: str) -> dict[str, Any]:
    """
    Extract schema, record count, and sample records from a JSON or JSONL file.

    Handles:
    - Single JSON object  {}
    - JSON array          [{}, {}]
    - JSONL               one JSON value per line

    Args:
        file_path: Absolute path to the JSON or JSONL file.

    Returns:
        {
            "schema": {...},        # inferred JSON schema
            "record_count": int,    # 1 for object, N for array/JSONL
            "sample_records": [...]  # first 5 records
        }
    """
    ext = Path(file_path).suffix.lower()

    # JSONL: explicit extension or fallback if regular JSON parse fails
    if ext == ".jsonl":
        records = _load_jsonl(file_path)
        schema: Any = {}
        for record in records[:100]:
            schema = _merge_schemas(schema, _infer_schema(record))
        return {
            "schema": schema,
            "record_count": len(records),
            "sample_records": records[:5],
        }

    # Try regular JSON first
    with open(file_path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fall back to JSONL parsing
        records = []
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped:
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError:
                    pass
        schema = {}
        for record in records[:100]:
            schema = _merge_schemas(schema, _infer_schema(record))
        return {
            "schema": schema,
            "record_count": len(records),
            "sample_records": records[:5],
        }

    # JSON array
    if isinstance(data, list):
        schema = {}
        for record in data[:100]:
            schema = _merge_schemas(schema, _infer_schema(record))
        return {
            "schema": schema,
            "record_count": len(data),
            "sample_records": data[:5],
        }

    # Single JSON object (or any other scalar)
    return {
        "schema": _infer_schema(data),
        "record_count": 1,
        "sample_records": [data],
    }
