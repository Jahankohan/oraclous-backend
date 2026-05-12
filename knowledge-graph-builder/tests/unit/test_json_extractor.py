"""
Unit tests for json_extractor.py

Covers:
- Single JSON object  {}
- JSON array          [{}, {}]
- JSONL               one JSON value per line
- Schema inference with nested objects
- record_count correctness
"""

import json
import os
import tempfile

import pytest

from app.services.json_extractor import _infer_schema, extract_json

# ─── _infer_schema unit tests ─────────────────────────────────────────────────


class TestInferSchema:
    @pytest.mark.unit
    def test_string_returns_string(self):
        assert _infer_schema("hello") == "string"

    @pytest.mark.unit
    def test_int_returns_integer(self):
        assert _infer_schema(42) == "integer"

    @pytest.mark.unit
    def test_float_returns_number(self):
        assert _infer_schema(3.14) == "number"

    @pytest.mark.unit
    def test_bool_returns_boolean(self):
        assert _infer_schema(True) == "boolean"

    @pytest.mark.unit
    def test_none_returns_null(self):
        assert _infer_schema(None) == "null"

    @pytest.mark.unit
    def test_dict_recurses(self):
        schema = _infer_schema({"name": "Alice", "age": 30})
        assert schema == {"name": "string", "age": "integer"}

    @pytest.mark.unit
    def test_nested_dict(self):
        schema = _infer_schema({"user": {"id": 1, "email": "a@b.com"}})
        assert schema == {"user": {"id": "integer", "email": "string"}}

    @pytest.mark.unit
    def test_empty_list(self):
        schema = _infer_schema([])
        assert schema == {"type": "array", "items": "unknown"}

    @pytest.mark.unit
    def test_homogeneous_int_list(self):
        schema = _infer_schema([1, 2, 3])
        assert schema == {"type": "array", "items": "integer"}

    @pytest.mark.unit
    def test_list_of_dicts(self):
        schema = _infer_schema([{"id": 1}, {"id": 2}])
        assert schema["type"] == "array"
        assert schema["items"] == {"id": "integer"}


# ─── extract_json: single object ─────────────────────────────────────────────


class TestExtractJSONObject:
    def _write(self, data, suffix=".json") -> str:
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        return path

    @pytest.mark.unit
    def test_single_object_record_count_is_1(self):
        path = self._write({"name": "Alice", "age": 30})
        try:
            result = extract_json(path)
        finally:
            os.unlink(path)
        assert result["record_count"] == 1

    @pytest.mark.unit
    def test_single_object_sample_records_contains_object(self):
        obj = {"name": "Alice", "age": 30}
        path = self._write(obj)
        try:
            result = extract_json(path)
        finally:
            os.unlink(path)
        assert result["sample_records"] == [obj]

    @pytest.mark.unit
    def test_single_object_schema(self):
        path = self._write({"x": 1, "y": "hello"})
        try:
            result = extract_json(path)
        finally:
            os.unlink(path)
        assert result["schema"] == {"x": "integer", "y": "string"}


# ─── extract_json: array ──────────────────────────────────────────────────────


class TestExtractJSONArray:
    def _write(self, data, suffix=".json") -> str:
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        return path

    @pytest.mark.unit
    def test_array_record_count(self):
        data = [{"id": i} for i in range(20)]
        path = self._write(data)
        try:
            result = extract_json(path)
        finally:
            os.unlink(path)
        assert result["record_count"] == 20

    @pytest.mark.unit
    def test_array_sample_records_capped_at_5(self):
        data = [{"id": i} for i in range(100)]
        path = self._write(data)
        try:
            result = extract_json(path)
        finally:
            os.unlink(path)
        assert len(result["sample_records"]) == 5
        assert result["sample_records"][0] == {"id": 0}

    @pytest.mark.unit
    def test_array_schema_inferred(self):
        data = [{"name": "Alice", "score": 95.5}, {"name": "Bob", "score": 88.0}]
        path = self._write(data)
        try:
            result = extract_json(path)
        finally:
            os.unlink(path)
        assert result["schema"]["name"] == "string"
        assert result["schema"]["score"] == "number"

    @pytest.mark.unit
    def test_empty_array(self):
        path = self._write([])
        try:
            result = extract_json(path)
        finally:
            os.unlink(path)
        assert result["record_count"] == 0
        assert result["sample_records"] == []


# ─── extract_json: JSONL ──────────────────────────────────────────────────────


class TestExtractJSONL:
    def _write_jsonl(self, records: list, suffix=".jsonl") -> str:
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        return path

    @pytest.mark.unit
    def test_jsonl_record_count(self):
        records = [{"id": i, "val": f"v{i}"} for i in range(30)]
        path = self._write_jsonl(records)
        try:
            result = extract_json(path)
        finally:
            os.unlink(path)
        assert result["record_count"] == 30

    @pytest.mark.unit
    def test_jsonl_sample_records(self):
        records = [{"id": i} for i in range(10)]
        path = self._write_jsonl(records)
        try:
            result = extract_json(path)
        finally:
            os.unlink(path)
        assert result["sample_records"] == records[:5]

    @pytest.mark.unit
    def test_jsonl_schema_merged(self):
        records = [{"x": 1}, {"x": 2, "y": "hello"}]
        path = self._write_jsonl(records)
        try:
            result = extract_json(path)
        finally:
            os.unlink(path)
        # x is present in all; y added from second record
        assert "x" in result["schema"]
        assert "y" in result["schema"]

    @pytest.mark.unit
    def test_jsonl_blank_lines_ignored(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write('{"id": 1}\n\n{"id": 2}\n')
        try:
            result = extract_json(path)
        finally:
            os.unlink(path)
        assert result["record_count"] == 2

    @pytest.mark.unit
    def test_json_file_with_jsonl_content_fallback(self):
        """A .json file containing JSONL is parsed via fallback."""
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write('{"a": 1}\n{"a": 2}\n')
        try:
            result = extract_json(path)
        finally:
            os.unlink(path)
        assert result["record_count"] == 2
