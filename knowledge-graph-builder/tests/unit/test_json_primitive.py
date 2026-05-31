"""Hermetic unit tests for the JSON primitive (TASK-222 / STORY-034).

Pure translation tests — small in-memory JSON / JSONL is written to a temp file
and the resulting `StructuralRepresentation` is asserted. No database.
"""

import json
import tempfile
from pathlib import Path

import pytest

from app.recipes.primitives import (
    ExtractionMode,
    JsonPrimitive,
    Primitive,
    StructuralRepresentation,
    UnitKind,
)

_RECORDS = [
    {"id": 1, "name": "Alice", "score": 9.5, "active": True},
    {"id": 2, "name": "Bob", "score": 7.0, "active": False},
    {"id": 3, "name": "Carol", "score": 8.2, "active": True},
    {"id": 4, "name": "Dave", "score": 6.1, "active": False},
    {"id": 5, "name": "Erin", "score": 9.9, "active": True},
    {"id": 6, "name": "Frank", "score": 5.5, "active": False},
]


def _write(text: str, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    )
    tmp.write(text)
    tmp.close()
    return tmp.name


@pytest.fixture
def json_array_path():
    path = _write(json.dumps(_RECORDS), ".json")
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def jsonl_path():
    path = _write("\n".join(json.dumps(r) for r in _RECORDS), ".jsonl")
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def single_object_path():
    path = _write(json.dumps(_RECORDS[0]), ".json")
    yield path
    Path(path).unlink(missing_ok=True)


class TestJsonPrimitive:
    def test_conforms_to_primitive_protocol(self):
        assert isinstance(JsonPrimitive(), Primitive)
        assert JsonPrimitive().source_type == "json"

    def test_array_sample_mode_structure(self, json_array_path):
        rep = JsonPrimitive().decompose(json_array_path, ExtractionMode.SAMPLE)
        assert isinstance(rep, StructuralRepresentation)
        assert rep.source_type == "json"

        sources = [u for u in rep.units if u.kind == UnitKind.SOURCE]
        fields = [u for u in rep.units if u.kind == UnitKind.FIELD]
        records = [u for u in rep.units if u.kind == UnitKind.RECORD]

        assert len(sources) == 1
        assert {f.name for f in fields} == {"id", "name", "score", "active"}
        # SAMPLE mode bounds records to the extractor's 5-record sample.
        assert len(records) == 5
        assert sources[0].metadata["record_count"] == 6

    def test_field_types_and_containment(self, json_array_path):
        rep = JsonPrimitive().decompose(json_array_path, ExtractionMode.SAMPLE)
        source = next(u for u in rep.units if u.kind == UnitKind.SOURCE)
        by_name = {u.name: u for u in rep.units if u.kind == UnitKind.FIELD}

        assert by_name["id"].data_type == "integer"
        assert by_name["name"].data_type == "string"
        assert by_name["score"].data_type == "number"
        assert by_name["active"].data_type == "boolean"
        assert by_name["name"].role == "free_text"
        # FIELD and RECORD units are parented to the source.
        for f in by_name.values():
            assert f.parent_id == source.unit_id
        for rec in (u for u in rep.units if u.kind == UnitKind.RECORD):
            assert rec.parent_id == source.unit_id

    def test_full_mode_emits_all_records(self, json_array_path):
        rep = JsonPrimitive().decompose(json_array_path, ExtractionMode.FULL)
        records = [u for u in rep.units if u.kind == UnitKind.RECORD]
        assert rep.mode == ExtractionMode.FULL
        assert len(records) == 6
        assert records[0].sample_values[0]["name"] == "Alice"
        assert records[5].sample_values[0]["name"] == "Frank"

    def test_jsonl_full_mode(self, jsonl_path):
        rep = JsonPrimitive().decompose(jsonl_path, ExtractionMode.FULL)
        records = [u for u in rep.units if u.kind == UnitKind.RECORD]
        fields = [u for u in rep.units if u.kind == UnitKind.FIELD]
        assert len(records) == 6
        assert {f.name for f in fields} == {"id", "name", "score", "active"}

    def test_single_object_yields_one_record(self, single_object_path):
        rep = JsonPrimitive().decompose(single_object_path, ExtractionMode.FULL)
        records = [u for u in rep.units if u.kind == UnitKind.RECORD]
        fields = [u for u in rep.units if u.kind == UnitKind.FIELD]
        assert len(records) == 1
        assert {f.name for f in fields} == {"id", "name", "score", "active"}

    def test_shape_signature_stable(self, json_array_path, jsonl_path):
        # An array file and a JSONL file with the same record shape match.
        a = JsonPrimitive().decompose(json_array_path, ExtractionMode.SAMPLE)
        b = JsonPrimitive().decompose(jsonl_path, ExtractionMode.FULL)
        assert a.shape_signature == b.shape_signature
        assert "name:string" in a.shape_signature
