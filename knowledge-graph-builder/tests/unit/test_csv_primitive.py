"""Hermetic unit tests for the CSV primitive (TASK-222 / STORY-034).

Pure translation tests — a small in-memory CSV is written to a temp file and
the resulting `StructuralRepresentation` is asserted. No database, no Docker.
"""

import tempfile
from pathlib import Path

import pytest

from app.recipes.primitives import (
    CsvPrimitive,
    ExtractionMode,
    Primitive,
    StructuralRepresentation,
    UnitKind,
)

_CSV = (
    "id,name,age,active\n"
    "1,Alice,30,true\n"
    "2,Bob,25,false\n"
    "3,Carol,41,true\n"
    "4,Dave,19,false\n"
    "5,Erin,55,true\n"
    "6,Frank,33,false\n"
)


def _write_csv(text: str) -> str:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    )
    tmp.write(text)
    tmp.close()
    return tmp.name


@pytest.fixture
def csv_path():
    path = _write_csv(_CSV)
    yield path
    Path(path).unlink(missing_ok=True)


class TestCsvPrimitive:
    def test_conforms_to_primitive_protocol(self):
        assert isinstance(CsvPrimitive(), Primitive)
        assert CsvPrimitive().source_type == "csv"

    def test_sample_mode_structure(self, csv_path):
        rep = CsvPrimitive().decompose(csv_path, ExtractionMode.SAMPLE)
        assert isinstance(rep, StructuralRepresentation)
        assert rep.source_type == "csv"
        assert rep.mode == ExtractionMode.SAMPLE

        sources = [u for u in rep.units if u.kind == UnitKind.SOURCE]
        columns = [u for u in rep.units if u.kind == UnitKind.COLUMN]
        records = [u for u in rep.units if u.kind == UnitKind.RECORD]

        assert len(sources) == 1
        assert len(columns) == 4
        # SAMPLE mode bounds records to the extractor's 5-row sample.
        assert len(records) == 5
        assert sources[0].metadata["row_count"] == 6

    def test_column_data_types_and_role(self, csv_path):
        rep = CsvPrimitive().decompose(csv_path, ExtractionMode.SAMPLE)
        by_name = {u.name: u for u in rep.units if u.kind == UnitKind.COLUMN}

        assert by_name["id"].data_type == "int"
        assert by_name["age"].data_type == "int"
        assert by_name["active"].data_type == "bool"
        assert by_name["name"].data_type == "str"
        # A string column gets the free_text role hint; typed columns do not.
        assert by_name["name"].role == "free_text"
        assert by_name["id"].role is None

    def test_columns_parented_to_source(self, csv_path):
        rep = CsvPrimitive().decompose(csv_path, ExtractionMode.SAMPLE)
        source = next(u for u in rep.units if u.kind == UnitKind.SOURCE)
        for col in (u for u in rep.units if u.kind == UnitKind.COLUMN):
            assert col.parent_id == source.unit_id
        for rec in (u for u in rep.units if u.kind == UnitKind.RECORD):
            assert rec.parent_id == source.unit_id

    def test_full_mode_emits_all_rows(self, csv_path):
        rep = CsvPrimitive().decompose(csv_path, ExtractionMode.FULL)
        records = [u for u in rep.units if u.kind == UnitKind.RECORD]
        assert rep.mode == ExtractionMode.FULL
        # FULL mode emits a record unit for every one of the 6 data rows.
        assert len(records) == 6
        assert records[0].sample_values[0]["name"] == "Alice"
        assert records[5].sample_values[0]["name"] == "Frank"

    def test_shape_signature_deterministic(self, csv_path):
        rep_a = CsvPrimitive().decompose(csv_path, ExtractionMode.SAMPLE)
        rep_b = CsvPrimitive().decompose(csv_path, ExtractionMode.FULL)
        # Signature depends on shape only, not mode.
        assert rep_a.shape_signature == rep_b.shape_signature
        assert "id:int" in rep_a.shape_signature
        assert "name:str" in rep_a.shape_signature
