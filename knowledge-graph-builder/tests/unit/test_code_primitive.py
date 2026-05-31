"""Hermetic unit tests for the code primitive (TASK-222 / STORY-034).

Pure translation tests — a tiny Python snippet is written to a temp file,
parsed via the AST stage, and the resulting `StructuralRepresentation` is
asserted. No database, no Docker. Skips cleanly if tree-sitter is unavailable.
"""

import tempfile
from pathlib import Path

import pytest

from app.recipes.primitives import (
    CodePrimitive,
    ExtractionMode,
    Primitive,
    StructuralRepresentation,
    UnitKind,
)
from app.recipes.primitives.code_primitive import file_metadata_from_path
from app.services.code_parser_service import _get_parser

_SNIPPET = '''\
"""A tiny module for primitive translation tests."""
import os


class Greeter:
    """Greets people."""

    def greet(self, name):
        return helper(name)


def helper(name):
    return os.getenv(name)
'''


def _has_python_parser() -> bool:
    return _get_parser("python") is not None


@pytest.fixture
def py_file():
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    )
    tmp.write(_SNIPPET)
    tmp.close()
    yield tmp.name
    Path(tmp.name).unlink(missing_ok=True)


@pytest.mark.skipif(
    not _has_python_parser(), reason="tree-sitter python grammar not installed"
)
class TestCodePrimitive:
    def test_conforms_to_primitive_protocol(self):
        assert isinstance(CodePrimitive(), Primitive)
        assert CodePrimitive().source_type == "code"

    def test_file_and_symbol_units(self, py_file):
        meta = file_metadata_from_path(py_file)
        rep = CodePrimitive().decompose(meta, ExtractionMode.FULL)
        assert isinstance(rep, StructuralRepresentation)
        assert rep.source_type == "code"

        files = [u for u in rep.units if u.kind == UnitKind.FILE]
        symbols = [u for u in rep.units if u.kind == UnitKind.SYMBOL]
        assert len(files) == 1
        assert files[0].data_type == "python"

        # Expect: Greeter class, greet method, helper function (plus Module
        # markers for the `import os` statement).
        names = {s.name for s in symbols}
        assert "Greeter" in names
        assert "greet" in names
        assert "helper" in names

    def test_method_parented_to_its_class(self, py_file):
        meta = file_metadata_from_path(py_file)
        rep = CodePrimitive().decompose(meta, ExtractionMode.FULL)
        symbols = {s.name: s for s in rep.units if s.kind == UnitKind.SYMBOL}
        cls = symbols["Greeter"]
        method = symbols["greet"]
        file_unit = next(u for u in rep.units if u.kind == UnitKind.FILE)

        # A class is parented to its file; a method is parented to its class.
        assert cls.parent_id == file_unit.unit_id
        assert method.parent_id == cls.unit_id
        assert method.metadata["symbol_type"] == "Function"
        assert method.metadata.get("is_method") is True

    def test_top_level_function_parented_to_file(self, py_file):
        meta = file_metadata_from_path(py_file)
        rep = CodePrimitive().decompose(meta, ExtractionMode.FULL)
        helper = next(
            u for u in rep.units if u.kind == UnitKind.SYMBOL and u.name == "helper"
        )
        file_unit = next(u for u in rep.units if u.kind == UnitKind.FILE)
        assert helper.parent_id == file_unit.unit_id

    def test_calls_metadata_present(self, py_file):
        meta = file_metadata_from_path(py_file)
        rep = CodePrimitive().decompose(meta, ExtractionMode.FULL)
        greet = next(
            u for u in rep.units if u.kind == UnitKind.SYMBOL and u.name == "greet"
        )
        # greet() calls helper() — the raw (unresolved) call is carried.
        assert "calls" in greet.metadata
        assert any("helper" in c for c in greet.metadata["calls"])

    def test_sample_mode_bounds_calls(self, py_file):
        meta = file_metadata_from_path(py_file)
        rep = CodePrimitive().decompose(meta, ExtractionMode.SAMPLE)
        assert rep.mode == ExtractionMode.SAMPLE
        for u in rep.units:
            if u.kind == UnitKind.SYMBOL and "calls" in u.metadata:
                assert len(u.metadata["calls"]) <= 10

    def test_accepts_list_of_files(self, py_file):
        meta = file_metadata_from_path(py_file)
        rep = CodePrimitive().decompose([meta], ExtractionMode.FULL)
        assert len([u for u in rep.units if u.kind == UnitKind.FILE]) == 1
