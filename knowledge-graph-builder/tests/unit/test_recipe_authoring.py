"""Hermetic unit tests for the recipe-authoring loop (TASK-225 / STORY-034).

Covers step 1 of the authoring loop — `sample_source` — and the primitive
registry it dispatches on. Pure, in-memory: a small CSV string and a small
Python snippet are written to temp files; the resulting SAMPLE-mode
`StructuralRepresentation` is asserted. No database, no Docker.

Step 2 (the agent authoring a recipe) is not code and is not tested here.
Step 3 (`save_recipe_draft`) is Postgres-backed — covered by the integration
test in `tests/integration/test_recipe_authoring.py`.
"""

import tempfile
from pathlib import Path

import pytest

from app.recipes.authoring import (
    PRIMITIVE_REGISTRY,
    UnknownPrimitiveKindError,
    sample_source,
)
from app.recipes.primitives import (
    ExtractionMode,
    Primitive,
    StructuralRepresentation,
    UnitKind,
)
from app.services.code_parser_service import _get_parser

# ---------------------------------------------------------------------------
# Fixtures — tiny in-memory sources
# ---------------------------------------------------------------------------

_CSV = "id,name,team\n1,Alice,Platform\n2,Bob,Platform\n3,Carol,Research\n"

_PY_SNIPPET = '''\
"""A tiny module for authoring-loop tests."""


def greet(name):
    return "hi " + name
'''


def _has_python_parser() -> bool:
    return _get_parser("python") is not None


@pytest.fixture
def csv_path():
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    )
    tmp.write(_CSV)
    tmp.close()
    yield tmp.name
    Path(tmp.name).unlink(missing_ok=True)


@pytest.fixture
def py_path():
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    )
    tmp.write(_PY_SNIPPET)
    tmp.close()
    yield tmp.name
    Path(tmp.name).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The primitive registry
# ---------------------------------------------------------------------------


class TestPrimitiveRegistry:
    def test_registry_has_the_six_explicit_kinds(self):
        """The registry keys are explicit `kind` strings — all six of them."""
        assert set(PRIMITIVE_REGISTRY) == {
            "csv",
            "json",
            "relational",
            "code",
            "markdown",
            "document",
        }

    def test_every_registered_value_is_a_primitive(self):
        """Each registry value conforms to the `Primitive` protocol."""
        for kind, primitive in PRIMITIVE_REGISTRY.items():
            assert isinstance(primitive, Primitive), kind

    def test_markdown_and_document_share_a_source_type_but_distinct_kinds(self):
        """The registry key disambiguates the two `source_type='text'` primitives."""
        assert PRIMITIVE_REGISTRY["markdown"].source_type == "text"
        assert PRIMITIVE_REGISTRY["document"].source_type == "text"
        # Distinct kinds, distinct primitive classes — that is why the key is
        # an explicit `kind`, not `source_type`.
        assert type(PRIMITIVE_REGISTRY["markdown"]) is not type(
            PRIMITIVE_REGISTRY["document"]
        )


# ---------------------------------------------------------------------------
# sample_source — registry dispatch + SAMPLE-mode representation
# ---------------------------------------------------------------------------


class TestSampleSource:
    def test_csv_dispatch_returns_sample_mode_representation(self, csv_path):
        """`sample_source(..., 'csv')` runs the CSV primitive in SAMPLE mode."""
        rep = sample_source(csv_path, "csv")

        assert isinstance(rep, StructuralRepresentation)
        assert rep.source_type == "csv"
        assert rep.mode == ExtractionMode.SAMPLE

        # SOURCE unit + one COLUMN per column.
        column_names = {u.name for u in rep.units if u.kind == UnitKind.COLUMN}
        assert column_names == {"id", "name", "team"}

        # SAMPLE mode emits bounded example values, not the whole source.
        name_col = next(
            u for u in rep.units if u.kind == UnitKind.COLUMN and u.name == "name"
        )
        assert name_col.sample_values  # bounded examples present
        assert len(name_col.sample_values) <= 5

    def test_code_dispatch_returns_sample_mode_representation(self, py_path):
        """`sample_source(..., 'code')` runs the code primitive in SAMPLE mode."""
        if not _has_python_parser():
            pytest.skip("tree-sitter python grammar not installed")

        from app.recipes.primitives.code_primitive import file_metadata_from_path

        meta = file_metadata_from_path(py_path)
        rep = sample_source(meta, "code")

        assert isinstance(rep, StructuralRepresentation)
        assert rep.source_type == "code"
        assert rep.mode == ExtractionMode.SAMPLE

        # One FILE unit, plus a SYMBOL unit for the `greet` function.
        assert any(u.kind == UnitKind.FILE for u in rep.units)
        symbol_names = {u.name for u in rep.units if u.kind == UnitKind.SYMBOL}
        assert "greet" in symbol_names

    def test_unknown_kind_raises_clear_error(self, csv_path):
        """An unregistered `kind` raises a clear error naming the supported kinds."""
        with pytest.raises(UnknownPrimitiveKindError) as excinfo:
            sample_source(csv_path, "spreadsheet")

        message = str(excinfo.value)
        assert "spreadsheet" in message
        # The error lists what *is* supported.
        assert "csv" in message and "code" in message

    def test_dispatch_picks_the_kind_not_the_source_type(self):
        """`markdown` and `document` are reachable by their distinct kinds.

        Both declare `source_type='text'`; dispatch is by the explicit kind.
        """
        markdown_rep = sample_source("# Title\n\nBody text.", "markdown")
        assert markdown_rep.source_type == "text"
        assert markdown_rep.mode == ExtractionMode.SAMPLE
        # A markdown document with one heading section.
        assert any(u.kind == UnitKind.DOCUMENT for u in markdown_rep.units)
        assert any(u.kind == UnitKind.CHUNK for u in markdown_rep.units)
