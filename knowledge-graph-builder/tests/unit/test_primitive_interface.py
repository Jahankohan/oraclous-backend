"""Hermetic unit tests for the primitive interface (TASK-222, STORY-034).

No runtime dependency (pure Pydantic model validation) — runs on the host per
the by-dependency testing policy.
"""

import pytest

from app.recipes.primitives import (
    ExtractionMode,
    Primitive,
    StructuralRepresentation,
    StructuralUnit,
    UnitKind,
)


def _sample_representation() -> StructuralRepresentation:
    return StructuralRepresentation(
        source_type="relational",
        shape_signature="employees(id,name,manager_id)",
        mode=ExtractionMode.SAMPLE,
        units=[
            StructuralUnit(
                kind=UnitKind.TABLE, unit_id="t:employees", name="employees"
            ),
            StructuralUnit(
                kind=UnitKind.COLUMN,
                unit_id="c:employees.manager_id",
                name="manager_id",
                role="foreign_key",
                parent_id="t:employees",
                metadata={"fk_target": "t:employees"},
            ),
        ],
    )


def test_structural_representation_round_trips_through_json():
    rep = _sample_representation()
    assert rep.mode is ExtractionMode.SAMPLE
    assert {u.kind for u in rep.units} == {UnitKind.TABLE, UnitKind.COLUMN}
    assert StructuralRepresentation.model_validate_json(rep.model_dump_json()) == rep


def test_unit_kind_rejects_an_unknown_kind():
    with pytest.raises(ValueError):
        StructuralUnit(kind="teleport", unit_id="x")


def test_structural_unit_requires_a_unit_id():
    with pytest.raises(ValueError):
        StructuralUnit(kind=UnitKind.TABLE)


def test_primitive_protocol_is_runtime_checkable():
    class FakeCsvPrimitive:
        source_type = "csv"

        def decompose(self, source, mode):
            return StructuralRepresentation(
                source_type="csv", shape_signature="", mode=mode, units=[]
            )

    assert isinstance(FakeCsvPrimitive(), Primitive)
    assert not isinstance(object(), Primitive)
