"""Hermetic unit tests for the relational primitive (TASK-222 / STORY-034).

Pure translation tests — a fabricated `SchemaSnapshot` (no DB connection) is
decomposed and the resulting `StructuralRepresentation` is asserted.
"""

from datetime import UTC, datetime

from app.recipes.primitives import (
    ExtractionMode,
    Primitive,
    RelationalPrimitive,
    StructuralRepresentation,
    UnitKind,
)
from app.services.database_connector_service import (
    ColumnMeta,
    DatabaseConnectorType,
    SchemaSnapshot,
    TableMeta,
)


def _snapshot() -> SchemaSnapshot:
    """A two-table snapshot: department (entity) and employee (FK → department)."""
    department = TableMeta(
        name="department",
        schema_name="public",
        columns=[
            ColumnMeta(
                name="id", data_type="integer", nullable=False, is_pk=True, is_fk=False
            ),
            ColumnMeta(
                name="name", data_type="text", nullable=False, is_pk=False, is_fk=False
            ),
        ],
    )
    employee = TableMeta(
        name="employee",
        schema_name="public",
        columns=[
            ColumnMeta(
                name="id", data_type="integer", nullable=False, is_pk=True, is_fk=False
            ),
            ColumnMeta(
                name="full_name",
                data_type="text",
                nullable=False,
                is_pk=False,
                is_fk=False,
            ),
            ColumnMeta(
                name="department_id",
                data_type="integer",
                nullable=True,
                is_pk=False,
                is_fk=True,
                fk_table="department",
                fk_column="id",
            ),
        ],
    )
    return SchemaSnapshot(
        connector_type=DatabaseConnectorType.POSTGRESQL,
        database="company",
        tables=[department, employee],
        captured_at=datetime.now(UTC),
    )


class TestRelationalPrimitive:
    def test_conforms_to_primitive_protocol(self):
        assert isinstance(RelationalPrimitive(), Primitive)
        assert RelationalPrimitive().source_type == "relational"

    def test_tables_and_columns(self):
        rep = RelationalPrimitive().decompose(_snapshot(), ExtractionMode.SAMPLE)
        assert isinstance(rep, StructuralRepresentation)
        assert rep.source_type == "relational"

        tables = [u for u in rep.units if u.kind == UnitKind.TABLE]
        columns = [u for u in rep.units if u.kind == UnitKind.COLUMN]
        assert {t.name for t in tables} == {"department", "employee"}
        # 2 department columns + 3 employee columns.
        assert len(columns) == 5

    def test_columns_parented_to_their_table(self):
        rep = RelationalPrimitive().decompose(_snapshot(), ExtractionMode.FULL)
        table_ids = {u.name: u.unit_id for u in rep.units if u.kind == UnitKind.TABLE}
        emp_cols = [
            u
            for u in rep.units
            if u.kind == UnitKind.COLUMN and u.unit_id.startswith("column:employee.")
        ]
        for col in emp_cols:
            assert col.parent_id == table_ids["employee"]

    def test_primary_key_role(self):
        rep = RelationalPrimitive().decompose(_snapshot(), ExtractionMode.SAMPLE)
        pk = next(
            u
            for u in rep.units
            if u.kind == UnitKind.COLUMN and u.unit_id == "column:department.id"
        )
        assert pk.role == "primary_key"
        assert pk.data_type == "integer"

    def test_foreign_key_role_and_fk_target(self):
        rep = RelationalPrimitive().decompose(_snapshot(), ExtractionMode.SAMPLE)
        fk = next(
            u
            for u in rep.units
            if u.kind == UnitKind.COLUMN
            and u.unit_id == "column:employee.department_id"
        )
        assert fk.role == "foreign_key"
        # fk_target carries the *table unit_id* of the referenced table.
        assert fk.metadata["fk_target"] == "table:department"
        assert fk.metadata["fk_target_present"] is True
        assert fk.metadata["fk_target_column"] == "id"

    def test_plain_column_has_no_role(self):
        rep = RelationalPrimitive().decompose(_snapshot(), ExtractionMode.SAMPLE)
        plain = next(
            u
            for u in rep.units
            if u.kind == UnitKind.COLUMN and u.unit_id == "column:employee.full_name"
        )
        assert plain.role is None

    def test_shape_signature_independent_of_table_order(self):
        snap = _snapshot()
        snap_reordered = SchemaSnapshot(
            connector_type=snap.connector_type,
            database=snap.database,
            tables=list(reversed(snap.tables)),
            captured_at=snap.captured_at,
        )
        a = RelationalPrimitive().decompose(snap, ExtractionMode.SAMPLE)
        b = RelationalPrimitive().decompose(snap_reordered, ExtractionMode.FULL)
        assert a.shape_signature == b.shape_signature
