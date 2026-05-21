"""Relational primitive — wraps the `SchemaSnapshot` schema model.

TASK-222 / STORY-034 / ADR-022.

Deterministically decomposes a relational database schema — a `SchemaSnapshot`
of `TableMeta` / `ColumnMeta` from `app.services.database_connector_service` —
into a `StructuralRepresentation`:

  * one `TABLE` unit per table;
  * one `COLUMN` unit per column, with `parent_id` at its table, the column's
    `data_type`, and a `role` of `primary_key` / `foreign_key` where the
    snapshot marks the column as such. Foreign-key columns also carry
    `metadata={"fk_target": "<table unit_id>"}` so the recipe engine can
    resolve the edge (recipe-spec §5.2).

The adapter takes a `SchemaSnapshot` it is *given* — it never opens a database
connection. Introspection (which does open a connection) is the connector's
job; this primitive is a pure structural translation of a snapshot.

It reuses the safe-identifier and classification reasoning in
`app.services.schema_mapper` only conceptually; no DB-touching code path is
invoked here.
"""

from __future__ import annotations

from typing import Any

from app.recipes.primitives.interface import (
    ExtractionMode,
    StructuralRepresentation,
    StructuralUnit,
    UnitKind,
)


def _table_unit_id(table_name: str) -> str:
    return f"table:{table_name}"


def _column_unit_id(table_name: str, column_name: str) -> str:
    return f"column:{table_name}.{column_name}"


class RelationalPrimitive:
    """Adapter turning a `SchemaSnapshot` into a `StructuralRepresentation`."""

    source_type: str = "relational"

    def decompose(self, source: Any, mode: ExtractionMode) -> StructuralRepresentation:
        """Decompose a `SchemaSnapshot` into structural units.

        Args:
            source: A `SchemaSnapshot` (duck-typed: `.tables`, each with
                `.name` and `.columns`; each column with `.name`,
                `.data_type`, `.is_pk`, `.is_fk`, `.fk_table`).
            mode: SAMPLE and FULL emit the same structure — a relational
                snapshot carries no row values, so there is nothing to bound.
        """
        snapshot = source
        units: list[StructuralUnit] = []

        # Names of tables actually present, used to decide whether an FK
        # target resolves to a unit_id we emit.
        known_tables = {t.name for t in snapshot.tables}

        for table in snapshot.tables:
            table_id = _table_unit_id(table.name)
            units.append(
                StructuralUnit(
                    kind=UnitKind.TABLE,
                    unit_id=table_id,
                    name=table.name,
                    metadata={
                        "schema_name": getattr(table, "schema_name", None),
                        "column_count": len(table.columns),
                    },
                )
            )

            for col in table.columns:
                role: str | None = None
                metadata: dict[str, Any] = {
                    "nullable": col.nullable,
                }
                if col.is_pk:
                    role = "primary_key"
                if col.is_fk:
                    # An FK that is also the PK keeps the foreign_key role —
                    # it is the edge-bearing column the recipe engine reads.
                    role = "foreign_key"
                    if col.fk_table:
                        metadata["fk_target"] = _table_unit_id(col.fk_table)
                        metadata["fk_target_present"] = col.fk_table in known_tables
                        if getattr(col, "fk_column", None):
                            metadata["fk_target_column"] = col.fk_column

                units.append(
                    StructuralUnit(
                        kind=UnitKind.COLUMN,
                        unit_id=_column_unit_id(table.name, col.name),
                        name=col.name,
                        data_type=col.data_type,
                        role=role,
                        parent_id=table_id,
                        metadata=metadata,
                    )
                )

        return StructuralRepresentation(
            source_type=self.source_type,
            shape_signature=self._shape_signature(snapshot),
            mode=mode,
            units=units,
        )

    @staticmethod
    def _shape_signature(snapshot: Any) -> str:
        """Deterministic descriptor — sorted table/column names + key flags.

        Stable across two snapshots with the same structural schema regardless
        of table/column ordering (recipe-spec §4 lookup key).
        """
        table_parts: list[str] = []
        for table in sorted(snapshot.tables, key=lambda t: t.name):
            col_parts: list[str] = []
            for col in sorted(table.columns, key=lambda c: c.name):
                flag = ""
                if col.is_pk:
                    flag += "P"
                if col.is_fk:
                    flag += "F"
                col_parts.append(f"{col.name}:{col.data_type}{flag}")
            table_parts.append(f"{table.name}[" + ",".join(col_parts) + "]")
        return "relational(" + ";".join(table_parts) + ")"
