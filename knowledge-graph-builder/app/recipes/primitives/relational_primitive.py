"""Relational primitive — wraps the `SchemaSnapshot` schema model.

TASK-222 / TASK-235 / STORY-034 / ADR-022.

Deterministically decomposes a relational database into a
`StructuralRepresentation`:

  * one `TABLE` unit per table;
  * one `COLUMN` unit per column, with `parent_id` at its table, the column's
    `data_type`, and a `role` of `primary_key` / `foreign_key` where the
    snapshot marks the column as such. Foreign-key columns also carry
    `metadata={"fk_target": "<table unit_id>"}` so the recipe engine can
    resolve the edge (recipe-spec §5.2);
  * one `RECORD` unit per table *row*, with `parent_id` at its `:Table` unit
    and the row's field values carried in ``sample_values[0]`` — keyed by
    column name, exactly as the CSV / JSON primitives carry a row. The recipe
    engine projects `node` / `property` / `edge` rules over these record units
    (unified-graph-model §2 — a database row IS the entity).

The adapter takes a `SchemaSnapshot` it is *given*, and the rows it is
*given* — it never opens a database connection. Introspection and row
fetching (which do open a connection) are the connector's job
(`app.services.database_connector_service`); this primitive is a pure
structural translation of a snapshot plus its rows.

Row data is supplied to `decompose` via the optional ``rows`` argument — a
``{table_name: [row_dict, ...]}`` mapping. When ``rows`` is omitted the
primitive emits structure only (the TASK-222 behaviour). The caller decides
how many rows to fetch:

  * `SAMPLE` mode — the caller passes a bounded sample of rows per table
    (the connector's ``extract_sample_data`` / ``fetch_rows`` with a small
    limit);
  * `FULL` mode — the caller passes all rows per table (the connector's
    ``fetch_rows`` paged to completion).

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

# Bounded number of record units emitted per table in SAMPLE mode when the
# caller has not already trimmed the supplied rows.
_SAMPLE_LIMIT = 5


def _table_unit_id(table_name: str) -> str:
    return f"table:{table_name}"


def _column_unit_id(table_name: str, column_name: str) -> str:
    return f"column:{table_name}.{column_name}"


def _record_unit_id(table_name: str, index: int) -> str:
    return f"record:{table_name}.{index}"


class RelationalPrimitive:
    """Adapter turning a `SchemaSnapshot` (+ rows) into a `StructuralRepresentation`."""

    source_type: str = "relational"

    def decompose(
        self,
        source: Any,
        mode: ExtractionMode,
        rows: dict[str, list[dict[str, Any]]] | None = None,
    ) -> StructuralRepresentation:
        """Decompose a `SchemaSnapshot` into structural units.

        Args:
            source: A `SchemaSnapshot` (duck-typed: `.tables`, each with
                `.name` and `.columns`; each column with `.name`,
                `.data_type`, `.is_pk`, `.is_fk`, `.fk_table`).
            mode: SAMPLE bounds the number of `RECORD` units emitted per table
                when the caller has not trimmed ``rows`` itself; FULL emits a
                `RECORD` unit for every supplied row. Both emit the same
                table/column structure.
            rows: Optional ``{table_name: [row_dict, ...]}`` — the rows to emit
                as `RECORD` units. A row dict is keyed by column name. Tables
                absent from this mapping (or the whole mapping being ``None``)
                emit structure only — no record units.
        """
        snapshot = source
        rows = rows or {}
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

            # RECORD units — one per row, parented at the table unit. A record
            # carries the row dict in `sample_values[0]`, consistent with the
            # CSV / JSON primitives so the engine reads it uniformly.
            #
            # The record unit's `name` is the *table name* — not a per-row
            # label — so a recipe rule can select exactly one table's rows with
            # `match: {unit_kind: "record", name: "<table>"}` (the per-row index
            # lives in `unit_id`). Without this a `record`-matching node rule
            # would match every table's rows under first-match-wins.
            table_rows = rows.get(table.name, [])
            if mode != ExtractionMode.FULL:
                table_rows = table_rows[:_SAMPLE_LIMIT]
            for idx, row in enumerate(table_rows):
                units.append(
                    StructuralUnit(
                        kind=UnitKind.RECORD,
                        unit_id=_record_unit_id(table.name, idx),
                        name=table.name,
                        parent_id=table_id,
                        sample_values=[dict(row)],
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
        of table/column ordering (recipe-spec §4 lookup key). Row values never
        affect the signature — the shape is the schema, not its data.
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
