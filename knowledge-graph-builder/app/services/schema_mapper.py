"""Schema Mapper Service — analyse SchemaSnapshot objects and emit GraphMappingRules.

Consumes the SchemaSnapshot produced by database_connector_service and produces a
declarative GraphMappingRules spec that tells RowTransformer (TASK-020) exactly how
to convert rows into graph nodes and edges.

Architecture invariants:
- graph_id is threaded through every output structure — no cross-tenant objects
- Pure analysis only — NO Neo4j writes (that is TASK-020's responsibility)
- One service per functionality — this file is self-contained
"""

from __future__ import annotations

import re as _re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from app.core.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Audit column names that do NOT count as "real" non-FK columns when
# deciding if a table qualifies as a junction table.
# ---------------------------------------------------------------------------
_AUDIT_COLUMN_NAMES: frozenset[str] = frozenset(
    {
        "created_at",
        "updated_at",
        "deleted_at",
        "created_by",
        "updated_by",
        "modified_at",
        "modified_by",
        "id",
    }
)

# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ColumnMapping:
    """Maps a source column to its Neo4j property name."""

    column_name: str
    neo4j_property: str  # camelCase of column_name


@dataclass
class RelationshipMapping:
    """Describes one relationship edge to be created in Neo4j."""

    from_table: str
    to_table: str
    rel_type: str  # e.g. WORKS_ON
    from_fk_column: str
    to_fk_column: str  # PK column on the target table
    via_junction: str | None  # junction table name if applicable


@dataclass
class TableMapping:
    """Full mapping specification for one source table."""

    table_name: str
    kind: Literal["entity_table", "junction_table", "self_ref_table"]
    neo4j_label: str  # PascalCase of table_name
    pk_column: str
    property_columns: list[ColumnMapping]
    relationships: list[RelationshipMapping]


@dataclass
class GraphMappingRules:
    """Top-level container — the output of SchemaMapper.map()."""

    connector_id: str
    graph_id: str  # tenant isolation — present on every output structure
    tables: list[TableMapping]
    generated_at: str  # ISO timestamp


# ---------------------------------------------------------------------------
# Naming utilities
# ---------------------------------------------------------------------------


def _to_camel_case(name: str) -> str:
    """Convert snake_case (or any underscore-separated) name to camelCase.

    Examples:
        user_name       → userName
        department_id   → departmentId
        created_at      → createdAt
        id              → id
    """
    parts = name.replace("-", "_").split("_")
    if not parts:
        return name
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


def _to_pascal_case(name: str) -> str:
    """Convert snake_case table name to PascalCase Neo4j label.

    Examples:
        employee        → Employee
        employee_project → EmployeeProject
        t_emp_proj      → TEmpProj
    """
    return "".join(word.capitalize() for word in name.replace("-", "_").split("_"))


def _validate_rel_type(rel_type: str) -> str:
    """Ensure rel_type is a valid Neo4j relationship type identifier.

    Raises ValueError if the derived type would produce unsafe Cypher.
    All rel types used in Cypher are derived from schema introspection (never
    direct user input) and validated here before any interpolation.
    """
    if not _re.match(r"^[A-Z_][A-Z0-9_]*$", rel_type):
        raise ValueError(
            f"Derived relationship type is not a valid Cypher identifier: {rel_type!r}"
        )
    return rel_type


# ---------------------------------------------------------------------------
# Relationship type derivation rules
# ---------------------------------------------------------------------------

# Self-referential FK patterns: column name → (outgoing rel, incoming rel)
# We use the outgoing direction (parent → child or manager → report).
_SELF_REF_PATTERNS: dict[str, str] = {
    "parent_id": "HAS_CHILD",
    "manager_id": "MANAGES",
    "supervisor_id": "SUPERVISES",
    "reports_to": "MANAGES",
    "parent": "HAS_CHILD",
}


def _rel_type_for_fk(
    fk_column: str,
    fk_table: str,
    source_table: str,
) -> str:
    """Derive a relationship type string from an FK column + its target table.

    Strategy (in priority order):
    1. Self-referential FK → use _SELF_REF_PATTERNS or fall back to REFERENCES_SELF
    2. Suffix stripping: column ends with _id or _ref → strip it, use the bare word
       e.g. department_id → BELONGS_TO_DEPARTMENT (but we simplify to BELONGS_TO)
    3. Default: BELONGS_TO (generic FK relationship)
    """
    col_lower = fk_column.lower()
    tbl_lower = fk_table.lower()

    # Self-referential
    if fk_table == source_table:
        # Exact match against known self-ref column name patterns
        if col_lower in _SELF_REF_PATTERNS:
            return _SELF_REF_PATTERNS[col_lower]
        return "REFERENCES_SELF"

    # Column name encodes semantic (e.g. manager_id on employee → MANAGED_BY)
    # Strip _id / _fk / _ref suffix to get the semantic word
    bare = _re.sub(r"_(id|fk|ref|key)$", "", col_lower)
    target_bare = _re.sub(r"s$", "", tbl_lower)  # naïve singularize

    if bare == target_bare or bare == tbl_lower:
        # Column name matches table name → generic BELONGS_TO
        return "BELONGS_TO"

    # If bare is something meaningful and different from the table name,
    # build: BELONGS_TO_<TABLE_UPPER>
    target_upper = _to_pascal_case(fk_table).upper()
    rel = "BELONGS_TO"
    return rel


def _junction_rel_type_from_name(
    table_name: str, from_table: str, to_table: str
) -> str | None:
    """Try to extract a verb from a junction table name.

    Returns an UPPER_SNAKE_CASE string if the name contains a recognisable verb,
    otherwise returns None (caller should invoke LLM).

    Heuristic: strip the two referenced table names (and common prefixes like
    t_, rel_, jn_, jt_) from the junction table name; if anything remains and
    it looks like a real word (≥3 chars, all alpha), use it as the verb.
    """
    name_clean = table_name.lower()

    # Strip common junction table prefixes
    for prefix in ("t_", "rel_", "jn_", "jt_", "map_", "link_", "xref_", "assoc_"):
        if name_clean.startswith(prefix):
            name_clean = name_clean[len(prefix) :]

    # Strip the two referenced table names from the junction name
    for ref in (from_table.lower(), to_table.lower()):
        name_clean = name_clean.replace(ref, "").strip("_")

    # Whatever is left — if it looks like a word, turn it into a rel type
    word = name_clean.strip("_")
    if word and len(word) >= 3 and word.isalpha():
        return word.upper()

    return None  # ambiguous — caller must use LLM


def _call_llm_for_rel_type(table_name: str, from_table: str, to_table: str) -> str:
    """Synchronous LLM call to name an ambiguous junction table relationship.

    Uses the OpenAI sync client.  Called at most once per ambiguous junction table.
    Falls back to ASSOCIATED_WITH if the API key is absent or the call fails.
    """
    try:
        from openai import OpenAI  # type: ignore

        from app.core.config import settings

        api_key = getattr(settings, "OPENAI_API_KEY", None)
        if not api_key:
            logger.warning(
                "schema_mapper: OPENAI_API_KEY not set — using fallback rel type for %r",
                table_name,
            )
            return "ASSOCIATED_WITH"

        prompt = (
            f"Table `{table_name}` is a junction table that joins `{from_table}` "
            f"and `{to_table}`. "
            f"Suggest a concise uppercase verb relationship type (e.g. WORKS_ON, ASSIGNED_TO, ENROLLED_IN). "
            f"Reply with ONLY the relationship type string, nothing else."
        )

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a graph schema expert. "
                        "Output only a valid Neo4j relationship type: ALL_CAPS_SNAKE_CASE."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=20,
        )
        raw = (response.choices[0].message.content or "").strip().upper()
        # Sanitise: keep only A-Z and underscores
        sanitised = _re.sub(r"[^A-Z_]", "_", raw).strip("_")
        if sanitised and _re.match(r"^[A-Z][A-Z0-9_]*$", sanitised):
            logger.info(
                "schema_mapper: LLM named junction %r → %s", table_name, sanitised
            )
            return sanitised
        logger.warning(
            "schema_mapper: LLM returned invalid rel type %r for %r — using fallback",
            raw,
            table_name,
        )
        return "ASSOCIATED_WITH"

    except Exception as exc:
        logger.warning(
            "schema_mapper: LLM call failed for junction %r: %s — using fallback",
            table_name,
            exc,
        )
        return "ASSOCIATED_WITH"


# ---------------------------------------------------------------------------
# FK grouping helpers
# ---------------------------------------------------------------------------


def _group_fk_columns(columns: list) -> list[list]:
    """Group FK columns that belong to composite FKs.

    In the current SchemaSnapshot model there is no explicit constraint-name
    field, so we group by (fk_table, fk_column) pairs that share the same
    target table.  Two FK columns pointing to the same target table are treated
    as a composite FK and emitted as a single RelationshipMapping.

    Returns a list of groups, where each group is a list of ColumnMeta objects.
    """
    # Group FK cols by target table
    groups: dict[str, list] = {}
    for col in columns:
        if not col.is_fk or col.fk_table is None:
            continue
        key = col.fk_table
        groups.setdefault(key, []).append(col)
    return list(groups.values())


# ---------------------------------------------------------------------------
# Table classification helpers
# ---------------------------------------------------------------------------

_FK_SUFFIX_PATTERN = _re.compile(r"_(id|fk|ref|key)$", _re.IGNORECASE)


def _find_pk(columns: list) -> str | None:
    """Return the name of the first PK column, or None."""
    for col in columns:
        if col.is_pk:
            return col.name
    return None


def _is_audit_column(col_name: str) -> bool:
    return col_name.lower() in _AUDIT_COLUMN_NAMES


def _classify_table_kind(
    table_name: str,
    columns: list,
) -> Literal["entity_table", "junction_table", "self_ref_table"]:
    """Classify a table as entity, junction, or self-referential.

    Rules:
    - junction_table: exactly 2 FK columns (ignoring audit columns and any
      single PK column) and no other non-FK, non-audit columns
    - self_ref_table: entity table that has at least one FK pointing back to
      itself (hierarchical / adjacency list pattern)
    - entity_table: everything else with a PK
    """
    fk_cols = [c for c in columns if c.is_fk and not c.is_pk]
    fk_tables = {c.fk_table for c in fk_cols if c.fk_table}

    # Non-FK, non-PK, non-audit columns
    non_fk_real_cols = [
        c
        for c in columns
        if not c.is_pk and not c.is_fk and not _is_audit_column(c.name)
    ]

    # Junction: exactly 2 distinct FK target tables + no meaningful payload cols
    if len(fk_tables) == 2 and len(non_fk_real_cols) == 0:
        return "junction_table"

    # Self-ref: any FK points back to the same table
    if any(c.fk_table == table_name for c in fk_cols):
        return "self_ref_table"

    return "entity_table"


# ---------------------------------------------------------------------------
# SchemaMapper
# ---------------------------------------------------------------------------


class SchemaMapper:
    """Analyse a SchemaSnapshot and produce GraphMappingRules.

    This is a pure analysis class — it makes NO writes to Neo4j.
    graph_id is injected into every output structure for multi-tenancy.
    """

    def map(
        self,
        schema_snapshot,  # SchemaSnapshot from database_connector_service
        connector_id: str,
        graph_id: str,
    ) -> GraphMappingRules:
        """Analyse snapshot and return a fully-populated GraphMappingRules.

        Args:
            schema_snapshot: SchemaSnapshot produced by a DatabaseConnector.
            connector_id:    ID of the source connector (for provenance).
            graph_id:        Tenant graph ID — threaded through all output structures.

        Returns:
            GraphMappingRules with graph_id set on the top-level object.
        """
        table_mappings: list[TableMapping] = []

        # Build a lookup: table_name → PK column name (needed for rel mappings)
        pk_lookup: dict[str, str] = {}
        for table_meta in schema_snapshot.tables:
            pk = _find_pk(table_meta.columns)
            if pk:
                pk_lookup[table_meta.name] = pk

        for table_meta in schema_snapshot.tables:
            mapping = self._map_table(table_meta, pk_lookup, graph_id)
            if mapping is not None:
                table_mappings.append(mapping)

        return GraphMappingRules(
            connector_id=connector_id,
            graph_id=graph_id,
            tables=table_mappings,
            generated_at=datetime.now(UTC).isoformat(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _map_table(
        self,
        table_meta,
        pk_lookup: dict[str, str],
        graph_id: str,  # kept for future per-table graph_id injection if needed
    ) -> TableMapping | None:
        """Map one TableMeta to a TableMapping, or return None if unmappable."""
        columns = table_meta.columns
        pk_col = _find_pk(columns)

        # Classify early so we can decide whether a missing PK is acceptable.
        # Junction tables do not need a PK to produce a valid RelationshipMapping.
        kind = _classify_table_kind(table_meta.name, columns)

        # Non-junction tables without a PK are skipped — we cannot create unique entity nodes.
        if pk_col is None and kind != "junction_table":
            logger.warning(
                "schema_mapper: table %r has no PK — skipping",
                table_meta.name,
            )
            return None
        neo4j_label = _to_pascal_case(table_meta.name)

        # Property columns: non-PK, non-FK columns
        property_columns = [
            ColumnMapping(
                column_name=col.name,
                neo4j_property=_to_camel_case(col.name),
            )
            for col in columns
            if not col.is_pk and not col.is_fk
        ]

        # FK → RelationshipMapping
        relationships = self._build_relationships(
            table_meta.name, columns, pk_lookup, kind
        )

        return TableMapping(
            table_name=table_meta.name,
            kind=kind,
            neo4j_label=neo4j_label,
            pk_column=pk_col or "",  # junction tables may have no PK
            property_columns=property_columns,
            relationships=relationships,
        )

    def _build_relationships(
        self,
        table_name: str,
        columns: list,
        pk_lookup: dict[str, str],
        kind: Literal["entity_table", "junction_table", "self_ref_table"],
    ) -> list[RelationshipMapping]:
        """Derive RelationshipMapping entries from FK columns.

        - For entity_table and self_ref_table: one RelationshipMapping per FK
          group (composite FKs sharing the same target table → one mapping).
        - For junction_table: one RelationshipMapping from table A → table B,
          via this junction table.
        """
        fk_cols = [c for c in columns if c.is_fk and c.fk_table is not None]
        if not fk_cols:
            return []

        # Group FK columns by target table (handles composite FKs)
        fk_groups = _group_fk_columns(columns)

        if kind == "junction_table":
            return self._build_junction_relationships(table_name, fk_groups, pk_lookup)

        # entity_table or self_ref_table — one rel per FK group
        rels: list[RelationshipMapping] = []
        for group in fk_groups:
            # Use the first column in the group as the representative FK column
            rep = group[0]
            target_table = rep.fk_table
            if not target_table:
                continue
            target_pk = pk_lookup.get(target_table, "id")
            rel_type = _rel_type_for_fk(rep.name, target_table, table_name)
            try:
                rel_type = _validate_rel_type(rel_type)
            except ValueError:
                logger.warning(
                    "schema_mapper: invalid rel_type %r for %r.%r → using REFERENCES",
                    rel_type,
                    table_name,
                    rep.name,
                )
                rel_type = "REFERENCES"

            rels.append(
                RelationshipMapping(
                    from_table=table_name,
                    to_table=target_table,
                    rel_type=rel_type,
                    from_fk_column=rep.name,
                    to_fk_column=target_pk,
                    via_junction=None,
                )
            )
        return rels

    def _build_junction_relationships(
        self,
        junction_table: str,
        fk_groups: list[list],
        pk_lookup: dict[str, str],
    ) -> list[RelationshipMapping]:
        """Build a single RelationshipMapping for a junction table (A → B via junction)."""
        if len(fk_groups) < 2:
            logger.warning(
                "schema_mapper: junction table %r has fewer than 2 FK groups — skipping",
                junction_table,
            )
            return []

        # Take the first two FK groups as the two sides of the relationship
        group_a = fk_groups[0]
        group_b = fk_groups[1]

        from_table = group_a[0].fk_table
        to_table = group_b[0].fk_table
        from_fk_col = group_a[0].name
        to_fk_col = group_b[0].name

        if not from_table or not to_table:
            return []

        # Derive relationship type from junction table name
        rel_type = _junction_rel_type_from_name(junction_table, from_table, to_table)
        if rel_type is None:
            # Ambiguous name — ask the LLM (max 1 call per table)
            logger.info(
                "schema_mapper: junction %r name is ambiguous — calling LLM",
                junction_table,
            )
            rel_type = _call_llm_for_rel_type(junction_table, from_table, to_table)

        try:
            rel_type = _validate_rel_type(rel_type)
        except ValueError:
            logger.warning(
                "schema_mapper: invalid rel_type %r for junction %r — using ASSOCIATED_WITH",
                rel_type,
                junction_table,
            )
            rel_type = "ASSOCIATED_WITH"

        return [
            RelationshipMapping(
                from_table=from_table,
                to_table=to_table,
                rel_type=rel_type,
                from_fk_column=from_fk_col,
                to_fk_column=to_fk_col,
                via_junction=junction_table,
            )
        ]
