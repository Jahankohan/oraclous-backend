"""Unit tests for app/services/schema_mapper.py.

All tests are pure Python — no Neo4j, no Docker, no network required.

Covers:
- entity_table classification
- junction_table classification (2 FK cols + audit cols)
- self_ref_table classification
- composite FK grouping → single RelationshipMapping
- LLM naming: mock LLM for opaque table name (`t_ep`) → verify call made
- graph_id present in all output structures
- camelCase / PascalCase name helpers
- Validation guard for unsafe rel type strings
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from app.services.database_connector_service import (
    ColumnMeta,
    DatabaseConnectorType,
    SchemaSnapshot,
    TableMeta,
)
from app.services.schema_mapper import (
    GraphMappingRules,
    SchemaMapper,
    _call_llm_for_rel_type,
    _classify_table_kind,
    _junction_rel_type_from_name,
    _rel_type_for_fk,
    _to_camel_case,
    _to_pascal_case,
    _validate_rel_type,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONNECTOR_ID = "conn-abc-123"
_GRAPH_ID = "graph-tenant-xyz"


def _make_snapshot(*tables: TableMeta) -> SchemaSnapshot:
    return SchemaSnapshot(
        connector_type=DatabaseConnectorType.POSTGRESQL,
        database="testdb",
        tables=list(tables),
        captured_at=datetime.now(UTC),
    )


def _col(
    name: str,
    *,
    is_pk: bool = False,
    is_fk: bool = False,
    fk_table: str | None = None,
    fk_column: str | None = None,
    data_type: str = "integer",
    nullable: bool = True,
) -> ColumnMeta:
    return ColumnMeta(
        name=name,
        data_type=data_type,
        nullable=nullable,
        is_pk=is_pk,
        is_fk=is_fk,
        fk_table=fk_table,
        fk_column=fk_column,
    )


def _table(name: str, *columns: ColumnMeta) -> TableMeta:
    return TableMeta(name=name, schema_name="public", columns=list(columns))


# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNameHelpers:
    def test_camel_case_single_word(self):
        assert _to_camel_case("id") == "id"

    def test_camel_case_two_words(self):
        assert _to_camel_case("user_name") == "userName"

    def test_camel_case_three_words(self):
        assert _to_camel_case("department_id") == "departmentId"

    def test_camel_case_created_at(self):
        assert _to_camel_case("created_at") == "createdAt"

    def test_pascal_case_simple(self):
        assert _to_pascal_case("employee") == "Employee"

    def test_pascal_case_multi_word(self):
        assert _to_pascal_case("employee_project") == "EmployeeProject"

    def test_pascal_case_prefix(self):
        assert _to_pascal_case("t_emp_proj") == "TEmpProj"


# ---------------------------------------------------------------------------
# Relationship type validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateRelType:
    def test_valid_type(self):
        assert _validate_rel_type("WORKS_ON") == "WORKS_ON"

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            _validate_rel_type("works on")

    def test_invalid_type_with_injection_chars(self):
        with pytest.raises(ValueError):
            _validate_rel_type("WORKS)-[:HACK")


# ---------------------------------------------------------------------------
# Table classification
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClassifyTableKind:
    def test_entity_table_basic(self):
        cols = [
            _col("id", is_pk=True),
            _col("name", data_type="varchar"),
            _col("email", data_type="varchar"),
        ]
        assert _classify_table_kind("user", cols) == "entity_table"

    def test_entity_table_with_fk_and_payload(self):
        """Table with FK but also non-FK payload columns → entity_table."""
        cols = [
            _col("id", is_pk=True),
            _col("department_id", is_fk=True, fk_table="department"),
            _col("name", data_type="varchar"),
        ]
        assert _classify_table_kind("employee", cols) == "entity_table"

    def test_junction_table_two_fks(self):
        """Exactly 2 FK columns + no payload → junction_table."""
        cols = [
            _col("employee_id", is_fk=True, fk_table="employee"),
            _col("project_id", is_fk=True, fk_table="project"),
        ]
        assert _classify_table_kind("employee_project", cols) == "junction_table"

    def test_junction_table_two_fks_plus_audit(self):
        """2 FK cols + audit cols only → still junction_table."""
        cols = [
            _col("employee_id", is_fk=True, fk_table="employee"),
            _col("project_id", is_fk=True, fk_table="project"),
            _col("created_at", data_type="timestamp"),
            _col("updated_at", data_type="timestamp"),
        ]
        assert _classify_table_kind("employee_project", cols) == "junction_table"

    def test_junction_table_with_surrogate_pk_and_audit(self):
        """Junction table may have an auto-increment 'id' PK — still junction."""
        cols = [
            _col("id", is_pk=True),
            _col("employee_id", is_fk=True, fk_table="employee"),
            _col("project_id", is_fk=True, fk_table="project"),
            _col("created_at", data_type="timestamp"),
        ]
        # 'id' is in _AUDIT_COLUMN_NAMES → ignored as non-FK payload
        assert _classify_table_kind("employee_project", cols) == "junction_table"

    def test_not_junction_with_payload(self):
        """2 FK cols + real payload col → entity_table, not junction."""
        cols = [
            _col("id", is_pk=True),
            _col("employee_id", is_fk=True, fk_table="employee"),
            _col("project_id", is_fk=True, fk_table="project"),
            _col("role", data_type="varchar"),  # real payload
        ]
        assert _classify_table_kind("employee_project", cols) == "entity_table"

    def test_self_ref_table(self):
        """FK pointing to same table's PK → self_ref_table."""
        cols = [
            _col("id", is_pk=True),
            _col("manager_id", is_fk=True, fk_table="employee"),
            _col("name", data_type="varchar"),
        ]
        assert _classify_table_kind("employee", cols) == "self_ref_table"

    def test_self_ref_with_parent_id(self):
        cols = [
            _col("id", is_pk=True),
            _col("parent_id", is_fk=True, fk_table="category"),
            _col("name", data_type="varchar"),
        ]
        assert _classify_table_kind("category", cols) == "self_ref_table"


# ---------------------------------------------------------------------------
# Junction table relationship type derivation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJunctionRelTypeName:
    def test_meaningful_verb_extracted(self):
        # employee_project → strip 'employee' and 'project' → '' → no verb
        # But 'employee_works_project' → 'works'
        result = _junction_rel_type_from_name("employee_works_project", "employee", "project")
        assert result == "WORKS"

    def test_no_verb_returns_none(self):
        # 'employee_project' → strip names → '' → None
        result = _junction_rel_type_from_name("employee_project", "employee", "project")
        assert result is None

    def test_opaque_name_returns_none(self):
        result = _junction_rel_type_from_name("t_ep", "employee", "project")
        assert result is None

    def test_opaque_with_rel_prefix_returns_none(self):
        result = _junction_rel_type_from_name("rel_012", "employee", "project")
        assert result is None


# ---------------------------------------------------------------------------
# FK → rel type derivation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRelTypeForFk:
    def test_department_id(self):
        result = _rel_type_for_fk("department_id", "department", "employee")
        assert result == "BELONGS_TO"

    def test_manager_id_self_ref(self):
        result = _rel_type_for_fk("manager_id", "employee", "employee")
        assert result == "MANAGES"

    def test_parent_id_self_ref(self):
        result = _rel_type_for_fk("parent_id", "category", "category")
        assert result == "HAS_CHILD"

    def test_supervisor_id_self_ref(self):
        result = _rel_type_for_fk("supervisor_id", "employee", "employee")
        assert result == "SUPERVISES"

    def test_unknown_self_ref(self):
        result = _rel_type_for_fk("reports_to_id", "employee", "employee")
        # Doesn't match _SELF_REF_PATTERNS exactly → REFERENCES_SELF fallback
        assert result == "REFERENCES_SELF"

    def test_generic_fk(self):
        result = _rel_type_for_fk("company_id", "company", "employee")
        assert result == "BELONGS_TO"


# ---------------------------------------------------------------------------
# SchemaMapper.map — full integration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSchemaMapperEntityTable:
    def test_basic_entity_table(self):
        snapshot = _make_snapshot(
            _table(
                "employee",
                _col("id", is_pk=True),
                _col("name", data_type="varchar"),
                _col("email", data_type="varchar"),
            )
        )
        mapper = SchemaMapper()
        rules = mapper.map(snapshot, _CONNECTOR_ID, _GRAPH_ID)

        assert rules.graph_id == _GRAPH_ID
        assert rules.connector_id == _CONNECTOR_ID
        assert len(rules.tables) == 1

        tm = rules.tables[0]
        assert tm.table_name == "employee"
        assert tm.kind == "entity_table"
        assert tm.neo4j_label == "Employee"
        assert tm.pk_column == "id"
        assert len(tm.relationships) == 0
        # Property columns: name, email (not id/pk)
        prop_names = {cm.column_name for cm in tm.property_columns}
        assert prop_names == {"name", "email"}

    def test_entity_table_with_fk_produces_relationship(self):
        dept_table = _table(
            "department",
            _col("id", is_pk=True),
            _col("name", data_type="varchar"),
        )
        emp_table = _table(
            "employee",
            _col("id", is_pk=True),
            _col("department_id", is_fk=True, fk_table="department"),
            _col("name", data_type="varchar"),
        )
        snapshot = _make_snapshot(dept_table, emp_table)
        mapper = SchemaMapper()
        rules = mapper.map(snapshot, _CONNECTOR_ID, _GRAPH_ID)

        emp_mapping = next(t for t in rules.tables if t.table_name == "employee")
        assert len(emp_mapping.relationships) == 1
        rel = emp_mapping.relationships[0]
        assert rel.from_table == "employee"
        assert rel.to_table == "department"
        assert rel.rel_type == "BELONGS_TO"
        assert rel.from_fk_column == "department_id"
        assert rel.via_junction is None

    def test_property_column_camel_case(self):
        snapshot = _make_snapshot(
            _table(
                "user_account",
                _col("id", is_pk=True),
                _col("first_name", data_type="varchar"),
                _col("last_login_at", data_type="timestamp"),
            )
        )
        mapper = SchemaMapper()
        rules = mapper.map(snapshot, _CONNECTOR_ID, _GRAPH_ID)
        tm = rules.tables[0]

        prop_map = {cm.column_name: cm.neo4j_property for cm in tm.property_columns}
        assert prop_map["first_name"] == "firstName"
        assert prop_map["last_login_at"] == "lastLoginAt"


@pytest.mark.unit
class TestSchemaMapperJunctionTable:
    def test_junction_two_fks(self):
        emp = _table("employee", _col("id", is_pk=True), _col("name", data_type="varchar"))
        proj = _table("project", _col("id", is_pk=True), _col("title", data_type="varchar"))
        junc = _table(
            "employee_project",
            _col("employee_id", is_fk=True, fk_table="employee"),
            _col("project_id", is_fk=True, fk_table="project"),
        )
        snapshot = _make_snapshot(emp, proj, junc)
        mapper = SchemaMapper()
        rules = mapper.map(snapshot, _CONNECTOR_ID, _GRAPH_ID)

        junc_mapping = next(t for t in rules.tables if t.table_name == "employee_project")
        assert junc_mapping.kind == "junction_table"
        assert len(junc_mapping.relationships) == 1
        rel = junc_mapping.relationships[0]
        assert rel.from_table == "employee"
        assert rel.to_table == "project"
        assert rel.via_junction == "employee_project"

    def test_junction_with_audit_columns(self):
        emp = _table("employee", _col("id", is_pk=True))
        proj = _table("project", _col("id", is_pk=True))
        junc = _table(
            "employee_project",
            _col("employee_id", is_fk=True, fk_table="employee"),
            _col("project_id", is_fk=True, fk_table="project"),
            _col("created_at", data_type="timestamp"),
            _col("updated_at", data_type="timestamp"),
        )
        snapshot = _make_snapshot(emp, proj, junc)
        mapper = SchemaMapper()
        rules = mapper.map(snapshot, _CONNECTOR_ID, _GRAPH_ID)

        junc_mapping = next(t for t in rules.tables if t.table_name == "employee_project")
        assert junc_mapping.kind == "junction_table"
        # Still exactly one relationship
        assert len(junc_mapping.relationships) == 1


@pytest.mark.unit
class TestSchemaMapperSelfRef:
    def test_self_ref_table(self):
        emp = _table(
            "employee",
            _col("id", is_pk=True),
            _col("manager_id", is_fk=True, fk_table="employee"),
            _col("name", data_type="varchar"),
        )
        snapshot = _make_snapshot(emp)
        mapper = SchemaMapper()
        rules = mapper.map(snapshot, _CONNECTOR_ID, _GRAPH_ID)

        emp_mapping = rules.tables[0]
        assert emp_mapping.kind == "self_ref_table"
        assert len(emp_mapping.relationships) == 1
        rel = emp_mapping.relationships[0]
        assert rel.from_table == "employee"
        assert rel.to_table == "employee"
        assert rel.rel_type == "MANAGES"
        assert rel.via_junction is None

    def test_category_parent_self_ref(self):
        cat = _table(
            "category",
            _col("id", is_pk=True),
            _col("parent_id", is_fk=True, fk_table="category"),
            _col("name", data_type="varchar"),
        )
        snapshot = _make_snapshot(cat)
        mapper = SchemaMapper()
        rules = mapper.map(snapshot, _CONNECTOR_ID, _GRAPH_ID)

        cat_mapping = rules.tables[0]
        assert cat_mapping.kind == "self_ref_table"
        rel = cat_mapping.relationships[0]
        assert rel.rel_type == "HAS_CHILD"


@pytest.mark.unit
class TestSchemaMapperCompositeFk:
    def test_composite_fk_grouped_as_single_relationship(self):
        """Two FK columns pointing to the same target table → one RelationshipMapping."""
        order = _table("order", _col("id", is_pk=True), _col("total", data_type="decimal"))
        # Composite FK: (billing_address_country, billing_address_city) both point to address
        # but since they have the same fk_table they are grouped into one RelationshipMapping
        order_detail = _table(
            "order_detail",
            _col("id", is_pk=True),
            _col("order_id", is_fk=True, fk_table="order", fk_column="id"),
            _col("address_line1", is_fk=True, fk_table="order", fk_column="id"),
        )
        snapshot = _make_snapshot(order, order_detail)
        mapper = SchemaMapper()
        rules = mapper.map(snapshot, _CONNECTOR_ID, _GRAPH_ID)

        od_mapping = next(t for t in rules.tables if t.table_name == "order_detail")
        # Even though there are 2 FK columns to the same table, they are grouped
        # → exactly 1 RelationshipMapping
        assert len(od_mapping.relationships) == 1
        rel = od_mapping.relationships[0]
        assert rel.from_table == "order_detail"
        assert rel.to_table == "order"

    def test_two_different_fk_targets_produce_two_relationships(self):
        dept = _table("department", _col("id", is_pk=True))
        location = _table("location", _col("id", is_pk=True))
        employee = _table(
            "employee",
            _col("id", is_pk=True),
            _col("department_id", is_fk=True, fk_table="department"),
            _col("location_id", is_fk=True, fk_table="location"),
            _col("name", data_type="varchar"),
        )
        snapshot = _make_snapshot(dept, location, employee)
        mapper = SchemaMapper()
        rules = mapper.map(snapshot, _CONNECTOR_ID, _GRAPH_ID)

        emp_mapping = next(t for t in rules.tables if t.table_name == "employee")
        assert emp_mapping.kind == "entity_table"
        assert len(emp_mapping.relationships) == 2
        targets = {rel.to_table for rel in emp_mapping.relationships}
        assert targets == {"department", "location"}


@pytest.mark.unit
class TestSchemaMapperLlmNaming:
    def test_llm_called_for_opaque_junction_name(self):
        """When the junction table name doesn't yield an obvious verb, LLM is called."""
        emp = _table("employee", _col("id", is_pk=True))
        proj = _table("project", _col("id", is_pk=True))
        # 't_ep' is opaque — no recognizable verb after stripping table names
        junc = _table(
            "t_ep",
            _col("employee_id", is_fk=True, fk_table="employee"),
            _col("project_id", is_fk=True, fk_table="project"),
        )
        snapshot = _make_snapshot(emp, proj, junc)
        mapper = SchemaMapper()

        with patch(
            "app.services.schema_mapper._call_llm_for_rel_type",
            return_value="ASSIGNED_TO",
        ) as mock_llm:
            rules = mapper.map(snapshot, _CONNECTOR_ID, _GRAPH_ID)

        # LLM should have been called exactly once (for the opaque junction)
        mock_llm.assert_called_once_with("t_ep", "employee", "project")

        junc_mapping = next(t for t in rules.tables if t.table_name == "t_ep")
        assert junc_mapping.relationships[0].rel_type == "ASSIGNED_TO"

    def test_llm_not_called_for_meaningful_junction_name(self):
        """When the junction name contains a clear verb, LLM is NOT called."""
        emp = _table("employee", _col("id", is_pk=True))
        proj = _table("project", _col("id", is_pk=True))
        # 'employee_works_project' → verb 'WORKS' extracted without LLM
        junc = _table(
            "employee_works_project",
            _col("employee_id", is_fk=True, fk_table="employee"),
            _col("project_id", is_fk=True, fk_table="project"),
        )
        snapshot = _make_snapshot(emp, proj, junc)
        mapper = SchemaMapper()

        with patch(
            "app.services.schema_mapper._call_llm_for_rel_type"
        ) as mock_llm:
            rules = mapper.map(snapshot, _CONNECTOR_ID, _GRAPH_ID)

        mock_llm.assert_not_called()

        junc_mapping = next(t for t in rules.tables if t.table_name == "employee_works_project")
        assert junc_mapping.relationships[0].rel_type == "WORKS"

    def test_llm_context_sent_correctly(self):
        """LLM receives the junction name, from_table, and to_table."""
        emp = _table("employee", _col("id", is_pk=True))
        proj = _table("project", _col("id", is_pk=True))
        junc = _table(
            "rel_012",
            _col("employee_id", is_fk=True, fk_table="employee"),
            _col("project_id", is_fk=True, fk_table="project"),
        )
        snapshot = _make_snapshot(emp, proj, junc)
        mapper = SchemaMapper()

        with patch(
            "app.services.schema_mapper._call_llm_for_rel_type",
            return_value="WORKS_ON",
        ) as mock_llm:
            mapper.map(snapshot, _CONNECTOR_ID, _GRAPH_ID)

        call_args = mock_llm.call_args
        assert call_args[0][0] == "rel_012"       # junction table name
        assert call_args[0][1] == "employee"       # from_table
        assert call_args[0][2] == "project"        # to_table


@pytest.mark.unit
class TestSchemaMapperGraphIdPresence:
    """Verify graph_id is threaded through every output structure."""

    def test_graph_id_on_mapping_rules(self):
        snapshot = _make_snapshot(
            _table("user", _col("id", is_pk=True), _col("name", data_type="varchar"))
        )
        mapper = SchemaMapper()
        rules = mapper.map(snapshot, _CONNECTOR_ID, _GRAPH_ID)

        assert rules.graph_id == _GRAPH_ID

    def test_graph_id_on_rules_with_multiple_tables(self):
        dept = _table("department", _col("id", is_pk=True))
        emp = _table(
            "employee",
            _col("id", is_pk=True),
            _col("department_id", is_fk=True, fk_table="department"),
        )
        snapshot = _make_snapshot(dept, emp)
        mapper = SchemaMapper()
        rules = mapper.map(snapshot, _CONNECTOR_ID, _GRAPH_ID)

        assert rules.graph_id == _GRAPH_ID
        # All tables are present in output
        assert len(rules.tables) == 2

    def test_connector_id_present(self):
        snapshot = _make_snapshot(
            _table("user", _col("id", is_pk=True))
        )
        mapper = SchemaMapper()
        rules = mapper.map(snapshot, _CONNECTOR_ID, _GRAPH_ID)

        assert rules.connector_id == _CONNECTOR_ID

    def test_generated_at_is_iso_timestamp(self):
        snapshot = _make_snapshot(
            _table("user", _col("id", is_pk=True))
        )
        mapper = SchemaMapper()
        rules = mapper.map(snapshot, _CONNECTOR_ID, _GRAPH_ID)

        # Should parse as ISO 8601 without error
        parsed = datetime.fromisoformat(rules.generated_at)
        assert parsed is not None

    def test_different_graph_ids_produce_different_rules(self):
        snapshot = _make_snapshot(
            _table("user", _col("id", is_pk=True))
        )
        mapper = SchemaMapper()
        rules_a = mapper.map(snapshot, _CONNECTOR_ID, "graph-A")
        rules_b = mapper.map(snapshot, _CONNECTOR_ID, "graph-B")

        assert rules_a.graph_id == "graph-A"
        assert rules_b.graph_id == "graph-B"


@pytest.mark.unit
class TestSchemaMapperEdgeCases:
    def test_table_with_no_pk_is_skipped(self):
        """Tables without a PK cannot form entity nodes — they are skipped."""
        no_pk = _table(
            "legacy_view",
            _col("name", data_type="varchar"),
            _col("value", data_type="varchar"),
        )
        snapshot = _make_snapshot(no_pk)
        mapper = SchemaMapper()
        rules = mapper.map(snapshot, _CONNECTOR_ID, _GRAPH_ID)

        assert len(rules.tables) == 0

    def test_empty_snapshot(self):
        snapshot = _make_snapshot()
        mapper = SchemaMapper()
        rules = mapper.map(snapshot, _CONNECTOR_ID, _GRAPH_ID)

        assert rules.graph_id == _GRAPH_ID
        assert rules.tables == []

    def test_fk_without_matching_pk_table_uses_id_fallback(self):
        """FK pointing to a table not in the snapshot — target PK defaults to 'id'."""
        employee = _table(
            "employee",
            _col("id", is_pk=True),
            _col("department_id", is_fk=True, fk_table="department"),
            _col("name", data_type="varchar"),
        )
        # 'department' table is NOT in the snapshot
        snapshot = _make_snapshot(employee)
        mapper = SchemaMapper()
        rules = mapper.map(snapshot, _CONNECTOR_ID, _GRAPH_ID)

        emp_mapping = rules.tables[0]
        assert len(emp_mapping.relationships) == 1
        rel = emp_mapping.relationships[0]
        assert rel.to_fk_column == "id"  # fallback
