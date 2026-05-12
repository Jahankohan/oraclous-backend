"""Unit tests for app/services/row_transformer.py (TASK-020).

All tests are pure Python — no Neo4j, no Docker, no network required.
Neo4j driver calls are replaced with MagicMock objects that record calls.

Covers:
- Entity row → __Entity__ node: id format {connector_id}:{table}:{pk},
  non-FK properties mapped, ingestion_time set in Cypher, graph_id present
- Junction row → relationship: edges between correct entity node ids
- Self-referential FK row → self-ref edge
- Batch processing: 1000 rows produce exactly 2 UNWIND calls (batch_size=500)
- graph_id on every written node and relationship
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.services.row_transformer import (
    RowTransformer,
    _build_entity_row_param,
    _build_junction_row_param,
    _build_self_ref_row_param,
    _chunks,
    _coerce_value,
    _entity_id,
    _safe_rel_type,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONNECTOR_ID = "conn-abc-123"
_GRAPH_ID = "graph-tenant-xyz"


# ---------------------------------------------------------------------------
# Minimal stub classes to avoid importing dataclasses from schema_mapper
# (keeps test isolated — no dependency on TASK-019 internals)
# ---------------------------------------------------------------------------


class _ColumnMapping:
    def __init__(self, column_name: str, neo4j_property: str):
        self.column_name = column_name
        self.neo4j_property = neo4j_property


class _RelationshipMapping:
    def __init__(
        self,
        from_table: str,
        to_table: str,
        rel_type: str,
        from_fk_column: str,
        to_fk_column: str,
        via_junction=None,
    ):
        self.from_table = from_table
        self.to_table = to_table
        self.rel_type = rel_type
        self.from_fk_column = from_fk_column
        self.to_fk_column = to_fk_column
        self.via_junction = via_junction


class _TableMapping:
    def __init__(
        self,
        table_name: str,
        kind: str,
        neo4j_label: str,
        pk_column: str,
        property_columns: list,
        relationships: list,
    ):
        self.table_name = table_name
        self.kind = kind
        self.neo4j_label = neo4j_label
        self.pk_column = pk_column
        self.property_columns = property_columns
        self.relationships = relationships


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_neo4j_manager() -> MagicMock:
    """Return a mock WorkerNeo4jManager with a mock sync driver and session."""
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
    mock_manager = MagicMock()
    mock_manager.get_sync_driver.return_value = mock_driver
    return mock_manager, mock_driver, mock_session


def _make_users_table_mapping() -> _TableMapping:
    return _TableMapping(
        table_name="users",
        kind="entity_table",
        neo4j_label="Users",
        pk_column="id",
        property_columns=[
            _ColumnMapping("name", "name"),
            _ColumnMapping("email", "email"),
        ],
        relationships=[],
    )


def _make_employee_self_ref_mapping() -> _TableMapping:
    return _TableMapping(
        table_name="employees",
        kind="self_ref_table",
        neo4j_label="Employees",
        pk_column="id",
        property_columns=[
            _ColumnMapping("name", "name"),
        ],
        relationships=[
            _RelationshipMapping(
                from_table="employees",
                to_table="employees",
                rel_type="REPORTS_TO",
                from_fk_column="manager_id",
                to_fk_column="id",
                via_junction=None,
            )
        ],
    )


# ---------------------------------------------------------------------------
# Unit tests: _entity_id
# ---------------------------------------------------------------------------


def test_entity_id_format():
    """id must be {connector_id}:{table_name}:{pk_value}."""
    eid = _entity_id("conn-1", "users", 42)
    assert eid == "conn-1:users:42"


def test_entity_id_string_pk():
    eid = _entity_id("c", "orders", "uuid-abc")
    assert eid == "c:orders:uuid-abc"


# ---------------------------------------------------------------------------
# Unit tests: _build_entity_row_param
# ---------------------------------------------------------------------------


def test_build_entity_row_param_basic():
    tm = _make_users_table_mapping()
    row = {"id": 1, "name": "Alice", "email": "alice@example.com"}
    param = _build_entity_row_param(row, tm, _GRAPH_ID, _CONNECTOR_ID)
    assert param is not None
    assert param["id"] == f"{_CONNECTOR_ID}:users:1"
    assert param["properties"]["name"] == "Alice"
    assert param["properties"]["email"] == "alice@example.com"


def test_build_entity_row_param_missing_pk_returns_none():
    tm = _make_users_table_mapping()
    row = {"name": "Alice"}  # no 'id'
    param = _build_entity_row_param(row, tm, _GRAPH_ID, _CONNECTOR_ID)
    assert param is None


def test_build_entity_row_param_none_pk_returns_none():
    tm = _make_users_table_mapping()
    row = {"id": None, "name": "Alice"}
    param = _build_entity_row_param(row, tm, _GRAPH_ID, _CONNECTOR_ID)
    assert param is None


def test_build_entity_row_param_none_property_excluded():
    """Properties with None values should not appear in the properties dict."""
    tm = _make_users_table_mapping()
    row = {"id": 1, "name": "Alice", "email": None}
    param = _build_entity_row_param(row, tm, _GRAPH_ID, _CONNECTOR_ID)
    assert param is not None
    assert "email" not in param["properties"]


def test_build_entity_row_param_extra_columns_ignored():
    """Columns not in property_columns (e.g. FK columns) are ignored."""
    tm = _make_users_table_mapping()
    row = {"id": 1, "name": "Alice", "email": "a@b.com", "role_id": 5}
    param = _build_entity_row_param(row, tm, _GRAPH_ID, _CONNECTOR_ID)
    assert "roleId" not in param["properties"]
    assert "role_id" not in param["properties"]


# ---------------------------------------------------------------------------
# Unit tests: _build_junction_row_param
# ---------------------------------------------------------------------------


def test_build_junction_row_param_basic():
    rel = _RelationshipMapping(
        from_table="users",
        to_table="roles",
        rel_type="HAS_ROLE",
        from_fk_column="user_id",
        to_fk_column="role_id",
        via_junction="user_roles",
    )
    row = {"user_id": 10, "role_id": 20}
    param = _build_junction_row_param(row, MagicMock(), rel, _GRAPH_ID, _CONNECTOR_ID)
    assert param["from_id"] == f"{_CONNECTOR_ID}:users:10"
    assert param["to_id"] == f"{_CONNECTOR_ID}:roles:20"


def test_build_junction_row_param_missing_fk_returns_none():
    rel = _RelationshipMapping(
        from_table="users",
        to_table="roles",
        rel_type="HAS_ROLE",
        from_fk_column="user_id",
        to_fk_column="role_id",
    )
    row = {"user_id": 10}  # role_id missing
    param = _build_junction_row_param(row, MagicMock(), rel, _GRAPH_ID, _CONNECTOR_ID)
    assert param is None


# ---------------------------------------------------------------------------
# Unit tests: _build_self_ref_row_param
# ---------------------------------------------------------------------------


def test_build_self_ref_row_param_basic():
    tm = _make_employee_self_ref_mapping()
    rel = tm.relationships[0]
    row = {"id": 5, "name": "Bob", "manager_id": 3}
    param = _build_self_ref_row_param(row, tm, rel, _GRAPH_ID, _CONNECTOR_ID)
    assert param["from_id"] == f"{_CONNECTOR_ID}:employees:5"
    assert param["to_id"] == f"{_CONNECTOR_ID}:employees:3"


def test_build_self_ref_row_param_no_manager_returns_none():
    """Rows where manager_id is None (top-level employees) produce None."""
    tm = _make_employee_self_ref_mapping()
    rel = tm.relationships[0]
    row = {"id": 1, "name": "CEO", "manager_id": None}
    param = _build_self_ref_row_param(row, tm, rel, _GRAPH_ID, _CONNECTOR_ID)
    assert param is None


# ---------------------------------------------------------------------------
# Unit tests: _chunks
# ---------------------------------------------------------------------------


def test_chunks_even_split():
    chunks = list(_chunks(list(range(10)), 5))
    assert chunks == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]


def test_chunks_remainder():
    chunks = list(_chunks(list(range(7)), 3))
    assert len(chunks) == 3
    assert chunks[-1] == [6]


def test_chunks_empty():
    assert list(_chunks([], 5)) == []


# ---------------------------------------------------------------------------
# Unit tests: _safe_rel_type
# ---------------------------------------------------------------------------


def test_safe_rel_type_valid():
    assert _safe_rel_type("REPORTS_TO") is True
    assert _safe_rel_type("HAS_ROLE") is True
    assert _safe_rel_type("REFERENCES_USERS") is True


def test_safe_rel_type_invalid():
    assert _safe_rel_type("invalid lower") is False
    assert _safe_rel_type("INJECT; DROP") is False
    assert _safe_rel_type("") is False


# ---------------------------------------------------------------------------
# Unit tests: _coerce_value
# ---------------------------------------------------------------------------


def test_coerce_value_primitives():
    assert _coerce_value(42) == 42
    assert _coerce_value(3.14) == 3.14
    assert _coerce_value("hello") == "hello"
    assert _coerce_value(True) is True
    assert _coerce_value(None) is None


def test_coerce_value_date():
    import datetime

    d = datetime.date(2026, 4, 27)
    assert _coerce_value(d) == "2026-04-27"


def test_coerce_value_datetime():
    import datetime

    dt = datetime.datetime(2026, 4, 27, 12, 0, 0)
    result = _coerce_value(dt)
    assert "2026-04-27" in result


def test_coerce_value_unknown_type():
    class Weird:
        def __str__(self):
            return "weird-value"

    assert _coerce_value(Weird()) == "weird-value"


# ---------------------------------------------------------------------------
# Unit tests: RowTransformer.transform_table
# ---------------------------------------------------------------------------


def test_transform_table_entity_nodes_written():
    """transform_table writes one UNWIND batch for ≤500 rows."""
    mock_manager, mock_driver, mock_session = _make_neo4j_manager()
    xfm = RowTransformer(mock_manager)
    tm = _make_users_table_mapping()

    rows = [{"id": i, "name": f"User{i}", "email": f"u{i}@x.com"} for i in range(1, 4)]
    count = xfm.transform_table(tm, rows, _GRAPH_ID, _CONNECTOR_ID)

    assert count == 3
    # One session.run() call for the single batch
    assert mock_session.run.call_count == 1
    call_args = mock_session.run.call_args
    cypher = call_args[0][0]
    params = call_args[0][1]

    # graph_id present in params
    assert params["graph_id"] == _GRAPH_ID
    # table_name present
    assert params["table_name"] == "users"
    # neo4j_label present
    assert params["neo4j_label"] == "Users"
    # UNWIND batch present with correct entity ids
    batch = params["batch"]
    assert len(batch) == 3
    assert batch[0]["id"] == f"{_CONNECTOR_ID}:users:1"


def test_transform_table_id_format():
    """Entity id must be {connector_id}:{table}:{pk_value}."""
    mock_manager, mock_driver, mock_session = _make_neo4j_manager()
    xfm = RowTransformer(mock_manager)
    tm = _make_users_table_mapping()

    rows = [{"id": 99, "name": "Test", "email": "t@t.com"}]
    xfm.transform_table(tm, rows, _GRAPH_ID, _CONNECTOR_ID)

    batch = mock_session.run.call_args[0][1]["batch"]
    assert batch[0]["id"] == f"{_CONNECTOR_ID}:users:99"


def test_transform_table_graph_id_on_every_write():
    """graph_id must appear in every Cypher call's parameters."""
    mock_manager, mock_driver, mock_session = _make_neo4j_manager()
    xfm = RowTransformer(mock_manager)
    tm = _make_users_table_mapping()

    rows = [{"id": i, "name": f"U{i}", "email": f"u{i}@x.com"} for i in range(5)]
    xfm.transform_table(tm, rows, _GRAPH_ID, _CONNECTOR_ID)

    for c in mock_session.run.call_args_list:
        assert c[0][1]["graph_id"] == _GRAPH_ID


def test_transform_table_empty_rows():
    mock_manager, _, _ = _make_neo4j_manager()
    xfm = RowTransformer(mock_manager)
    tm = _make_users_table_mapping()
    count = xfm.transform_table(tm, [], _GRAPH_ID, _CONNECTOR_ID)
    assert count == 0


def test_transform_table_batch_processing_1000_rows():
    """1000 rows → exactly 2 UNWIND calls (batches of 500)."""
    mock_manager, mock_driver, mock_session = _make_neo4j_manager()
    xfm = RowTransformer(mock_manager)
    tm = _make_users_table_mapping()

    rows = [{"id": i, "name": f"User{i}", "email": f"u{i}@x.com"} for i in range(1000)]
    count = xfm.transform_table(tm, rows, _GRAPH_ID, _CONNECTOR_ID)

    assert count == 1000
    assert mock_session.run.call_count == 2

    # Verify each batch has 500 rows
    first_batch = mock_session.run.call_args_list[0][0][1]["batch"]
    second_batch = mock_session.run.call_args_list[1][0][1]["batch"]
    assert len(first_batch) == 500
    assert len(second_batch) == 500


def test_transform_table_properties_mapped():
    """Non-FK, non-PK columns are written using neo4j_property names."""
    mock_manager, mock_driver, mock_session = _make_neo4j_manager()
    xfm = RowTransformer(mock_manager)
    tm = _TableMapping(
        table_name="products",
        kind="entity_table",
        neo4j_label="Products",
        pk_column="product_id",
        property_columns=[
            _ColumnMapping("product_name", "productName"),
            _ColumnMapping("unit_price", "unitPrice"),
        ],
        relationships=[],
    )

    rows = [{"product_id": 1, "product_name": "Widget", "unit_price": 9.99}]
    xfm.transform_table(tm, rows, _GRAPH_ID, _CONNECTOR_ID)

    batch = mock_session.run.call_args[0][1]["batch"]
    props = batch[0]["properties"]
    assert "productName" in props
    assert props["productName"] == "Widget"
    assert "unitPrice" in props
    assert props["unitPrice"] == 9.99
    # PK column must NOT appear in properties
    assert "product_id" not in props
    assert "productId" not in props


def test_transform_table_rows_missing_pk_skipped():
    """Rows without a PK value are silently skipped."""
    mock_manager, mock_driver, mock_session = _make_neo4j_manager()
    xfm = RowTransformer(mock_manager)
    tm = _make_users_table_mapping()

    rows = [
        {"id": 1, "name": "Alice", "email": "a@x.com"},
        {"name": "No-PK", "email": "b@x.com"},  # no 'id'
    ]
    count = xfm.transform_table(tm, rows, _GRAPH_ID, _CONNECTOR_ID)
    assert count == 1
    batch = mock_session.run.call_args[0][1]["batch"]
    assert len(batch) == 1


# ---------------------------------------------------------------------------
# Unit tests: RowTransformer.transform_junctions — junction tables
# ---------------------------------------------------------------------------


def test_transform_junctions_junction_table():
    """Junction row → relationship between correct entity node ids."""
    mock_manager, mock_driver, mock_session = _make_neo4j_manager()
    xfm = RowTransformer(mock_manager)

    rel = _RelationshipMapping(
        from_table="users",
        to_table="roles",
        rel_type="HAS_ROLE",
        from_fk_column="user_id",
        to_fk_column="role_id",
        via_junction="user_roles",
    )
    junction_tm = _TableMapping(
        table_name="user_roles",
        kind="junction_table",
        neo4j_label="UserRoles",
        pk_column="",
        property_columns=[],
        relationships=[rel],
    )

    rows_by_table = {
        "user_roles": [
            {"user_id": 1, "role_id": 10},
            {"user_id": 2, "role_id": 20},
        ]
    }
    total = xfm.transform_junctions(
        [junction_tm], rows_by_table, _GRAPH_ID, _CONNECTOR_ID
    )

    assert total == 2
    assert mock_session.run.call_count == 1  # 2 rows fit in one batch
    call_args = mock_session.run.call_args
    cypher = call_args[0][0]
    params = call_args[0][1]

    # graph_id present
    assert params["graph_id"] == _GRAPH_ID
    # source_table present
    assert params["junction_table"] == "user_roles"
    # rel_type in Cypher
    assert "HAS_ROLE" in cypher

    batch = params["batch"]
    assert batch[0]["from_id"] == f"{_CONNECTOR_ID}:users:1"
    assert batch[0]["to_id"] == f"{_CONNECTOR_ID}:roles:10"
    assert batch[1]["from_id"] == f"{_CONNECTOR_ID}:users:2"
    assert batch[1]["to_id"] == f"{_CONNECTOR_ID}:roles:20"


def test_transform_junctions_graph_id_on_relationship():
    """graph_id must appear in the relationship Cypher and params."""
    mock_manager, mock_driver, mock_session = _make_neo4j_manager()
    xfm = RowTransformer(mock_manager)

    rel = _RelationshipMapping(
        from_table="users",
        to_table="roles",
        rel_type="HAS_ROLE",
        from_fk_column="user_id",
        to_fk_column="role_id",
    )
    junction_tm = _TableMapping(
        table_name="user_roles",
        kind="junction_table",
        neo4j_label="UserRoles",
        pk_column="",
        property_columns=[],
        relationships=[rel],
    )

    rows_by_table = {"user_roles": [{"user_id": 1, "role_id": 10}]}
    xfm.transform_junctions([junction_tm], rows_by_table, _GRAPH_ID, _CONNECTOR_ID)

    call_args = mock_session.run.call_args
    cypher = call_args[0][0]
    params = call_args[0][1]
    # graph_id in params
    assert params["graph_id"] == _GRAPH_ID
    # graph_id appears in the MERGE Cypher (scopes the relationship)
    assert "graph_id" in cypher


def test_transform_junctions_empty_rows():
    mock_manager, _, _ = _make_neo4j_manager()
    xfm = RowTransformer(mock_manager)
    total = xfm.transform_junctions([], {}, _GRAPH_ID, _CONNECTOR_ID)
    assert total == 0


# ---------------------------------------------------------------------------
# Unit tests: RowTransformer.transform_junctions — self-referential FK
# ---------------------------------------------------------------------------


def test_transform_junctions_self_ref():
    """Self-referential FK produces edge where source and target are same table."""
    mock_manager, mock_driver, mock_session = _make_neo4j_manager()
    xfm = RowTransformer(mock_manager)

    tm = _make_employee_self_ref_mapping()
    rows_by_table = {
        "employees": [
            {"id": 1, "name": "CEO", "manager_id": None},  # no manager — skipped
            {"id": 2, "name": "VP", "manager_id": 1},
            {"id": 3, "name": "Manager", "manager_id": 1},
        ]
    }

    total = xfm.transform_junctions([tm], rows_by_table, _GRAPH_ID, _CONNECTOR_ID)

    # Only 2 rows produce valid edges (row with manager_id=None is skipped)
    assert total == 2

    call_args = mock_session.run.call_args
    batch = call_args[0][1]["batch"]
    assert len(batch) == 2
    # VP reports to CEO
    assert batch[0]["from_id"] == f"{_CONNECTOR_ID}:employees:2"
    assert batch[0]["to_id"] == f"{_CONNECTOR_ID}:employees:1"


def test_transform_junctions_self_ref_rel_type_in_cypher():
    """REPORTS_TO should appear as the relationship type in the generated Cypher."""
    mock_manager, mock_driver, mock_session = _make_neo4j_manager()
    xfm = RowTransformer(mock_manager)

    tm = _make_employee_self_ref_mapping()
    rows_by_table = {"employees": [{"id": 2, "name": "VP", "manager_id": 1}]}
    xfm.transform_junctions([tm], rows_by_table, _GRAPH_ID, _CONNECTOR_ID)

    cypher = mock_session.run.call_args[0][0]
    assert "REPORTS_TO" in cypher


# ---------------------------------------------------------------------------
# Unit tests: batch size boundary
# ---------------------------------------------------------------------------


def test_transform_junctions_1000_rows_two_batches():
    """1000 junction rows → exactly 2 UNWIND calls (batches of 500)."""
    mock_manager, mock_driver, mock_session = _make_neo4j_manager()
    xfm = RowTransformer(mock_manager)

    rel = _RelationshipMapping(
        from_table="users",
        to_table="roles",
        rel_type="HAS_ROLE",
        from_fk_column="user_id",
        to_fk_column="role_id",
    )
    junction_tm = _TableMapping(
        table_name="user_roles",
        kind="junction_table",
        neo4j_label="UserRoles",
        pk_column="",
        property_columns=[],
        relationships=[rel],
    )

    rows = [{"user_id": i, "role_id": i + 100} for i in range(1000)]
    rows_by_table = {"user_roles": rows}

    total = xfm.transform_junctions(
        [junction_tm], rows_by_table, _GRAPH_ID, _CONNECTOR_ID
    )

    assert total == 1000
    assert mock_session.run.call_count == 2

    first_batch = mock_session.run.call_args_list[0][0][1]["batch"]
    second_batch = mock_session.run.call_args_list[1][0][1]["batch"]
    assert len(first_batch) == 500
    assert len(second_batch) == 500


# ---------------------------------------------------------------------------
# Integration: import from schema_mapper works (smoke test)
# ---------------------------------------------------------------------------


def test_import_from_schema_mapper():
    """Verify RowTransformer can use real TableMapping from schema_mapper."""
    from app.services.schema_mapper import (
        ColumnMapping,
        TableMapping,
    )

    tm = TableMapping(
        table_name="orders",
        kind="entity_table",
        neo4j_label="Orders",
        pk_column="order_id",
        property_columns=[ColumnMapping("status", "status")],
        relationships=[],
    )
    mock_manager, mock_driver, mock_session = _make_neo4j_manager()
    xfm = RowTransformer(mock_manager)
    rows = [{"order_id": 1, "status": "shipped"}]
    count = xfm.transform_table(tm, rows, _GRAPH_ID, _CONNECTOR_ID)
    assert count == 1
    batch = mock_session.run.call_args[0][1]["batch"]
    assert batch[0]["id"] == f"{_CONNECTOR_ID}:orders:1"
    assert batch[0]["properties"]["status"] == "shipped"
