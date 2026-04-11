"""Unit tests for pure helper functions in database_connector_service.py.

No Neo4j or DB connections required — all helpers are pure Python functions.
"""

from datetime import UTC, datetime

import pytest
from fastapi import HTTPException

from app.services.database_connector_service import (
    ColumnMeta,
    DatabaseConnectorType,
    SchemaSnapshot,
    TableMeta,
    _diff_snapshots,
    _entity_label,
    _infer_mongo_fk_table,
    _rel_type,
    _to_pascal_case,
    validate_host_ssrf,
)

# ---------------------------------------------------------------------------
# validate_host_ssrf
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateHostSsrf:
    def test_ipv6_loopback_blocked(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_host_ssrf("::1")
        assert exc_info.value.status_code == 422

    def test_link_local_169_254_blocked(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_host_ssrf("169.254.169.254")
        assert exc_info.value.status_code == 422

    def test_metadata_google_internal_blocked(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_host_ssrf("metadata.google.internal")
        assert exc_info.value.status_code == 422

    def test_valid_public_ip_allowed(self):
        # Should not raise
        validate_host_ssrf("8.8.8.8")

    def test_private_10_block_blocked(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_host_ssrf("10.0.0.1")
        assert exc_info.value.status_code == 422

    def test_localhost_blocked(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_host_ssrf("localhost")
        assert exc_info.value.status_code == 422


# ---------------------------------------------------------------------------
# _diff_snapshots
# ---------------------------------------------------------------------------


def _make_snapshot(tables: list[TableMeta]) -> SchemaSnapshot:
    return SchemaSnapshot(
        connector_type=DatabaseConnectorType.POSTGRESQL,
        database="testdb",
        tables=tables,
        captured_at=datetime.now(UTC),
    )


def _simple_table(name: str, columns: list[ColumnMeta] | None = None) -> TableMeta:
    cols = columns or [
        ColumnMeta(
            name="id", data_type="integer", nullable=False, is_pk=True, is_fk=False
        )
    ]
    return TableMeta(name=name, schema_name="public", columns=cols)


@pytest.mark.unit
class TestDiffSnapshots:
    def test_added_table(self):
        prev = _make_snapshot([_simple_table("users")])
        cur = _make_snapshot([_simple_table("users"), _simple_table("orders")])
        diff = _diff_snapshots(prev, cur)
        assert "orders" in diff["added_tables"]
        assert diff["removed_tables"] == []

    def test_removed_table(self):
        prev = _make_snapshot([_simple_table("users"), _simple_table("orders")])
        cur = _make_snapshot([_simple_table("users")])
        diff = _diff_snapshots(prev, cur)
        assert "orders" in diff["removed_tables"]
        assert diff["added_tables"] == []

    def test_altered_column_type(self):
        col_prev = ColumnMeta(
            name="email", data_type="varchar", nullable=True, is_pk=False, is_fk=False
        )
        col_cur = ColumnMeta(
            name="email", data_type="text", nullable=True, is_pk=False, is_fk=False
        )
        prev = _make_snapshot([_simple_table("users", [col_prev])])
        cur = _make_snapshot([_simple_table("users", [col_cur])])
        diff = _diff_snapshots(prev, cur)
        assert len(diff["altered_tables"]) == 1
        assert "email" in diff["altered_tables"][0]["type_changed"]

    def test_no_changes(self):
        snapshot = _make_snapshot([_simple_table("users")])
        diff = _diff_snapshots(snapshot, snapshot)
        assert diff["added_tables"] == []
        assert diff["removed_tables"] == []
        assert diff["altered_tables"] == []


# ---------------------------------------------------------------------------
# _to_pascal_case
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToPascalCase:
    def test_hyphenated_name(self):
        assert _to_pascal_case("user-profile") == "UserProfile"

    def test_underscored_name(self):
        assert _to_pascal_case("order_line_item") == "OrderLineItem"

    def test_numeric_prefix_edge_case(self):
        # Numbers are kept as-is; capitalize() on a digit segment is a no-op
        assert _to_pascal_case("2fa_tokens") == "2faTokens"

    def test_single_word(self):
        assert _to_pascal_case("users") == "Users"


# ---------------------------------------------------------------------------
# _infer_mongo_fk_table
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInferMongoFkTable:
    def test_suffix_underscore_id(self):
        assert _infer_mongo_fk_table("user_id") == "user"

    def test_suffix_Id(self):
        assert _infer_mongo_fk_table("orderId") == "order"

    def test_suffix_Ref(self):
        assert _infer_mongo_fk_table("accountRef") == "account"

    def test_suffix_ref(self):
        assert _infer_mongo_fk_table("productref") == "product"

    def test_non_matching_name(self):
        assert _infer_mongo_fk_table("email") is None

    def test_suffix_only_returns_none(self):
        # Field name IS the suffix — nothing before it, so should return None
        assert _infer_mongo_fk_table("_id") is None


# ---------------------------------------------------------------------------
# _rel_type
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRelType:
    def test_valid_label_returns_correct_type(self):
        assert _rel_type("Order") == "REFERENCES_ORDER"

    def test_mixed_case_label(self):
        assert _rel_type("OrderLineItem") == "REFERENCES_ORDERLINEITEM"

    def test_invalid_cypher_identifier_raises(self):
        # A label that contains a character producing an invalid identifier
        with pytest.raises(ValueError, match="not a valid Cypher identifier"):
            _rel_type("table-with-hyphens")

    def test_label_with_spaces_raises(self):
        with pytest.raises(ValueError, match="not a valid Cypher identifier"):
            _rel_type("my table")


# ---------------------------------------------------------------------------
# _entity_label
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEntityLabel:
    def test_dotted_schema_table_name(self):
        # Only the table part (after last dot) should be used
        assert _entity_label("public.order_items") == "OrderItems"

    def test_standard_table_name(self):
        assert _entity_label("user_accounts") == "UserAccounts"

    def test_no_dot(self):
        assert _entity_label("products") == "Products"
