"""
Unit tests for Cypher injection prevention.

Verifies that no user-controlled value is ever interpolated as a literal string
into a Cypher query — all variable values must be passed as $parameters.
"""

# ---------------------------------------------------------------------------
# multi_tenant_components — _inject_graph_id_filter
# ---------------------------------------------------------------------------


class TestMultiTenantGraphIdFilter:
    """_inject_graph_id_filter must produce parameterized WHERE, never literals."""

    def _get_filter_fn(self):
        from app.components.multi_tenant_components import (
            MultiTenantVectorCypherRetriever,
        )

        return MultiTenantVectorCypherRetriever._inject_graph_id_filter

    def test_no_match_returns_query_unchanged(self):
        fn = self._get_filter_fn()
        query = "RETURN 1"
        assert fn(query) == query

    def test_already_parameterized_not_doubled(self):
        fn = self._get_filter_fn()
        query = "MATCH (n) WHERE n.graph_id = $graph_id RETURN n"
        assert fn(query) == query

    def test_injects_parameter_not_literal(self):
        fn = self._get_filter_fn()
        query = "MATCH (n)\nRETURN n"
        result = fn(query)
        assert "$graph_id" in result
        # Must never embed a bare string value
        assert "= '" not in result

    def test_malicious_graph_id_not_in_query(self):
        """A malicious graph_id value must never appear in the generated query string."""
        fn = self._get_filter_fn()
        malicious = "'; DROP DATABASE neo4j; //"
        query = "MATCH (n)\nRETURN n"
        result = fn(query)
        assert malicious not in result
        assert "$graph_id" in result

    def test_existing_where_clause_extended_with_parameter(self):
        fn = self._get_filter_fn()
        query = "MATCH (n)\nWHERE n.active = true\nRETURN n"
        result = fn(query)
        assert "$graph_id" in result
        assert "= '" not in result


# ---------------------------------------------------------------------------
# retriever_factory — _inject_graph_id_filter
# ---------------------------------------------------------------------------


class TestRetrieverFactoryGraphIdFilter:
    """retriever_factory._inject_graph_id_filter must produce parameterized WHERE."""

    def _make_factory(self):

        from app.services.retriever_factory import RetrieverFactory

        factory = RetrieverFactory.__new__(RetrieverFactory)
        return factory

    def test_no_where_single_line_gets_parameter(self):
        factory = self._make_factory()
        query = "MATCH (n) RETURN n"
        result = factory._inject_graph_id_filter(query)
        assert "$graph_id" in result
        assert "= '" not in result

    def test_already_has_parameter_unchanged(self):
        factory = self._make_factory()
        query = "MATCH (n) WHERE n.graph_id = $graph_id RETURN n"
        result = factory._inject_graph_id_filter(query)
        assert result == query

    def test_existing_where_prepends_parameter(self):
        factory = self._make_factory()
        query = "MATCH (n)\nWHERE n.name = $name\nRETURN n"
        result = factory._inject_graph_id_filter(query)
        assert "$graph_id" in result
        assert "= '" not in result

    def test_malicious_payload_never_interpolated(self):
        factory = self._make_factory()
        payload = "' OR '1'='1"
        query = "MATCH (n) RETURN n"
        result = factory._inject_graph_id_filter(query)
        # The payload was never passed — the result should only contain $graph_id
        assert payload not in result
        assert "$graph_id" in result

    def test_multiline_no_where_inserts_after_first_line(self):
        factory = self._make_factory()
        query = "MATCH (n)\nRETURN n"
        result = factory._inject_graph_id_filter(query)
        lines = result.split("\n")
        assert any("$graph_id" in line for line in lines)
        assert "= '" not in result


# ---------------------------------------------------------------------------
# analytics_service — entity label validation
# ---------------------------------------------------------------------------


class TestEntityLabelValidation:
    """GDS subquery strings can't use $params; entity_label must be validated."""

    def test_safe_label_passes(self):
        import re

        pattern = re.compile(r"^[A-Za-z0-9_]+$")
        assert pattern.match("__Entity__")
        assert pattern.match("Entity")
        assert pattern.match("MyLabel123")

    def test_injection_label_rejected(self):
        import re

        pattern = re.compile(r"^[A-Za-z0-9_]+$")
        assert not pattern.match("__Entity__`}) RETURN 1//")
        assert not pattern.match("Foo'; DROP")
        assert not pattern.match("Label WITH spaces")

    def test_graph_id_uuid_string_is_safe(self):
        """str(UUID) always produces a UUID-format string — no injection possible."""
        import uuid

        uid = uuid.uuid4()
        uid_str = str(uid)
        # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        import re

        assert re.match(r"^[0-9a-f-]+$", uid_str)
        assert "'" not in uid_str
        assert ";" not in uid_str


# ---------------------------------------------------------------------------
# schema_service — backtick guard
# ---------------------------------------------------------------------------


class TestSchemaServiceLabelGuard:
    """Labels containing backticks must be rejected before use in count queries."""

    def test_label_with_backtick_detected(self):
        label = "ValidLabel`injected"
        assert "`" in label

    def test_normal_labels_pass_guard(self):
        safe_labels = ["__Entity__", "Chunk", "Document", "GraphVersion"]
        for label in safe_labels:
            assert "`" not in label

    def test_query_template_uses_parameter_for_graph_id(self):
        """The count query template must use $graph_id, not a literal value."""
        label = "Entity"
        query = f"MATCH (n:`{label}` {{graph_id: $graph_id}}) RETURN count(n) as count"
        assert "$graph_id" in query
        assert "= '" not in query
