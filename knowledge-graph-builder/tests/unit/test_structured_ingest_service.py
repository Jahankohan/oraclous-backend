"""Unit tests for StructuredIngestService (STORY-9).

Verifies:
- _record_to_properties projects JSON correctly (primitives, arrays,
  nested dicts, None handling)
- _is_safe_label / _is_safe_rel_type reject unsafe inputs
- ingest_records skips records missing the id field
- MERGE Cypher uses parameterized graph_id + id
- relationship mappings produce edges
- invalid labels and rel types raise ValueError before any I/O
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.structured_ingest_service import (
    RelationshipMapping,
    StructuredIngestService,
    _is_safe_label,
    _is_safe_rel_type,
    _record_to_properties,
)


class TestSafetyValidators:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "label,expected",
        [
            ("EvidenceRecord", True),
            ("Company", True),
            ("__Entity__", True),
            ("Foo_Bar123", True),
            ("Foo-Bar", False),  # hyphen not allowed
            ("Foo Bar", False),  # space not allowed
            ("Foo;DROP", False),  # injection attempt
            ("", False),
            (None, False),
            (123, False),
        ],
    )
    def test_is_safe_label(self, label, expected):
        assert _is_safe_label(label) is expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "rel,expected",
        [
            ("USES", True),
            ("PART_OF", True),
            ("ABOUT_2024", True),
            ("uses", False),  # lowercase not allowed
            ("USES;", False),  # semicolon not allowed
            ("", False),
            (None, False),
        ],
    )
    def test_is_safe_rel_type(self, rel, expected):
        assert _is_safe_rel_type(rel) is expected


class TestRecordToProperties:
    @pytest.mark.unit
    def test_primitives_pass_through(self):
        result = _record_to_properties(
            {"id": "x", "name": "Foo", "count": 42, "active": True, "price": 1.5},
            id_field="id",
        )
        assert result["name"] == "Foo"
        assert result["count"] == 42
        assert result["active"] is True
        assert result["price"] == 1.5

    @pytest.mark.unit
    def test_none_values_dropped(self):
        result = _record_to_properties(
            {"id": "x", "description": None, "tag": "active"}, id_field="id"
        )
        assert "description" not in result
        assert result["tag"] == "active"

    @pytest.mark.unit
    def test_list_of_primitives_kept_as_array(self):
        result = _record_to_properties(
            {"id": "x", "tags": ["a", "b", "c"]}, id_field="id"
        )
        assert result["tags"] == ["a", "b", "c"]

    @pytest.mark.unit
    def test_nested_dict_json_stringified(self):
        import json as _json

        result = _record_to_properties(
            {"id": "x", "address": {"city": "Utrecht"}}, id_field="id"
        )
        assert _json.loads(result["address"]) == {"city": "Utrecht"}

    @pytest.mark.unit
    def test_list_of_dicts_json_stringified(self):
        import json as _json

        result = _record_to_properties(
            {"id": "x", "contacts": [{"name": "A"}, {"name": "B"}]},
            id_field="id",
        )
        parsed = _json.loads(result["contacts"])
        assert parsed == [{"name": "A"}, {"name": "B"}]


class TestIngestRecordsValidation:
    @pytest.mark.unit
    async def test_rejects_invalid_label(self):
        svc = StructuredIngestService()
        with pytest.raises(ValueError, match="Invalid label"):
            await svc.ingest_records("g1", records=[{"id": "a"}], label="Bad-Label")

    @pytest.mark.unit
    async def test_rejects_invalid_rel_type(self):
        svc = StructuredIngestService()
        with pytest.raises(ValueError, match="Invalid rel_type"):
            await svc.ingest_records(
                "g1",
                records=[{"id": "a", "company_id": "c1"}],
                label="Record",
                relationships=[
                    RelationshipMapping(
                        from_field="company_id",
                        to_label="Company",
                        rel_type="lowercase_not_allowed",
                    )
                ],
            )

    @pytest.mark.unit
    async def test_rejects_invalid_relationship_to_label(self):
        svc = StructuredIngestService()
        with pytest.raises(ValueError, match="Invalid relationship to_label"):
            await svc.ingest_records(
                "g1",
                records=[],
                label="Record",
                relationships=[
                    RelationshipMapping(
                        from_field="x",
                        to_label="Bad-Label",
                        rel_type="REL",
                    )
                ],
            )


class TestIngestRecords:
    @pytest.mark.unit
    async def test_skips_record_missing_id(self):
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=[])
        with patch("app.services.structured_ingest_service.neo4j_client", mock_client):
            svc = StructuredIngestService()
            report = await svc.ingest_records(
                "g1",
                records=[{"name": "no_id"}, {"id": "x", "name": "ok"}],
                label="Record",
            )
        assert report.records_processed == 1
        assert report.skipped == 1
        assert "missing_or_invalid_id" in report.skip_reasons

    @pytest.mark.unit
    async def test_skips_non_dict_record(self):
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=[])
        with patch("app.services.structured_ingest_service.neo4j_client", mock_client):
            svc = StructuredIngestService()
            report = await svc.ingest_records(
                "g1", records=["not a dict", {"id": "a"}], label="Record"
            )
        assert report.skipped == 1
        assert "not_a_dict" in report.skip_reasons

    @pytest.mark.unit
    async def test_merge_passes_graph_id_and_id_as_params(self):
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=[])
        with patch("app.services.structured_ingest_service.neo4j_client", mock_client):
            svc = StructuredIngestService()
            await svc.ingest_records(
                "my-graph",
                records=[{"id": "rec-1", "name": "Test"}],
                label="Record",
            )
        first_call = mock_client.execute_query.await_args_list[0]
        params = first_call.args[1]
        assert params["id"] == "rec-1"
        assert params["gid"] == "my-graph"
        assert params["props"]["name"] == "Test"
        # id removed from props so MERGE key isn't overwritten
        assert "id" not in params["props"]

    @pytest.mark.unit
    async def test_relationship_mapping_creates_edge(self):
        """For a record with company_id, a target Company entity is
        MERGE'd and an ABOUT edge is created."""
        mock_client = MagicMock()
        # Calls: 1 merge_entity, 2 relationship (existence check + MERGE)
        mock_client.execute_query = AsyncMock(
            side_effect=[
                None,  # MERGE entity
                [{"n": 0}],  # existence check for target — doesn't exist
                None,  # MERGE relationship
            ]
        )
        with patch("app.services.structured_ingest_service.neo4j_client", mock_client):
            svc = StructuredIngestService()
            report = await svc.ingest_records(
                "g1",
                records=[{"id": "ev-1", "company_id": "comp-1"}],
                label="EvidenceRecord",
                relationships=[
                    RelationshipMapping(
                        from_field="company_id",
                        to_label="Company",
                        rel_type="ABOUT",
                    )
                ],
            )
        assert report.records_processed == 1
        assert report.relationships_created == 1
        assert report.related_entities_created == 1

    @pytest.mark.unit
    async def test_relationship_skipped_when_field_missing(self):
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=[])
        with patch("app.services.structured_ingest_service.neo4j_client", mock_client):
            svc = StructuredIngestService()
            report = await svc.ingest_records(
                "g1",
                records=[{"id": "ev-1"}],  # no company_id
                label="EvidenceRecord",
                relationships=[
                    RelationshipMapping(
                        from_field="company_id",
                        to_label="Company",
                        rel_type="ABOUT",
                    )
                ],
            )
        assert report.records_processed == 1
        assert report.relationships_created == 0

    @pytest.mark.unit
    async def test_list_of_target_ids_creates_multiple_edges(self):
        """When ``from_field`` carries a list, one edge per element."""
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(
            side_effect=[
                None,  # merge entity
                [{"n": 0}],  # target 1 existence
                None,  # target 1 merge rel
                [{"n": 1}],  # target 2 existence — existed before
                None,  # target 2 merge rel
            ]
        )
        with patch("app.services.structured_ingest_service.neo4j_client", mock_client):
            svc = StructuredIngestService()
            report = await svc.ingest_records(
                "g1",
                records=[{"id": "ev-1", "tags": ["t1", "t2"]}],
                label="EvidenceRecord",
                relationships=[
                    RelationshipMapping(
                        from_field="tags",
                        to_label="Tag",
                        rel_type="TAGGED_WITH",
                    )
                ],
            )
        # Two edges created, one newly-created target entity (t1; t2
        # already existed)
        assert report.relationships_created == 2
        assert report.related_entities_created == 1

    @pytest.mark.unit
    async def test_empty_record_list_is_no_op(self):
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=[])
        with patch("app.services.structured_ingest_service.neo4j_client", mock_client):
            svc = StructuredIngestService()
            report = await svc.ingest_records("g1", records=[], label="X")
        assert report.records_processed == 0
        assert report.skipped == 0
        mock_client.execute_query.assert_not_called()
