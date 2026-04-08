"""
Unit tests for graph_schemas.py Pydantic models.

Tests banned-property enforcement on EntityNodeProperties, RelationshipProperties
validation, LLMExtractionOutput cross-reference validation, and MigrationOrphanLog.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from app.schemas.graph_schemas import (
    BANNED_NODE_PROPERTIES,
    EntityNodeProperties,
    ExtractedEntity,
    ExtractedRelationship,
    LLMExtractionOutput,
    MigrationOrphanLog,
    RelationshipProperties,
)

# ---------------------------------------------------------------------------
# Tests: BANNED_NODE_PROPERTIES constant
# ---------------------------------------------------------------------------


class TestBannedNodePropertiesConstant:
    @pytest.mark.unit
    def test_contains_job_title(self):
        assert "job_title" in BANNED_NODE_PROPERTIES

    @pytest.mark.unit
    def test_contains_position(self):
        assert "position" in BANNED_NODE_PROPERTIES

    @pytest.mark.unit
    def test_contains_role(self):
        assert "role" in BANNED_NODE_PROPERTIES

    @pytest.mark.unit
    def test_contains_seniority(self):
        assert "seniority" in BANNED_NODE_PROPERTIES

    @pytest.mark.unit
    def test_contains_proficiency(self):
        assert "proficiency" in BANNED_NODE_PROPERTIES

    @pytest.mark.unit
    def test_contains_ownership_pct(self):
        assert "ownership_pct" in BANNED_NODE_PROPERTIES

    @pytest.mark.unit
    def test_is_a_set(self):
        assert isinstance(BANNED_NODE_PROPERTIES, set)


# ---------------------------------------------------------------------------
# Tests: EntityNodeProperties — banned property rejection
# ---------------------------------------------------------------------------


class TestEntityNodeProperties:
    @pytest.mark.unit
    def test_valid_properties_pass(self):
        props = EntityNodeProperties(name="Alice Chen", description="A person")
        assert props.name == "Alice Chen"

    @pytest.mark.unit
    def test_extra_non_banned_props_pass(self):
        props = EntityNodeProperties(
            name="Acme Corp", extra={"industry": "Software", "founded_year": 2010}
        )
        assert props.extra["industry"] == "Software"

    @pytest.mark.unit
    def test_job_title_in_extra_raises(self):
        with pytest.raises(ValidationError, match="relational context"):
            EntityNodeProperties(name="Alice", extra={"job_title": "CEO"})

    @pytest.mark.unit
    def test_position_in_extra_raises(self):
        with pytest.raises(ValidationError, match="relational context"):
            EntityNodeProperties(name="Alice", extra={"position": "Director"})

    @pytest.mark.unit
    def test_role_in_extra_raises(self):
        with pytest.raises(ValidationError, match="relational context"):
            EntityNodeProperties(name="Alice", extra={"role": "Lead"})

    @pytest.mark.unit
    def test_multiple_banned_props_raises(self):
        with pytest.raises(ValidationError):
            EntityNodeProperties(
                name="Alice", extra={"job_title": "CEO", "seniority": "Senior"}
            )

    @pytest.mark.unit
    def test_empty_extra_passes(self):
        props = EntityNodeProperties(name="Alice", extra={})
        assert props.extra == {}

    @pytest.mark.unit
    def test_none_extra_passes(self):
        props = EntityNodeProperties(name="Alice", extra=None)
        assert props.extra is None

    @pytest.mark.unit
    def test_all_banned_props_rejected(self):
        """Every property in BANNED_NODE_PROPERTIES must raise when placed on a node."""
        for prop in BANNED_NODE_PROPERTIES:
            with pytest.raises(ValidationError, match="relational context"):
                EntityNodeProperties(name="Test", extra={prop: "some_value"})


# ---------------------------------------------------------------------------
# Tests: RelationshipProperties
# ---------------------------------------------------------------------------


class TestRelationshipProperties:
    @pytest.mark.unit
    def test_valid_relationship_properties(self):
        props = RelationshipProperties(source_chunk_id="chunk-abc", confidence=0.95)
        assert props.source_chunk_id == "chunk-abc"
        assert props.confidence == 0.95

    @pytest.mark.unit
    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValidationError):
            RelationshipProperties(source_chunk_id="chunk", confidence=-0.1)

    @pytest.mark.unit
    def test_confidence_above_one_raises(self):
        with pytest.raises(ValidationError):
            RelationshipProperties(source_chunk_id="chunk", confidence=1.1)

    @pytest.mark.unit
    def test_confidence_at_zero_passes(self):
        props = RelationshipProperties(source_chunk_id="chunk", confidence=0.0)
        assert props.confidence == 0.0

    @pytest.mark.unit
    def test_confidence_at_one_passes(self):
        props = RelationshipProperties(source_chunk_id="chunk", confidence=1.0)
        assert props.confidence == 1.0

    @pytest.mark.unit
    def test_optional_fields_default_to_none(self):
        props = RelationshipProperties(source_chunk_id="chunk", confidence=0.8)
        assert props.valid_from is None
        assert props.valid_to is None

    @pytest.mark.unit
    def test_ingested_at_defaults_to_now(self):
        props = RelationshipProperties(source_chunk_id="chunk", confidence=0.8)
        assert isinstance(props.ingested_at, datetime)

    @pytest.mark.unit
    def test_relationship_can_carry_positional_property_as_extra(self):
        """Relationship properties CAN carry job_title, position, etc. — only nodes cannot."""
        props = RelationshipProperties(
            source_chunk_id="chunk",
            confidence=0.9,
            extra={"position": "CTO", "start_date": "2021-03"},
        )
        assert props.extra["position"] == "CTO"

    @pytest.mark.unit
    def test_source_chunk_id_required(self):
        with pytest.raises(ValidationError):
            RelationshipProperties(confidence=0.9)


# ---------------------------------------------------------------------------
# Tests: ExtractedEntity
# ---------------------------------------------------------------------------


class TestExtractedEntity:
    @pytest.mark.unit
    def test_valid_entity(self):
        entity = ExtractedEntity(
            id="alice-001",
            label="Person",
            properties=EntityNodeProperties(name="Alice Chen"),
        )
        assert entity.id == "alice-001"
        assert entity.label == "Person"

    @pytest.mark.unit
    def test_entity_with_banned_property_raises(self):
        with pytest.raises(ValidationError):
            ExtractedEntity(
                id="alice-001",
                label="Person",
                properties=EntityNodeProperties(
                    name="Alice", extra={"job_title": "CEO"}
                ),
            )


# ---------------------------------------------------------------------------
# Tests: ExtractedRelationship
# ---------------------------------------------------------------------------


class TestExtractedRelationship:
    @pytest.mark.unit
    def test_valid_relationship(self):
        rel = ExtractedRelationship(
            start_node_id="alice-001",
            end_node_id="acme-001",
            type="WORKS_FOR",
            properties=RelationshipProperties(
                source_chunk_id="chunk-xyz",
                confidence=0.93,
                extra={"position": "CTO", "start_date": "March 2021"},
            ),
        )
        assert rel.type == "WORKS_FOR"
        assert rel.properties.extra["position"] == "CTO"

    @pytest.mark.unit
    def test_relationship_type_must_be_uppercase_snake_case(self):
        with pytest.raises(ValidationError):
            ExtractedRelationship(
                start_node_id="a",
                end_node_id="b",
                type="works_for",  # lowercase — invalid
                properties=RelationshipProperties(source_chunk_id="c", confidence=0.9),
            )

    @pytest.mark.unit
    def test_relationship_type_with_numbers_allowed(self):
        rel = ExtractedRelationship(
            start_node_id="a",
            end_node_id="b",
            type="MEMBER_OF_V2",
            properties=RelationshipProperties(source_chunk_id="c", confidence=0.9),
        )
        assert rel.type == "MEMBER_OF_V2"


# ---------------------------------------------------------------------------
# Tests: LLMExtractionOutput — cross-reference validation
# ---------------------------------------------------------------------------


class TestLLMExtractionOutput:
    def _make_entity(
        self, entity_id: str, label: str = "Person", name: str = "Alice"
    ) -> ExtractedEntity:
        return ExtractedEntity(
            id=entity_id, label=label, properties=EntityNodeProperties(name=name)
        )

    def _make_rel(
        self, start: str, end: str, rel_type: str = "WORKS_FOR"
    ) -> ExtractedRelationship:
        return ExtractedRelationship(
            start_node_id=start,
            end_node_id=end,
            type=rel_type,
            properties=RelationshipProperties(source_chunk_id="chunk", confidence=0.9),
        )

    @pytest.mark.unit
    def test_valid_extraction_output(self):
        output = LLMExtractionOutput(
            nodes=[
                self._make_entity("alice", "Person", "Alice"),
                self._make_entity("acme", "Company", "Acme Corp"),
            ],
            relationships=[self._make_rel("alice", "acme")],
        )
        assert len(output.nodes) == 2
        assert len(output.relationships) == 1

    @pytest.mark.unit
    def test_invalid_start_node_reference_raises(self):
        with pytest.raises(ValidationError, match="not in extracted nodes"):
            LLMExtractionOutput(
                nodes=[self._make_entity("acme", "Company", "Acme")],
                relationships=[self._make_rel("alice", "acme")],  # alice not in nodes
            )

    @pytest.mark.unit
    def test_invalid_end_node_reference_raises(self):
        with pytest.raises(ValidationError, match="not in extracted nodes"):
            LLMExtractionOutput(
                nodes=[self._make_entity("alice", "Person", "Alice")],
                relationships=[self._make_rel("alice", "acme")],  # acme not in nodes
            )

    @pytest.mark.unit
    def test_empty_nodes_and_relationships_passes(self):
        output = LLMExtractionOutput(nodes=[], relationships=[])
        assert output.nodes == []

    @pytest.mark.unit
    def test_node_with_banned_property_fails_at_top_level(self):
        with pytest.raises(ValidationError):
            LLMExtractionOutput(
                nodes=[
                    ExtractedEntity(
                        id="alice",
                        label="Person",
                        properties=EntityNodeProperties(
                            name="Alice", extra={"job_title": "CEO"}
                        ),
                    )
                ],
                relationships=[],
            )


# ---------------------------------------------------------------------------
# Tests: MigrationOrphanLog
# ---------------------------------------------------------------------------


class TestMigrationOrphanLog:
    @pytest.mark.unit
    def test_valid_orphan_log(self):
        log = MigrationOrphanLog(
            graph_id="graph-123",
            entity_id="e-001",
            entity_name="Alice Chen",
            entity_type="Person",
            orphaned_properties={"job_title": "CEO"},
            source_chunk_ids=["chunk-abc"],
        )
        assert log.status == "pending"
        assert isinstance(log.detected_at, datetime)

    @pytest.mark.unit
    def test_status_defaults_to_pending(self):
        log = MigrationOrphanLog(
            graph_id="g",
            entity_id="e",
            entity_name="N",
            entity_type="Person",
            orphaned_properties={},
            source_chunk_ids=[],
        )
        assert log.status == "pending"

    @pytest.mark.unit
    def test_custom_status_accepted(self):
        log = MigrationOrphanLog(
            graph_id="g",
            entity_id="e",
            entity_name="N",
            entity_type="Person",
            orphaned_properties={},
            source_chunk_ids=[],
            status="re_extracted",
        )
        assert log.status == "re_extracted"
