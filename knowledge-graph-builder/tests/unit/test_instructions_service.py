"""
Unit tests for InstructionsResolver, InstructionsCompiler, and related schemas.

Tests:
- GraphInstructions / IngestionOverrides model validation
- InstructionsResolver merge rules
- InstructionsCompiler prompt block generation
- IngestDataRequest.resolved_overrides() backwards compat
"""

import warnings
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.schemas.graph_schemas import (
    EntityTypeRule,
    ExtractionDensity,
    GraphInstructions,
    IngestDataRequest,
    IngestionOverrides,
    RelationshipRule,
)
from app.services.instructions_service import (
    InstructionsCompiler,
    InstructionsResolver,
    ResolvedInstructions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_resolver(instructions_json=None) -> InstructionsResolver:
    resolver = InstructionsResolver()
    resolver._load_from_neo4j = AsyncMock(
        return_value=(
            GraphInstructions(**instructions_json) if instructions_json else None
        )
    )
    return resolver


# ---------------------------------------------------------------------------
# Tests: GraphInstructions model validation
# ---------------------------------------------------------------------------


class TestGraphInstructions:
    @pytest.mark.unit
    def test_all_defaults_valid(self):
        gi = GraphInstructions()
        assert gi.extraction_density == ExtractionDensity.BALANCED
        assert gi.language == "en"
        assert gi.entity_types is None

    @pytest.mark.unit
    def test_custom_prompt_suffix_max_length(self):
        """custom_prompt_suffix must not exceed 2000 chars."""
        with pytest.raises(ValidationError):
            GraphInstructions(custom_prompt_suffix="x" * 2001)

    @pytest.mark.unit
    def test_custom_prompt_suffix_at_limit_passes(self):
        gi = GraphInstructions(custom_prompt_suffix="x" * 2000)
        assert len(gi.custom_prompt_suffix) == 2000

    @pytest.mark.unit
    def test_entity_type_rule_structure(self):
        gi = GraphInstructions(
            entity_types=[
                EntityTypeRule(
                    name="Person", description="A human", examples=["Alice", "Bob"]
                )
            ]
        )
        assert gi.entity_types[0].name == "Person"
        assert gi.entity_types[0].examples == ["Alice", "Bob"]

    @pytest.mark.unit
    def test_extraction_density_sparse(self):
        gi = GraphInstructions(extraction_density="sparse")
        assert gi.extraction_density == ExtractionDensity.SPARSE

    @pytest.mark.unit
    def test_extraction_density_dense(self):
        gi = GraphInstructions(extraction_density="dense")
        assert gi.extraction_density == ExtractionDensity.DENSE


# ---------------------------------------------------------------------------
# Tests: IngestionOverrides model validation
# ---------------------------------------------------------------------------


class TestIngestionOverrides:
    @pytest.mark.unit
    def test_all_none_defaults(self):
        ov = IngestionOverrides()
        assert ov.additional_focus is None
        assert ov.override_density is None
        assert ov.extra_entity_types is None
        assert ov.schema_evolution_hint is None

    @pytest.mark.unit
    def test_override_density_accepted(self):
        ov = IngestionOverrides(override_density="dense")
        assert ov.override_density == ExtractionDensity.DENSE

    @pytest.mark.unit
    def test_extra_entity_types_accepted(self):
        ov = IngestionOverrides(extra_entity_types=[EntityTypeRule(name="Drug")])
        assert ov.extra_entity_types[0].name == "Drug"


# ---------------------------------------------------------------------------
# Tests: IngestDataRequest.resolved_overrides() backwards compat
# ---------------------------------------------------------------------------


class TestIngestDataRequestResolvedOverrides:
    @pytest.mark.unit
    def test_returns_none_when_neither_set(self):
        req = IngestDataRequest(content="hello world test content")
        assert req.resolved_overrides() is None

    @pytest.mark.unit
    def test_returns_overrides_when_set(self):
        req = IngestDataRequest(
            content="hello world test content",
            overrides=IngestionOverrides(additional_focus="focus on dates"),
        )
        result = req.resolved_overrides()
        assert result is not None
        assert result.additional_focus == "focus on dates"

    @pytest.mark.unit
    def test_wraps_deprecated_instructions_as_additional_focus(self):
        req = IngestDataRequest(
            content="hello world test content",
            instructions="focus on legal clauses",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = req.resolved_overrides()

        assert result is not None
        assert result.additional_focus == "focus on legal clauses"
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    @pytest.mark.unit
    def test_overrides_takes_precedence_over_instructions(self):
        """When both are set, overrides wins — no wrapping."""
        req = IngestDataRequest(
            content="hello world test content",
            overrides=IngestionOverrides(additional_focus="structured"),
            instructions="deprecated hint",
        )
        result = req.resolved_overrides()
        assert result.additional_focus == "structured"


# ---------------------------------------------------------------------------
# Tests: InstructionsResolver — merge rules
# ---------------------------------------------------------------------------


class TestInstructionsResolver:
    @pytest.mark.unit
    async def test_returns_defaults_when_no_instructions_stored(self):
        resolver = _make_resolver(instructions_json=None)
        resolved = await resolver.resolve("graph-1")
        assert resolved.extraction_density == ExtractionDensity.BALANCED
        assert resolved.entity_types is None

    @pytest.mark.unit
    async def test_loads_graph_instructions_when_present(self):
        resolver = _make_resolver(
            {"domain": "HR org chart", "extraction_density": "sparse"}
        )
        resolved = await resolver.resolve("graph-1")
        assert resolved.domain == "HR org chart"
        assert resolved.extraction_density == ExtractionDensity.SPARSE

    @pytest.mark.unit
    async def test_override_density_replaces_graph_density(self):
        """sparse graph + dense override → resolved dense."""
        resolver = _make_resolver({"extraction_density": "sparse"})
        overrides = IngestionOverrides(override_density=ExtractionDensity.DENSE)
        resolved = await resolver.resolve("graph-1", overrides)
        assert resolved.extraction_density == ExtractionDensity.DENSE

    @pytest.mark.unit
    async def test_additional_focus_appended_to_focus_areas(self):
        resolver = _make_resolver({"focus_areas": ["employment history"]})
        overrides = IngestionOverrides(additional_focus="board memberships")
        resolved = await resolver.resolve("graph-1", overrides)
        assert "employment history" in resolved.focus_areas
        assert "board memberships" in resolved.focus_areas

    @pytest.mark.unit
    async def test_additional_focus_creates_list_when_graph_has_none(self):
        resolver = _make_resolver({})  # no focus_areas
        overrides = IngestionOverrides(additional_focus="key dates")
        resolved = await resolver.resolve("graph-1", overrides)
        assert resolved.focus_areas == ["key dates"]

    @pytest.mark.unit
    async def test_extra_entity_types_appended(self):
        resolver = _make_resolver({"entity_types": [{"name": "Person"}]})
        overrides = IngestionOverrides(extra_entity_types=[EntityTypeRule(name="Drug")])
        resolved = await resolver.resolve("graph-1", overrides)
        assert len(resolved.entity_types) == 2
        names = [et.name for et in resolved.entity_types]
        assert "Person" in names
        assert "Drug" in names

    @pytest.mark.unit
    async def test_schema_evolution_hint_appended_to_prompt_suffix(self):
        resolver = _make_resolver({"custom_prompt_suffix": "Existing suffix."})
        overrides = IngestionOverrides(schema_evolution_hint="New hint.")
        resolved = await resolver.resolve("graph-1", overrides)
        assert "Existing suffix." in resolved.custom_prompt_suffix
        assert "New hint." in resolved.custom_prompt_suffix

    @pytest.mark.unit
    async def test_no_overrides_returns_graph_instructions_unchanged(self):
        resolver = _make_resolver({"domain": "Legal contracts", "language": "fr"})
        resolved = await resolver.resolve("graph-1", overrides=None)
        assert resolved.domain == "Legal contracts"
        assert resolved.language == "fr"

    @pytest.mark.unit
    async def test_graph_id_passed_to_neo4j_loader(self):
        resolver = InstructionsResolver()
        resolver._load_from_neo4j = AsyncMock(return_value=None)
        await resolver.resolve("specific-graph-id")
        resolver._load_from_neo4j.assert_awaited_once_with("specific-graph-id")


# ---------------------------------------------------------------------------
# Tests: InstructionsCompiler — prompt block generation
# ---------------------------------------------------------------------------


class TestInstructionsCompiler:
    def _compile(self, **kwargs) -> str:
        resolved = ResolvedInstructions(**kwargs)
        return InstructionsCompiler().to_prompt(resolved)

    @pytest.mark.unit
    def test_default_instructions_omits_domain_entity_and_focus_blocks(self):
        """All-default resolved instructions only produce the rules block."""
        result = self._compile()
        assert "Extraction Context" not in result
        assert "Entity Types" not in result
        assert "Focus Areas" not in result
        assert "Extraction Rules" in result

    @pytest.mark.unit
    def test_domain_block_included_when_set(self):
        result = self._compile(domain="Legal contracts")
        assert "Legal contracts" in result
        assert "Extraction Context" in result

    @pytest.mark.unit
    def test_domain_block_absent_when_none(self):
        result = self._compile(domain=None)
        assert "Extraction Context" not in result

    @pytest.mark.unit
    def test_entity_types_block_in_ontology_mode(self):
        result = self._compile(
            entity_types=[EntityTypeRule(name="Person", description="A human")]
        )
        assert "Entity Types" in result
        assert "Person" in result
        assert "Do NOT create entity types not listed above." in result

    @pytest.mark.unit
    def test_entity_types_block_absent_in_free_form_mode(self):
        result = self._compile(entity_types=None)
        assert "Entity Types" not in result

    @pytest.mark.unit
    def test_relationship_types_block_included(self):
        result = self._compile(
            relationship_types=[
                RelationshipRule(
                    name="WORKS_FOR",
                    source_type="Person",
                    target_type="Company",
                    store_as_edge_property=["job_title", "start_date"],
                )
            ]
        )
        assert "WORKS_FOR" in result
        assert "Relationship Types" in result
        assert "job_title" in result

    @pytest.mark.unit
    def test_rules_block_includes_density(self):
        result = self._compile(extraction_density=ExtractionDensity.SPARSE)
        assert "sparse" in result

    @pytest.mark.unit
    def test_rules_block_includes_edge_property_fields(self):
        result = self._compile(edge_property_fields=["job_title", "position"])
        assert "job_title" in result
        assert "MUST be stored on relationships" in result

    @pytest.mark.unit
    def test_focus_areas_block_included_when_set(self):
        result = self._compile(focus_areas=["employment history", "board roles"])
        assert "Focus Areas" in result
        assert "employment history" in result

    @pytest.mark.unit
    def test_focus_areas_block_absent_when_empty(self):
        result = self._compile(focus_areas=[])
        assert "Focus Areas" not in result

    @pytest.mark.unit
    def test_custom_prompt_suffix_appended(self):
        result = self._compile(custom_prompt_suffix="Only extract named individuals.")
        assert "Only extract named individuals." in result

    @pytest.mark.unit
    def test_language_included_in_rules(self):
        result = self._compile(language="fr")
        assert "fr" in result
