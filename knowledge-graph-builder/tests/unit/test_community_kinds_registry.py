"""Unit tests for the community-kind registry (STORY-4a).

The registry is the source of truth for which community kinds the platform
can read and detect. These tests pin the contract: keys present, helpers
behave deterministically, unknown kinds raise the expected error.
"""

import pytest

from app.schemas.community_kinds import (
    COMMUNITY_KINDS,
    CommunityKindSpec,
    UnknownCommunityKindError,
    all_kinds,
    get_kind,
    kind_for_community_label,
)


class TestRegistryEntries:
    @pytest.mark.unit
    def test_entity_kind_is_registered_with_expected_shape(self):
        spec = COMMUNITY_KINDS["entity"]
        assert spec.community_label == "__Community__"
        assert spec.member_label == "__Entity__"
        assert spec.member_rel == "IN_COMMUNITY"
        assert spec.id_property == "id"
        assert spec.size_property == "entity_count"
        assert spec.hierarchical is True
        assert spec.member_rel_has_level is True
        assert spec.detector_task_name is not None
        # STORY-4c — entity uses the existing Celery-detector convention
        assert spec.embedding_property == "embedding"
        assert spec.index_name == "community_embeddings_entity"

    @pytest.mark.unit
    def test_chunk_kind_is_registered_with_expected_shape(self):
        spec = COMMUNITY_KINDS["chunk"]
        assert spec.community_label == "ChunkCommunity"
        assert spec.member_label == "Chunk"
        assert spec.member_rel == "IN_CHUNK_COMMUNITY"
        assert spec.id_property == "community_id"
        assert spec.size_property == "size"
        assert spec.hierarchical is False
        assert spec.member_rel_has_level is False
        # Read-only kind today
        assert spec.detector_task_name is None
        # STORY-4c — chunk uses the summary_* namespace for embeddings
        assert spec.embedding_property == "summary_embedding"
        assert spec.index_name == "community_embeddings_chunk"

    @pytest.mark.unit
    def test_index_names_are_unique_per_kind(self):
        """Each kind must have a unique vector-index name so the indexes
        don't collide in Neo4j."""
        names = [spec.index_name for spec in COMMUNITY_KINDS.values()]
        assert len(names) == len(set(names)), (
            f"Index names collide across kinds: {names}"
        )

    @pytest.mark.unit
    def test_registry_keys_match_kind_field(self):
        """The dict key must equal the spec's own ``kind`` field — otherwise
        ``all_kinds()`` consumers will get inconsistent identifiers."""
        for key, spec in COMMUNITY_KINDS.items():
            assert key == spec.kind, (
                f"Registry key {key!r} doesn't match spec.kind {spec.kind!r}"
            )

    @pytest.mark.unit
    def test_specs_are_frozen(self):
        """Mutation of a registered spec is rejected so callers can't
        accidentally rewrite the registry at runtime."""
        spec = COMMUNITY_KINDS["entity"]
        with pytest.raises((AttributeError, Exception)):
            spec.community_label = "Mutated"  # type: ignore[misc]


class TestGetKind:
    @pytest.mark.unit
    def test_returns_registered_spec(self):
        spec = get_kind("entity")
        assert isinstance(spec, CommunityKindSpec)
        assert spec.kind == "entity"

    @pytest.mark.unit
    def test_raises_unknown_community_kind_error(self):
        with pytest.raises(UnknownCommunityKindError) as exc_info:
            get_kind("bogus")
        # The error must enumerate valid kinds so callers can correct the bug
        assert "bogus" in str(exc_info.value)
        assert "entity" in str(exc_info.value)
        assert "chunk" in str(exc_info.value)
        # ValueError lineage lets endpoints catch with a single except
        assert isinstance(exc_info.value, ValueError)

    @pytest.mark.unit
    def test_error_exposes_kind_and_valid_kinds(self):
        try:
            get_kind("nope")
        except UnknownCommunityKindError as exc:
            assert exc.kind == "nope"
            assert "entity" in exc.valid_kinds
            assert "chunk" in exc.valid_kinds


class TestAllKinds:
    @pytest.mark.unit
    def test_returns_every_registered_spec(self):
        kinds = all_kinds()
        names = [spec.kind for spec in kinds]
        assert "entity" in names
        assert "chunk" in names

    @pytest.mark.unit
    def test_preserves_insertion_order(self):
        """Insertion order matters for endpoints that probe kinds in order
        (e.g., get_community_detail without ?kind=). Entity must come first
        so the common path stays a single round trip."""
        kinds = all_kinds()
        first_two = [spec.kind for spec in kinds[:2]]
        assert first_two == ["entity", "chunk"]


class TestKindForCommunityLabel:
    @pytest.mark.unit
    def test_entity_label_resolves_to_entity_kind(self):
        spec = kind_for_community_label("__Community__")
        assert spec is not None
        assert spec.kind == "entity"

    @pytest.mark.unit
    def test_chunk_label_resolves_to_chunk_kind(self):
        spec = kind_for_community_label("ChunkCommunity")
        assert spec is not None
        assert spec.kind == "chunk"

    @pytest.mark.unit
    def test_unknown_label_returns_none(self):
        assert kind_for_community_label("Foo") is None
