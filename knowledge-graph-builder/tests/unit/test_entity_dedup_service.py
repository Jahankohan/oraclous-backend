"""Unit tests for EntityDeduplicationService (STORY-6).

Verifies:
- canonical_name normalization (suffix stripping, list-coercion, edge cases)
- pass 1 writes canonical_name on entities missing one
- pass 2 picks the deterministic keeper (lowest elementId)
- pass 3 emits the right vector-index Cypher and skips same-canonical pairs
- pass 4 consolidates parallel relationships and skips structural types
- dry_run never calls write APIs
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.entity_dedup_service import (
    EntityDeduplicationService,
)


class TestComputeCanonicalName:
    """Pure-function pass — no Neo4j involvement, easy to exercise hard."""

    @pytest.mark.unit
    def test_lowercases(self):
        assert EntityDeduplicationService.compute_canonical_name("Eurail") == "eurail"

    @pytest.mark.unit
    def test_strips_bv(self):
        assert (
            EntityDeduplicationService.compute_canonical_name("Eurail B.V.") == "eurail"
        )

    @pytest.mark.unit
    def test_strips_group(self):
        assert (
            EntityDeduplicationService.compute_canonical_name("Eurail Group")
            == "eurail"
        )

    @pytest.mark.unit
    def test_strips_dotcom(self):
        assert (
            EntityDeduplicationService.compute_canonical_name("Eurail.com") == "eurail"
        )

    @pytest.mark.unit
    def test_strips_repeated_suffixes(self):
        """``Eurail Group B.V.`` should strip both suffixes."""
        assert (
            EntityDeduplicationService.compute_canonical_name("Eurail Group B.V.")
            == "eurail"
        )

    @pytest.mark.unit
    def test_strips_gmbh(self):
        assert (
            EntityDeduplicationService.compute_canonical_name("Munich GmbH") == "munich"
        )

    @pytest.mark.unit
    def test_collapses_whitespace(self):
        assert (
            EntityDeduplicationService.compute_canonical_name("Eurail   Group  B.V.")
            == "eurail"
        )

    @pytest.mark.unit
    def test_returns_none_for_none(self):
        assert EntityDeduplicationService.compute_canonical_name(None) is None

    @pytest.mark.unit
    def test_returns_none_for_empty_string(self):
        assert EntityDeduplicationService.compute_canonical_name("") is None
        assert EntityDeduplicationService.compute_canonical_name("   ") is None

    @pytest.mark.unit
    def test_returns_none_for_non_string(self):
        assert EntityDeduplicationService.compute_canonical_name(123) is None
        assert EntityDeduplicationService.compute_canonical_name({"a": 1}) is None

    @pytest.mark.unit
    def test_string_array_picks_first_valid_string(self):
        """Some entities have ``name`` as a list because of a separate
        extractor bug — the service must tolerate that, not crash."""
        result = EntityDeduplicationService.compute_canonical_name(
            ["Eurail B.V.", "Eurail Group"]
        )
        # Picks the first list entry and normalizes it
        assert result == "eurail"

    @pytest.mark.unit
    def test_string_array_with_no_valid_string(self):
        assert EntityDeduplicationService.compute_canonical_name([None, ""]) is None

    @pytest.mark.unit
    def test_preserves_internal_substring_that_matches_suffix(self):
        """``Group`` is a suffix but ``GroupMe`` is not — strip should
        only operate at the trailing whitespace boundary."""
        # "groupme inc" → "groupme" (strips Inc, but not interior "group")
        assert (
            EntityDeduplicationService.compute_canonical_name("GroupMe Inc")
            == "groupme"
        )


class TestCanonicalPass:
    @pytest.mark.unit
    async def test_writes_canonical_on_each_missing(self):
        mock_client = MagicMock()
        rows = [
            {"eid": "elem-1", "name": "Eurail B.V."},
            {"eid": "elem-2", "name": "Eurail Group"},
        ]
        # First call returns the listing; subsequent two calls are writes.
        mock_client.execute_query = AsyncMock(side_effect=[rows, None, None])
        with patch("app.services.entity_dedup_service.neo4j_client", mock_client):
            svc = EntityDeduplicationService()
            added, skipped = await svc._pass_canonical("g1", dry_run=False)
        assert added == 2
        assert skipped == 0
        # Each write call carried canonical_name == "eurail"
        write1 = mock_client.execute_query.await_args_list[1].args[1]
        assert write1["cn"] == "eurail"

    @pytest.mark.unit
    async def test_skips_bad_names(self):
        mock_client = MagicMock()
        rows = [
            {"eid": "elem-1", "name": None},
            {"eid": "elem-2", "name": ""},
            {"eid": "elem-3", "name": "valid"},
        ]
        mock_client.execute_query = AsyncMock(side_effect=[rows, None])
        with patch("app.services.entity_dedup_service.neo4j_client", mock_client):
            svc = EntityDeduplicationService()
            added, skipped = await svc._pass_canonical("g1", dry_run=False)
        assert added == 1
        assert skipped == 2

    @pytest.mark.unit
    async def test_dry_run_makes_no_write_calls(self):
        mock_client = MagicMock()
        rows = [{"eid": "elem-1", "name": "Eurail B.V."}]
        mock_client.execute_query = AsyncMock(return_value=rows)
        with patch("app.services.entity_dedup_service.neo4j_client", mock_client):
            svc = EntityDeduplicationService()
            added, _ = await svc._pass_canonical("g1", dry_run=True)
        assert added == 1
        # Only one call — the SELECT — no write
        assert mock_client.execute_query.await_count == 1


class TestMergeCanonicalPass:
    @pytest.mark.unit
    async def test_no_duplicate_groups_means_zero_merges(self):
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=[])
        with patch("app.services.entity_dedup_service.neo4j_client", mock_client):
            svc = EntityDeduplicationService()
            merged = await svc._pass_merge_canonical("g1", dry_run=False)
        assert merged == 0

    @pytest.mark.unit
    async def test_picks_lowest_elementid_as_keeper(self):
        """For a group of duplicates, the lexicographically lowest
        elementId becomes the keeper — deterministic across re-runs."""
        groups = [{"cn": "eurail", "eids": ["z-elem", "a-elem", "m-elem"]}]
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=groups)
        with patch("app.services.entity_dedup_service.neo4j_client", mock_client):
            svc = EntityDeduplicationService()
            svc._consolidate_into_keeper = AsyncMock(return_value=2)
            await svc._pass_merge_canonical("g1", dry_run=False)
        keeper_arg = svc._consolidate_into_keeper.await_args.args[1]
        losers_arg = svc._consolidate_into_keeper.await_args.args[2]
        assert keeper_arg == "a-elem"
        assert sorted(losers_arg) == ["m-elem", "z-elem"]

    @pytest.mark.unit
    async def test_dry_run_doesnt_consolidate(self):
        groups = [{"cn": "eurail", "eids": ["a", "b", "c"]}]
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=groups)
        with patch("app.services.entity_dedup_service.neo4j_client", mock_client):
            svc = EntityDeduplicationService()
            svc._consolidate_into_keeper = AsyncMock(return_value=2)
            merged = await svc._pass_merge_canonical("g1", dry_run=True)
        # Reports 2 merges (would happen) but doesn't actually call consolidate
        assert merged == 2
        svc._consolidate_into_keeper.assert_not_called()


class TestEmbeddingPass:
    @pytest.mark.unit
    async def test_cypher_uses_vector_index_and_dim_filter(self):
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=[])
        with patch("app.services.entity_dedup_service.neo4j_client", mock_client):
            svc = EntityDeduplicationService()
            await svc._pass_embedding("g1", threshold=0.92, dry_run=False)
        cypher = mock_client.execute_query.call_args[0][0]
        params = mock_client.execute_query.call_args[0][1]
        assert "entity_embeddings" in cypher
        assert "$expected_dim" in cypher
        assert "elementId(a) < elementId(b)" in cypher
        assert params["threshold"] == 0.92
        assert params["expected_dim"] == 3072

    @pytest.mark.unit
    async def test_skips_pairs_with_same_canonical_name(self):
        """The Cypher must exclude pairs that already share a
        canonical_name — pass 2 handles those."""
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=[])
        with patch("app.services.entity_dedup_service.neo4j_client", mock_client):
            svc = EntityDeduplicationService()
            await svc._pass_embedding("g1", threshold=0.92, dry_run=False)
        cypher = mock_client.execute_query.call_args[0][0]
        assert "a.canonical_name <> b.canonical_name" in cypher

    @pytest.mark.unit
    async def test_dry_run_doesnt_consolidate(self):
        pairs = [{"keeper_eid": "a", "loser_eid": "b", "score": 0.95}]
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=pairs)
        with patch("app.services.entity_dedup_service.neo4j_client", mock_client):
            svc = EntityDeduplicationService()
            svc._consolidate_into_keeper = AsyncMock(return_value=1)
            merged = await svc._pass_embedding("g1", threshold=0.92, dry_run=True)
        assert merged == 1
        svc._consolidate_into_keeper.assert_not_called()


class TestRelationshipPass:
    @pytest.mark.unit
    async def test_excluded_types_in_cypher(self):
        """SIMILAR_TO and friends must be filtered OUT of the pass."""
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=[])
        with patch("app.services.entity_dedup_service.neo4j_client", mock_client):
            svc = EntityDeduplicationService()
            await svc._pass_relationships("g1", dry_run=False)
        cypher = mock_client.execute_query.call_args[0][0]
        # All structural rel types must appear in the NOT-IN clause
        assert "'SIMILAR_TO'" in cypher
        assert "'IN_COMMUNITY'" in cypher
        assert "'FROM_CHUNK'" in cypher
        assert "'PARENT_COMMUNITY'" in cypher
        assert "'IN_CHUNK_COMMUNITY'" in cypher
        # Filter shape: NOT type(r) IN [...]
        assert "NOT type(r) IN" in cypher

    @pytest.mark.unit
    async def test_consolidates_duplicate_triples(self):
        """Triple with 3 parallel edges → 2 removed, keeper gets +2 count."""
        triple_rows = [
            {
                "a_eid": "ea",
                "b_eid": "eb",
                "rel_type": "USES",
                "rel_eids": ["r-zzz", "r-aaa", "r-mmm"],
                "dup_count": 3,
            }
        ]
        mock_client = MagicMock()
        # 1 SELECT, then 2 writes (set count + delete losers)
        mock_client.execute_query = AsyncMock(side_effect=[triple_rows, None, None])
        with patch("app.services.entity_dedup_service.neo4j_client", mock_client):
            svc = EntityDeduplicationService()
            removed = await svc._pass_relationships("g1", dry_run=False)
        assert removed == 2
        # SET call carries delta=2 and keeper is lowest elementId ("r-aaa")
        set_call = mock_client.execute_query.await_args_list[1]
        set_params = set_call.args[1]
        assert set_params["keeper"] == "r-aaa"
        assert set_params["delta"] == 2

    @pytest.mark.unit
    async def test_dry_run_writes_nothing(self):
        triple_rows = [
            {
                "a_eid": "ea",
                "b_eid": "eb",
                "rel_type": "USES",
                "rel_eids": ["r1", "r2"],
                "dup_count": 2,
            }
        ]
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=triple_rows)
        with patch("app.services.entity_dedup_service.neo4j_client", mock_client):
            svc = EntityDeduplicationService()
            removed = await svc._pass_relationships("g1", dry_run=True)
        assert removed == 1  # reported as if it would happen
        # Only one call total — the SELECT
        assert mock_client.execute_query.await_count == 1


class TestDeduplicateOrchestration:
    @pytest.mark.unit
    async def test_runs_all_passes_by_default(self):
        mock_client = MagicMock()
        # Each pass returns empty so we just trace pass-running.
        mock_client.execute_query = AsyncMock(return_value=[])
        with patch("app.services.entity_dedup_service.neo4j_client", mock_client):
            svc = EntityDeduplicationService()
            report = await svc.deduplicate("g1")
        assert report.passes_run == [
            "canonical",
            "merge_canonical",
            "embedding",
            "relationships",
        ]

    @pytest.mark.unit
    async def test_can_select_subset_of_passes(self):
        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(return_value=[])
        with patch("app.services.entity_dedup_service.neo4j_client", mock_client):
            svc = EntityDeduplicationService()
            report = await svc.deduplicate("g1", passes=["relationships"])
        assert report.passes_run == ["relationships"]
        assert report.canonical_names_added == 0
        assert report.embedding_merges == 0
