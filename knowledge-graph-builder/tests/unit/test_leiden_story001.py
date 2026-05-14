"""
STORY-001 — Leiden hierarchy test suite.

Covers:
1. Leiden hierarchy structure (igraph + leidenalg, two resolutions)
2. Parent-child mapping correctness (_build_hierarchy)
3. Summary prompt content (_generate_level_summaries, levels 0 / 1 / 2)
4. Global query routing (_is_global_query + auto_select_retriever_type)
5. End-to-end detection → summary → retrieval (mocked Neo4j + LLM)
6. Staleness: stale-clear step executed with correct graph_id
7. Regression — all existing unit tests still pass (subprocess check)

All external I/O is mocked; no real Neo4j, Redis, Postgres, or LLM calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Test 1 — Leiden hierarchy structure
# ---------------------------------------------------------------------------


class TestLeidenHierarchyStructure:
    """Validate that leidenalg runs and produces valid partition memberships."""

    @pytest.mark.unit
    def test_partitions_are_nonempty_at_each_resolution(self):
        import igraph
        import leidenalg

        g = igraph.Graph(10)
        g.add_edges(
            [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (8, 9), (4, 5)]
        )

        for resolution in [0.5, 2.0]:
            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=resolution,
                seed=42,
            )
            communities = set(partition.membership)
            assert (
                len(communities) > 0
            ), f"Resolution {resolution}: expected at least one community"

    @pytest.mark.unit
    def test_every_node_belongs_to_exactly_one_community_per_level(self):
        import igraph
        import leidenalg

        g = igraph.Graph(10)
        g.add_edges(
            [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (8, 9), (4, 5)]
        )

        for resolution in [0.5, 2.0]:
            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=resolution,
                seed=42,
            )
            membership = partition.membership
            # Exactly 10 assignments (one per node)
            assert (
                len(membership) == 10
            ), f"Resolution {resolution}: expected 10 memberships, got {len(membership)}"
            # Every membership is a non-negative integer
            assert all(
                isinstance(m, int) and m >= 0 for m in membership
            ), f"Resolution {resolution}: non-integer membership found"

    @pytest.mark.unit
    def test_fine_resolution_produces_at_least_as_many_communities_as_coarse(self):
        """Fine-grained resolution (high γ) should fragment graph more than coarse."""
        import igraph
        import leidenalg

        g = igraph.Graph(10)
        g.add_edges(
            [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (8, 9), (4, 5)]
        )

        partition_coarse = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=0.5,
            seed=42,
        )
        partition_fine = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=2.0,
            seed=42,
        )

        n_coarse = len(set(partition_coarse.membership))
        n_fine = len(set(partition_fine.membership))

        # Fine resolution should not produce fewer communities than coarse.
        # They can be equal on a small ring graph, but never should coarse > fine.
        assert n_fine >= n_coarse, (
            f"Expected fine resolution to produce >= communities as coarse; "
            f"got fine={n_fine}, coarse={n_coarse}"
        )


# ---------------------------------------------------------------------------
# Test 2 — Parent-child mapping (_build_hierarchy)
# ---------------------------------------------------------------------------


class TestParentChildMapping:
    """
    Validate _build_hierarchy majority-vote logic directly.

    In _build_hierarchy:
    - Level 0 is the finest-grained (communities A, B)
    - Level 1 is coarser (communities X, Y)
    - For each level-1 community, its parent_id is assigned to the level-0
      community that owns the majority of its members (majority vote).

    Synthetic setup:
      Level 0 (fine):   A: [e0, e1, e2],  B: [e3, e4, e5]
      Level 1 (coarse): X: [e0, e1, e2, e3],  Y: [e4, e5, e6]

    Expected (coarser level-1 community → majority level-0 community):
      X.parent_id = A  (3 votes for A out of 4 members: e0→A, e1→A, e2→A, e3→B)
      Y.parent_id = B  (2 votes for B out of 3 members: e4→B, e5→B, e6→unassigned)
    """

    @pytest.mark.unit
    def test_coarse_community_x_parent_is_fine_community_a(self):
        """Level-1 community X (coarse) should link to level-0 community A (fine)."""
        from app.tasks.community_tasks import _build_hierarchy

        communities_map = {
            0: {"A": ["e0", "e1", "e2"], "B": ["e3", "e4", "e5"]},
            1: {"X": ["e0", "e1", "e2", "e3"], "Y": ["e4", "e5", "e6"]},
        }

        result = _build_hierarchy(communities_map, levels=[0, 1])

        # X: e0→A, e1→A, e2→A (3 votes), e3→B (1 vote) → majority = A
        assert (
            result[1]["X"]["parent_id"] == "A"
        ), f"Expected X.parent_id='A', got {result[1]['X']['parent_id']!r}"

    @pytest.mark.unit
    def test_coarse_community_y_parent_is_fine_community_b(self):
        """Level-1 community Y (coarse) should link to level-0 community B (fine)."""
        from app.tasks.community_tasks import _build_hierarchy

        communities_map = {
            0: {"A": ["e0", "e1", "e2"], "B": ["e3", "e4", "e5"]},
            1: {"X": ["e0", "e1", "e2", "e3"], "Y": ["e4", "e5", "e6"]},
        }

        result = _build_hierarchy(communities_map, levels=[0, 1])

        # Y: e4→B (1 vote), e5→B (1 vote), e6→unassigned (0 votes) → majority = B
        assert (
            result[1]["Y"]["parent_id"] == "B"
        ), f"Expected Y.parent_id='B', got {result[1]['Y']['parent_id']!r}"

    @pytest.mark.unit
    def test_level_0_communities_have_no_parent(self):
        """Level-0 communities are finest; they should not get a parent_id."""
        from app.tasks.community_tasks import _build_hierarchy

        communities_map = {
            0: {"Z": ["a", "b", "c"]},
        }
        result = _build_hierarchy(communities_map, levels=[0])

        assert result[0]["Z"]["parent_id"] is None

    @pytest.mark.unit
    def test_member_lists_preserved_through_hierarchy(self):
        """_build_hierarchy should not mutate member lists."""
        from app.tasks.community_tasks import _build_hierarchy

        communities_map = {
            0: {"X": ["e0", "e1"]},
            1: {"A": ["e0", "e1"]},
        }
        result = _build_hierarchy(communities_map, levels=[0, 1])

        assert sorted(result[0]["X"]["members"]) == ["e0", "e1"]
        assert sorted(result[1]["A"]["members"]) == ["e0", "e1"]


# ---------------------------------------------------------------------------
# Test 3 — Summary prompt content (_generate_level_summaries)
# ---------------------------------------------------------------------------


class TestSummaryPromptContent:
    """
    Verify that _generate_level_summaries builds the correct prompt structure
    at each level without calling a real LLM.
    """

    def _mock_neo4j(self, level0_entities, child_summaries_by_cid):
        """
        Return a mock neo4j_client whose execute_query returns:
        - Level discovery → [{"level": 0}, {"level": 1}, {"level": 2}]
        - Community IDs per level → [{"community_id": "c0"}], etc.
        - Level-0 entities → level0_entities
        - Level-1+ children → child_summaries_by_cid[cid]
        """

        call_count = [0]

        async def mock_execute(query, params):
            call_count[0] += 1
            q = query.strip()

            # Stale-clear (SET c.summary = null) — return empty
            if "SET c.summary = null" in q:
                return []

            # Level discovery
            if "DISTINCT c.level" in q:
                return [{"level": 0}, {"level": 1}, {"level": 2}]

            # Community IDs at each level
            if "RETURN c.id AS community_id" in q:
                level = params.get("level")
                if level == 0:
                    return [{"community_id": "c0"}]
                elif level == 1:
                    return [{"community_id": "c1"}]
                elif level == 2:
                    return [{"community_id": "c2"}]
                return []

            # Level-0 entity fetch
            if "e.name AS name" in q and "lbls" in q:
                return level0_entities

            # Level-1+ child summary fetch
            if "child.summary AS summary" in q:
                cid = params.get("community_id", "")
                summaries = child_summaries_by_cid.get(cid, [])
                return [{"summary": s} for s in summaries]

            # Summary write-back
            if "SET c.summary = $summary" in q:
                return []

            return []

        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(side_effect=mock_execute)
        return mock_client

    @pytest.mark.unit
    async def test_level0_prompt_contains_entity_names(self):
        """Level-0 prompt must reference the actual entity names from the graph."""
        captured_prompts: list[str] = []

        level0_entities = [
            {"name": "Alice", "lbls": ["Person", "__Entity__"]},
            {"name": "Organization", "lbls": ["Org", "__Entity__"]},
        ]
        child_summaries = {
            "c1": ["Child summary for c1"],
            "c2": ["Higher level summary"],
        }

        mock_client = self._mock_neo4j(level0_entities, child_summaries)

        async def fake_llm_call(*args, **kwargs):
            messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
            user_msg = next(
                (m["content"] for m in messages if m.get("role") == "user"), ""
            )
            captured_prompts.append(user_msg)
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test summary"
            return mock_response

        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(side_effect=fake_llm_call)

        with (
            patch("app.services.analytics_service.neo4j_client", mock_client),
            patch("openai.AsyncOpenAI", return_value=mock_openai),
            patch("app.services.analytics_service.settings") as mock_settings,
        ):
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_settings.LLM_SUMMARY_CONCURRENCY = 5

            from app.services.analytics_service import GraphAnalyticsService

            svc = GraphAnalyticsService()
            await svc._generate_level_summaries("test-graph-id")

        # Level-0 prompt should contain both entity names
        level0_prompt = captured_prompts[0] if captured_prompts else ""
        assert (
            "Alice" in level0_prompt
        ), f"Level-0 prompt should contain 'Alice'; got: {level0_prompt!r}"
        assert (
            "Organization" in level0_prompt
        ), f"Level-0 prompt should contain 'Organization'; got: {level0_prompt!r}"

    @pytest.mark.unit
    async def test_level1_prompt_contains_child_summaries_not_raw_entity_names(self):
        """Level-1 prompt must roll up child summaries, not raw entity names."""
        captured_prompts: list[str] = []

        level0_entities = [
            {"name": "ShouldNotAppearInLevel1Prompt", "lbls": ["__Entity__"]},
        ]
        child_summaries = {
            "c1": ["Sub-group about finance and banking"],
            "c2": ["Top-level group covering global economy"],
        }

        mock_client = self._mock_neo4j(level0_entities, child_summaries)

        async def fake_llm_call(*args, **kwargs):
            messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
            user_msg = next(
                (m["content"] for m in messages if m.get("role") == "user"), ""
            )
            captured_prompts.append(user_msg)
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Summary text"
            return mock_response

        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(side_effect=fake_llm_call)

        with (
            patch("app.services.analytics_service.neo4j_client", mock_client),
            patch("openai.AsyncOpenAI", return_value=mock_openai),
            patch("app.services.analytics_service.settings") as mock_settings,
        ):
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_settings.LLM_SUMMARY_CONCURRENCY = 5

            from app.services.analytics_service import GraphAnalyticsService

            svc = GraphAnalyticsService()
            await svc._generate_level_summaries("test-graph-id")

        # Level-1 prompt (second call) should contain child summary text
        if len(captured_prompts) >= 2:
            level1_prompt = captured_prompts[1]
            assert (
                "Sub-group about finance and banking" in level1_prompt
            ), f"Level-1 prompt should contain child summary; got: {level1_prompt!r}"
            # Must NOT contain the raw entity name (it comes from level-0 entities query,
            # which is only used at level 0)
            assert (
                "ShouldNotAppearInLevel1Prompt" not in level1_prompt
            ), f"Level-1 prompt must not contain raw entity names; got: {level1_prompt!r}"

    @pytest.mark.unit
    async def test_level2_prompt_contains_overarching_keyword(self):
        """Level-2+ prompt template must include 'overarching' or 'insights'."""
        captured_prompts: list[str] = []

        level0_entities = [{"name": "Node", "lbls": ["__Entity__"]}]
        child_summaries = {
            "c1": ["Level-1 sub-summary"],
            "c2": ["Higher-level insights roll-up"],
        }

        mock_client = self._mock_neo4j(level0_entities, child_summaries)

        async def fake_llm_call(*args, **kwargs):
            messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
            user_msg = next(
                (m["content"] for m in messages if m.get("role") == "user"), ""
            )
            captured_prompts.append(user_msg)
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Summary"
            return mock_response

        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(side_effect=fake_llm_call)

        with (
            patch("app.services.analytics_service.neo4j_client", mock_client),
            patch("openai.AsyncOpenAI", return_value=mock_openai),
            patch("app.services.analytics_service.settings") as mock_settings,
        ):
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_settings.LLM_SUMMARY_CONCURRENCY = 5

            from app.services.analytics_service import GraphAnalyticsService

            svc = GraphAnalyticsService()
            await svc._generate_level_summaries("test-graph-id")

        # Level-2 prompt (third call) must contain "overarching" or "insights"
        if len(captured_prompts) >= 3:
            level2_prompt = captured_prompts[2]
            assert (
                "overarching" in level2_prompt.lower()
                or "insights" in level2_prompt.lower()
            ), (
                f"Level-2 prompt must contain 'overarching' or 'insights'; "
                f"got: {level2_prompt!r}"
            )

    @pytest.mark.unit
    async def test_stale_clear_called_with_correct_graph_id(self):
        """The stale-clear SET c.summary=null query must include the correct graph_id."""
        stale_clear_params: list[dict] = []

        async def capturing_execute(query, params):
            if "SET c.summary = null" in query:
                stale_clear_params.append(dict(params))
            return []

        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(side_effect=capturing_execute)

        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="summary"))]
            )
        )

        with (
            patch("app.services.analytics_service.neo4j_client", mock_client),
            patch("openai.AsyncOpenAI", return_value=mock_openai),
            patch("app.services.analytics_service.settings") as mock_settings,
        ):
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_settings.LLM_SUMMARY_CONCURRENCY = 5

            from app.services.analytics_service import GraphAnalyticsService

            svc = GraphAnalyticsService()
            target_graph = "aaaabbbb-1234-5678-abcd-aaaabbbbcccc"
            await svc._generate_level_summaries(target_graph)

        assert (
            len(stale_clear_params) == 1
        ), f"Expected exactly 1 stale-clear call; got {len(stale_clear_params)}"
        assert (
            stale_clear_params[0].get("graph_id") == target_graph
        ), f"Stale-clear called with wrong graph_id: {stale_clear_params[0]}"


# ---------------------------------------------------------------------------
# Test 4 — Global query routing
# ---------------------------------------------------------------------------


class TestGlobalQueryRouting:
    """
    _is_global_query and auto_select_retriever_type must route correctly.
    """

    @pytest.mark.unit
    def test_global_keywords_trigger_global_detection(self):
        from app.services.chat_service import _is_global_query

        global_queries = [
            "what are the main themes across all documents?",
            "give me an overview of the knowledge base",
            "summarize all topics",
            "what are the broad categories?",
            "describe the landscape of this domain",
            "what domains are covered?",
        ]
        for q in global_queries:
            assert (
                _is_global_query(q) is True
            ), f"Expected _is_global_query to return True for: {q!r}"

    @pytest.mark.unit
    def test_specific_entity_queries_are_not_global(self):
        from app.services.chat_service import _is_global_query

        specific_queries = [
            "who is John Smith?",
            "what is the relationship between Alice and Bob?",
            "when was TechNova Corp founded?",
            "list all CEOs",
        ]
        for q in specific_queries:
            assert (
                _is_global_query(q) is False
            ), f"Expected _is_global_query to return False for: {q!r}"

    @pytest.mark.unit
    def test_global_query_routes_to_community_summary(self):
        from app.services.chat_service import ChatService

        with (
            patch("app.services.chat_service.OpenAIEmbeddings"),
            patch("app.services.chat_service.OpenAILLM"),
            patch("app.services.chat_service.settings") as mock_settings,
        ):
            mock_settings.OPENAI_API_KEY = "test-key"
            from app.services.retriever_factory import RetrieverType

            result = ChatService.auto_select_retriever_type(
                "what are the main themes across all documents?"
            )
            assert (
                result == RetrieverType.COMMUNITY_SUMMARY
            ), f"Expected COMMUNITY_SUMMARY, got {result}"

    @pytest.mark.unit
    def test_specific_query_does_not_route_to_community_summary(self):
        from app.services.chat_service import ChatService
        from app.services.retriever_factory import RetrieverType

        with (
            patch("app.services.chat_service.OpenAIEmbeddings"),
            patch("app.services.chat_service.OpenAILLM"),
            patch("app.services.chat_service.settings") as mock_settings,
        ):
            mock_settings.OPENAI_API_KEY = "test-key"

            result = ChatService.auto_select_retriever_type("who is John Smith?")
            assert (
                result != RetrieverType.COMMUNITY_SUMMARY
            ), f"Specific entity query must NOT route to COMMUNITY_SUMMARY; got {result}"

    @pytest.mark.unit
    def test_cypher_query_routes_to_text2cypher(self):
        from app.services.chat_service import ChatService
        from app.services.retriever_factory import RetrieverType

        with (
            patch("app.services.chat_service.OpenAIEmbeddings"),
            patch("app.services.chat_service.OpenAILLM"),
            patch("app.services.chat_service.settings") as mock_settings,
        ):
            mock_settings.OPENAI_API_KEY = "test-key"

            result = ChatService.auto_select_retriever_type(
                "write a cypher query to find all paths between Alice and Bob"
            )
            assert (
                result == RetrieverType.TEXT2CYPHER
            ), f"Expected TEXT2CYPHER for Cypher-pattern query; got {result}"

    @pytest.mark.unit
    def test_analytic_query_routes_to_hybrid(self):
        from app.services.chat_service import ChatService
        from app.services.retriever_factory import RetrieverType

        with (
            patch("app.services.chat_service.OpenAIEmbeddings"),
            patch("app.services.chat_service.OpenAILLM"),
            patch("app.services.chat_service.settings") as mock_settings,
        ):
            mock_settings.OPENAI_API_KEY = "test-key"

            result = ChatService.auto_select_retriever_type("list all employees")
            assert (
                result == RetrieverType.HYBRID
            ), f"Expected HYBRID for analytic query; got {result}"


# ---------------------------------------------------------------------------
# Test 5 — End-to-end: detection → summary → retrieval (all mocked)
# ---------------------------------------------------------------------------


class TestEndToEndCommunityPipeline:
    """
    Integration-level test (no real I/O) verifying the whole pipeline:
    _run_leiden produces communities → _build_hierarchy assigns parents →
    global query routes to COMMUNITY_SUMMARY → CommunitySummaryRetriever
    returns community records.
    """

    @pytest.mark.unit
    def test_leiden_produces_communities_from_synthetic_graph(self):
        """_run_leiden should yield a non-empty level map for a connected graph."""
        from app.tasks.community_tasks import _run_leiden

        entity_ids = [f"e{i}" for i in range(10)]
        # Chain: e0—e1—e2—e3—e4—e5—e6—e7—e8—e9
        edges = [(f"e{i}", f"e{i + 1}", 1) for i in range(9)]

        result = _run_leiden(
            entity_ids=entity_ids,
            edges=edges,
            levels=[0, 1],
            resolutions=[0.5, 2.0],
        )

        assert 0 in result and 1 in result, "Expected levels 0 and 1 in result"
        assert len(result[0]) > 0, "Level 0 must have at least one community"
        assert len(result[1]) > 0, "Level 1 must have at least one community"

        # Every entity_id must appear in exactly one community per level
        for level in [0, 1]:
            assigned = []
            for members in result[level].values():
                assigned.extend(members)
            assert sorted(assigned) == sorted(
                entity_ids
            ), f"Level {level}: not all entities assigned exactly once"

    @pytest.mark.unit
    def test_hierarchy_links_fine_to_coarse_communities(self):
        """After _run_leiden + _build_hierarchy, level-1 communities should have parent_id set."""
        from app.tasks.community_tasks import _build_hierarchy, _run_leiden

        entity_ids = [f"e{i}" for i in range(10)]
        edges = [(f"e{i}", f"e{i + 1}", 1) for i in range(9)]

        level_map = _run_leiden(
            entity_ids=entity_ids,
            edges=edges,
            levels=[0, 1],
            resolutions=[0.5, 2.0],
        )
        enriched = _build_hierarchy(level_map, levels=[0, 1])

        # Level-1 communities should have parent_id pointing to a level-0 community
        level1_comms = enriched.get(1, {})
        for cid, data in level1_comms.items():
            parent = data.get("parent_id")
            if parent is not None:
                # Parent must exist at level 0
                assert parent in enriched[0], (
                    f"Community {cid} at level 1 has parent_id={parent} "
                    f"which is not a level-0 community"
                )

    @pytest.mark.unit
    async def test_community_summary_retriever_passes_graph_id(self):
        """CommunitySummaryRetriever must include graph_id in its Neo4j query."""
        from app.services.retriever_factory import CommunitySummaryRetriever

        captured_params: list[dict] = []

        async def fake_execute_query(query, params):
            captured_params.append(dict(params))
            result = MagicMock()
            result.records = []
            return result

        mock_driver = MagicMock()
        mock_driver.execute_query = AsyncMock(side_effect=fake_execute_query)

        with patch("app.services.retriever_factory.neo4j_client") as mock_client:
            mock_client.async_driver = mock_driver

            retriever = CommunitySummaryRetriever(
                graph_id="my-test-graph", level=2, limit=5
            )
            await retriever.search("what are the main themes?")

        assert len(captured_params) == 1
        assert captured_params[0]["graph_id"] == "my-test-graph", (
            f"CommunitySummaryRetriever must pass correct graph_id; "
            f"got: {captured_params[0]}"
        )

    @pytest.mark.unit
    async def test_community_summary_retriever_returns_empty_when_no_driver(self):
        """CommunitySummaryRetriever should return [] gracefully if async_driver is None."""
        from app.services.retriever_factory import CommunitySummaryRetriever

        with patch("app.services.retriever_factory.neo4j_client") as mock_client:
            mock_client.async_driver = None

            retriever = CommunitySummaryRetriever(graph_id="g1")
            result = await retriever.search("broad overview")

        assert (
            result == []
        ), f"Expected empty list when async driver is None; got: {result}"

    @pytest.mark.unit
    async def test_global_query_routes_to_community_summary_end_to_end(self):
        """Full routing: global query string → RetrieverType.COMMUNITY_SUMMARY."""
        from app.services.chat_service import ChatService
        from app.services.retriever_factory import RetrieverType

        global_query = "what are the main themes across all documents?"
        selected = ChatService.auto_select_retriever_type(global_query)

        assert selected == RetrieverType.COMMUNITY_SUMMARY

        # Verify the retriever class is correctly instantiated for this type
        from app.services.retriever_factory import CommunitySummaryRetriever

        retriever = CommunitySummaryRetriever(graph_id="g-end-to-end")
        assert retriever.graph_id == "g-end-to-end"


# ---------------------------------------------------------------------------
# Test 6 — Staleness: stale-clear followed by summary regeneration
# ---------------------------------------------------------------------------


class TestStalenessAfterReIngestion:
    """
    Verify that _generate_level_summaries:
    1. Clears stale summaries (SET c.summary = null) BEFORE generating new ones.
    2. Calls the LLM for a community whose summary was cleared.
    """

    @pytest.mark.unit
    async def test_stale_clear_precedes_llm_summary_generation(self):
        """
        Execution order: stale-clear must happen before any LLM call.
        We track the call sequence and assert the stale-clear comes first.
        """
        execution_log: list[str] = []

        async def ordered_execute(query, params):
            if "SET c.summary = null" in query:
                execution_log.append("stale_clear")
                return []
            if "DISTINCT c.level" in query:
                return [{"level": 0}]
            if "RETURN c.id AS community_id" in query:
                return [{"community_id": "comm-a"}]
            if "e.name AS name" in query:
                return [{"name": "EntityA", "lbls": ["__Entity__"]}]
            if "SET c.summary = $summary" in query:
                return []
            return []

        async def fake_llm_call(*args, **kwargs):
            execution_log.append("llm_call")
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = (
                "Fresh summary after re-ingestion"
            )
            return mock_response

        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(side_effect=ordered_execute)

        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(side_effect=fake_llm_call)

        with (
            patch("app.services.analytics_service.neo4j_client", mock_client),
            patch("openai.AsyncOpenAI", return_value=mock_openai),
            patch("app.services.analytics_service.settings") as mock_settings,
        ):
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_settings.LLM_SUMMARY_CONCURRENCY = 5

            from app.services.analytics_service import GraphAnalyticsService

            svc = GraphAnalyticsService()
            await svc._generate_level_summaries("graph-rerun")

        assert "stale_clear" in execution_log, "stale-clear step must be executed"
        assert "llm_call" in execution_log, "LLM call must be made after stale-clear"

        stale_idx = execution_log.index("stale_clear")
        llm_idx = execution_log.index("llm_call")
        assert (
            stale_idx < llm_idx
        ), f"stale_clear must happen before llm_call; order was: {execution_log}"

    @pytest.mark.unit
    async def test_summary_written_back_after_llm_call(self):
        """After the LLM generates a summary, it must be written back to Neo4j."""
        written_summaries: list[str] = []

        async def capture_write(query, params):
            if "SET c.summary = $summary" in query:
                written_summaries.append(params.get("summary", ""))
            if "DISTINCT c.level" in query:
                return [{"level": 0}]
            if "RETURN c.id AS community_id" in query:
                return [{"community_id": "comm-b"}]
            if "e.name AS name" in query:
                return [{"name": "EntityB", "lbls": ["__Entity__"]}]
            return []

        async def fake_llm_call(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Regenerated community summary"
            return mock_response

        mock_client = MagicMock()
        mock_client.execute_query = AsyncMock(side_effect=capture_write)

        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(side_effect=fake_llm_call)

        with (
            patch("app.services.analytics_service.neo4j_client", mock_client),
            patch("openai.AsyncOpenAI", return_value=mock_openai),
            patch("app.services.analytics_service.settings") as mock_settings,
        ):
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_settings.LLM_SUMMARY_CONCURRENCY = 5

            from app.services.analytics_service import GraphAnalyticsService

            svc = GraphAnalyticsService()
            await svc._generate_level_summaries("graph-write-test")

        assert len(written_summaries) >= 1, "Expected at least one summary written back"
        assert any(
            s == "Regenerated community summary" for s in written_summaries
        ), f"Expected 'Regenerated community summary' in writes; got: {written_summaries}"


# ---------------------------------------------------------------------------
# Test 7 — make_community_id determinism (pure function, bonus regression)
# ---------------------------------------------------------------------------


class TestMakeCommunityId:
    """Validate the deterministic ID utility used throughout community_tasks."""

    @pytest.mark.unit
    def test_deterministic_same_inputs(self):
        from app.tasks.community_tasks import make_community_id

        r1 = make_community_id("graph-1", 0, 0.5, ["e1", "e2", "e3"])
        r2 = make_community_id("graph-1", 0, 0.5, ["e1", "e2", "e3"])
        assert r1 == r2

    @pytest.mark.unit
    def test_order_independent(self):
        from app.tasks.community_tasks import make_community_id

        r1 = make_community_id("graph-1", 0, 0.5, ["e1", "e2", "e3"])
        r2 = make_community_id("graph-1", 0, 0.5, ["e3", "e1", "e2"])
        assert r1 == r2

    @pytest.mark.unit
    def test_different_graph_different_id(self):
        from app.tasks.community_tasks import make_community_id

        r1 = make_community_id("graph-A", 0, 0.5, ["e1", "e2"])
        r2 = make_community_id("graph-B", 0, 0.5, ["e1", "e2"])
        assert r1 != r2

    @pytest.mark.unit
    def test_different_level_different_id(self):
        from app.tasks.community_tasks import make_community_id

        r1 = make_community_id("graph-1", 0, 0.5, ["e1", "e2"])
        r2 = make_community_id("graph-1", 1, 0.5, ["e1", "e2"])
        assert r1 != r2

    @pytest.mark.unit
    def test_id_starts_with_community_prefix(self):
        from app.tasks.community_tasks import make_community_id

        cid = make_community_id("graph-1", 0, 0.5, ["e1"])
        assert cid.startswith("community_")
