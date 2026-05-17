"""Unit tests for the seven graph algorithm tools (TASK-033 / STORY-020).

Every test:
- Uses a mock Neo4j async driver — no real database required
- Verifies graph_id is present in Cypher params (as $gid)
- Verifies ToolNotPermittedError is raised when tool not in allowlist
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.schemas.agent_schemas import NodeResult, PathResult
from app.services.agent_tools import AgentToolkit, ToolNotPermittedError

# ── Test helpers ──────────────────────────────────────────────────────────────


class _FakeNode(dict):
    """Dict subclass that behaves like a Neo4j Node for dict() conversion."""


def _make_driver(records=None):
    mock_result = MagicMock()
    mock_result.records = records or []
    driver = AsyncMock()
    driver.execute_query = AsyncMock(return_value=mock_result)
    return driver, mock_result


def _toolkit(driver, tools: list[str], embedder=None) -> AgentToolkit:
    return AgentToolkit(driver, allowed_tools=tools, embedder=embedder)


# ── ToolNotPermittedError ─────────────────────────────────────────────────────


class TestToolNotPermitted:
    async def test_graph_search_blocked(self):
        driver, _ = _make_driver()
        with pytest.raises(ToolNotPermittedError):
            await _toolkit(driver, []).graph_search("g", "q")

    async def test_community_members_blocked(self):
        driver, _ = _make_driver()
        with pytest.raises(ToolNotPermittedError):
            await _toolkit(driver, []).community_members("g", "c")

    async def test_neighbors_blocked(self):
        driver, _ = _make_driver()
        with pytest.raises(ToolNotPermittedError):
            await _toolkit(driver, []).neighbors("g", "n")

    async def test_degree_centrality_blocked(self):
        driver, _ = _make_driver()
        with pytest.raises(ToolNotPermittedError):
            await _toolkit(driver, []).degree_centrality("g", "Person")

    async def test_shortest_path_blocked(self):
        driver, _ = _make_driver()
        with pytest.raises(ToolNotPermittedError):
            await _toolkit(driver, []).shortest_path("g", "a::b", "c::d")

    async def test_taint_trace_blocked(self):
        driver, _ = _make_driver()
        with pytest.raises(ToolNotPermittedError):
            await _toolkit(driver, []).taint_trace("g", "mod::fn")

    async def test_temporal_slice_blocked(self):
        driver, _ = _make_driver()
        with pytest.raises(ToolNotPermittedError):
            await _toolkit(driver, []).temporal_slice("g", "Event", 1000)

    async def test_error_carries_tool_name(self):
        driver, _ = _make_driver()
        try:
            await _toolkit(driver, []).taint_trace("g", "mod::fn")
        except ToolNotPermittedError as exc:
            assert exc.tool_name == "taint_trace"

    async def test_allowed_tool_does_not_raise(self):
        driver, _ = _make_driver()
        # Should NOT raise — driver returns empty records
        result = await _toolkit(driver, ["community_members"]).community_members(
            "g", "c"
        )
        assert result == []


# ── graph_search ──────────────────────────────────────────────────────────────


class TestGraphSearch:
    def _make_rec(self, props: dict):
        node = _FakeNode(props)
        rec = {"node": node, "score": 0.9}
        return rec

    async def test_returns_node_results(self):
        rec = self._make_rec({"id": "n1", "label": "Foo", "graph_id": "g1"})
        driver, _ = _make_driver(records=[rec])
        embedder = AsyncMock()
        embedder.embed_query = AsyncMock(return_value=[0.1, 0.2])
        results = await _toolkit(driver, ["graph_search"], embedder).graph_search(
            "g1", "query"
        )
        assert len(results) == 1
        assert isinstance(results[0], NodeResult)

    async def test_embed_query_called_with_text(self):
        driver, _ = _make_driver()
        embedder = AsyncMock()
        embedder.embed_query = AsyncMock(return_value=[0.0])
        await _toolkit(driver, ["graph_search"], embedder).graph_search(
            "g1", "my question"
        )
        embedder.embed_query.assert_called_once_with("my question")

    async def test_graph_id_in_params(self):
        # TASK-205: graph_search now scopes with `graph_id IN $gids`. A
        # single-graph call wraps the id into a one-element list — which
        # is equivalent to the pre-TASK-205 `= $gid` filter.
        driver, _ = _make_driver()
        embedder = AsyncMock()
        embedder.embed_query = AsyncMock(return_value=[0.0])
        await _toolkit(driver, ["graph_search"], embedder).graph_search(
            "target-graph", "q"
        )
        params = driver.execute_query.call_args[0][1]
        assert params["gids"] == ["target-graph"]

    async def test_graph_search_spans_multiple_graphs(self):
        # TASK-205: a list of graph ids is passed straight through to the
        # `IN $gids` filter, de-duplicated and order-preserving.
        driver, _ = _make_driver()
        embedder = AsyncMock()
        embedder.embed_query = AsyncMock(return_value=[0.0])
        await _toolkit(driver, ["graph_search"], embedder).graph_search(
            ["src-graph", "linked-graph", "src-graph"], "q"
        )
        params = driver.execute_query.call_args[0][1]
        assert params["gids"] == ["src-graph", "linked-graph"]

    async def test_max_results_in_params(self):
        driver, _ = _make_driver()
        embedder = AsyncMock()
        embedder.embed_query = AsyncMock(return_value=[0.0])
        await _toolkit(driver, ["graph_search"], embedder).graph_search(
            "g", "q", max_results=7
        )
        params = driver.execute_query.call_args[0][1]
        assert params["max_results"] == 7


# ── community_members ─────────────────────────────────────────────────────────


class TestCommunityMembers:
    async def test_returns_node_results(self):
        rec = {"n": _FakeNode({"id": "n1", "label": "Member", "graph_id": "g1"})}
        driver, _ = _make_driver(records=[rec])
        results = await _toolkit(driver, ["community_members"]).community_members(
            "g1", "c42"
        )
        assert len(results) == 1
        assert isinstance(results[0], NodeResult)

    async def test_graph_id_in_params(self):
        # TASK-205: community_members scopes with `graph_id IN $gids`.
        driver, _ = _make_driver()
        await _toolkit(driver, ["community_members"]).community_members("g-xyz", "c1")
        params = driver.execute_query.call_args[0][1]
        assert params["gids"] == ["g-xyz"]

    async def test_community_id_in_params(self):
        driver, _ = _make_driver()
        await _toolkit(driver, ["community_members"]).community_members("g", "comm-99")
        params = driver.execute_query.call_args[0][1]
        assert params["cid"] == "comm-99"

    async def test_empty_returns_empty_list(self):
        driver, _ = _make_driver(records=[])
        results = await _toolkit(driver, ["community_members"]).community_members(
            "g", "c"
        )
        assert results == []


# ── neighbors ─────────────────────────────────────────────────────────────────


class TestNeighbors:
    async def test_returns_node_results(self):
        rec = {"m": _FakeNode({"id": "m1", "label": "Neighbor", "graph_id": "g1"})}
        driver, _ = _make_driver(records=[rec])
        results = await _toolkit(driver, ["neighbors"]).neighbors("g1", "n1")
        assert len(results) == 1

    async def test_depth_embedded_as_literal_in_cypher(self):
        driver, _ = _make_driver()
        await _toolkit(driver, ["neighbors"]).neighbors("g1", "n1", depth=3)
        cypher = driver.execute_query.call_args[0][0]
        assert "*1..3" in cypher

    async def test_depth_is_not_a_param(self):
        driver, _ = _make_driver()
        await _toolkit(driver, ["neighbors"]).neighbors("g1", "n1", depth=3)
        params = driver.execute_query.call_args[0][1]
        assert "depth" not in params

    async def test_depth_capped_at_max(self):
        driver, _ = _make_driver()
        await _toolkit(driver, ["neighbors"]).neighbors("g1", "n1", depth=9999)
        cypher = driver.execute_query.call_args[0][0]
        assert "*1..20" in cypher

    async def test_edge_type_filter_included(self):
        driver, _ = _make_driver()
        await _toolkit(driver, ["neighbors"]).neighbors("g1", "n1", edge_type="CALLS")
        cypher = driver.execute_query.call_args[0][0]
        params = driver.execute_query.call_args[0][1]
        assert "edge_type" in params
        assert params["edge_type"] == "CALLS"
        assert "TYPE(rel)" in cypher

    async def test_no_edge_type_no_filter(self):
        driver, _ = _make_driver()
        await _toolkit(driver, ["neighbors"]).neighbors("g1", "n1")
        cypher = driver.execute_query.call_args[0][0]
        params = driver.execute_query.call_args[0][1]
        assert "edge_type" not in params
        assert "TYPE(rel)" not in cypher

    async def test_graph_id_in_params(self):
        driver, _ = _make_driver()
        await _toolkit(driver, ["neighbors"]).neighbors("g-abc", "n1")
        params = driver.execute_query.call_args[0][1]
        assert params["gid"] == "g-abc"


# ── degree_centrality ─────────────────────────────────────────────────────────


class TestDegreeCentrality:
    async def test_returns_node_results(self):
        rec = {
            "n": _FakeNode({"id": "n1", "label": "Person", "graph_id": "g1"}),
            "degree": 5,
        }
        driver, _ = _make_driver(records=[rec])
        results = await _toolkit(driver, ["degree_centrality"]).degree_centrality(
            "g1", "Person"
        )
        assert len(results) == 1

    async def test_node_label_embedded_in_cypher(self):
        driver, _ = _make_driver()
        await _toolkit(driver, ["degree_centrality"]).degree_centrality("g1", "Concept")
        cypher = driver.execute_query.call_args[0][0]
        assert ":Concept" in cypher

    async def test_graph_id_in_params(self):
        # TASK-205: degree_centrality scopes with `graph_id IN $gids`.
        driver, _ = _make_driver()
        await _toolkit(driver, ["degree_centrality"]).degree_centrality(
            "g-123", "Entity"
        )
        params = driver.execute_query.call_args[0][1]
        assert params["gids"] == ["g-123"]

    async def test_top_n_in_params(self):
        driver, _ = _make_driver()
        await _toolkit(driver, ["degree_centrality"]).degree_centrality(
            "g", "X", top_n=3
        )
        params = driver.execute_query.call_args[0][1]
        assert params["top_n"] == 3

    async def test_invalid_label_raises_value_error(self):
        driver, _ = _make_driver()
        with pytest.raises(ValueError):
            await _toolkit(driver, ["degree_centrality"]).degree_centrality(
                "g", "DROP; MATCH(n)"
            )

    async def test_label_not_a_cypher_param(self):
        driver, _ = _make_driver()
        await _toolkit(driver, ["degree_centrality"]).degree_centrality("g", "Thing")
        params = driver.execute_query.call_args[0][1]
        assert "node_label" not in params


# ── shortest_path ─────────────────────────────────────────────────────────────


class TestShortestPath:
    async def test_returns_none_when_no_path(self):
        driver, _ = _make_driver(records=[])
        result = await _toolkit(driver, ["shortest_path"]).shortest_path(
            "g", "a::X", "b::Y"
        )
        assert result is None

    async def test_returns_path_result(self):
        n1 = _FakeNode(
            {"id": "n1", "label": "A", "qualified_name": "mod::A", "graph_id": "g"}
        )
        n2 = _FakeNode(
            {"id": "n2", "label": "B", "qualified_name": "mod::B", "graph_id": "g"}
        )
        mock_path = MagicMock()
        mock_path.nodes = [n1, n2]
        rec = {"p": mock_path, "hop_count": 1}
        driver, _ = _make_driver(records=[rec])
        result = await _toolkit(driver, ["shortest_path"]).shortest_path(
            "g", "mod::A", "mod::B"
        )
        assert result is not None
        assert isinstance(result, PathResult)
        assert result.hop_count == 1
        assert len(result.nodes) == 2

    async def test_graph_id_in_params(self):
        driver, _ = _make_driver()
        await _toolkit(driver, ["shortest_path"]).shortest_path("g-999", "a", "b")
        params = driver.execute_query.call_args[0][1]
        assert params["gid"] == "g-999"

    async def test_from_and_to_in_params(self):
        driver, _ = _make_driver()
        await _toolkit(driver, ["shortest_path"]).shortest_path("g", "src::A", "dst::B")
        params = driver.execute_query.call_args[0][1]
        assert params["from"] == "src::A"
        assert params["to"] == "dst::B"


# ── taint_trace ───────────────────────────────────────────────────────────────


class TestTaintTrace:
    async def test_returns_node_results(self):
        rec = {"sink": _FakeNode({"id": "s1", "label": "Sink", "graph_id": "g"})}
        driver, _ = _make_driver(records=[rec])
        results = await _toolkit(driver, ["taint_trace"]).taint_trace("g", "src::fn")
        assert len(results) == 1

    async def test_depth_embedded_as_literal_in_cypher(self):
        driver, _ = _make_driver()
        await _toolkit(driver, ["taint_trace"]).taint_trace("g", "src::main", depth=5)
        cypher = driver.execute_query.call_args[0][0]
        assert "*1..5" in cypher

    async def test_depth_is_not_a_param(self):
        driver, _ = _make_driver()
        await _toolkit(driver, ["taint_trace"]).taint_trace("g", "src::main", depth=5)
        params = driver.execute_query.call_args[0][1]
        assert "depth" not in params

    async def test_flows_to_in_cypher(self):
        driver, _ = _make_driver()
        await _toolkit(driver, ["taint_trace"]).taint_trace("g", "src::main")
        cypher = driver.execute_query.call_args[0][0]
        assert "FLOWS_TO" in cypher

    async def test_graph_id_in_params(self):
        driver, _ = _make_driver()
        await _toolkit(driver, ["taint_trace"]).taint_trace("g-taint", "src::fn")
        params = driver.execute_query.call_args[0][1]
        assert params["gid"] == "g-taint"

    async def test_depth_capped_at_max(self):
        driver, _ = _make_driver()
        await _toolkit(driver, ["taint_trace"]).taint_trace("g", "src::fn", depth=9999)
        cypher = driver.execute_query.call_args[0][0]
        assert "*1..20" in cypher


# ── temporal_slice ────────────────────────────────────────────────────────────


class TestTemporalSlice:
    async def test_returns_node_results(self):
        rec = {
            "n": _FakeNode(
                {"id": "e1", "label": "Event", "graph_id": "g", "valid_from": 0}
            )
        }
        driver, _ = _make_driver(records=[rec])
        results = await _toolkit(driver, ["temporal_slice"]).temporal_slice(
            "g", "Event", 500
        )
        assert len(results) == 1

    async def test_graph_id_in_params(self):
        # TASK-205: temporal_slice scopes with `graph_id IN $gids`.
        driver, _ = _make_driver()
        await _toolkit(driver, ["temporal_slice"]).temporal_slice(
            "g-time", "Event", 1000
        )
        params = driver.execute_query.call_args[0][1]
        assert params["gids"] == ["g-time"]

    async def test_at_time_in_params(self):
        driver, _ = _make_driver()
        await _toolkit(driver, ["temporal_slice"]).temporal_slice("g", "Event", 12345)
        params = driver.execute_query.call_args[0][1]
        assert params["at_time"] == 12345

    async def test_valid_from_and_valid_to_in_cypher(self):
        driver, _ = _make_driver()
        await _toolkit(driver, ["temporal_slice"]).temporal_slice("g", "Event", 0)
        cypher = driver.execute_query.call_args[0][0]
        assert "valid_from" in cypher
        assert "valid_to" in cypher

    async def test_node_label_embedded_in_cypher(self):
        driver, _ = _make_driver()
        await _toolkit(driver, ["temporal_slice"]).temporal_slice("g", "Snapshot", 0)
        cypher = driver.execute_query.call_args[0][0]
        assert ":Snapshot" in cypher

    async def test_invalid_label_raises_value_error(self):
        driver, _ = _make_driver()
        with pytest.raises(ValueError):
            await _toolkit(driver, ["temporal_slice"]).temporal_slice(
                "g", "'; DROP TABLE events; --", 0
            )

    async def test_max_results_in_params(self):
        driver, _ = _make_driver()
        await _toolkit(driver, ["temporal_slice"]).temporal_slice(
            "g", "Event", 0, max_results=25
        )
        params = driver.execute_query.call_args[0][1]
        assert params["max_results"] == 25


# ── find_communities (STORY-4d) ──────────────────────────────────────────────


def _vector_rec(community_id, score, summary, size=10, keywords=None, excerpt=None):
    """Mock one row from db.index.vector.queryNodes."""
    return {
        "community_id": community_id,
        "summary": summary,
        "size": size,
        "keywords": keywords,
        "excerpt": excerpt,
        "score": score,
    }


class TestFindCommunities:
    async def test_blocked_when_not_in_allowlist(self):
        driver, _ = _make_driver()
        embedder = MagicMock()
        embedder.embed_query = MagicMock(return_value=[0.0] * 3072)
        with pytest.raises(ToolNotPermittedError):
            await _toolkit(driver, [], embedder=embedder).find_communities(
                "g1", "query"
            )

    async def test_searches_all_kinds_by_default(self):
        """``kind=None`` issues one Cypher call per registered kind."""
        driver, _ = _make_driver()
        # Each call returns one match
        results = [
            MagicMock(records=[_vector_rec("c-entity", 0.9, "entity sum")]),
            MagicMock(
                records=[
                    _vector_rec(
                        "cc-chunk", 0.95, "chunk sum", keywords='["x"]', excerpt="e"
                    )
                ]
            ),
        ]
        driver.execute_query = AsyncMock(side_effect=results)
        embedder = MagicMock()
        embedder.embed_query = MagicMock(return_value=[0.0] * 3072)
        out = await _toolkit(
            driver, ["find_communities"], embedder=embedder
        ).find_communities("g1", "test query")

        # Two calls: one per kind
        assert driver.execute_query.await_count == 2
        # Merged by score descending — chunk first (0.95) then entity (0.9)
        assert len(out) == 2
        assert out[0].id == "cc-chunk"
        assert out[1].id == "c-entity"

    async def test_explicit_kind_restricts_to_one_index(self):
        driver, _ = _make_driver()
        driver.execute_query = AsyncMock(
            return_value=MagicMock(records=[_vector_rec("cc-1", 0.8, "s")])
        )
        embedder = MagicMock()
        embedder.embed_query = MagicMock(return_value=[0.0] * 3072)
        out = await _toolkit(
            driver, ["find_communities"], embedder=embedder
        ).find_communities("g1", "q", kind="chunk")
        # Only one call — the chunk index
        assert driver.execute_query.await_count == 1
        cypher = driver.execute_query.call_args[0][0]
        assert "community_embeddings_chunk" in cypher
        assert len(out) == 1

    async def test_graph_id_passed_via_params(self):
        driver, _ = _make_driver()
        driver.execute_query = AsyncMock(return_value=MagicMock(records=[]))
        embedder = MagicMock()
        embedder.embed_query = MagicMock(return_value=[0.0] * 3072)
        await _toolkit(
            driver, ["find_communities"], embedder=embedder
        ).find_communities("my-graph", "q", kind="chunk")
        params = driver.execute_query.call_args[0][1]
        # TASK-205: find_communities scopes with `graph_id IN $gids`.
        assert params["gids"] == ["my-graph"]

    async def test_keywords_parsed_from_json(self):
        driver, _ = _make_driver()
        driver.execute_query = AsyncMock(
            return_value=MagicMock(
                records=[
                    _vector_rec(
                        "cc-1", 0.9, "s", keywords='["k1","k2","k3"]', excerpt="e"
                    )
                ]
            )
        )
        embedder = MagicMock()
        embedder.embed_query = MagicMock(return_value=[0.0] * 3072)
        out = await _toolkit(
            driver, ["find_communities"], embedder=embedder
        ).find_communities("g", "q", kind="chunk")
        assert out[0].properties["keywords"] == ["k1", "k2", "k3"]

    async def test_index_failure_does_not_abort_other_kinds(self):
        """If one kind's vector index doesn't exist, the others still
        succeed — log + continue rather than crash the tool call."""
        driver, _ = _make_driver()
        driver.execute_query = AsyncMock(
            side_effect=[
                RuntimeError("entity index missing"),
                MagicMock(records=[_vector_rec("cc-1", 0.7, "chunk")]),
            ]
        )
        embedder = MagicMock()
        embedder.embed_query = MagicMock(return_value=[0.0] * 3072)
        out = await _toolkit(
            driver, ["find_communities"], embedder=embedder
        ).find_communities("g", "q")
        # Entity errored, chunk succeeded
        assert len(out) == 1
        assert out[0].id == "cc-1"

    async def test_top_k_caps_merged_result(self):
        driver, _ = _make_driver()
        # Return 4 from each kind
        rec = lambda cid, s: _vector_rec(cid, s, "x")  # noqa: E731
        driver.execute_query = AsyncMock(
            side_effect=[
                MagicMock(records=[rec(f"e{i}", 0.5 + i * 0.01) for i in range(4)]),
                MagicMock(records=[rec(f"c{i}", 0.6 + i * 0.01) for i in range(4)]),
            ]
        )
        embedder = MagicMock()
        embedder.embed_query = MagicMock(return_value=[0.0] * 3072)
        out = await _toolkit(
            driver, ["find_communities"], embedder=embedder
        ).find_communities("g", "q", top_k=3)
        assert len(out) == 3


# ── describe_community (STORY-4d) ────────────────────────────────────────────


class TestDescribeCommunity:
    async def test_blocked_when_not_in_allowlist(self):
        driver, _ = _make_driver()
        with pytest.raises(ToolNotPermittedError):
            await _toolkit(driver, []).describe_community("g", "cc-1")

    async def test_probes_registry_when_kind_omitted(self):
        """No kind hint → entity probed first; if it returns no rows
        the chunk path is tried."""
        driver, _ = _make_driver()
        # First call (entity probe) → no records. Second call (chunk
        # probe) → one record. Third call (chunk member fetch) → empty.
        driver.execute_query = AsyncMock(
            side_effect=[
                MagicMock(records=[]),
                MagicMock(
                    records=[
                        {
                            "community_id": "cc-1",
                            "summary": "Test cluster",
                            "size": 42,
                            "keywords": '["x"]',
                            "excerpt": "ex",
                        }
                    ]
                ),
                MagicMock(records=[]),
            ]
        )
        out = await _toolkit(driver, ["describe_community"]).describe_community(
            "g", "cc-1"
        )
        assert len(out) == 1
        assert out[0].id == "cc-1"
        assert out[0].properties["kind"] == "chunk"
        assert out[0].properties["keywords"] == ["x"]
        assert out[0].properties["excerpt"] == "ex"
        assert out[0].properties["size"] == 42

    async def test_explicit_kind_skips_probe(self):
        driver, _ = _make_driver()
        driver.execute_query = AsyncMock(
            side_effect=[
                MagicMock(
                    records=[
                        {
                            "community_id": "cc-1",
                            "summary": "s",
                            "size": 5,
                            "keywords": None,
                            "excerpt": None,
                        }
                    ]
                ),
                MagicMock(records=[]),
            ]
        )
        await _toolkit(driver, ["describe_community"]).describe_community(
            "g", "cc-1", kind="chunk"
        )
        # Only 2 calls: 1 for the chunk row, 1 for members. No entity probe.
        assert driver.execute_query.await_count == 2

    async def test_returns_empty_when_no_kind_matches(self):
        driver, _ = _make_driver()
        # Both probes return no records
        driver.execute_query = AsyncMock(
            side_effect=[MagicMock(records=[]), MagicMock(records=[])]
        )
        out = await _toolkit(driver, ["describe_community"]).describe_community(
            "g", "nonexistent"
        )
        assert out == []

    async def test_sample_members_capped_at_5(self):
        driver, _ = _make_driver()
        driver.execute_query = AsyncMock(
            side_effect=[
                MagicMock(
                    records=[
                        {
                            "community_id": "cc-1",
                            "summary": "s",
                            "size": 100,
                            "keywords": None,
                            "excerpt": None,
                        }
                    ]
                ),
                # 5 members returned (the Cypher LIMIT 5 already capped)
                MagicMock(
                    records=[
                        {"member_id": f"m{i}", "member_preview": f"text-{i}"}
                        for i in range(5)
                    ]
                ),
            ]
        )
        out = await _toolkit(driver, ["describe_community"]).describe_community(
            "g", "cc-1", kind="chunk"
        )
        assert len(out[0].properties["sample_members"]) == 5

    async def test_graph_id_passed_via_params(self):
        driver, _ = _make_driver()
        driver.execute_query = AsyncMock(
            side_effect=[
                MagicMock(
                    records=[
                        {
                            "community_id": "cc-1",
                            "summary": "s",
                            "size": 5,
                            "keywords": None,
                            "excerpt": None,
                        }
                    ]
                ),
                MagicMock(records=[]),
            ]
        )
        await _toolkit(driver, ["describe_community"]).describe_community(
            "my-graph", "cc-1", kind="chunk"
        )
        # The first call is the community-row fetch
        params = driver.execute_query.call_args_list[0][0][1]
        # TASK-205: describe_community scopes with `graph_id IN $gids`.
        assert params["gids"] == ["my-graph"]
