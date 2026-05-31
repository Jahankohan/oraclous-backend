"""Unit tests for FederationService.

Verifies:
- Permission validation (ownership + federatable flag, fail-closed)
- Multi-tenancy: cross-tenant access raises FederationError 403
- Max graph_ids limit enforcement
- UNION ALL query builds correct per-graph subqueries (graph_id in params)
- SAME_AS deduplication produces CrossGraphLink for matching entities
- Schema validation: too few graph_ids, duplicates
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.schemas.federation_schemas import FederatedEntity, FederatedQueryOptions
from app.services.federation_service import FederationError, FederationService

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_async_driver(rows: list):
    """Return a mock AsyncDriver whose session().run().data() yields *rows*.

    driver.session() is a SYNC call returning an async context manager,
    so driver itself is a MagicMock (not AsyncMock).
    """
    mock_result = AsyncMock()
    mock_result.data = AsyncMock(return_value=rows)

    mock_session = MagicMock()
    mock_session.run = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    return mock_driver, mock_session


def _owned_federatable(graph_id: str, user_id: str) -> dict:
    return {
        "graph_id": graph_id,
        "user_id": user_id,
        "name": f"Graph {graph_id}",
        "federatable": True,
    }


def _owned_non_federatable(graph_id: str, user_id: str) -> dict:
    return {
        "graph_id": graph_id,
        "user_id": user_id,
        "name": f"Graph {graph_id}",
        "federatable": False,
    }


# ─── Validation tests ─────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_rejects_cross_tenant_graph():
    """FederationError 403 when graph belongs to a different user."""
    attacker_id = "user-2"
    # Both graphs exist but owned by user-1, not the attacker
    rows = [
        {
            "graph_id": "graph-a",
            "user_id": "user-1",
            "name": "Graph A",
            "federatable": True,
        },
        {
            "graph_id": "graph-b",
            "user_id": "user-1",
            "name": "Graph B",
            "federatable": True,
        },
    ]
    driver, _ = _make_async_driver(rows)
    svc = FederationService(async_driver=driver)

    with pytest.raises(FederationError) as exc_info:
        await svc._validate_and_filter(attacker_id, ["graph-a", "graph-b"])

    assert exc_info.value.status_code == 403
    # 403 body must NOT leak graph_id — prevents enumeration attack (ORA-89)
    assert "graph-a" not in str(exc_info.value)
    assert "graph-b" not in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_rejects_non_federatable_graph():
    """FederationError 400 when graph.federatable=false."""
    user_id = "user-1"
    graph_ids = ["graph-a", "graph-b"]
    rows = [
        _owned_federatable("graph-a", user_id),
        _owned_non_federatable("graph-b", user_id),
    ]
    driver, _ = _make_async_driver(rows)
    svc = FederationService(async_driver=driver)

    with pytest.raises(FederationError) as exc_info:
        await svc._validate_and_filter(user_id, graph_ids)

    assert exc_info.value.status_code == 400
    # Error must NOT leak the graph_id (ORA-89 security fix)
    assert "graph-b" not in str(exc_info.value)
    assert "graph-a" not in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_rejects_missing_graph():
    """FederationError 400 when a requested graph_id doesn't exist."""
    user_id = "user-1"
    driver, _ = _make_async_driver([_owned_federatable("graph-a", user_id)])
    svc = FederationService(async_driver=driver)

    with pytest.raises(FederationError) as exc_info:
        await svc._validate_and_filter(user_id, ["graph-a", "missing-graph"])

    assert exc_info.value.status_code == 400
    assert "missing-graph" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_rejects_too_many_graphs():
    """FederationError 400 when more than MAX_GRAPH_IDS requested."""
    from app.schemas.federation_schemas import MAX_GRAPH_IDS

    driver, _ = _make_async_driver([])
    svc = FederationService(async_driver=driver)
    too_many = [f"graph-{i}" for i in range(MAX_GRAPH_IDS + 1)]

    with pytest.raises(FederationError) as exc_info:
        await svc._validate_and_filter("user-1", too_many)

    assert exc_info.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_passes_for_all_owned_and_federatable():
    """No error when all graphs are owned and federatable."""
    user_id = "user-1"
    graph_ids = ["graph-a", "graph-b"]
    rows = [_owned_federatable(gid, user_id) for gid in graph_ids]
    driver, _ = _make_async_driver(rows)
    svc = FederationService(async_driver=driver)

    result = await svc._validate_and_filter(user_id, graph_ids)
    assert {r["graph_id"] for r in result} == set(graph_ids)


# ─── Regression tests (ORA-102) ───────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_uses_owner_user_id_not_user_id():
    """Regression for ORA-102: query must read g.owner_user_id, not g.user_id.

    The Graph node stores ownership under owner_user_id (set by rebac_service).
    Using g.user_id (which is always None) caused all federated queries to 403.
    """
    import inspect

    from app.services.federation_service import FederationService

    source = inspect.getsource(FederationService._validate_and_filter)
    assert "g.owner_user_id" in source, (
        "Regression: _validate_and_filter must read g.owner_user_id (not g.user_id)"
    )
    assert "g.user_id AS user_id" not in source, (
        "Regression: _validate_and_filter must NOT read g.user_id (it is always None)"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_matches_system_namespace():
    """Regression for ORA-102: query must filter Graph nodes by namespace='__system__'.

    Without the namespace filter, non-ReBAC Graph nodes could be matched,
    causing false positives/negatives in ownership checks.
    """
    import inspect

    from app.services.federation_service import FederationService

    source = inspect.getsource(FederationService._validate_and_filter)
    assert "__system__" in source, (
        "Regression: _validate_and_filter must filter by namespace='__system__'"
    )


# ─── Query builder tests ──────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_entity_union_passes_graph_ids_as_params():
    """Each graph_id must appear as a separate parameter (not string interpolation)."""
    user_id = "user-1"
    graph_ids = ["graph-x", "graph-y"]
    entity_rows = [
        {
            "entity_id": "e1",
            "name": "Alice",
            "type": "Person",
            "source_graph_id": "graph-x",
        },
        {
            "entity_id": "e2",
            "name": "Bob",
            "type": "Person",
            "source_graph_id": "graph-y",
        },
    ]
    validation_rows = [_owned_federatable(gid, user_id) for gid in graph_ids]

    # First call: validation. Second call: entity UNION.
    call_count = 0
    mock_result = AsyncMock()

    async def side_effect_data():
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return validation_rows
        return entity_rows

    mock_result.data = side_effect_data

    mock_session = MagicMock()
    mock_session.run = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    driver = MagicMock()
    driver.session.return_value = mock_session

    svc = FederationService(async_driver=driver)
    await svc.federated_query(
        user_id=user_id,
        graph_ids=graph_ids,
        search_term="Alice",
        options=FederatedQueryOptions(deduplicate_entities=False),
    )

    # Verify run was called twice (validation + union)
    assert mock_session.run.call_count == 2

    # Second call must pass gid_0 and gid_1 params (not string-interpolated values)
    _, union_call_kwargs = mock_session.run.call_args_list[1]
    # call is positional: run(query, params)
    union_params = mock_session.run.call_args_list[1].args[1]
    assert "gid_0" in union_params
    assert "gid_1" in union_params
    assert union_params["gid_0"] == "graph-x"
    assert union_params["gid_1"] == "graph-y"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_federated_query_returns_source_attribution():
    """Every entity in the response must have source_graph_id set."""
    user_id = "user-1"
    graph_ids = ["graph-a", "graph-b"]
    validation_rows = [_owned_federatable(gid, user_id) for gid in graph_ids]
    entity_rows = [
        {
            "entity_id": "e1",
            "name": "Alice",
            "type": "Person",
            "source_graph_id": "graph-a",
        },
        {
            "entity_id": "e2",
            "name": "Bob",
            "type": "Person",
            "source_graph_id": "graph-b",
        },
    ]

    call_count = 0
    mock_result = AsyncMock()

    async def side_effect_data():
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return validation_rows
        return entity_rows

    mock_result.data = side_effect_data
    mock_session = MagicMock()
    mock_session.run = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    driver = MagicMock()
    driver.session.return_value = mock_session

    svc = FederationService(async_driver=driver)
    result = await svc.federated_query(
        user_id=user_id,
        graph_ids=graph_ids,
        search_term="test",
        options=FederatedQueryOptions(deduplicate_entities=False),
    )

    assert result["status"] == "ok"
    assert result["graphs_queried"] == graph_ids
    for entity in result["entities"]:
        assert entity.source_graph_id in graph_ids


# ─── SAME_AS deduplication tests ──────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_same_as_deduplication_produces_link_for_matching_entities():
    """Two entities with same name+type from different graphs → SAME_AS link."""
    entities = [
        FederatedEntity(
            entity_id="e1",
            name="Alice Chen",
            type="Person",
            source_graph_id="graph-a",
            source_graph_name="HR",
        ),
        FederatedEntity(
            entity_id="e2",
            name="Alice Chen",
            type="Person",
            source_graph_id="graph-b",
            source_graph_name="Projects",
        ),
    ]

    driver = MagicMock()
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.execute_write = AsyncMock(return_value=None)
    driver.session.return_value = mock_session

    svc = FederationService(async_driver=driver)
    links = await svc._resolve_same_as(entities)

    assert len(links) == 1
    link = links[0]
    assert link.link_type == "SAME_AS"
    assert link.confidence >= 0.85
    assert {link.graph_a, link.graph_b} == {"graph-a", "graph-b"}
    # execute_write must have been called to persist the SAME_AS link (ORA-142)
    mock_session.execute_write.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_store_same_as_links_is_awaited():
    """Regression for ORA-142: _store_same_as_links must be awaited, not fire-and-forget.

    Verifies execute_write is called with an async callable and the result is awaited.
    """
    driver = MagicMock()
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.execute_write = AsyncMock(return_value=None)
    driver.session.return_value = mock_session

    svc = FederationService(async_driver=driver)
    await svc._store_same_as_links([("id_a", "id_b", 0.99, "graph-a", "graph-b")])

    mock_session.execute_write.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_same_as_no_link_for_same_graph():
    """Two entities with same name+type in the SAME graph must NOT produce SAME_AS."""
    entities = [
        FederatedEntity(
            entity_id="e1",
            name="Alice Chen",
            type="Person",
            source_graph_id="graph-a",
            source_graph_name="HR",
        ),
        FederatedEntity(
            entity_id="e2",
            name="Alice Chen",
            type="Person",
            source_graph_id="graph-a",
            source_graph_name="HR",
        ),
    ]

    driver = AsyncMock()
    svc = FederationService(async_driver=driver)
    links = await svc._resolve_same_as(entities)

    assert links == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_same_as_no_link_for_different_types():
    """Same name but different types → no SAME_AS link."""
    entities = [
        FederatedEntity(
            entity_id="e1",
            name="Apollo",
            type="Project",
            source_graph_id="graph-a",
            source_graph_name="HR",
        ),
        FederatedEntity(
            entity_id="e2",
            name="Apollo",
            type="SpaceMission",
            source_graph_id="graph-b",
            source_graph_name="Projects",
        ),
    ]

    driver = AsyncMock()
    svc = FederationService(async_driver=driver)
    links = await svc._resolve_same_as(entities)

    assert links == []


# ─── ORA-217 regression: valid CALL { UNION ALL } syntax ─────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_entity_union_cypher_uses_single_outer_call():
    """ORA-217 regression: generated Cypher must NOT use CALL {} UNION ALL CALL {}.

    Neo4j 5.23+ rejects queries that conclude with a CALL subquery.
    The fix wraps all UNION ALL branches in a single outer CALL block.
    """
    import inspect

    source = inspect.getsource(FederationService._execute_entity_union)
    # The old (broken) pattern had each branch wrapped as its own CALL
    # joined at the top level: CALL{} UNION ALL CALL{}
    # The fixed pattern has one outer CALL containing all branches.
    # Verify the branch-building loop does NOT produce per-branch CALL wrappers.
    assert 'f"CALL {\\n"' not in source and "f'CALL {\\n'" not in source, (
        "ORA-217 regression: branches must NOT be individually wrapped in CALL"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_entity_union_cypher_structure_single_call_wrapping():
    """ORA-217: _execute_entity_union must emit CALL { <union branches> } RETURN."""
    graph_ids = ["graph-x", "graph-y", "graph-z"]

    captured_cypher: list[str] = []
    captured_params: list[dict] = []

    mock_result = AsyncMock()
    mock_result.data = AsyncMock(return_value=[])

    mock_session = MagicMock()

    async def capture_run(cypher, params=None):
        captured_cypher.append(cypher)
        captured_params.append(params or {})
        return mock_result

    mock_session.run = capture_run
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    driver = MagicMock()
    driver.session.return_value = mock_session

    svc = FederationService(async_driver=driver)
    await svc._execute_entity_union(
        graph_ids=graph_ids, search_term="test", max_per_graph=10
    )

    assert len(captured_cypher) == 1, "Expected exactly one Cypher execution"
    cypher = captured_cypher[0]

    # Must start with a single outer CALL block
    assert cypher.startswith("CALL {"), f"Cypher must start with 'CALL {{': {cypher!r}"

    # Must have UNION ALL inside (not at the top level)
    call_start = cypher.index("CALL {")
    call_close = cypher.rindex("}")
    inner_body = cypher[call_start + len("CALL {") : call_close]
    assert "UNION ALL" in inner_body, (
        f"UNION ALL must be inside the outer CALL block: {cypher!r}"
    )

    # Outer RETURN must follow the closing brace
    tail = cypher[call_close + 1 :].strip()
    assert tail.startswith("RETURN"), (
        f"Query must conclude with RETURN after CALL block: {cypher!r}"
    )

    # All graph_id params parameterized
    params = captured_params[0]
    for i in range(len(graph_ids)):
        assert f"gid_{i}" in params, f"Missing param gid_{i}"


# ─── Pydantic schema validation tests ────────────────────────────────────────


@pytest.mark.unit
def test_federated_query_request_rejects_duplicate_graph_ids():
    from pydantic import ValidationError

    from app.schemas.federation_schemas import FederatedQueryRequest

    with pytest.raises(ValidationError) as exc_info:
        FederatedQueryRequest(graph_ids=["g1", "g1"], query="test")

    assert "unique" in str(exc_info.value).lower()


@pytest.mark.unit
def test_federated_query_request_rejects_single_graph():
    from pydantic import ValidationError

    from app.schemas.federation_schemas import FederatedQueryRequest

    with pytest.raises(ValidationError):
        FederatedQueryRequest(graph_ids=["g1"], query="test")


@pytest.mark.unit
def test_federated_query_request_rejects_too_many_graphs():
    from pydantic import ValidationError

    from app.schemas.federation_schemas import MAX_GRAPH_IDS, FederatedQueryRequest

    too_many = [f"g{i}" for i in range(MAX_GRAPH_IDS + 1)]
    with pytest.raises(ValidationError):
        FederatedQueryRequest(graph_ids=too_many, query="test")
