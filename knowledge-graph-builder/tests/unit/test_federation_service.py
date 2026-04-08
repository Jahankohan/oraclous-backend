"""Unit tests for FederationService.

Verifies:
- Permission validation (ownership + federatable flag, fail-closed)
- Multi-tenancy: cross-tenant access raises FederationError 403
- Max graph_ids limit enforcement
- UNION ALL query builds correct per-graph subqueries (graph_id in params)
- SAME_AS deduplication produces CrossGraphLink for matching entities
- Schema validation: too few graph_ids, duplicates
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.federation_service import FederationError, FederationService
from app.schemas.federation_schemas import FederatedQueryOptions, FederatedEntity


# Patch OpenAIEmbeddings for the entire test module — no real API key needed
@pytest.fixture(autouse=True)
def mock_embedder(monkeypatch):
    mock = MagicMock()
    mock.embed_query.return_value = [0.1] * 1536
    monkeypatch.setattr(
        "app.services.federation_service.OpenAIEmbeddings",
        lambda **kwargs: mock,
    )
    return mock


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
    return {"graph_id": graph_id, "user_id": user_id, "name": f"Graph {graph_id}", "federatable": True}


def _owned_non_federatable(graph_id: str, user_id: str) -> dict:
    return {"graph_id": graph_id, "user_id": user_id, "name": f"Graph {graph_id}", "federatable": False}


# ─── Validation tests ─────────────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_rejects_cross_tenant_graph():
    """FederationError 403 when graph belongs to a different user."""
    attacker_id = "user-2"
    # Both graphs exist but owned by user-1, not the attacker
    rows = [
        {"graph_id": "graph-a", "user_id": "user-1", "name": "Graph A", "federatable": True},
        {"graph_id": "graph-b", "user_id": "user-1", "name": "Graph B", "federatable": True},
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
    driver, _ = _make_async_driver(
        [_owned_federatable("graph-a", user_id)]
    )
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


# ─── Query builder tests ──────────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.asyncio
async def test_entity_union_passes_graph_ids_as_params():
    """Each graph_id must appear as a separate parameter (not string interpolation)."""
    user_id = "user-1"
    graph_ids = ["graph-x", "graph-y"]
    entity_rows = [
        {"entity_id": "e1", "name": "Alice", "type": "Person", "source_graph_id": "graph-x"},
        {"entity_id": "e2", "name": "Bob", "type": "Person", "source_graph_id": "graph-y"},
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
    result = await svc.federated_query(
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
        {"entity_id": "e1", "name": "Alice", "type": "Person", "source_graph_id": "graph-a"},
        {"entity_id": "e2", "name": "Bob", "type": "Person", "source_graph_id": "graph-b"},
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
            entity_id="e1", name="Alice Chen", type="Person",
            source_graph_id="graph-a", source_graph_name="HR"
        ),
        FederatedEntity(
            entity_id="e2", name="Alice Chen", type="Person",
            source_graph_id="graph-b", source_graph_name="Projects"
        ),
    ]

    driver = MagicMock()
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.execute_write = AsyncMock(return_value=None)
    driver.session.return_value = mock_session

    svc = FederationService(async_driver=driver)

    with patch("asyncio.create_task") as mock_create_task:
        links = await svc._resolve_same_as(entities)

    assert len(links) == 1
    link = links[0]
    assert link.link_type == "SAME_AS"
    assert link.confidence >= 0.85
    assert {link.graph_a, link.graph_b} == {"graph-a", "graph-b"}
    # create_task must have been called to store the link
    mock_create_task.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_same_as_no_link_for_same_graph():
    """Two entities with same name+type in the SAME graph must NOT produce SAME_AS."""
    entities = [
        FederatedEntity(
            entity_id="e1", name="Alice Chen", type="Person",
            source_graph_id="graph-a", source_graph_name="HR"
        ),
        FederatedEntity(
            entity_id="e2", name="Alice Chen", type="Person",
            source_graph_id="graph-a", source_graph_name="HR"
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
            entity_id="e1", name="Apollo", type="Project",
            source_graph_id="graph-a", source_graph_name="HR"
        ),
        FederatedEntity(
            entity_id="e2", name="Apollo", type="SpaceMission",
            source_graph_id="graph-b", source_graph_name="Projects"
        ),
    ]

    driver = AsyncMock()
    svc = FederationService(async_driver=driver)
    links = await svc._resolve_same_as(entities)

    assert links == []


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
    from app.schemas.federation_schemas import FederatedQueryRequest, MAX_GRAPH_IDS

    too_many = [f"g{i}" for i in range(MAX_GRAPH_IDS + 1)]
    with pytest.raises(ValidationError):
        FederatedQueryRequest(graph_ids=too_many, query="test")


# ─── ORA-103 regression tests — vector search uses real embedding ─────────────

@pytest.mark.unit
@pytest.mark.asyncio
async def test_vector_search_calls_embed_query_with_query_text(mock_embedder):
    """federated_vector_search must call embed_query(query_text), never pass [] (ORA-103)."""
    user_id = "user-1"
    graph_ids = ["graph-a", "graph-b"]
    validation_rows = [_owned_federatable(gid, user_id) for gid in graph_ids]
    vector_rows = [
        {
            "chunk_id": "c1", "text": "hello world", "score": 0.92,
            "source_graph_id": "graph-a", "entity_name": None, "entity_type": None,
        }
    ]

    call_count = 0
    mock_result = AsyncMock()

    async def side_effect_data():
        nonlocal call_count
        call_count += 1
        return validation_rows if call_count == 1 else vector_rows

    mock_result.data = side_effect_data
    mock_session = MagicMock()
    mock_session.run = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    driver = MagicMock()
    driver.session.return_value = mock_session

    svc = FederationService(async_driver=driver)
    await svc.federated_vector_search(
        user_id=user_id,
        graph_ids=graph_ids,
        query_text="machine learning",
        top_k=5,
    )

    # embed_query must have been called with the exact query text
    mock_embedder.embed_query.assert_called_once_with("machine learning")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_vector_search_passes_real_vector_to_neo4j(mock_embedder):
    """The Neo4j session must receive the real embedding vector, not [] (ORA-103)."""
    FAKE_VECTOR = [0.42] * 1536
    mock_embedder.embed_query.return_value = FAKE_VECTOR

    user_id = "user-1"
    graph_ids = ["graph-a", "graph-b"]
    validation_rows = [_owned_federatable(gid, user_id) for gid in graph_ids]

    call_count = 0
    mock_result = AsyncMock()

    async def side_effect_data():
        nonlocal call_count
        call_count += 1
        return validation_rows if call_count == 1 else []

    mock_result.data = side_effect_data
    mock_session = MagicMock()
    mock_session.run = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    driver = MagicMock()
    driver.session.return_value = mock_session

    svc = FederationService(async_driver=driver)
    await svc.federated_vector_search(
        user_id=user_id,
        graph_ids=graph_ids,
        query_text="some query",
        top_k=5,
    )

    # Second session.run call is the vector search — check it received the real vector
    vector_search_call = mock_session.run.call_args_list[1]
    vector_params = vector_search_call.args[1]
    assert vector_params["query_vector"] == FAKE_VECTOR, (
        "Neo4j must receive the real embedding vector, not an empty list"
    )
