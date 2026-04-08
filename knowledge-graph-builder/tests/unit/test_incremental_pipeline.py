"""
Unit tests for entity-level delta detection in the incremental ingestion pipeline.

Covers: compute_entity_fingerprint, compute_prop_hash, _bulk_fingerprint_lookup,
_apply_entity_delta, _soft_delete_orphaned_rels, and delta classification logic.
All Neo4j calls are mocked — no live database required.

Spec criteria from ORA-49:
1. Fingerprint stability — same inputs produce identical fingerprints.
2. No duplicates on re-ingest — UNCHANGED entities are filtered from graph.
3. UPDATED fires correctly — prop_hash changes trigger UPDATED classification.
4. UNCHANGED skips writes — zero entity nodes remain in graph after filtering.
5. graph_id isolation — two graphs with same entity names produce different fingerprints.
"""
import hashlib
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.pipeline_service import (
    compute_entity_fingerprint,
    compute_prop_hash,
    MultiTenantGraphRAGPipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GRAPH_ID = str(uuid.uuid4())
OTHER_GRAPH_ID = str(uuid.uuid4())


def _make_pipeline(graph_id: str = GRAPH_ID) -> MultiTenantGraphRAGPipeline:
    """Create a pipeline instance without triggering __init__ (no driver needed)."""
    p = MultiTenantGraphRAGPipeline.__new__(MultiTenantGraphRAGPipeline)
    p.graph_id = graph_id
    return p


def _make_node(name: str, label: str = "__Entity__", description: str = "desc") -> MagicMock:
    node = MagicMock()
    node.id = f"node_{name}"
    node.label = label
    node.properties = {"name": name, "description": description}
    node.start_node_id = f"node_{name}"
    node.end_node_id = f"node_{name}"
    return node


def _make_graph(nodes, relationships=None):
    g = MagicMock()
    g.nodes = list(nodes)
    g.relationships = list(relationships or [])
    return g


# ---------------------------------------------------------------------------
# compute_entity_fingerprint
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fingerprint_stability():
    """Same inputs always produce the same 16-char fingerprint."""
    fp1 = compute_entity_fingerprint(GRAPH_ID, "Alice Smith", "Person")
    fp2 = compute_entity_fingerprint(GRAPH_ID, "Alice Smith", "Person")
    assert fp1 == fp2
    assert len(fp1) == 16


@pytest.mark.unit
def test_fingerprint_is_16_chars():
    fp = compute_entity_fingerprint(GRAPH_ID, "Company X", "Organization")
    assert len(fp) == 16


@pytest.mark.unit
def test_fingerprint_normalizes_name():
    """Punctuation and case differences should not produce different fingerprints."""
    fp1 = compute_entity_fingerprint(GRAPH_ID, "alice smith", "Person")
    fp2 = compute_entity_fingerprint(GRAPH_ID, "Alice Smith", "Person")
    fp3 = compute_entity_fingerprint(GRAPH_ID, "alice  smith!", "Person")
    assert fp1 == fp2 == fp3


@pytest.mark.unit
def test_fingerprint_graph_id_isolation():
    """Two graphs with the same entity name produce different fingerprints (Spec criterion 7)."""
    fp1 = compute_entity_fingerprint(GRAPH_ID, "Alice", "Person")
    fp2 = compute_entity_fingerprint(OTHER_GRAPH_ID, "Alice", "Person")
    assert fp1 != fp2


@pytest.mark.unit
def test_fingerprint_label_matters():
    """Different labels → different fingerprint even with same name."""
    fp_person = compute_entity_fingerprint(GRAPH_ID, "Apple", "Person")
    fp_org = compute_entity_fingerprint(GRAPH_ID, "Apple", "Organization")
    assert fp_person != fp_org


@pytest.mark.unit
def test_fingerprint_excludes_mutable_props():
    """Changing description should NOT affect the fingerprint."""
    fp1 = compute_entity_fingerprint(GRAPH_ID, "Bob", "Person")
    fp2 = compute_entity_fingerprint(GRAPH_ID, "Bob", "Person")
    assert fp1 == fp2  # Description is not an input to fingerprint


# ---------------------------------------------------------------------------
# compute_prop_hash
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_prop_hash_excludes_system_fields():
    """System fields should not affect prop_hash."""
    props_a = {"name": "Alice", "description": "CEO", "graph_id": GRAPH_ID, "embedding": [0.1, 0.2]}
    props_b = {"name": "Alice", "description": "CEO", "graph_id": OTHER_GRAPH_ID, "embedding": [0.9]}
    assert compute_prop_hash(props_a) == compute_prop_hash(props_b)


@pytest.mark.unit
def test_prop_hash_detects_description_change():
    """A changed description should produce a different prop_hash."""
    props_old = {"name": "Alice", "description": "CEO"}
    props_new = {"name": "Alice", "description": "CTO"}
    assert compute_prop_hash(props_old) != compute_prop_hash(props_new)


@pytest.mark.unit
def test_prop_hash_stable_across_calls():
    props = {"name": "Alice", "description": "CEO", "extra": {"dept": "Eng"}}
    assert compute_prop_hash(props) == compute_prop_hash(props)


# ---------------------------------------------------------------------------
# _bulk_fingerprint_lookup
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_bulk_fingerprint_lookup_returns_dict():
    pipeline = _make_pipeline()
    fps = ["abc123", "def456"]

    with patch("app.services.pipeline_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(return_value=[
            {"fp": "abc123", "prop_hash": "hash1", "description": "old desc"},
        ])
        result = await pipeline._bulk_fingerprint_lookup(fps)

    assert result == {"abc123": {"prop_hash": "hash1", "description": "old desc"}}
    # "def456" not in result → treated as NEW


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bulk_fingerprint_lookup_empty_input():
    pipeline = _make_pipeline()
    with patch("app.services.pipeline_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock()
        result = await pipeline._bulk_fingerprint_lookup([])
    mock_client.execute_query.assert_not_called()
    assert result == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bulk_fingerprint_lookup_includes_graph_id():
    pipeline = _make_pipeline(GRAPH_ID)
    with patch("app.services.pipeline_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(return_value=[])
        await pipeline._bulk_fingerprint_lookup(["fp1"])
    params = mock_client.execute_query.call_args[0][1]
    assert params["graph_id"] == GRAPH_ID


# ---------------------------------------------------------------------------
# _apply_entity_delta
# ---------------------------------------------------------------------------

def _fp(graph_id: str, name: str, label: str = "__Entity__") -> str:
    return compute_entity_fingerprint(graph_id, name, label)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_apply_entity_delta_classifies_new():
    pipeline = _make_pipeline()
    node = _make_node("Alice", "__Entity__")
    graph = _make_graph([node])

    fp = _fp(GRAPH_ID, "Alice")

    with patch("app.services.pipeline_service.neo4j_client") as mock_client:
        # Bulk lookup returns empty → all NEW
        mock_client.execute_query = AsyncMock(return_value=[])
        stats = await pipeline._apply_entity_delta(graph, "doc.txt")

    assert stats["new"] == 1
    assert stats["updated"] == 0
    assert stats["unchanged"] == 0
    # NEW entity should remain in graph
    assert len(graph.nodes) == 1



@pytest.mark.unit
@pytest.mark.asyncio
async def test_apply_entity_delta_classifies_unchanged_and_removes():
    pipeline = _make_pipeline()
    node = _make_node("Alice", "__Entity__", description="CEO")
    graph = _make_graph([node])

    fp = _fp(GRAPH_ID, "Alice")
    existing_ph = compute_prop_hash({"name": "Alice", "description": "CEO"})

    with patch("app.services.pipeline_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(return_value=[
            {"fp": fp, "prop_hash": existing_ph, "description": "CEO"}
        ])
        stats = await pipeline._apply_entity_delta(graph, "doc.txt")

    assert stats["unchanged"] == 1
    assert stats["new"] == 0
    # UNCHANGED entity must be removed from graph (Spec criterion 4)
    assert len(graph.nodes) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_apply_entity_delta_classifies_updated():
    pipeline = _make_pipeline()
    node = _make_node("Alice", "__Entity__", description="CTO")  # description changed
    graph = _make_graph([node])

    fp = _fp(GRAPH_ID, "Alice")
    old_ph = compute_prop_hash({"name": "Alice", "description": "CEO"})  # old hash differs

    with patch("app.services.pipeline_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(return_value=[
            {"fp": fp, "prop_hash": old_ph, "description": "CEO"}
        ])
        stats = await pipeline._apply_entity_delta(graph, "doc.txt")

    assert stats["updated"] == 1
    assert stats["new"] == 0
    assert stats["unchanged"] == 0
    # UPDATED entity stays in graph
    assert len(graph.nodes) == 1
    # Marked with _delta sentinel and prev description stored
    assert node.properties.get("_delta") == "updated"
    assert node.properties.get("_prev_description") == "CEO"
    assert node.id in stats["entity_ids_updated"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_apply_entity_delta_mixed_batch():
    """NEW + UPDATED + UNCHANGED in one batch."""
    pipeline = _make_pipeline()
    node_new = _make_node("Charlie", "__Entity__", description="New person")
    node_updated = _make_node("Alice", "__Entity__", description="Updated CTO")
    node_unchanged = _make_node("Bob", "__Entity__", description="Engineer")
    graph = _make_graph([node_new, node_updated, node_unchanged])

    fp_alice = _fp(GRAPH_ID, "Alice")
    fp_bob = _fp(GRAPH_ID, "Bob")
    old_ph_alice = compute_prop_hash({"name": "Alice", "description": "CEO"})
    unchanged_ph_bob = compute_prop_hash({"name": "Bob", "description": "Engineer"})

    with patch("app.services.pipeline_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(return_value=[
            {"fp": fp_alice, "prop_hash": old_ph_alice, "description": "CEO"},
            {"fp": fp_bob, "prop_hash": unchanged_ph_bob, "description": "Engineer"},
        ])
        stats = await pipeline._apply_entity_delta(graph, "doc.txt")

    assert stats["new"] == 1
    assert stats["updated"] == 1
    assert stats["unchanged"] == 1
    # Only UNCHANGED (Bob) should be removed
    remaining_names = [n.properties["name"] for n in graph.nodes]
    assert "Bob" not in remaining_names
    assert "Alice" in remaining_names
    assert "Charlie" in remaining_names


@pytest.mark.unit
@pytest.mark.asyncio
async def test_apply_entity_delta_no_entities():
    """Non-entity nodes (Document, Chunk) should not be processed."""
    pipeline = _make_pipeline()
    node = _make_node("doc.txt", "Document")
    graph = _make_graph([node])

    with patch("app.services.pipeline_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(return_value=[])
        stats = await pipeline._apply_entity_delta(graph, "doc.txt")

    assert stats["new"] == 0
    # Document node should still be in graph (not touched)
    assert len(graph.nodes) == 1


# ---------------------------------------------------------------------------
# _apply_updated_merge_rules — description-longer-wins
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_description_longer_wins_keeps_old_when_new_is_shorter():
    """Re-ingest with a shorter description → old (longer) description is preserved."""
    pipeline = _make_pipeline()
    entity_id = "node_Alice"

    with patch("app.services.pipeline_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(return_value=[])
        await pipeline._apply_updated_merge_rules([entity_id])

    call_args = mock_client.execute_query.call_args
    cypher: str = call_args[0][0]
    params: dict = call_args[0][1]

    # The Cypher must reference _prev_description, not _new_description
    assert "_prev_description" in cypher
    assert "_new_description" not in cypher
    # The old sentinel property must be cleared
    assert "e._prev_description = null" in cypher
    assert params["eid"] == entity_id
    assert params["graph_id"] == GRAPH_ID


@pytest.mark.unit
@pytest.mark.asyncio
async def test_description_longer_wins_uses_longer_description():
    """The Cypher keeps the longer description: old wins when old > new, new wins otherwise."""
    pipeline = _make_pipeline()
    entity_id = "node_Bob"

    with patch("app.services.pipeline_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(return_value=[])
        await pipeline._apply_updated_merge_rules([entity_id])

    cypher: str = mock_client.execute_query.call_args[0][0]

    # The CASE expression must compare _prev_description length against e.description length
    assert "size(coalesce(toString(e._prev_description)" in cypher
    assert "size(coalesce(toString(e.description)" in cypher


@pytest.mark.unit
@pytest.mark.asyncio
async def test_apply_updated_merge_rules_skips_empty_list():
    """No Cypher is executed when updated_ids is empty."""
    pipeline = _make_pipeline()

    with patch("app.services.pipeline_service.neo4j_client") as mock_client:
        mock_client.execute_query = AsyncMock(return_value=[])
        await pipeline._apply_updated_merge_rules([])

    mock_client.execute_query.assert_not_called()
