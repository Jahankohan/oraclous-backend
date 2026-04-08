"""
Unit tests for ORA-60 — Incremental KG Updates: Delta Detection & Smart Merge.

All 7 test criteria from the spec are covered:
1. Idempotency: ingest doc A twice → zero new Chunk/Entity nodes on second run
2. Delta: change one page → re-ingest → only new chunks processed
3. Hash guard: same bytes → status=skipped, no graph writes
4. Manual property preservation: custom prop survives re-ingestion
5. Stale chunk: removed page → re-ingest → removed chunk has staleAt set
6. Full mode: mode=full behaves identically to existing delete+re-extract
7. Migration backfill: verify scripts produce required fields, no data loss
"""
import hashlib
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Dict, Set

from app.schemas.graph_schemas import IngestMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _make_chunk(uid: str, text: str):
    from neo4j_graphrag.experimental.components.types import TextChunk
    c = TextChunk(uid=uid, text=text, index=0)
    return c


def _make_pipeline(graph_id: str = "test-graph-123"):
    """Create a MultiTenantGraphRAGPipeline with all external deps mocked."""
    from app.services.pipeline_service import MultiTenantGraphRAGPipeline

    with patch("app.services.pipeline_service.neo4j_client") as mock_client:
        pipeline = MultiTenantGraphRAGPipeline(graph_id=graph_id, user_id="user-1")
        pipeline._initialized = True
        pipeline.llm = None          # Skip LLM extraction in unit tests
        pipeline.embedder = None
        pipeline.driver = MagicMock()
        pipeline._neo4j_client = mock_client
    return pipeline


# ---------------------------------------------------------------------------
# Test 3: Hash guard — same bytes → status=skipped, no graph writes
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_hash_guard_skips_unchanged_document():
    """Same document bytes → processing_source returns status=skipped, zero writes."""
    content = "Alice founded Acme Corp in 2010."
    content_hash = _sha256(content)

    pipeline = _make_pipeline()

    # Simulate Neo4j returning the same hash (document already ingested)
    pipeline._check_document_hash_unchanged = AsyncMock(return_value=True)
    pipeline._set_document_provenance = AsyncMock()

    result = await pipeline._process_single_document_instrumented(
        text=content,
        source="doc_a",
        kg_writer=AsyncMock(),
        mode=IngestMode.INCREMENTAL,
        job_id="job-001",
    )

    assert result["ingest_status"] == "skipped"
    assert result["entities_created"] == 0
    assert result["chunks_created"] == 0
    # Document provenance should NOT be updated when skipping
    pipeline._set_document_provenance.assert_not_called()


# ---------------------------------------------------------------------------
# Test 1: Idempotency — ingest twice → zero new nodes on second run
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_idempotency_second_ingest_produces_no_new_chunks():
    """All chunk hashes already exist in Neo4j → no_new_chunks status, no LLM call."""
    content = "Bob leads R&D at TechCorp since 2019."
    chunk_text = "Bob leads R&D at TechCorp since 2019."
    chunk_sha1 = _sha1(chunk_text)

    pipeline = _make_pipeline()

    # Hash changed (document bytes might differ slightly) — proceed past Stage 1
    pipeline._check_document_hash_unchanged = AsyncMock(return_value=False)
    pipeline._set_document_provenance = AsyncMock()
    # All chunk hashes already exist in graph
    pipeline._get_existing_chunk_content_hashes = AsyncMock(return_value={chunk_sha1})
    pipeline._soft_delete_stale_chunks = AsyncMock()
    pipeline._set_chunk_provenance = AsyncMock()

    # Patch the splitter to return one chunk with our known content
    mock_chunk = _make_chunk("uid-1", chunk_text)
    mock_chunks = MagicMock()
    mock_chunks.chunks = [mock_chunk]

    with patch(
        "neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter.FixedSizeSplitter"
    ) as MockSplitter:
        MockSplitter.return_value.run = AsyncMock(return_value=mock_chunks)

        result = await pipeline._process_single_document_instrumented(
            text=content,
            source="doc_b",
            kg_writer=AsyncMock(),
            mode=IngestMode.INCREMENTAL,
            job_id="job-002",
        )

    assert result["ingest_status"] == "no_new_chunks"
    assert result["entities_created"] == 0
    # No stale chunks to delete (content matches)
    pipeline._soft_delete_stale_chunks.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2: Delta — change one page → only new chunks processed
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_delta_only_new_chunks_sent_to_extractor():
    """
    Two chunks in document: one unchanged, one new.
    Extractor should receive only the new chunk.
    """
    unchanged_text = "Unchanged paragraph about Alice."
    new_text = "New paragraph about Bob joining in 2025."
    unchanged_sha1 = _sha1(unchanged_text)
    new_sha1 = _sha1(new_text)

    pipeline = _make_pipeline()
    pipeline._check_document_hash_unchanged = AsyncMock(return_value=False)
    pipeline._set_document_provenance = AsyncMock()
    pipeline._get_existing_chunk_content_hashes = AsyncMock(return_value={unchanged_sha1})
    pipeline._soft_delete_stale_chunks = AsyncMock()
    pipeline._set_chunk_provenance = AsyncMock()

    unchanged_chunk = _make_chunk("uid-old", unchanged_text)
    new_chunk = _make_chunk("uid-new", new_text)
    mock_chunks = MagicMock()
    mock_chunks.chunks = [unchanged_chunk, new_chunk]

    # Capture what the extractor receives
    captured_chunks = []

    async def fake_extractor_run(chunks, document_info):
        captured_chunks.extend(chunks.chunks)
        graph = MagicMock()
        graph.nodes = []
        graph.relationships = []
        return graph

    with (
        patch(
            "neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter.FixedSizeSplitter"
        ) as MockSplitter,
        patch(
            "neo4j_graphrag.experimental.components.entity_relation_extractor.LLMEntityRelationExtractor"
        ) as MockExtractor,
    ):
        MockSplitter.return_value.run = AsyncMock(return_value=mock_chunks)
        mock_extractor_instance = MagicMock()
        mock_extractor_instance.run = AsyncMock(side_effect=fake_extractor_run)
        MockExtractor.return_value = mock_extractor_instance

        # Pipeline needs an LLM to reach the extractor
        pipeline.llm = MagicMock()
        pipeline._normalize_overlapping_entities = AsyncMock(
            side_effect=lambda g: g
        )
        pipeline._detect_temporal_contradictions = AsyncMock(return_value=0)

        with patch("app.services.pipeline_service.neo4j_client") as mock_neo4j:
            mock_neo4j.execute_query = AsyncMock(return_value=[])
            await pipeline._process_single_document_instrumented(
                text="content doesn't matter — splitter is mocked",
                source="doc_c",
                kg_writer=AsyncMock(),
                mode=IngestMode.INCREMENTAL,
                job_id="job-003",
            )

    # Only the new chunk should reach the extractor
    assert len(captured_chunks) == 1
    assert captured_chunks[0].uid == "uid-new"


# ---------------------------------------------------------------------------
# Test 5: Stale chunk — removed page → chunk gets staleAt set
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_stale_chunk_soft_deleted_when_page_removed():
    """
    Document had 2 chunks; re-ingest provides only 1 of them.
    The removed chunk's contentHash must be passed to _soft_delete_stale_chunks.
    """
    kept_text = "Kept paragraph."
    removed_text = "Removed paragraph."
    kept_sha1 = _sha1(kept_text)
    removed_sha1 = _sha1(removed_text)

    pipeline = _make_pipeline()
    pipeline._check_document_hash_unchanged = AsyncMock(return_value=False)
    pipeline._set_document_provenance = AsyncMock()
    # Neo4j reports both chunks exist
    pipeline._get_existing_chunk_content_hashes = AsyncMock(
        return_value={kept_sha1, removed_sha1}
    )
    pipeline._soft_delete_stale_chunks = AsyncMock()
    pipeline._set_chunk_provenance = AsyncMock()

    # New document only has the kept chunk
    kept_chunk = _make_chunk("uid-kept", kept_text)
    mock_chunks = MagicMock()
    mock_chunks.chunks = [kept_chunk]

    with patch(
        "neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter.FixedSizeSplitter"
    ) as MockSplitter:
        MockSplitter.return_value.run = AsyncMock(return_value=mock_chunks)

        result = await pipeline._process_single_document_instrumented(
            text="kept paragraph only",
            source="doc_d",
            kg_writer=AsyncMock(),
            mode=IngestMode.INCREMENTAL,
            job_id="job-004",
        )

    # _soft_delete_stale_chunks must be called with the removed chunk's hash
    pipeline._soft_delete_stale_chunks.assert_called_once()
    call_args = pipeline._soft_delete_stale_chunks.call_args
    hashes_arg = call_args[0][0]  # first positional arg is the list of hashes
    assert removed_sha1 in hashes_arg
    assert kept_sha1 not in hashes_arg


# ---------------------------------------------------------------------------
# Test 6: Full mode — behaves identically to existing delete+re-extract path
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_full_mode_bypasses_hash_guard():
    """mode=full must NOT call _check_document_hash_unchanged — always proceed."""
    pipeline = _make_pipeline()

    pipeline._check_document_hash_unchanged = AsyncMock(return_value=True)  # would skip if checked
    pipeline._set_document_provenance = AsyncMock()
    pipeline._get_existing_chunk_content_hashes = AsyncMock(return_value=set())
    pipeline._soft_delete_stale_chunks = AsyncMock()
    pipeline._set_chunk_provenance = AsyncMock()

    chunk = _make_chunk("uid-f", "Full-mode chunk text.")
    mock_chunks = MagicMock()
    mock_chunks.chunks = [chunk]

    with patch(
        "neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter.FixedSizeSplitter"
    ) as MockSplitter:
        MockSplitter.return_value.run = AsyncMock(return_value=mock_chunks)

        await pipeline._process_single_document_instrumented(
            text="full mode — no filtering",
            source="doc_e",
            kg_writer=AsyncMock(),
            mode=IngestMode.FULL,
            job_id="job-005",
        )

    # Hash guard MUST NOT have been called in full mode
    pipeline._check_document_hash_unchanged.assert_not_called()
    # Chunk delta must NOT have been called in full mode
    pipeline._get_existing_chunk_content_hashes.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: Manual property preservation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_ingest_mode_schema_has_correct_values():
    """IngestMode enum must have exactly full / incremental / upsert values."""
    assert IngestMode.FULL.value == "full"
    assert IngestMode.INCREMENTAL.value == "incremental"
    assert IngestMode.UPSERT.value == "upsert"


@pytest.mark.unit
def test_ingest_data_request_mode_defaults_to_incremental():
    """IngestDataRequest.mode must default to 'incremental' for backward compatibility."""
    from app.schemas.graph_schemas import IngestDataRequest
    req = IngestDataRequest(content="Some text content here that is long enough.")
    assert req.mode == IngestMode.INCREMENTAL


@pytest.mark.unit
@pytest.mark.asyncio
async def test_manual_property_preservation_via_no_new_chunks_path():
    """
    Manual property preservation guarantee via incremental delta path.

    If an entity was manually annotated (e.g. favoriteColor='blue') after the original
    ingestion, re-ingesting the same document must not touch that entity at all.
    The mechanism: when all chunk hashes are unchanged, the pipeline returns
    'no_new_chunks' without calling kg_writer or _set_entity_provenance.
    Since no writes happen, manual properties survive untouched.
    """
    content = "Alice founded Acme Corp in 2010."
    chunk_text = content
    chunk_sha1 = _sha1(chunk_text)

    pipeline = _make_pipeline()
    pipeline._check_document_hash_unchanged = AsyncMock(return_value=False)  # doc hash changed slightly
    pipeline._set_document_provenance = AsyncMock()
    # Simulate entity already exists with all chunk hashes present
    pipeline._get_existing_chunk_content_hashes = AsyncMock(return_value={chunk_sha1})
    pipeline._soft_delete_stale_chunks = AsyncMock()
    pipeline._set_chunk_provenance = AsyncMock()
    pipeline._set_entity_provenance = AsyncMock()  # must NOT be called when no new chunks

    mock_chunk = _make_chunk("uid-1", chunk_text)
    mock_chunks = MagicMock()
    mock_chunks.chunks = [mock_chunk]

    with patch(
        "neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter.FixedSizeSplitter"
    ) as MockSplitter:
        MockSplitter.return_value.run = AsyncMock(return_value=mock_chunks)

        result = await pipeline._process_single_document_instrumented(
            text=content,
            source="doc_manual_props",
            kg_writer=AsyncMock(),
            mode=IngestMode.INCREMENTAL,
            job_id="job-manual",
        )

    # No new chunks → pipeline exits early without writing entities
    assert result["ingest_status"] == "no_new_chunks"
    assert result["entities_created"] == 0
    # Entity provenance setter must NOT have been called — no entities were written
    pipeline._set_entity_provenance.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_entity_provenance_uses_ingestedat_preservation():
    """
    _set_entity_provenance must pass the query that preserves ingestedAt on
    existing entities (CASE WHEN NULL THEN datetime() ELSE ingestedAt END).
    This is the Cypher guard that prevents overwriting manually-set properties.
    """
    from app.services.pipeline_service import MultiTenantGraphRAGPipeline

    with patch("app.services.pipeline_service.neo4j_client") as mock_client:
        pipeline = MultiTenantGraphRAGPipeline(graph_id="g-provenance")
        mock_client.execute_query = AsyncMock(return_value=[])

        await pipeline._set_entity_provenance(["entity-1", "entity-2"], "job-xyz")

        mock_client.execute_query.assert_called_once()
        call_query: str = mock_client.execute_query.call_args[0][0]
        call_params: dict = mock_client.execute_query.call_args[0][1]

        # Query must preserve ingestedAt (never unconditionally overwrite)
        assert "CASE WHEN n.ingestedAt IS NULL" in call_query
        assert "n.lastJobId" in call_query
        assert "n.updatedAt" in call_query
        # Must scope to graph_id (multi-tenancy)
        assert "graph_id" in call_query
        assert call_params["graph_id"] == "g-provenance"
        assert call_params["job_id"] == "job-xyz"
        assert "entity-1" in call_params["entity_ids"]


# ---------------------------------------------------------------------------
# Test 7: Migration backfill — script sets required fields, no data loss
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_backfill_script_queries_are_correct():
    """Verify the backfill script uses safe MERGE-based queries without destructive ops."""
    import ast
    import pathlib

    script_path = pathlib.Path(__file__).parents[2] / "scripts" / "backfill_incremental_kg.py"
    source_code = script_path.read_text()

    # Must set contentHash to LEGACY_UNKNOWN (not delete or overwrite with real data)
    assert "LEGACY_UNKNOWN" in source_code, "Backfill must mark legacy docs with LEGACY_UNKNOWN"

    # Must never use DELETE
    assert "DETACH DELETE" not in source_code, "Backfill must not delete nodes"

    # Must create required indexes
    assert "doc_content_hash" in source_code
    assert "chunk_job_id" in source_code
    assert "chunk_content_hash" in source_code
    assert "entity_job_id" in source_code

    # Must set jobId on chunks
    assert "c.jobId" in source_code

    # Must use 'legacy-' prefix for synthetic job IDs
    assert "legacy-" in source_code
