"""SIMILAR_TO edge generator (STORY-7).

Walks the existing Neo4j vector indexes (``text_embeddings_primary`` on
``:Chunk``, ``entity_embeddings`` on ``:__Entity__``) to find pairs of
nodes whose cosine similarity is above a threshold, and writes a
``:SIMILAR_TO`` edge between them.

Design choices
--------------
- **Undirected pairs**: SIMILAR_TO is semantically undirected but Neo4j
  edges are directed. We dedupe by always storing the edge from the
  node with the lexicographically-smaller id to the larger one
  (``a.id < b.id`` filter). Means agents must traverse in both
  directions when walking SIMILAR_TO.
- **Score on the edge**: stored as ``r.score`` so downstream callers
  (community detection, future find_similar agent tool) can rank by
  similarity strength.
- **Tenant isolation**: ``graph_id`` is filtered on the source node, on
  the neighbour, AND stored on the edge. Three layers because the
  vector index has no per-graph partitioning, so a neighbour from
  another tenant could leak through if we didn't filter.
- **Different thresholds per target**: chunks (long-form text) are
  semantically uniform and we use 0.85; entities (short names) embed
  more noisily and need 0.92 to avoid false-positive merges.
- **`a.id < b.id` ordering** depends on every node having a stable ``id``
  property. :Chunk and :__Entity__ both do — chunks via
  neo4j_graphrag's chunk_id, entities via the deduplicator.

Not in scope for STORY-7
------------------------
- Auto-triggering on ingest — kept explicit via the endpoint. The
  ``SIMILARITY_AUTO_TRIGGER_ON_INGEST`` setting (default False) is a
  hook for a future commit to add pipeline-side scheduling without
  breaking existing ingest performance.
- Reciprocal (``a→b`` AND ``b→a``) edges — single edge per pair only.
- Multi-graph similarity (federation) — that's federation_service's job.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from app.core.config import settings
from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client

logger = get_logger(__name__)

# Expected embedding dimensionality. Matches what ensure_community_vector_indexes
# and the legacy create_vector_indexes script configure for every embedding
# index in the deployment. Nodes whose ``embedding`` doesn't match this
# dimension are filtered out at query time so a single corrupt row doesn't
# crash the whole build (the vector index call raises IllegalArgumentException
# when dimensions disagree).
_EXPECTED_EMBEDDING_DIM = getattr(settings, "EMBEDDING_DIMENSIONS", 3072)


# Target catalog — explicit so adding a new node-type target is one entry.
@dataclass(frozen=True, slots=True)
class _SimilarityTarget:
    label: str
    index_name: str
    default_threshold: float


_TARGETS: dict[str, _SimilarityTarget] = {
    "chunks": _SimilarityTarget(
        label="Chunk",
        index_name="text_embeddings_primary",
        default_threshold=0.85,
    ),
    "entities": _SimilarityTarget(
        label="__Entity__",
        index_name="entity_embeddings",
        default_threshold=0.92,
    ),
}


TargetLiteral = Literal["chunks", "entities", "all"]


@dataclass(slots=True)
class SimilarityReport:
    """Outcome of one build_similarities run."""

    target: str
    chunk_edges_created: int = 0
    entity_edges_created: int = 0
    chunks_processed: int = 0
    entities_processed: int = 0
    elapsed_seconds: float = 0.0
    force_rebuild: bool = False

    def total_edges(self) -> int:
        return self.chunk_edges_created + self.entity_edges_created


class SimilarityService:
    """Compute SIMILAR_TO edges between nodes in the same graph."""

    async def build_similarities(
        self,
        graph_id: str,
        target: TargetLiteral = "all",
        threshold_chunks: float | None = None,
        threshold_entities: float | None = None,
        top_k: int = 5,
        force_rebuild: bool = False,
        job_id: str | None = None,
    ) -> SimilarityReport:
        """Build SIMILAR_TO edges for one or both targets.

        Args:
            graph_id: Tenant graph id (scoped on source, neighbour, and
                edge property for defense-in-depth).
            target: "chunks", "entities", or "all".
            threshold_chunks: Override default 0.85 for chunk pairs.
            threshold_entities: Override default 0.92 for entity pairs.
            top_k: Max neighbours considered per source node. We query
                ``top_k+1`` against the vector index because the source
                node itself comes back at score=1.0.
            force_rebuild: When True, delete all existing SIMILAR_TO
                edges for the target(s) before computing fresh ones.
            job_id: Optional ingestion job id. When provided, the
                resulting edge counts are added to the job's
                ``similarity_relationships`` Postgres column.
        """
        start = time.time()
        report = SimilarityReport(target=target, force_rebuild=force_rebuild)

        targets_to_run: list[str] = (
            ["chunks", "entities"] if target == "all" else [target]
        )
        if target != "all" and target not in _TARGETS:
            raise ValueError(
                f"Unknown similarity target {target!r}. "
                f"Valid: {list(_TARGETS.keys())} or 'all'."
            )

        threshold_overrides = {
            "chunks": threshold_chunks,
            "entities": threshold_entities,
        }

        for tname in targets_to_run:
            spec = _TARGETS[tname]
            t_threshold = (
                threshold_overrides[tname]
                if threshold_overrides[tname] is not None
                else spec.default_threshold
            )

            if force_rebuild:
                await self._delete_existing(graph_id, spec)

            counts = await self._build_for_target(
                graph_id=graph_id,
                spec=spec,
                threshold=t_threshold,
                top_k=top_k,
            )

            if tname == "chunks":
                report.chunk_edges_created = counts["edges_created"]
                report.chunks_processed = counts["nodes_processed"]
            else:
                report.entity_edges_created = counts["edges_created"]
                report.entities_processed = counts["nodes_processed"]

        report.elapsed_seconds = round(time.time() - start, 2)

        # Mirror to the IngestionJob counter if a job_id was provided.
        # Best-effort — failure here doesn't fail the whole call.
        if job_id and report.total_edges() > 0:
            try:
                await self._increment_job_counter(job_id, report.total_edges())
            except Exception as exc:
                logger.warning(
                    "similarity_service: failed to update job_id=%s counter: %s",
                    job_id,
                    exc,
                )

        logger.info(
            "similarity build complete graph_id=%s target=%s "
            "chunks=%d entities=%d elapsed=%.2fs",
            graph_id,
            target,
            report.chunk_edges_created,
            report.entity_edges_created,
            report.elapsed_seconds,
        )
        return report

    # ── private ──────────────────────────────────────────────────────────────

    async def _build_for_target(
        self,
        graph_id: str,
        spec: _SimilarityTarget,
        threshold: float,
        top_k: int,
    ) -> dict[str, int]:
        """Run the vector-search loop for one target label.

        Labels, index names, and property names come from the compile-time
        ``_TARGETS`` catalog — never from request input — so f-string
        interpolation is safe. The ``graph_id`` is parameterized, as are
        threshold and top_k.

        The query is one round trip — Neo4j handles the per-source
        vector lookup and pair-dedup via the ``a.id < b.id`` filter.
        """
        # ``top_k + 1`` because the source node itself comes back at
        # score=1.0 in the vector index result, so we strip it via the
        # elementId-inequality filter.
        #
        # ``size(a.embedding) = $expected_dim`` defends against corrupt
        # rows whose ``embedding`` property holds something other than a
        # proper 3072-dim vector. Without this filter, a single corrupt
        # row would crash the entire build (the vector index raises
        # IllegalArgumentException on dimension mismatch).
        #
        # Pair-dedup uses ``elementId()`` rather than ``a.id`` because not
        # every label has a user-facing ``id`` property (e.g. :Chunk in
        # this deployment only has ``index`` / ``text`` / ``embedding``).
        # elementId is always unique within Neo4j.
        query = (
            f"MATCH (a:`{spec.label}` {{graph_id: $gid}}) "
            "WHERE a.embedding IS NOT NULL "
            "  AND size(a.embedding) = $expected_dim "
            f"CALL db.index.vector.queryNodes('{spec.index_name}', "
            "$top_k_plus_one, a.embedding) "
            "YIELD node AS b, score "
            f"WHERE b:`{spec.label}` AND b.graph_id = $gid "
            "  AND size(b.embedding) = $expected_dim "
            "  AND elementId(a) < elementId(b) "
            "  AND score >= $threshold "
            "MERGE (a)-[r:SIMILAR_TO {graph_id: $gid}]->(b) "
            "ON CREATE SET r.score = score, r.created_at = datetime() "
            "ON MATCH SET r.score = score, r.updated_at = datetime() "
            "RETURN count(DISTINCT r) AS edges_created, "
            "       count(DISTINCT a) AS nodes_processed"
        )
        rows = await neo4j_client.execute_query(
            query,
            {
                "gid": graph_id,
                "top_k_plus_one": top_k + 1,
                "threshold": threshold,
                "expected_dim": _EXPECTED_EMBEDDING_DIM,
            },
        )
        if not rows:
            return {"edges_created": 0, "nodes_processed": 0}
        return {
            "edges_created": rows[0].get("edges_created") or 0,
            "nodes_processed": rows[0].get("nodes_processed") or 0,
        }

    async def _delete_existing(self, graph_id: str, spec: _SimilarityTarget) -> None:
        """Delete all SIMILAR_TO edges for this label + tenant. Used by
        force_rebuild before the fresh pass."""
        query = (
            f"MATCH (:`{spec.label}` {{graph_id: $gid}})"
            "-[r:SIMILAR_TO {graph_id: $gid}]->"
            f"(:`{spec.label}` {{graph_id: $gid}}) "
            "DELETE r"
        )
        await neo4j_client.execute_write_query(query, {"gid": graph_id})

    async def _increment_job_counter(self, job_id: str, increment: int) -> None:
        """Add to ``IngestionJob.similarity_relationships`` for one job."""
        from sqlalchemy import update

        from app.core.database import async_session_maker
        from app.models.graph import IngestionJob

        async with async_session_maker() as session:
            await session.execute(
                update(IngestionJob)
                .where(IngestionJob.id == job_id)
                .values(
                    similarity_relationships=(
                        IngestionJob.similarity_relationships + increment
                    )
                )
            )
            await session.commit()


# Module-level singleton — service is stateless.
similarity_service = SimilarityService()
