"""Entity & relationship deduplication service (STORY-6).

Post-ingestion cleanup: consolidates duplicate :__Entity__ nodes and
collapses parallel relationships into single edges with a ``count``
property. The pipeline's writer is inside neo4j_graphrag's
``SimpleKGPipeline`` (we don't control its MERGE shapes), so the
practical equivalent of "prevent duplication at ingest" is a service
that runs immediately after the pipeline — same pattern as STORY-7's
``SimilarityService``.

The service runs four passes (each opt-in via the request body):

1. ``canonical`` — Populate ``canonical_name`` on every entity missing
   one. Lowercase + strip common corporate suffixes (B.V., GmbH, Ltd,
   .com, Group, etc.). Pure enrichment — no merges, no deletes.

2. ``merge_canonical`` — Group entities by ``canonical_name`` within a
   graph. When N > 1: pick one canonical (lowest elementId for
   determinism), rebind every incoming and outgoing edge to it,
   accumulate the discarded names as ``aliases``, delete the duplicates.

3. ``embedding`` — For each entity with a valid 3072-dim embedding,
   query the ``entity_embeddings`` vector index for nearest neighbours.
   Pairs above the threshold AND not yet sharing a canonical group are
   merged via the same rebind-and-delete pattern as pass 2.

4. ``relationships`` — For each ``(a)-[r:TYPE]->(b)`` triple with
   multiple edges of the same TYPE, reduce to one edge carrying
   ``count = N``, ``first_seen``, ``last_seen``. Skips structural
   relationships (``:SIMILAR_TO``, ``:IN_COMMUNITY``, ``:FROM_CHUNK``,
   ``:PARENT_COMMUNITY``, ``:IN_CHUNK_COMMUNITY``) that legitimately
   have multiple instances or carry semantic meaning per instance.

All four passes are idempotent: re-running produces zero new actions
once the graph is fully deduped. ``dry_run=True`` reports what would
happen without writing.

The legacy ``MultiTenantEntityDeduplicator`` (in components/entity_resolver)
remains for the pipeline's automatic post-write step. STORY-6 doesn't
remove it; it adds a stronger on-demand path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from app.core.config import settings
from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client

logger = get_logger(__name__)


# Word-boundary corporate suffixes — must be preceded by whitespace OR be
# the entire string. Prevents "GroupMe" from losing "Group". All compared
# case-insensitively.
_CORPORATE_SUFFIXES: tuple[str, ...] = (
    "b.v.",
    "bv",
    "n.v.",
    "nv",
    "gmbh",
    "ltd.",
    "ltd",
    "inc.",
    "inc",
    "corp.",
    "corp",
    "corporation",
    "llc",
    "l.l.c.",
    "group",
    "holding",
    "holdings",
)

# Attached suffixes — strip when at end of string without requiring a
# leading space. ``.com``, ``.io`` etc. don't have a word boundary so
# the space-prefix rule above doesn't catch them.
_ATTACHED_SUFFIXES: tuple[str, ...] = (
    ".com",
    ".io",
    ".net",
    ".org",
)

# Relationship types that legitimately have multiple parallel instances
# between the same nodes. The relationship-consolidation pass skips these.
_STRUCTURAL_REL_TYPES: tuple[str, ...] = (
    "SIMILAR_TO",
    "IN_COMMUNITY",
    "FROM_CHUNK",
    "PARENT_COMMUNITY",
    "IN_CHUNK_COMMUNITY",
    "NEXT_CHUNK",
    "FROM_DOCUMENT",
)


_EMBEDDING_DIM = getattr(settings, "EMBEDDING_DIMENSIONS", 3072)


PassLiteral = Literal["canonical", "merge_canonical", "embedding", "relationships"]


@dataclass(slots=True)
class DedupReport:
    """Outcome of running deduplicate() for one graph."""

    passes_run: list[str] = field(default_factory=list)
    canonical_names_added: int = 0
    canonical_merges: int = 0
    embedding_merges: int = 0
    relationship_consolidations: int = 0
    entities_processed: int = 0
    relationships_processed: int = 0
    skipped_bad_names: int = 0
    elapsed_seconds: float = 0.0
    dry_run: bool = False


class EntityDeduplicationService:
    """Post-ingestion entity + relationship cleanup."""

    async def deduplicate(
        self,
        graph_id: str,
        passes: list[PassLiteral] | None = None,
        fuzzy_threshold: float = 0.92,
        dry_run: bool = False,
    ) -> DedupReport:
        """Run the requested dedup passes against the graph.

        Args:
            graph_id: Tenant graph id (every Cypher is scoped to it).
            passes: Subset of {"canonical", "merge_canonical", "embedding",
                "relationships"}. None means all four.
            fuzzy_threshold: Cosine threshold for the embedding pass.
                Default 0.92 — entity-name embeddings false-positive
                easily at lower thresholds.
            dry_run: When True, no writes happen; the report reflects
                what each pass WOULD do.
        """
        start = time.time()
        report = DedupReport(dry_run=dry_run)
        all_passes: list[PassLiteral] = [
            "canonical",
            "merge_canonical",
            "embedding",
            "relationships",
        ]
        passes_to_run = passes or all_passes
        report.passes_run = list(passes_to_run)

        if "canonical" in passes_to_run:
            added, skipped = await self._pass_canonical(graph_id, dry_run)
            report.canonical_names_added = added
            report.skipped_bad_names = skipped

        if "merge_canonical" in passes_to_run:
            report.canonical_merges = await self._pass_merge_canonical(
                graph_id, dry_run
            )

        if "embedding" in passes_to_run:
            report.embedding_merges = await self._pass_embedding(
                graph_id, fuzzy_threshold, dry_run
            )

        if "relationships" in passes_to_run:
            report.relationship_consolidations = await self._pass_relationships(
                graph_id, dry_run
            )

        # Counters that frame the work done.
        report.entities_processed = await self._count_entities(graph_id)
        report.relationships_processed = await self._count_relationships(graph_id)
        report.elapsed_seconds = round(time.time() - start, 2)
        logger.info(
            "entity dedup complete graph_id=%s passes=%s "
            "canonical=%d merge_canonical=%d embedding=%d rel=%d "
            "dry_run=%s elapsed=%.2fs",
            graph_id,
            passes_to_run,
            report.canonical_names_added,
            report.canonical_merges,
            report.embedding_merges,
            report.relationship_consolidations,
            dry_run,
            report.elapsed_seconds,
        )
        return report

    # ── Pass 1: canonical-name normalization ─────────────────────────────────

    @staticmethod
    def compute_canonical_name(raw_name: Any) -> str | None:
        """Compute a canonical form of an entity name.

        Returns None for unusable inputs (None, non-string, empty after
        normalization). Tolerates StringArray-corrupted entries by
        picking the first element. Lowercases, strips a configurable
        set of corporate suffixes, collapses whitespace.
        """
        # Tolerate corruption: name field is sometimes a list when the
        # extractor merged variants incorrectly.
        if isinstance(raw_name, list | tuple):
            raw_name = next(
                (x for x in raw_name if isinstance(x, str) and x.strip()), None
            )
        if not isinstance(raw_name, str):
            return None
        s = raw_name.strip()
        if not s:
            return None
        # Lowercase + collapse interior whitespace
        s = " ".join(s.lower().split())
        # Suffix stripping — repeat until no suffix matches (handles
        # double-suffix cases like "Eurail Group B.V.")
        changed = True
        while changed:
            changed = False
            # Word-boundary suffixes (require space before, or whole-string match)
            for suffix in _CORPORATE_SUFFIXES:
                if s.endswith(" " + suffix) or s == suffix:
                    s = s[: -len(suffix)].rstrip()
                    changed = True
                    break
            if changed:
                continue
            # Attached suffixes (no space required)
            for suffix in _ATTACHED_SUFFIXES:
                if s.endswith(suffix) and len(s) > len(suffix):
                    s = s[: -len(suffix)].rstrip()
                    changed = True
                    break
        return s or None

    async def _pass_canonical(self, graph_id: str, dry_run: bool) -> tuple[int, int]:
        """Populate canonical_name on every entity missing one.

        Returns (added, skipped_bad_names).
        """
        rows = await neo4j_client.execute_query(
            (
                "MATCH (e:__Entity__ {graph_id: $gid}) "
                "WHERE e.canonical_name IS NULL "
                "RETURN elementId(e) AS eid, e.name AS name"
            ),
            {"gid": graph_id},
        )
        added = 0
        skipped = 0
        for row in rows:
            cn = self.compute_canonical_name(row["name"])
            if cn is None:
                skipped += 1
                continue
            added += 1
            if dry_run:
                continue
            await neo4j_client.execute_query(
                "MATCH (e:__Entity__) WHERE elementId(e) = $eid SET e.canonical_name = $cn",
                {"eid": row["eid"], "cn": cn},
            )
        return added, skipped

    # ── Pass 2: merge by canonical_name ──────────────────────────────────────

    async def _pass_merge_canonical(self, graph_id: str, dry_run: bool) -> int:
        """For each canonical_name with multiple entities, consolidate
        into one. Returns the number of duplicates that were merged
        away (i.e. the count of deleted entities)."""
        groups = await neo4j_client.execute_query(
            (
                "MATCH (e:__Entity__ {graph_id: $gid}) "
                "WHERE e.canonical_name IS NOT NULL "
                "WITH e.canonical_name AS cn, "
                "     collect(elementId(e)) AS eids "
                "WHERE size(eids) > 1 "
                "RETURN cn, eids"
            ),
            {"gid": graph_id},
        )
        merged_count = 0
        for grp in groups:
            eids = grp["eids"]
            if len(eids) < 2:
                continue
            # Deterministic winner = lexicographically lowest elementId
            sorted_eids = sorted(eids)
            keeper = sorted_eids[0]
            losers = sorted_eids[1:]
            if dry_run:
                merged_count += len(losers)
                continue
            try:
                deleted = await self._consolidate_into_keeper(
                    graph_id, keeper, losers, source="canonical"
                )
                merged_count += deleted
            except Exception as exc:
                logger.warning(
                    "merge_canonical failed for canonical_name=%r: %s",
                    grp["cn"],
                    exc,
                )
        return merged_count

    # ── Pass 3: embedding-based fuzzy merge ──────────────────────────────────

    async def _pass_embedding(
        self, graph_id: str, threshold: float, dry_run: bool
    ) -> int:
        """For each entity with a 3072-dim embedding, find vector-index
        neighbors above ``threshold`` that share neither entity id nor
        canonical_name with it. Merge each such pair into the lower-
        elementId entity."""
        # Single Cypher walks the index, dedups pairs via elementId
        # ordering, and emits keeper/loser ids. Then we batch the merges
        # client-side so we can detect transitive duplicates.
        rows = await neo4j_client.execute_query(
            (
                "MATCH (a:__Entity__ {graph_id: $gid}) "
                "WHERE a.embedding IS NOT NULL "
                "  AND size(a.embedding) = $expected_dim "
                "CALL db.index.vector.queryNodes('entity_embeddings', "
                "$top_k_plus_one, a.embedding) "
                "YIELD node AS b, score "
                "WHERE b:__Entity__ AND b.graph_id = $gid "
                "  AND size(b.embedding) = $expected_dim "
                "  AND elementId(a) < elementId(b) "
                "  AND score >= $threshold "
                "  AND (a.canonical_name IS NULL "
                "       OR b.canonical_name IS NULL "
                "       OR a.canonical_name <> b.canonical_name) "
                "RETURN elementId(a) AS keeper_eid, "
                "       elementId(b) AS loser_eid, score "
                "ORDER BY score DESC"
            ),
            {
                "gid": graph_id,
                "top_k_plus_one": 6,
                "threshold": threshold,
                "expected_dim": _EMBEDDING_DIM,
            },
        )
        merged_count = 0
        # Track entities already merged so we don't try to merge a
        # loser into something that itself got merged away.
        merged_away: set[str] = set()
        for r in rows:
            keeper_eid = r["keeper_eid"]
            loser_eid = r["loser_eid"]
            if keeper_eid in merged_away or loser_eid in merged_away:
                continue
            if dry_run:
                merged_count += 1
                merged_away.add(loser_eid)
                continue
            try:
                deleted = await self._consolidate_into_keeper(
                    graph_id, keeper_eid, [loser_eid], source="embedding"
                )
                merged_count += deleted
                merged_away.add(loser_eid)
            except Exception as exc:
                logger.warning(
                    "embedding merge failed (keeper=%s loser=%s): %s",
                    keeper_eid,
                    loser_eid,
                    exc,
                )
        return merged_count

    # ── Pass 4: parallel-relationship consolidation ──────────────────────────

    async def _pass_relationships(self, graph_id: str, dry_run: bool) -> int:
        """Consolidate parallel relationships between the same two
        entities (same TYPE) into a single edge with ``count``.

        Returns the number of duplicate edges that were removed (i.e.
        for a triple with N parallel edges, this contributes N-1).
        """
        # Excluded types live as a Cypher literal list — compile-time
        # constants, safe to inline.
        excluded = ", ".join(f"'{t}'" for t in _STRUCTURAL_REL_TYPES)
        rows = await neo4j_client.execute_query(
            (
                f"MATCH (a:__Entity__ {{graph_id: $gid}})-[r]->(b:__Entity__ {{graph_id: $gid}}) "
                f"WHERE NOT type(r) IN [{excluded}] "
                "WITH a, b, type(r) AS rel_type, collect(r) AS rels "
                "WHERE size(rels) > 1 "
                "RETURN elementId(a) AS a_eid, elementId(b) AS b_eid, "
                "       rel_type, "
                "       [rel IN rels | elementId(rel)] AS rel_eids, "
                "       size(rels) AS dup_count"
            ),
            {"gid": graph_id},
        )
        removed = 0
        for r in rows:
            rel_eids = r["rel_eids"]
            if len(rel_eids) < 2:
                continue
            # Keep the lowest-elementId edge; delete the rest; bump
            # count on the keeper.
            sorted_rel_eids = sorted(rel_eids)
            keeper_rel = sorted_rel_eids[0]
            loser_rels = sorted_rel_eids[1:]
            removed += len(loser_rels)
            if dry_run:
                continue
            try:
                await neo4j_client.execute_query(
                    (
                        "MATCH ()-[r]->() WHERE elementId(r) = $keeper "
                        "SET r.count = coalesce(r.count, 1) + $delta, "
                        "    r.last_seen = datetime()"
                    ),
                    {"keeper": keeper_rel, "delta": len(loser_rels)},
                )
                await neo4j_client.execute_query(
                    "MATCH ()-[r]->() WHERE elementId(r) IN $eids DELETE r",
                    {"eids": loser_rels},
                )
            except Exception as exc:
                logger.warning(
                    "relationship consolidation failed (keeper=%s): %s",
                    keeper_rel,
                    exc,
                )
        return removed

    # ── Helpers ──────────────────────────────────────────────────────────────

    async def _consolidate_into_keeper(
        self,
        graph_id: str,
        keeper_eid: str,
        loser_eids: list[str],
        source: str,
    ) -> int:
        """Rebind every edge of each loser to the keeper, accumulate
        names into the keeper's ``aliases`` list, then delete losers.
        Returns the count of losers actually deleted.

        ``source`` is recorded in a structured log so post-hoc analysis
        can see whether canonical or embedding triggered the merge.
        """
        deleted = 0
        for loser_eid in loser_eids:
            # Skip self-merge (same elementId)
            if loser_eid == keeper_eid:
                continue
            # Accumulate alias
            await neo4j_client.execute_query(
                (
                    "MATCH (keeper), (loser) "
                    "WHERE elementId(keeper) = $keeper AND elementId(loser) = $loser "
                    "WITH keeper, loser, "
                    "     coalesce(keeper.aliases, []) AS prev_aliases, "
                    "     loser.name AS loser_name "
                    "SET keeper.aliases = CASE "
                    "      WHEN loser_name IS NOT NULL AND NOT loser_name IN prev_aliases "
                    "        THEN prev_aliases + [loser_name] "
                    "      ELSE prev_aliases END"
                ),
                {"keeper": keeper_eid, "loser": loser_eid},
            )
            # Move incoming edges
            await neo4j_client.execute_query(
                (
                    "MATCH (other)-[r]->(loser), (keeper) "
                    "WHERE elementId(loser) = $loser AND elementId(keeper) = $keeper "
                    "WITH other, r, keeper, type(r) AS rt, properties(r) AS props "
                    "CALL apoc.create.relationship(other, rt, props, keeper) YIELD rel "
                    "DELETE r "
                    "RETURN count(rel) AS moved"
                ),
                {"loser": loser_eid, "keeper": keeper_eid},
            )
            # Move outgoing edges
            await neo4j_client.execute_query(
                (
                    "MATCH (loser)-[r]->(other), (keeper) "
                    "WHERE elementId(loser) = $loser AND elementId(keeper) = $keeper "
                    "WITH r, other, keeper, type(r) AS rt, properties(r) AS props "
                    "CALL apoc.create.relationship(keeper, rt, props, other) YIELD rel "
                    "DELETE r "
                    "RETURN count(rel) AS moved"
                ),
                {"loser": loser_eid, "keeper": keeper_eid},
            )
            # Delete the loser
            await neo4j_client.execute_query(
                "MATCH (loser) WHERE elementId(loser) = $loser DETACH DELETE loser",
                {"loser": loser_eid},
            )
            deleted += 1
        if deleted > 0:
            logger.info(
                "consolidated %d entities into keeper=%s via %s (graph_id=%s)",
                deleted,
                keeper_eid,
                source,
                graph_id,
            )
        return deleted

    async def _count_entities(self, graph_id: str) -> int:
        rows = await neo4j_client.execute_query(
            "MATCH (e:__Entity__ {graph_id: $gid}) RETURN count(e) AS n",
            {"gid": graph_id},
        )
        return rows[0]["n"] if rows else 0

    async def _count_relationships(self, graph_id: str) -> int:
        rows = await neo4j_client.execute_query(
            "MATCH (:__Entity__ {graph_id: $gid})-[r]->(:__Entity__ {graph_id: $gid}) "
            "RETURN count(r) AS n",
            {"gid": graph_id},
        )
        return rows[0]["n"] if rows else 0


# Module-level singleton — service is stateless.
entity_dedup_service = EntityDeduplicationService()
