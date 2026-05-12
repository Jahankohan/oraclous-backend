"""
Entity Deduplication Component for Neo4j GraphRAG Pipeline

Native component that integrates into the neo4j_graphrag pipeline to consolidate
duplicate entities across chunks into single entities with multiple chunk relationships.

This approach creates a cleaner graph by:
1. Identifying duplicate entities across chunks
2. Merging them into a single canonical entity
3. Linking the canonical entity to all relevant chunks
4. Removing duplicate entity nodes

Also provides EntityResolver (TASK-010): a four-signal scorer that combines
embedding similarity, name similarity, type compatibility, and context overlap
to produce a final confidence score for SAME_AS candidates, then creates
SAME_AS links for scores above the store threshold.
"""

import re
import time
from typing import Any

import jellyfish
from neo4j import AsyncSession, Driver, Session
from neo4j.graph import Node
from neo4j_graphrag.experimental.components.types import Neo4jGraph
from neo4j_graphrag.experimental.pipeline.component import Component

from app.core.logging import get_logger
from app.schemas.federation_schemas import SameAsCandidate
from app.services.llm_service import disambiguate_entities

logger = get_logger(__name__)

# Allowlisted relationship types for deduplication fallback — prevents Cypher injection
# via runtime rel_type values passed to apoc.cypher.doIt() or string concatenation.
# Must cover all types produced by the LLM extractor prompt in pipeline_service.py
# (RELATIONSHIP_PROPERTY_PROMPT_TEMPLATE) plus any structural types used elsewhere.
_ALLOWED_REL_TYPES: frozenset[str] = frozenset(
    {
        # ── LLM extractor types (pipeline_service.py RELATIONSHIP_PROPERTY_PROMPT_TEMPLATE) ──
        "WORKS_FOR",
        "REPORTS_TO",
        "HAS_SKILL",
        "MEMBER_OF",
        "INVESTED_IN",
        "CITES",
        "AUTHORED",
        "WORKS_ON",
        "DEPENDS_ON",
        "ACQUIRED_BY",
        "PARTNER_OF",
        # ── Additional types used elsewhere in the codebase ──
        "FOUNDED",
        "LEADS",
        "MANAGES",
        "DEVELOPED",
        "RELATED_TO",
        "PART_OF",
        "OWNS",
        "LOCATED_IN",
        "SAME_AS",
        "SIMILAR_TO",
    }
)

# ── EntityResolver — four-signal SAME_AS scorer ───────────────────────────────

# Scoring weights (must sum to 1.0)
EMBEDDING_WEIGHT = 0.4
NAME_WEIGHT = 0.3
TYPE_WEIGHT = 0.2
CONTEXT_WEIGHT = 0.1

# Decision thresholds
STORE_THRESHOLD = 0.85  # create SAME_AS link immediately
AMBIGUOUS_LOWER = 0.60  # pass to LLM disambiguation (TASK-011)

# Name length cap — prevents CPU DoS via extremely long names before regex/Jaro-Winkler
_MAX_NAME_LEN = 1000

# Compatible type pairs (order-independent) that receive a partial type score
_COMPATIBLE_TYPE_PAIRS: frozenset[frozenset[str]] = frozenset(
    [
        frozenset({"organization", "company"}),
        frozenset({"person", "individual"}),
        frozenset({"location", "place"}),
    ]
)

# Legal-suffix tokens stripped during name normalization
_LEGAL_SUFFIX_RE = re.compile(
    r"\b(inc|ltd|corp|llc|co|limited|incorporated|corporation|plc|gmbh|sa|srl|bv|nv|ag)\b\.?",
    re.IGNORECASE,
)

# Punctuation strip pattern (keeps spaces and alphanumeric)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, remove common legal suffixes."""
    name = name[:_MAX_NAME_LEN]
    name = name.lower()
    name = _LEGAL_SUFFIX_RE.sub("", name)
    name = _PUNCT_RE.sub("", name)
    return " ".join(name.split())  # collapse whitespace


class EntityResolver:
    """Four-signal scorer for cross-graph SAME_AS candidate resolution.

    Signals and weights:
        embedding_similarity * 0.4  — cosine similarity from vector search
        name_similarity      * 0.3  — Jaro-Winkler on normalized names
        type_compatibility   * 0.2  — exact / compatible / incompatible type pairs
        context_overlap      * 0.1  — Jaccard of 1-hop neighbor names

    Decision thresholds:
        final_score >= STORE_THRESHOLD (0.85)          → create SAME_AS link
        AMBIGUOUS_LOWER (0.60) <= score < 0.85         → log for TASK-011
        score < AMBIGUOUS_LOWER                        → discard
    """

    # ── Signal: embedding similarity ─────────────────────────────────────────

    @staticmethod
    def _embedding_score(candidate: SameAsCandidate) -> float:
        """Return the cosine similarity score from the vector search (already in [0,1])."""
        return float(candidate["score"])

    # ── Signal: name similarity ───────────────────────────────────────────────

    @staticmethod
    def _name_score(entity_a: dict, entity_b: dict) -> float:
        """Jaro-Winkler similarity on normalized entity names."""
        name_a = _normalize_name(entity_a.get("name") or "")
        name_b = _normalize_name(entity_b.get("name") or "")
        if not name_a or not name_b:
            return 0.0
        return jellyfish.jaro_winkler_similarity(name_a, name_b)

    # ── Signal: type compatibility ────────────────────────────────────────────

    @staticmethod
    def _type_score(entity_a: dict, entity_b: dict) -> float:
        """1.0 for same type, 0.5 for compatible pairs, 0.0 otherwise."""
        type_a = (entity_a.get("type") or "").strip().lower()
        type_b = (entity_b.get("type") or "").strip().lower()
        if not type_a or not type_b:
            return 0.0
        if type_a == type_b:
            return 1.0
        pair = frozenset({type_a, type_b})
        if pair in _COMPATIBLE_TYPE_PAIRS:
            return 0.5
        return 0.0

    # ── Signal: context overlap ───────────────────────────────────────────────

    @staticmethod
    async def _context_score(
        entity_a: dict,
        entity_b: dict,
        session: AsyncSession,
        graph_id_a: str,
        graph_id_b: str,
    ) -> float:
        """Jaccard similarity of 1-hop neighbor entity names.

        Returns 0.0 (no ZeroDivisionError) when either entity has no neighbors.
        """
        neighbors_a = await EntityResolver._get_neighbor_names(
            session, entity_a.get("entity_id", ""), graph_id_a
        )
        neighbors_b = await EntityResolver._get_neighbor_names(
            session, entity_b.get("entity_id", ""), graph_id_b
        )
        if not neighbors_a or not neighbors_b:
            return 0.0
        intersection = neighbors_a & neighbors_b
        union = neighbors_a | neighbors_b
        if not union:
            return 0.0
        return len(intersection) / len(union)

    @staticmethod
    async def _get_neighbor_names(
        session: AsyncSession, entity_id: str, graph_id: str
    ) -> set[str]:
        """Return the set of normalized neighbor entity names for a given entity."""
        if not entity_id:
            return set()
        query = """
        MATCH (e:__Entity__)
        WHERE elementId(e) = $entity_id AND e.graph_id = $graph_id
        MATCH (e)-[]->(neighbor:__Entity__)
        WHERE neighbor.graph_id = $graph_id
        RETURN neighbor.name AS name
        UNION
        MATCH (e:__Entity__)
        WHERE elementId(e) = $entity_id AND e.graph_id = $graph_id
        MATCH (neighbor:__Entity__)-[]->(e)
        WHERE neighbor.graph_id = $graph_id
        RETURN neighbor.name AS name
        """
        try:
            result = await session.run(
                query, {"entity_id": entity_id, "graph_id": graph_id}
            )
            rows = await result.data()
            return {_normalize_name(row["name"]) for row in rows if row["name"]}
        except Exception as exc:
            logger.warning(
                "failed to fetch neighbor names for entity %s: %s", entity_id, exc
            )
            return set()

    # ── Composite score ───────────────────────────────────────────────────────

    @staticmethod
    async def score(
        entity_a: dict,
        candidate: SameAsCandidate,
        session: AsyncSession,
        graph_id_a: str,
        graph_id_b: str,
    ) -> float:
        """Return the weighted four-signal confidence score.

        Weights: embedding=0.4, name=0.3, type=0.2, context=0.1  (sum=1.0)
        """
        entity_b = candidate["entity"]
        embedding_sim = EntityResolver._embedding_score(candidate)
        name_sim = EntityResolver._name_score(entity_a, entity_b)
        type_compat = EntityResolver._type_score(entity_a, entity_b)
        ctx_overlap = await EntityResolver._context_score(
            entity_a, entity_b, session, graph_id_a, graph_id_b
        )
        return (
            embedding_sim * EMBEDDING_WEIGHT
            + name_sim * NAME_WEIGHT
            + type_compat * TYPE_WEIGHT
            + ctx_overlap * CONTEXT_WEIGHT
        )

    # ── Resolution & link creation ────────────────────────────────────────────

    @staticmethod
    async def resolve_and_link(
        entity_a: dict,
        candidates: list[SameAsCandidate],
        session: AsyncSession,
        graph_id_a: str,
        target_graph_ids: list[str],
    ) -> list[dict]:
        """Score each candidate and create SAME_AS links for high-confidence pairs.

        For each candidate:
          - final_score >= STORE_THRESHOLD (0.85): create SAME_AS with method='multi-signal'
          - AMBIGUOUS_LOWER (0.60) <= final_score < 0.85: send to LLM disambiguation
            - YES + HIGH   → SAME_AS at original score, method='llm-disambiguated'
            - YES + MEDIUM → SAME_AS at score * 0.9, method='llm-disambiguated'
            - NO / LOW     → skip (appended to returned ambiguous list for audit)
          - final_score < AMBIGUOUS_LOWER: discard

        Returns a list of dicts for candidates in the ambiguous zone that the LLM
        rejected (or that could not be resolved), for audit / downstream use.
        """
        ambiguous: list[dict] = []

        for candidate in candidates:
            entity_b = candidate["entity"]
            graph_id_b = entity_b.get("source_graph_id")
            if not graph_id_b:
                logger.warning(
                    "skipping candidate with missing source_graph_id: %r",
                    entity_b.get("entity_id", "?"),
                )
                continue

            final_score = await EntityResolver.score(
                entity_a, candidate, session, graph_id_a, graph_id_b
            )

            if final_score >= STORE_THRESHOLD:
                await EntityResolver._create_same_as_link(
                    session,
                    entity_a.get("entity_id", ""),
                    entity_b.get("entity_id", ""),
                    final_score,
                    graph_id_a=graph_id_a,
                    graph_id_b=graph_id_b,
                )
            elif final_score >= AMBIGUOUS_LOWER:
                # LLM disambiguation for the 0.60–0.85 ambiguous zone
                logger.info(
                    "ambiguous candidate — sending to LLM: %s <-> %s score=%.3f",
                    entity_a.get("name", entity_a.get("entity_id", "?")),
                    entity_b.get("name", entity_b.get("entity_id", "?")),
                    final_score,
                )
                should_link, multiplier = await EntityResolver._llm_disambiguate(
                    entity_a, entity_b, session, graph_id_a, graph_id_b
                )
                if should_link:
                    effective_score = final_score * multiplier
                    await EntityResolver._create_same_as_link(
                        session,
                        entity_a.get("entity_id", ""),
                        entity_b.get("entity_id", ""),
                        effective_score,
                        graph_id_a=graph_id_a,
                        graph_id_b=graph_id_b,
                        method="llm-disambiguated",
                    )
                else:
                    logger.info(
                        "LLM rejected SAME_AS: %s <-> %s score=%.3f",
                        entity_a.get("name", entity_a.get("entity_id", "?")),
                        entity_b.get("name", entity_b.get("entity_id", "?")),
                        final_score,
                    )
                    ambiguous.append(
                        {
                            "entity_a": entity_a,
                            "entity_b": entity_b,
                            "score": final_score,
                            "graph_id_a": graph_id_a,
                            "graph_id_b": graph_id_b,
                        }
                    )
            # else: discard — below AMBIGUOUS_LOWER

        return ambiguous

    @staticmethod
    async def _create_same_as_link(
        session: AsyncSession,
        id_a: str,
        id_b: str,
        score: float,
        graph_id_a: str,
        graph_id_b: str,
        method: str = "multi-signal",
    ) -> None:
        """Persist a bidirectional SAME_AS relationship using idempotent MERGE.

        graph_id constraints on both MATCH clauses prevent cross-tenant writes:
        a MATCH that mismatches graph_id returns no row and the MERGE is skipped.
        """
        query = """
        MATCH (a:__Entity__ {graph_id: $graph_id_a}) WHERE elementId(a) = $id_a
        MATCH (b:__Entity__ {graph_id: $graph_id_b}) WHERE elementId(b) = $id_b
        MERGE (a)-[:SAME_AS {confidence: $score, method: $method, created_at: datetime()}]->(b)
        MERGE (b)-[:SAME_AS {confidence: $score, method: $method, created_at: datetime()}]->(a)
        """
        try:

            async def _write(tx) -> None:
                await tx.run(
                    query,
                    {
                        "id_a": id_a,
                        "id_b": id_b,
                        "score": score,
                        "method": method,
                        "graph_id_a": graph_id_a,
                        "graph_id_b": graph_id_b,
                    },
                )

            await session.execute_write(_write)
            logger.info(
                "created SAME_AS link: %s <-> %s confidence=%.3f method=%s",
                id_a,
                id_b,
                score,
                method,
            )
        except Exception as exc:
            logger.warning(
                "failed to create SAME_AS link %s <-> %s: %s", id_a, id_b, exc
            )

    # ── LLM disambiguation for ambiguous candidates ───────────────────────────

    @staticmethod
    async def _fetch_neighbor_names_for_context(
        session: AsyncSession,
        entity_id: str,
        graph_id: str,
    ) -> list[str]:
        """Fetch up to 3 outgoing neighbor names for LLM context.

        Uses parameterized Cypher with explicit graph_id filtering on both
        anchor and neighbor nodes — tenant isolation is enforced at the query level.
        The returned names are only used in the LLM prompt string; they are never
        interpolated into Cypher query text.
        """
        if not entity_id:
            return []
        query = """
        MATCH (e {graph_id: $graph_id})
        WHERE elementId(e) = $entity_id
        MATCH (e)-[]->(n:__Entity__ {graph_id: $graph_id})
        RETURN n.name AS name
        LIMIT 3
        """
        try:
            result = await session.run(
                query, {"graph_id": graph_id, "entity_id": entity_id}
            )
            rows = await result.data()
            return [row["name"] for row in rows if row.get("name")]
        except Exception as exc:
            logger.warning(
                "_fetch_neighbor_names_for_context failed for entity %s: %s",
                entity_id,
                exc,
            )
            return []

    @staticmethod
    async def _llm_disambiguate(
        entity_a: dict,
        entity_b: dict,
        session: AsyncSession,
        graph_id_a: str,
        graph_id_b: str,
    ) -> tuple[bool, float]:
        """Ask the LLM whether entity_a and entity_b are the same real-world entity.

        Returns (should_link, score_multiplier):
            (True, 1.0)   — YES + HIGH confidence
            (True, 0.9)   — YES + MEDIUM confidence
            (False, 0.0)  — NO, or YES + LOW, or any error (fail-safe)
        """
        context_a = await EntityResolver._fetch_neighbor_names_for_context(
            session, entity_a.get("entity_id", ""), graph_id_a
        )
        context_b = await EntityResolver._fetch_neighbor_names_for_context(
            session, entity_b.get("entity_id", ""), graph_id_b
        )

        result = await disambiguate_entities(
            name_a=entity_a.get("name") or "",
            type_a=entity_a.get("type") or "",
            context_a=context_a,
            name_b=entity_b.get("name") or "",
            type_b=entity_b.get("type") or "",
            context_b=context_b,
        )

        decision = result.get("decision", "NO")
        confidence = result.get("confidence", "LOW")

        if decision == "YES" and confidence == "HIGH":
            return True, 1.0
        if decision == "YES" and confidence == "MEDIUM":
            return True, 0.9
        return False, 0.0


class MultiTenantEntityDeduplicator(Component):
    """
    Native neo4j_graphrag component for entity deduplication with multi-tenant support.

    This component:
    1. Finds entities with identical names across different chunks
    2. Merges duplicate entities into a single canonical entity
    3. Links the canonical entity to all chunks where it appears
    4. Removes duplicate entity nodes
    5. Preserves all relationships and properties
    6. Supports multi-tenant isolation using graph_id
    """

    def __init__(
        self,
        driver: Driver,
        graph_id: str,
        similarity_threshold: float = 0.85,
        enable_fuzzy_matching: bool = False,
        neo4j_database: str | None = None,
    ):
        """
        Initialize the entity deduplicator component.

        Args:
            driver: Neo4j driver instance
            graph_id: Tenant graph identifier for multi-tenant isolation
            similarity_threshold: Threshold for similarity matching (0.0-1.0)
            enable_fuzzy_matching: Whether to enable fuzzy string matching
            neo4j_database: Neo4j database name (optional)
        """
        super().__init__()
        self.driver = driver
        self.graph_id = graph_id
        self.similarity_threshold = similarity_threshold
        self.enable_fuzzy_matching = enable_fuzzy_matching
        self.neo4j_database = neo4j_database

    async def run(self, graph: Neo4jGraph) -> Neo4jGraph:
        """
        Run entity deduplication on the graph and return the updated graph.

        This method follows the neo4j_graphrag component pattern:
        - Takes a Neo4jGraph as input
        - Processes the graph data by consolidating duplicate entities
        - Returns the modified graph

        Args:
            graph: The Neo4jGraph containing entities and relationships

        Returns:
            The same graph (deduplication happens in-database)
        """
        logger.info(f"🔄 Running entity deduplication for graph {self.graph_id}")

        start_time = time.time()

        try:
            # Step 1: Exact name matching (most common case)
            exact_deduplications = await self._deduplicate_exact_matches()

            # Step 2: Fuzzy matching (if enabled)
            fuzzy_deduplications = 0
            if self.enable_fuzzy_matching:
                fuzzy_deduplications = await self._deduplicate_fuzzy_matches()

            total_deduplicated = exact_deduplications + fuzzy_deduplications
            duration = time.time() - start_time

            logger.info(
                f"✅ Entity deduplication completed for graph {self.graph_id}: "
                f"{total_deduplicated} entities deduplicated in {duration:.2f}s "
                f"(exact: {exact_deduplications}, fuzzy: {fuzzy_deduplications})"
            )

        except Exception as e:
            logger.error(
                f"❌ Entity deduplication failed for graph {self.graph_id}: {e}"
            )
            # Don't raise - return original graph to continue pipeline

        # Return the original graph (deduplication happens in-database)
        return graph

    async def _deduplicate_exact_matches(self) -> int:
        """
        Find and consolidate entities with identical names across different chunks.

        Strategy:
        1. Find groups of entities with the same name across different chunks
        2. For each group, keep the first entity as canonical
        3. Move all FROM_CHUNK relationships to the canonical entity
        4. Move all other relationships to the canonical entity
        5. Delete duplicate entities

        Returns:
            Number of entities that were deduplicated
        """
        entities_deduplicated = 0

        with self.driver.session(database=self.neo4j_database) as session:
            # Find entities with the same name in different chunks
            find_duplicates_query = """
            MATCH (e:__Entity__)-[:FROM_CHUNK]->(c:Chunk)
            WHERE e.graph_id = $graph_id
            WITH e.name as entity_name, collect(DISTINCT e) as entities, collect(DISTINCT c) as chunks
            WHERE size(entities) > 1
            RETURN entity_name, entities, chunks
            """

            result = session.run(find_duplicates_query, graph_id=self.graph_id)
            duplicate_groups = list(result)

            logger.info(f"Found {len(duplicate_groups)} entity groups with duplicates")

            for record in duplicate_groups:
                entity_name = record["entity_name"]
                entities = record["entities"]
                chunks = record["chunks"]

                try:
                    deduplicated_count = await self._consolidate_entity_group(
                        session, entity_name, entities, chunks
                    )
                    entities_deduplicated += deduplicated_count

                except Exception as e:
                    logger.error(
                        f"Failed to deduplicate entity group '{entity_name}': {e}"
                    )

        return entities_deduplicated

    async def _consolidate_entity_group(
        self,
        session: Session,
        entity_name: str,
        entities: list[Node],
        chunks: list[Node],
    ) -> int:
        """
        Consolidate a group of duplicate entities into a single canonical entity.

        Args:
            session: Neo4j session
            entity_name: Name of the entities to consolidate
            entities: List of duplicate entity nodes
            chunks: List of chunks where these entities appear

        Returns:
            Number of entities that were consolidated (duplicates removed)
        """
        if len(entities) <= 1:
            return 0

        # Use the first entity as the canonical one
        canonical_entity_id = entities[0].element_id
        duplicate_entity_ids = [e.element_id for e in entities[1:]]

        logger.debug(
            f"Consolidating {len(duplicate_entity_ids)} duplicates of '{entity_name}' "
            f"into canonical entity {canonical_entity_id}"
        )

        # Step 1: Connect canonical entity to all chunks
        for chunk in chunks:
            chunk_id = chunk.element_id

            # Connect canonical entity to this chunk if not already connected
            session.run(
                """
                MATCH (canonical) WHERE elementId(canonical) = $canonical_id
                MATCH (chunk) WHERE elementId(chunk) = $chunk_id
                MERGE (canonical)-[:FROM_CHUNK]->(chunk)
            """,
                canonical_id=canonical_entity_id,
                chunk_id=chunk_id,
            )

        # Step 2: Move relationships from duplicates to canonical (without APOC)
        for duplicate_id in duplicate_entity_ids:
            # Handle outgoing relationships manually by relationship type
            # First get all relationship types and targets
            # Handle outgoing relationships manually by relationship type
            # First get all relationship types and targets
            outgoing_rels = session.run(
                """
                MATCH (duplicate)-[r]->(target)
                WHERE elementId(duplicate) = $duplicate_id AND type(r) <> 'FROM_CHUNK'
                RETURN elementId(target) as target_id, type(r) as rel_type, properties(r) as rel_props
            """,
                duplicate_id=duplicate_id,
            )

            # Recreate each relationship for canonical entity
            for rel_record in list(outgoing_rels):
                rel_type = rel_record["rel_type"]
                target_id = rel_record["target_id"]
                rel_props: dict[str, Any] = rel_record["rel_props"] or {}

                # Use parameterized query with predefined relationship types
                # Handle all relationship types dynamically
                query = """
                    MATCH (canonical) WHERE elementId(canonical) = $canonical_id
                    MATCH (target) WHERE elementId(target) = $target_id
                    CALL apoc.create.relationship(canonical, $rel_type, $rel_props, target) YIELD rel
                    RETURN rel
                """
                try:
                    session.run(
                        query,
                        canonical_id=canonical_entity_id,
                        target_id=target_id,
                        rel_type=rel_type,
                        rel_props=rel_props,
                    )
                except Exception:
                    # Fallback without APOC - create specific relationship types
                    self._create_relationship_fallback(
                        session, canonical_entity_id, target_id, rel_type, rel_props
                    )

            # Handle incoming relationships
            incoming_rels = session.run(
                """
                MATCH (source)-[r]->(duplicate)
                WHERE elementId(duplicate) = $duplicate_id AND type(r) <> 'FROM_CHUNK'
                RETURN elementId(source) as source_id, type(r) as rel_type, properties(r) as rel_props
            """,
                duplicate_id=duplicate_id,
            )

            # Recreate each incoming relationship for canonical entity
            for rel_record in list(incoming_rels):
                rel_type = rel_record["rel_type"]
                source_id = rel_record["source_id"]
                rel_props: dict[str, Any] = rel_record["rel_props"] or {}

                # Handle all relationship types dynamically
                query = """
                    MATCH (source) WHERE elementId(source) = $source_id
                    MATCH (canonical) WHERE elementId(canonical) = $canonical_id
                    CALL apoc.create.relationship(source, $rel_type, $rel_props, canonical) YIELD rel
                    RETURN rel
                """
                try:
                    session.run(
                        query,
                        source_id=source_id,
                        canonical_id=canonical_entity_id,
                        rel_type=rel_type,
                        rel_props=rel_props,
                    )
                except Exception:
                    # Fallback without APOC
                    self._create_relationship_fallback(
                        session, source_id, canonical_entity_id, rel_type, rel_props
                    )

        # Step 3: Delete duplicate entities and their relationships
        for duplicate_id in duplicate_entity_ids:
            session.run(
                """
                MATCH (duplicate) WHERE elementId(duplicate) = $duplicate_id
                DETACH DELETE duplicate
            """,
                duplicate_id=duplicate_id,
            )

        logger.debug(
            f"✅ Consolidated '{entity_name}': removed {len(duplicate_entity_ids)} duplicates, "
            f"canonical entity now linked to {len(chunks)} chunks"
        )

        return len(duplicate_entity_ids)

    def _create_relationship_fallback(
        self,
        session: Session,
        source_id: str,
        target_id: str,
        rel_type: str,
        rel_props: dict[str, Any],
    ) -> None:
        """
        Fallback method to create relationships without APOC procedures.

        Only relationship types in _ALLOWED_REL_TYPES are accepted; any other
        type is silently skipped with a warning to prevent Cypher injection.
        No apoc.cypher.doIt() or string concatenation into Cypher query text.
        """
        if rel_type not in _ALLOWED_REL_TYPES:
            logger.warning(
                "skipping relationship of unrecognised type %r during deduplication"
                " -- not in allowlist",
                rel_type,
            )
            return
        self._create_known_relationship(
            session, source_id, target_id, rel_type, rel_props
        )

    def _create_known_relationship(
        self,
        session: Session,
        source_id: str,
        target_id: str,
        rel_type: str,
        rel_props: dict[str, Any],
    ) -> None:
        """
        Create relationships for known/predefined relationship types.
        This is the ultimate fallback when APOC is not available.
        """
        if rel_type == "WORKS_FOR":
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:WORKS_FOR]->(target)
                SET r = $rel_props
            """
        elif rel_type == "FOUNDED":
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:FOUNDED]->(target)
                SET r = $rel_props
            """
        elif rel_type == "LEADS":
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:LEADS]->(target)
                SET r = $rel_props
            """
        elif rel_type == "MANAGES":
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:MANAGES]->(target)
                SET r = $rel_props
            """
        elif rel_type == "DEVELOPED":
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:DEVELOPED]->(target)
                SET r = $rel_props
            """
        elif rel_type == "REPORTS_TO":
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:REPORTS_TO]->(target)
                SET r = $rel_props
            """
        elif rel_type == "HAS_SKILL":
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:HAS_SKILL]->(target)
                SET r = $rel_props
            """
        elif rel_type == "MEMBER_OF":
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:MEMBER_OF]->(target)
                SET r = $rel_props
            """
        elif rel_type == "INVESTED_IN":
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:INVESTED_IN]->(target)
                SET r = $rel_props
            """
        elif rel_type == "CITES":
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:CITES]->(target)
                SET r = $rel_props
            """
        elif rel_type == "AUTHORED":
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:AUTHORED]->(target)
                SET r = $rel_props
            """
        elif rel_type == "WORKS_ON":
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:WORKS_ON]->(target)
                SET r = $rel_props
            """
        elif rel_type == "DEPENDS_ON":
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:DEPENDS_ON]->(target)
                SET r = $rel_props
            """
        elif rel_type == "ACQUIRED_BY":
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:ACQUIRED_BY]->(target)
                SET r = $rel_props
            """
        elif rel_type == "PARTNER_OF":
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:PARTNER_OF]->(target)
                SET r = $rel_props
            """
        elif rel_type == "RELATED_TO":
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:RELATED_TO]->(target)
                SET r = $rel_props
            """
        elif rel_type == "PART_OF":
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:PART_OF]->(target)
                SET r = $rel_props
            """
        elif rel_type == "OWNS":
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:OWNS]->(target)
                SET r = $rel_props
            """
        elif rel_type == "LOCATED_IN":
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:LOCATED_IN]->(target)
                SET r = $rel_props
            """
        elif rel_type == "SAME_AS":
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:SAME_AS]->(target)
                SET r = $rel_props
            """
        elif rel_type == "SIMILAR_TO":
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:SIMILAR_TO]->(target)
                SET r = $rel_props
            """
        else:
            # _create_relationship_fallback() already checked the allowlist;
            # this branch should never be reached.
            logger.warning(
                "skipping relationship of unrecognised type %r in _create_known_relationship",
                rel_type,
            )
            return

        session.run(
            query, source_id=source_id, target_id=target_id, rel_props=rel_props
        )

    async def _deduplicate_fuzzy_matches(self) -> int:
        """
        Find and consolidate entities with similar names using fuzzy matching.

        This method uses APOC procedures for similarity calculations when available.
        Falls back gracefully when APOC is not installed.

        Returns:
            Number of entities that were deduplicated via fuzzy matching
        """
        entities_deduplicated = 0

        # Query to find entities with similar names using fuzzy matching
        query = """
        MATCH (e1:__Entity__)-[:FROM_CHUNK]->(c1:Chunk)
        MATCH (e2:__Entity__)-[:FROM_CHUNK]->(c2:Chunk)
        WHERE e1.graph_id = $graph_id
        AND e2.graph_id = $graph_id
        AND c1.index <> c2.index
        AND elementId(e1) < elementId(e2)
        AND NOT e1.name = e2.name  // Exclude exact matches
        // Simple fuzzy matching using CONTAINS or APOC similarity
        AND (
            e1.name CONTAINS e2.name OR
            e2.name CONTAINS e1.name OR
            (EXISTS {
                CALL apoc.text.levenshteinSimilarity(e1.name, e2.name) YIELD value
                WHERE value > $threshold
            })
        )
        WITH e1, e2,
             CASE
                WHEN EXISTS { CALL apoc.text.levenshteinSimilarity(e1.name, e2.name) YIELD value }
                THEN apoc.text.levenshteinSimilarity(e1.name, e2.name)
                ELSE 0.9
             END as similarity
        WHERE similarity > $threshold
        WITH e1.name as entity_name, collect(DISTINCT e1) + collect(DISTINCT e2) as entities
        WHERE size(entities) > 1
        RETURN entity_name, entities
        """

        try:
            with self.driver.session(database=self.neo4j_database) as session:
                # Find potential fuzzy matches
                result = session.run(
                    query, graph_id=self.graph_id, threshold=self.similarity_threshold
                )
                fuzzy_groups = list(result)

                logger.info(
                    f"Found {len(fuzzy_groups)} entity groups with fuzzy matches"
                )

                # Process each group of similar entities
                for record in fuzzy_groups:
                    entity_name = record["entity_name"]
                    entities = record["entities"]

                    if len(entities) > 1:
                        try:
                            # Get chunks for these entities
                            chunks: list[Node] = []
                            for entity in entities:
                                entity_chunks = session.run(
                                    """
                                    MATCH (e)-[:FROM_CHUNK]->(c:Chunk)
                                    WHERE elementId(e) = $entity_id
                                    RETURN c
                                """,
                                    entity_id=entity.element_id,
                                )
                                chunks.extend([record["c"] for record in entity_chunks])

                            # Remove duplicates by element_id
                            unique_chunks_dict = {
                                chunk.element_id: chunk for chunk in chunks
                            }
                            unique_chunks: list[Node] = list(
                                unique_chunks_dict.values()
                            )

                            # Consolidate the fuzzy matched entities
                            deduplicated_count = await self._consolidate_entity_group(
                                session, entity_name, entities, unique_chunks
                            )
                            entities_deduplicated += deduplicated_count

                        except Exception as e:
                            logger.error(
                                f"Failed to deduplicate fuzzy entity group '{entity_name}': {e}"
                            )

        except Exception as e:
            # APOC might not be available, that's ok
            logger.warning(
                f"Fuzzy matching requires APOC procedures (falling back to exact matching only): {e}"
            )

        return entities_deduplicated
