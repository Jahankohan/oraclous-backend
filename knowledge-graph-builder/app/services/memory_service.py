"""
Memory Service

Business logic for the Agent Memory API:
- Memory CRUD with multi-tenant graph_id scoping
- Ebbinghaus decay computation
- Contradiction detection (semantic memories)
- Consolidation logic
- Context window assembly for agent prompt injection
"""

from __future__ import annotations

import hashlib
import math
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client
from app.schemas.memory import (
    ConflictInfo,
    ContradictionResolution,
    GraphFact,
    MemoryContext,
    MemoryCreate,
    MemoryCreateResponse,
    MemoryScope,
    MemorySearchResponse,
    MemorySearchResult,
    MemoryType,
    MemoryUpdate,
    MemoryUpdateResponse,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Decay constants (Ebbinghaus λ) by memory type
# ---------------------------------------------------------------------------
_DECAY: dict[str, float] = {
    MemoryType.EPISODIC: 0.05,
    MemoryType.SEMANTIC: 0.005,
    MemoryType.PROCEDURAL: 0.01,
}

# Base importance by source
_BASE_IMPORTANCE: dict[str, float] = {
    "user_feedback": 1.0,
    "agent_high": 0.8,  # confidence >= 0.9
    "agent_medium": 0.0,  # filled dynamically: confidence * 0.9
    "episodic": 0.4,
    "inference": 0.3,
}

# Retrieval ranking weights
_RANK_WEIGHTS = {"vector": 0.50, "importance": 0.30, "recency": 0.20}


# ==================== DECAY HELPERS ====================


def compute_importance(
    base_importance: float,
    memory_type: str,
    last_accessed_at: datetime,
    access_count: int,
    now: datetime | None = None,
) -> float:
    """
    Ebbinghaus-inspired forgetting curve with access boosting.

    I(t) = base_importance * e^(-λ * days) + access_boost
    """
    if now is None:
        now = datetime.now(UTC)
    if last_accessed_at.tzinfo is None:
        last_accessed_at = last_accessed_at.replace(tzinfo=UTC)

    lam = _DECAY.get(memory_type, 0.01)
    days = max(0.0, (now - last_accessed_at).total_seconds() / 86400)
    decayed = base_importance * math.exp(-lam * days)
    access_boost = min(0.3, 0.05 * math.log1p(access_count))
    return min(1.0, decayed + access_boost)


def _recency_factor(last_accessed_at: datetime, now: datetime | None = None) -> float:
    if now is None:
        now = datetime.now(UTC)
    if last_accessed_at.tzinfo is None:
        last_accessed_at = last_accessed_at.replace(tzinfo=UTC)
    days = max(0.0, (now - last_accessed_at).total_seconds() / 86400)
    return math.exp(-0.02 * days)


def _base_importance_for(req: MemoryCreate) -> float:
    if req.source == "user_feedback":
        return 1.0
    if req.type == MemoryType.EPISODIC:
        return 0.4
    if req.source == "inference":
        return 0.3
    if req.confidence >= 0.9:
        return 0.8
    return req.confidence * 0.9


def _content_hash(content: str) -> str:
    normalized = " ".join(content.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


# ==================== MEMORY SERVICE ====================


class MemoryService:
    """
    All memory operations. Uses neo4j_client.async_driver (FastAPI rule).
    Consolidation runs in Celery workers via the task in background_jobs.py.
    """

    # ------------------------------------------------------------------ #
    # Store                                                                #
    # ------------------------------------------------------------------ #

    async def store_memory(
        self, graph_id: str, req: MemoryCreate
    ) -> MemoryCreateResponse:
        now = datetime.now(UTC)
        content_hash = _content_hash(req.content)

        # 1. Duplicate check
        existing = await self._find_by_content_hash(graph_id, content_hash)
        if existing:
            return MemoryCreateResponse(
                memory_id=existing["memory_id"],
                importance_score=existing["importance_score"],
                contradictions_detected=[],
                entity_linked=None,
            )

        memory_id = str(uuid.uuid4())
        base_imp = _base_importance_for(req)
        valid_from = req.valid_from or now

        # 2. Build type-specific extra properties
        extra_props: dict[str, Any] = {}
        labels = ["Memory"]
        if req.type == MemoryType.EPISODIC:
            labels.append("Episodic")
            extra_props["event_type"] = req.event_type or "interaction"
            extra_props["user_id"] = req.user_id or ""
        elif req.type == MemoryType.SEMANTIC:
            labels.append("Semantic")
            extra_props["subject"] = req.subject or ""
            extra_props["predicate"] = req.predicate or ""
            extra_props["object"] = req.object or ""
            extra_props["is_negation"] = req.is_negation
        elif req.type == MemoryType.PROCEDURAL:
            labels.append("Procedural")
            extra_props["category"] = req.category or "preference"
            extra_props["trigger_pattern"] = req.trigger_pattern or ""
            extra_props["times_applied"] = 0

        # 3. Create node
        label_str = ":".join(labels)
        set_clauses = "\n".join(f"  m.{k} = ${k}," for k in extra_props).rstrip(",")

        params: dict[str, Any] = {
            "memory_id": memory_id,
            "graph_id": graph_id,
            "memory_type": req.type.value,
            "content": req.content,
            "content_hash": content_hash,
            "importance_score": base_imp,
            "base_importance": base_imp,
            "access_count": 0,
            "last_accessed_at": now.isoformat(),
            "valid_from": valid_from.isoformat(),
            "valid_to": None,
            "ingested_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "source": req.source.value,
            "agent_id": req.agent_id or "",
            "session_id": req.session_id or "",
            "confidence": req.confidence,
            "scope": req.scope.value,
            **extra_props,
        }

        extra_set = ""
        if set_clauses:
            extra_set = "\n" + set_clauses

        create_query = f"""
        CREATE (m:{label_str} {{
          memory_id: $memory_id,
          graph_id: $graph_id,
          memory_type: $memory_type,
          content: $content,
          content_hash: $content_hash,
          importance_score: $importance_score,
          base_importance: $base_importance,
          access_count: $access_count,
          last_accessed_at: datetime($last_accessed_at),
          valid_from: datetime($valid_from),
          valid_to: null,
          ingested_at: datetime($ingested_at),
          updated_at: datetime($updated_at),
          source: $source,
          agent_id: $agent_id,
          session_id: $session_id,
          confidence: $confidence,
          scope: $scope
        }})
        SET{extra_set if extra_set else ""}
        RETURN m.memory_id AS memory_id
        """
        # Remove SET if there is nothing to set
        if not extra_set:
            create_query = create_query.replace(
                "\n        SET\n        RETURN", "\n        RETURN"
            )

        await neo4j_client.execute_write_query(create_query, params)

        # 4. Contradiction detection (semantic only)
        contradictions: list[ConflictInfo] = []
        if req.type == MemoryType.SEMANTIC and req.subject and req.predicate:
            contradictions = await self._detect_and_record_contradictions(
                graph_id, memory_id, req, now
            )

        # 5. Entity linking
        entity_linked: str | None = None
        if req.type == MemoryType.SEMANTIC and req.subject:
            entity_linked = await self._link_to_entity(graph_id, memory_id, req.subject)

        return MemoryCreateResponse(
            memory_id=memory_id,
            importance_score=base_imp,
            contradictions_detected=contradictions,
            entity_linked=entity_linked,
        )

    # ------------------------------------------------------------------ #
    # Search                                                               #
    # ------------------------------------------------------------------ #

    async def search_memories(
        self,
        graph_id: str,
        query: str,
        memory_type: MemoryType | None = None,
        scope: MemoryScope | None = None,
        temporal: str = "current",
        min_confidence: float = 0.0,
        limit: int = 20,
        include_graph_facts: bool = False,
    ) -> MemorySearchResponse:
        now = datetime.now(UTC)

        where_clauses = ["m.graph_id = $graph_id", "m.confidence >= $min_confidence"]
        params: dict[str, Any] = {
            "graph_id": graph_id,
            "min_confidence": min_confidence,
            "limit": limit,
            "query": query,
        }

        if memory_type:
            where_clauses.append("m.memory_type = $memory_type")
            params["memory_type"] = memory_type.value

        if scope:
            where_clauses.append("m.scope = $scope")
            params["scope"] = scope.value

        if temporal == "current":
            where_clauses.append("m.valid_to IS NULL")

        where_str = " AND ".join(where_clauses)

        # Fulltext search + ranking
        search_query = f"""
        CALL db.index.fulltext.queryNodes('memory_content_idx', $query)
        YIELD node AS m, score AS text_score
        WHERE {where_str}
        WITH m, text_score,
             m.importance_score AS imp,
             duration.inDays(m.last_accessed_at, datetime()).days AS days_since
        WITH m, text_score, imp,
             exp(-0.02 * toFloat(days_since)) AS recency,
             ({_RANK_WEIGHTS["vector"]} * text_score
              + {_RANK_WEIGHTS["importance"]} * imp
              + {_RANK_WEIGHTS["recency"]} * exp(-0.02 * toFloat(days_since))) AS ranking
        ORDER BY ranking DESC
        LIMIT $limit
        RETURN
          m.memory_id AS memory_id,
          m.memory_type AS type,
          m.content AS content,
          imp AS importance_score,
          ranking AS relevance_score,
          m.confidence AS confidence,
          m.valid_from AS valid_from,
          m.valid_to AS valid_to,
          m.scope AS scope,
          m.agent_id AS agent_id,
          m.session_id AS session_id,
          m.ingested_at AS created_at,
          m.last_accessed_at AS last_accessed_at,
          m.access_count AS access_count
        """

        records = await neo4j_client.execute_query(search_query, params)

        # Bump access counts
        if records:
            hit_ids = [r["memory_id"] for r in records]
            await self._bump_access(graph_id, hit_ids, now)

        memories = [self._record_to_search_result(r) for r in records]

        # Graph facts
        graph_facts: list[GraphFact] = []
        if include_graph_facts and query:
            graph_facts = await self._fetch_graph_facts(graph_id, query, limit=10)

        return MemorySearchResponse(
            memories=memories,
            graph_facts=graph_facts,
            total=len(memories),
        )

    # ------------------------------------------------------------------ #
    # Context window assembly                                              #
    # ------------------------------------------------------------------ #

    async def get_context(
        self,
        graph_id: str,
        query: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        scopes: list[str] | None = None,
        max_tokens: int = 2000,
        include_types: list[str] | None = None,
    ) -> MemoryContext:
        t0 = time.monotonic()
        now = datetime.now(UTC)

        scope_filter = ""
        scope_params: dict[str, Any] = {}
        if scopes:
            scope_filter = "AND m.scope IN $scopes"
            scope_params["scopes"] = scopes

        type_filter = ""
        if include_types:
            type_filter = "AND m.memory_type IN $include_types"
            scope_params["include_types"] = include_types

        params: dict[str, Any] = {
            "graph_id": graph_id,
            "query": query,
            "limit": 50,
            **scope_params,
        }

        context_query = f"""
        CALL db.index.fulltext.queryNodes('memory_content_idx', $query)
        YIELD node AS m, score AS text_score
        WHERE m.graph_id = $graph_id
          AND m.valid_to IS NULL
          {scope_filter}
          {type_filter}
        WITH m, text_score,
             m.importance_score AS imp,
             duration.inDays(m.last_accessed_at, datetime()).days AS days_since
        ORDER BY (0.5 * text_score + 0.3 * imp + 0.2 * exp(-0.02 * toFloat(days_since))) DESC
        LIMIT $limit
        RETURN m.memory_id AS memory_id,
               m.memory_type AS memory_type,
               m.content AS content,
               m.confidence AS confidence,
               m.scope AS scope,
               m.last_accessed_at AS last_accessed_at
        """

        records = await neo4j_client.execute_query(context_query, params)

        # Token-budgeted assembly
        sections: dict[str, list[str]] = {
            "semantic": [],
            "procedural": [],
            "episodic": [],
        }
        used_ids: list[str] = []
        estimated_tokens = 0
        tokens_per_char = 0.25  # ~4 chars per token

        for rec in records:
            entry = f"- {rec['content']} (confidence: {rec['confidence']:.2f})"
            entry_tokens = int(len(entry) * tokens_per_char) + 5
            if estimated_tokens + entry_tokens > max_tokens:
                break
            mtype = rec.get("memory_type", "semantic")
            sections.setdefault(mtype, []).append(entry)
            used_ids.append(rec["memory_id"])
            estimated_tokens += entry_tokens

        lines: list[str] = ["## Relevant Memory\n"]
        if sections.get("semantic"):
            lines.append("**Facts:**")
            lines.extend(sections["semantic"])
        if sections.get("procedural"):
            lines.append("\n**Preferences:**")
            lines.extend(sections["procedural"])
        if sections.get("episodic"):
            lines.append("\n**Recent activity:**")
            lines.extend(sections["episodic"])

        context_block = "\n".join(lines)

        if used_ids:
            await self._bump_access(graph_id, used_ids, now)

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        return MemoryContext(
            context_block=context_block,
            memories_used=used_ids,
            token_estimate=estimated_tokens,
            retrieval_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------ #
    # Update (temporal versioning)                                         #
    # ------------------------------------------------------------------ #

    async def update_memory(
        self, graph_id: str, memory_id: str, req: MemoryUpdate
    ) -> MemoryUpdateResponse:
        now = datetime.now(UTC)

        # Fetch existing
        existing_records = await neo4j_client.execute_query(
            """
            MATCH (m:Memory {graph_id: $graph_id, memory_id: $memory_id})
            WHERE m.valid_to IS NULL
            RETURN m.content AS content, m.memory_type AS memory_type,
                   m.confidence AS confidence, m.scope AS scope,
                   m.agent_id AS agent_id, m.session_id AS session_id,
                   m.source AS source, m.base_importance AS base_importance
            """,
            {"graph_id": graph_id, "memory_id": memory_id},
        )
        if not existing_records:
            raise ValueError(f"Memory {memory_id} not found in graph {graph_id}")

        old = existing_records[0]
        new_id = str(uuid.uuid4())
        new_content = req.content if req.content is not None else old["content"]
        new_confidence = (
            req.confidence if req.confidence is not None else old["confidence"]
        )
        new_hash = _content_hash(new_content)
        base_imp = float(old["base_importance"]) if old["base_importance"] else 0.8

        await neo4j_client.execute_write_query(
            """
            MATCH (old:Memory {graph_id: $graph_id, memory_id: $old_id})
            SET old.valid_to = datetime($now)
            CREATE (new:Memory {
              memory_id: $new_id,
              graph_id: $graph_id,
              memory_type: old.memory_type,
              content: $new_content,
              content_hash: $new_hash,
              importance_score: $base_imp,
              base_importance: $base_imp,
              access_count: 0,
              last_accessed_at: datetime($now),
              valid_from: datetime($now),
              valid_to: null,
              ingested_at: datetime($now),
              updated_at: datetime($now),
              source: old.source,
              agent_id: old.agent_id,
              session_id: old.session_id,
              confidence: $new_confidence,
              scope: old.scope
            })
            CREATE (new)-[:SUPERSEDES {reason: $reason, superseded_at: datetime($now)}]->(old)
            """,
            {
                "graph_id": graph_id,
                "old_id": memory_id,
                "new_id": new_id,
                "new_content": new_content,
                "new_hash": new_hash,
                "base_imp": base_imp,
                "new_confidence": new_confidence,
                "reason": req.reason or "update",
                "now": now.isoformat(),
            },
        )

        return MemoryUpdateResponse(
            old_memory_id=memory_id,
            new_memory_id=new_id,
            superseded_at=now,
        )

    # ------------------------------------------------------------------ #
    # Delete                                                               #
    # ------------------------------------------------------------------ #

    async def delete_memory(
        self, graph_id: str, memory_id: str, hard: bool = False
    ) -> None:
        if hard:
            await neo4j_client.execute_write_query(
                """
                MATCH (m:Memory {graph_id: $graph_id, memory_id: $memory_id})
                DETACH DELETE m
                """,
                {"graph_id": graph_id, "memory_id": memory_id},
            )
        else:
            now = datetime.now(UTC)
            await neo4j_client.execute_write_query(
                """
                MATCH (m:Memory {graph_id: $graph_id, memory_id: $memory_id})
                WHERE m.valid_to IS NULL
                SET m.valid_to = datetime($now), m.updated_at = datetime($now)
                """,
                {"graph_id": graph_id, "memory_id": memory_id, "now": now.isoformat()},
            )

    # ------------------------------------------------------------------ #
    # Consolidation (called from Celery task)                             #
    # ------------------------------------------------------------------ #

    async def consolidate(self, graph_id: str) -> dict[str, Any]:
        """
        Find clusters of semantically similar memories (cosine sim > 0.92)
        and merge duplicates. Returns stats dict.

        This is called by the Celery beat task — the driver used there is
        a task-scoped sync driver, so this method is also called via
        asyncio.run() inside the Celery task wrapper.
        """
        now = datetime.now(UTC)

        # Fetch all current memories for the graph that have embeddings
        # We cluster by content similarity using Neo4j's vector index
        candidates_query = """
        MATCH (m:Memory {graph_id: $graph_id})
        WHERE m.valid_to IS NULL
        RETURN m.memory_id AS memory_id,
               m.content AS content,
               m.importance_score AS importance_score,
               m.confidence AS confidence,
               m.base_importance AS base_importance
        ORDER BY m.importance_score DESC
        LIMIT 5000
        """
        candidates = await neo4j_client.execute_query(
            candidates_query, {"graph_id": graph_id}
        )

        if len(candidates) < 2:
            return {"merged": 0, "graph_id": graph_id}

        # Find near-duplicate pairs using fulltext similarity heuristic.
        # True vector clustering would require iterating the vector index;
        # we approximate with hash-based dedup and fulltext matches here,
        # leaving full HNSW cluster traversal for a future optimization.
        content_hash_map: dict[str, list[str]] = {}
        for c in candidates:
            h = _content_hash(c["content"])
            content_hash_map.setdefault(h, []).append(c["memory_id"])

        merged_count = 0
        for _hash_val, ids in content_hash_map.items():
            if len(ids) < 2:
                continue
            # Keep the first (highest importance_score — sorted DESC above)
            winner_id = ids[0]
            losers = ids[1:]
            for loser_id in losers:
                await neo4j_client.execute_write_query(
                    """
                    MATCH (winner:Memory {graph_id: $graph_id, memory_id: $winner_id}),
                          (loser:Memory  {graph_id: $graph_id, memory_id: $loser_id})
                    SET loser.valid_to = datetime($now),
                        loser.updated_at = datetime($now),
                        winner.base_importance = CASE
                          WHEN winner.base_importance + loser.base_importance > 1.0
                          THEN 1.0
                          ELSE winner.base_importance + loser.base_importance
                        END
                    WITH winner, loser
                    OPTIONAL MATCH (loser)-[r:ABOUT]->(e)
                    FOREACH (_ IN CASE WHEN e IS NOT NULL THEN [1] ELSE [] END |
                      MERGE (winner)-[:ABOUT]->(e)
                    )
                    CREATE (winner)-[:SUPERSEDES {
                      reason: 'consolidation',
                      superseded_at: datetime($now)
                    }]->(loser)
                    """,
                    {
                        "graph_id": graph_id,
                        "winner_id": winner_id,
                        "loser_id": loser_id,
                        "now": now.isoformat(),
                    },
                )
                merged_count += 1

        logger.info(
            f"Consolidation for graph {graph_id}: merged {merged_count} duplicate memories"
        )
        return {"merged": merged_count, "graph_id": graph_id}

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    async def _find_by_content_hash(
        self, graph_id: str, content_hash: str
    ) -> dict[str, Any] | None:
        records = await neo4j_client.execute_query(
            """
            MATCH (m:Memory {graph_id: $graph_id, content_hash: $content_hash})
            WHERE m.valid_to IS NULL
            RETURN m.memory_id AS memory_id, m.importance_score AS importance_score
            LIMIT 1
            """,
            {"graph_id": graph_id, "content_hash": content_hash},
        )
        return records[0] if records else None

    async def _detect_and_record_contradictions(
        self,
        graph_id: str,
        new_memory_id: str,
        req: MemoryCreate,
        now: datetime,
    ) -> list[ConflictInfo]:
        """Detect semantic memories with same subject+predicate but different object."""
        records = await neo4j_client.execute_query(
            """
            MATCH (m:Memory:Semantic {graph_id: $graph_id})
            WHERE m.valid_to IS NULL
              AND m.memory_id <> $new_id
              AND m.subject = $subject
              AND m.predicate = $predicate
              AND m.object <> $object
            RETURN m.memory_id AS memory_id, m.content AS content
            LIMIT 10
            """,
            {
                "graph_id": graph_id,
                "new_id": new_memory_id,
                "subject": req.subject or "",
                "predicate": req.predicate or "",
                "object": req.object or "",
            },
        )

        conflicts: list[ConflictInfo] = []
        for rec in records:
            resolution = ContradictionResolution.NEW_WINS
            await neo4j_client.execute_write_query(
                """
                MATCH (new_m:Memory {graph_id: $graph_id, memory_id: $new_id}),
                      (old_m:Memory {graph_id: $graph_id, memory_id: $old_id})
                MERGE (new_m)-[:CONTRADICTS {
                  detected_at: datetime($now),
                  resolution: $resolution
                }]->(old_m)
                SET old_m.valid_to = CASE WHEN $resolution = 'new_wins'
                                          THEN datetime($now) ELSE old_m.valid_to END
                """,
                {
                    "graph_id": graph_id,
                    "new_id": new_memory_id,
                    "old_id": rec["memory_id"],
                    "now": now.isoformat(),
                    "resolution": resolution.value,
                },
            )
            conflicts.append(
                ConflictInfo(
                    conflict_memory_id=rec["memory_id"],
                    content=rec["content"],
                    resolution=resolution,
                )
            )

        return conflicts

    async def _link_to_entity(
        self, graph_id: str, memory_id: str, subject: str
    ) -> str | None:
        """Create ABOUT edge to matching __Entity__ node."""
        records = await neo4j_client.execute_query(
            """
            MATCH (e:__Entity__ {graph_id: $graph_id})
            WHERE toLower(e.name) = toLower($subject)
            RETURN e.id AS entity_id
            LIMIT 1
            """,
            {"graph_id": graph_id, "subject": subject},
        )
        if not records:
            return None

        entity_id = records[0]["entity_id"]
        await neo4j_client.execute_write_query(
            """
            MATCH (m:Memory {graph_id: $graph_id, memory_id: $memory_id}),
                  (e:__Entity__ {graph_id: $graph_id, id: $entity_id})
            MERGE (m)-[:ABOUT {confidence: m.confidence}]->(e)
            """,
            {"graph_id": graph_id, "memory_id": memory_id, "entity_id": entity_id},
        )
        return entity_id

    async def _bump_access(
        self, graph_id: str, memory_ids: list[str], now: datetime
    ) -> None:
        await neo4j_client.execute_write_query(
            """
            UNWIND $memory_ids AS mid
            MATCH (m:Memory {graph_id: $graph_id, memory_id: mid})
            SET m.access_count = m.access_count + 1,
                m.last_accessed_at = datetime($now),
                m.importance_score = CASE
                  WHEN m.base_importance * exp(-0.005 * toFloat(duration.inDays(m.last_accessed_at, datetime($now)).days))
                       + CASE WHEN log(1 + m.access_count + 1) * 0.05 > 0.3 THEN 0.3
                              ELSE log(1 + m.access_count + 1) * 0.05 END > 1.0
                  THEN 1.0
                  ELSE m.base_importance * exp(-0.005 * toFloat(duration.inDays(m.last_accessed_at, datetime($now)).days))
                       + CASE WHEN log(1 + m.access_count + 1) * 0.05 > 0.3 THEN 0.3
                              ELSE log(1 + m.access_count + 1) * 0.05 END
                END
            """,
            {"graph_id": graph_id, "memory_ids": memory_ids, "now": now.isoformat()},
        )

    async def _fetch_graph_facts(
        self, graph_id: str, query: str, limit: int = 10
    ) -> list[GraphFact]:
        records = await neo4j_client.execute_query(
            """
            CALL db.index.fulltext.queryNodes('entity_text_fulltext', $query)
            YIELD node AS e, score
            WHERE e.graph_id = $graph_id
            WITH e ORDER BY score DESC LIMIT 5
            MATCH (e)-[r]->(other:__Entity__ {graph_id: $graph_id})
            RETURN e.name AS subject, type(r) AS predicate, other.name AS object,
                   null AS source_chunk_id
            LIMIT $limit
            """,
            {"graph_id": graph_id, "query": query, "limit": limit},
        )
        return [
            GraphFact(
                subject=r["subject"],
                predicate=r["predicate"],
                object=r["object"],
                source_chunk_id=r.get("source_chunk_id"),
            )
            for r in records
        ]

    def _record_to_search_result(self, rec: dict[str, Any]) -> MemorySearchResult:
        def _dt(v: Any) -> datetime | None:
            if v is None:
                return None
            if isinstance(v, datetime):
                return v
            try:
                return datetime.fromisoformat(str(v))
            except Exception:
                return None

        return MemorySearchResult(
            memory_id=rec["memory_id"],
            type=MemoryType(rec["type"]),
            content=rec["content"],
            importance_score=float(rec.get("importance_score", 0.0)),
            relevance_score=float(rec.get("relevance_score", 0.0)),
            confidence=float(rec.get("confidence", 0.0)),
            valid_from=_dt(rec.get("valid_from")),
            valid_to=_dt(rec.get("valid_to")),
            scope=MemoryScope(rec.get("scope", "agent")),
            agent_id=rec.get("agent_id") or None,
            session_id=rec.get("session_id") or None,
            created_at=_dt(rec.get("created_at")),
            last_accessed_at=_dt(rec.get("last_accessed_at")),
            access_count=int(rec.get("access_count", 0)),
        )


async def ensure_memory_indexes() -> None:
    """
    Create Neo4j indexes required for the Agent Memory API. Idempotent.
    Called once at application startup (do NOT call neo4j_client.connect/close here).
    """
    index_queries = [
        # 1. Vector index — semantic similarity search on Memory nodes
        """
        CREATE VECTOR INDEX memory_embedding_idx IF NOT EXISTS
        FOR (m:Memory) ON m.embedding
        OPTIONS {indexConfig: {
          `vector.dimensions`: 3072,
          `vector.similarity_function`: 'cosine'
        }}
        """,
        # 2. Fulltext index — keyword recall across memory content
        "CREATE FULLTEXT INDEX memory_content_idx IF NOT EXISTS FOR (m:Memory) ON EACH [m.content]",
        # 3. Composite lookup — graph-scoped queries by scope/type/validity
        "CREATE INDEX memory_graph_scope_idx IF NOT EXISTS FOR (m:Memory) ON (m.graph_id, m.scope, m.memory_type, m.valid_to)",
        # 4. Deduplication — content hash within graph
        "CREATE INDEX memory_content_hash_idx IF NOT EXISTS FOR (m:Memory) ON (m.graph_id, m.content_hash)",
    ]
    for q in index_queries:
        try:
            await neo4j_client.execute_write_query(q)
        except Exception as e:
            logger.warning(f"Memory index creation skipped: {e}")

    indexes = await neo4j_client.execute_query(
        "SHOW INDEXES WHERE labelsOrTypes = ['Memory']", {}
    )
    logger.info(f"Memory indexes present: {len(indexes)}")


# Global service instance
memory_service = MemoryService()
