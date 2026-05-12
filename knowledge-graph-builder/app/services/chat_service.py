"""
Chat Service using Neo4j GraphRAG
Enhanced implementation supporting all retriever types with factory pattern,
strict graph-grounded responses, hallucination prevention, entity anchor
detection, multi-hop reasoning, and auto retriever selection.
"""

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.generation.types import RagResultModel
from neo4j_graphrag.llm import OpenAILLM
from opentelemetry import trace as otel_trace

from app.core.config import settings
from app.core.errors import KGBError
from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client
from app.core.telemetry import get_tracer
from app.schemas.chat_schemas import TemporalMode
from app.schemas.graph_schemas import TemporalFilter
from app.schemas.retriever_schemas import (
    HybridCypherRetrieverConfig,
    HybridRetrieverConfig,
    RetrieverConfig,
    Text2CypherRetrieverConfig,
    VectorCypherRetrieverConfig,
    VectorRetrieverConfig,
    get_default_retriever_config,
)
from app.services.fulltext_index_service import fulltext_index_manager
from app.services.query_cache_service import QueryCacheService
from app.services.retriever_factory import (
    CommunitySummaryRetriever,
    RetrieverType,
    retriever_factory,
)

logger = get_logger(__name__)
_chat_tracer = get_tracer("oraclous.chat")

# ---------------------------------------------------------------------------
# LLM outage exceptions — used for graceful degradation.
# Import openai lazily to avoid hard dependency at module load time.
# ---------------------------------------------------------------------------
try:
    import openai as _openai

    _LLM_DOWN_EXCEPTIONS: tuple = (
        _openai.APITimeoutError,
        _openai.RateLimitError,
        _openai.APIConnectionError,
    )
except ImportError:
    _LLM_DOWN_EXCEPTIONS = (TimeoutError,)

# ---------------------------------------------------------------------------
# Prompt template that strictly grounds responses to retrieved graph context.
# Uses {context} and {query_text} placeholders consumed by neo4j-graphrag.
# ---------------------------------------------------------------------------
STRICT_GROUNDING_PROMPT = """\
You are a knowledge graph assistant. Your ONLY job is to answer questions \
using information explicitly present in the Context section below.

RULES (non-negotiable):
1. Answer SOLELY from the Context. Do NOT use external knowledge or training data.
2. For every factual claim, reference the specific graph node or relationship \
that supports it (e.g. "[Entity: TechNova Corp]").
3. If the Context does not contain enough information to answer the question, \
respond EXACTLY with the following prefix and nothing else:
   INSUFFICIENT_DATA: <brief reason why context is inadequate>
4. Never guess, speculate, or extrapolate beyond what is in the Context.

Context:
{context}

Question: {query_text}

Answer (cite graph nodes/relationships for each fact):"""

# Prefix used to detect insufficient-context responses from the LLM.
_INSUFFICIENT_PREFIX = "INSUFFICIENT_DATA:"

# Minimum number of retriever items required to attempt an answer.
_MIN_CONTEXT_ITEMS = 1

# --------------------------------------------------------------------------
# Query heuristics for auto retriever selection
# --------------------------------------------------------------------------
_ANALYTIC_PATTERNS = re.compile(
    r"\b(list all|find all|show all|count|how many|enumerate|which .* are)\b",
    re.IGNORECASE,
)
_RELATIONSHIP_PATTERNS = re.compile(
    r"\b(relationship|connected|related|partner|between|link|associat|collaborat)\b",
    re.IGNORECASE,
)
_CYPHER_PATTERNS = re.compile(
    r"\b(query|cypher|match|return|where clause|subgraph|path between)\b",
    re.IGNORECASE,
)

# Multi-hop 2-hop expansion query — always filters by graph_id (multi-tenancy).
_MULTIHOP_CYPHER = """\
MATCH (anchor:__Entity__ {graph_id: $graph_id})
WHERE toLower(anchor.name) CONTAINS toLower($entity_name)
MATCH (anchor)-[r1]->(hop1:__Entity__ {graph_id: $graph_id})
OPTIONAL MATCH (hop1)-[r2]->(hop2:__Entity__ {graph_id: $graph_id})
RETURN
    anchor.name        AS anchor_name,
    anchor.description AS anchor_desc,
    type(r1)           AS rel1,
    hop1.name          AS hop1_name,
    hop1.description   AS hop1_desc,
    type(r2)           AS rel2,
    hop2.name          AS hop2_name,
    hop2.description   AS hop2_desc
LIMIT 20
"""

# Words that must not be treated as entity candidates.
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "can",
        "could",
        "about",
        "what",
        "who",
        "which",
        "when",
        "where",
        "why",
        "how",
        "tell",
        "me",
        "give",
        "show",
        "find",
        "list",
        "explain",
        "describe",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "and",
        "or",
        "but",
        "not",
        "with",
        "from",
        "by",
        "its",
        "their",
        "this",
        "that",
        "these",
        "those",
        "any",
        "all",
        "some",
    }
)

# ---------------------------------------------------------------------------
# Global query detection — routes broad/thematic questions to community
# summaries instead of vector/fulltext retrieval.
# ---------------------------------------------------------------------------
_GLOBAL_KEYWORDS = frozenset(
    {
        "overview",
        "themes",
        "theme",
        "main topics",
        "across all",
        "summarize",
        "summarise",
        "what are the",
        "broad",
        "general",
        "landscape",
        "areas",
        "categories",
        "domains",
    }
)


def _is_global_query(query: str) -> bool:
    """Return True when the query is broad/thematic and benefits from community summaries."""
    q_lower = query.lower()
    return any(kw in q_lower for kw in _GLOBAL_KEYWORDS)


@dataclass
class GroundedSearchResult:
    """Structured result from a grounded GraphRAG search."""

    answer: str
    sources: list[dict[str, Any]]
    confidence: float
    is_grounded: bool
    retriever_used: str
    retriever_result: Any | None = None


class ChatService:
    """
    Chat service using Neo4j GraphRAG with hallucination prevention.

    Features:
    - Support for all 5 Neo4j GraphRAG retriever types
    - Strict graph-grounded prompt — LLM only uses retrieved context
    - Structured source citation extracted from retriever results
    - Insufficient-context detection with no-data response
    - Confidence scoring based on retrieval relevance scores
    - Multi-tenant isolation with graph_id
    - Automatic full-text index management
    """

    def __init__(
        self,
        graph_id: str,
        retriever_type: RetrieverType = RetrieverType.VECTOR_CYPHER,
        retriever_config: dict[str, Any] | None = None,
        cache: QueryCacheService | None = None,
    ):
        self.graph_id = graph_id
        self.retriever_type = retriever_type
        self.retriever_config_dict = retriever_config or {}

        # COMMUNITY_SUMMARY bypasses the GraphRAG retriever pipeline — skip
        # RetrieverConfig construction (CommunitySummaryRetriever is instantiated
        # directly in _setup_retriever).
        if retriever_type == RetrieverType.COMMUNITY_SUMMARY:
            self.retriever_config = None  # type: ignore[assignment]
        else:
            default_config = get_default_retriever_config(retriever_type, graph_id)

            if retriever_type == RetrieverType.VECTOR:
                typed_config = cast(VectorRetrieverConfig, default_config)
            elif retriever_type == RetrieverType.VECTOR_CYPHER:
                typed_config = cast(VectorCypherRetrieverConfig, default_config)
            elif retriever_type == RetrieverType.HYBRID:
                typed_config = cast(HybridRetrieverConfig, default_config)
            elif retriever_type == RetrieverType.HYBRID_CYPHER:
                typed_config = cast(HybridCypherRetrieverConfig, default_config)
            elif retriever_type == RetrieverType.TEXT2CYPHER:
                typed_config = cast(Text2CypherRetrieverConfig, default_config)
            else:
                raise ValueError(f"Unsupported retriever type: {retriever_type}")

            self.retriever_config = RetrieverConfig(
                type=retriever_type, config=typed_config
            )

        self.embedder = OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY, model="text-embedding-3-large"
        )

        self.llm = OpenAILLM(
            model_name="gpt-4o",
            api_key=settings.OPENAI_API_KEY,
            model_params={"temperature": 0.1},
        )

        self.retriever = None
        self.rag = None
        self._cache: QueryCacheService | None = cache

        logger.info(
            f"ChatService initialized for graph {graph_id} "
            f"with {retriever_type.value} retriever"
        )

    async def initialize(self):
        """Async initialization of retriever and GraphRAG components."""
        await self._setup_retriever()

        if self.retriever_type == RetrieverType.COMMUNITY_SUMMARY:
            # CommunitySummaryRetriever does not use GraphRAG — skip rag setup.
            logger.info(
                f"CommunitySummaryRetriever initialized for graph {self.graph_id}"
            )
            return

        if self.retriever:
            self.rag = GraphRAG(retriever=self.retriever, llm=self.llm)
            logger.info(f"GraphRAG initialized successfully for graph {self.graph_id}")
        else:
            raise RuntimeError("Failed to initialize retriever and GraphRAG")

    async def _setup_retriever(self):
        """Set up retriever using factory pattern with full-text index management."""
        try:
            # CommunitySummaryRetriever is instantiated directly — it does not go
            # through the RetrieverFactory / GraphRAG pipeline.
            if self.retriever_type == RetrieverType.COMMUNITY_SUMMARY:
                self.retriever = CommunitySummaryRetriever(graph_id=self.graph_id)
                logger.info(
                    f"Created CommunitySummaryRetriever for graph {self.graph_id}"
                )
                return

            if self.retriever_type in [
                RetrieverType.HYBRID,
                RetrieverType.HYBRID_CYPHER,
            ]:
                await fulltext_index_manager.setup_default_indexes(self.graph_id)

            self.retriever = await retriever_factory.create_retriever(
                retriever_config=self.retriever_config, graph_id=self.graph_id
            )

            if not self.retriever:
                raise RuntimeError(
                    f"Factory failed to create {self.retriever_type.value} retriever"
                )

        except Exception as e:
            logger.error(f"Failed to setup retriever: {e}")
            try:
                fallback_config = get_default_retriever_config(
                    RetrieverType.VECTOR, self.graph_id
                )
                typed_fallback = cast(VectorRetrieverConfig, fallback_config)
                fallback_retriever_config = RetrieverConfig(
                    type=RetrieverType.VECTOR, config=typed_fallback
                )
                self.retriever = await retriever_factory.create_retriever(
                    retriever_config=fallback_retriever_config, graph_id=self.graph_id
                )
                self.retriever_config = fallback_retriever_config
                self.retriever_type = RetrieverType.VECTOR
                logger.warning(
                    f"Using fallback vector retriever for graph {self.graph_id}"
                )
            except Exception as fallback_error:
                logger.error(f"Fallback retriever creation failed: {fallback_error}")
                raise RuntimeError("Failed to create any retriever") from fallback_error

    # ------------------------------------------------------------------
    # Temporal WHERE clause builder (TASK-007)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_temporal_filter(
        temporal_mode: TemporalMode | None,
        temporal_at: datetime | None,
        temporal_since: datetime | None,
    ) -> tuple[str, dict[str, Any]]:
        """Build a trusted Cypher WHERE clause fragment and parameter dict.

        The clause text is constructed from internal constants only — it never
        contains user-supplied strings.  All datetime values are bound as Cypher
        parameters ($temporal_at / $temporal_since) and must be forwarded via
        query_params; they are never interpolated into the query string.

        Modes:
            POINT_IN_TIME:   relationships where event_time <= $temporal_at
                             AND (event_time_end IS NULL OR event_time_end >= $temporal_at)
            KNOWLEDGE_AS_OF: relationships where ingestion_time <= $temporal_at
            CHANGES_SINCE:   relationships where ingestion_time > $temporal_since

        Returns:
            (clause, params) — both empty when temporal_mode is None, preserving
            identical behavior to pre-TASK-007 callers (backward compatible).
        """
        if temporal_mode == TemporalMode.POINT_IN_TIME:
            clause = (
                "(r.event_time IS NULL OR r.event_time <= $temporal_at)"
                " AND (r.event_time_end IS NULL OR r.event_time_end >= $temporal_at)"
            )
            params: dict[str, Any] = {
                "temporal_at": temporal_at.isoformat() if temporal_at else None
            }
        elif temporal_mode == TemporalMode.KNOWLEDGE_AS_OF:
            clause = "r.ingestion_time <= $temporal_at"
            params = {"temporal_at": temporal_at.isoformat() if temporal_at else None}
        elif temporal_mode == TemporalMode.CHANGES_SINCE:
            clause = "r.ingestion_time > $temporal_since"
            params = {
                "temporal_since": (
                    temporal_since.isoformat() if temporal_since else None
                )
            }
        else:
            return "", {}

        return clause, params

    async def search_cached(
        self,
        query_text: str,
        retriever_config: dict[str, Any] | None = None,
        return_context: bool = False,
        examples: str = "",
        temporal_filter: TemporalFilter | None = None,
        temporal_mode: TemporalMode | None = None,
        temporal_at: datetime | None = None,
        temporal_since: datetime | None = None,
    ) -> "tuple[GroundedSearchResult, bool]":
        """Cache-aware wrapper around search().

        Checks the Redis query cache before invoking the full RAG pipeline.
        Cache key includes graph_id and retriever_type — guarantees cross-tenant
        isolation (Architecture Rule: graph_id on every key).

        Temporal queries are NOT cached: a temporal filter changes the result set
        and would require a different key per timestamp, making invalidation
        impractical. Only stateless (non-temporal) queries are cached.

        Returns:
            (result, cache_hit) — cache_hit is True when the result came from Redis.
        """
        # Skip cache entirely when temporal filtering is active:
        # results are time-scoped and may change without a new ingest event.
        is_temporal = bool(temporal_mode or temporal_filter)

        if self._cache and not is_temporal:
            cached_dict = await self._cache.get(
                self.graph_id, query_text, self.retriever_type.value
            )
            if cached_dict is not None:
                logger.info(
                    "Cache HIT for graph %s retriever %s",
                    self.graph_id,
                    self.retriever_type.value,
                )
                result = GroundedSearchResult(
                    answer=cached_dict.get("answer", ""),
                    sources=cached_dict.get("sources", []),
                    confidence=cached_dict.get("confidence", 0.0),
                    is_grounded=cached_dict.get("is_grounded", True),
                    retriever_used=cached_dict.get(
                        "retriever_used", self.retriever_type.value
                    ),
                )
                return result, True

        # Cache miss (or cache disabled / temporal query) — run full pipeline.
        result = await self.search(
            query_text=query_text,
            retriever_config=retriever_config,
            return_context=return_context,
            examples=examples,
            temporal_filter=temporal_filter,
            temporal_mode=temporal_mode,
            temporal_at=temporal_at,
            temporal_since=temporal_since,
        )

        if self._cache and not is_temporal:
            await self._cache.set(
                self.graph_id,
                query_text,
                self.retriever_type.value,
                {
                    "answer": result.answer,
                    "sources": result.sources,
                    "confidence": result.confidence,
                    "is_grounded": result.is_grounded,
                    "retriever_used": result.retriever_used,
                },
            )

        return result, False

    async def search(
        self,
        query_text: str,
        retriever_config: dict[str, Any] | None = None,
        return_context: bool = False,
        examples: str = "",
        temporal_filter: TemporalFilter | None = None,
        temporal_mode: TemporalMode | None = None,
        temporal_at: datetime | None = None,
        temporal_since: datetime | None = None,
    ) -> GroundedSearchResult:
        """
        Perform a graph-grounded GraphRAG search with hallucination prevention.

        Always retrieves context internally to:
        - Detect insufficient data before passing to the LLM
        - Extract source citations from retrieved graph nodes
        - Calculate confidence from retrieval scores
        - Use the strict grounding prompt to prevent external knowledge leakage

        Args:
            query_text: User's question
            retriever_config: Configuration for retriever (e.g., top_k)
            return_context: Whether to include retriever_result in the returned object
            examples: Examples for few-shot learning
            temporal_filter: Legacy TemporalFilter for multi-hop enrichment scoping
            temporal_mode: Bitemporal query mode (point_in_time / knowledge_as_of /
                changes_since).  When set, injects a Cypher WHERE clause on
                relationship temporal properties via query_params.  Omitting this
                produces identical behavior to the pre-TASK-007 baseline.
            temporal_at: Reference timestamp for point_in_time and knowledge_as_of.
            temporal_since: Reference timestamp for changes_since.

        Returns:
            GroundedSearchResult with answer, sources, confidence, and grounding flag
        """
        # Build temporal WHERE clause + params (TASK-007).
        # An empty clause means no temporal filtering — backward compatible.
        temporal_clause, temporal_params = self._build_temporal_filter(
            temporal_mode, temporal_at, temporal_since
        )

        with _chat_tracer.start_as_current_span(
            "chat.query",
            kind=otel_trace.SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("graph_id", self.graph_id)
            span.set_attribute("chat.retriever_type", self.retriever_type.value)
            span.set_attribute("chat.query.length", len(query_text))
            if temporal_mode:
                span.set_attribute("chat.temporal_mode", temporal_mode.value)
            return await self._search_inner(
                span,
                query_text,
                retriever_config,
                return_context,
                examples,
                temporal_filter,
                temporal_clause,
                temporal_params,
            )

    async def _search_inner(
        self,
        span,
        query_text: str,
        retriever_config: dict[str, Any] | None,
        return_context: bool,
        examples: str,
        temporal_filter: TemporalFilter | None,
        temporal_clause: str = "",
        temporal_params: dict[str, Any] | None = None,
    ) -> "GroundedSearchResult":
        try:
            # ------------------------------------------------------------------
            # Community summary fast path — bypasses GraphRAG vector pipeline.
            # ------------------------------------------------------------------
            if self.retriever_type == RetrieverType.COMMUNITY_SUMMARY:
                return await self._community_summary_search(span, query_text)

            if not self.rag:
                await self.initialize()

            if not self.rag:
                raise RuntimeError("GraphRAG not properly initialized")

            logger.info(f"Processing grounded search for graph {self.graph_id}")

            # Merge temporal query parameters into the retriever_config so that
            # Cypher-capable retrievers (VECTOR_CYPHER, HYBRID_CYPHER) can bind
            # the $temporal_at / $temporal_since placeholders in their queries.
            # Values are always passed as parameters — never interpolated into query text.
            effective_retriever_config = dict(retriever_config or {"top_k": 5})
            if temporal_clause and temporal_params:
                existing_params = effective_retriever_config.get("query_params") or {}
                effective_retriever_config["query_params"] = {
                    **existing_params,
                    **temporal_params,
                }

            # Always request context so we can inspect retrieved items.
            raw_result: RagResultModel = self.rag.search(
                query_text=query_text,
                retriever_config=effective_retriever_config,
                return_context=True,
                examples=examples,
                prompt_template=STRICT_GROUNDING_PROMPT,
            )

            # Collect and sort retriever items by relevance score (desc).
            retriever_items = []
            if raw_result.retriever_result:
                retriever_items = list(
                    getattr(raw_result.retriever_result, "items", []) or []
                )
            retriever_items.sort(
                key=lambda item: getattr(item, "score", None) or 0.0, reverse=True
            )

            # If retrieval is sparse, attempt 2-hop entity-anchor enrichment.
            if len(retriever_items) < 3:
                entity_candidates = self.detect_entity_candidates(query_text)
                if entity_candidates:
                    multihop_rows = await self._multihop_enrich(
                        entity_candidates, temporal_filter=temporal_filter
                    )
                    if multihop_rows:
                        logger.info(
                            f"Multi-hop enrichment added {len(multihop_rows)} rows "
                            f"for graph {self.graph_id}"
                        )
                        # Re-run GraphRAG with enriched context hint in retriever_config.
                        # We append the hop data as additional context in a second pass
                        # only when the original retrieval returned nothing useful.
                        if len(retriever_items) < _MIN_CONTEXT_ITEMS:
                            hop_context = "; ".join(
                                f"{r['anchor']} -[{r['rel1']}]-> {r['hop1_name']}"
                                for r in multihop_rows
                                if r.get("anchor") and r.get("hop1_name")
                            )
                            augmented_query = (
                                f"{query_text}\n\n[Graph context: {hop_context}]"
                            )
                            raw_result = self.rag.search(
                                query_text=augmented_query,
                                retriever_config=effective_retriever_config,
                                return_context=True,
                                examples=examples,
                                prompt_template=STRICT_GROUNDING_PROMPT,
                            )
                            if raw_result.retriever_result:
                                retriever_items = list(
                                    getattr(raw_result.retriever_result, "items", [])
                                    or []
                                )
                                retriever_items.sort(
                                    key=lambda item: getattr(item, "score", None)
                                    or 0.0,
                                    reverse=True,
                                )

            # No context retrieved — return structured no-data response.
            if len(retriever_items) < _MIN_CONTEXT_ITEMS:
                logger.warning(
                    f"No graph context retrieved for graph {self.graph_id}. "
                    "Returning insufficient-data response."
                )
                return GroundedSearchResult(
                    answer=(
                        "The knowledge graph does not contain sufficient data "
                        "to answer this question."
                    ),
                    sources=[],
                    confidence=0.0,
                    is_grounded=False,
                    retriever_used=self.retriever_type.value,
                    retriever_result=(
                        raw_result.retriever_result if return_context else None
                    ),
                )

            sources = self._extract_sources(retriever_items)
            confidence = self._calculate_confidence(retriever_items)

            # Detect if the LLM itself signalled insufficient data.
            answer = raw_result.answer or ""
            is_grounded = not answer.strip().startswith(_INSUFFICIENT_PREFIX)

            if not is_grounded:
                answer = (
                    "The knowledge graph does not contain sufficient data "
                    "to answer this question."
                )
                confidence = max(confidence * 0.3, 0.0)

            logger.info(
                f"Grounded search complete — grounded={is_grounded}, "
                f"confidence={confidence:.2f}, sources={len(sources)}"
            )

            span.set_attribute("chat.is_grounded", is_grounded)
            span.set_attribute("chat.confidence", round(confidence, 4))
            span.set_attribute("chat.sources_count", len(sources))
            span.set_attribute("chat.hallucination_flag", not is_grounded)

            return GroundedSearchResult(
                answer=answer,
                sources=sources,
                confidence=confidence,
                is_grounded=is_grounded,
                retriever_used=self.retriever_type.value,
                retriever_result=(
                    raw_result.retriever_result if return_context else None
                ),
            )

        except _LLM_DOWN_EXCEPTIONS as e:
            # LLM timeout or rate-limit — return available graph context without an answer.
            _llm_err_code, _llm_err_msg = KGBError.LLM_UNAVAILABLE
            logger.warning(
                f"LLM unavailable for graph {self.graph_id} [{_llm_err_code}]: {e}"
            )
            span.record_exception(e)
            span.set_status(otel_trace.StatusCode.ERROR, str(e))
            return GroundedSearchResult(
                answer=None,  # type: ignore[arg-type]
                sources=[],
                confidence=0.0,
                is_grounded=False,
                retriever_used=self.retriever_type.value,
                retriever_result=None,
            )
        except Exception as e:
            logger.error(f"GraphRAG search failed for graph {self.graph_id}: {e}")
            span.record_exception(e)
            span.set_status(otel_trace.StatusCode.ERROR, str(e))
            return GroundedSearchResult(
                answer=f"An error occurred while searching the knowledge graph: {str(e)}",
                sources=[],
                confidence=0.0,
                is_grounded=False,
                retriever_used=self.retriever_type.value,
            )

    # ------------------------------------------------------------------
    # Community summary retrieval
    # ------------------------------------------------------------------

    async def _community_summary_search(
        self, span, query_text: str
    ) -> GroundedSearchResult:
        """
        Retrieve Leiden community summaries and synthesise an answer via the LLM.

        Bypasses the GraphRAG vector pipeline.  Summaries from Neo4j are passed
        as plain text context; they are never interpolated into Cypher queries.
        """
        if not self.retriever:
            await self.initialize()

        assert isinstance(self.retriever, CommunitySummaryRetriever)

        community_docs = await self.retriever.search(query_text)

        if not community_docs:
            logger.warning("No community summaries found for graph %s", self.graph_id)
            span.set_attribute("chat.is_grounded", False)
            span.set_attribute("chat.confidence", 0.0)
            span.set_attribute("chat.sources_count", 0)
            span.set_attribute("chat.hallucination_flag", True)
            return GroundedSearchResult(
                answer=(
                    "The knowledge graph does not contain sufficient data "
                    "to answer this question."
                ),
                sources=[],
                confidence=0.0,
                is_grounded=False,
                retriever_used=RetrieverType.COMMUNITY_SUMMARY.value,
            )

        # Build plain-text context from community summaries.
        context_parts = [
            f"[Community {doc['community_id']} — {doc['entity_count']} entities]\n"
            f"{doc['summary']}"
            for doc in community_docs
        ]
        context = "\n\n".join(context_parts)

        prompt = STRICT_GROUNDING_PROMPT.format(context=context, query_text=query_text)
        try:
            llm_response = await self.llm.ainvoke(prompt)
            answer = llm_response.content
        except _LLM_DOWN_EXCEPTIONS as e:
            _llm_err_code, _llm_err_msg = KGBError.LLM_UNAVAILABLE
            logger.warning(
                f"LLM unavailable during community summary search for graph "
                f"{self.graph_id} [{_llm_err_code}]: {e}"
            )
            # Return context nodes without an LLM-synthesised answer.
            sources = [
                {
                    "node_id": doc["community_id"],
                    "node_labels": ["__Community__"],
                    "content": doc["summary"][:500],
                    "relevance_score": None,
                    "properties": {"entity_count": doc["entity_count"]},
                }
                for doc in community_docs
            ]
            span.set_attribute("chat.is_grounded", False)
            span.set_attribute("chat.confidence", 0.0)
            span.set_attribute("chat.sources_count", len(sources))
            span.set_attribute("chat.hallucination_flag", True)
            return GroundedSearchResult(
                answer=None,  # type: ignore[arg-type]
                sources=sources,
                confidence=0.0,
                is_grounded=False,
                retriever_used=RetrieverType.COMMUNITY_SUMMARY.value,
            )

        is_grounded = not answer.strip().startswith(_INSUFFICIENT_PREFIX)
        if not is_grounded:
            answer = (
                "The knowledge graph does not contain sufficient data "
                "to answer this question."
            )

        sources = [
            {
                "node_id": doc["community_id"],
                "node_labels": ["__Community__"],
                "content": doc["summary"][:500],
                "relevance_score": None,
                "properties": {"entity_count": doc["entity_count"]},
            }
            for doc in community_docs
        ]
        confidence = 0.7 if is_grounded else 0.0

        logger.info(
            "Community summary search complete — grounded=%s, sources=%d",
            is_grounded,
            len(sources),
        )
        span.set_attribute("chat.is_grounded", is_grounded)
        span.set_attribute("chat.confidence", confidence)
        span.set_attribute("chat.sources_count", len(sources))
        span.set_attribute("chat.hallucination_flag", not is_grounded)

        return GroundedSearchResult(
            answer=answer,
            sources=sources,
            confidence=confidence,
            is_grounded=is_grounded,
            retriever_used=RetrieverType.COMMUNITY_SUMMARY.value,
        )

    # ------------------------------------------------------------------
    # Auto retriever selection
    # ------------------------------------------------------------------

    @staticmethod
    def auto_select_retriever_type(query: str) -> RetrieverType:
        """
        Choose the most appropriate retriever type based on query characteristics.

        Rules (evaluated in priority order):
        1. Global/thematic keywords → COMMUNITY_SUMMARY for broad overview queries
        2. Cypher/graph-query keywords → TEXT2CYPHER for precise traversal
        3. Analytic / enumeration keywords → HYBRID for broader coverage
        4. Relationship / connectivity keywords → VECTOR_CYPHER for graph traversal
        5. Default → VECTOR_CYPHER (balanced precision + context)
        """
        if _is_global_query(query):
            return RetrieverType.COMMUNITY_SUMMARY
        if _CYPHER_PATTERNS.search(query):
            return RetrieverType.TEXT2CYPHER
        if _ANALYTIC_PATTERNS.search(query):
            return RetrieverType.HYBRID
        if _RELATIONSHIP_PATTERNS.search(query):
            return RetrieverType.VECTOR_CYPHER
        return RetrieverType.VECTOR_CYPHER

    # ------------------------------------------------------------------
    # Entity anchor detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_entity_candidates(query: str) -> list[str]:
        """
        Extract likely named-entity tokens from a query string.

        Heuristic: consecutive capitalised words (e.g. "TechNova Corp")
        that do not appear in the stopword list and have length > 2.
        Returns de-duplicated candidates preserving order.
        """
        # Split on whitespace/punctuation but keep original case.
        tokens = re.split(r"[\s,;:!?.()\[\]\"']+", query)
        candidates: list[str] = []
        seen: set = set()

        # Merge consecutive capital-starting tokens into multi-word names.
        buffer: list[str] = []
        for tok in tokens:
            if len(tok) > 2 and tok[0].isupper() and tok.lower() not in _STOPWORDS:
                buffer.append(tok)
            else:
                if buffer:
                    phrase = " ".join(buffer)
                    if phrase not in seen:
                        candidates.append(phrase)
                        seen.add(phrase)
                    buffer = []
        if buffer:
            phrase = " ".join(buffer)
            if phrase not in seen:
                candidates.append(phrase)

        return candidates

    # ------------------------------------------------------------------
    # Multi-hop enrichment
    # ------------------------------------------------------------------

    async def _multihop_enrich(
        self,
        entity_candidates: list[str],
        top_k: int = 3,
        temporal_filter: TemporalFilter | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run 2-hop Cypher traversal anchored on detected entity candidates.

        Uses the AsyncDriver (FastAPI-safe). Returns a list of context dicts
        that can be injected alongside retriever results.
        Always filters by self.graph_id — multi-tenancy enforced.
        When temporal_filter is set, restricts r1 to facts valid at the given time.
        """
        enriched: list[dict[str, Any]] = []
        driver = neo4j_client.async_driver
        if driver is None:
            logger.warning("Async driver unavailable — skipping multi-hop enrichment")
            return enriched

        # Build optional temporal WHERE clause for the first-hop relationship.
        # ORA-138: Use Cypher parameters ($tf_pit) instead of f-string datetime
        # interpolation to avoid injection and enable rel_valid_from/to indexes.
        temporal_clause = ""
        temporal_params: dict[str, Any] = {}
        if temporal_filter:
            if temporal_filter.current_only:
                temporal_clause = "AND r1.valid_to IS NULL"
            elif temporal_filter.point_in_time:
                temporal_params["tf_pit"] = temporal_filter.point_in_time.isoformat()
                temporal_clause = (
                    "AND (r1.valid_from IS NULL OR r1.valid_from <= datetime($tf_pit))"
                    " AND (r1.valid_to IS NULL OR r1.valid_to > datetime($tf_pit))"
                )

        cypher = (
            f"""
MATCH (anchor:__Entity__ {{graph_id: $graph_id}})
WHERE toLower(anchor.name) CONTAINS toLower($entity_name)
MATCH (anchor)-[r1]->(hop1:__Entity__ {{graph_id: $graph_id}})
WHERE true {temporal_clause}
OPTIONAL MATCH (hop1)-[r2]->(hop2:__Entity__ {{graph_id: $graph_id}})
RETURN
    anchor.name        AS anchor_name,
    anchor.description AS anchor_desc,
    type(r1)           AS rel1,
    hop1.name          AS hop1_name,
    hop1.description   AS hop1_desc,
    type(r2)           AS rel2,
    hop2.name          AS hop2_name,
    hop2.description   AS hop2_desc
LIMIT 20
"""
            if temporal_clause
            else _MULTIHOP_CYPHER
        )

        for entity_name in entity_candidates[:top_k]:
            try:
                result = await driver.execute_query(
                    cypher,
                    {
                        "graph_id": self.graph_id,
                        "entity_name": entity_name,
                        **temporal_params,
                    },
                )
                records = result.records if hasattr(result, "records") else result[0]
                for rec in records:
                    entry: dict[str, Any] = {
                        "anchor": rec.get("anchor_name"),
                        "anchor_desc": rec.get("anchor_desc"),
                        "hop1_name": rec.get("hop1_name"),
                        "hop1_desc": rec.get("hop1_desc"),
                        "rel1": rec.get("rel1"),
                        "hop2_name": rec.get("hop2_name"),
                        "hop2_desc": rec.get("hop2_desc"),
                        "rel2": rec.get("rel2"),
                        "graph_id": self.graph_id,
                        "_source": "multihop",
                    }
                    enriched.append(entry)
            except Exception as exc:
                logger.warning(
                    f"Multi-hop traversal failed for entity '{entity_name}' "
                    f"in graph {self.graph_id}: {exc}"
                )

        return enriched

    # ------------------------------------------------------------------
    # Streaming search
    # ------------------------------------------------------------------

    async def stream_search(
        self,
        query_text: str,
        retriever_config: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """
        Streaming variant of search().

        Yields Server-Sent Events (SSE) formatted strings:
          - data: {"type": "source", ...}  — one per retrieved source
          - data: {"type": "answer_chunk", "text": "..."}  — answer text chunks
          - data: {"type": "done", "confidence": 0.9, "is_grounded": true}

        Falls back to a single chunk if the underlying GraphRAG doesn't
        support token-level streaming.
        """
        try:
            if not self.rag:
                await self.initialize()

            if not self.rag:
                raise RuntimeError("GraphRAG not properly initialized")

            raw_result: RagResultModel = self.rag.search(
                query_text=query_text,
                retriever_config=retriever_config or {"top_k": 5},
                return_context=True,
                prompt_template=STRICT_GROUNDING_PROMPT,
            )

            retriever_items = []
            if raw_result.retriever_result:
                retriever_items = list(
                    getattr(raw_result.retriever_result, "items", []) or []
                )
            retriever_items.sort(
                key=lambda item: getattr(item, "score", None) or 0.0,
                reverse=True,
            )

            import json

            # Emit each source first.
            sources = self._extract_sources(retriever_items)
            for src in sources:
                yield f"data: {json.dumps({'type': 'source', **src})}\n\n"

            if len(retriever_items) < _MIN_CONTEXT_ITEMS:
                answer = (
                    "The knowledge graph does not contain sufficient data "
                    "to answer this question."
                )
                yield f"data: {json.dumps({'type': 'answer_chunk', 'text': answer})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'confidence': 0.0, 'is_grounded': False})}\n\n"
                return

            answer = raw_result.answer or ""
            is_grounded = not answer.strip().startswith(_INSUFFICIENT_PREFIX)
            if not is_grounded:
                answer = (
                    "The knowledge graph does not contain sufficient data "
                    "to answer this question."
                )

            confidence = self._calculate_confidence(retriever_items)
            if not is_grounded:
                confidence = max(confidence * 0.3, 0.0)

            # Emit answer in word-level chunks to simulate streaming.
            words = answer.split(" ")
            for i in range(0, len(words), 10):
                chunk = " ".join(words[i : i + 10])
                if i + 10 < len(words):
                    chunk += " "
                yield f"data: {json.dumps({'type': 'answer_chunk', 'text': chunk})}\n\n"

            yield (
                f"data: {json.dumps({'type': 'done', 'confidence': confidence, 'is_grounded': is_grounded, 'retriever_used': self.retriever_type.value})}\n\n"
            )

        except Exception as exc:
            import json

            logger.error(f"Streaming search failed for graph {self.graph_id}: {exc}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_sources(retriever_items: list[Any]) -> list[dict[str, Any]]:
        """
        Extract structured source citations from retriever result items.

        Each item's `metadata` dict typically contains node properties
        (id, labels, name, text, score) depending on the retriever type
        and the configured return_properties / retrieval_query.
        """
        sources: list[dict[str, Any]] = []
        for item in retriever_items:
            metadata: dict[str, Any] = getattr(item, "metadata", {}) or {}
            content: str = getattr(item, "content", "") or ""

            source: dict[str, Any] = {
                "node_id": metadata.get("id") or metadata.get("elementId"),
                "node_labels": metadata.get("labels"),
                "content": content[:500] if content else None,
                "relevance_score": (
                    getattr(item, "score", None) or metadata.get("score")
                ),
                "properties": {
                    k: v
                    for k, v in metadata.items()
                    if k not in {"id", "elementId", "labels", "score", "embedding"}
                },
            }
            sources.append(source)
        return sources

    @staticmethod
    def _calculate_confidence(retriever_items: list[Any]) -> float:
        """
        Compute a confidence score [0, 1] from retriever relevance scores.

        Uses the mean of the top-3 scores (or fewer if less are available),
        clamped to [0, 1].  Falls back to a low baseline if no scores exist.
        """
        scores: list[float] = []
        for item in retriever_items[:3]:
            score = getattr(item, "score", None)
            if score is not None:
                try:
                    scores.append(float(score))
                except (TypeError, ValueError):
                    pass

        if not scores:
            # Context exists but scores are unavailable — moderate confidence.
            return 0.5

        return min(max(sum(scores) / len(scores), 0.0), 1.0)
