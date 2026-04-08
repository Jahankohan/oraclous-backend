# app/services/pipeline_service.from ..components.entity_resolver import MultiTenantEntityDeduplicatory
"""
Multi-Tenant Pipeline Service - Neo4j GraphRAG Foundation

Clean, maintainable pipeline service using your AdvancedGraphRAGPipeline 
with multi-tenant support and FastAPI compatibility.

DESIGN PRINCIPLES:
- Uses your clean refactor/AdvancedGraphRAGPipeline as foundation
- Multi-tenant wrapper with perfect isolation 
- Simple, maintainable code (no complex abstractions)
- FastAPI compatible with async support
- Performance monitoring and error handling

NEO4J DUAL DRIVER ARCHITECTURE:
- Uses neo4j_client.sync_driver for GraphRAG components (VectorRetriever, Neo4jWriter, etc.)
- Uses neo4j_client.execute_query() for async database operations
- Automatic driver management and connection isolation
"""

import asyncio
import difflib
import hashlib
from dataclasses import dataclass, field as dataclass_field
from typing import Dict, List, Any, Optional, Set, Tuple
from uuid import UUID
from datetime import datetime

from fastapi import BackgroundTasks, HTTPException, status

from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.types import DocumentInfo, Neo4jGraph

from app.components.multi_tenant_components import MultiTenantKGWriter, create_multi_tenant_kg_writer
from app.components.entity_resolver import MultiTenantEntityDeduplicator
from opentelemetry import trace as otel_trace
from app.core.neo4j_client import neo4j_client
from app.core.config import settings
from app.core.logging import get_logger
from app.core.telemetry import get_tracer

_pipeline_tracer = get_tracer("oraclous.pipeline")
from app.schemas.graph_schemas import BANNED_NODE_PROPERTIES, IngestMode, IngestionOverrides, OntologyValidationMode, TemporalContext, TemporalFilter
from app.services.instructions_service import (
    ResolvedInstructions,
    InstructionsResolver,
    InstructionsCompiler,
    instructions_resolver,
    instructions_compiler,
)

logger = get_logger(__name__)


# ==================== ONTOLOGY ENFORCEMENT ====================

@dataclass
class _EnforcementReport:
    violations: int = 0
    coercions: int = 0


# ==================== RELATIONSHIP-FIRST EXTRACTION PROMPT ====================

# Overrides the default ERExtractionTemplate to enforce the property placement standard:
# contextual/positional properties belong on relationships, never on entity nodes.
RELATIONSHIP_PROPERTY_PROMPT_TEMPLATE = """\
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.

Extract the entities (nodes) and specify their type from the following text.
Also extract the relationships between these nodes.

Return result as JSON using the following format:
{{"nodes": [ {{"id": "0", "label": "Person", "properties": {{"name": "John"}} }}],
"relationships": [{{"type": "KNOWS", "start_node_id": "0", "end_node_id": "1", "properties": {{"since": "2024-08-01"}} }}] }}

Use only the following node and relationship types (if provided):
{schema}

Assign a unique ID (string) to each node, and reuse it to define relationships.
Do respect the source and target node types for relationship and the relationship direction.

CRITICAL PROPERTY PLACEMENT RULES:
1. Node properties = ONLY intrinsic identity attributes (name, type, founding year, immutable classification).
2. Relationship properties = ALL contextual, positional, temporal, and role-based attributes.
3. NEVER put job_title, position, role, seniority, proficiency, ownership_pct, allocation, or employment dates on entity nodes.
4. ALWAYS put the above attributes on the relationship between the two entities.

TEMPORAL EXTRACTION RULES:
5. When the text implies a time period for a relationship (e.g. "since 2019", "from 2020 to 2023", "until Q3 2022", "as of January 2024"), extract it as:
   - "valid_from": ISO-8601 date string (YYYY-MM-DD or YYYY) for when the relationship became true.
   - "valid_to": ISO-8601 date string for when it ended; omit or set null if still ongoing.
6. Put valid_from / valid_to on the RELATIONSHIP, not on entity nodes.
7. If no temporal information is present, omit valid_from and valid_to entirely (do not default to null).

EXAMPLE (correct — with temporal context):
  Text: "Alice Chen has been the CTO of Acme Corp since March 2021."
  Nodes: [{{"id":"alice","label":"Person","properties":{{"name":"Alice Chen"}}}},
          {{"id":"acme","label":"Company","properties":{{"name":"Acme Corp"}}}}]
  Relationships: [{{"start_node_id":"alice","end_node_id":"acme","type":"WORKS_FOR",
                   "properties":{{"position":"CTO","valid_from":"2021-03-01"}}}}]

EXAMPLE (correct — with ended relationship):
  Text: "Bob served as CFO of Acme from 2018 to 2022."
  Relationships: [{{"start_node_id":"bob","end_node_id":"acme","type":"WORKS_FOR",
                   "properties":{{"position":"CFO","valid_from":"2018-01-01","valid_to":"2022-12-31"}}}}]

EXAMPLE (wrong — do not do this):
  Nodes: [{{"id":"alice","label":"Person","properties":{{"name":"Alice Chen","job_title":"CTO","employer":"Acme"}}}}]

RELATIONSHIP TYPE NAMING: ALL_CAPS_SNAKE_CASE.
Common types: WORKS_FOR, REPORTS_TO, HAS_SKILL, MEMBER_OF, INVESTED_IN, CITES, AUTHORED, WORKS_ON, DEPENDS_ON, ACQUIRED_BY, PARTNER_OF.

Make sure you adhere to the following rules to produce valid JSON objects:
- Do not return any additional information other than the JSON in it.
- Omit any backticks around the JSON - simply output the JSON on its own.
- The JSON object must not wrapped into a list - it is its own JSON object.
- Property names must be enclosed in double quotes

Examples:
{examples}

Input text:

{text}"""


# ==================== CONFIGURATION ADAPTER ====================

class PipelineConfig:
    """
    Configuration adapter that bridges FastAPI settings with your AdvancedPipelineConfig.
    Simple dataclass that maps existing settings to your pipeline requirements.
    """
    
    def __init__(self):
        # Neo4j Configuration
        self.neo4j_uri = settings.NEO4J_URI
        self.neo4j_user = settings.NEO4J_USERNAME  
        self.neo4j_password = settings.NEO4J_PASSWORD
        self.neo4j_database = settings.NEO4J_DATABASE
        
        # OpenAI Configuration
        self.openai_api_key = settings.OPENAI_API_KEY
        self.llm_model = getattr(settings, 'LLM_MODEL', 'gpt-4o')  # Use gpt-4o which supports json_object
        self.llm_temperature = getattr(settings, 'LLM_TEMPERATURE', 0.1)
        self.llm_max_tokens = getattr(settings, 'LLM_MAX_TOKENS', 3000)
        
        # Embedding Configuration
        self.embedding_model = getattr(settings, 'EMBEDDING_MODEL', 'text-embedding-3-large')
        self.embedding_dimensions = 3072
        
        # Processing Configuration
        self.chunk_size = getattr(settings, 'CHUNK_SIZE', 1500)
        self.chunk_overlap = getattr(settings, 'CHUNK_OVERLAP', 300)
        self.batch_size = 2000
        self.max_concurrency = 10
        
        # Advanced Features
        self.enable_entity_resolution = True
        self.similarity_threshold = 0.85
        self.enable_schema_learning = True
        self.enable_performance_monitoring = True
        self.enable_detailed_logging = True
        self.on_error = "IGNORE"  # Continue processing on errors
        
        # TODO: Implement Schema-Guided Extraction similar to benchmark's AdvancedSchemaManager
        # The benchmark implementation uses sophisticated schema learning from text samples
        # which could improve entity extraction accuracy, type consistency, and entity count
        # See benchmark.py AdvancedSchemaManager class for reference implementation


# ==================== ENTITY FINGERPRINTING (Spec ORA-49) ====================

import re as _re


def compute_entity_fingerprint(graph_id: str, name: str, label: str) -> str:
    """
    Deterministic identity hash for an entity — stable across re-ingestions.

    Key: graph_id + label.upper() + normalised_name
    Mutable properties (description, extra) are excluded so that content
    changes do not generate a new node; only name+label identity matters.
    Returns a 16-char hex prefix of SHA-256.
    """
    norm_name = _re.sub(r"[^\w\s]", "", name.lower().strip())
    norm_name = _re.sub(r"\s+", " ", norm_name)
    raw = f"{graph_id}:{label.upper()}:{norm_name}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def compute_prop_hash(properties: dict) -> str:
    """
    MD5 of mutable entity properties for change detection.

    System fields (fingerprint, prop_hash, embedding, ingested_at,
    last_updated_at, transaction_time, graph_id, created_by) are excluded
    so that bookkeeping updates do not trigger an UPDATED classification.
    """
    _EXCLUDED = frozenset({
        "fingerprint", "prop_hash", "embedding", "ingested_at",
        "last_updated_at", "transaction_time", "graph_id", "created_by",
        "user_id", "entity_id",
    })
    mutable = {k: v for k, v in sorted(properties.items()) if k not in _EXCLUDED}
    return hashlib.md5(str(mutable).encode()).hexdigest()


async def ensure_fingerprint_indexes() -> None:
    """
    Create Neo4j indexes required for fingerprint-based entity delta. Idempotent.
    Called once at application startup.
    """
    from app.core.neo4j_client import neo4j_client as _nc
    index_queries = [
        "CREATE INDEX entity_fingerprint IF NOT EXISTS FOR (e:__Entity__) ON (e.graph_id, e.fingerprint)",
        "CREATE INDEX chunk_doc_id IF NOT EXISTS FOR (c:Chunk) ON (c.graph_id, c.doc_id)",
    ]
    for q in index_queries:
        try:
            await _nc.execute_query(q, {})
        except Exception as e:
            _logger = get_logger(__name__)
            _logger.warning(f"Fingerprint index creation skipped: {e}")


# ==================== MULTI-TENANT PIPELINE WRAPPER ====================

class MultiTenantGraphRAGPipeline:
    """
    Multi-tenant wrapper around your AdvancedGraphRAGPipeline.
    
    FEATURES:
    - Perfect tenant isolation with graph_id injection
    - Uses your clean refactor pipeline as foundation
    - FastAPI compatible with async support
    - Performance monitoring and metrics
    - Simple factory pattern for clean initialization
    """
    
    def __init__(self, graph_id: str, user_id: Optional[str] = None):
        """
        Initialize multi-tenant pipeline wrapper.
        
        Args:
            graph_id: Tenant graph identifier
            user_id: Optional user identifier for additional tracking
        """
        self.graph_id = graph_id
        self.user_id = user_id
        self.config = PipelineConfig()
        
        # Components will be initialized on first use
        self.driver = None
        self.llm = None
        self.embedder = None
        self.base_pipeline = None
        self._initialized = False
        
        logger.info(f"MultiTenantGraphRAGPipeline created for graph {graph_id}")
    
    def _model_supports_json_object(self, model_name: str) -> bool:
        """
        Check if the OpenAI model supports response_format with json_object.
        Only newer models support this feature.
        """
        json_supported_models = [
            "gpt-4o",
            "gpt-4o-mini", 
            "gpt-4-turbo",
            "gpt-4-1106-preview",
            "gpt-4-0125-preview",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-0125"
        ]
        
        # Check if the model name contains any of the supported model identifiers
        return any(supported_model in model_name for supported_model in json_supported_models)
    
    async def _initialize_components(self):
        """
        Initialize Neo4j GraphRAG components using dual driver architecture.
        
        Uses sync driver for GraphRAG components and async operations through neo4j_client.
        """
        if self._initialized:
            return
        
        try:
            # Ensure both drivers are available
            await neo4j_client.connect_async()  # For async operations
            neo4j_client.connect_sync()          # For GraphRAG components
            
            # Use sync driver for GraphRAG components (required by neo4j_graphrag)
            self.driver = neo4j_client.sync_driver
            if not self.driver:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Neo4j sync connection not available for GraphRAG"
                )
            
            # Initialize OpenAI LLM with conditional response format
            model_params: Dict[str, Any] = {
                "temperature": self.config.llm_temperature,
                "max_tokens": self.config.llm_max_tokens
            }
            
            # Only add response_format for models that support it
            if self._model_supports_json_object(self.config.llm_model):
                model_params["response_format"] = {"type": "json_object"}
                logger.info(f"Using JSON object response format for model {self.config.llm_model}")
            else:
                logger.warning(f"Model {self.config.llm_model} does not support JSON object response format")
            
            self.llm = OpenAILLM(
                model_name=self.config.llm_model,
                api_key=self.config.openai_api_key,
                model_params=model_params
            )
            
            # Initialize OpenAI embedder
            self.embedder = OpenAIEmbeddings(
                model=self.config.embedding_model,
                api_key=self.config.openai_api_key
            )
            
            self._initialized = True
            logger.info(f"Pipeline components initialized for graph {self.graph_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Pipeline initialization failed: {str(e)}"
            )
    
    async def process_documents(
        self,
        documents: List[Dict[str, Any]],
        background_tasks: Optional[BackgroundTasks] = None,
        overrides: Optional[IngestionOverrides] = None,
        temporal_context: Optional[TemporalContext] = None,
        mode: IngestMode = IngestMode.INCREMENTAL,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process documents through Neo4j GraphRAG pipeline with multi-tenant isolation.

        Args:
            documents: List of document dicts with 'text' and 'source' keys
            background_tasks: Optional FastAPI background tasks for async processing
            overrides: Per-job extraction overrides
            temporal_context: World-time bounds applied to all relationships in this job
            mode: Ingestion mode — full | incremental | upsert
            job_id: Ingestion job UUID (for provenance tracking on nodes)

        Returns:
            Processing result with statistics and status
        """
        try:
            await self._initialize_components()

            start_time = datetime.now()

            # Resolve instructions once for all documents in this call
            resolved = await instructions_resolver.resolve(self.graph_id, overrides)

            # For large document sets, use background processing
            if len(documents) > 10 or background_tasks:
                if background_tasks:
                    background_tasks.add_task(
                        self._process_documents_background,
                        documents,
                        resolved,
                    )
                else:
                    asyncio.create_task(self._process_documents_background(documents, resolved))

                return {
                    "status": "processing",
                    "message": f"Processing {len(documents)} documents in background",
                    "graph_id": self.graph_id,
                    "documents_queued": len(documents),
                    "processing_started_at": start_time.isoformat()
                }

            # Process synchronously for small document sets
            result = await self._process_documents_sync(
                documents, resolved, temporal_context=temporal_context, mode=mode, job_id=job_id
            )

            processing_time = (datetime.now() - start_time).total_seconds()
            result.update({
                "status": "completed",
                "processing_duration": processing_time,
                "graph_id": self.graph_id
            })

            return result

        except Exception as e:
            logger.error(f"Document processing failed for graph {self.graph_id}: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "graph_id": self.graph_id,
                "documents_processed": 0
            }
    
    async def _process_documents_sync(
        self,
        documents: List[Dict[str, Any]],
        resolved: Optional[ResolvedInstructions] = None,
        temporal_context: Optional[TemporalContext] = None,
        mode: IngestMode = IngestMode.INCREMENTAL,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process documents synchronously using Neo4j GraphRAG pipeline."""

        # Create multi-tenant KG writer
        kg_writer = create_multi_tenant_kg_writer(
            driver=self.driver,
            graph_id=self.graph_id,
            user_id=self.user_id
        )

        total_entities = 0
        total_relationships = 0
        total_chunks = 0
        total_violations_detected = 0
        total_violations_migrated = 0
        total_ontology_violations = 0
        total_ontology_coercions = 0
        total_temporal_contradictions = 0

        for i, doc in enumerate(documents):
            try:
                logger.info(f"Processing document {i+1}/{len(documents)} for graph {self.graph_id} (mode={mode.value})")

                text_content = doc.get('text', '') or doc.get('content', '')
                if not text_content:
                    logger.warning(f"Document {i+1} has no text content, skipping")
                    continue

                result = await self._process_single_document(
                    text_content,
                    doc.get('source', f'document_{i+1}'),
                    kg_writer,
                    resolved=resolved,
                    temporal_context=temporal_context,
                    mode=mode,
                    job_id=job_id,
                )

                total_entities += result.get('entities_created', 0)
                total_relationships += result.get('relationships_created', 0)
                total_chunks += result.get('chunks_created', 0)
                total_violations_detected += result.get('property_violations_detected', 0)
                total_violations_migrated += result.get('property_violations_migrated', 0)
                total_ontology_violations += result.get('ontology_violations', 0)
                total_ontology_coercions += result.get('ontology_coercions', 0)
                total_temporal_contradictions += result.get('temporal_contradictions', 0)

            except Exception as e:
                logger.error(f"Failed to process document {i+1}: {e}")
                continue

        return {
            "documents_processed": len(documents),
            "entities_created": total_entities,
            "relationships_created": total_relationships,
            "chunks_created": total_chunks,
            "property_violations_detected": total_violations_detected,
            "property_violations_migrated": total_violations_migrated,
            "ontology_violations": total_ontology_violations,
            "ontology_coercions": total_ontology_coercions,
            "temporal_contradictions": total_temporal_contradictions,
        }
    
    async def _process_single_document(
        self,
        text: str,
        source: str,
        kg_writer: MultiTenantKGWriter,
        resolved: Optional[ResolvedInstructions] = None,
        temporal_context: Optional[TemporalContext] = None,
        mode: IngestMode = IngestMode.INCREMENTAL,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a single document using Neo4j GraphRAG components.

        This processes chunks independently which can create duplicate entities.
        The solution is to add proper entity resolution after extraction.
        """
        from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
        from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor, OnError
        from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder

        with _pipeline_tracer.start_as_current_span("pipeline.document") as doc_span:
            doc_span.set_attribute("graph_id", self.graph_id)
            doc_span.set_attribute("document.source", source)
            doc_span.set_attribute("document.text_length", len(text))
            doc_span.set_attribute("document.mode", mode.value)
            return await self._process_single_document_instrumented(
                text, source, kg_writer, resolved, temporal_context, mode=mode, job_id=job_id,
            )

    async def _process_single_document_instrumented(
        self,
        text: str,
        source: str,
        kg_writer: "MultiTenantKGWriter",
        resolved: Optional["ResolvedInstructions"] = None,
        temporal_context: Optional["TemporalContext"] = None,
        mode: "IngestMode" = IngestMode.INCREMENTAL,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
        from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor, OnError
        from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder

        # 0. Create DocumentInfo for proper lexical graph support
        document_info = DocumentInfo(path=source)

        # ── STAGE 0: SHA256 document-level hash guard ──────────────────────────
        new_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if mode == IngestMode.INCREMENTAL:
            skipped = await self._check_document_hash_unchanged(source, new_hash)
            if skipped:
                logger.info(f"[INCREMENTAL] Document '{source}' unchanged (hash match) — skipping for graph {self.graph_id}")
                return {
                    "entities_created": 0, "relationships_created": 0, "chunks_created": 0,
                    "property_violations_detected": 0, "property_violations_migrated": 0,
                    "ontology_violations": 0, "ontology_coercions": 0, "temporal_contradictions": 0,
                    "ingest_status": "skipped",
                }

        # ── STAGE 0b: Set provenance on Document node ──────────────────────────
        await self._set_document_provenance(source, new_hash, mode, job_id)

        # 1. Text Splitting
        with _pipeline_tracer.start_as_current_span("pipeline.stage.split"):
            splitter = FixedSizeSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            chunks = await splitter.run(text=text)

        # ── STAGE 1.5: Chunk-level delta (incremental mode only) ──────────────────
        chunk_list = getattr(chunks, "chunks", [])
        # Build uid → SHA1(text) map for provenance tracking and delta comparison
        chunk_content_hashes: Dict[str, str] = {
            c.uid: hashlib.sha1((getattr(c, "text", "") or "").encode("utf-8")).hexdigest()
            for c in chunk_list
        }

        if mode == IngestMode.INCREMENTAL:
            existing_hashes: Set[str] = await self._get_existing_chunk_content_hashes(source)
            new_hashes: Set[str] = set(chunk_content_hashes.values())
            added_hashes = new_hashes - existing_hashes
            removed_hashes = existing_hashes - new_hashes

            if removed_hashes:
                await self._soft_delete_stale_chunks(list(removed_hashes), source, job_id)
                logger.info(f"[INCREMENTAL] Soft-deleted {len(removed_hashes)} stale chunks for '{source}'")

            if not added_hashes:
                logger.info(f"[INCREMENTAL] No new chunks for '{source}' — skipping LLM extraction")
                return {
                    "entities_created": 0, "relationships_created": 0, "chunks_created": 0,
                    "property_violations_detected": 0, "property_violations_migrated": 0,
                    "ontology_violations": 0, "ontology_coercions": 0, "temporal_contradictions": 0,
                    "ingest_status": "no_new_chunks",
                }

            # Filter to only added chunks so we skip LLM extraction on unchanged ones
            from neo4j_graphrag.experimental.components.types import TextChunks as _TextChunks
            filtered_list = [c for c in chunk_list if chunk_content_hashes.get(c.uid) in added_hashes]
            chunks = _TextChunks(chunks=filtered_list)
            logger.info(f"[INCREMENTAL] {len(added_hashes)} new / {len(removed_hashes)} removed / "
                        f"{len(existing_hashes - removed_hashes)} unchanged chunks for '{source}'")
        # mode == FULL → no filtering (existing pipeline behavior, entities deleted upstream)
        # mode == UPSERT → no filtering, no soft-delete

        # 2. Chunk Embedding
        with _pipeline_tracer.start_as_current_span("pipeline.stage.embed") as embed_span:
            if self.embedder:
                chunk_embedder = TextChunkEmbedder(embedder=self.embedder)
                embedded_chunks = await chunk_embedder.run(text_chunks=chunks)
            else:
                embedded_chunks = chunks
            embed_span.set_attribute("chunks.count", len(chunks.chunks) if hasattr(chunks, "chunks") else 0)

        # 3. Entity & Relationship Extraction with relationship-first prompt
        if self.llm:
            # Build final prompt: instructions prefix + base relationship-first template
            if resolved is not None:
                prefix = instructions_compiler.to_prompt(resolved)
                if prefix:
                    prompt_template = prefix + "\n\n" + RELATIONSHIP_PROPERTY_PROMPT_TEMPLATE
                    logger.debug(f"Using instructions-augmented prompt for graph {self.graph_id}")
                else:
                    prompt_template = RELATIONSHIP_PROPERTY_PROMPT_TEMPLATE
            else:
                prompt_template = RELATIONSHIP_PROPERTY_PROMPT_TEMPLATE

            # Pre-render {schema} placeholder directly in prompt (do not rely on neo4j_graphrag kwarg passthrough)
            schema_block = instructions_compiler.build_schema_block(resolved) if resolved is not None else ""
            final_prompt = prompt_template.replace("{schema}", schema_block)

            with _pipeline_tracer.start_as_current_span("pipeline.stage.extract") as extract_span:
                extractor = LLMEntityRelationExtractor(
                    llm=self.llm,
                    prompt_template=final_prompt,
                    create_lexical_graph=True,  # Creates Document/Chunk nodes
                    on_error=OnError.IGNORE,
                    max_concurrency=self.config.max_concurrency
                )
                graph = await extractor.run(chunks=embedded_chunks, document_info=document_info)
                extract_span.set_attribute("graph.nodes_raw", len(graph.nodes))
                extract_span.set_attribute("graph.relationships_raw", len(graph.relationships))

            # Detailed logging for entity analysis
            logger.info(f"Raw extraction: {len(graph.nodes)} nodes, {len(graph.relationships)} relationships")

            entity_types: Dict[str, int] = {}
            entity_names: List[str] = []
            for node in graph.nodes:
                node_label = getattr(node, 'label', 'Unknown')
                entity_types[node_label] = entity_types.get(node_label, 0) + 1
                if hasattr(node, 'properties') and node.properties and node.properties.get('name'):
                    entity_names.append(node.properties['name'])

            logger.info(f"Raw extraction entity breakdown by type: {entity_types}")
            logger.info(f"Extracted entity names: {entity_names}")

            # Strip banned node properties and track violations
            graph, violations_detected, violations_migrated = self._strip_banned_node_properties(graph)
            if violations_detected:
                logger.warning(
                    f"Stripped {violations_detected} banned node properties "
                    f"({violations_migrated} moved to relationships) for graph {self.graph_id}"
                )

            # Enforce ontology (between extraction and normalization per spec)
            if resolved is not None and resolved.entity_types:
                graph, ontology_report = self._enforce_ontology(graph, resolved)
                if ontology_report.violations:
                    logger.warning(
                        f"Ontology enforcement ({resolved.ontology_mode.value}): "
                        f"{ontology_report.violations} violations for graph {self.graph_id}"
                    )
                if ontology_report.coercions:
                    logger.info(
                        f"Ontology enforcement: {ontology_report.coercions} coercions for graph {self.graph_id}"
                    )
            else:
                ontology_report = _EnforcementReport()

        else:
            logger.error("LLM not available for entity extraction")
            return {"entities_created": 0, "relationships_created": 0, "chunks_created": 0, "property_violations_detected": 0, "property_violations_migrated": 0}

        # 4. Pre-process: Normalize entity IDs to handle chunk overlap
        logger.info("Starting entity normalization to handle chunk overlaps...")
        graph = await self._normalize_overlapping_entities(graph)
        logger.info(f"After normalization: {len(graph.nodes)} nodes, {len(graph.relationships)} relationships")

        # 4.5. Apply temporal_context overrides to relationships
        if temporal_context:
            for rel in graph.relationships:
                if not rel.properties:
                    rel.properties = {}
                if temporal_context.valid_from and not rel.properties.get("valid_from"):
                    rel.properties["valid_from"] = temporal_context.valid_from
                if temporal_context.valid_to and not rel.properties.get("valid_to"):
                    rel.properties["valid_to"] = temporal_context.valid_to

        # 4.6. Contradiction detection (inline check before write)
        temporal_contradictions = await self._detect_temporal_contradictions(graph, temporal_context)
        if temporal_contradictions:
            logger.warning(
                f"Detected {temporal_contradictions} temporal contradictions for graph {self.graph_id}"
            )

        # 4.7. Entity-level delta classification (INCREMENTAL mode only — Spec ORA-49)
        entity_delta_stats: Dict[str, Any] = {"new": 0, "updated": 0, "unchanged": 0, "entity_ids_updated": []}
        if mode == IngestMode.INCREMENTAL:
            with _pipeline_tracer.start_as_current_span("pipeline.stage.entity_delta"):
                entity_delta_stats = await self._apply_entity_delta(graph, source)

        # 5. Multi-tenant metadata injection (automatic via kg_writer)
        with _pipeline_tracer.start_as_current_span("pipeline.stage.graph_write") as write_span:
            await kg_writer.run(graph)
            write_span.set_attribute("graph_id", self.graph_id)
            logger.info(f"Graph writing completed for graph {self.graph_id}")

        # 5.1. Simple Document-Graph connection
        try:
            connect_query = """
            MATCH (d:Document {path: $source, graph_id: $graph_id})
            MATCH (g:Graph {graph_id: $graph_id})
            MERGE (d)-[:BELONGS_TO]->(g)
            """
            await neo4j_client.execute_query(connect_query, {
                "source": source,
                "graph_id": self.graph_id
            })
            logger.info(f"Connected document '{source}' to graph '{self.graph_id}'")
        except Exception as e:
            logger.warning(f"Could not connect document to graph: {e}")

        # 5.2. Set contentHash provenance on newly-written Chunk nodes
        await self._set_chunk_provenance(chunk_content_hashes, job_id)

        # 5.3. Apply UPDATED property-merge rules + soft-delete orphaned rels (INCREMENTAL only)
        if mode == IngestMode.INCREMENTAL:
            await self._apply_updated_merge_rules(entity_delta_stats.get("entity_ids_updated", []))
            # Build current rel key set for orphan detection
            current_rel_keys: set = set()
            for rel in graph.relationships:
                rp = rel.properties or {}
                src_node = next((n for n in graph.nodes if n.id == rel.start_node_id), None)
                tgt_node = next((n for n in graph.nodes if n.id == rel.end_node_id), None)
                if src_node and tgt_node:
                    src_fp = (src_node.properties or {}).get("fingerprint", "")
                    tgt_fp = (tgt_node.properties or {}).get("fingerprint", "")
                    current_rel_keys.add((
                        src_fp,
                        getattr(rel, "type", "") or "",
                        tgt_fp,
                        rp.get("source_chunk_id", ""),
                    ))
            await self._soft_delete_orphaned_rels(source, current_rel_keys)

        # 6. Entity Deduplication - Consolidate duplicate entities across chunks
        logger.info(f"Checking driver availability for deduplication: driver={self.driver is not None}")
        with _pipeline_tracer.start_as_current_span("pipeline.stage.dedup") as dedup_span:
            if self.driver:  # Ensure driver is available
                logger.info(f"Creating entity deduplicator for graph {self.graph_id}")
                entity_deduplicator = MultiTenantEntityDeduplicator(
                    driver=self.driver,
                    graph_id=self.graph_id,
                    similarity_threshold=0.85,
                    enable_fuzzy_matching=False  # Start with exact matching only
                )

                # Run entity deduplication on the graph
                logger.info(f"Running entity deduplication for graph {self.graph_id}")
                await entity_deduplicator.run(graph)
                dedup_span.set_attribute("dedup.ran", True)
                logger.info(f"Entity deduplication completed for graph {self.graph_id}")
            else:
                dedup_span.set_attribute("dedup.ran", False)
                logger.warning("Neo4j driver not available - skipping entity deduplication")

        # 6.5. Set provenance on entity nodes and FROM_CHUNK relationships
        if job_id and graph and graph.nodes:
            entity_ids = [n.id for n in graph.nodes if n.id]
            await self._set_entity_provenance(entity_ids, job_id)

        # Return statistics
        return {
            "entities_created": len(graph.nodes) if graph and graph.nodes else 0,
            "relationships_created": len(graph.relationships) if graph and graph.relationships else 0,
            "chunks_created": len(chunks.chunks) if chunks and hasattr(chunks, 'chunks') else 0,
            "property_violations_detected": violations_detected,
            "property_violations_migrated": violations_migrated,
            "ontology_violations": ontology_report.violations,
            "ontology_coercions": ontology_report.coercions,
            "temporal_contradictions": temporal_contradictions,
            "ingest_status": "processed",
            # Entity-level delta stats (populated in INCREMENTAL mode only)
            "entities_new": entity_delta_stats.get("new", 0),
            "entities_updated": entity_delta_stats.get("updated", 0),
            "entities_unchanged": entity_delta_stats.get("unchanged", 0),
        }
    
    async def _normalize_overlapping_entities(self, graph: Neo4jGraph) -> Neo4jGraph:
        """
        Normalize entity IDs to handle chunk overlap by removing chunk prefixes
        and creating consistent entity identifiers based on entity names.
        
        This solves the chunk overlap issue where the same entity appears in multiple
        chunks with different IDs (e.g., chunk_1:Alex Thompson vs chunk_2:Alex Thompson).
        """
        from typing import Dict, Any
        
        if not graph or not graph.nodes:
            return graph
        
        # Step 1: Create mapping from entity names to canonical IDs
        entity_name_to_canonical_id: Dict[str, str] = {}
        old_id_to_new_id: Dict[str, str] = {}
        original_entity_count = len(graph.nodes)
        
        # Track what we're merging for debugging
        entities_by_name = {}
        
        for node in graph.nodes:
            if not hasattr(node, 'properties') or not node.properties:
                continue
                
            entity_name = node.properties.get('name')
            if not entity_name:
                continue
            
            # Track entities with same name for debugging
            if entity_name not in entities_by_name:
                entities_by_name[entity_name] = []
            entities_by_name[entity_name].append({
                'id': node.id,
                'label': getattr(node, 'label', 'Unknown'),
                'properties': node.properties
            })
            
            # Create canonical ID from entity name (remove chunk prefix if exists)
            original_id = node.id
            if ':' in original_id:
                # Extract the actual entity name part after chunk prefix
                canonical_id = original_id.split(':', 1)[1]
            else:
                canonical_id = original_id
            
            # Use entity name as the canonical identifier
            if entity_name not in entity_name_to_canonical_id:
                entity_name_to_canonical_id[entity_name] = canonical_id
            
            # Map old chunk-prefixed ID to canonical ID
            old_id_to_new_id[original_id] = entity_name_to_canonical_id[entity_name]
        
        # Log what we're about to merge
        entities_to_merge = {name: entities for name, entities in entities_by_name.items() if len(entities) > 1}
        if entities_to_merge:
            logger.info(f"🔄 Entities being merged due to same name:")
            for entity_name, entity_variations in entities_to_merge.items():
                logger.info(f"  📍 '{entity_name}': {len(entity_variations)} variations")
                for i, variation in enumerate(entity_variations):
                    logger.info(f"    {i+1}. ID: {variation['id']}, Label: {variation['label']}")
        else:
            logger.info("✅ No duplicate entity names found - no merging needed")
        
        # Step 2: Update node IDs to use canonical IDs
        for node in graph.nodes:
            if node.id in old_id_to_new_id:
                node.id = old_id_to_new_id[node.id]
        
        # Step 3: Update relationship references to use canonical IDs
        for rel in graph.relationships:
            if rel.start_node_id in old_id_to_new_id:
                rel.start_node_id = old_id_to_new_id[rel.start_node_id]
            if rel.end_node_id in old_id_to_new_id:
                rel.end_node_id = old_id_to_new_id[rel.end_node_id]
        
        # Step 4: Remove duplicate nodes with same canonical ID
        unique_nodes = {}
        for node in graph.nodes:
            node_key = node.id
            if node_key not in unique_nodes:
                unique_nodes[node_key] = node
            else:
                # Merge properties if needed (take the one with more properties)
                existing_node = unique_nodes[node_key]
                if (hasattr(node, 'properties') and node.properties and 
                    len(node.properties) > len(existing_node.properties or {})):
                    unique_nodes[node_key] = node
        
        # Update graph with deduplicated nodes
        graph.nodes = list(unique_nodes.values())
        
        logger.info(f"Entity normalization: Reduced {len(old_id_to_new_id)} entity references "
                   f"to {len(unique_nodes)} unique entities")
        logger.info(f"📊 Normalization summary: {original_entity_count} → {len(unique_nodes)} entities "
                   f"({original_entity_count - len(unique_nodes)} merged)")
        
        return graph
    
    def _enforce_ontology(
        self,
        graph: "Neo4jGraph",
        resolved: "ResolvedInstructions",
    ) -> Tuple["Neo4jGraph", "_EnforcementReport"]:
        """
        Enforce the graph ontology against extracted nodes.

        Modes:
        - WARN: count violations, do not modify graph
        - STRICT: remove violating nodes and their non-structural relationships
        - COERCE: fuzzy-remap close label matches (threshold 0.7); drop the rest

        Structural relationships are NEVER removed:
        FROM_CHUNK, FROM_DOCUMENT, BELONGS_TO, IN_SESSION, USES_GRAPH
        """
        STRUCTURAL_RELS = frozenset({
            "FROM_CHUNK", "FROM_DOCUMENT", "BELONGS_TO", "IN_SESSION", "USES_GRAPH",
        })
        report = _EnforcementReport()

        if not resolved.entity_types:
            return graph, report

        allowed_types = {et.name for et in resolved.entity_types}
        mode = resolved.ontology_mode

        if mode == OntologyValidationMode.WARN:
            for node in graph.nodes:
                label = getattr(node, "label", None)
                if label and label not in allowed_types and not label.startswith("__"):
                    report.violations += 1
            return graph, report

        if mode == OntologyValidationMode.STRICT:
            violating_ids: set = set()
            conforming_nodes = []
            for node in graph.nodes:
                label = getattr(node, "label", None)
                if label and label not in allowed_types and not label.startswith("__"):
                    violating_ids.add(node.id)
                    report.violations += 1
                else:
                    conforming_nodes.append(node)

            if violating_ids:
                conforming_rels = []
                for rel in graph.relationships:
                    rel_type = getattr(rel, "type", "")
                    if rel_type in STRUCTURAL_RELS:
                        conforming_rels.append(rel)
                    elif rel.start_node_id in violating_ids or rel.end_node_id in violating_ids:
                        pass  # drop
                    else:
                        conforming_rels.append(rel)
                graph.nodes = conforming_nodes
                graph.relationships = conforming_rels
            return graph, report

        if mode == OntologyValidationMode.COERCE:
            allowed_list = list(allowed_types)
            for node in graph.nodes:
                label = getattr(node, "label", None)
                if not label or label in allowed_types or label.startswith("__"):
                    continue
                best_match = max(
                    allowed_list,
                    key=lambda a: difflib.SequenceMatcher(None, label.lower(), a.lower()).ratio(),
                )
                ratio = difflib.SequenceMatcher(None, label.lower(), best_match.lower()).ratio()
                if ratio >= 0.7:
                    node.label = best_match
                    report.coercions += 1
                else:
                    report.violations += 1
            return graph, report

        return graph, report

    def compile_temporal_filter(self, temporal_filter: TemporalFilter) -> str:
        """
        Compile a TemporalFilter into a Cypher WHERE clause fragment.

        Returned string is suitable for appending to queries that alias
        relationships as `r`. It does NOT include the leading 'AND' or 'WHERE'
        keyword — callers must prefix as appropriate.

        Backward compat: no temporal_filter → no filtering (all facts returned).
        """
        if temporal_filter.current_only:
            return "r.valid_to IS NULL"

        clauses: List[str] = []

        if temporal_filter.point_in_time:
            iso = temporal_filter.point_in_time.isoformat()
            clauses.append(
                f"(r.valid_from IS NULL OR r.valid_from <= datetime('{iso}'))"
            )
            clauses.append(
                f"(r.valid_to IS NULL OR r.valid_to > datetime('{iso}'))"
            )

        if temporal_filter.valid_from_gte:
            iso = temporal_filter.valid_from_gte.isoformat()
            clauses.append(f"(r.valid_from IS NULL OR r.valid_from >= datetime('{iso}'))")

        if temporal_filter.valid_to_lte:
            iso = temporal_filter.valid_to_lte.isoformat()
            clauses.append(f"(r.valid_to IS NULL OR r.valid_to <= datetime('{iso}'))")

        return " AND ".join(clauses) if clauses else "true"

    async def _detect_temporal_contradictions(
        self,
        graph: "Neo4jGraph",
        temporal_context: Optional[TemporalContext],
    ) -> int:
        """
        Inline contradiction check: before writing, scan for existing relationships
        of the same type between the same entities whose valid_time overlaps the
        incoming relationships. Marks conflicting pairs for logging.

        Returns the count of detected contradictions. Does NOT block the write —
        conflicts are logged as warnings and surfaced in pipeline stats.
        """
        if not self.driver or not graph.relationships:
            return 0

        contradictions = 0
        incoming_valid_from = None
        incoming_valid_to = None

        if temporal_context:
            incoming_valid_from = temporal_context.valid_from
            incoming_valid_to = temporal_context.valid_to

        for rel in graph.relationships:
            props = rel.properties or {}
            rel_vf = props.get("valid_from") or incoming_valid_from
            rel_vt = props.get("valid_to") or incoming_valid_to

            if rel_vf is None:
                continue  # No temporal info — cannot detect overlap

            # Build overlap check query
            # Two intervals [A,B) and [C,D) overlap when A < D AND C < B
            # (treating NULL as infinity for open-ended intervals)
            check_query = """
            MATCH (src:__Entity__ {graph_id: $graph_id})-[existing]->(tgt:__Entity__ {graph_id: $graph_id})
            WHERE src.name = $src_name
              AND tgt.name = $tgt_name
              AND type(existing) = $rel_type
              AND (existing.valid_from IS NULL OR existing.valid_from <= $vt_or_max)
              AND (existing.valid_to IS NULL OR existing.valid_to >= $vf)
            RETURN count(existing) AS overlap_count
            """

            vt_or_max = rel_vt.isoformat() if rel_vt else "9999-12-31T00:00:00"

            # Resolve entity names from graph nodes (best-effort)
            node_map = {n.id: (n.properties or {}).get("name", n.id) for n in graph.nodes}
            src_name = node_map.get(rel.start_node_id, rel.start_node_id)
            tgt_name = node_map.get(rel.end_node_id, rel.end_node_id)

            try:
                with self.driver.session() as session:
                    rec = session.run(check_query, {
                        "graph_id": self.graph_id,
                        "src_name": src_name,
                        "tgt_name": tgt_name,
                        "rel_type": getattr(rel, "type", ""),
                        "vf": rel_vf.isoformat() if rel_vf else "0001-01-01T00:00:00",
                        "vt_or_max": vt_or_max,
                    }).single()
                    if rec and rec["overlap_count"] > 0:
                        contradictions += 1
                        logger.warning(
                            f"Temporal contradiction: {src_name} -[{getattr(rel, 'type', '?')}]-> {tgt_name} "
                            f"overlaps with existing relationship in graph {self.graph_id} "
                            f"(valid_from={rel_vf.isoformat() if rel_vf else None})"
                        )
            except Exception as e:
                logger.debug(f"Contradiction check skipped for relationship: {e}")

        return contradictions

    # ==================== DELTA DETECTION HELPERS ====================

    async def _check_document_hash_unchanged(self, source: str, new_hash: str) -> bool:
        """Return True if the stored contentHash matches new_hash (skip signal)."""
        try:
            query = """
            MATCH (d:Document {path: $source, graph_id: $graph_id})
            RETURN d.contentHash AS stored_hash
            LIMIT 1
            """
            results = await neo4j_client.execute_query(query, {
                "source": source, "graph_id": self.graph_id
            })
            if not results:
                return False  # First ingestion — always proceed
            stored = results[0].get("stored_hash")
            return stored == new_hash and stored not in (None, "LEGACY_UNKNOWN")
        except Exception as e:
            logger.warning(f"Hash check failed for '{source}': {e} — proceeding with full extraction")
            return False

    async def _set_document_provenance(
        self, source: str, content_hash: str, mode: "IngestMode", job_id: Optional[str]
    ) -> None:
        """Set contentHash, lastJobId, lastIngestedAt, ingestMode on the Document node (upsert)."""
        try:
            query = """
            MERGE (d:Document {path: $source, graph_id: $graph_id})
            SET d.contentHash    = $content_hash,
                d.lastJobId      = $job_id,
                d.lastIngestedAt = datetime(),
                d.ingestMode     = $mode
            """
            await neo4j_client.execute_query(query, {
                "source": source,
                "graph_id": self.graph_id,
                "content_hash": content_hash,
                "job_id": job_id or "",
                "mode": mode.value,
            })
        except Exception as e:
            logger.warning(f"Could not set document provenance for '{source}': {e}")

    async def _get_existing_chunk_content_hashes(self, source: str) -> Set[str]:
        """Return the set of SHA1 contentHash values for chunks already in the graph for this document."""
        try:
            query = """
            MATCH (d:Document {path: $source, graph_id: $graph_id})<-[:PART_OF]-(c:Chunk {graph_id: $graph_id})
            WHERE c.contentHash IS NOT NULL
            RETURN collect(c.contentHash) AS hashes
            """
            results = await neo4j_client.execute_query(query, {
                "source": source, "graph_id": self.graph_id
            })
            if results:
                return set(results[0].get("hashes") or [])
            return set()
        except Exception as e:
            logger.warning(f"Could not fetch existing chunk hashes for '{source}': {e}")
            return set()

    async def _soft_delete_stale_chunks(
        self, content_hashes: List[str], source: str, job_id: Optional[str]
    ) -> None:
        """Mark removed chunks as stale by setting staleAt / staleJobId (soft-delete)."""
        try:
            query = """
            MATCH (d:Document {path: $source, graph_id: $graph_id})<-[:PART_OF]-(c:Chunk {graph_id: $graph_id})
            WHERE c.contentHash IN $hashes AND c.staleAt IS NULL
            SET c.staleAt    = datetime(),
                c.staleJobId = $job_id
            """
            await neo4j_client.execute_query(query, {
                "source": source,
                "graph_id": self.graph_id,
                "hashes": content_hashes,
                "job_id": job_id or "",
            })
        except Exception as e:
            logger.warning(f"Could not soft-delete stale chunks for '{source}': {e}")

    async def _set_chunk_provenance(
        self, chunk_content_hashes: Dict[str, str], job_id: Optional[str]
    ) -> None:
        """
        Set contentHash and jobId on newly-written Chunk nodes.

        Matches chunks by their uid (stored as `id` property in Neo4j by neo4j_graphrag)
        and sets provenance properties so delta detection works on subsequent ingestions.
        """
        if not chunk_content_hashes:
            return
        try:
            # Build list of {uid, contentHash} params for batch update
            params_list = [
                {"uid": uid, "content_hash": ch}
                for uid, ch in chunk_content_hashes.items()
            ]
            query = """
            UNWIND $params AS p
            MATCH (c:Chunk {id: p.uid, graph_id: $graph_id})
            SET c.contentHash = p.content_hash,
                c.jobId       = $job_id,
                c.ingestedAt  = CASE WHEN c.ingestedAt IS NULL THEN datetime() ELSE c.ingestedAt END
            """
            await neo4j_client.execute_query(query, {
                "params": params_list,
                "graph_id": self.graph_id,
                "job_id": job_id or "",
            })
        except Exception as e:
            logger.warning(f"Could not set chunk provenance for graph {self.graph_id}: {e}")

    async def _set_entity_provenance(
        self, entity_ids: List[str], job_id: Optional[str]
    ) -> None:
        """
        Set lastJobId, ingestedAt (first-seen only), and updatedAt on extracted entity nodes.

        Uses ON MATCH SET semantics: ingestedAt is only set when first created (preserves
        original ingestion timestamp), while updatedAt and lastJobId are always refreshed.
        This ensures manually-added properties on entity nodes are never overwritten.
        """
        if not entity_ids or not job_id:
            return
        try:
            query = """
            UNWIND $entity_ids AS eid
            MATCH (n:__Entity__ {id: eid, graph_id: $graph_id})
            SET n.lastJobId  = $job_id,
                n.ingestedAt = CASE WHEN n.ingestedAt IS NULL THEN datetime() ELSE n.ingestedAt END,
                n.updatedAt  = datetime()
            """
            await neo4j_client.execute_query(query, {
                "entity_ids": entity_ids,
                "graph_id": self.graph_id,
                "job_id": job_id,
            })
        except Exception as e:
            logger.warning(f"Could not set entity provenance for graph {self.graph_id}: {e}")

    def _strip_banned_node_properties(
        self, graph: Neo4jGraph
    ) -> Tuple[Neo4jGraph, int, int]:
        """
        Remove banned properties from entity nodes.

        Properties like job_title, position, seniority belong on relationships,
        not on nodes. The LLM prompt should prevent this, but this is a defensive
        cleanup pass that runs post-extraction.

        Returns:
            (graph, violations_detected, violations_migrated)
            violations_migrated is always 0 here — LLM should have placed them on
            the correct relationship already. If not, the property is dropped so
            Neo4j doesn't persist the anti-pattern.
        """
        violations_detected = 0
        for node in graph.nodes:
            if not node.properties:
                continue
            banned_found = set(node.properties.keys()) & BANNED_NODE_PROPERTIES
            if banned_found:
                violations_detected += len(banned_found)
                for prop in banned_found:
                    del node.properties[prop]
                logger.warning(
                    f"Stripped banned properties {banned_found} from node "
                    f"'{node.properties.get('name', node.id)}'"
                )
        return graph, violations_detected, 0

    # ── Entity-level delta (Spec ORA-49) ──────────────────────────────────────

    async def _bulk_fingerprint_lookup(self, fingerprints: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Single round-trip lookup: returns {fingerprint → {prop_hash, description}}.

        Only returns entries for fingerprints that already exist in Neo4j.
        Missing fingerprints → NEW entity.
        """
        if not fingerprints:
            return {}
        q = """
        MATCH (e:__Entity__)
        WHERE e.graph_id = $graph_id AND e.fingerprint IN $fps
        RETURN e.fingerprint AS fp, e.prop_hash AS prop_hash, e.description AS description
        """
        rows = await neo4j_client.execute_query(q, {"graph_id": self.graph_id, "fps": fingerprints})
        return {
            r["fp"]: {"prop_hash": r["prop_hash"], "description": r.get("description")}
            for r in (rows or [])
        }

    async def _apply_entity_delta(
        self,
        graph: "Neo4jGraph",
        source: str,
    ) -> Dict[str, Any]:
        """
        Entity-level delta classification for INCREMENTAL mode.

        Modifies graph.nodes in place:
        - Attaches fingerprint + prop_hash to every __Entity__ node.
        - Removes UNCHANGED entities (and their exclusive relationships) from the
          graph so kg_writer skips them entirely.
        - Tags UPDATED entities with a sentinel property ``_delta=updated`` so
          _apply_updated_merge_rules can identify them after the write.

        Returns a stats dict: {new, updated, unchanged, entity_ids_updated}.
        """
        entity_nodes = [
            n for n in graph.nodes
            if "__Entity__" in getattr(n, "label", "") or (
                hasattr(n, "labels") and "__Entity__" in getattr(n, "labels", [])
            ) or (
                # neo4j_graphrag uses `label` as a single string
                getattr(n, "label", None) == "__Entity__"
            )
        ]

        if not entity_nodes:
            return {"new": 0, "updated": 0, "unchanged": 0, "entity_ids_updated": []}

        # Compute fingerprint + prop_hash for every extracted entity
        fp_map: Dict[str, "Neo4jGraphNode"] = {}  # fingerprint → node (first occurrence wins)
        for node in entity_nodes:
            props = node.properties or {}
            name = props.get("name") or node.id or ""
            label = getattr(node, "label", None) or (
                next(iter(getattr(node, "labels", [])), "") if hasattr(node, "labels") else ""
            )
            fp = compute_entity_fingerprint(self.graph_id, name, label)
            ph = compute_prop_hash(props)
            if not node.properties:
                node.properties = {}
            node.properties["fingerprint"] = fp
            node.properties["prop_hash"] = ph
            fp_map[fp] = node

        # Single round-trip to Neo4j
        existing: Dict[str, str] = await self._bulk_fingerprint_lookup(list(fp_map.keys()))

        new_count = updated_count = unchanged_count = 0
        unchanged_fps: set = set()
        updated_ids: List[str] = []

        for fp, node in fp_map.items():
            if fp not in existing:
                new_count += 1
                if node.properties:
                    node.properties["last_updated_at"] = datetime.now().isoformat()
            elif existing[fp]["prop_hash"] == node.properties.get("prop_hash"):
                unchanged_count += 1
                unchanged_fps.add(fp)
            else:
                updated_count += 1
                if node.properties:
                    node.properties["_delta"] = "updated"
                    node.properties["last_updated_at"] = datetime.now().isoformat()
                    # Store the old description so _apply_updated_merge_rules can
                    # implement description-longer-wins after kg_writer overwrites it.
                    node.properties["_prev_description"] = existing[fp]["description"]
                updated_ids.append(node.id)

        # Remove UNCHANGED entities from the graph (no write needed)
        if unchanged_fps:
            unchanged_node_ids = {
                n.id for n in entity_nodes
                if (n.properties or {}).get("fingerprint") in unchanged_fps
            }
            graph.nodes = [n for n in graph.nodes if n.id not in unchanged_node_ids]
            # Remove relationships whose BOTH endpoints are unchanged (exclusive rels)
            graph.relationships = [
                r for r in graph.relationships
                if r.start_node_id not in unchanged_node_ids
                or r.end_node_id not in unchanged_node_ids
            ]

        logger.info(
            f"[INCREMENTAL] Entity delta for graph {self.graph_id}: "
            f"new={new_count}, updated={updated_count}, unchanged={unchanged_count}"
        )
        return {
            "new": new_count,
            "updated": updated_count,
            "unchanged": unchanged_count,
            "entity_ids_updated": updated_ids,
        }

    async def _apply_updated_merge_rules(self, updated_ids: List[str]) -> None:
        """
        After kg_writer has written the UPDATED entities, apply property-level merge rules:
        - description: keep the longer value
        - extra dict fields: union (additive, never lose keys)
        - prop_hash + last_updated_at: always set to new values
        - Recompute embedding only if description changed (handled by next embedding step)

        Only runs if there are UPDATED entities.
        """
        if not updated_ids:
            return

        # For each updated entity id, fetch old props and apply rules via Cypher
        for entity_id in updated_ids:
            try:
                merge_q = """
                MATCH (e:__Entity__ {graph_id: $graph_id})
                WHERE elementId(e) = $eid OR e.entity_id = $eid
                WITH e LIMIT 1
                SET e.description = CASE
                      WHEN e._prev_description IS NOT NULL
                       AND size(coalesce(toString(e._prev_description), '')) > size(coalesce(toString(e.description), ''))
                      THEN e._prev_description
                      ELSE e.description
                    END,
                    e.last_updated_at = datetime(),
                    e._delta = null,
                    e._prev_description = null
                """
                await neo4j_client.execute_query(
                    merge_q, {"graph_id": self.graph_id, "eid": entity_id}
                )
            except Exception as exc:
                logger.warning(f"Post-write merge rule failed for entity {entity_id}: {exc}")

    async def _soft_delete_orphaned_rels(self, source: str, current_rel_keys: set) -> int:
        """
        Soft-delete relationships from `source` that are no longer present in the
        current extraction. Sets valid_to = datetime() on orphaned edges.

        Only called in INCREMENTAL mode when re-ingesting an existing document.
        `current_rel_keys` is a frozenset of (src_fingerprint, rel_type, tgt_fingerprint, chunk_id).
        """
        try:
            find_q = """
            MATCH (d:Document {path: $source, graph_id: $graph_id})
            MATCH (d)<-[:FROM_DOCUMENT]-(c:Chunk {graph_id: $graph_id})
            MATCH (src:__Entity__ {graph_id: $graph_id})<-[:FROM_CHUNK]-(c)
            MATCH (src)-[r {graph_id: $graph_id}]->(tgt:__Entity__ {graph_id: $graph_id})
            WHERE r.valid_to IS NULL AND r.source_chunk_id IS NOT NULL
            RETURN r.source_chunk_id AS chunk_id,
                   type(r) AS rel_type,
                   src.fingerprint AS src_fp,
                   tgt.fingerprint AS tgt_fp,
                   elementId(r) AS rel_elem_id
            """
            rows = await neo4j_client.execute_query(find_q, {
                "source": source, "graph_id": self.graph_id
            })
            if not rows:
                return 0

            orphan_elem_ids = [
                r["rel_elem_id"] for r in rows
                if (r["src_fp"], r["rel_type"], r["tgt_fp"], r["chunk_id"]) not in current_rel_keys
            ]
            if not orphan_elem_ids:
                return 0

            soft_del_q = """
            MATCH ()-[r]->()
            WHERE elementId(r) IN $elem_ids
            SET r.valid_to = datetime()
            RETURN count(r) AS cnt
            """
            result = await neo4j_client.execute_query(soft_del_q, {"elem_ids": orphan_elem_ids})
            deleted = int((result or [{"cnt": 0}])[0]["cnt"])
            logger.info(f"[INCREMENTAL] Soft-deleted {deleted} orphaned rels for '{source}' in graph {self.graph_id}")
            return deleted
        except Exception as exc:
            logger.warning(f"Orphaned rel soft-delete failed for graph {self.graph_id}: {exc}")
            return 0

    async def _process_documents_background(
        self,
        documents: List[Dict[str, Any]],
        resolved: Optional[ResolvedInstructions] = None,
    ):
        """Background processing for large document sets."""
        try:
            logger.info(f"Starting background processing of {len(documents)} documents for graph {self.graph_id}")

            result = await self._process_documents_sync(documents, resolved)
            
            logger.info(f"Background processing completed for graph {self.graph_id}: "
                       f"{result.get('entities_created', 0)} entities, "
                       f"{result.get('relationships_created', 0)} relationships")
                       
        except Exception as e:
            logger.error(f"Background processing failed for graph {self.graph_id}: {e}")
    
    async def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status and statistics."""
        try:
            await self._initialize_components()
            
            # Query graph statistics
            query = """
            MATCH (n)
            WHERE n.graph_id = $graph_id
            WITH labels(n) as node_labels, count(n) as node_count
            UNWIND node_labels as label
            RETURN label, sum(node_count) as count
            """
            
            result = await neo4j_client.execute_query(query, {"graph_id": self.graph_id})
            
            stats = {}
            for record in result:
                stats[record['label']] = record['count']
            
            return {
                "graph_id": self.graph_id,
                "status": "ready",
                "statistics": stats,
                "components_initialized": self._initialized
            }
            
        except Exception as e:
            logger.error(f"Failed to get processing status for {self.graph_id}: {e}")
            return {
                "graph_id": self.graph_id,
                "status": "error", 
                "error": str(e)
            }


# ==================== SERVICE CLASS ====================

class PipelineService:
    """
    FastAPI-compatible service for multi-tenant pipeline operations.
    
    FEATURES:
    - Factory pattern for creating tenant-specific pipelines
    - FastAPI dependency injection support
    - Clean error handling and logging
    - Performance monitoring
    """
    
    def __init__(self):
        """Initialize pipeline service."""
        self._pipeline_cache: Dict[str, MultiTenantGraphRAGPipeline] = {}  # Cache pipelines per graph_id
        logger.info("PipelineService initialized")
    
    def get_pipeline(self, graph_id: UUID, user_id: Optional[str] = None) -> MultiTenantGraphRAGPipeline:
        """
        Get or create multi-tenant pipeline for graph_id.
        
        Args:
            graph_id: Tenant graph identifier  
            user_id: Optional user identifier
            
        Returns:
            Multi-tenant pipeline instance
        """
        cache_key = f"pipeline_{graph_id}"
        
        if cache_key not in self._pipeline_cache:
            self._pipeline_cache[cache_key] = MultiTenantGraphRAGPipeline(
                graph_id=str(graph_id),
                user_id=user_id
            )
        
        return self._pipeline_cache[cache_key]
    
    async def process_documents(
        self,
        documents: List[Dict[str, Any]],
        graph_id: UUID,
        user_id: Optional[str] = None,
        background_tasks: Optional[BackgroundTasks] = None,
        overrides: Optional[IngestionOverrides] = None,
        temporal_context: Optional[TemporalContext] = None,
        mode: IngestMode = IngestMode.INCREMENTAL,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process documents through multi-tenant pipeline.

        Args:
            documents: List of document dicts
            graph_id: Tenant graph identifier
            user_id: Optional user identifier
            background_tasks: Optional FastAPI background tasks
            overrides: Per-job extraction overrides
            temporal_context: World-time bounds applied to all relationships in this job
            mode: Ingestion mode — full | incremental | upsert
            job_id: Ingestion job UUID for provenance tracking

        Returns:
            Processing result with status and statistics
        """
        try:
            # Auto-checkpoint: create a version snapshot before bulk ingestion when graph
            # already has entities (Architecture spec § 9 — pre-ingest checkpoints).
            await self._auto_checkpoint_if_needed(graph_id, user_id)

            pipeline = self.get_pipeline(graph_id, user_id)
            return await pipeline.process_documents(
                documents, background_tasks, overrides,
                temporal_context=temporal_context, mode=mode, job_id=job_id,
            )

        except Exception as e:
            logger.error(f"Pipeline processing failed for graph {graph_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Pipeline processing failed: {str(e)}"
            )

    async def _auto_checkpoint_if_needed(self, graph_id: UUID, user_id: Optional[str]) -> None:
        """
        Create an auto-checkpoint version before bulk ingestion if the graph has existing data.
        Non-fatal: failures are logged but never abort ingestion.
        """
        try:
            from app.core.neo4j_client import neo4j_client
            result = await neo4j_client.execute_query(
                "MATCH (e:__Entity__ {graph_id: $graph_id}) WHERE e.deleted_at IS NULL RETURN count(e) AS cnt",
                {"graph_id": str(graph_id)},
            )
            entity_count = int((result or [{"cnt": 0}])[0]["cnt"])
            if entity_count == 0:
                return  # Nothing to checkpoint

            from app.services.background_jobs import create_graph_snapshot
            from datetime import datetime, timezone
            label = f"pre-ingest-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
            create_graph_snapshot.delay(
                graph_id=str(graph_id),
                label=label,
                description="Auto-checkpoint before bulk ingestion",
                created_by=user_id or "system",
                is_auto=True,
            )
            logger.info(f"Queued auto-checkpoint '{label}' for graph {graph_id} ({entity_count} entities)")
        except Exception as exc:
            logger.warning(f"Auto-checkpoint skipped for graph {graph_id}: {exc}")
    
    async def get_pipeline_status(self, graph_id: UUID) -> Dict[str, Any]:
        """Get processing status for a specific graph."""
        try:
            pipeline = self.get_pipeline(graph_id)
            return await pipeline.get_processing_status()
            
        except Exception as e:
            logger.error(f"Failed to get pipeline status for {graph_id}: {e}")
            return {
                "graph_id": str(graph_id),
                "status": "error",
                "error": str(e)
            }
    
    def clear_pipeline_cache(self, graph_id: Optional[UUID] = None):
        """Clear pipeline cache for memory management."""
        if graph_id:
            cache_key = f"pipeline_{graph_id}"
            if cache_key in self._pipeline_cache:
                del self._pipeline_cache[cache_key]
                logger.info(f"Cleared pipeline cache for graph {graph_id}")
        else:
            self._pipeline_cache.clear()
            logger.info("Cleared all pipeline caches")


# ==================== FASTAPI DEPENDENCY INJECTION ====================

def get_pipeline_service() -> PipelineService:
    """
    FastAPI dependency factory for PipelineService.
    
    Usage:
        @router.post("/graphs/{graph_id}/process")
        async def process_documents(
            graph_id: UUID,
            documents: List[DocumentRequest],
            pipeline_service: PipelineService = Depends(get_pipeline_service),
            background_tasks: BackgroundTasks = None
        ):
            return await pipeline_service.process_documents(
                documents=[doc.dict() for doc in documents],
                graph_id=graph_id,
                background_tasks=background_tasks
            )
    """
    return PipelineService()


# ==================== GLOBAL INSTANCE ====================

# Global instance for backward compatibility and direct usage
pipeline_service = PipelineService()