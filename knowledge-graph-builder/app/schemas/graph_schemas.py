from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

# ==================== ONTOLOGY VALIDATION MODE ====================


class OntologyValidationMode(StrEnum):
    WARN = "warn"  # Count violations but do not modify the extracted graph
    STRICT = "strict"  # Remove nodes whose label is not in the allowed set
    COERCE = "coerce"  # Fuzzy-remap close matches; drop the rest (threshold 0.7)


class IngestMode(StrEnum):
    FULL = "full"  # Delete all chunks+entities → re-extract (backward-compat)
    INCREMENTAL = (
        "incremental"  # SHA256 hash guard → only process changed chunks (default)
    )
    UPSERT = "upsert"  # Process all chunks, MERGE everywhere, never delete


# Properties that describe relational context — must live on edges, never on entity nodes.
BANNED_NODE_PROPERTIES = {
    "job_title",
    "position",
    "role",
    "title",
    "employer",
    "proficiency",
    "seniority",
    "ownership_pct",
    "allocation",
    "start_date_of_employment",
    "end_date_of_employment",
}


class RelationshipProperties(BaseModel):
    source_chunk_id: str = Field(..., description="ID of the source Chunk node")
    confidence: float = Field(..., ge=0.0, le=1.0)
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Bitemporal fields — valid_time tracks when the fact was true in the world
    valid_from: datetime | None = Field(
        None, description="When this fact became true in the world"
    )
    valid_to: datetime | None = Field(
        None, description="When this fact stopped being true (null = still valid)"
    )
    # transaction_time is always set server-side — never trust client for this value
    transaction_time: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this fact was recorded in the graph (server-set)",
    )
    extra: dict[str, Any] | None = Field(default_factory=dict)
    # Bitemporal provenance fields (TASK-005)
    event_time: datetime | None = Field(
        None,
        description="When this fact was true in the real world (nullable; set by LLM or caller).",
    )
    event_time_end: datetime | None = Field(
        None,
        description="When this fact stopped being true in the real world (null = still true).",
    )
    ingestion_time: datetime | None = Field(
        None,
        description="When the system first learned this fact (server-set at write time).",
    )
    ingestion_source: str | None = Field(
        None,
        description="Which document, connector, or API provided this fact.",
    )

    @field_validator("valid_from", "valid_to", mode="before")
    @classmethod
    def coerce_temporal_string(cls, v: Any) -> datetime | None:
        """Accept ISO-8601 strings from LLM output and coerce to datetime."""
        if v is None or isinstance(v, datetime):
            return v
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return None
            try:
                # Try ISO format first, fall back to dateutil if available
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                return None  # Unparseable temporal string — treat as missing
        return None

    @model_validator(mode="after")
    def validate_temporal_range(self) -> "RelationshipProperties":
        if self.valid_from and self.valid_to and self.valid_from > self.valid_to:
            raise ValueError(
                f"valid_from ({self.valid_from.isoformat()}) must be ≤ valid_to ({self.valid_to.isoformat()})"
            )
        return self


class EntityNodeProperties(BaseModel):
    name: str
    description: str | None = None
    extra: dict[str, Any] | None = Field(default_factory=dict)
    # Bitemporal provenance fields (TASK-005)
    event_time: datetime | None = Field(
        None,
        description="When this entity fact was true in the real world (nullable).",
    )
    event_time_end: datetime | None = Field(
        None,
        description="When this entity fact stopped being true (null = still true).",
    )
    ingestion_time: datetime | None = Field(
        None,
        description="When the system first learned about this entity (server-set at write time).",
    )
    ingestion_source: str | None = Field(
        None,
        description="Which document, connector, or API provided this entity.",
    )

    @field_validator("extra")
    @classmethod
    def reject_banned_properties(
        cls, v: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if not v:
            return v
        violations = set(v.keys()) & BANNED_NODE_PROPERTIES
        if violations:
            raise ValueError(
                f"Properties {violations} are relational context — place them on "
                f"the relationship, not the entity node."
            )
        return v


class ExtractedEntity(BaseModel):
    id: str
    label: str
    properties: EntityNodeProperties


class ExtractedRelationship(BaseModel):
    start_node_id: str
    end_node_id: str
    type: str = Field(..., pattern=r"^[A-Z][A-Z0-9_]*$")
    properties: RelationshipProperties


class LLMExtractionOutput(BaseModel):
    nodes: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]

    @field_validator("relationships")
    @classmethod
    def validate_node_references(
        cls, rels: list[ExtractedRelationship], info: Any
    ) -> list[ExtractedRelationship]:
        if "nodes" not in info.data:
            return rels
        node_ids = {n.id for n in info.data["nodes"]}
        for rel in rels:
            if rel.start_node_id not in node_ids:
                raise ValueError(
                    f"start_node_id '{rel.start_node_id}' not in extracted nodes"
                )
            if rel.end_node_id not in node_ids:
                raise ValueError(
                    f"end_node_id '{rel.end_node_id}' not in extracted nodes"
                )
        return rels


class MigrationOrphanLog(BaseModel):
    graph_id: str
    entity_id: str
    entity_name: str
    entity_type: str
    orphaned_properties: dict[str, Any]
    source_chunk_ids: list[str]
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(default="pending")  # pending | re_extracted | manual_review


# ==================== TEMPORAL MODELS ====================


class TemporalContext(BaseModel):
    """Per-ingestion temporal context — specifies world-time bounds for ingested facts.

    When provided, `valid_from` / `valid_to` are applied to all relationships written
    during that ingestion job, overriding values extracted by the LLM.
    """

    valid_from: datetime | None = Field(
        None,
        description="Start of world-time validity for facts in this document.",
    )
    valid_to: datetime | None = Field(
        None,
        description="End of world-time validity. Null means still valid.",
    )
    source_date: str | None = Field(
        None,
        description="Human-readable document date, e.g. 'Q3 2023'. Stored as metadata; not used for filtering.",
    )

    @model_validator(mode="after")
    def validate_range(self) -> "TemporalContext":
        if self.valid_from and self.valid_to and self.valid_from > self.valid_to:
            raise ValueError("valid_from must be ≤ valid_to")
        return self


class TemporalFilter(BaseModel):
    """Query-time temporal filter — scopes entity/relationship retrieval to a point or range in time."""

    point_in_time: datetime | None = Field(
        None,
        description="Return only facts valid at this instant (valid_from ≤ t < valid_to or valid_to IS NULL).",
    )
    valid_from_gte: datetime | None = Field(
        None,
        description="Lower bound filter on valid_from (inclusive).",
    )
    valid_to_lte: datetime | None = Field(
        None,
        description="Upper bound filter on valid_to (inclusive). Null entries are always included.",
    )
    current_only: bool = Field(
        False,
        description="When True, return only currently-valid facts (valid_to IS NULL). "
        "Takes precedence over point_in_time when both are set.",
    )

    @model_validator(mode="after")
    def validate_filter(self) -> "TemporalFilter":
        if self.point_in_time and self.current_only:
            raise ValueError("point_in_time and current_only are mutually exclusive")
        return self


class UpdateTemporalBoundsRequest(BaseModel):
    """Request body for PATCH /graphs/{id}/entities/{entity_id}/temporal."""

    valid_from: datetime | None = Field(None, description="New world-time start.")
    valid_to: datetime | None = Field(
        None, description="New world-time end. Null clears the end date."
    )

    @model_validator(mode="after")
    def validate_range(self) -> "UpdateTemporalBoundsRequest":
        if self.valid_from and self.valid_to and self.valid_from > self.valid_to:
            raise ValueError("valid_from must be ≤ valid_to")
        return self


class TimelineEvent(BaseModel):
    """A single entry in a graph timeline."""

    event_type: str = Field(
        ..., description="'entity_created', 'entity_updated', or 'relationship'"
    )
    entity_id: str
    entity_name: str | None = None
    entity_label: str | None = None
    relationship_type: str | None = None
    related_entity_name: str | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    transaction_time: datetime | None = None


class TimelineResponse(BaseModel):
    """Response for GET /graphs/{id}/timeline."""

    graph_id: str
    entity_id: str | None = None
    events: list[TimelineEvent]
    total: int


# ==================== USER CONTEXT & INSTRUCTIONS MODELS ====================


class ExtractionDensity(StrEnum):
    SPARSE = "sparse"
    BALANCED = "balanced"
    DENSE = "dense"


class EntityTypeDefinition(BaseModel):
    """Canonical entity type definition for ontology-guided extraction."""

    name: str = Field(..., description="Entity type label, e.g. Person, Company, Drug")
    description: str | None = Field(
        None, description="What qualifies as this entity type"
    )
    examples: list[str] | None = Field(
        None, description="Canonical examples to guide LLM"
    )
    properties: dict[str, str] | None = Field(
        None,
        description="Property name -> description map. Tells the LLM what attributes to capture on this type.",
    )


# Backward-compat alias — existing code that imports EntityTypeRule still works
EntityTypeRule = EntityTypeDefinition


class RelationshipTypeDefinition(BaseModel):
    """Canonical relationship type definition for ontology-guided extraction."""

    name: str = Field(..., description="Relationship type, e.g. WORKS_FOR, PRESCRIBES")
    source_type: str | None = Field(None, description="Expected source entity type")
    target_type: str | None = Field(None, description="Expected target entity type")
    properties: list[str] | None = Field(
        None,
        description="Properties that must be stored on this relationship, not on entity nodes.",
    )
    # Deprecated — kept for backward compatibility with stored GraphInstructions JSON
    store_as_edge_property: list[str] | None = Field(
        None,
        description="Deprecated. Use `properties` instead.",
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_store_as_edge_property(cls, values: Any) -> Any:
        """Silently promote old `store_as_edge_property` → `properties` on read."""
        if isinstance(values, dict):
            old = values.get("store_as_edge_property")
            if old and not values.get("properties"):
                values["properties"] = old
        return values


# Backward-compat alias
RelationshipRule = RelationshipTypeDefinition


class GraphInstructions(BaseModel):
    """Graph-scoped extraction instructions — persisted on the Graph node."""

    domain: str | None = Field(
        None,
        description="Domain hint for the LLM extractor, e.g. 'HR org chart', 'pharmaceutical research'.",
    )
    extraction_density: ExtractionDensity = Field(
        ExtractionDensity.BALANCED,
        description="Controls how aggressively entities are extracted.",
    )
    entity_types: list[EntityTypeDefinition] | None = Field(
        None,
        description="Preferred entity types. When provided: ontology-guided mode. When absent: free-form.",
    )
    relationship_types: list[RelationshipTypeDefinition] | None = Field(
        None,
        description="Preferred relationship types with edge-property rules.",
    )
    ontology_mode: OntologyValidationMode = Field(
        OntologyValidationMode.WARN,
        description="How strictly to enforce ontology during extraction.",
    )
    edge_property_fields: list[str] | None = Field(
        None,
        description="Global list of property names that must ALWAYS be stored on relationships, not nodes.",
    )
    focus_areas: list[str] | None = Field(
        None,
        description="Free-text hints about what to focus on.",
    )
    ignore_patterns: list[str] | None = Field(
        None,
        description="Entity names or patterns to ignore during extraction.",
    )
    language: str | None = Field(
        "en",
        description="Primary language of source documents. ISO 639-1 code.",
    )
    custom_prompt_suffix: str | None = Field(
        None,
        max_length=2000,
        description="Free-text instruction appended verbatim to the LLM extraction prompt.",
    )


class IngestionOverrides(BaseModel):
    """Per-ingestion overrides that supplement (not replace) graph-level instructions."""

    additional_focus: str | None = Field(
        None,
        description="One-time focus hint for this specific document.",
    )
    override_density: ExtractionDensity | None = Field(
        None,
        description="Override extraction density for this job only.",
    )
    extra_entity_types: list[EntityTypeDefinition] | None = Field(
        None,
        description="Additional entity types to extract beyond graph defaults.",
    )
    schema_evolution_hint: str | None = Field(
        None,
        description="Instruction about how schema should evolve from this document.",
    )


class GraphInstructionsResponse(BaseModel):
    graph_id: UUID
    instructions: GraphInstructions
    version: int
    updated_at: datetime


# ==================== ONTOLOGY REQUEST/RESPONSE MODELS ====================


class OntologyResponse(BaseModel):
    """Ontology configuration retrieved from a graph."""

    graph_id: UUID
    entity_types: list[EntityTypeDefinition]
    relationship_types: list[RelationshipTypeDefinition]
    ontology_mode: OntologyValidationMode
    version: int
    updated_at: datetime


class OntologySetRequest(BaseModel):
    """Set (replace) the ontology on a graph."""

    entity_types: list[EntityTypeDefinition] = Field(
        ..., description="Allowed entity type definitions."
    )
    relationship_types: list[RelationshipTypeDefinition] | None = Field(
        None, description="Allowed relationship type definitions."
    )
    ontology_mode: OntologyValidationMode = Field(
        OntologyValidationMode.WARN,
        description="Enforcement mode applied during extraction.",
    )


class OntologyPatchRequest(BaseModel):
    """Partial update for the ontology — add/remove individual types."""

    add_entity_types: list[EntityTypeDefinition] | None = Field(
        None, description="Entity types to add or replace (matched by name)."
    )
    remove_entity_types: list[str] | None = Field(
        None, description="Entity type names to remove."
    )
    add_relationship_types: list[RelationshipTypeDefinition] | None = Field(
        None, description="Relationship types to add or replace (matched by name)."
    )
    remove_relationship_types: list[str] | None = Field(
        None, description="Relationship type names to remove."
    )
    ontology_mode: OntologyValidationMode | None = Field(
        None, description="Update the enforcement mode."
    )


class OntologyValidationReport(BaseModel):
    """Dry-run Cypher scan result — no graph modifications."""

    graph_id: UUID
    scanned_entities: int
    violation_count: int
    coercion_candidates: int
    violations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Sample of violating entities (name, label, element_id).",
    )


class RetroactiveApplyRequest(BaseModel):
    """Request to apply the current ontology to already-ingested nodes."""

    dry_run: bool = Field(
        True,
        description="When True, scan only — no writes. When False, apply enforcement.",
    )


class RetroactiveApplyResponse(BaseModel):
    """Result of a retroactive ontology apply operation."""

    graph_id: UUID
    dry_run: bool
    mode: OntologyValidationMode
    violations_found: int
    coercions_applied: int
    deletions_applied: int
    celery_task_id: str | None = Field(
        None,
        description="Set when the operation was dispatched as a Celery background task (>10k nodes).",
    )


# ==================== GRAPH CRUD MODELS ====================


class GraphCreate(BaseModel):
    """Schema for creating a new knowledge graph"""

    name: str = Field(
        ..., min_length=1, max_length=255, examples=["Company Knowledge Base"]
    )
    description: str | None = Field(
        None, max_length=1000, examples=["Internal org chart and company relationships"]
    )
    schema_config: dict[str, Any] | None = Field(None)

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Company Knowledge Base",
                "description": "Internal org chart and company relationships",
            }
        }
    }


class GraphUpdate(BaseModel):
    """Schema for updating a knowledge graph"""

    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = Field(None, max_length=1000)
    schema_config: dict[str, Any] | None = Field(None)
    federatable: bool | None = Field(
        None, description="Enable this graph for cross-graph federation queries"
    )
    federation_group: str | None = Field(
        None, max_length=255, description="Optional named federation group tag"
    )
    auto_snapshot_on_ingestion: bool | None = Field(
        None, description="Auto-snapshot after each ingestion (max 1 per 24h)"
    )
    snapshot_strategy: str | None = Field(
        None,
        description="Snapshot strategy (placeholder — materialized snapshots deferred to Phase 4)",
    )


class GraphResponse(BaseModel):
    """Schema for knowledge graph response"""

    id: UUID
    name: str
    description: str | None
    user_id: UUID
    schema_config: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime
    node_count: int
    relationship_count: int
    status: str
    last_optimized: datetime | None = Field(
        default=None, description="When this graph was last optimized"
    )
    optimization_count: int = Field(
        default=0, description="Number of optimizations performed"
    )
    last_optimization_type: str | None = Field(
        default=None, description="Type of last optimization"
    )
    has_instructions: bool = Field(
        default=False,
        description="Whether graph-level extraction instructions are configured",
    )
    instructions_version: int | None = Field(
        default=None, description="Current instructions version"
    )
    federatable: bool = Field(
        default=False,
        description="Whether this graph is enabled for cross-graph federation",
    )
    federation_group: str | None = Field(
        default=None, description="Named federation group tag"
    )
    auto_snapshot_on_ingestion: bool = Field(
        default=False, description="Auto-snapshot after each ingestion (max 1 per 24h)"
    )
    snapshot_strategy: str | None = Field(
        default=None,
        description="Snapshot strategy placeholder (materialized snapshots deferred to Phase 4)",
    )

    class Config:
        from_attributes = True


class SchemaLearnRequest(BaseModel):
    """Request for learning schema from text"""

    text_sample: str = Field(
        ..., min_length=50, description="Sample text to learn schema from"
    )
    domain_context: str | None = Field(
        None, description="Domain context (e.g., 'medical', 'legal')"
    )
    evolution_mode: str | None = Field(
        "guided", description="Schema evolution mode: strict, guided, permissive"
    )
    max_entities: int | None = Field(20, description="Maximum number of entity types")
    max_relationships: int | None = Field(
        15, description="Maximum number of relationship types"
    )


class IngestDataRequest(BaseModel):
    """Enhanced request for data ingestion with schema evolution"""

    content: str = Field(
        ...,
        min_length=10,
        description="Text content to ingest and extract entities from",
    )
    source_type: str = Field(
        default="text", description="Content type: text, pdf, url, api"
    )
    mode: IngestMode = Field(
        default=IngestMode.INCREMENTAL,
        description=(
            "Ingestion mode: 'incremental' (default) detects changes via SHA256 hash and only "
            "re-processes changed chunks; 'full' deletes all entities and re-extracts; "
            "'upsert' processes all chunks without deleting."
        ),
    )
    graph_schema: dict[str, list[str]] | None = None

    # Per-job extraction overrides (preferred)
    overrides: IngestionOverrides | None = Field(
        None,
        description="Per-job extraction overrides merged with graph-level instructions.",
    )

    # Per-job temporal context — pins world-time bounds for all facts in this document
    temporal_context: TemporalContext | None = Field(
        None,
        description="World-time bounds for facts in this document. Overrides LLM-extracted valid_from/valid_to.",
    )

    # DEPRECATED — kept for backwards compat only; wrapped to overrides.additional_focus
    instructions: str | None = Field(
        None, description="Deprecated. Use overrides.additional_focus instead."
    )

    # Schema evolution parameters
    evolution_mode: str | None = Field(
        "guided", description="Schema evolution mode: strict, guided, permissive"
    )
    max_entities: int | None = Field(20, description="Maximum entity types allowed")
    max_relationships: int | None = Field(
        15, description="Maximum relationship types allowed"
    )
    allow_schema_evolution: bool | None = Field(
        True, description="Allow schema to evolve during ingestion"
    )

    # Relationship property enforcement
    enforce_relationship_properties: bool = Field(
        default=True,
        description="Validate that contextual properties are on relationships, not nodes",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "content": "TechNova Corporation was founded in 2015 by Dr. Sarah Chen and Marcus Webb. The company acquired DataStream Inc. in 2023.",
                "source_type": "text",
                "overrides": {
                    "additional_focus": "Focus on executive roles and acquisition details",
                    "override_density": "dense",
                },
            }
        }
    }

    def resolved_overrides(self) -> Optional["IngestionOverrides"]:
        """
        Return the effective IngestionOverrides, applying backwards compat wrapper:
        if only deprecated `instructions` is set, wrap it as additional_focus.
        """
        import warnings

        if self.overrides is not None:
            return self.overrides
        if self.instructions is not None:
            warnings.warn(
                "IngestDataRequest.instructions is deprecated. Use overrides.additional_focus instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return IngestionOverrides(additional_focus=self.instructions)
        return None


class IngestionJobResponse(BaseModel):
    """Schema for ingestion job response"""

    id: UUID
    graph_id: UUID
    source_type: str | None
    status: str
    progress: int
    error_message: str | None = None
    extracted_entities: int
    extracted_relationships: int
    processed_chunks: int = 0
    similarity_relationships: int = 0
    communities_detected: int = 0
    property_violations_detected: int = Field(
        default=0, description="Banned node properties found during extraction"
    )
    property_violations_migrated: int = Field(
        default=0, description="Banned properties moved to relationships"
    )
    ontology_violations: int = Field(
        default=0, description="Entities that violated the ontology during extraction"
    )
    ontology_coercions: int = Field(
        default=0,
        description="Entities coerced to a closer ontology type during extraction",
    )
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    """Health check response schema"""

    status: str
    service: str
    version: str
    timestamp: datetime
    dependencies: dict[str, Any]


class SchemaValidationRequest(BaseModel):
    """Request for validating a schema"""

    entities: list[str] = Field(..., description="List of entity types")
    relationships: list[str] = Field(..., description="List of relationship types")


class SchemaValidationResponse(BaseModel):
    """Response for schema validation"""

    valid: bool
    entities_count: int
    relationships_count: int
    warnings: list[str] = []
    errors: list[str] = []


class SchemaEvolutionSettings(BaseModel):
    """Settings for schema evolution"""

    mode: str = Field("guided", description="Evolution mode")
    max_entities: int = Field(20, description="Maximum entities")
    max_relationships: int = Field(15, description="Maximum relationships")
    evolution_threshold: float = Field(
        0.3, description="Threshold for triggering evolution"
    )
    auto_consolidate: bool = Field(
        True, description="Automatically consolidate similar entities"
    )


class GraphConfiguration(BaseModel):
    """Complete graph configuration"""

    graph_schema: dict[str, list[str]]
    evolution_settings: SchemaEvolutionSettings
    domain_context: str | None = None
    custom_instructions: str | None = None


# ==================== COMMUNITY DETECTION SCHEMAS ====================


class CommunityDetectRequest(BaseModel):
    force_rebuild: bool = Field(
        False, description="Force re-detection even if status is active"
    )
    levels: int = Field(3, ge=1, le=5, description="Number of hierarchy levels")
    min_entities: int | None = Field(
        None, description="Override minimum entity threshold"
    )


class CommunityDetectResponse(BaseModel):
    job_id: str
    graph_id: str
    status: str
    estimated_entities: int | None = None


class CommunityItem(BaseModel):
    community_id: str
    level: int
    entity_count: int
    weight: float | None = None
    summary: str | None = None
    parent_id: str | None = None
    status: str


class CommunityListResponse(BaseModel):
    communities: list[CommunityItem]
    total: int
    detection_status: str
    last_detected_at: str | None = None


class CommunityMember(BaseModel):
    entity_id: str
    entity_name: str | None = None
    entity_type: str | None = None


class CommunityDetailResponse(BaseModel):
    community_id: str
    level: int
    summary: str | None = None
    entity_count: int
    algorithm: str | None = None
    status: str | None = None
    parent_community: dict[str, Any] | None = None
    child_communities: list[dict[str, Any]] = []
    members: list[CommunityMember] = []
    created_at: Any | None = None
    last_updated: Any | None = None


class CommunityStatusResponse(BaseModel):
    status: str
    last_detected_at: str | None = None
    communities_by_level: dict[str, int] = {}
    entity_count_at_detection: int = 0
    current_entity_count: int = 0


# ==================== VERSIONING SCHEMAS ====================


class VersionCreateRequest(BaseModel):
    label: str | None = Field(
        None,
        max_length=200,
        description="Human-readable label, e.g. 'pre-Q4-ingestion'",
    )
    description: str | None = Field(None, max_length=1000)


class VersionResponse(BaseModel):
    version_id: str
    graph_id: str
    version_number: int
    label: str | None = None
    description: str | None = None
    captured_at: Any  # Neo4j datetime — serialised to str by endpoint
    created_by: str
    parent_version_id: str | None = None
    is_auto: bool = False
    entity_count: int = 0
    relationship_count: int = 0
    created_at: Any


class VersionListResponse(BaseModel):
    versions: list[VersionResponse]
    total: int


class DiffSummary(BaseModel):
    entities_added: int = 0
    entities_deleted: int = 0
    relationships_added: int = 0
    relationships_deleted: int = 0
    property_changes: int = 0


class DiffChange(BaseModel):
    type: (
        str  # entity_added | entity_deleted | relationship_added | relationship_deleted
    )
    entity_id: str | None = None
    rel_id: str | None = None
    name: str | None = None
    entity_type: str | None = None
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    timestamp: Any | None = None


class DiffVersionInfo(BaseModel):
    version_id: str
    label: str | None = None
    captured_at: Any


class VersionDiffResponse(BaseModel):
    from_version: DiffVersionInfo
    to_version: DiffVersionInfo
    summary: DiffSummary
    changes: list[DiffChange] = []
    offset: int = 0
    limit: int = 100
    has_more: bool = False


class RollbackRequest(BaseModel):
    confirm: bool = Field(False, description="Must be true to execute rollback")
    create_checkpoint: bool = Field(
        True, description="Auto-snapshot current state before rolling back"
    )


class RollbackResponse(BaseModel):
    checkpoint_version_id: str | None = None
    entities_restored: int = 0
    entities_soft_deleted: int = 0
    relationships_restored: int = 0
    relationships_soft_deleted: int = 0
    message: str
    staleness_pct: float = 0.0


class AsyncRollbackResponse(BaseModel):
    """Returned when rollback is dispatched asynchronously (graph > 10K nodes)."""

    rollback_job_id: str
    status: str  # always "pending" at dispatch time
    message: str


class RollbackJobResponse(BaseModel):
    rollback_job_id: str
    graph_id: str
    version_id: str
    mode: str
    status: str  # pending | running | done | failed
    progress: int = 0
    entities_restored: int = 0
    entities_soft_deleted: int = 0
    relationships_restored: int = 0
    relationships_soft_deleted: int = 0
    checkpoint_version_id: str | None = None
    error_message: str | None = None
    performed_by: str
    scope: dict[str, Any] | None = None
    celery_task_id: str | None = None
    started_at: Any | None = None
    completed_at: Any | None = None
    created_at: Any | None = None
