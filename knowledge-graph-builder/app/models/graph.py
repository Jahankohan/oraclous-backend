import uuid

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func

from app.core.database import Base


class KnowledgeGraph(Base):
    """Knowledge graph metadata model"""

    __tablename__ = "knowledge_graphs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    schema_config = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    node_count = Column(Integer, default=0)
    relationship_count = Column(Integer, default=0)
    status = Column(String(50), default="active")
    last_optimized = Column(DateTime(timezone=True), nullable=True)
    optimization_count = Column(Integer, default=0)
    last_optimization_type = Column(String(50), nullable=True)
    similarity_relationships = Column(Integer, default=0)
    communities_count = Column(Integer, default=0)
    communities_detected_at = Column(DateTime(timezone=True), nullable=True)
    communities_status = Column(String(20), default="not_detected")
    entity_count_at_detection = Column(Integer, default=0)
    entity_delta_since_detection = Column(Integer, default=0)
    # Auto-snapshot on ingestion: cap 1 per 24h per graph
    auto_snapshot_on_ingestion = Column(Boolean, default=False)
    auto_snapshot_last_at = Column(DateTime(timezone=True), nullable=True)


class IngestionJob(Base):
    """Data ingestion job tracking"""

    __tablename__ = "ingestion_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    graph_id = Column(UUID(as_uuid=True), nullable=False)
    source_type = Column(String(50))  # 'text', 'pdf', 'url', 'api'
    filename = Column(String(512), nullable=True)
    source_content = Column(Text)
    status = Column(String(50), default="pending")
    progress = Column(Integer, default=0)
    error_message = Column(Text)
    extracted_entities = Column(Integer, default=0)
    extracted_relationships = Column(Integer, default=0)
    processed_chunks = Column(Integer, default=0)
    evolution_mode = Column(String, default="guided")
    schema_before = Column(JSON, nullable=True)
    schema_after = Column(JSON, nullable=True)
    schema_evolved = Column(Boolean, default=False)
    similarity_relationships = Column(Integer, default=0)
    communities_detected = Column(Integer, default=0)
    schema_evolution_count = Column(Integer, default=0)
    entity_deduplication_count = Column(Integer, default=0)
    credits_consumed = Column(String(20), default="0")
    # Ingest mode: full | incremental | upsert (default: incremental)
    ingest_mode = Column(
        String(20), default="incremental", nullable=False, server_default="incremental"
    )
    # Provenance: captures graph_instructions + overrides + resolved at job start time
    effective_instructions = Column(JSON, nullable=True)
    # Ontology enforcement stats
    ontology_violations = Column(Integer, nullable=False, server_default="0")
    ontology_coercions = Column(Integer, nullable=False, server_default="0")
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class GraphRollbackJob(Base):
    """Tracks async rollback jobs for large graphs (>10K nodes)."""

    __tablename__ = "graph_rollback_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    graph_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    version_id = Column(String(255), nullable=False)
    mode = Column(String(50), default="full")
    status = Column(String(50), default="pending")  # pending/running/done/failed
    progress = Column(Integer, default=0)
    entities_restored = Column(Integer, default=0)
    entities_soft_deleted = Column(Integer, default=0)
    relationships_restored = Column(Integer, default=0)
    relationships_soft_deleted = Column(Integer, default=0)
    checkpoint_version_id = Column(String(255), nullable=True)
    error_message = Column(Text, nullable=True)
    performed_by = Column(String(255), nullable=False)
    scope = Column(JSON, nullable=True)
    celery_task_id = Column(String(255), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Connector(Base):
    """External data source connector registry"""

    __tablename__ = "connectors"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    graph_id = Column(Text, nullable=False, index=True)
    user_id = Column(Text, nullable=False, index=True)
    name = Column(Text, nullable=False)
    connector_type = Column(
        Text, nullable=False
    )  # github, notion, linear, confluence, slack, rest_api, webhook_receiver
    status = Column(
        Text, nullable=False, server_default="active"
    )  # active, paused, error
    config = Column(JSONB, nullable=False)
    schedule = Column(Text, nullable=True)  # cron expression; NULL = webhook-only
    last_synced_at = Column(DateTime(timezone=True), nullable=True)
    last_sync_cursor = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ConnectorSyncLog(Base):
    """Sync history for connectors"""

    __tablename__ = "connector_sync_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    connector_id = Column(
        UUID(as_uuid=True),
        ForeignKey("connectors.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    finished_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(Text, nullable=True)  # success, error, partial
    items_processed = Column(Integer, nullable=False, server_default="0")
    entities_extracted = Column(Integer, nullable=False, server_default="0")
    error_message = Column(Text, nullable=True)
    sync_metadata = Column("metadata", JSONB, nullable=True)  # reserved name workaround


class WebhookEvent(Base):
    """Inbound webhook event queue with deduplication"""

    __tablename__ = "webhook_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    connector_id = Column(
        UUID(as_uuid=True),
        ForeignKey("connectors.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_type = Column(Text, nullable=True)
    payload_hash = Column(Text, nullable=False)  # SHA-256 of raw payload for dedup
    payload = Column(JSONB, nullable=False)
    received_at = Column(DateTime(timezone=True), server_default=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(
        Text, nullable=False, server_default="pending"
    )  # pending, processed, error, duplicate
    error_message = Column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("connector_id", "payload_hash", name="uq_webhook_dedup"),
    )


class FallbackJobQueue(Base):
    """
    PostgreSQL fallback queue for Celery tasks when the broker (Redis) is unavailable.

    When a .delay() call raises a broker connection error, the task metadata is
    written here with status='pending'.  A recovery process (manual or automated)
    can scan this table and re-dispatch the tasks once the broker is healthy again.
    """

    __tablename__ = "job_queue"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # The Celery task name (e.g. "app.services.background_jobs.process_ingestion_job")
    task_name = Column(String(255), nullable=False, index=True)
    # JSON-serialised positional args for the task
    args = Column(JSON, nullable=False, server_default="[]")
    # JSON-serialised keyword args for the task
    kwargs = Column(JSON, nullable=False, server_default="{}")
    # Lifecycle status: pending → dispatched / failed
    status = Column(String(50), nullable=False, default="pending", index=True)
    # Human-readable error from the broker that caused the fallback
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
