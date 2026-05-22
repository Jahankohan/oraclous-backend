"""The MCP capability registry — what the projection turns into tools.

A `CapabilitySpec` declares one exposed REST operation and its I/O class. The
projection mechanism (`projection.py`) reads the registry and builds the MCP
tools mechanically — there is no hand-authored tool per endpoint.

The registry below is the **curated primitive set** (ADR-024 D8-R): a deliberate
selection of generic primitives across the capability families that have REST
endpoints — not the mechanical union of all ~150 endpoints. Each spec carries
the endpoint's Pydantic request/response model classes so the projection can
publish a *fully typed* input schema (ADR-023 D4 — no untyped `body: dict`).

Curation rules applied here:

  * Only families with an existing REST endpoint are projected — the projection
    dispatches into existing routes only, it never adds a REST capability.
  * `recipe.*` is intentionally absent: the STORY-034 recipe pipeline has **no
    REST surface** (it was never exposed over REST), so it cannot be projected
    in TASK-230. Exposing the recipe pipeline over REST is a separate gap and
    is out of scope for STORY-035's non-goals.
  * Dangerous operations (ADR-023 D6 — `permissions`, `service-accounts`) and
    bespoke/workflow-shaped endpoints (chat-session CRUD, evaluation runs, org
    dashboards) are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# --- Pydantic request/response models for the curated endpoints --------------
# Imported so each `CapabilitySpec` can name the endpoint's real models; the
# projection annotates the tool's `body` parameter with the request model so
# FastMCP publishes a full nested JSON schema instead of an untyped object.
from app.schemas.agent_schemas import (
    AgentCreate,
    AgentCreateResponse,
    AgentResponse,
    AgentUpdate,
)
from app.schemas.chat_schemas import ChatRequest, ChatResponse
from app.schemas.connector_schemas import (
    ConnectorListResponse,
    ConnectorResponse,
    DbConnectorResponse,
    RegisterConnectorRequest,
    RegisterDbConnectorRequest,
    TriggerDbSyncRequest,
    TriggerDbSyncResponse,
    UpdateConnectorRequest,
)
from app.schemas.federation_schemas import (
    FederatedQueryRequest,
    FederatedQueryResponse,
    FederatedVectorSearchRequest,
    FederatedVectorSearchResponse,
)
from app.schemas.graph_schemas import (
    CommunityDetectRequest,
    CommunityDetectResponse,
    CommunityStatusResponse,
    CommunitySummarizeRequest,
    CommunitySummarizeResponse,
    GraphCreate,
    GraphResponse,
    GraphUpdate,
    IngestDataRequest,
    IngestionJobResponse,
    OntologyResponse,
    OntologySetRequest,
    StructuredIngestRequest,
    StructuredIngestResponse,
)
from app.schemas.memory import (
    ConsolidateResponse,
    MemoryCreate,
    MemoryCreateResponse,
    MemorySearchResponse,
    MemoryUpdate,
    MemoryUpdateResponse,
)
from app.schemas.multimodal import MultiModalJobResponse


class IOClass(str, Enum):
    """The four I/O classes of the REST surface (ADR-023 D3).

    The surface is uniform in its auth/scoping pattern but not in its I/O shape;
    each class has exactly one projection builder.
    """

    PLAIN = "plain"  # request/response  -> one typed tool
    UPLOAD = "upload"  # file upload      -> one upload tool
    STREAMING = "streaming"  # SSE        -> one collecting tool
    ASYNC_JOB = "async_job"  # enqueue+poll -> a submit tool + a status tool


@dataclass(frozen=True)
class CapabilitySpec:
    """One exposed REST operation, declared for mechanical projection.

    `name` is the namespaced MCP tool name (ADR-024 D7-R — `graph.*`, `ingest.*`,
    …). `path` is a REST path template; `{...}` placeholders are filled from the
    tool call's `path_params`.
    """

    name: str
    io_class: IOClass
    method: str
    path: str
    description: str

    # Plain / streaming / async-job: how the call's inputs map to the request.
    path_params: tuple[str, ...] = ()
    query_params: tuple[str, ...] = ()
    has_body: bool = False

    # Typed schemas (TASK-230). `body_model` is the endpoint's Pydantic
    # request-body model class — used as the `body` parameter's annotation so
    # FastMCP publishes a full nested JSON schema (ADR-023 D4). `result_model`
    # is the response model class — surfaced in the tool description.
    body_model: type | None = None
    result_model: type | None = None

    # ASYNC_JOB only — the paired status (poll) tool.
    status_name: str | None = None
    status_method: str = "GET"
    status_path: str | None = None
    status_path_params: tuple[str, ...] = ()
    status_description: str = ""
    status_result_model: type | None = None
    job_id_field: str = "id"

    # Free-form notes.
    notes: str = ""


# --- The curated registry ----------------------------------------------------
# ADR-024 D8-R: a curated primitive set across the families that have REST
# endpoints. Every `body`-carrying spec names its request model, so the
# projection publishes a fully typed input schema.

REGISTRY: tuple[CapabilitySpec, ...] = (
    # === graph.* — graph lifecycle and structured ingestion =================
    CapabilitySpec(
        name="graph.create",
        io_class=IOClass.PLAIN,
        method="POST",
        path="/api/v1/graphs",
        description="Create a new knowledge graph.",
        has_body=True,
        body_model=GraphCreate,
        result_model=GraphResponse,
    ),
    CapabilitySpec(
        name="graph.get",
        io_class=IOClass.PLAIN,
        method="GET",
        path="/api/v1/graphs/{graph_id}",
        description="Get a knowledge graph by id.",
        path_params=("graph_id",),
        result_model=GraphResponse,
    ),
    CapabilitySpec(
        name="graph.list",
        io_class=IOClass.PLAIN,
        method="GET",
        path="/api/v1/graphs",
        description="List the caller's knowledge graphs.",
        result_model=GraphResponse,
    ),
    CapabilitySpec(
        name="graph.update",
        io_class=IOClass.PLAIN,
        method="PUT",
        path="/api/v1/graphs/{graph_id}",
        description="Update a knowledge graph's name or description.",
        path_params=("graph_id",),
        has_body=True,
        body_model=GraphUpdate,
        result_model=GraphResponse,
    ),
    CapabilitySpec(
        name="graph.ask",
        io_class=IOClass.STREAMING,
        method="POST",
        path="/api/v1/chat/stream",
        description=(
            "Ask a question over a graph. The endpoint streams the answer; the "
            "projection collects the whole stream and returns it as one result."
        ),
        has_body=True,
        body_model=ChatRequest,
        result_model=ChatResponse,
    ),
    CapabilitySpec(
        name="graph.ingest_records",
        io_class=IOClass.PLAIN,
        method="POST",
        path="/api/v1/graphs/{graph_id}/ingest-records",
        description=(
            "Ingest structured records (rows/objects) into a graph as nodes "
            "and edges, mapped by the supplied field schema."
        ),
        path_params=("graph_id",),
        has_body=True,
        body_model=StructuredIngestRequest,
        result_model=StructuredIngestResponse,
    ),
    # === schema.* — ontology introspection (folded into graph family) =======
    CapabilitySpec(
        name="schema.get_ontology",
        io_class=IOClass.PLAIN,
        method="GET",
        path="/api/v1/graphs/{graph_id}/ontology",
        description="Get a graph's ontology (its node and edge type schema).",
        path_params=("graph_id",),
        result_model=OntologyResponse,
    ),
    CapabilitySpec(
        name="schema.set_ontology",
        io_class=IOClass.PLAIN,
        method="POST",
        path="/api/v1/graphs/{graph_id}/ontology",
        description="Set or replace a graph's ontology.",
        path_params=("graph_id",),
        has_body=True,
        body_model=OntologySetRequest,
        result_model=OntologyResponse,
    ),
    # === ingest.* — text ingestion (async job) and document upload ==========
    CapabilitySpec(
        name="ingest.text",
        io_class=IOClass.ASYNC_JOB,
        method="POST",
        path="/api/v1/graphs/{graph_id}/ingest",
        description="Submit a text-ingestion job over a graph; returns a job id.",
        path_params=("graph_id",),
        has_body=True,
        body_model=IngestDataRequest,
        result_model=IngestionJobResponse,
        status_name="ingest.job_status",
        status_method="GET",
        status_path="/api/v1/graphs/{graph_id}/jobs/{job_id}",
        status_path_params=("graph_id", "job_id"),
        status_description="Poll the status of an ingestion job.",
        status_result_model=IngestionJobResponse,
        job_id_field="id",
    ),
    CapabilitySpec(
        name="ingest.document",
        io_class=IOClass.UPLOAD,
        method="POST",
        path="/api/v1/graphs/{graph_id}/ingest/document",
        description=(
            "Ingest a PDF/DOCX document into a graph. The file is supplied as "
            "base64 content; the projection forwards it as a multipart upload."
        ),
        path_params=("graph_id",),
        result_model=MultiModalJobResponse,
    ),
    CapabilitySpec(
        name="ingest.image",
        io_class=IOClass.UPLOAD,
        method="POST",
        path="/api/v1/graphs/{graph_id}/ingest/image",
        description=(
            "Ingest an image into a graph via vision extraction. The file is "
            "supplied as base64 content; forwarded as a multipart upload."
        ),
        path_params=("graph_id",),
        result_model=MultiModalJobResponse,
    ),
    # === community.* — community detection results ==========================
    CapabilitySpec(
        name="community.detect",
        io_class=IOClass.PLAIN,
        method="POST",
        path="/api/v1/graphs/{graph_id}/communities/detect",
        description="Run community detection over a graph.",
        path_params=("graph_id",),
        has_body=True,
        body_model=CommunityDetectRequest,
        result_model=CommunityDetectResponse,
    ),
    CapabilitySpec(
        name="community.status",
        io_class=IOClass.PLAIN,
        method="GET",
        path="/api/v1/graphs/{graph_id}/communities/status",
        description="Get the status of community detection for a graph.",
        path_params=("graph_id",),
        query_params=("kind",),
        result_model=CommunityStatusResponse,
    ),
    CapabilitySpec(
        name="community.summarize",
        io_class=IOClass.PLAIN,
        method="POST",
        path="/api/v1/graphs/{graph_id}/communities/summarize",
        description="Generate natural-language summaries of a graph's communities.",
        path_params=("graph_id",),
        has_body=True,
        body_model=CommunitySummarizeRequest,
        result_model=CommunitySummarizeResponse,
    ),
    # === agent.* — agent definitions ========================================
    CapabilitySpec(
        name="agent.create",
        io_class=IOClass.PLAIN,
        method="POST",
        path="/api/v1/graphs/{graph_id}/agents",
        description="Create an agent definition scoped to a graph.",
        path_params=("graph_id",),
        has_body=True,
        body_model=AgentCreate,
        result_model=AgentCreateResponse,
    ),
    CapabilitySpec(
        name="agent.get",
        io_class=IOClass.PLAIN,
        method="GET",
        path="/api/v1/graphs/{graph_id}/agents/{agent_id}",
        description="Get an agent definition by id.",
        path_params=("graph_id", "agent_id"),
        result_model=AgentResponse,
    ),
    CapabilitySpec(
        name="agent.list",
        io_class=IOClass.PLAIN,
        method="GET",
        path="/api/v1/graphs/{graph_id}/agents",
        description="List the agent definitions in a graph.",
        path_params=("graph_id",),
        result_model=AgentResponse,
    ),
    CapabilitySpec(
        name="agent.update",
        io_class=IOClass.PLAIN,
        method="PATCH",
        path="/api/v1/graphs/{graph_id}/agents/{agent_id}",
        description="Update an agent definition.",
        path_params=("graph_id", "agent_id"),
        has_body=True,
        body_model=AgentUpdate,
        result_model=AgentResponse,
    ),
    # === memory.* — memory read/write =======================================
    CapabilitySpec(
        name="memory.store",
        io_class=IOClass.PLAIN,
        method="POST",
        path="/api/v1/graphs/{graph_id}/memories",
        description="Store a memory item in a graph's memory store.",
        path_params=("graph_id",),
        has_body=True,
        body_model=MemoryCreate,
        result_model=MemoryCreateResponse,
    ),
    CapabilitySpec(
        name="memory.search",
        io_class=IOClass.PLAIN,
        method="GET",
        path="/api/v1/graphs/{graph_id}/memories/search",
        description="Search a graph's memory store by query text.",
        path_params=("graph_id",),
        query_params=("query", "type", "scope", "temporal", "min_confidence", "limit"),
        result_model=MemorySearchResponse,
    ),
    CapabilitySpec(
        name="memory.update",
        io_class=IOClass.PLAIN,
        method="PATCH",
        path="/api/v1/graphs/{graph_id}/memories/{memory_id}",
        description="Update a stored memory item.",
        path_params=("graph_id", "memory_id"),
        has_body=True,
        body_model=MemoryUpdate,
        result_model=MemoryUpdateResponse,
    ),
    CapabilitySpec(
        name="memory.consolidate",
        io_class=IOClass.PLAIN,
        method="POST",
        path="/api/v1/graphs/{graph_id}/memories/consolidate",
        description="Consolidate a graph's memory store (merge and prune).",
        path_params=("graph_id",),
        result_model=ConsolidateResponse,
    ),
    # === connector.* — data-source connections ==============================
    CapabilitySpec(
        name="connector.register",
        io_class=IOClass.PLAIN,
        method="POST",
        path="/api/v1/graphs/{graph_id}/connectors",
        description="Register a data-source connector for a graph.",
        path_params=("graph_id",),
        has_body=True,
        body_model=RegisterConnectorRequest,
        result_model=ConnectorResponse,
    ),
    CapabilitySpec(
        name="connector.list",
        io_class=IOClass.PLAIN,
        method="GET",
        path="/api/v1/graphs/{graph_id}/connectors",
        description="List a graph's data-source connectors.",
        path_params=("graph_id",),
        result_model=ConnectorListResponse,
    ),
    CapabilitySpec(
        name="connector.get",
        io_class=IOClass.PLAIN,
        method="GET",
        path="/api/v1/graphs/{graph_id}/connectors/{connector_id}",
        description="Get a data-source connector by id.",
        path_params=("graph_id", "connector_id"),
        result_model=ConnectorResponse,
    ),
    CapabilitySpec(
        name="connector.update",
        io_class=IOClass.PLAIN,
        method="PATCH",
        path="/api/v1/graphs/{graph_id}/connectors/{connector_id}",
        description="Update a data-source connector.",
        path_params=("graph_id", "connector_id"),
        has_body=True,
        body_model=UpdateConnectorRequest,
        result_model=ConnectorResponse,
    ),
    CapabilitySpec(
        name="connector.register_database",
        io_class=IOClass.PLAIN,
        method="POST",
        path="/api/v1/graphs/{graph_id}/connectors/database",
        description="Register a database connector for a graph.",
        path_params=("graph_id",),
        has_body=True,
        body_model=RegisterDbConnectorRequest,
        result_model=DbConnectorResponse,
    ),
    CapabilitySpec(
        name="connector.sync_database",
        io_class=IOClass.PLAIN,
        method="POST",
        path="/api/v1/graphs/{graph_id}/connectors/database/{connector_id}/sync",
        description="Trigger a sync run for a database connector.",
        path_params=("graph_id", "connector_id"),
        has_body=True,
        body_model=TriggerDbSyncRequest,
        result_model=TriggerDbSyncResponse,
    ),
    # === federation.* — cross-graph federation ==============================
    CapabilitySpec(
        name="federation.query",
        io_class=IOClass.PLAIN,
        method="POST",
        path="/api/v1/graphs/federate/query",
        description="Run a federated query across multiple graphs.",
        has_body=True,
        body_model=FederatedQueryRequest,
        result_model=FederatedQueryResponse,
    ),
    CapabilitySpec(
        name="federation.vector_search",
        io_class=IOClass.PLAIN,
        method="POST",
        path="/api/v1/graphs/federate/vector-search",
        description="Run a federated vector search across multiple graphs.",
        has_body=True,
        body_model=FederatedVectorSearchRequest,
        result_model=FederatedVectorSearchResponse,
    ),
)
