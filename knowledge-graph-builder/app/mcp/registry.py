"""The MCP capability registry — what the projection turns into tools.

A `CapabilitySpec` declares one exposed REST operation and its I/O class. The
projection mechanism (`projection.py`) reads the registry and builds the MCP
tools mechanically — there is no hand-authored tool per endpoint.

The registry below is a **representative set** — one capability per I/O class,
enough to prove every projection pattern end-to-end (TASK-229). The full
*curated* set (ADR-024 D8-R — selected by the ADR-023 D2 structural test) and
the per-tool typed schemas are TASK-230; the exposure allowlist governing what
the registry may contain is TASK-232.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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

    # ASYNC_JOB only — the paired status (poll) tool.
    status_name: str | None = None
    status_method: str = "GET"
    status_path: str | None = None
    status_path_params: tuple[str, ...] = ()
    status_description: str = ""
    job_id_field: str = "id"

    # Free-form notes carried for TASK-230 (typed schemas).
    notes: str = ""


# --- The representative registry --------------------------------------------
# One capability per I/O class. Each is generic (ADR-023 D2): it names a platform
# primitive (graph, ingest) and does one primitive thing.

REGISTRY: tuple[CapabilitySpec, ...] = (
    # PLAIN — request/response.
    CapabilitySpec(
        name="graph.create",
        io_class=IOClass.PLAIN,
        method="POST",
        path="/api/v1/graphs",
        description="Create a new knowledge graph.",
        has_body=True,
    ),
    CapabilitySpec(
        name="graph.get",
        io_class=IOClass.PLAIN,
        method="GET",
        path="/api/v1/graphs/{graph_id}",
        description="Get a knowledge graph by id.",
        path_params=("graph_id",),
    ),
    CapabilitySpec(
        name="graph.list",
        io_class=IOClass.PLAIN,
        method="GET",
        path="/api/v1/graphs",
        description="List the caller's knowledge graphs.",
    ),
    # UPLOAD — file upload (multipart/form-data).
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
    ),
    # STREAMING — SSE; the stream is collected into one tool result.
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
    ),
    # ASYNC_JOB — enqueue + poll; one spec yields a submit tool and a status tool.
    CapabilitySpec(
        name="ingest.text",
        io_class=IOClass.ASYNC_JOB,
        method="POST",
        path="/api/v1/graphs/{graph_id}/ingest",
        description="Submit a text-ingestion job over a graph; returns a job id.",
        path_params=("graph_id",),
        has_body=True,
        status_name="ingest.job_status",
        status_method="GET",
        status_path="/api/v1/graphs/{graph_id}/jobs/{job_id}",
        status_path_params=("graph_id", "job_id"),
        status_description="Poll the status of an ingestion job.",
        job_id_field="id",
    ),
)
