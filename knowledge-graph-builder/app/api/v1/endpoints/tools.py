"""GET /api/v1/tools/manifest — static tool discovery endpoint for oraclous-core.

Assembled once at import time from the ORA-343 spec (Section 4 & 10.1).
No database or Neo4j call; safe to cache with a short TTL by callers.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.config import settings

router = APIRouter()

# ---------------------------------------------------------------------------
# Manifest — built once at module load, never mutated.
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "kg.create_graph",
        "description": "Create a new knowledge graph for the caller's tenant.",
        "input_schema": {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "org_id": {"type": "string"},
            },
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "graph_id": {"type": "string"},
                "name": {"type": "string"},
                "status": {"type": "string", "enum": ["active"]},
                "created_at": {"type": "string", "format": "date-time"},
            },
        },
        "idempotent": False,
        "async": False,
    },
    {
        "name": "kg.ingest",
        "description": "Ingest data into a graph; optionally wait for completion.",
        "input_schema": {
            "type": "object",
            "required": ["graph_id", "data"],
            "properties": {
                "graph_id": {"type": "string"},
                "data": {
                    "type": "object",
                    "properties": {
                        "source_type": {
                            "type": "string",
                            "enum": ["text", "url", "file_path", "structured"],
                        },
                        "content": {"type": "string"},
                        "metadata": {"type": "object"},
                    },
                },
                "mode": {
                    "type": "string",
                    "enum": ["entity", "relationship", "full"],
                },
                "wait": {"type": "boolean", "default": False},
                "wait_timeout_s": {"type": "integer", "default": 120},
            },
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["queued", "completed", "failed"],
                },
                "poll_url": {"type": "string"},
                "entities_created": {"type": "integer"},
                "relationships_created": {"type": "integer"},
                "duration_ms": {"type": "integer"},
            },
        },
        "idempotent": False,
        "async": True,
    },
    {
        "name": "kg.run_recipe",
        "description": "Run a recipe-based ingest against a graph.",
        "input_schema": {
            "type": "object",
            "required": ["graph_id", "recipe_id"],
            "properties": {
                "graph_id": {"type": "string"},
                "recipe_id": {"type": "string"},
                "records": {"type": "array", "items": {"type": "object"}},
                "wait": {"type": "boolean", "default": False},
                "wait_timeout_s": {"type": "integer", "default": 120},
            },
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["queued", "completed", "failed"],
                },
                "poll_url": {"type": "string"},
                "entities_created": {"type": "integer"},
                "relationships_created": {"type": "integer"},
                "duration_ms": {"type": "integer"},
            },
        },
        "idempotent": False,
        "async": True,
    },
    {
        "name": "kg.query",
        "description": "Execute a parameterized Cypher query against a graph.",
        "input_schema": {
            "type": "object",
            "required": ["graph_id", "cypher"],
            "properties": {
                "graph_id": {"type": "string"},
                "cypher": {"type": "string"},
                "params": {"type": "object"},
                "limit": {"type": "integer", "default": 50, "maximum": 500},
            },
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "rows": {"type": "array", "items": {"type": "object"}},
                "count": {"type": "integer"},
                "truncated": {"type": "boolean"},
            },
        },
        "idempotent": True,
        "async": False,
    },
    {
        "name": "kg.ask",
        "description": "Ask a natural-language question against a graph (RAG).",
        "input_schema": {
            "type": "object",
            "required": ["graph_id", "question"],
            "properties": {
                "graph_id": {"type": "string"},
                "question": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["keyword", "similarity", "hybrid", "graph_traversal"],
                    "default": "hybrid",
                },
                "session_id": {"type": "string"},
            },
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity_id": {"type": "string"},
                            "label": {"type": "string"},
                            "snippet": {"type": "string"},
                            "score": {"type": "number"},
                        },
                    },
                },
                "session_id": {"type": "string"},
            },
        },
        "idempotent": True,
        "async": False,
    },
    {
        "name": "kg.store_fact",
        "description": "Store a memory fact in an agent's memory graph.",
        "input_schema": {
            "type": "object",
            "required": ["graph_id", "content"],
            "properties": {
                "graph_id": {"type": "string"},
                "content": {"type": "string"},
                "type": {
                    "type": "string",
                    "enum": ["fact", "observation", "reflection", "procedure"],
                },
                "scope": {
                    "type": "string",
                    "enum": ["agent", "session", "global"],
                    "default": "agent",
                },
                "agent_id": {"type": "string"},
                "session_id": {"type": "string"},
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 1.0,
                },
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "created_at": {"type": "string", "format": "date-time"},
            },
        },
        "idempotent": False,
        "async": False,
    },
    {
        "name": "kg.recall",
        "description": "Search an agent's memory graph for relevant facts.",
        "input_schema": {
            "type": "object",
            "required": ["graph_id", "query"],
            "properties": {
                "graph_id": {"type": "string"},
                "query": {"type": "string"},
                "type": {
                    "type": ["string", "null"],
                    "enum": [
                        "fact",
                        "observation",
                        "reflection",
                        "procedure",
                        None,
                    ],
                },
                "scope": {
                    "type": ["string", "null"],
                    "enum": ["agent", "session", "global", None],
                },
                "temporal": {
                    "type": "string",
                    "enum": ["current", "all"],
                    "default": "current",
                },
                "min_confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.0,
                },
                "limit": {"type": "integer", "default": 20, "maximum": 100},
            },
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "memories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "string"},
                            "content": {"type": "string"},
                            "type": {"type": "string"},
                            "scope": {"type": "string"},
                            "confidence": {"type": "number"},
                            "relevance_score": {"type": "number"},
                            "created_at": {"type": "string", "format": "date-time"},
                        },
                    },
                },
                "total": {"type": "integer"},
            },
        },
        "idempotent": True,
        "async": False,
    },
    {
        "name": "kg.federate",
        "description": "Run a federated query across multiple graphs.",
        "input_schema": {
            "type": "object",
            "required": ["graph_ids", "query"],
            "properties": {
                "graph_ids": {"type": "array", "items": {"type": "string"}},
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "graph_id": {"type": "string"},
                            "entity_id": {"type": "string"},
                            "label": {"type": "string"},
                            "type": {"type": "string"},
                            "score": {"type": "number"},
                        },
                    },
                },
            },
        },
        "idempotent": True,
        "async": False,
    },
    {
        "name": "kg.job_status",
        "description": "Poll the status of an async ingest or recipe job.",
        "input_schema": {
            "type": "object",
            "required": ["graph_id", "job_id"],
            "properties": {
                "graph_id": {"type": "string"},
                "job_id": {"type": "string"},
            },
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["queued", "running", "completed", "failed"],
                },
                "progress_pct": {"type": "integer"},
                "entities_created": {"type": "integer"},
                "error": {"type": ["string", "null"]},
                "started_at": {"type": ["string", "null"], "format": "date-time"},
                "completed_at": {
                    "type": ["string", "null"],
                    "format": "date-time",
                },
            },
        },
        "idempotent": True,
        "async": False,
    },
]


def _build_manifest() -> dict:
    return {
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "base_url": settings.PUBLIC_BASE_URL,
        "tools": _TOOLS,
    }


# Built once at startup; settings are immutable after that.
_MANIFEST = _build_manifest()


@router.get(
    "/tools/manifest",
    summary="Tool manifest for oraclous-core registration",
    tags=["tools"],
    response_class=JSONResponse,
)
async def get_tools_manifest() -> JSONResponse:
    """
    Return the static tool manifest consumed by oraclous-core for tool
    registration.  No auth required — this endpoint is public for service
    discovery.  oraclous-core should cache the response with a ≤5-minute TTL.
    """
    return JSONResponse(content=_MANIFEST)
