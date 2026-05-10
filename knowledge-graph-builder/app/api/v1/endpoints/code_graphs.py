"""
Code Knowledge Graph API endpoints.

POST /graphs/{graphId}/code-ingest    — trigger code repository ingestion (202)
GET  /graphs/{graphId}/code/symbols   — list symbols with filters
POST /graphs/{graphId}/code/query     — run one of 7 code-intelligence query types

Job status polling reuses the existing:
    GET /graphs/{graphId}/jobs/{jobId}
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user_id, get_database, verify_graph_access
from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client
from app.models.graph import IngestionJob
from app.schemas.code_schemas import (
    CodeDepth,
    CodeIngestRequest,
    CodeIngestResponse,
    CodeQueryRequest,
    CodeQueryResult,
    SymbolItem,
    SymbolListResponse,
    SymbolType,
)
from app.services.background_job_service import background_job_service

router = APIRouter()
logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# POST /graphs/{graph_id}/code-ingest
# ─────────────────────────────────────────────────────────────────────────────


@router.post(
    "/graphs/{graph_id}/code-ingest",
    response_model=CodeIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a code repository into the Code Knowledge Graph",
    responses={
        403: {"description": "Graph belongs to another user"},
        404: {"description": "Graph not found"},
        503: {"description": "Service unavailable"},
    },
)
async def code_ingest(
    graph_id: UUID,
    request: CodeIngestRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
) -> CodeIngestResponse:
    """
    Submit a code repository for ingestion into the Code Knowledge Graph.

    One of `repo_path` (local absolute path) or `git_url` (remote) is required.
    The response contains a `job_id` — poll `GET /graphs/{id}/jobs/{job_id}` until
    `status` is `completed` or `failed`.

    **Depth auto-switch:** if `depth` is not explicitly set and the repository
    contains more than `CODE_LARGE_REPO_DEPTH_THRESHOLD` files (default 5 000),
    the worker automatically falls back to `depth: file` and records a warning in
    the job status response.

    **Cross-graph edges are prohibited.** External function calls resolve to
    `Dependency` nodes only.
    """
    await verify_graph_access(str(graph_id), "write", user_id)

    # Verify graph exists in Neo4j
    if not neo4j_client.async_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available",
        )
    result = await neo4j_client.async_driver.execute_query(
        "MATCH (g:Graph:__Platform__ {graph_id: $graph_id}) RETURN g.graph_id LIMIT 1",
        {"graph_id": str(graph_id)},
    )
    if not result.records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Graph not found"
        )

    # Determine effective depth (None means auto; will be resolved by the worker)
    effective_depth = request.depth.value if request.depth else None

    # Serialise request parameters into the job row so the Celery worker can
    # reconstruct them without holding a reference to the HTTP request object.
    params: dict[str, Any] = {
        "repo_path": request.repo_path,
        "git_url": request.git_url,
        "branch": request.branch,
        "mode": request.mode.value,
        "depth": effective_depth,
        "languages": (
            [lang.value for lang in request.languages] if request.languages else None
        ),
        "exclude_patterns": request.exclude_patterns,
    }

    job = IngestionJob(
        graph_id=graph_id,
        source_type="code",
        source_content=json.dumps(params),
        status="pending",
        ingest_mode=request.mode.value,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    job_result = await background_job_service.start_code_ingest_job(str(job.id), user_id)
    if job_result["status"] == "failed":
        logger.error(f"Failed to start code ingest job: {job_result['message']}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start code ingestion job",
        )

    logger.info(f"Code ingest job {job.id} queued for graph {graph_id}")
    return CodeIngestResponse(
        job_id=str(job.id),
        graph_id=str(graph_id),
        status="queued",
        mode=request.mode,
        depth=request.depth or CodeDepth.function,
        depth_auto_switched=False,  # worker sets this; not known at queue time
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /graphs/{graph_id}/code/symbols
# ─────────────────────────────────────────────────────────────────────────────


@router.get(
    "/graphs/{graph_id}/code/symbols",
    response_model=SymbolListResponse,
    summary="List code symbols in a graph",
    responses={
        403: {"description": "Graph belongs to another user"},
        503: {"description": "Neo4j not available"},
    },
)
async def list_symbols(
    graph_id: UUID,
    type: SymbolType | None = Query(None, description="Filter by symbol type"),
    language: str | None = Query(None, description="Filter by language (e.g. python)"),
    q: str | None = Query(
        None, description="Full-text search across name, qualified_name, docstring"
    ),
    file_path: str | None = Query(
        None, description="Filter by file path (substring match)"
    ),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    user_id: str = Depends(get_current_user_id),
) -> SymbolListResponse:
    """
    Return code symbols stored in the Code Knowledge Graph for a given graph.

    All results are scoped to the authenticated user's graph (`graph_id` filter
    applied in every Cypher query — multi-tenancy guaranteed).
    """
    await verify_graph_access(str(graph_id), "read", user_id)

    if not neo4j_client.async_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available",
        )

    # Build a label filter. All code symbol labels share the same property set.
    label_filter = f":{type.value}" if type else ":Function|Class|Variable|Module|File"

    # Build WHERE clauses
    where_parts: list[str] = ["n.graph_id = $graph_id"]
    params: dict[str, Any] = {
        "graph_id": str(graph_id),
        "limit": limit,
        "offset": offset,
    }

    if language:
        where_parts.append("n.language = $language")
        params["language"] = language

    if file_path:
        where_parts.append(
            "n.file_path CONTAINS $file_path OR n.path CONTAINS $file_path"
        )
        params["file_path"] = file_path

    if q:
        where_parts.append(
            "(n.name CONTAINS $q OR n.qualified_name CONTAINS $q OR n.docstring CONTAINS $q)"
        )
        params["q"] = q

    where_clause = " AND ".join(where_parts)

    count_query = (
        f"MATCH (n{label_filter}) WHERE {where_clause} RETURN count(n) AS total"
    )
    data_query = f"""
        MATCH (n{label_filter})
        WHERE {where_clause}
        OPTIONAL MATCH (n)-[:DEFINED_IN]->(f:File {{graph_id: $graph_id}})
        RETURN
            labels(n)[0]           AS sym_type,
            n.name                 AS name,
            coalesce(n.qualified_name, n.name) AS qualified_name,
            coalesce(f.path, n.path) AS file_path,
            n.start_line           AS start_line,
            n.end_line             AS end_line,
            n.signature            AS signature,
            n.docstring            AS docstring,
            n.language             AS language,
            n.is_async             AS is_async,
            n.is_method            AS is_method,
            n.is_test              AS is_test,
            n.visibility           AS visibility
        ORDER BY n.qualified_name
        SKIP $offset
        LIMIT $limit
    """

    count_result = await neo4j_client.async_driver.execute_query(count_query, params)
    total = count_result.records[0]["total"] if count_result.records else 0

    data_result = await neo4j_client.async_driver.execute_query(data_query, params)
    symbols: list[SymbolItem] = []
    for record in data_result.records:
        sym_type_str = record.get("sym_type") or "Function"
        try:
            sym_type = SymbolType(sym_type_str)
        except ValueError:
            sym_type = SymbolType.Function
        symbols.append(
            SymbolItem(
                type=sym_type,
                name=record.get("name") or "",
                qualified_name=record.get("qualified_name") or "",
                file_path=record.get("file_path"),
                start_line=record.get("start_line"),
                end_line=record.get("end_line"),
                signature=record.get("signature"),
                docstring=record.get("docstring"),
                language=record.get("language"),
                is_async=record.get("is_async"),
                is_method=record.get("is_method"),
                is_test=record.get("is_test"),
                visibility=record.get("visibility"),
            )
        )

    return SymbolListResponse(total=total, symbols=symbols)


# ─────────────────────────────────────────────────────────────────────────────
# POST /graphs/{graph_id}/code/query
# ─────────────────────────────────────────────────────────────────────────────


@router.post(
    "/graphs/{graph_id}/code/query",
    response_model=CodeQueryResult,
    summary="Run a code-intelligence query",
    responses={
        400: {"description": "Missing required param for query type"},
        403: {"description": "Graph belongs to another user"},
        503: {"description": "Neo4j not available"},
    },
)
async def code_query(
    graph_id: UUID,
    request: CodeQueryRequest,
    user_id: str = Depends(get_current_user_id),
) -> CodeQueryResult:
    """
    Run one of 7 code-intelligence queries against the Code Knowledge Graph.

    All queries filter by `graph_id` — multi-tenancy is enforced at query level.

    | `query_type`        | Required `params` keys     |
    |---------------------|---------------------------|
    | `callers`           | `function_name`            |
    | `callees`           | `function_name`            |
    | `dead_code`         | _(none)_; optional `language` |
    | `circular_imports`  | _(none)_                   |
    | `inheritance_chain` | `class_name`               |
    | `file_deps`         | `file_path`                |
    | `semantic_search`   | `query_text`; optional `top_k` |
    | `data_flow`         | `source_symbol`; optional `depth` (default 10), `direction` (`forward`\|`backward`\|`both`) |
    """
    await verify_graph_access(str(graph_id), "read", user_id)

    if not neo4j_client.async_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available",
        )

    gid = str(graph_id)
    p = request.params
    limit = request.limit
    qt = request.query_type.value

    try:
        records = await _run_code_query(gid, qt, p, limit, request.include_tests)
    except _QueryParamError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from None

    return CodeQueryResult(
        query_type=request.query_type, results=records, total=len(records)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal query dispatcher
# ─────────────────────────────────────────────────────────────────────────────


class _QueryParamError(ValueError):
    pass


async def _run_code_query(
    graph_id: str,
    query_type: str,
    params: dict[str, Any],
    limit: int,
    include_tests: bool,
) -> list[dict[str, Any]]:
    driver = neo4j_client.async_driver

    if query_type == "callers":
        fn = params.get("function_name")
        if not fn:
            raise _QueryParamError("'function_name' is required for query_type=callers")
        result = await driver.execute_query(
            """
            MATCH (caller:Function {graph_id: $graph_id})-[:CALLS]->(callee:Function {graph_id: $graph_id})
            WHERE callee.name = $fn OR callee.qualified_name = $fn
            RETURN caller.qualified_name AS caller, caller.start_line AS line,
                   caller.language AS language
            ORDER BY caller.qualified_name
            LIMIT $limit
            """,
            {"graph_id": graph_id, "fn": fn, "limit": limit},
        )
        return [dict(r) for r in result.records]

    if query_type == "callees":
        fn = params.get("function_name")
        if not fn:
            raise _QueryParamError("'function_name' is required for query_type=callees")
        result = await driver.execute_query(
            """
            MATCH (caller:Function {graph_id: $graph_id})-[:CALLS]->(callee:Function {graph_id: $graph_id})
            WHERE caller.name = $fn OR caller.qualified_name = $fn
            RETURN callee.qualified_name AS callee, callee.start_line AS line,
                   callee.language AS language
            ORDER BY callee.qualified_name
            LIMIT $limit
            """,
            {"graph_id": graph_id, "fn": fn, "limit": limit},
        )
        return [dict(r) for r in result.records]

    if query_type == "dead_code":
        lang_filter = ""
        qp: dict[str, Any] = {"graph_id": graph_id, "limit": limit}
        if params.get("language"):
            lang_filter = "AND f.language = $language"
            qp["language"] = params["language"]
        test_filter = "" if include_tests else "AND NOT f.is_test"
        result = await driver.execute_query(
            f"""
            MATCH (f:Function {{graph_id: $graph_id}})
            WHERE NOT EXISTS {{ MATCH ()-[:CALLS]->(f) }}
              AND NOT f.name IN ['__init__', '__new__', '__del__', 'main']
              AND NOT f.name STARTS WITH '__'
              {test_filter}
              {lang_filter}
            RETURN f.qualified_name AS qualified_name, f.language AS language,
                   f.start_line AS start_line, f.is_test AS is_test
            ORDER BY f.qualified_name
            LIMIT $limit
            """,
            qp,
        )
        return [dict(r) for r in result.records]

    if query_type == "circular_imports":
        result = await driver.execute_query(
            """
            MATCH path = (start:File {graph_id: $graph_id})-[:IMPORTS*2..10]->(start)
            WITH [node IN nodes(path) | node.path] AS cycle
            RETURN DISTINCT cycle
            ORDER BY size(cycle)
            LIMIT $limit
            """,
            {"graph_id": graph_id, "limit": limit},
        )
        return [{"cycle": r["cycle"]} for r in result.records]

    if query_type == "inheritance_chain":
        cls = params.get("class_name")
        if not cls:
            raise _QueryParamError(
                "'class_name' is required for query_type=inheritance_chain"
            )
        result = await driver.execute_query(
            """
            MATCH path = (child:Class {graph_id: $graph_id})-[:INHERITS*0..]->(ancestor:Class {graph_id: $graph_id})
            WHERE child.name = $cls OR child.qualified_name = $cls
            RETURN [node IN nodes(path) | node.qualified_name] AS hierarchy,
                   length(path) AS depth
            ORDER BY depth
            LIMIT $limit
            """,
            {"graph_id": graph_id, "cls": cls, "limit": limit},
        )
        return [
            {"hierarchy": r["hierarchy"], "depth": r["depth"]} for r in result.records
        ]

    if query_type == "file_deps":
        fp = params.get("file_path")
        if not fp:
            raise _QueryParamError("'file_path' is required for query_type=file_deps")
        result = await driver.execute_query(
            """
            MATCH (f:File {graph_id: $graph_id})-[rel:IMPORTS]->(target)
            WHERE f.path = $fp OR f.path ENDS WITH $fp
            RETURN
                CASE WHEN target:Module  THEN target.name
                     WHEN target:File    THEN target.path
                     WHEN target:Dependency THEN target.name
                     ELSE toString(elementId(target))
                END AS import_target,
                labels(target)[0] AS target_type,
                rel.alias AS alias,
                rel.line_number AS line_number,
                rel.is_relative AS is_relative
            ORDER BY rel.line_number
            LIMIT $limit
            """,
            {"graph_id": graph_id, "fp": fp, "limit": limit},
        )
        return [dict(r) for r in result.records]

    if query_type == "semantic_search":
        query_text = params.get("query_text")
        if not query_text:
            raise _QueryParamError(
                "'query_text' is required for query_type=semantic_search"
            )
        top_k = int(params.get("top_k", min(limit, 10)))

        # Generate embedding for the query text via the existing LLM service
        from app.services.llm_service import LLMService

        llm_service = LLMService()
        query_embedding = await llm_service.generate_embedding(query_text)

        result = await driver.execute_query(
            """
            CALL db.index.vector.queryNodes('function_embedding', $top_k, $embedding)
            YIELD node AS f, score
            WHERE f.graph_id = $graph_id
            RETURN f.qualified_name AS qualified_name, f.signature AS signature,
                   f.docstring AS docstring, score
            ORDER BY score DESC
            LIMIT $top_k
            """,
            {"graph_id": graph_id, "top_k": top_k, "embedding": query_embedding},
        )
        return [dict(r) for r in result.records]

    if query_type == "data_flow":
        source_symbol = params.get("source_symbol")
        if not source_symbol:
            raise _QueryParamError(
                "'source_symbol' is required for query_type=data_flow"
            )
        # Clamp depth to safe integer range; embed as literal — Neo4j disallows
        # runtime parameters inside variable-length path range syntax (*1..$n).
        depth = max(1, min(int(params.get("depth", 10)), 50))
        direction = params.get("direction", "forward")

        if direction == "backward":
            cypher = f"""
            MATCH (start {{graph_id: $graph_id}})
            WHERE start.qualified_name = $source_symbol OR start.id = $source_symbol
            MATCH path = (source)-[:FLOWS_TO*1..{depth}]->(start)
            RETURN
                [node IN nodes(path) | coalesce(node.qualified_name, node.name)] AS path_nodes,
                [node IN nodes(path) | labels(node)[0]]                          AS path_labels,
                length(path)                                                       AS depth
            ORDER BY depth
            LIMIT $limit
            """
        elif direction == "both":
            cypher = f"""
            MATCH (start {{graph_id: $graph_id}})
            WHERE start.qualified_name = $source_symbol OR start.id = $source_symbol
            MATCH path = (start)-[:FLOWS_TO*1..{depth}]-(sink)
            RETURN
                [node IN nodes(path) | coalesce(node.qualified_name, node.name)] AS path_nodes,
                [node IN nodes(path) | labels(node)[0]]                          AS path_labels,
                length(path)                                                       AS depth
            ORDER BY depth
            LIMIT $limit
            """
        else:
            # default: forward
            cypher = f"""
            MATCH (start {{graph_id: $graph_id}})
            WHERE start.qualified_name = $source_symbol OR start.id = $source_symbol
            MATCH path = (start)-[:FLOWS_TO*1..{depth}]->(sink)
            RETURN
                [node IN nodes(path) | coalesce(node.qualified_name, node.name)] AS path_nodes,
                [node IN nodes(path) | labels(node)[0]]                          AS path_labels,
                length(path)                                                       AS depth
            ORDER BY depth
            LIMIT $limit
            """

        result = await driver.execute_query(
            cypher,
            {
                "graph_id": graph_id,
                "source_symbol": source_symbol,
                "limit": limit,
            },
        )
        return [
            {
                "path_nodes": r["path_nodes"],
                "path_labels": r["path_labels"],
                "depth": r["depth"],
            }
            for r in result.records
        ]

    raise _QueryParamError(f"Unknown query_type: {query_type}")
