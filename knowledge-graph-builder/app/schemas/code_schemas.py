"""
Pydantic schemas for Code Knowledge Graph API.

Covers:
- POST /graphs/{graphId}/code-ingest
- GET  /graphs/{graphId}/code/symbols
- POST /graphs/{graphId}/code/query
- GET  /graphs/{graphId}/jobs/{jobId}  (extended with code-specific fields)
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class CodeIngestMode(StrEnum):
    full = "full"
    incremental = "incremental"


class CodeDepth(StrEnum):
    file = "file"
    function = "function"


class SymbolType(StrEnum):
    Function = "Function"
    Class = "Class"
    Variable = "Variable"
    Module = "Module"
    File = "File"


class CodeLanguage(StrEnum):
    python = "python"
    typescript = "typescript"
    javascript = "javascript"
    go = "go"
    java = "java"


class CodeQueryType(StrEnum):
    callers = "callers"
    callees = "callees"
    dead_code = "dead_code"
    circular_imports = "circular_imports"
    inheritance_chain = "inheritance_chain"
    file_deps = "file_deps"
    semantic_search = "semantic_search"
    data_flow = "data_flow"


# ─────────────────────────────────────────────────────────────────────────────
# Code Ingest Request / Response
# ─────────────────────────────────────────────────────────────────────────────


class CodeIngestRequest(BaseModel):
    repo_path: str | None = Field(None, description="Absolute local path to repository")
    git_url: str | None = Field(None, description="Remote git URL to clone")
    branch: str = Field("main", description="Branch to checkout when using git_url")
    mode: CodeIngestMode = Field(
        CodeIngestMode.incremental, description="full | incremental"
    )
    depth: CodeDepth | None = Field(
        None,
        description=(
            "file = file-level import graph only; "
            "function = full symbol extraction (default). "
            "Auto-switches to 'file' for large repos unless explicitly set."
        ),
    )
    languages: list[CodeLanguage] | None = Field(
        None,
        description="Limit parsing to these languages. Defaults to all supported.",
    )
    exclude_patterns: list[str] = Field(
        default_factory=list,
        description="Glob patterns to exclude (e.g. ['**/test_*', '**/__pycache__/**'])",
    )

    @model_validator(mode="after")
    def require_source(self) -> CodeIngestRequest:
        if not self.repo_path and not self.git_url:
            raise ValueError("One of repo_path or git_url is required")
        return self


class CodeIngestResponse(BaseModel):
    job_id: str
    graph_id: str
    status: str
    mode: CodeIngestMode
    depth: CodeDepth
    depth_auto_switched: bool = Field(
        False,
        description="True when depth was auto-switched to 'file' due to large repo",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Job Status (code-specific extension)
# ─────────────────────────────────────────────────────────────────────────────


class CodeJobStatus(BaseModel):
    job_id: str
    graph_id: str
    status: str
    progress: int = 0
    files_scanned: int = 0
    files_changed: int = 0
    symbols_added: int = 0
    symbols_updated: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Symbol List
# ─────────────────────────────────────────────────────────────────────────────


class SymbolItem(BaseModel):
    type: SymbolType
    qualified_name: str
    name: str
    file_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    signature: str | None = None
    docstring: str | None = None
    language: str | None = None
    is_async: bool | None = None
    is_method: bool | None = None
    is_test: bool | None = None
    visibility: str | None = None


class SymbolListResponse(BaseModel):
    total: int
    symbols: list[SymbolItem]


# ─────────────────────────────────────────────────────────────────────────────
# Code Query
# ─────────────────────────────────────────────────────────────────────────────


class CodeQueryRequest(BaseModel):
    query_type: CodeQueryType
    params: dict[str, Any] = Field(default_factory=dict)
    limit: int = Field(50, ge=1, le=500)
    include_tests: bool = Field(
        False,
        description="Include test functions in dead_code results",
    )


class CodeQueryResult(BaseModel):
    query_type: CodeQueryType
    results: list[dict[str, Any]]
    total: int
