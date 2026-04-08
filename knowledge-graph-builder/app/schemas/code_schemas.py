"""
Pydantic schemas for Code Knowledge Graph API.

Covers:
- POST /graphs/{graphId}/code-ingest
- GET  /graphs/{graphId}/code/symbols
- POST /graphs/{graphId}/code/query
- GET  /graphs/{graphId}/jobs/{jobId}  (extended with code-specific fields)
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class CodeIngestMode(str, Enum):
    full = "full"
    incremental = "incremental"


class CodeDepth(str, Enum):
    file = "file"
    function = "function"


class SymbolType(str, Enum):
    Function = "Function"
    Class = "Class"
    Variable = "Variable"
    Module = "Module"
    File = "File"


class CodeLanguage(str, Enum):
    python = "python"
    typescript = "typescript"
    javascript = "javascript"
    go = "go"
    java = "java"


class CodeQueryType(str, Enum):
    callers = "callers"
    callees = "callees"
    dead_code = "dead_code"
    circular_imports = "circular_imports"
    inheritance_chain = "inheritance_chain"
    file_deps = "file_deps"
    semantic_search = "semantic_search"


# ─────────────────────────────────────────────────────────────────────────────
# Code Ingest Request / Response
# ─────────────────────────────────────────────────────────────────────────────

class CodeIngestRequest(BaseModel):
    repo_path: Optional[str] = Field(None, description="Absolute local path to repository")
    git_url: Optional[str] = Field(None, description="Remote git URL to clone")
    branch: str = Field("main", description="Branch to checkout when using git_url")
    mode: CodeIngestMode = Field(CodeIngestMode.incremental, description="full | incremental")
    depth: Optional[CodeDepth] = Field(
        None,
        description=(
            "file = file-level import graph only; "
            "function = full symbol extraction (default). "
            "Auto-switches to 'file' for large repos unless explicitly set."
        ),
    )
    languages: Optional[List[CodeLanguage]] = Field(
        None,
        description="Limit parsing to these languages. Defaults to all supported.",
    )
    exclude_patterns: List[str] = Field(
        default_factory=list,
        description="Glob patterns to exclude (e.g. ['**/test_*', '**/__pycache__/**'])",
    )

    @model_validator(mode="after")
    def require_source(self) -> "CodeIngestRequest":
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
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Symbol List
# ─────────────────────────────────────────────────────────────────────────────

class SymbolItem(BaseModel):
    type: SymbolType
    qualified_name: str
    name: str
    file_path: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    signature: Optional[str] = None
    docstring: Optional[str] = None
    language: Optional[str] = None
    is_async: Optional[bool] = None
    is_method: Optional[bool] = None
    is_test: Optional[bool] = None
    visibility: Optional[str] = None


class SymbolListResponse(BaseModel):
    total: int
    symbols: List[SymbolItem]


# ─────────────────────────────────────────────────────────────────────────────
# Code Query
# ─────────────────────────────────────────────────────────────────────────────

class CodeQueryRequest(BaseModel):
    query_type: CodeQueryType
    params: Dict[str, Any] = Field(default_factory=dict)
    limit: int = Field(50, ge=1, le=500)
    include_tests: bool = Field(
        False,
        description="Include test functions in dead_code results",
    )


class CodeQueryResult(BaseModel):
    query_type: CodeQueryType
    results: List[Dict[str, Any]]
    total: int
