"""
Code Knowledge Graph Parser Service

Implements the 6-stage code ingestion pipeline from ORA-69 spec:
  Stage 0 — Repository Bootstrap  (file discovery + dependency manifest parsing)
  Stage 1 — Delta Detection       (SHA-256 hash comparison against existing File nodes)
  Stage 2 — AST Parsing           (Tree-sitter; language-specific symbol extraction)
  Stage 3 — Cross-File Resolution (call graph + import graph edge creation)
  Stage 4 — Embedding Generation  (Function + Class nodes via existing OpenAI embedder)
  Stage 5 — Neo4j Write           (deterministic MERGE; dependency-safe write order)
  Stage 6 — Stale Cleanup         (async; 7-day TTL matching ORA-60 pattern)

Architecture rules honoured:
  - All Cypher queries filter by graph_id (multi-tenancy)
  - Deterministic entity resolution — no LLM deduplication
  - Uses AsyncDriver (FastAPI) and sync Driver (Celery) via dual-driver pattern
  - Uses existing OpenAIEmbeddings from neo4j_graphrag
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import subprocess
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Language → file extension mapping
# ─────────────────────────────────────────────────────────────────────────────

LANGUAGE_EXTENSIONS: Dict[str, str] = {
    ".py":   "python",
    ".ts":   "typescript",
    ".tsx":  "typescript",
    ".js":   "javascript",
    ".jsx":  "javascript",
    ".go":   "go",
    ".java": "java",
}

MANIFEST_FILES = {
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "package.json",
    "go.mod",
    "pom.xml",
    "build.gradle",
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FileMetadata:
    path: str           # relative to repo root
    abs_path: str
    language: str
    size_bytes: int
    content_hash: str
    is_test: bool


@dataclass
class RawSymbol:
    """Symbol extracted from AST before cross-file resolution."""
    symbol_type: str                  # "Function" | "Class" | "Variable" | "Module"
    name: str
    qualified_name: str
    language: str
    file_path: str                    # relative to repo root
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    signature: Optional[str] = None
    is_async: bool = False
    is_method: bool = False
    is_abstract: bool = False
    is_test: bool = False
    visibility: str = "public"
    type_annotation: Optional[str] = None
    value_preview: Optional[str] = None
    parent_class: Optional[str] = None       # qualified class name for methods
    raw_calls: List[str] = field(default_factory=list)      # unresolved call refs
    raw_imports: List[Dict[str, Any]] = field(default_factory=list)  # {target, alias, line, relative}
    raw_bases: List[str] = field(default_factory=list)      # base class names (Class only)


@dataclass
class DependencyNode:
    name: str
    version_constraint: str = ""
    dep_type: str = "runtime"


@dataclass
class IngestStats:
    files_scanned: int = 0
    files_changed: int = 0
    symbols_added: int = 0
    symbols_updated: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Tree-sitter lazy loader (avoids import-time failures if not installed)
# ─────────────────────────────────────────────────────────────────────────────

_PARSERS: Dict[str, Any] = {}


def _get_parser(language: str) -> Optional[Any]:
    """Return a cached tree-sitter Parser for the given language, or None."""
    if language in _PARSERS:
        return _PARSERS[language]
    try:
        from tree_sitter import Language, Parser  # type: ignore

        if language == "python":
            import tree_sitter_python as mod  # type: ignore
            lang = Language(mod.language())
        elif language in ("typescript", "tsx"):
            import tree_sitter_typescript as mod  # type: ignore
            lang = Language(mod.language_typescript())
        elif language == "javascript":
            import tree_sitter_javascript as mod  # type: ignore
            lang = Language(mod.language())
        elif language == "go":
            import tree_sitter_go as mod  # type: ignore
            lang = Language(mod.language())
        elif language == "java":
            import tree_sitter_java as mod  # type: ignore
            lang = Language(mod.language())
        else:
            return None

        parser = Parser(lang)
        _PARSERS[language] = parser
        return parser
    except Exception as exc:
        logger.warning(f"tree-sitter grammar for '{language}' unavailable: {exc}")
        _PARSERS[language] = None
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0 — Repository Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_repository(
    repo_path: Optional[str],
    git_url: Optional[str],
    branch: str,
    allowed_languages: Optional[Set[str]],
    exclude_patterns: List[str],
) -> Tuple[str, List[FileMetadata], List[DependencyNode]]:
    """
    Clone (if needed) and walk the repository.
    Returns (resolved_repo_path, file_metadata_list, dependency_list).
    """
    if git_url:
        tmp = tempfile.mkdtemp(prefix="kg_code_")
        logger.info(f"Cloning {git_url}@{branch} → {tmp}")
        subprocess.run(
            ["git", "clone", "--depth=1", "--branch", branch, git_url, tmp],
            check=True,
            capture_output=True,
        )
        resolved_path = tmp
    else:
        resolved_path = repo_path  # type: ignore[assignment]

    files: List[FileMetadata] = []
    deps: List[DependencyNode] = []

    for root, dirs, filenames in os.walk(resolved_path):
        # Skip hidden directories (.git, .venv, node_modules, etc.)
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d not in ("node_modules", "__pycache__", "venv", ".venv")
        ]

        for fname in filenames:
            abs_path = os.path.join(root, fname)
            rel_path = os.path.relpath(abs_path, resolved_path)

            # Check exclude patterns
            if any(fnmatch.fnmatch(rel_path, pat) for pat in exclude_patterns):
                continue

            # Dependency manifests
            if fname in MANIFEST_FILES:
                deps.extend(_parse_manifest(abs_path, fname))
                continue

            ext = Path(fname).suffix.lower()
            if ext not in LANGUAGE_EXTENSIONS:
                continue

            lang = LANGUAGE_EXTENSIONS[ext]
            if allowed_languages and lang not in allowed_languages:
                continue

            try:
                content = Path(abs_path).read_bytes()
                content_hash = hashlib.sha256(content).hexdigest()
                is_test = _is_test_path(rel_path)

                files.append(FileMetadata(
                    path=rel_path,
                    abs_path=abs_path,
                    language=lang,
                    size_bytes=len(content),
                    content_hash=content_hash,
                    is_test=is_test,
                ))
            except OSError as e:
                logger.warning(f"Could not read {abs_path}: {e}")

    return resolved_path, files, deps


def _is_test_path(rel_path: str) -> bool:
    parts = Path(rel_path).parts
    return any(
        part.startswith("test_") or part in ("tests", "test")
        for part in parts
    )


def _parse_manifest(abs_path: str, filename: str) -> List[DependencyNode]:
    deps: List[DependencyNode] = []
    try:
        content = Path(abs_path).read_text(encoding="utf-8", errors="ignore")
        if filename == "requirements.txt":
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    m = re.match(r"^([A-Za-z0-9_\-\.]+)\s*([>=<!\^~].*)?$", line)
                    if m:
                        deps.append(DependencyNode(name=m.group(1), version_constraint=m.group(2) or ""))
        elif filename == "package.json":
            import json
            data = json.loads(content)
            for section, dep_type in [("dependencies", "runtime"), ("devDependencies", "dev"), ("optionalDependencies", "optional")]:
                for name, ver in data.get(section, {}).items():
                    deps.append(DependencyNode(name=name, version_constraint=ver, dep_type=dep_type))
        elif filename == "go.mod":
            for line in content.splitlines():
                m = re.match(r"^\s+(\S+)\s+(\S+)", line)
                if m:
                    deps.append(DependencyNode(name=m.group(1), version_constraint=m.group(2)))
    except Exception as e:
        logger.warning(f"Failed to parse manifest {abs_path}: {e}")
    return deps


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Delta Detection (async, uses AsyncDriver)
# ─────────────────────────────────────────────────────────────────────────────

async def detect_deltas(
    graph_id: str,
    files: List[FileMetadata],
    async_driver: Any,
) -> Tuple[List[FileMetadata], List[FileMetadata]]:
    """
    Returns (new_files, changed_files). Unchanged files are skipped.
    Marks stale child nodes for changed files (sets stale_at).
    """
    if not files:
        return [], []

    # Batch lookup existing File nodes
    paths = [f.path for f in files]
    query = """
    UNWIND $paths AS p
    MATCH (f:File {graph_id: $graph_id, path: p})
    RETURN f.path AS path, f.content_hash AS hash
    """
    result = await async_driver.execute_query(
        query,
        {"graph_id": graph_id, "paths": paths},
        database_=settings.NEO4J_DATABASE,
    )
    existing: Dict[str, str] = {rec["path"]: rec["hash"] for rec in result.records}

    new_files: List[FileMetadata] = []
    changed_files: List[FileMetadata] = []

    for f in files:
        if f.path not in existing:
            new_files.append(f)
        elif existing[f.path] != f.content_hash:
            changed_files.append(f)
        # else: unchanged — skip

    # Mark stale for changed files
    if changed_files:
        stale_paths = [f.path for f in changed_files]
        stale_query = """
        UNWIND $paths AS p
        MATCH (f:File {graph_id: $graph_id, path: p})
        OPTIONAL MATCH (f)<-[:DEFINED_IN]-(sym)
        WHERE sym:Function OR sym:Class OR sym:Variable
        SET sym.stale_at = datetime()
        """
        await async_driver.execute_query(
            stale_query,
            {"graph_id": graph_id, "paths": stale_paths},
            database_=settings.NEO4J_DATABASE,
        )

    return new_files, changed_files


def detect_deltas_sync(
    graph_id: str,
    files: List[FileMetadata],
    session: Any,
) -> Tuple[List[FileMetadata], List[FileMetadata]]:
    """Sync version for Celery worker context."""
    if not files:
        return [], []

    paths = [f.path for f in files]
    result = session.run(
        """
        UNWIND $paths AS p
        MATCH (f:File {graph_id: $graph_id, path: p})
        RETURN f.path AS path, f.content_hash AS hash
        """,
        {"graph_id": graph_id, "paths": paths},
    )
    existing: Dict[str, str] = {rec["path"]: rec["hash"] for rec in result}

    new_files: List[FileMetadata] = []
    changed_files: List[FileMetadata] = []
    stale_paths: List[str] = []

    for f in files:
        if f.path not in existing:
            new_files.append(f)
        elif existing[f.path] != f.content_hash:
            changed_files.append(f)
            stale_paths.append(f.path)

    if stale_paths:
        session.run(
            """
            UNWIND $paths AS p
            MATCH (f:File {graph_id: $graph_id, path: p})
            OPTIONAL MATCH (f)<-[:DEFINED_IN]-(sym)
            WHERE sym:Function OR sym:Class OR sym:Variable
            SET sym.stale_at = datetime()
            """,
            {"graph_id": graph_id, "paths": stale_paths},
        )

    return new_files, changed_files


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — AST Parsing (tree-sitter)
# ─────────────────────────────────────────────────────────────────────────────

def parse_file(file_meta: FileMetadata) -> List[RawSymbol]:
    """Parse a single file; returns all symbols found."""
    parser = _get_parser(file_meta.language)
    if parser is None:
        return []

    try:
        content_bytes = Path(file_meta.abs_path).read_bytes()
        tree = parser.parse(content_bytes)
        content_str = content_bytes.decode("utf-8", errors="replace")

        extractor = _EXTRACTORS.get(file_meta.language)
        if extractor is None:
            return []

        return extractor(tree, content_str, file_meta)
    except Exception as e:
        logger.warning(f"Failed to parse {file_meta.path}: {e}")
        return []


def parse_files_parallel(files: List[FileMetadata], max_workers: int = 4) -> List[RawSymbol]:
    """Parse multiple files in parallel using ThreadPoolExecutor."""
    all_symbols: List[RawSymbol] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for symbols in executor.map(parse_file, files):
            all_symbols.extend(symbols)
    return all_symbols


# ─────────────────────────────────────────────────────────────────────────────
# Language-specific AST extractors
# ─────────────────────────────────────────────────────────────────────────────

def _module_name_from_path(rel_path: str) -> str:
    """Convert file relative path to dotted module name."""
    p = Path(rel_path)
    parts = list(p.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _node_text(node: Any, content: str) -> str:
    return content[node.start_byte:node.end_byte]


def _first_string_child(node: Any, content: str) -> Optional[str]:
    """Extract first string literal child (docstring)."""
    for child in node.children:
        if child.type in ("string", "interpreted_string_literal", "raw_string_literal"):
            text = _node_text(child, content).strip("\"'` ")
            # Remove triple-quote markers
            for q in ('"""', "'''", '`'):
                text = text.strip(q)
            return text.strip()
        if child.type == "block":
            return _first_string_child(child, content)
    return None


def _qualified(module: str, *parts: Optional[str]) -> str:
    segments = [s for s in [module, *parts] if s]
    return ".".join(segments)


# ── Python ────────────────────────────────────────────────────────────────────

def _extract_python(tree: Any, content: str, meta: FileMetadata) -> List[RawSymbol]:
    module_name = _module_name_from_path(meta.path)
    symbols: List[RawSymbol] = [
        RawSymbol(
            symbol_type="Module",
            name=module_name.split(".")[-1],
            qualified_name=module_name,
            language="python",
            file_path=meta.path,
            start_line=1,
            end_line=content.count("\n") + 1,
        )
    ]

    def walk(node: Any, class_context: Optional[str] = None) -> None:
        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                cls_name = _node_text(name_node, content)
                qname = _qualified(module_name, class_context, cls_name)
                bases = []
                arg_node = node.child_by_field_name("superclasses")
                if arg_node:
                    for ch in arg_node.children:
                        if ch.type == "identifier":
                            bases.append(_node_text(ch, content))
                symbols.append(RawSymbol(
                    symbol_type="Class",
                    name=cls_name,
                    qualified_name=qname,
                    language="python",
                    file_path=meta.path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    docstring=_first_string_child(node.child_by_field_name("body") or node, content),
                    is_test=meta.is_test,
                    raw_bases=bases,
                ))
                body = node.child_by_field_name("body")
                if body:
                    for child in body.children:
                        walk(child, qname)

        elif node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                fn_name = _node_text(name_node, content)
                qname = _qualified(module_name, class_context, fn_name)
                is_async = node.parent and node.parent.type == "decorated_definition" or \
                    any(ch.type == "async" for ch in node.children)
                params = node.child_by_field_name("parameters")
                ret = node.child_by_field_name("return_type")
                sig = ""
                if params:
                    sig = _node_text(params, content)
                if ret:
                    sig += " -> " + _node_text(ret, content).lstrip("->").strip()

                body = node.child_by_field_name("body")
                raw_calls = _collect_python_calls(body, content) if body else []
                is_test_fn = meta.is_test or fn_name.startswith("test_")

                symbols.append(RawSymbol(
                    symbol_type="Function",
                    name=fn_name,
                    qualified_name=qname,
                    language="python",
                    file_path=meta.path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    docstring=_first_string_child(body, content) if body else None,
                    signature=sig,
                    is_async=is_async,
                    is_method=class_context is not None,
                    is_test=is_test_fn,
                    parent_class=class_context,
                    raw_calls=raw_calls,
                ))

        elif node.type in ("import_statement", "import_from_statement"):
            _collect_python_import(node, content, meta.path, module_name, symbols)

        elif node.type in ("expression_statement", "assignment") and class_context is None:
            _collect_python_variable(node, content, meta, module_name, class_context, symbols)

        else:
            for child in node.children:
                walk(child, class_context)

    for child in tree.root_node.children:
        walk(child)

    return symbols


def _collect_python_calls(body_node: Any, content: str) -> List[str]:
    calls: List[str] = []
    if body_node is None:
        return calls

    def walk(n: Any) -> None:
        if n.type == "call":
            fn = n.child_by_field_name("function")
            if fn:
                calls.append(_node_text(fn, content))
        for ch in n.children:
            walk(ch)

    walk(body_node)
    return calls


def _collect_python_import(node: Any, content: str, file_path: str, module_name: str, symbols: List[RawSymbol]) -> None:
    line = node.start_point[0] + 1
    text = _node_text(node, content)
    is_relative = text.strip().startswith("from .")
    if node.type == "import_statement":
        for ch in node.children:
            if ch.type in ("dotted_name", "aliased_import"):
                target = _node_text(ch, content).split(" as ")[0].strip()
                alias = _node_text(ch, content).split(" as ")[-1].strip() if " as " in _node_text(ch, content) else ""
                symbols.append(RawSymbol(
                    symbol_type="Module",
                    name=target.split(".")[-1],
                    qualified_name=target,
                    language="python",
                    file_path=file_path,
                    start_line=line,
                    end_line=line,
                    raw_imports=[{"target": target, "alias": alias, "line": line, "relative": False}],
                ))
    elif node.type == "import_from_statement":
        from_name = ""
        names: List[str] = []
        for ch in node.children:
            if ch.type == "dotted_name":
                from_name = _node_text(ch, content)
            elif ch.type in ("identifier", "aliased_import"):
                names.append(_node_text(ch, content).split(" as ")[0].strip())
        for n in names or [from_name]:
            target = f"{from_name}.{n}" if from_name and n and n != from_name else from_name or n
            symbols.append(RawSymbol(
                symbol_type="Module",
                name=n,
                qualified_name=target,
                language="python",
                file_path=file_path,
                start_line=line,
                end_line=line,
                raw_imports=[{"target": target, "alias": "", "line": line, "relative": is_relative}],
            ))


def _collect_python_variable(node: Any, content: str, meta: FileMetadata, module_name: str, class_context: Optional[str], symbols: List[RawSymbol]) -> None:
    text = _node_text(node, content).strip()
    # Only module-level or class-level annotated assignments
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^\s=]+)(?:\s*=\s*(.+))?$", text)
    if not m:
        return
    var_name = m.group(1)
    type_ann = m.group(2)
    val = (m.group(3) or "")[:200]
    scope = "class" if class_context else "module"
    qname = _qualified(module_name, class_context, var_name)
    symbols.append(RawSymbol(
        symbol_type="Variable",
        name=var_name,
        qualified_name=qname,
        language="python",
        file_path=meta.path,
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        type_annotation=type_ann,
        value_preview=val,
        parent_class=class_context,
    ))


# ── TypeScript / JavaScript ────────────────────────────────────────────────────

def _extract_ts_js(tree: Any, content: str, meta: FileMetadata) -> List[RawSymbol]:
    module_name = _module_name_from_path(meta.path)
    symbols: List[RawSymbol] = []

    def walk(node: Any, class_context: Optional[str] = None) -> None:
        t = node.type

        if t == "class_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                cls_name = _node_text(name_node, content)
                qname = _qualified(module_name, cls_name)
                bases: List[str] = []
                heritage = node.child_by_field_name("class_heritage")
                if heritage:
                    for ch in heritage.children:
                        if ch.type == "identifier":
                            bases.append(_node_text(ch, content))
                symbols.append(RawSymbol(
                    symbol_type="Class",
                    name=cls_name,
                    qualified_name=qname,
                    language=meta.language,
                    file_path=meta.path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    is_test=meta.is_test,
                    raw_bases=bases,
                ))
                body = node.child_by_field_name("body")
                if body:
                    for ch in body.children:
                        walk(ch, qname)

        elif t in ("function_declaration", "method_definition", "function"):
            name_node = node.child_by_field_name("name")
            if name_node:
                fn_name = _node_text(name_node, content)
                qname = _qualified(module_name, class_context, fn_name)
                is_async = any(ch.type == "async" for ch in node.children)
                is_test_fn = meta.is_test or fn_name.startswith("test_") or fn_name.startswith("it(") or fn_name.startswith("describe(")
                symbols.append(RawSymbol(
                    symbol_type="Function",
                    name=fn_name,
                    qualified_name=qname,
                    language=meta.language,
                    file_path=meta.path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    is_async=is_async,
                    is_method=class_context is not None,
                    is_test=is_test_fn,
                    parent_class=class_context,
                ))

        elif t == "import_declaration":
            line = node.start_point[0] + 1
            source = None
            for ch in node.children:
                if ch.type == "string":
                    source = _node_text(ch, content).strip("\"'")
            if source:
                symbols.append(RawSymbol(
                    symbol_type="Module",
                    name=source.split("/")[-1],
                    qualified_name=source,
                    language=meta.language,
                    file_path=meta.path,
                    start_line=line,
                    end_line=line,
                    raw_imports=[{"target": source, "alias": "", "line": line, "relative": source.startswith(".")}],
                ))

        else:
            for ch in node.children:
                walk(ch, class_context)

    for ch in tree.root_node.children:
        walk(ch)
    return symbols


# ── Go ────────────────────────────────────────────────────────────────────────

def _extract_go(tree: Any, content: str, meta: FileMetadata) -> List[RawSymbol]:
    # Derive package name from directory
    pkg = Path(meta.path).parent.name or "main"
    symbols: List[RawSymbol] = []

    def walk(node: Any) -> None:
        t = node.type
        if t == "function_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                fn_name = _node_text(name_node, content)
                qname = f"{pkg}.{fn_name}"
                is_test = meta.is_test or fn_name.startswith("Test") or fn_name.startswith("Benchmark")
                symbols.append(RawSymbol(
                    symbol_type="Function",
                    name=fn_name,
                    qualified_name=qname,
                    language="go",
                    file_path=meta.path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    is_test=is_test,
                ))
        elif t == "method_declaration":
            name_node = node.child_by_field_name("name")
            recv = node.child_by_field_name("receiver")
            cls_name = ""
            if recv:
                for ch in recv.children:
                    if ch.type in ("parameter_declaration",):
                        for c in ch.children:
                            if c.type in ("type_identifier", "pointer_type"):
                                cls_name = _node_text(c, content).lstrip("*")
                                break
            if name_node:
                fn_name = _node_text(name_node, content)
                qname = f"{pkg}.{cls_name}.{fn_name}" if cls_name else f"{pkg}.{fn_name}"
                symbols.append(RawSymbol(
                    symbol_type="Function",
                    name=fn_name,
                    qualified_name=qname,
                    language="go",
                    file_path=meta.path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    is_method=bool(cls_name),
                    parent_class=f"{pkg}.{cls_name}" if cls_name else None,
                    is_test=meta.is_test,
                ))
        elif t == "type_declaration":
            for ch in node.children:
                if ch.type == "type_spec":
                    name_node = ch.child_by_field_name("name")
                    type_node = ch.child_by_field_name("type")
                    if name_node and type_node and type_node.type == "struct_type":
                        cls_name = _node_text(name_node, content)
                        symbols.append(RawSymbol(
                            symbol_type="Class",
                            name=cls_name,
                            qualified_name=f"{pkg}.{cls_name}",
                            language="go",
                            file_path=meta.path,
                            start_line=ch.start_point[0] + 1,
                            end_line=ch.end_point[0] + 1,
                        ))
        elif t == "import_declaration":
            for ch in node.children:
                if ch.type == "import_spec_list":
                    for spec in ch.children:
                        if spec.type == "import_spec":
                            path_node = spec.child_by_field_name("path")
                            if path_node:
                                target = _node_text(path_node, content).strip("\"")
                                symbols.append(RawSymbol(
                                    symbol_type="Module",
                                    name=target.split("/")[-1],
                                    qualified_name=target,
                                    language="go",
                                    file_path=meta.path,
                                    start_line=spec.start_point[0] + 1,
                                    end_line=spec.end_point[0] + 1,
                                    raw_imports=[{"target": target, "alias": "", "line": spec.start_point[0] + 1, "relative": False}],
                                ))
        else:
            for ch in node.children:
                walk(ch)

    for ch in tree.root_node.children:
        walk(ch)
    return symbols


# ── Java ─────────────────────────────────────────────────────────────────────

def _extract_java(tree: Any, content: str, meta: FileMetadata) -> List[RawSymbol]:
    # Derive package from first package_declaration
    pkg = ""
    for ch in tree.root_node.children:
        if ch.type == "package_declaration":
            pkg = _node_text(ch, content).replace("package ", "").strip(";").strip()
            break

    symbols: List[RawSymbol] = []

    def walk(node: Any, class_context: Optional[str] = None) -> None:
        t = node.type
        if t == "class_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                cls_name = _node_text(name_node, content)
                qname = f"{pkg}.{cls_name}" if pkg else cls_name
                bases: List[str] = []
                sup = node.child_by_field_name("superclass")
                if sup:
                    for ch in sup.children:
                        if ch.type == "type_identifier":
                            bases.append(_node_text(ch, content))
                symbols.append(RawSymbol(
                    symbol_type="Class",
                    name=cls_name,
                    qualified_name=qname,
                    language="java",
                    file_path=meta.path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    is_test=meta.is_test,
                    raw_bases=bases,
                ))
                body = node.child_by_field_name("body")
                if body:
                    for ch in body.children:
                        walk(ch, qname)
        elif t == "method_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                fn_name = _node_text(name_node, content)
                qname = f"{class_context}.{fn_name}" if class_context else fn_name
                is_test = meta.is_test or fn_name.startswith("test")
                symbols.append(RawSymbol(
                    symbol_type="Function",
                    name=fn_name,
                    qualified_name=qname,
                    language="java",
                    file_path=meta.path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    is_method=bool(class_context),
                    is_test=is_test,
                    parent_class=class_context,
                ))
        elif t == "import_declaration":
            text = _node_text(node, content).replace("import ", "").strip(";").strip()
            symbols.append(RawSymbol(
                symbol_type="Module",
                name=text.split(".")[-1],
                qualified_name=text,
                language="java",
                file_path=meta.path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                raw_imports=[{"target": text, "alias": "", "line": node.start_point[0] + 1, "relative": False}],
            ))
        else:
            for ch in node.children:
                walk(ch, class_context)

    for ch in tree.root_node.children:
        walk(ch)
    return symbols


_EXTRACTORS: Dict[str, Callable] = {
    "python":     _extract_python,
    "typescript": _extract_ts_js,
    "javascript": _extract_ts_js,
    "go":         _extract_go,
    "java":       _extract_java,
}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Cross-File Resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_symbols(
    symbols: List[RawSymbol],
    all_file_metas: List[FileMetadata],
) -> Tuple[List[RawSymbol], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (symbols, calls_edges, imports_edges, inherits_edges).
    All edges are dicts ready for Neo4j writes.
    """
    # Build symbol table: qualified_name → RawSymbol
    sym_table: Dict[str, RawSymbol] = {}
    for s in symbols:
        if s.symbol_type in ("Function", "Class", "Module"):
            sym_table[s.qualified_name] = s

    calls_edges: List[Dict[str, Any]] = []
    imports_edges: List[Dict[str, Any]] = []
    inherits_edges: List[Dict[str, Any]] = []

    for sym in symbols:
        # CALLS edges (Function only)
        if sym.symbol_type == "Function":
            for raw_call in sym.raw_calls:
                # Try exact match, then suffix match
                callee = sym_table.get(raw_call)
                if callee is None:
                    for qn, s in sym_table.items():
                        if qn.endswith("." + raw_call) and s.symbol_type == "Function":
                            callee = s
                            break
                if callee:
                    calls_edges.append({
                        "caller_qname": sym.qualified_name,
                        "callee_qname": callee.qualified_name,
                        "line_number": sym.start_line,
                        "argument_count": 0,
                    })
                # Unresolvable → skip (external deps handled via IMPORTS)

        # IMPORTS edges (Module import markers embedded in symbols)
        for imp in sym.raw_imports:
            target = imp["target"]
            is_internal = target in sym_table
            imports_edges.append({
                "source_file": sym.file_path,
                "target": target,
                "is_internal": is_internal,
                "line_number": imp.get("line", 0),
                "alias": imp.get("alias", ""),
                "is_relative": imp.get("relative", False),
            })

        # INHERITS edges (Class only)
        if sym.symbol_type == "Class":
            for order, base_name in enumerate(sym.raw_bases):
                parent = sym_table.get(base_name)
                if parent is None:
                    for qn, s in sym_table.items():
                        if qn.endswith("." + base_name) and s.symbol_type == "Class":
                            parent = s
                            break
                if parent:
                    inherits_edges.append({
                        "child_qname": sym.qualified_name,
                        "parent_qname": parent.qualified_name,
                        "order": order,
                    })

    return symbols, calls_edges, imports_edges, inherits_edges


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Embedding Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_embeddings(symbols: List[RawSymbol]) -> Dict[str, List[float]]:
    """
    Generates embeddings for Function and Class nodes.
    Returns {qualified_name: embedding_vector}.
    Uses OpenAIEmbeddings (synchronous embed_query).
    """
    if not settings.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set — skipping code symbol embeddings")
        return {}

    try:
        from neo4j_graphrag.embeddings import OpenAIEmbeddings  # type: ignore
        embedder = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY,
        )
    except Exception as e:
        logger.warning(f"Failed to initialize embedder: {e}")
        return {}

    embeddable = [
        s for s in symbols
        if s.symbol_type in ("Function", "Class")
    ]

    result: Dict[str, List[float]] = {}
    batch_size = settings.CODE_EMBEDDING_BATCH_SIZE

    for i in range(0, len(embeddable), batch_size):
        batch = embeddable[i:i + batch_size]
        for sym in batch:
            try:
                if sym.symbol_type == "Function":
                    text = f"{sym.qualified_name}\n{sym.signature or ''}\n{sym.docstring or ''}".strip()
                else:
                    text = f"{sym.qualified_name}\n{sym.docstring or ''}".strip()
                result[sym.qualified_name] = embedder.embed_query(text)
            except Exception as e:
                logger.warning(f"Embedding failed for {sym.qualified_name}: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Neo4j Write (sync, for Celery)
# ─────────────────────────────────────────────────────────────────────────────

def write_code_graph_sync(
    graph_id: str,
    session: Any,
    file_metas: List[FileMetadata],
    symbols: List[RawSymbol],
    deps: List[DependencyNode],
    calls_edges: List[Dict[str, Any]],
    imports_edges: List[Dict[str, Any]],
    inherits_edges: List[Dict[str, Any]],
    embeddings: Dict[str, List[float]],
    stats: IngestStats,
) -> None:
    """Write all code KG nodes and relationships (sync driver, batched)."""
    batch_size = settings.BATCH_SIZE

    # 1. File nodes
    for batch in _chunks(file_metas, batch_size):
        params = [
            {
                "graph_id": graph_id,
                "file_id": str(uuid.uuid4()),
                "path": f.path,
                "language": f.language,
                "size_bytes": f.size_bytes,
                "content_hash": f.content_hash,
                "last_parsed_at": datetime.now(timezone.utc).isoformat(),
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "stale_at": None,
            }
            for f in batch
        ]
        session.run(
            """
            UNWIND $rows AS row
            MERGE (f:File {graph_id: row.graph_id, path: row.path})
            ON CREATE SET
                f.file_id = row.file_id,
                f.language = row.language,
                f.size_bytes = row.size_bytes,
                f.content_hash = row.content_hash,
                f.last_parsed_at = datetime(row.last_parsed_at),
                f.ingested_at = datetime(row.ingested_at),
                f.stale_at = null
            ON MATCH SET
                f.language = row.language,
                f.size_bytes = row.size_bytes,
                f.content_hash = row.content_hash,
                f.last_parsed_at = datetime(row.last_parsed_at),
                f.stale_at = null
            """,
            {"rows": params},
        )

    # 2. Module nodes
    modules = [s for s in symbols if s.symbol_type == "Module" and s.raw_imports]
    for batch in _chunks(modules, batch_size):
        params = [
            {"graph_id": graph_id, "name": s.qualified_name, "language": s.language}
            for s in batch
        ]
        session.run(
            """
            UNWIND $rows AS row
            MERGE (m:Module {graph_id: row.graph_id, name: row.name})
            ON CREATE SET m.language = row.language
            """,
            {"rows": params},
        )

    # 3. Dependency nodes
    for batch in _chunks(deps, batch_size):
        params = [
            {"graph_id": graph_id, "dep_id": str(uuid.uuid4()),
             "name": d.name, "version_constraint": d.version_constraint, "dep_type": d.dep_type}
            for d in batch
        ]
        session.run(
            """
            UNWIND $rows AS row
            MERGE (d:Dependency {graph_id: row.graph_id, name: row.name})
            ON CREATE SET d.dep_id = row.dep_id, d.version_constraint = row.version_constraint,
                          d.dep_type = row.dep_type
            ON MATCH SET d.version_constraint = row.version_constraint
            """,
            {"rows": params},
        )

    # 4. Class nodes
    classes = [s for s in symbols if s.symbol_type == "Class"]
    for batch in _chunks(classes, batch_size):
        params = [
            {
                "graph_id": graph_id,
                "class_id": str(uuid.uuid4()),
                "name": s.name,
                "qualified_name": s.qualified_name,
                "language": s.language,
                "start_line": s.start_line,
                "end_line": s.end_line,
                "docstring": s.docstring or "",
                "is_abstract": s.is_abstract,
                "visibility": s.visibility,
                "embedding": embeddings.get(s.qualified_name),
            }
            for s in batch
        ]
        session.run(
            """
            UNWIND $rows AS row
            MERGE (c:Class {graph_id: row.graph_id, qualified_name: row.qualified_name})
            ON CREATE SET
                c.class_id = row.class_id,
                c.name = row.name,
                c.language = row.language,
                c.start_line = row.start_line,
                c.end_line = row.end_line,
                c.docstring = row.docstring,
                c.is_abstract = row.is_abstract,
                c.visibility = row.visibility,
                c.stale_at = null
            ON MATCH SET
                c.start_line = row.start_line,
                c.end_line = row.end_line,
                c.docstring = row.docstring,
                c.stale_at = null
            WITH c, row
            CALL { WITH c, row
                   WHERE row.embedding IS NOT NULL
                   SET c.embedding = row.embedding } IN TRANSACTIONS
            """,
            {"rows": params},
        )
        stats.symbols_added += len(batch)

    # 5. Function nodes
    functions = [s for s in symbols if s.symbol_type == "Function"]
    for batch in _chunks(functions, batch_size):
        params = [
            {
                "graph_id": graph_id,
                "function_id": str(uuid.uuid4()),
                "name": s.name,
                "qualified_name": s.qualified_name,
                "language": s.language,
                "start_line": s.start_line,
                "end_line": s.end_line,
                "signature": s.signature or "",
                "docstring": s.docstring or "",
                "is_async": s.is_async,
                "is_method": s.is_method,
                "is_test": s.is_test,
                "visibility": s.visibility,
                "embedding": embeddings.get(s.qualified_name),
            }
            for s in batch
        ]
        session.run(
            """
            UNWIND $rows AS row
            MERGE (f:Function {graph_id: row.graph_id, qualified_name: row.qualified_name})
            ON CREATE SET
                f.function_id = row.function_id,
                f.name = row.name,
                f.language = row.language,
                f.start_line = row.start_line,
                f.end_line = row.end_line,
                f.signature = row.signature,
                f.docstring = row.docstring,
                f.is_async = row.is_async,
                f.is_method = row.is_method,
                f.is_test = row.is_test,
                f.visibility = row.visibility,
                f.stale_at = null
            ON MATCH SET
                f.start_line = row.start_line,
                f.end_line = row.end_line,
                f.signature = row.signature,
                f.docstring = row.docstring,
                f.is_async = row.is_async,
                f.is_method = row.is_method,
                f.is_test = row.is_test,
                f.stale_at = null
            WITH f, row
            CALL { WITH f, row
                   WHERE row.embedding IS NOT NULL
                   SET f.embedding = row.embedding } IN TRANSACTIONS
            """,
            {"rows": params},
        )
        stats.symbols_added += len(batch)

    # 6. Variable nodes
    variables = [s for s in symbols if s.symbol_type == "Variable"]
    for batch in _chunks(variables, batch_size):
        params = [
            {
                "graph_id": graph_id,
                "variable_id": str(uuid.uuid4()),
                "name": s.name,
                "qualified_name": s.qualified_name,
                "language": s.language,
                "type_annotation": s.type_annotation or "",
                "value_preview": s.value_preview or "",
                "scope": "class" if s.parent_class else "module",
            }
            for s in batch
        ]
        session.run(
            """
            UNWIND $rows AS row
            MERGE (v:Variable {graph_id: row.graph_id, qualified_name: row.qualified_name})
            ON CREATE SET
                v.variable_id = row.variable_id,
                v.name = row.name,
                v.language = row.language,
                v.type_annotation = row.type_annotation,
                v.value_preview = row.value_preview,
                v.scope = row.scope
            ON MATCH SET
                v.type_annotation = row.type_annotation,
                v.value_preview = row.value_preview
            """,
            {"rows": params},
        )

    # 7. Structural relationships: DEFINED_IN, METHOD_OF, SCOPED_TO
    non_module_syms = [s for s in symbols if s.symbol_type in ("Function", "Class", "Variable")]
    for batch in _chunks(non_module_syms, batch_size):
        params = [
            {
                "graph_id": graph_id,
                "qualified_name": s.qualified_name,
                "file_path": s.file_path,
                "start_line": s.start_line,
                "end_line": s.end_line,
                "sym_type": s.symbol_type,
                "parent_class": s.parent_class,
            }
            for s in batch
        ]
        session.run(
            """
            UNWIND $rows AS row
            MATCH (f:File {graph_id: row.graph_id, path: row.file_path})
            CALL {
                WITH row, f
                WHERE row.sym_type = 'Class'
                MATCH (c:Class {graph_id: row.graph_id, qualified_name: row.qualified_name})
                MERGE (c)-[:DEFINED_IN {start_line: row.start_line, end_line: row.end_line}]->(f)
              UNION
                WITH row, f
                WHERE row.sym_type = 'Function'
                MATCH (fn:Function {graph_id: row.graph_id, qualified_name: row.qualified_name})
                MERGE (fn)-[:DEFINED_IN {start_line: row.start_line, end_line: row.end_line}]->(f)
              UNION
                WITH row, f
                WHERE row.sym_type = 'Variable'
                MATCH (v:Variable {graph_id: row.graph_id, qualified_name: row.qualified_name})
                MERGE (v)-[:DEFINED_IN]->(f)
            } IN TRANSACTIONS
            """,
            {"rows": params},
        )
        # METHOD_OF
        methods = [p for p in params if p["sym_type"] == "Function" and p["parent_class"]]
        if methods:
            session.run(
                """
                UNWIND $rows AS row
                MATCH (fn:Function {graph_id: row.graph_id, qualified_name: row.qualified_name})
                MATCH (cls:Class {graph_id: row.graph_id, qualified_name: row.parent_class})
                MERGE (fn)-[:METHOD_OF]->(cls)
                """,
                {"rows": methods, "graph_id": graph_id},
            )
        # SCOPED_TO
        scoped_vars = [p for p in params if p["sym_type"] == "Variable" and p["parent_class"]]
        if scoped_vars:
            session.run(
                """
                UNWIND $rows AS row
                MATCH (v:Variable {graph_id: row.graph_id, qualified_name: row.qualified_name})
                MATCH (cls:Class {graph_id: row.graph_id, qualified_name: row.parent_class})
                MERGE (v)-[:SCOPED_TO]->(cls)
                """,
                {"rows": scoped_vars, "graph_id": graph_id},
            )

    # 8. IMPORTS relationships
    for batch in _chunks(imports_edges, batch_size):
        session.run(
            """
            UNWIND $rows AS row
            MATCH (src:File {graph_id: $graph_id, path: row.source_file})
            CALL {
                WITH src, row
                WHERE row.is_internal = true
                MATCH (tgt:Module {graph_id: $graph_id, name: row.target})
                MERGE (src)-[:IMPORTS {line_number: row.line_number, alias: row.alias,
                                        is_relative: row.is_relative}]->(tgt)
              UNION
                WITH src, row
                WHERE row.is_internal = false
                MERGE (d:Dependency {graph_id: $graph_id, name: row.target})
                ON CREATE SET d.dep_id = randomUUID(), d.dep_type = 'runtime', d.version_constraint = ''
                MERGE (src)-[:IMPORTS {line_number: row.line_number, alias: row.alias,
                                        is_relative: row.is_relative}]->(d)
            } IN TRANSACTIONS
            """,
            {"rows": batch, "graph_id": graph_id},
        )

    # 9. INHERITS relationships
    for batch in _chunks(inherits_edges, batch_size):
        session.run(
            """
            UNWIND $rows AS row
            MATCH (child:Class {graph_id: $graph_id, qualified_name: row.child_qname})
            MATCH (parent:Class {graph_id: $graph_id, qualified_name: row.parent_qname})
            MERGE (child)-[:INHERITS {order: row.order}]->(parent)
            """,
            {"rows": batch, "graph_id": graph_id},
        )

    # 10. CALLS relationships (last — all functions must exist)
    for batch in _chunks(calls_edges, batch_size):
        session.run(
            """
            UNWIND $rows AS row
            MATCH (caller:Function {graph_id: $graph_id, qualified_name: row.caller_qname})
            MATCH (callee:Function {graph_id: $graph_id, qualified_name: row.callee_qname})
            MERGE (caller)-[:CALLS {line_number: row.line_number,
                                     argument_count: row.argument_count}]->(callee)
            """,
            {"rows": batch, "graph_id": graph_id},
        )


def _chunks(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — Stale Cleanup (sync)
# ─────────────────────────────────────────────────────────────────────────────

def cleanup_stale_code_nodes_sync(graph_id: str, session: Any, ttl_days: int = 7) -> int:
    """Delete code symbol nodes past TTL. Returns count deleted."""
    result = session.run(
        """
        MATCH (n)
        WHERE (n:Function OR n:Class OR n:Variable)
          AND n.graph_id = $graph_id
          AND n.stale_at IS NOT NULL
          AND n.stale_at < datetime() - duration({days: $ttl_days})
        CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 500 ROWS
        RETURN count(n) AS deleted
        """,
        {"graph_id": graph_id, "ttl_days": ttl_days},
    )
    rec = result.single()
    return int(rec["deleted"]) if rec else 0


# ─────────────────────────────────────────────────────────────────────────────
# Schema Initialisation (idempotent — safe to run on live DB)
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA_STATEMENTS = [
    # Constraints
    "CREATE CONSTRAINT file_unique     IF NOT EXISTS FOR (f:File)       REQUIRE (f.graph_id, f.path)           IS UNIQUE",
    "CREATE CONSTRAINT function_unique IF NOT EXISTS FOR (f:Function)   REQUIRE (f.graph_id, f.qualified_name) IS UNIQUE",
    "CREATE CONSTRAINT class_unique    IF NOT EXISTS FOR (c:Class)      REQUIRE (c.graph_id, c.qualified_name) IS UNIQUE",
    "CREATE CONSTRAINT module_unique   IF NOT EXISTS FOR (m:Module)     REQUIRE (m.graph_id, m.name)           IS UNIQUE",
    "CREATE CONSTRAINT dep_unique      IF NOT EXISTS FOR (d:Dependency) REQUIRE (d.graph_id, d.name)           IS UNIQUE",
    # Indexes
    "CREATE INDEX file_graph_path   IF NOT EXISTS FOR (f:File)       ON (f.graph_id, f.path)",
    "CREATE INDEX class_graph_name  IF NOT EXISTS FOR (c:Class)      ON (c.graph_id, c.name)",
    "CREATE INDEX func_graph_name   IF NOT EXISTS FOR (f:Function)   ON (f.graph_id, f.name)",
    "CREATE INDEX func_graph_qname  IF NOT EXISTS FOR (f:Function)   ON (f.graph_id, f.qualified_name)",
    "CREATE INDEX var_graph_name    IF NOT EXISTS FOR (v:Variable)   ON (v.graph_id, v.name)",
    "CREATE INDEX dep_graph_name    IF NOT EXISTS FOR (d:Dependency) ON (d.graph_id, d.name)",
    # Full-text index
    """CREATE FULLTEXT INDEX code_symbol_search IF NOT EXISTS
       FOR (n:Function|Class|Variable|Module)
       ON EACH [n.name, n.qualified_name, n.docstring]""",
    # Vector indexes
    """CREATE VECTOR INDEX function_embedding IF NOT EXISTS
       FOR (f:Function) ON (f.embedding)
       OPTIONS {indexConfig: {`vector.dimensions`: 512, `vector.similarity_function`: 'cosine'}}""",
    """CREATE VECTOR INDEX class_embedding IF NOT EXISTS
       FOR (c:Class) ON (c.embedding)
       OPTIONS {indexConfig: {`vector.dimensions`: 512, `vector.similarity_function`: 'cosine'}}""",
]


async def ensure_code_schema(async_driver: Any) -> None:
    """Apply all code KG constraints and indexes (idempotent)."""
    for stmt in _SCHEMA_STATEMENTS:
        try:
            await async_driver.execute_query(stmt, database_=settings.NEO4J_DATABASE)
        except Exception as e:
            logger.warning(f"Schema statement warning (likely already exists): {e}")
    logger.info("Code KG schema constraints and indexes applied")
