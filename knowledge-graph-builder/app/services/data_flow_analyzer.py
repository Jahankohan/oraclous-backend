"""
Data Flow Analyzer — intra-procedural Python data flow analysis.

Produces FLOWS_TO edges in Neo4j connecting existing Function/Variable/Class
nodes (written by code_parser_service.py).

Phase 1 scope: Python intra-procedural only.

Flow tracking rules:
  1. Parameter → local var via assignment  → FLOWS_TO {via: "assignment"}
  2. Local var  → return value             → FLOWS_TO {via: "return"}
  3. Local var  → function argument        → FLOWS_TO {via: "argument"}
  4. Augmented assignment: x = x + y      → both x and y flow into new x

Taint source heuristics:
  Function name contains: view, handler, endpoint, route, request
  OR decorator includes: @app.route, @router., @api_view
  → first parameter marked with taint: "user_input"

Architecture invariants honoured:
  - graph_id on every FLOWS_TO edge — no exceptions
  - Parameterised Cypher only — never string-interpolate IDs or graph_id
  - WorkerNeo4jManager (sync NullPool) — never async_driver here
  - Python built-in ast module — not Tree-sitter
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Taint heuristics
# ─────────────────────────────────────────────────────────────────────────────

_TAINT_NAME_PATTERNS = re.compile(
    r"(view|handler|endpoint|route|request)", re.IGNORECASE
)
_TAINT_DECORATOR_PATTERNS = re.compile(r"(app\.route|router\.|api_view)", re.IGNORECASE)


def _is_taint_source(func_name: str, decorator_names: list[str]) -> bool:
    """Return True if the function is an HTTP entry-point (taint source)."""
    if _TAINT_NAME_PATTERNS.search(func_name):
        return True
    for dec in decorator_names:
        if _TAINT_DECORATOR_PATTERNS.search(dec):
            return True
    return False


def _decorator_text(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Extract decorator expressions as plain strings."""
    texts: list[str] = []
    for dec in node.decorator_list:
        try:
            texts.append(ast.unparse(dec))
        except Exception:
            pass
    return texts


# ─────────────────────────────────────────────────────────────────────────────
# Internal data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _FlowEdge:
    """Represents one FLOWS_TO edge to write."""

    source_qualified_name: str  # qualified_name of source node
    target_qualified_name: str  # qualified_name of target node
    via: str  # "assignment" | "return" | "argument"
    source_line: int
    source_file: str


@dataclass
class _TaintMark:
    """A variable node that should have taint: "user_input" set."""

    qualified_name: str


@dataclass
class _FunctionAnalysis:
    edges: list[_FlowEdge] = field(default_factory=list)
    taint_marks: list[_TaintMark] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Intra-procedural flow analysis
# ─────────────────────────────────────────────────────────────────────────────


class _FunctionFlowVisitor(ast.NodeVisitor):
    """
    Walk a single function body and emit flow edges.

    Tracks a set of "tainted" names (parameters + variables derived from
    parameters) within the function scope.  The tracked set grows as we see
    assignments whose RHS references already-tracked names.
    """

    def __init__(
        self,
        func_qname: str,
        param_names: list[str],
        module_qname: str,
        source_file: str,
        is_taint_source: bool,
    ) -> None:
        self.func_qname = func_qname
        self.module_qname = module_qname
        self.source_file = source_file

        # tracked_vars: name → qualified_name of the source node in Neo4j
        # For parameters: the Function node is the source (they're not Variable nodes
        # in the KG unless declared at module level).  We model them as flowing FROM
        # the Function node.
        self.tracked_vars: dict[str, str] = {}

        # Build param set; if taint source, mark them
        self.is_taint_source = is_taint_source
        self.param_names = param_names
        for pname in param_names:
            # Each parameter tracks its own Variable node qname so that
            # FLOWS_TO edges start from the param node (required for taint
            # traversal: MATCH (src {taint:'user_input'})-[:FLOWS_TO*]->).
            self.tracked_vars[pname] = f"{func_qname}.{pname}"

        self.edges: list[_FlowEdge] = []
        self.taint_marks: list[_TaintMark] = []

        # Emit Function → param_Variable edges so that API queries starting
        # from the Function node (source_symbol = func_qname) can traverse
        # into the parameter flow graph.  These are written before any
        # assignment edges, so subsequent MATCH (src = param_Variable) finds
        # the node already created by the MERGE in this earlier edge row.
        for pname in param_names:
            param_qname = f"{func_qname}.{pname}"
            self.edges.append(
                _FlowEdge(
                    source_qualified_name=func_qname,
                    target_qualified_name=param_qname,
                    via="parameter",
                    source_line=0,
                    source_file=source_file,
                )
            )

        # If taint source, the first non-self parameter carries taint.
        # The Variable node qualified_name would be module_qname.func_qname.param
        # but those are not guaranteed to exist in Neo4j (only module-level
        # annotated variables are written by code_parser_service).
        # We record taint marks for best-effort: the write step MERGEs them only
        # if the node already exists.
        if is_taint_source:
            for pname in param_names:
                if pname not in ("self", "cls"):
                    qname = f"{func_qname}.{pname}"
                    self.taint_marks.append(_TaintMark(qualified_name=qname))
                    break  # first user-facing param only

    def _names_in_expr(self, node: ast.expr) -> list[str]:
        """Collect all Name node ids referenced in an expression."""
        names: list[str] = []
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                names.append(n.id)
        return names

    def _expr_refs_tracked(self, node: ast.expr) -> list[str]:
        """Return tracked variable names referenced in *node*."""
        return [n for n in self._names_in_expr(node) if n in self.tracked_vars]

    def visit_Assign(self, node: ast.Assign) -> None:
        """Handle simple and augmented-style assignments: x = ..."""
        rhs_tracked = self._expr_refs_tracked(node.value)

        for target in node.targets:
            if isinstance(target, ast.Name):
                tname = target.id
                tgt_qname = f"{self.func_qname}.{tname}"

                for src_name in rhs_tracked:
                    src_qname = self.tracked_vars[src_name]
                    self.edges.append(
                        _FlowEdge(
                            source_qualified_name=src_qname,
                            target_qualified_name=tgt_qname,
                            via="assignment",
                            source_line=node.lineno,
                            source_file=self.source_file,
                        )
                    )

                # The target now becomes tracked (derived from a tracked source)
                if rhs_tracked:
                    self.tracked_vars[tname] = tgt_qname

        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Handle annotated assignments: x: int = ..."""
        if node.value is None:
            return
        rhs_tracked = self._expr_refs_tracked(node.value)
        if isinstance(node.target, ast.Name):
            tname = node.target.id
            tgt_qname = f"{self.func_qname}.{tname}"
            for src_name in rhs_tracked:
                src_qname = self.tracked_vars[src_name]
                self.edges.append(
                    _FlowEdge(
                        source_qualified_name=src_qname,
                        target_qualified_name=tgt_qname,
                        via="assignment",
                        source_line=node.lineno,
                        source_file=self.source_file,
                    )
                )
            if rhs_tracked:
                self.tracked_vars[tname] = tgt_qname
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Handle augmented assignments: x += y, x = x + y style."""
        rhs_tracked = self._expr_refs_tracked(node.value)
        if isinstance(node.target, ast.Name):
            tname = node.target.id
            tgt_qname = f"{self.func_qname}.{tname}"
            # target also flows into itself (augmented)
            all_sources = list(rhs_tracked)
            if tname in self.tracked_vars:
                all_sources.append(tname)
            for src_name in all_sources:
                src_qname = self.tracked_vars.get(
                    src_name, f"{self.func_qname}.{src_name}"
                )
                self.edges.append(
                    _FlowEdge(
                        source_qualified_name=src_qname,
                        target_qualified_name=tgt_qname,
                        via="assignment",
                        source_line=node.lineno,
                        source_file=self.source_file,
                    )
                )
            if all_sources:
                self.tracked_vars[tname] = tgt_qname
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        """Handle return statements: tracked var → function return."""
        if node.value is None:
            return
        rhs_tracked = self._expr_refs_tracked(node.value)
        for src_name in rhs_tracked:
            src_qname = self.tracked_vars[src_name]
            # Use a synthetic .__return__ node to avoid colliding with the
            # existing Function node that already has the same qualified_name.
            self.edges.append(
                _FlowEdge(
                    source_qualified_name=src_qname,
                    target_qualified_name=f"{self.func_qname}.__return__",
                    via="return",
                    source_line=node.lineno,
                    source_file=self.source_file,
                )
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Handle function calls: tracked var passed as argument → FLOWS_TO call site."""
        # Derive a call site qualified name
        try:
            callee_str = ast.unparse(node.func)
        except Exception:
            callee_str = "unknown"

        # For positional arguments
        for arg in node.args:
            tracked = self._expr_refs_tracked(arg)
            for src_name in tracked:
                src_qname = self.tracked_vars[src_name]
                # Target is a synthetic "call site" qualified name:
                # module_qname.callee — best effort
                tgt_qname = f"{self.module_qname}.{callee_str}"
                self.edges.append(
                    _FlowEdge(
                        source_qualified_name=src_qname,
                        target_qualified_name=tgt_qname,
                        via="argument",
                        source_line=node.lineno,
                        source_file=self.source_file,
                    )
                )

        # For keyword arguments
        for kw in node.keywords:
            if kw.value:
                tracked = self._expr_refs_tracked(kw.value)
                for src_name in tracked:
                    src_qname = self.tracked_vars[src_name]
                    tgt_qname = f"{self.module_qname}.{callee_str}"
                    self.edges.append(
                        _FlowEdge(
                            source_qualified_name=src_qname,
                            target_qualified_name=tgt_qname,
                            via="argument",
                            source_line=node.lineno,
                            source_file=self.source_file,
                        )
                    )

        self.generic_visit(node)


def _analyze_function_ast(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    module_qname: str,
    source_file: str,
) -> _FunctionAnalysis:
    """Analyse a single function AST node; return edges + taint marks."""
    func_name = func_node.name
    func_qname = f"{module_qname}.{func_name}" if module_qname else func_name

    # Collect parameter names (skip *args, **kwargs annotations — just names)
    param_names: list[str] = []
    args = func_node.args
    for arg in args.args + args.posonlyargs + args.kwonlyargs:
        param_names.append(arg.arg)
    if args.vararg:
        param_names.append(args.vararg.arg)
    if args.kwarg:
        param_names.append(args.kwarg.arg)

    decorator_texts = _decorator_text(func_node)
    taint = _is_taint_source(func_name, decorator_texts)

    visitor = _FunctionFlowVisitor(
        func_qname=func_qname,
        param_names=param_names,
        module_qname=module_qname,
        source_file=source_file,
        is_taint_source=taint,
    )
    visitor.visit(func_node)

    return _FunctionAnalysis(edges=visitor.edges, taint_marks=visitor.taint_marks)


def _module_qname_from_path(file_path: str) -> str:
    """Convert a file path to a dotted module name (mirrors code_parser_service)."""
    p = Path(file_path)
    parts = list(p.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Neo4j write helpers
# ─────────────────────────────────────────────────────────────────────────────

_MERGE_FLOWS_TO = """
UNWIND $edges AS e
MATCH (src {graph_id: $graph_id, qualified_name: e.src_qname})
MERGE (tgt:Variable {graph_id: $graph_id, qualified_name: e.tgt_qname})
ON CREATE SET
    tgt.variable_id  = randomUUID(),
    tgt.name         = split(e.tgt_qname, '.')[-1],
    tgt.is_local     = true,
    tgt.source_file  = e.source_file
MERGE (src)-[r:FLOWS_TO {
    via: e.via,
    graph_id: $graph_id,
    source_file: e.source_file,
    source_line: e.source_line
}]->(tgt)
RETURN count(r) AS written
"""

_MARK_TAINT = """
UNWIND $marks AS m
MERGE (n:Variable {graph_id: $graph_id, qualified_name: m.qname})
ON CREATE SET
    n.variable_id = randomUUID(),
    n.name        = split(m.qname, '.')[-1],
    n.is_local    = true
SET n.taint = 'user_input'
"""


def _write_edges_sync(
    session: Any,
    graph_id: str,
    flow_edges: list[_FlowEdge],
    taint_marks: list[_TaintMark],
    batch_size: int = 500,
) -> int:
    """Write FLOWS_TO edges + taint marks using a sync Neo4j session."""
    total_written = 0

    # Write taint marks FIRST so that param Variable nodes exist before edges
    # try to MATCH them as src.  Without this ordering, edges whose src is a
    # tainted parameter would fail with "node not found" because the Variable
    # node is only created by the taint-mark MERGE.
    if taint_marks:
        mark_rows = [{"qname": m.qualified_name} for m in taint_marks]
        try:
            session.run(_MARK_TAINT, {"marks": mark_rows, "graph_id": graph_id})
        except Exception as exc:
            logger.warning(f"Taint mark write failed (non-fatal): {exc}")

    # Write edges in batches (src Variable nodes from taint marks now exist)
    edge_rows = [
        {
            "src_qname": e.source_qualified_name,
            "tgt_qname": e.target_qualified_name,
            "via": e.via,
            "source_file": e.source_file,
            "source_line": e.source_line,
        }
        for e in flow_edges
    ]

    for i in range(0, len(edge_rows), batch_size):
        batch = edge_rows[i : i + batch_size]
        result = session.run(
            _MERGE_FLOWS_TO,
            {"edges": batch, "graph_id": graph_id},
        )
        rec = result.single()
        if rec:
            total_written += int(rec["written"])

    return total_written


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────


class DataFlowAnalyzer:
    """
    Intra-procedural Python data flow analyzer.

    Reads Python source via the built-in `ast` module, emits FLOWS_TO edges
    to Neo4j using a sync driver session (WorkerNeo4jManager / NullPool).

    Usage in Celery context:
        with WorkerNeo4jManager() as neo4j:
            driver = neo4j.get_sync_driver()
            analyzer = DataFlowAnalyzer(driver)
            count = analyzer.analyze_file(abs_path, graph_id)
    """

    def __init__(self, sync_driver: Any | None = None) -> None:
        """
        Args:
            sync_driver: Neo4j sync driver (GraphDatabase.driver).
                         Required for any method that writes to Neo4j.
                         Pass None for pure AST analysis (testing / dry-run).
        """
        self._driver = sync_driver

    # ------------------------------------------------------------------
    # Core: analyse a single function's source
    # ------------------------------------------------------------------

    def analyze_function(
        self,
        function_node_id: str,
        source_code: str,
        graph_id: str,
        source_file: str = "<unknown>",
    ) -> int:
        """
        Parse *source_code* as a Python module, find all function definitions,
        and write FLOWS_TO edges for each one.

        Args:
            function_node_id: The qualified_name of the function in the KG.
                              Used to scope the analysis to this function only.
            source_code:       Full Python source (the function's enclosing module).
            graph_id:          Multi-tenancy scope key.
            source_file:       Source path for edge metadata.

        Returns:
            Number of FLOWS_TO edges written to Neo4j.
        """
        try:
            tree = ast.parse(source_code, filename=source_file)
        except SyntaxError as exc:
            logger.warning(f"DataFlowAnalyzer: syntax error in {source_file}: {exc}")
            return 0

        module_qname = _module_qname_from_path(source_file)
        all_edges: list[_FlowEdge] = []
        all_taint: list[_TaintMark] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fqname = f"{module_qname}.{node.name}" if module_qname else node.name
                if fqname != function_node_id and node.name != function_node_id:
                    continue  # only analyse the requested function
                analysis = _analyze_function_ast(node, module_qname, source_file)
                all_edges.extend(analysis.edges)
                all_taint.extend(analysis.taint_marks)

        if not all_edges and not all_taint:
            return 0

        if self._driver is None:
            logger.debug(
                f"DataFlowAnalyzer: no driver — dry-run for {function_node_id}"
            )
            return len(all_edges)

        with self._driver.session() as session:
            return _write_edges_sync(session, graph_id, all_edges, all_taint)

    # ------------------------------------------------------------------
    # Analyse an entire file
    # ------------------------------------------------------------------

    def analyze_file(
        self,
        file_path: str,
        graph_id: str,
        module_qname: str | None = None,
    ) -> int:
        """
        Parse a Python file, analyse every function definition, and write
        FLOWS_TO edges to Neo4j.

        Args:
            file_path:    Absolute path used to read the file.
            graph_id:     Multi-tenancy scope key.
            module_qname: Override the dotted module name used for qualified-name
                          computation.  When omitted, derived from *file_path*.
                          Pass the repo-relative path (FileMetadata.path) so that
                          qualified names match what code_parser_service writes.

        Returns:
            Total number of FLOWS_TO edges written.
        """
        try:
            source_code = Path(file_path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning(f"DataFlowAnalyzer: cannot read {file_path}: {exc}")
            return 0

        try:
            tree = ast.parse(source_code, filename=file_path)
        except SyntaxError as exc:
            logger.warning(f"DataFlowAnalyzer: syntax error in {file_path}: {exc}")
            return 0

        if module_qname is None:
            module_qname = _module_qname_from_path(file_path)
        all_edges: list[_FlowEdge] = []
        all_taint: list[_TaintMark] = []

        # Walk only top-level and class-level function definitions
        # (we do not recurse into nested functions — Phase 1 scope)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                analysis = _analyze_function_ast(node, module_qname, file_path)
                all_edges.extend(analysis.edges)
                all_taint.extend(analysis.taint_marks)

        if not all_edges and not all_taint:
            return 0

        if self._driver is None:
            logger.debug(
                f"DataFlowAnalyzer: no driver — dry-run for {file_path}, "
                f"{len(all_edges)} edges found"
            )
            return len(all_edges)

        with self._driver.session() as session:
            written = _write_edges_sync(session, graph_id, all_edges, all_taint)

        logger.info(
            f"DataFlowAnalyzer: {file_path} — {written} FLOWS_TO edges written "
            f"({len(all_taint)} taint marks)"
        )
        return written

    # ------------------------------------------------------------------
    # Convenience: analyse all Python files from a file_meta list
    # ------------------------------------------------------------------

    def analyze_files(
        self,
        file_metas: list[Any],
        graph_id: str,
    ) -> int:
        """
        Analyse all Python files in *file_metas* (FileMetadata from code_parser_service).

        Args:
            file_metas: List of FileMetadata objects; only language=="python" are processed.
            graph_id:   Multi-tenancy scope key.

        Returns:
            Total FLOWS_TO edges written across all files.
        """
        total = 0
        for fm in file_metas:
            if getattr(fm, "language", None) != "python":
                continue
            try:
                # Use the repo-relative path (fm.path) for module qualified-name
                # computation so that names match code_parser_service output.
                rel_path: str = getattr(fm, "path", None) or fm.abs_path
                module_qname = _module_qname_from_path(rel_path)
                total += self.analyze_file(fm.abs_path, graph_id, module_qname)
            except Exception as exc:
                logger.warning(
                    f"DataFlowAnalyzer: error analysing {fm.abs_path}: {exc}"
                )
        return total
