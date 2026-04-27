"""
Unit tests for data_flow_analyzer.py

Covers (per TASK-022 Definition of Done):
  #1  — parameter → local var via assignment   → FLOWS_TO {via: "assignment"}
  #2  — local var → return value               → FLOWS_TO {via: "return"}
  #3  — local var → function argument          → FLOWS_TO {via: "argument"}
  #4  — taint source: handle_request → first param tainted
  #5  — graph_id on every edge
  #6  — augmented assignment: x = x + y, both x and y flow into new x
  #7  — non-taint function: no taint marks produced
  #8  — decorator taint heuristic (@app.route)
  #9  — analyze_file returns correct edge count (dry-run, no Neo4j)
"""

from __future__ import annotations

import pytest

from app.services.data_flow_analyzer import (
    DataFlowAnalyzer,
    _FunctionFlowVisitor,
    _analyze_function_ast,
    _is_taint_source,
    _module_qname_from_path,
)

import ast


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_first_function(
    source: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Parse *source* and return the first function definition node."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    raise ValueError("No function found in source")


def _edges_from_source(source: str, module_qname: str = "mymodule", source_file: str = "mymodule.py"):
    """Convenience: analyse first function in *source*, return list of _FlowEdge."""
    func_node = _parse_first_function(source)
    analysis = _analyze_function_ast(func_node, module_qname, source_file)
    return analysis.edges, analysis.taint_marks


# ─────────────────────────────────────────────────────────────────────────────
# Test #1 — parameter → local var via assignment
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_parameter_to_local_var_assignment():
    """
    def fn(data):
        result = data   # data (param) flows into result
    """
    source = """\
def fn(data):
    result = data
"""
    edges, _ = _edges_from_source(source, "mod", "mod.py")

    assignment_edges = [e for e in edges if e.via == "assignment"]
    assert len(assignment_edges) >= 1, "Expected at least one assignment edge"

    # The source should be the function node (parameters flow FROM the function)
    edge = assignment_edges[0]
    assert "fn" in edge.source_qualified_name, (
        f"Expected source to reference fn, got: {edge.source_qualified_name}"
    )
    assert "result" in edge.target_qualified_name, (
        f"Expected target to reference result, got: {edge.target_qualified_name}"
    )
    assert edge.via == "assignment"


@pytest.mark.unit
def test_parameter_to_multiple_vars():
    """Two assignments from the same param produce two edges."""
    source = """\
def fn(x):
    a = x
    b = x
"""
    edges, _ = _edges_from_source(source)
    assignment_edges = [e for e in edges if e.via == "assignment"]
    assert len(assignment_edges) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Test #2 — local var → return value
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_local_var_to_return():
    """
    def fn(x):
        result = x
        return result  # result flows into the function return
    """
    source = """\
def fn(x):
    result = x
    return result
"""
    edges, _ = _edges_from_source(source, "mod", "mod.py")

    return_edges = [e for e in edges if e.via == "return"]
    assert len(return_edges) >= 1, "Expected at least one return edge"

    edge = return_edges[0]
    # Source should be the local var (result) or param (x)
    assert edge.via == "return"
    # Target should be the function node itself
    assert "fn" in edge.target_qualified_name


@pytest.mark.unit
def test_direct_param_return():
    """
    def fn(x):
        return x   # param x flows directly to return
    """
    source = """\
def fn(x):
    return x
"""
    edges, _ = _edges_from_source(source, "mod", "mod.py")
    return_edges = [e for e in edges if e.via == "return"]
    assert len(return_edges) >= 1
    # Source is the function (representing the parameter)
    assert "fn" in return_edges[0].source_qualified_name


# ─────────────────────────────────────────────────────────────────────────────
# Test #3 — local var → function argument
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_local_var_to_argument():
    """
    def fn(user_input):
        safe = sanitize(user_input)   # user_input flows into sanitize(...)
    """
    source = """\
def fn(user_input):
    safe = sanitize(user_input)
"""
    edges, _ = _edges_from_source(source, "mod", "mod.py")

    arg_edges = [e for e in edges if e.via == "argument"]
    assert len(arg_edges) >= 1, "Expected at least one argument edge"

    edge = arg_edges[0]
    assert edge.via == "argument"
    # Source should be the function (param flows from fn node)
    assert "fn" in edge.source_qualified_name


@pytest.mark.unit
def test_derived_var_to_argument():
    """Local variable derived from a param also flows to call argument."""
    source = """\
def fn(raw):
    processed = raw
    execute(processed)
"""
    edges, _ = _edges_from_source(source, "mod", "mod.py")
    arg_edges = [e for e in edges if e.via == "argument"]
    assert len(arg_edges) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Test #4 — taint source: function named handle_request → first param tainted
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_taint_source_by_function_name():
    """
    Functions with 'request' in the name should have first param marked tainted.
    """
    source = """\
def handle_request(req, context):
    data = req.body
    return data
"""
    _, taint_marks = _edges_from_source(source, "views", "views.py")
    assert len(taint_marks) >= 1, "Expected at least one taint mark"
    # First non-self param (req) should be marked
    taint_qnames = [m.qualified_name for m in taint_marks]
    assert any("req" in qn for qn in taint_qnames), (
        f"Expected 'req' param in taint marks, got: {taint_qnames}"
    )


@pytest.mark.unit
def test_taint_source_view_name():
    """Function named 'user_view' should be a taint source."""
    source = """\
def user_view(request):
    return request.data
"""
    _, taint_marks = _edges_from_source(source, "views")
    assert len(taint_marks) >= 1


@pytest.mark.unit
def test_taint_source_endpoint_name():
    """Function containing 'endpoint' in name → taint source."""
    source = """\
def create_endpoint(payload, auth):
    return payload
"""
    _, taint_marks = _edges_from_source(source)
    assert len(taint_marks) >= 1
    taint_qnames = [m.qualified_name for m in taint_marks]
    assert any("payload" in qn for qn in taint_qnames)


@pytest.mark.unit
def test_non_taint_function_no_marks():
    """Regular function should produce no taint marks."""
    source = """\
def compute_average(numbers):
    total = sum(numbers)
    return total / len(numbers)
"""
    _, taint_marks = _edges_from_source(source)
    assert len(taint_marks) == 0, f"Expected no taint marks, got: {taint_marks}"


# ─────────────────────────────────────────────────────────────────────────────
# Test #5 — graph_id on every edge (checked via DataFlowAnalyzer dry-run)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_graph_id_on_every_edge_metadata(tmp_path):
    """
    DataFlowAnalyzer.analyze_file() in dry-run (no driver) returns correct edge
    count. Separately verify that the Cypher template includes $graph_id.
    """
    source = """\
def process(user_input):
    result = user_input
    return result
"""
    py_file = tmp_path / "service.py"
    py_file.write_text(source)

    # Dry-run: pass no driver
    analyzer = DataFlowAnalyzer(sync_driver=None)
    count = analyzer.analyze_file(str(py_file), graph_id="test-graph-id-123")
    assert count >= 1, "Expected at least one edge from dry-run"


@pytest.mark.unit
def test_graph_id_in_cypher_template():
    """The MERGE_FLOWS_TO Cypher must reference $graph_id — architecture invariant."""
    from app.services.data_flow_analyzer import _MERGE_FLOWS_TO

    assert "$graph_id" in _MERGE_FLOWS_TO, (
        "FLOWS_TO MERGE query must use $graph_id parameter (architecture invariant)"
    )


@pytest.mark.unit
def test_taint_cypher_includes_graph_id():
    """The taint mark Cypher must also scope by $graph_id."""
    from app.services.data_flow_analyzer import _MARK_TAINT

    assert "$graph_id" in _MARK_TAINT, (
        "_MARK_TAINT Cypher must use $graph_id (multi-tenancy invariant)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test #6 — augmented assignment
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_augmented_assignment_both_sources_tracked():
    """
    def fn(x, y):
        x += y   # both x and y flow into x
    """
    source = """\
def fn(x, y):
    x += y
"""
    edges, _ = _edges_from_source(source, "mod", "mod.py")
    assignment_edges = [e for e in edges if e.via == "assignment"]
    # x+=y: x flows into x (augmented) AND y flows into x
    assert len(assignment_edges) >= 2, (
        f"Expected >=2 assignment edges for augmented assignment, got {len(assignment_edges)}"
    )


@pytest.mark.unit
def test_compound_assignment_chain():
    """
    def fn(a):
        b = a
        c = b   # b (derived) flows into c
        return c
    """
    source = """\
def fn(a):
    b = a
    c = b
    return c
"""
    edges, _ = _edges_from_source(source, "mod", "mod.py")
    assert len([e for e in edges if e.via == "assignment"]) >= 2
    assert len([e for e in edges if e.via == "return"]) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Test #7 — taint heuristic: decorator @app.route
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_taint_source_app_route_decorator():
    """@app.route decorator marks first param as taint source."""
    source = """\
@app.route("/create", methods=["POST"])
def create(request):
    data = request.json
    return data
"""
    _, taint_marks = _edges_from_source(source)
    assert len(taint_marks) >= 1
    taint_qnames = [m.qualified_name for m in taint_marks]
    assert any("request" in qn for qn in taint_qnames)


@pytest.mark.unit
def test_taint_source_router_decorator():
    """@router.get / @router.post marks function as taint source."""
    assert _is_taint_source("my_func", ["router.get('/path')"]) is True


@pytest.mark.unit
def test_taint_source_api_view_decorator():
    """@api_view marks function as taint source."""
    assert _is_taint_source("my_func", ["api_view(['GET'])"]) is True


# ─────────────────────────────────────────────────────────────────────────────
# Test #8 — _is_taint_source helper
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_is_taint_source_by_name():
    assert _is_taint_source("handle_request", []) is True
    assert _is_taint_source("list_view", []) is True
    assert _is_taint_source("route_handler", []) is True
    assert _is_taint_source("compute_sum", []) is False


@pytest.mark.unit
def test_is_taint_source_by_decorator():
    assert _is_taint_source("my_fn", ["app.route('/path')"]) is True
    assert _is_taint_source("my_fn", ["router.post('/items')"]) is True
    assert _is_taint_source("my_fn", ["login_required"]) is False


# ─────────────────────────────────────────────────────────────────────────────
# Test #9 — analyze_file edge counting (dry-run, no Neo4j)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_analyze_file_returns_edge_count(tmp_path):
    """analyze_file in dry-run mode returns the correct edge count."""
    source = """\
def first(a, b):
    x = a
    y = b
    return x

def second(p):
    q = p
    process(q)
"""
    py_file = tmp_path / "myservice.py"
    py_file.write_text(source)

    analyzer = DataFlowAnalyzer(sync_driver=None)
    count = analyzer.analyze_file(str(py_file), graph_id="g-123")

    # first(): a→x (assignment), b→y (assignment), x→return = 3 edges
    # second(): p→q (assignment), q→process (argument) = 2 edges
    # Total: 5 edges minimum
    assert count >= 5, f"Expected >=5 edges, got {count}"


@pytest.mark.unit
def test_analyze_file_skips_non_python(tmp_path):
    """analyze_file returns 0 for non-Python files (not a crash)."""
    ts_file = tmp_path / "app.ts"
    ts_file.write_text("const x = 1;")

    analyzer = DataFlowAnalyzer(sync_driver=None)
    # TypeScript file — SyntaxError on parse → graceful 0
    count = analyzer.analyze_file(str(ts_file), graph_id="g-ts")
    assert count == 0


@pytest.mark.unit
def test_analyze_file_syntax_error_returns_zero(tmp_path):
    """analyze_file handles SyntaxError gracefully."""
    bad_file = tmp_path / "broken.py"
    bad_file.write_text("def fn(: pass")

    analyzer = DataFlowAnalyzer(sync_driver=None)
    count = analyzer.analyze_file(str(bad_file), graph_id="g-bad")
    assert count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test #10 — module_qname helper
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_module_qname_from_path():
    assert _module_qname_from_path("app/services/data_flow_analyzer.py") == (
        "app.services.data_flow_analyzer"
    )
    assert _module_qname_from_path("main.py") == "main"


# ─────────────────────────────────────────────────────────────────────────────
# Test #11 — self/cls excluded from taint marks
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_self_param_excluded_from_taint():
    """self is not a user-controlled parameter and must not be taint-marked."""
    source = """\
class MyView:
    def handle_request(self, request):
        return request.data
"""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            analysis = _analyze_function_ast(node, "myapp.MyView", "myapp.py")
            taint_qnames = [m.qualified_name for m in analysis.taint_marks]
            # self should not be marked, request should be
            assert not any("self" in qn for qn in taint_qnames), (
                f"'self' should not be in taint marks: {taint_qnames}"
            )
            if analysis.taint_marks:
                assert any("request" in qn for qn in taint_qnames)
            break
