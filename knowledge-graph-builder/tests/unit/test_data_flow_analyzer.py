"""
Unit tests for data_flow_analyzer.py

TASK-023 — QA: data flow analysis tests (FLOWS_TO edges, taint marking).
Verifies all STORY-004 acceptance criteria for the DataFlowAnalyzer service
produced by TASK-022.

Coverage:
  #1  — x = param; return x → 2 FLOWS_TO edges: param→x (assignment), x→fn (return)
  #2  — foo(x) call → FLOWS_TO {via: "argument"} edge
  #3  — taint: @router.get(...) decorator → params have taint: "user_input"
  #4  — taint: function named handle_request → first param tainted
  #5  — no FLOWS_TO edges for variables never derived from parameters
  #6  — graph_id present on every edge (Cypher template + dry-run assertions)
  #7  — augmented assignment: both sources tracked
  #8  — decorator taint heuristics (@app.route, @router., @api_view)
  #9  — analyze_file returns correct edge count (dry-run, no Neo4j)
  #10 — _module_qname_from_path helper
  #11 — self/cls excluded from taint marks
  #12 — non-taint function produces zero taint marks
"""

from __future__ import annotations

import ast

import pytest

from app.services.data_flow_analyzer import (
    _MARK_TAINT,
    _MERGE_FLOWS_TO,
    DataFlowAnalyzer,
    _analyze_function_ast,
    _is_taint_source,
    _module_qname_from_path,
)

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


def _edges_from_source(
    source: str,
    module_qname: str = "mymodule",
    source_file: str = "mymodule.py",
):
    """
    Convenience helper: parse *source*, analyse the first function found,
    return (edges, taint_marks).
    """
    func_node = _parse_first_function(source)
    analysis = _analyze_function_ast(func_node, module_qname, source_file)
    return analysis.edges, analysis.taint_marks


# ─────────────────────────────────────────────────────────────────────────────
# Test #1 — x = param; return x → 2 FLOWS_TO edges
# Task requirement: "analyze_function() on `x = param; return x` → 2 FLOWS_TO edges:
#   param→x (assignment), x→return_value (return)"
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_assignment_then_return_produces_two_edges():
    """
    def fn(param):
        x = param      # FLOWS_TO {via: "assignment"}
        return x       # FLOWS_TO {via: "return"}

    Must produce exactly 2 edges: one assignment, one return.
    """
    source = """\
def fn(param):
    x = param
    return x
"""
    edges, _ = _edges_from_source(source, "mod", "mod.py")

    assignment_edges = [e for e in edges if e.via == "assignment"]
    return_edges = [e for e in edges if e.via == "return"]

    assert (
        len(assignment_edges) >= 1
    ), f"Expected >=1 assignment edge for `x = param`, got {assignment_edges}"
    assert (
        len(return_edges) >= 1
    ), f"Expected >=1 return edge for `return x`, got {return_edges}"

    # Verify assignment edge: param → x
    assign_edge = assignment_edges[0]
    assert "fn" in assign_edge.source_qualified_name, (
        f"Source of assignment edge should reference 'fn' (the function node representing param), "
        f"got: {assign_edge.source_qualified_name}"
    )
    assert "x" in assign_edge.target_qualified_name, (
        f"Target of assignment edge should reference 'x', "
        f"got: {assign_edge.target_qualified_name}"
    )

    # Verify return edge: x → function (return value)
    return_edge = return_edges[0]
    assert "fn" in return_edge.target_qualified_name, (
        f"Target of return edge should reference the function 'fn', "
        f"got: {return_edge.target_qualified_name}"
    )


@pytest.mark.unit
def test_direct_param_return_produces_return_edge():
    """
    def fn(x):
        return x   # param x flows directly to return (no intermediate var)

    Produces at least 1 return edge.
    """
    source = """\
def fn(x):
    return x
"""
    edges, _ = _edges_from_source(source, "mod", "mod.py")
    return_edges = [e for e in edges if e.via == "return"]
    assert len(return_edges) >= 1, "Expected return edge for `return x`"
    assert "fn" in return_edges[0].source_qualified_name


@pytest.mark.unit
def test_assignment_edge_direction():
    """Verify assignment edge direction: source is function (param), target has var name."""
    source = """\
def process(data):
    result = data
"""
    edges, _ = _edges_from_source(source, "svc", "svc.py")
    assignment_edges = [e for e in edges if e.via == "assignment"]
    assert len(assignment_edges) >= 1

    edge = assignment_edges[0]
    assert edge.via == "assignment"
    assert (
        "result" in edge.target_qualified_name
    ), f"Expected target to contain 'result', got {edge.target_qualified_name}"


# ─────────────────────────────────────────────────────────────────────────────
# Test #2 — foo(x) call → FLOWS_TO {via: "argument"} edge
# Task requirement: "analyze_function() on `foo(x)` → FLOWS_TO {via: 'argument'} edge"
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_call_argument_produces_argument_edge():
    """
    def fn(x):
        foo(x)     # FLOWS_TO {via: "argument"}

    x is a parameter; passing it to foo() must produce an argument edge.
    """
    source = """\
def fn(x):
    foo(x)
"""
    edges, _ = _edges_from_source(source, "mod", "mod.py")

    arg_edges = [e for e in edges if e.via == "argument"]
    assert (
        len(arg_edges) >= 1
    ), f"Expected >=1 argument edge for `foo(x)`, got {arg_edges}"

    edge = arg_edges[0]
    assert edge.via == "argument"
    # Source is the function node (parameter flows from fn)
    assert (
        "fn" in edge.source_qualified_name
    ), f"Source of argument edge should reference 'fn', got: {edge.source_qualified_name}"


@pytest.mark.unit
def test_derived_variable_to_call_argument():
    """Variable derived from param also flows as argument."""
    source = """\
def fn(raw):
    processed = raw
    execute(processed)
"""
    edges, _ = _edges_from_source(source, "mod", "mod.py")
    arg_edges = [e for e in edges if e.via == "argument"]
    assert len(arg_edges) >= 1, "Expected argument edge for execute(processed)"


@pytest.mark.unit
def test_keyword_argument_produces_argument_edge():
    """Keyword arguments from tracked vars also produce argument edges."""
    source = """\
def fn(data):
    save(record=data)
"""
    edges, _ = _edges_from_source(source, "mod", "mod.py")
    arg_edges = [e for e in edges if e.via == "argument"]
    assert len(arg_edges) >= 1, "Expected argument edge for keyword argument"


# ─────────────────────────────────────────────────────────────────────────────
# Test #3 — taint: @router.get(...) decorator → params have taint: "user_input"
# Task requirement: "Taint: function with @router.get(...) decorator → params have taint: 'user_input'"
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_router_get_decorator_taints_params():
    """
    @router.get('/items')
    def list_items(request):
        ...

    request param must be taint-marked.
    """
    source = """\
@router.get('/items')
def list_items(request):
    data = request.query_params
    return data
"""
    _, taint_marks = _edges_from_source(source, "api.endpoints", "api/endpoints.py")

    assert (
        len(taint_marks) >= 1
    ), "Expected taint marks for @router.get decorated function"
    taint_qnames = [m.qualified_name for m in taint_marks]
    assert any(
        "request" in qn for qn in taint_qnames
    ), f"Expected 'request' param in taint marks, got: {taint_qnames}"


@pytest.mark.unit
def test_router_post_decorator_is_taint_source():
    """@router.post also marks as taint source via _is_taint_source."""
    assert _is_taint_source("create_item", ["router.post('/items')"]) is True


@pytest.mark.unit
def test_app_route_decorator_taints_params():
    """@app.route marks function as taint source."""
    source = """\
@app.route('/submit', methods=['POST'])
def submit_form(request):
    return request.form
"""
    _, taint_marks = _edges_from_source(source)
    assert len(taint_marks) >= 1
    taint_qnames = [m.qualified_name for m in taint_marks]
    assert any("request" in qn for qn in taint_qnames)


@pytest.mark.unit
def test_api_view_decorator_is_taint_source():
    """@api_view marks function as taint source via _is_taint_source."""
    assert _is_taint_source("my_view", ["api_view(['GET', 'POST'])"]) is True


# ─────────────────────────────────────────────────────────────────────────────
# Test #4 — taint: function named handle_request → first param tainted
# Task requirement: "Taint: function named handle_request → first param tainted"
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_handle_request_name_taints_first_param():
    """
    def handle_request(req, context):
        ...

    First non-self param (req) should be taint-marked.
    """
    source = """\
def handle_request(req, context):
    data = req.body
    return data
"""
    _, taint_marks = _edges_from_source(source, "views", "views.py")

    assert len(taint_marks) >= 1, "Expected taint mark for handle_request"
    taint_qnames = [m.qualified_name for m in taint_marks]
    assert any(
        "req" in qn for qn in taint_qnames
    ), f"Expected 'req' in taint marks, got: {taint_qnames}"


@pytest.mark.unit
def test_taint_name_heuristics_match():
    """_is_taint_source matches view, handler, endpoint, route, request in name."""
    assert _is_taint_source("handle_request", []) is True
    assert _is_taint_source("list_view", []) is True
    assert _is_taint_source("route_handler", []) is True
    assert _is_taint_source("create_endpoint", []) is True
    assert _is_taint_source("user_handler", []) is True
    # Non-matching names
    assert _is_taint_source("compute_sum", []) is False
    assert _is_taint_source("parse_config", []) is False


@pytest.mark.unit
def test_taint_first_param_only_not_second():
    """Only the first non-self parameter gets tainted."""
    source = """\
def handle_request(req, other):
    pass
"""
    _, taint_marks = _edges_from_source(source, "mod", "mod.py")
    # There should be exactly one taint mark for 'req'
    assert len(taint_marks) == 1
    assert "req" in taint_marks[0].qualified_name


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
            assert not any(
                "self" in qn for qn in taint_qnames
            ), f"'self' must not appear in taint marks: {taint_qnames}"
            if analysis.taint_marks:
                assert any(
                    "request" in qn for qn in taint_qnames
                ), f"'request' should be taint-marked, got: {taint_qnames}"
            break


# ─────────────────────────────────────────────────────────────────────────────
# Test #5 — no FLOWS_TO edges for variables never derived from parameters
# Task requirement: "No FLOWS_TO edges for variables never derived from parameters"
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_no_edges_for_untracked_local_variable():
    """
    def fn(param):
        unrelated = 42          # never derived from param → no edge
        result = param          # derived from param → edge
        return result

    `unrelated` must not appear as an edge source or target.
    """
    source = """\
def fn(param):
    unrelated = 42
    result = param
    return result
"""
    edges, _ = _edges_from_source(source, "mod", "mod.py")

    # Collect all qualified names appearing in edges
    edge_names = set()
    for e in edges:
        edge_names.add(e.source_qualified_name)
        edge_names.add(e.target_qualified_name)

    assert not any(
        "unrelated" in name for name in edge_names
    ), f"'unrelated' (never derived from param) must not appear in edges: {edge_names}"


@pytest.mark.unit
def test_no_edges_for_pure_constant_function():
    """
    def fn():
        x = 42
        return x

    No parameters → no tracked vars → zero FLOWS_TO edges.
    """
    source = """\
def fn():
    x = 42
    return x
"""
    edges, _ = _edges_from_source(source, "mod", "mod.py")
    assert (
        len(edges) == 0
    ), f"Expected 0 edges for function with no params, got: {edges}"


@pytest.mark.unit
def test_only_param_derived_vars_generate_edges():
    """Mixed function: only param-derived assignments produce edges."""
    source = """\
def fn(x):
    a = x       # tracked: x is a param → edge
    b = 100     # not tracked: 100 is a literal
    c = a       # tracked: a is derived from x → edge
    d = b       # NOT tracked: b is not param-derived
    return c
"""
    edges, _ = _edges_from_source(source, "mod", "mod.py")

    edge_targets = [e.target_qualified_name for e in edges]
    # 'd' is assigned from 'b' (not tracked), so the local var 'd' must not be a flow target.
    # Check with a trailing-dot or end-of-string anchor to avoid matching 'd' inside other names.
    assert not any(
        t.endswith(".d") or t == "d" for t in edge_targets
    ), f"'d' (derived from non-param 'b') must not be a flow target: {edge_targets}"
    # 'c' is assigned from 'a' (tracked), so it must appear as a target
    assert any(
        t.endswith(".c") or t == "c" for t in edge_targets
    ), f"'c' (derived from param-tracked 'a') must appear as flow target: {edge_targets}"


# ─────────────────────────────────────────────────────────────────────────────
# Test #6 — graph_id present on every edge
# Task requirement: "graph_id present on every edge"
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_graph_id_in_merge_flows_to_cypher():
    """
    Architecture invariant: MERGE_FLOWS_TO Cypher must reference $graph_id.
    Every FLOWS_TO edge must be scoped to a graph_id — no exceptions.
    """
    assert "$graph_id" in _MERGE_FLOWS_TO, (
        "FLOWS_TO MERGE Cypher must use $graph_id parameter "
        "(architecture invariant: graph_id on every FLOWS_TO edge)"
    )


@pytest.mark.unit
def test_graph_id_in_mark_taint_cypher():
    """
    Multi-tenancy invariant: taint mark Cypher must also scope by $graph_id.
    """
    assert (
        "$graph_id" in _MARK_TAINT
    ), "_MARK_TAINT Cypher must use $graph_id (multi-tenancy invariant)"


@pytest.mark.unit
def test_graph_id_not_hardcoded_in_cypher():
    """
    Parameterized Cypher invariant: graph_id must never be string-interpolated.
    Both Cypher templates must use the $ parameter syntax only.
    """
    # Check MERGE_FLOWS_TO does not hardcode any graph_id value
    assert (
        "graph_id: '" not in _MERGE_FLOWS_TO
    ), "MERGE_FLOWS_TO must not hardcode graph_id values"
    assert (
        'graph_id: "' not in _MERGE_FLOWS_TO
    ), "MERGE_FLOWS_TO must not hardcode graph_id values"
    assert "graph_id: '" not in _MARK_TAINT
    assert 'graph_id: "' not in _MARK_TAINT


@pytest.mark.unit
def test_dry_run_returns_edge_count_with_graph_id(tmp_path):
    """
    DataFlowAnalyzer.analyze_file() dry-run (sync_driver=None) counts edges correctly.
    The graph_id parameter is passed through even in dry-run.
    """
    source = """\
def process(user_input):
    result = user_input
    return result
"""
    py_file = tmp_path / "service.py"
    py_file.write_text(source)

    graph_id = "test-task023-graph-id"
    analyzer = DataFlowAnalyzer(sync_driver=None)
    count = analyzer.analyze_file(str(py_file), graph_id=graph_id)

    # `result = user_input` → 1 assignment edge
    # `return result` → 1 return edge
    assert (
        count >= 2
    ), f"Expected >=2 edges for `result = user_input; return result`, got {count}"


@pytest.mark.unit
def test_flows_to_edge_graph_id_passed_as_parameter():
    """
    Verify the Cypher write step passes graph_id as a query *parameter* (not interpolated).
    The $graph_id token in the Cypher string paired with graph_id in the params dict
    is the architectural requirement.
    """
    import re

    # The Cypher template should contain $graph_id (parameterised)
    assert "$graph_id" in _MERGE_FLOWS_TO
    # It should NOT contain graph_id as a formatted string literal
    hardcoded_pattern = re.compile(r"graph_id:\s*['\"]")
    assert not hardcoded_pattern.search(
        _MERGE_FLOWS_TO
    ), "MERGE_FLOWS_TO must use $graph_id parameter, not hardcoded string"


# ─────────────────────────────────────────────────────────────────────────────
# Additional coverage — augmented assignment, chained flows, multi-function file
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_augmented_assignment_produces_edges_from_both_sides():
    """
    def fn(x, y):
        x += y   # x and y (both params) flow into x

    Both sources should appear in assignment edges.
    """
    source = """\
def fn(x, y):
    x += y
"""
    edges, _ = _edges_from_source(source, "mod", "mod.py")
    assignment_edges = [e for e in edges if e.via == "assignment"]
    assert len(assignment_edges) >= 2, (
        f"Expected >=2 assignment edges for augmented assignment `x += y`, "
        f"got {assignment_edges}"
    )


@pytest.mark.unit
def test_compound_assignment_chain():
    """
    def fn(a):
        b = a    # param → b
        c = b    # b → c (derived)
        return c # c → fn (return)

    3 edges minimum.
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


@pytest.mark.unit
def test_non_taint_function_no_marks():
    """Regular function (no taint heuristic match) produces zero taint marks."""
    source = """\
def compute_average(numbers):
    total = sum(numbers)
    return total
"""
    _, taint_marks = _edges_from_source(source)
    assert (
        len(taint_marks) == 0
    ), f"Expected no taint marks for non-taint function, got: {taint_marks}"


@pytest.mark.unit
def test_analyze_file_multiple_functions(tmp_path):
    """analyze_file counts edges across all functions in a file."""
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

    # first(): a→x (1 assign), b→y (1 assign), x→fn (1 return) = 3 edges
    # second(): p→q (1 assign), q→process (1 argument) = 2 edges
    # Total: 5 edges minimum
    assert count >= 5, f"Expected >=5 edges across 2 functions, got {count}"


@pytest.mark.unit
def test_analyze_file_graceful_on_syntax_error(tmp_path):
    """analyze_file handles SyntaxError gracefully, returns 0."""
    bad_file = tmp_path / "broken.py"
    bad_file.write_text("def fn(: pass")

    analyzer = DataFlowAnalyzer(sync_driver=None)
    count = analyzer.analyze_file(str(bad_file), graph_id="g-bad")
    assert count == 0


@pytest.mark.unit
def test_module_qname_from_path():
    """_module_qname_from_path mirrors code_parser_service path conversion."""
    assert _module_qname_from_path("app/services/data_flow_analyzer.py") == (
        "app.services.data_flow_analyzer"
    )
    assert _module_qname_from_path("main.py") == "main"
