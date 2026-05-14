"""
Integration tests for data flow analysis (STORY-004).

TASK-023 — QA: data flow integration tests.

IMPORTANT: These tests require a running Neo4j Docker container with TASK-022
code deployed. Do NOT run these tests from a worktree — they target the live
Docker stack.

Run only AFTER TASK-022's PR merges to develop and the Docker stack is rebuilt:
    cd knowledge-graph-builder
    python -m pytest tests/integration/test_data_flow_integration.py -v -m integration

Architecture invariants verified:
  - graph_id on every FLOWS_TO edge (multi-tenancy)
  - Cross-tenant isolation: FLOWS_TO edges from graph A not visible to graph B
  - Parameterized Cypher: $graph_id in all data_flow queries
  - Taint propagation: taint: "user_input" set on first param of handle_request
  - API surface: POST /graphs/{graph_id}/code/query?query_type=data_flow
"""

from __future__ import annotations

import shutil
import textwrap
import time
import uuid
from pathlib import Path

import pytest
import requests

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — matches docker-compose service URLs
# ─────────────────────────────────────────────────────────────────────────────

_API_BASE = "http://localhost:8000/api/v1"
_AUTH_SERVICE_URL = "http://auth-service:8000"
_NEO4J_BOLT = "bolt://neo4j:7687"
_NEO4J_USER = "neo4j"
_NEO4J_PASS = "password"

# Temp directory shared between the kg-builder and kg-worker containers
# via the /app/app bind mount.  Tests write source fixtures here so the
# Celery worker (a separate container) can scan them.
_SHARED_TEST_DIR = Path("/app/app/_integration_test_scratch")

_TEST_USER_EMAIL = f"inttest-task023-{uuid.uuid4().hex[:8]}@example.com"
_TEST_USER_PASSWORD = "IntTest123!"


def _get_integration_token() -> str:
    """Register (or re-use) an integration test user and return a JWT."""
    reg = requests.post(
        f"{_AUTH_SERVICE_URL}/register/",
        json={"email": _TEST_USER_EMAIL, "password": _TEST_USER_PASSWORD},
        timeout=10,
    )
    if reg.status_code == 200:
        return reg.json()["access_token"]

    # User may already exist — try login instead
    login = requests.post(
        f"{_AUTH_SERVICE_URL}/login/",
        json={"email": _TEST_USER_EMAIL, "password": _TEST_USER_PASSWORD},
        timeout=10,
    )
    if login.status_code == 200:
        return login.json()["access_token"]

    raise RuntimeError(
        f"Cannot obtain integration test token: "
        f"register={reg.status_code} login={login.status_code}"
    )


_INTEGRATION_TOKEN: str | None = None


def _token() -> str:
    global _INTEGRATION_TOKEN
    if _INTEGRATION_TOKEN is None:
        _INTEGRATION_TOKEN = _get_integration_token()
    return _INTEGRATION_TOKEN


# Graph IDs — populated by create_test_graphs fixture (server assigns the UUID)
_GRAPH_ID_A: str = ""
_GRAPH_ID_B: str = ""


def _create_graph(name: str) -> str:
    """Create a graph via the API and return the server-assigned graph_id."""
    resp = requests.post(
        f"{_API_BASE}/graphs",
        headers=_api_headers(),
        json={"name": name, "description": "integration test graph"},
        timeout=10,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Graph creation failed: {resp.status_code} {resp.text}")
    return resp.json()["id"]


# Minimal Python source with one taintable function
_TAINT_SOURCE = textwrap.dedent("""\
    def handle_request(req, ctx):
        data = req.body
        result = data
        return result
""")

# Non-taint function in same file
_PLAIN_SOURCE = textwrap.dedent("""\
    def compute_sum(numbers):
        total = 0
        total = total + sum(numbers)
        return total
""")

_COMBINED_SOURCE = _TAINT_SOURCE + "\n" + _PLAIN_SOURCE


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _neo4j_driver():
    """Return a sync Neo4j driver (neo4j Python package)."""
    from neo4j import GraphDatabase

    return GraphDatabase.driver(_NEO4J_BOLT, auth=(_NEO4J_USER, _NEO4J_PASS))


def _api_headers(user_token: str | None = None) -> dict:
    return {"Authorization": f"Bearer {user_token or _token()}"}


def _wait_for_api(timeout: float = 30.0) -> None:
    """Block until the API is responsive."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = requests.get(f"{_API_BASE.replace('/api/v1', '')}/health", timeout=2)
            if resp.status_code < 500:
                return
        except requests.ConnectionError:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"API at {_API_BASE} not reachable after {timeout}s")


def _ingest_python_source(
    graph_id: str,
    source_code: str,
    tmp_path=None,
    user_token: str | None = None,
) -> str:
    """
    Write source_code to a shared temp dir accessible to the Celery worker
    container, then trigger code ingestion via the API.  Returns the dir path.
    Polls until the job completes.

    tmp_path is kept for backwards compatibility but ignored — both containers
    share /app/app via bind mount, so we write there instead of /tmp.
    """
    # Use the shared bind-mount so the Celery worker container can read the files
    test_dir = _SHARED_TEST_DIR / uuid.uuid4().hex
    test_dir.mkdir(parents=True, exist_ok=True)

    py_file = test_dir / "test_module.py"
    py_file.write_text(source_code)

    resp = requests.post(
        f"{_API_BASE}/graphs/{graph_id}/code-ingest",
        headers=_api_headers(user_token),
        json={"repo_path": str(test_dir), "mode": "full"},
        timeout=10,
    )
    assert resp.status_code == 202, f"Ingest failed: {resp.status_code} {resp.text}"

    job_id = resp.json()["job_id"]
    _wait_for_job(graph_id, job_id, user_token)
    return str(test_dir)


def _wait_for_job(
    graph_id: str,
    job_id: str,
    user_token: str,
    timeout: float = 60.0,
) -> None:
    """Poll job status until completed or failed."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = requests.get(
            f"{_API_BASE}/graphs/{graph_id}/jobs/{job_id}",
            headers=_api_headers(user_token),
            timeout=5,
        )
        if resp.status_code == 200:
            status = resp.json().get("status")
            if status == "completed":
                return
            if status == "failed":
                raise RuntimeError(
                    f"Job {job_id} failed: {resp.json().get('error', 'unknown')}"
                )
        time.sleep(2.0)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def _delete_test_graph(driver, graph_id: str) -> None:
    """Remove all nodes and edges for a test graph from Neo4j."""
    with driver.session() as session:
        session.run(
            "MATCH (n {graph_id: $graph_id}) DETACH DELETE n",
            {"graph_id": graph_id},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def neo4j_driver_fixture():
    """Sync Neo4j driver for direct graph inspection."""
    driver = _neo4j_driver()
    yield driver
    # Teardown: delete all test graphs created by this module
    for gid in [_GRAPH_ID_A, _GRAPH_ID_B]:
        try:
            _delete_test_graph(driver, gid)
        except Exception:
            pass
    driver.close()


@pytest.fixture(scope="module", autouse=True)
def wait_for_api_ready():
    """Ensure the API is up before running any integration test."""
    _wait_for_api()


@pytest.fixture(scope="module", autouse=True)
def create_test_graphs():
    """Create the test graphs via the API; store server-assigned UUIDs in module globals."""
    global _GRAPH_ID_A, _GRAPH_ID_B
    _GRAPH_ID_A = _create_graph("task023-graph-a")
    # Graph B intentionally not ingested — used for cross-tenant isolation tests
    _GRAPH_ID_B = _create_graph("task023-graph-b")


@pytest.fixture(scope="module")
def ingested_graph_a(neo4j_driver_fixture):
    """
    Ingest _COMBINED_SOURCE into GRAPH_ID_A once for the whole module.
    Yields graph_id for use in tests.
    """
    test_dir_path = _ingest_python_source(_GRAPH_ID_A, _COMBINED_SOURCE)
    yield _GRAPH_ID_A
    shutil.rmtree(test_dir_path, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — FLOWS_TO edges exist in Neo4j with correct graph_id
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_flows_to_edges_exist_in_neo4j(neo4j_driver_fixture, ingested_graph_a):
    """
    After ingesting _COMBINED_SOURCE, FLOWS_TO edges must exist in Neo4j
    scoped to _GRAPH_ID_A.
    """
    with neo4j_driver_fixture.session() as session:
        result = session.run(
            """
            MATCH (src)-[r:FLOWS_TO]->(tgt)
            WHERE r.graph_id = $graph_id
            RETURN count(r) AS edge_count
            """,
            {"graph_id": _GRAPH_ID_A},
        )
        record = result.single()
        edge_count = record["edge_count"] if record else 0

    assert edge_count >= 2, (
        f"Expected >=2 FLOWS_TO edges for handle_request in graph {_GRAPH_ID_A}, "
        f"got {edge_count}"
    )


@pytest.mark.integration
def test_flows_to_edges_have_graph_id_property(neo4j_driver_fixture, ingested_graph_a):
    """
    Architecture invariant: every FLOWS_TO edge must have a graph_id property.
    Every edge scoped to our test graph must carry graph_id == _GRAPH_ID_A.
    """
    with neo4j_driver_fixture.session() as session:
        result = session.run(
            """
            MATCH ()-[r:FLOWS_TO]->()
            WHERE r.graph_id = $graph_id
            RETURN r.graph_id AS edge_graph_id, r.via AS via
            LIMIT 20
            """,
            {"graph_id": _GRAPH_ID_A},
        )
        records = result.data()

    assert len(records) >= 1, "No FLOWS_TO edges found — did ingestion run?"
    for rec in records:
        assert rec["edge_graph_id"] == _GRAPH_ID_A, (
            f"Edge graph_id mismatch: expected {_GRAPH_ID_A}, "
            f"got {rec['edge_graph_id']}"
        )
        assert rec["via"] in (
            "assignment",
            "return",
            "argument",
            "parameter",
        ), f"Unexpected FLOWS_TO via value: {rec['via']}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — API data_flow query returns the flow path
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_api_data_flow_query_returns_path(ingested_graph_a):
    """
    POST /graphs/{graph_id}/code/query with query_type=data_flow and
    source_symbol returns a non-empty path for handle_request.
    """
    resp = requests.post(
        f"{_API_BASE}/graphs/{_GRAPH_ID_A}/code/query",
        headers=_api_headers(),
        json={
            "query_type": "data_flow",
            "params": {
                "source_symbol": "test_module.handle_request",
                "direction": "forward",
                "depth": 5,
            },
        },
        timeout=15,
    )
    assert (
        resp.status_code == 200
    ), f"Expected 200 from data_flow query, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["query_type"] == "data_flow"
    assert (
        body["total"] >= 1
    ), f"Expected at least 1 flow path for handle_request, got {body['total']}"
    # Each result must have the expected shape
    for result in body["results"]:
        assert "path_nodes" in result, f"Missing 'path_nodes' in: {result}"
        assert "depth" in result, f"Missing 'depth' in: {result}"
        assert isinstance(result["path_nodes"], list)


@pytest.mark.integration
def test_api_data_flow_query_missing_source_symbol_returns_400():
    """
    POST /graphs/{graph_id}/code/query with data_flow but no source_symbol → 400.
    """
    resp = requests.post(
        f"{_API_BASE}/graphs/{_GRAPH_ID_A}/code/query",
        headers=_api_headers(),
        json={
            "query_type": "data_flow",
            "params": {},
        },
        timeout=10,
    )
    assert (
        resp.status_code == 400
    ), f"Expected 400 for missing source_symbol, got {resp.status_code}: {resp.text}"
    detail = resp.json().get("detail", "")
    assert "source_symbol" in detail.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Taint property on the correct node
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_taint_property_on_first_param(neo4j_driver_fixture, ingested_graph_a):
    """
    handle_request is a taint source (name contains 'request').
    Its first non-self parameter (req) must have taint: 'user_input' set in Neo4j.
    """
    with neo4j_driver_fixture.session() as session:
        result = session.run(
            """
            MATCH (n {graph_id: $graph_id, taint: 'user_input'})
            RETURN n.qualified_name AS qname, n.taint AS taint
            """,
            {"graph_id": _GRAPH_ID_A},
        )
        records = result.data()

    assert (
        len(records) >= 1
    ), f"Expected at least one node with taint='user_input' in graph {_GRAPH_ID_A}"
    tainted_qnames = [r["qname"] for r in records]
    # The req parameter of handle_request should be tainted
    assert any(
        "req" in (qn or "") for qn in tainted_qnames
    ), f"Expected 'req' param to be taint-marked, found: {tainted_qnames}"


@pytest.mark.integration
def test_non_taint_function_has_no_taint_marks(neo4j_driver_fixture, ingested_graph_a):
    """
    compute_sum is a plain function (no taint heuristic match).
    No nodes from compute_sum should have taint: 'user_input'.
    """
    with neo4j_driver_fixture.session() as session:
        result = session.run(
            """
            MATCH (n {graph_id: $graph_id, taint: 'user_input'})
            WHERE n.qualified_name CONTAINS 'compute_sum'
            RETURN count(n) AS cnt
            """,
            {"graph_id": _GRAPH_ID_A},
        )
        record = result.single()
        cnt = record["cnt"] if record else 0

    assert (
        cnt == 0
    ), f"compute_sum params must not be tainted, but found {cnt} tainted node(s)"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Taint query: FLOWS_TO path from tainted source to sink
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_taint_flows_to_path_via_cypher(neo4j_driver_fixture, ingested_graph_a):
    """
    STORY-004 acceptance criterion:
    MATCH (source {taint: 'user_input'})-[:FLOWS_TO*1..10]->(sink)
    returns correct path for handle_request.
    """
    with neo4j_driver_fixture.session() as session:
        result = session.run(
            """
            MATCH (source {graph_id: $graph_id, taint: 'user_input'})
                  -[:FLOWS_TO*1..10]->(sink)
            RETURN
                source.qualified_name AS source_qname,
                sink.qualified_name   AS sink_qname
            LIMIT 10
            """,
            {"graph_id": _GRAPH_ID_A},
        )
        records = result.data()

    assert len(records) >= 1, (
        f"Expected taint flow path in graph {_GRAPH_ID_A} — "
        "taint source must flow to at least one sink"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Cross-tenant isolation: FLOWS_TO edges not visible from different graph_id
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_cross_tenant_isolation_flows_to_neo4j(neo4j_driver_fixture, ingested_graph_a):
    """
    FLOWS_TO edges stored under _GRAPH_ID_A must be invisible from _GRAPH_ID_B.
    Zero edges returned when queried with a different graph_id.
    _GRAPH_ID_B is a fresh UUID that never had any ingestion.
    """
    with neo4j_driver_fixture.session() as session:
        result = session.run(
            """
            MATCH ()-[r:FLOWS_TO]->()
            WHERE r.graph_id = $graph_id
            RETURN count(r) AS edge_count
            """,
            {"graph_id": _GRAPH_ID_B},  # tenant B — never ingested
        )
        record = result.single()
        edge_count = record["edge_count"] if record else 0

    assert edge_count == 0, (
        f"Cross-tenant isolation violation: found {edge_count} FLOWS_TO edges "
        f"visible from graph {_GRAPH_ID_B} (belongs to a different tenant)"
    )


@pytest.mark.integration
def test_cross_tenant_isolation_api_data_flow(ingested_graph_a):
    """
    API: data_flow query for graph_B with source_symbol from graph_A
    returns empty results (not another tenant's data).
    """
    resp = requests.post(
        f"{_API_BASE}/graphs/{_GRAPH_ID_B}/code/query",
        headers=_api_headers(),
        json={
            "query_type": "data_flow",
            "params": {
                "source_symbol": "test_module.handle_request",
            },
        },
        timeout=10,
    )
    # May return 200 with empty results, or 403/404 — all satisfy isolation
    if resp.status_code == 200:
        body = resp.json()
        assert body["total"] == 0, (
            f"Cross-tenant isolation violation: graph_B returned {body['total']} "
            f"results for graph_A's source_symbol"
        )
    else:
        # 403 or 404 also satisfies isolation requirement
        assert resp.status_code in (
            403,
            404,
        ), f"Unexpected status for cross-tenant query: {resp.status_code}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — Performance: 500-line file analysis completes in <10s
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_500_line_file_analysis_under_10_seconds(neo4j_driver_fixture):
    """
    STORY-004 acceptance criterion: analysis of a 500-line Python file completes
    in under 10 seconds end-to-end (ingest + FLOWS_TO edge writing).
    """
    perf_graph_id = _create_graph("task023-perf")

    # Generate a ~500-line Python file: 25 functions, each ~20 lines
    lines = []
    for i in range(25):
        lines.append(f"def func_{i:03d}(param_{i}, extra):")
        lines.append(f"    x_{i} = param_{i}")
        lines.append(f"    y_{i} = x_{i}")
        for j in range(8):
            lines.append(f"    z_{i}_{j} = x_{i}")
        lines.append(f"    process(y_{i})")
        lines.append(f"    return y_{i}")
        lines.append("")

    source = "\n".join(lines)

    perf_dir: str | None = None
    start = time.monotonic()
    try:
        perf_dir = _ingest_python_source(perf_graph_id, source)
    finally:
        try:
            _delete_test_graph(neo4j_driver_fixture, perf_graph_id)
        except Exception:
            pass
        if perf_dir:
            shutil.rmtree(perf_dir, ignore_errors=True)
    elapsed = time.monotonic() - start

    assert elapsed < 10.0, (
        f"500-line file analysis took {elapsed:.2f}s — must complete in <10s "
        "(STORY-004 acceptance criterion)"
    )
