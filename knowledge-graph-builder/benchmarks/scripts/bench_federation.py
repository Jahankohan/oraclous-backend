"""
Benchmark 4: Federation Overhead
==================================
Creates the same query across 1, 2, and 5 graphs (same content) and measures
response time, computing overhead per additional graph.

Note: This benchmark requires multiple pre-populated graphs. You can create them
by running bench_ingestion.py with different GRAPH_ID values.

Usage:
    ORACLOUS_API_KEY=<key> GRAPH_IDS=<id1>,<id2>,<id3>,<id4>,<id5> python bench_federation.py

Environment variables:
    ORACLOUS_API_KEY    — Bearer token for the knowledge-graph-builder service
    GRAPH_IDS           — Comma-separated list of at least 5 graph UUIDs
                          (all should be pre-populated with the same wikipedia_100 dataset)
    KGB_BASE_URL        — Base URL for knowledge-graph-builder (default: http://localhost:8003/api/v1)
    FEDERATION_ENDPOINT — Full URL for the federation endpoint
                          (default: http://localhost:8003/api/v1/federation/query)
    REPETITIONS         — Number of times to repeat each test (default: 10)

Outputs:
    results/federation.json  — response times for 1, 2, 5 graphs + overhead computation

Methodology:
    - Same 10 queries issued against 1-graph, 2-graph, and 5-graph setups
    - Each scenario repeated REPETITIONS times to get stable percentiles
    - Overhead per additional graph = (5-graph P95 - 1-graph P95) / 4
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KGB_BASE_URL = os.environ.get("KGB_BASE_URL", "http://localhost:8003/api/v1")
FEDERATION_ENDPOINT = os.environ.get(
    "FEDERATION_ENDPOINT", f"{KGB_BASE_URL}/federation/query"
)
ORACLOUS_API_KEY = os.environ.get("ORACLOUS_API_KEY", "")
_GRAPH_IDS_ENV = os.environ.get("GRAPH_IDS", "")
GRAPH_IDS: list[str] = [g.strip() for g in _GRAPH_IDS_ENV.split(",") if g.strip()]
REPETITIONS = int(os.environ.get("REPETITIONS", "10"))

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Hardcoded benchmark queries for reproducibility
BENCHMARK_QUERIES: list[str] = [
    "What is machine learning?",
    "How does a knowledge graph work?",
    "What is retrieval-augmented generation?",
    "Explain the transformer architecture.",
    "What is the relationship between embeddings and semantic search?",
    "What is Neo4j and what is it used for?",
    "How does vector similarity search work?",
    "What is role-based access control?",
    "What are the main types of neural networks?",
    "How does fine-tuning improve language models?",
]

# Scenarios: (label, number_of_graphs)
SCENARIOS = [(1, "1_graph"), (2, "2_graphs"), (5, "5_graphs")]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_headers() -> dict[str, str]:
    if not ORACLOUS_API_KEY:
        print(
            "ERROR: ORACLOUS_API_KEY environment variable is not set.", file=sys.stderr
        )
        sys.exit(1)
    return {
        "Authorization": f"Bearer {ORACLOUS_API_KEY}",
        "Content-Type": "application/json",
    }


def percentile(values: list[float], pct: float) -> float:
    """Compute the p-th percentile (0–100) of a list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct / 100
    lo = int(k)
    hi = lo + 1
    if hi >= len(sorted_vals):
        return sorted_vals[-1]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def federation_query(
    session: requests.Session,
    query: str,
    graph_ids: list[str],
) -> tuple[float, bool, str | None]:
    """
    POST a federated query and return (elapsed_s, success, error).

    The federation endpoint is expected to accept:
        POST /api/v1/federation/query
        {
          "query": "...",
          "graph_ids": ["id1", "id2", ...]
        }

    Falls back to sequential single-graph chat if the federation endpoint is
    not available (HTTP 404/405), so the benchmark still produces useful
    single-graph baseline data.
    """
    payload = {"query": query, "graph_ids": graph_ids}

    t_start = time.perf_counter()
    try:
        resp = session.post(
            FEDERATION_ENDPOINT,
            json=payload,
            headers=get_headers(),
            timeout=120,
        )
        elapsed = time.perf_counter() - t_start

        if resp.status_code in (404, 405):
            # Federation endpoint not yet deployed — fall back to sequential chat
            elapsed, success, error = _sequential_chat_fallback(
                session, query, graph_ids
            )
            return elapsed, success, "fallback:sequential"

        return elapsed, resp.ok, (resp.text[:200] if not resp.ok else None)

    except requests.RequestException as exc:
        elapsed = time.perf_counter() - t_start
        return elapsed, False, str(exc)


def _sequential_chat_fallback(
    session: requests.Session,
    query: str,
    graph_ids: list[str],
) -> tuple[float, bool, str | None]:
    """
    Fallback: run the same chat query against each graph sequentially and
    measure total elapsed time. Used when the federation endpoint is not available.
    """
    t_start = time.perf_counter()
    all_ok = True
    last_error: str | None = None

    for graph_id in graph_ids:
        url = f"{KGB_BASE_URL}/graphs/{graph_id}/chat"
        payload = {
            "query": query,
            "graph_id": graph_id,
            "return_context": False,
            "include_sources": False,
        }
        try:
            resp = session.post(url, json=payload, headers=get_headers(), timeout=60)
            if not resp.ok:
                all_ok = False
                last_error = resp.text[:200]
        except requests.RequestException as exc:
            all_ok = False
            last_error = str(exc)

    elapsed = time.perf_counter() - t_start
    return elapsed, all_ok, last_error


def run_scenario(
    session: requests.Session,
    n_graphs: int,
    label: str,
) -> dict:
    """Run all queries REPETITIONS times for a given number of graphs."""
    if len(GRAPH_IDS) < n_graphs:
        print(
            f"  SKIP {label}: need {n_graphs} graphs but only {len(GRAPH_IDS)} provided in GRAPH_IDS.",
            file=sys.stderr,
        )
        return {
            "n_graphs": n_graphs,
            "label": label,
            "skipped": True,
            "reason": f"Need {n_graphs} graph IDs, only {len(GRAPH_IDS)} provided",
        }

    graphs = GRAPH_IDS[:n_graphs]
    per_run: list[dict] = []
    elapsed_times: list[float] = []

    for rep in range(1, REPETITIONS + 1):
        for q_idx, query in enumerate(BENCHMARK_QUERIES, 1):
            elapsed, success, error = federation_query(session, query, graphs)
            run = {
                "rep": rep,
                "query_idx": q_idx,
                "query": query,
                "graph_ids": graphs,
                "elapsed_s": round(elapsed, 3),
                "success": success,
                "error": error,
            }
            per_run.append(run)
            if success:
                elapsed_times.append(elapsed)

            status_icon = "OK" if success else "FAIL"
            print(
                f"    rep={rep}/{REPETITIONS}  q={q_idx:2d}/{len(BENCHMARK_QUERIES)}  "
                f"[{status_icon}]  {elapsed:.2f}s  {query[:40]}"
            )

    successful_runs = [r for r in per_run if r["success"]]
    return {
        "n_graphs": n_graphs,
        "label": label,
        "graph_ids": graphs,
        "skipped": False,
        "total_runs": len(per_run),
        "successful": len(successful_runs),
        "failed": len(per_run) - len(successful_runs),
        "p50_s": round(percentile(elapsed_times, 50), 3) if elapsed_times else None,
        "p95_s": round(percentile(elapsed_times, 95), 3) if elapsed_times else None,
        "p99_s": round(percentile(elapsed_times, 99), 3) if elapsed_times else None,
        "mean_s": round(statistics.mean(elapsed_times), 3) if elapsed_times else None,
        "per_run": per_run,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Oraclous Benchmark 4: Federation Overhead")
    print("=" * 60)

    if not GRAPH_IDS:
        print(
            "ERROR: Set GRAPH_IDS to a comma-separated list of at least 5 graph UUIDs.\n"
            "Example: GRAPH_IDS=uuid1,uuid2,uuid3,uuid4,uuid5",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Graph IDs provided:  {len(GRAPH_IDS)}")
    print(f"Federation endpoint: {FEDERATION_ENDPOINT}")
    print(f"Repetitions:         {REPETITIONS}")
    print(f"Queries per rep:     {len(BENCHMARK_QUERIES)}")
    print()

    session = requests.Session()
    scenario_results: list[dict] = []

    for n_graphs, label in SCENARIOS:
        print(f"--- Scenario: {label} ({n_graphs} graph(s)) ---")
        result = run_scenario(session, n_graphs, label)
        scenario_results.append(result)
        if not result.get("skipped"):
            print(
                f"  P50={result['p50_s']}s  P95={result['p95_s']}s  "
                f"P99={result['p99_s']}s  mean={result['mean_s']}s"
            )
        print()

    # Compute overhead per additional graph
    overhead_per_graph: float | None = None
    p95_1 = next(
        (
            r.get("p95_s")
            for r in scenario_results
            if r["n_graphs"] == 1 and not r.get("skipped")
        ),
        None,
    )
    p95_5 = next(
        (
            r.get("p95_s")
            for r in scenario_results
            if r["n_graphs"] == 5 and not r.get("skipped")
        ),
        None,
    )
    if p95_1 is not None and p95_5 is not None:
        overhead_per_graph = round((p95_5 - p95_1) / 4, 3)

    output = {
        "benchmark": "federation_overhead",
        "graph_ids_available": GRAPH_IDS,
        "repetitions": REPETITIONS,
        "queries_per_rep": len(BENCHMARK_QUERIES),
        "overhead_per_additional_graph_s": overhead_per_graph,
        "scenarios": scenario_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "federation.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS — Federation Overhead")
    print("=" * 60)
    print(f"  {'Scenario':<15} {'P50':>8} {'P95':>8} {'P99':>8} {'mean':>8}")
    print(f"  {'-' * 15} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    for r in scenario_results:
        if r.get("skipped"):
            print(f"  {r['label']:<15} SKIPPED ({r.get('reason', '')})")
        else:
            print(
                f"  {r['label']:<15} {str(r['p50_s']) + 's':>8} {str(r['p95_s']) + 's':>8} "
                f"{str(r['p99_s']) + 's':>8} {str(r['mean_s']) + 's':>8}"
            )
    print()
    if overhead_per_graph is not None:
        print(f"  Overhead per additional graph (P95): {overhead_per_graph}s")
    else:
        print("  Overhead per additional graph: n/a (insufficient data)")

    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
