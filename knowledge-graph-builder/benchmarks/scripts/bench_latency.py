"""
Benchmark 3: Chat Latency
==========================
Runs 100 sequential chat queries against the ingested graph for each available
retriever type, measuring P50, P95, and P99 response times.

Usage:
    ORACLOUS_API_KEY=<key> GRAPH_ID=<id> python bench_latency.py

Environment variables:
    ORACLOUS_API_KEY  — Bearer token for the knowledge-graph-builder service
    GRAPH_ID          — UUID of the target graph (populated by bench_ingestion.py)
    KGB_BASE_URL      — Base URL for knowledge-graph-builder (default: http://localhost:8003/api/v1)
    RETRIEVER_TYPES   — Comma-separated list of retriever types to test
                        (default: vector,vector_cypher,hybrid,hybrid_cypher,text2cypher)

Outputs:
    results/latency.json  — per-query timings + aggregate P50/P95/P99 per retriever

Targets:
    P95 < 5s for vector retriever (PASS / FAIL printed at end)
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
GRAPH_ID = os.environ.get("GRAPH_ID", "")
ORACLOUS_API_KEY = os.environ.get("ORACLOUS_API_KEY", "")

_DEFAULT_RETRIEVERS = "vector,vector_cypher,hybrid,hybrid_cypher,text2cypher"
RETRIEVER_TYPES: list[str] = [
    r.strip()
    for r in os.environ.get("RETRIEVER_TYPES", _DEFAULT_RETRIEVERS).split(",")
    if r.strip()
]

TARGET_P95_VECTOR_S = 5.0  # seconds

RESULTS_DIR = Path(__file__).parent.parent / "results"

# ---------------------------------------------------------------------------
# 100 diverse benchmark questions — hardcoded for reproducibility
# Mix of factual, multi-hop, and summary questions
# ---------------------------------------------------------------------------

BENCHMARK_QUESTIONS: list[str] = [
    # Factual — single entity
    "What is Python programming language?",
    "What is machine learning?",
    "How does deep learning work?",
    "What is a knowledge graph?",
    "What is Neo4j used for?",
    "What is FastAPI?",
    "What is Docker and how does it help developers?",
    "What is Kubernetes?",
    "What are microservices?",
    "What is a REST API?",
    # Technical concepts
    "How does GraphQL differ from REST?",
    "What is PostgreSQL?",
    "What is Redis used for?",
    "How does Apache Kafka work?",
    "What is Celery used for in Python?",
    "What is TensorFlow?",
    "What is PyTorch?",
    "What is Hugging Face?",
    "What is BERT in natural language processing?",
    "What is GPT-4?",
    # Advanced concepts
    "What is retrieval-augmented generation?",
    "How do vector databases work?",
    "What are embeddings in machine learning?",
    "What is semantic search?",
    "What is fine-tuning in deep learning?",
    "What is reinforcement learning from human feedback?",
    "How does the attention mechanism work?",
    "What is an encoder-decoder architecture?",
    "What is named-entity recognition?",
    "What is information extraction?",
    # Graph and knowledge
    "What is knowledge representation?",
    "What is an ontology in information science?",
    "What is RDF?",
    "What is SPARQL?",
    "What is the Cypher query language?",
    "What is a property graph?",
    "What is entity resolution?",
    "What is coreference resolution?",
    "What is relation extraction?",
    "What is text mining?",
    # ML techniques
    "What is sentiment analysis?",
    "What is topic modeling?",
    "What is Latent Dirichlet Allocation?",
    "What is word embedding?",
    "How does Word2vec work?",
    "What is GloVe in machine learning?",
    "What is a recurrent neural network?",
    "What is long short-term memory?",
    "What is a gated recurrent unit?",
    "What is a convolutional neural network?",
    # Modern architectures
    "What is ResNet?",
    "What is transfer learning?",
    "What is zero-shot learning?",
    "What is few-shot learning?",
    "What is prompt engineering?",
    "What is chain-of-thought prompting?",
    "How does a retrieval system work?",
    "What is BM25?",
    "What is TF-IDF?",
    "How does PageRank work?",
    # Search and retrieval
    "What is Apache Lucene?",
    "What is Elasticsearch?",
    "What is OpenSearch?",
    "What is Chroma used for?",
    "What is Pinecone?",
    "What is Weaviate?",
    "What is Milvus?",
    "What is FAISS?",
    "How is cosine similarity computed?",
    "What is Euclidean distance?",
    # Algorithms
    "How does k-nearest neighbors work?",
    "What is HNSW?",
    "What is approximate nearest neighbor search?",
    # Security and multi-tenancy
    "What is multi-tenancy in software?",
    "What is role-based access control?",
    "What is attribute-based access control?",
    "What is relationship-based access control?",
    "What is a JSON Web Token?",
    "What is OAuth?",
    "What is OpenID Connect?",
    # Multi-hop reasoning
    "How do knowledge graphs relate to semantic search?",
    "What is the relationship between embeddings and vector databases?",
    "How does RAG use knowledge graphs?",
    "What connects BERT and transformer models?",
    "How does fine-tuning relate to transfer learning?",
    # Summary / global
    "What are the key concepts in natural language processing?",
    "What are the main types of neural networks?",
    "What are the main approaches to information retrieval?",
    "What are the different types of graph databases?",
    "How do access control systems work?",
    "What are the main embedding models available?",
    "What are the key components of a knowledge graph system?",
    "What are the differences between vector and graph retrieval?",
    # Data concepts
    "What is bitemporal modeling?",
    "What is slowly changing dimension?",
    "What is data versioning?",
    "What is change data capture?",
    "What is an ETL pipeline?",
    "What is data lineage?",
    "What is data provenance?",
    "How does data pipeline work?",
    "What is FastText?",
    "What is a Transformer model?",
]

assert len(BENCHMARK_QUESTIONS) == 100, f"Expected 100 questions, got {len(BENCHMARK_QUESTIONS)}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_headers() -> dict[str, str]:
    if not ORACLOUS_API_KEY:
        print("ERROR: ORACLOUS_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    return {"Authorization": f"Bearer {ORACLOUS_API_KEY}", "Content-Type": "application/json"}


def percentile(values: list[float], pct: float) -> float:
    """Compute the p-th percentile (0–100) of a list of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct / 100
    lo = int(k)
    hi = lo + 1
    if hi >= len(sorted_vals):
        return sorted_vals[-1]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def run_queries_for_retriever(
    session: requests.Session,
    retriever_type: str,
) -> list[dict]:
    """Run all 100 benchmark questions with the given retriever type."""
    url = f"{KGB_BASE_URL}/graphs/{GRAPH_ID}/chat"
    results: list[dict] = []

    for i, question in enumerate(BENCHMARK_QUESTIONS, 1):
        payload = {
            "query": question,
            "graph_id": GRAPH_ID,
            "retriever_type": retriever_type,
            "return_context": False,
            "include_sources": False,
        }

        t_start = time.perf_counter()
        try:
            resp = session.post(url, json=payload, headers=get_headers(), timeout=60)
            elapsed = time.perf_counter() - t_start
            success = resp.ok
            error = None if resp.ok else resp.text[:200]
        except requests.RequestException as exc:
            elapsed = time.perf_counter() - t_start
            success = False
            error = str(exc)

        results.append(
            {
                "question_idx": i,
                "question": question,
                "elapsed_s": round(elapsed, 3),
                "success": success,
                "error": error,
            }
        )

        status_icon = "OK" if success else "FAIL"
        print(f"    [{i:3d}/100] [{status_icon}] {elapsed:.2f}s  {question[:50]}")

    return results


def compute_stats(results: list[dict]) -> dict:
    """Compute latency statistics for a set of query results."""
    ok = [r for r in results if r["success"]]
    times = [r["elapsed_s"] for r in ok]
    return {
        "total_queries": len(results),
        "successful": len(ok),
        "failed": len(results) - len(ok),
        "p50_s": round(percentile(times, 50), 3) if times else None,
        "p95_s": round(percentile(times, 95), 3) if times else None,
        "p99_s": round(percentile(times, 99), 3) if times else None,
        "mean_s": round(statistics.mean(times), 3) if times else None,
        "min_s": round(min(times), 3) if times else None,
        "max_s": round(max(times), 3) if times else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Oraclous Benchmark 3: Chat Latency")
    print("=" * 60)

    if not GRAPH_ID:
        print("ERROR: Set GRAPH_ID to the UUID of your target graph.", file=sys.stderr)
        sys.exit(1)

    print(f"Retriever types to test: {', '.join(RETRIEVER_TYPES)}")
    print(f"Questions per retriever: {len(BENCHMARK_QUESTIONS)}")
    print()

    session = requests.Session()
    per_retriever: dict[str, dict] = {}

    for retriever_type in RETRIEVER_TYPES:
        print(f"--- Testing retriever: {retriever_type} ---")
        results = run_queries_for_retriever(session, retriever_type)
        stats = compute_stats(results)
        per_retriever[retriever_type] = {
            "stats": stats,
            "per_query": results,
        }
        print(
            f"  P50={stats['p50_s']}s  P95={stats['p95_s']}s  P99={stats['p99_s']}s  "
            f"mean={stats['mean_s']}s  ({stats['successful']}/{stats['total_queries']} OK)"
        )
        print()

    # Determine target status for vector retriever
    vector_stats = per_retriever.get("vector", {}).get("stats", {})
    vector_p95 = vector_stats.get("p95_s")
    vector_target_met = vector_p95 is not None and vector_p95 < TARGET_P95_VECTOR_S

    output = {
        "benchmark": "chat_latency",
        "graph_id": GRAPH_ID,
        "retriever_types_tested": RETRIEVER_TYPES,
        "questions_per_retriever": len(BENCHMARK_QUESTIONS),
        "target_vector_p95_s": TARGET_P95_VECTOR_S,
        "vector_target_met": vector_target_met,
        "per_retriever": per_retriever,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "latency.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS — Chat Latency")
    print("=" * 60)
    print(f"  {'Retriever':<20} {'P50':>8} {'P95':>8} {'P99':>8} {'mean':>8} {'OK':>6}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for rt, data in per_retriever.items():
        s = data["stats"]
        print(
            f"  {rt:<20} {str(s['p50_s']) + 's':>8} {str(s['p95_s']) + 's':>8} "
            f"{str(s['p99_s']) + 's':>8} {str(s['mean_s']) + 's':>8} "
            f"{s['successful']}/{s['total_queries']:>3}"
        )
    print()

    if vector_p95 is not None:
        result_str = "PASS" if vector_target_met else "FAIL"
        print(f"  TARGET vector P95 <{TARGET_P95_VECTOR_S}s: {result_str} ({vector_p95}s)")
    else:
        print(f"  TARGET vector P95 <{TARGET_P95_VECTOR_S}s: SKIP (vector retriever not tested or no results)")

    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
