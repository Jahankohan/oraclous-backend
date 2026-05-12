"""
Benchmark 1: Ingestion Throughput
==================================
Fetches 100 Wikipedia article summaries and ingests them one at a time into
a knowledge graph, measuring docs/min and P95 ingestion time per doc.

Usage:
    ORACLOUS_API_KEY=<key> GRAPH_ID=<id> python bench_ingestion.py

Environment variables:
    ORACLOUS_API_KEY  — Bearer token for the knowledge-graph-builder service
    GRAPH_ID          — UUID of the target graph (must already exist and be writable)
    KGB_BASE_URL      — Base URL for knowledge-graph-builder (default: http://localhost:8003/api/v1)

Outputs:
    results/ingestion.json  — per-doc timings + aggregate stats

Targets:
    docs/min  > 20  (PASS / FAIL printed at end)
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

WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

# Fixed list of 100 article titles for reproducibility
ARTICLE_TITLES: list[str] = [
    "Python_(programming_language)",
    "Machine_learning",
    "Artificial_intelligence",
    "Deep_learning",
    "Natural_language_processing",
    "Neural_network_(machine_learning)",
    "Transformer_(machine_learning_model)",
    "Graph_database",
    "Knowledge_graph",
    "Neo4j",
    "FastAPI",
    "Docker_(software)",
    "Kubernetes",
    "Microservices",
    "REST",
    "GraphQL",
    "PostgreSQL",
    "Redis",
    "Apache_Kafka",
    "Celery_(software)",
    "TensorFlow",
    "PyTorch",
    "Hugging_Face",
    "BERT_(language_model)",
    "GPT-4",
    "Large_language_model",
    "Retrieval-augmented_generation",
    "Vector_database",
    "Embeddings_(machine_learning)",
    "Semantic_search",
    "Fine-tuning_(deep_learning)",
    "Reinforcement_learning_from_human_feedback",
    "Attention_mechanism",
    "Encoder-decoder",
    "Named-entity_recognition",
    "Information_extraction",
    "Knowledge_representation_and_reasoning",
    "Ontology_(information_science)",
    "Resource_Description_Framework",
    "SPARQL",
    "Cypher_query_language",
    "Property_graph",
    "Entity_resolution",
    "Coreference_resolution",
    "Relation_extraction",
    "Event_extraction",
    "Text_mining",
    "Sentiment_analysis",
    "Topic_model",
    "Latent_Dirichlet_allocation",
    "Word_embedding",
    "Word2vec",
    "GloVe_(machine_learning)",
    "FastText",
    "Recurrent_neural_network",
    "Long_short-term_memory",
    "Gated_recurrent_unit",
    "Convolutional_neural_network",
    "ResNet",
    "Transfer_learning",
    "Zero-shot_learning",
    "Few-shot_learning",
    "Prompt_engineering",
    "Chain-of-thought_prompting",
    "Retrieval_system",
    "BM25",
    "TF-IDF",
    "PageRank",
    "Apache_Lucene",
    "Elasticsearch",
    "OpenSearch_(software)",
    "Chroma_(database)",
    "Pinecone_(vector_database)",
    "Weaviate",
    "Milvus_(software)",
    "FAISS",
    "Cosine_similarity",
    "Euclidean_distance",
    "k-nearest_neighbors_algorithm",
    "Hierarchical_navigable_small_world",
    "Approximate_nearest_neighbor_search",
    "Multi-tenancy",
    "Role-based_access_control",
    "Attribute-based_access_control",
    "Relationship-based_access_control",
    "JSON_Web_Token",
    "OAuth",
    "OpenID_Connect",
    "Cypher_(query_language)",
    "Labeled_property_graph",
    "RDF_triplestore",
    "Linked_data",
    "Data_lineage",
    "Data_provenance",
    "Bitemporal_modeling",
    "Slowly_changing_dimension",
    "Data_versioning",
    "Change_data_capture",
    "ETL",
    "Data_pipeline",
]

# Ensure exactly 100 titles
assert len(ARTICLE_TITLES) == 100, f"Expected 100 titles, got {len(ARTICLE_TITLES)}"

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
RESULTS_DIR = Path(__file__).parent.parent / "results"

TARGET_DOCS_PER_MIN = 20


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


def fetch_wikipedia_summaries() -> list[dict]:
    """Fetch summaries from Wikipedia REST API. Returns list of article dicts."""
    print(f"Fetching {len(ARTICLE_TITLES)} Wikipedia article summaries...")
    articles: list[dict] = []
    session = requests.Session()

    for i, title in enumerate(ARTICLE_TITLES, 1):
        try:
            resp = session.get(WIKIPEDIA_API.format(title=title), timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                articles.append(
                    {
                        "title": data.get("title", title),
                        "content": data.get("extract", ""),
                        "url": data.get("content_urls", {})
                        .get("desktop", {})
                        .get("page", ""),
                        "wikipedia_title": title,
                    }
                )
            else:
                print(f"  [{i:3d}] WARNING: HTTP {resp.status_code} for {title}")
                articles.append(
                    {
                        "title": title.replace("_", " "),
                        "content": f"Article about {title.replace('_', ' ')} (fetch failed with HTTP {resp.status_code}).",
                        "url": f"https://en.wikipedia.org/wiki/{title}",
                        "wikipedia_title": title,
                    }
                )
        except requests.RequestException as exc:
            print(f"  [{i:3d}] WARNING: request error for {title}: {exc}")
            articles.append(
                {
                    "title": title.replace("_", " "),
                    "content": f"Article about {title.replace('_', ' ')} (fetch failed: {exc}).",
                    "url": f"https://en.wikipedia.org/wiki/{title}",
                    "wikipedia_title": title,
                }
            )

        if i % 10 == 0:
            print(f"  {i}/100 fetched")

    return articles


def save_dataset(articles: list[dict]) -> Path:
    """Save articles to datasets/wikipedia_100.jsonl."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATASETS_DIR / "wikipedia_100.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for article in articles:
            fh.write(json.dumps(article, ensure_ascii=False) + "\n")
    print(f"Saved {len(articles)} articles to {out_path}")
    return out_path


def load_dataset() -> list[dict]:
    """Load articles from datasets/wikipedia_100.jsonl if it exists."""
    path = DATASETS_DIR / "wikipedia_100.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def ingest_article(session: requests.Session, article: dict) -> dict:
    """POST one article to the ingest endpoint. Returns timing + job info."""
    if not GRAPH_ID:
        print("ERROR: GRAPH_ID environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    url = f"{KGB_BASE_URL}/graphs/{GRAPH_ID}/ingest"
    payload = {
        "content": f"{article['title']}\n\n{article['content']}",
        "source_type": "text",
        "mode": "incremental",
    }

    t_start = time.perf_counter()
    resp = session.post(url, json=payload, headers=get_headers(), timeout=120)
    elapsed = time.perf_counter() - t_start

    result: dict = {
        "title": article["title"],
        "elapsed_s": round(elapsed, 3),
        "http_status": resp.status_code,
        "success": resp.status_code in (200, 201),
    }

    if resp.ok:
        body = resp.json()
        result["job_id"] = body.get("id")
        result["job_status"] = body.get("status")
    else:
        result["error"] = resp.text[:200]

    return result


def compute_percentile(values: list[float], pct: float) -> float:
    """Return the p-th percentile of a sorted list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, int(len(sorted_vals) * pct / 100) - 1)
    return sorted_vals[idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Oraclous Benchmark 1: Ingestion Throughput")
    print("=" * 60)

    if not GRAPH_ID:
        print("ERROR: Set GRAPH_ID to the UUID of your target graph.", file=sys.stderr)
        sys.exit(1)

    # Step 1: Fetch or load dataset
    articles = load_dataset()
    if len(articles) == 100:
        print(f"Loaded {len(articles)} articles from cache.")
    else:
        articles = fetch_wikipedia_summaries()
        save_dataset(articles)

    # Step 2: Ingest one at a time
    print(f"\nIngesting {len(articles)} articles into graph {GRAPH_ID} ...")
    session = requests.Session()
    per_doc_results: list[dict] = []
    bench_start = time.perf_counter()

    for i, article in enumerate(articles, 1):
        result = ingest_article(session, article)
        per_doc_results.append(result)
        status_icon = "OK" if result["success"] else "FAIL"
        print(
            f"  [{i:3d}/100] [{status_icon}] {article['title'][:50]:50s}  "
            f"{result['elapsed_s']:.2f}s"
        )

    bench_elapsed = time.perf_counter() - bench_start

    # Step 3: Compute aggregate stats
    success_results = [r for r in per_doc_results if r["success"]]
    failed_results = [r for r in per_doc_results if not r["success"]]
    elapsed_times = [r["elapsed_s"] for r in success_results]

    docs_per_min = (
        (len(success_results) / bench_elapsed) * 60 if bench_elapsed > 0 else 0
    )
    p50 = compute_percentile(elapsed_times, 50)
    p95 = compute_percentile(elapsed_times, 95)
    p99 = compute_percentile(elapsed_times, 99)
    mean_elapsed = statistics.mean(elapsed_times) if elapsed_times else 0

    aggregate = {
        "total_docs": len(articles),
        "successful": len(success_results),
        "failed": len(failed_results),
        "total_elapsed_s": round(bench_elapsed, 2),
        "docs_per_min": round(docs_per_min, 2),
        "latency_mean_s": round(mean_elapsed, 3),
        "latency_p50_s": round(p50, 3),
        "latency_p95_s": round(p95, 3),
        "latency_p99_s": round(p99, 3),
        "target_docs_per_min": TARGET_DOCS_PER_MIN,
        "target_met": docs_per_min >= TARGET_DOCS_PER_MIN,
    }

    output = {
        "benchmark": "ingestion_throughput",
        "graph_id": GRAPH_ID,
        "aggregate": aggregate,
        "per_doc": per_doc_results,
    }

    # Step 4: Write results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "ingestion.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    # Step 5: Print summary
    print("\n" + "=" * 60)
    print("RESULTS — Ingestion Throughput")
    print("=" * 60)
    print(f"  Total docs:         {len(articles)}")
    print(f"  Successful:         {len(success_results)}")
    print(f"  Failed:             {len(failed_results)}")
    print(f"  Total elapsed:      {bench_elapsed:.1f}s")
    print(f"  Throughput:         {docs_per_min:.1f} docs/min")
    print(f"  Latency mean:       {mean_elapsed:.2f}s")
    print(f"  Latency P50:        {p50:.2f}s")
    print(f"  Latency P95:        {p95:.2f}s")
    print(f"  Latency P99:        {p99:.2f}s")
    print()

    if aggregate["target_met"]:
        print(
            f"  TARGET >={TARGET_DOCS_PER_MIN} docs/min: PASS ({docs_per_min:.1f} docs/min)"
        )
    else:
        print(
            f"  TARGET >={TARGET_DOCS_PER_MIN} docs/min: FAIL ({docs_per_min:.1f} docs/min)"
        )

    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
