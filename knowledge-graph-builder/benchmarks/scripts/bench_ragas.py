"""
Benchmark 2: RAGAS Retrieval Quality
======================================
Uses the 100 Wikipedia articles ingested in Benchmark 1 as a knowledge base.
For each article, calls an LLM to generate a question + ground-truth answer,
then queries the graph chat endpoint and evaluates with the RAGAS evaluation
endpoint.

Usage:
    ORACLOUS_API_KEY=<key> GRAPH_ID=<id> ANTHROPIC_API_KEY=<key> python bench_ragas.py

Environment variables:
    ORACLOUS_API_KEY  — Bearer token for the knowledge-graph-builder service
    GRAPH_ID          — UUID of the target graph (populated by bench_ingestion.py)
    ANTHROPIC_API_KEY — Anthropic API key for Q&A pair generation
    KGB_BASE_URL      — Base URL for knowledge-graph-builder (default: http://localhost:8003/api/v1)
    SKIP_QA_GEN       — Set to "1" to skip Q&A generation and use cached qa_pairs_100.json

Outputs:
    datasets/qa_pairs_100.json  — generated Q&A pairs
    results/ragas.json          — per-question RAGAS scores + aggregate

Targets:
    faithfulness      > 0.85  (PASS / FAIL printed at end)
    answer_relevance  > 0.80  (PASS / FAIL printed at end)
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
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SKIP_QA_GEN = os.environ.get("SKIP_QA_GEN", "0") == "1"

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-haiku-4-5"

TARGET_FAITHFULNESS = 0.85
TARGET_ANSWER_RELEVANCE = 0.80

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
RESULTS_DIR = Path(__file__).parent.parent / "results"

QA_GENERATION_PROMPT = """\
You are creating a benchmark dataset for a knowledge graph retrieval system.
Given the following Wikipedia article excerpt, generate exactly ONE question that:
1. Can be answered from the article text alone
2. Is specific and factual (not vague or open-ended)
3. Requires understanding the content, not just keyword matching

Then provide the ground truth answer — a concise, complete sentence.

Article title: {title}
Article text:
{content}

Respond in this exact JSON format (no other text):
{{
  "question": "...",
  "ground_truth": "..."
}}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_headers() -> dict[str, str]:
    if not ORACLOUS_API_KEY:
        print("ERROR: ORACLOUS_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    return {"Authorization": f"Bearer {ORACLOUS_API_KEY}", "Content-Type": "application/json"}


def get_anthropic_headers() -> dict[str, str]:
    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    return {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }


def load_articles() -> list[dict]:
    """Load articles from datasets/wikipedia_100.jsonl."""
    path = DATASETS_DIR / "wikipedia_100.jsonl"
    if not path.exists():
        print(f"ERROR: {path} not found. Run bench_ingestion.py first.", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def generate_qa_pair(session: requests.Session, article: dict) -> dict | None:
    """Call Anthropic to generate a Q&A pair for one article."""
    content_excerpt = article["content"][:1500]  # Truncate long articles
    if not content_excerpt.strip():
        return None

    prompt = QA_GENERATION_PROMPT.format(
        title=article["title"],
        content=content_excerpt,
    )

    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": prompt}],
    }

    for attempt in range(3):
        try:
            resp = session.post(
                ANTHROPIC_MESSAGES_URL,
                json=payload,
                headers=get_anthropic_headers(),
                timeout=30,
            )
            if resp.status_code == 200:
                body = resp.json()
                text = body["content"][0]["text"].strip()
                qa = json.loads(text)
                return {
                    "wikipedia_title": article.get("wikipedia_title", article["title"]),
                    "article_title": article["title"],
                    "question": qa["question"],
                    "ground_truth": qa["ground_truth"],
                }
            elif resp.status_code == 529 or resp.status_code == 429:
                # Rate limited — back off
                wait = 2 ** attempt
                print(f"    Rate limited (HTTP {resp.status_code}), waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    WARNING: Anthropic HTTP {resp.status_code}: {resp.text[:100]}")
                return None
        except (requests.RequestException, json.JSONDecodeError, KeyError) as exc:
            print(f"    WARNING: Q&A gen error (attempt {attempt + 1}): {exc}")
            time.sleep(1)

    return None


def generate_qa_pairs(articles: list[dict]) -> list[dict]:
    """Generate Q&A pairs for all articles via Anthropic."""
    print(f"Generating {len(articles)} Q&A pairs via {ANTHROPIC_MODEL} ...")
    session = requests.Session()
    pairs: list[dict] = []

    for i, article in enumerate(articles, 1):
        pair = generate_qa_pair(session, article)
        if pair:
            pairs.append(pair)
            print(f"  [{i:3d}/100] OK  Q: {pair['question'][:60]}")
        else:
            print(f"  [{i:3d}/100] SKIP (no content or generation failed): {article['title']}")
            # Provide a fallback to keep dataset at ~100 pairs
            pairs.append(
                {
                    "wikipedia_title": article.get("wikipedia_title", article["title"]),
                    "article_title": article["title"],
                    "question": f"What is {article['title']}?",
                    "ground_truth": article["content"][:200] if article.get("content") else "No information available.",
                }
            )

        # Small delay to respect API rate limits
        if i % 10 == 0:
            time.sleep(1)

    return pairs


def save_qa_pairs(pairs: list[dict]) -> Path:
    """Save Q&A pairs to datasets/qa_pairs_100.json."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATASETS_DIR / "qa_pairs_100.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(pairs, fh, indent=2, ensure_ascii=False)
    print(f"Saved {len(pairs)} Q&A pairs to {out_path}")
    return out_path


def load_qa_pairs() -> list[dict]:
    """Load cached Q&A pairs from datasets/qa_pairs_100.json."""
    path = DATASETS_DIR / "qa_pairs_100.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def chat_query(session: requests.Session, question: str) -> dict:
    """Call POST /graphs/{graph_id}/chat and return answer + contexts."""
    if not GRAPH_ID:
        print("ERROR: GRAPH_ID environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    url = f"{KGB_BASE_URL}/graphs/{GRAPH_ID}/chat"
    payload = {
        "query": question,
        "graph_id": GRAPH_ID,
        "return_context": True,
        "include_sources": True,
    }

    resp = session.post(url, json=payload, headers=get_headers(), timeout=60)
    resp.raise_for_status()
    body = resp.json()

    # Extract retrieved contexts from sources
    contexts: list[str] = []
    if body.get("sources"):
        for src in body["sources"]:
            if src.get("content"):
                contexts.append(src["content"])
    elif body.get("context") and body["context"].get("sources"):
        for src in body["context"]["sources"]:
            if src.get("content"):
                contexts.append(src["content"])

    return {
        "answer": body.get("answer", ""),
        "contexts": contexts,
        "is_grounded": body.get("is_grounded", False),
        "retriever_type": body.get("retriever_type", "unknown"),
    }


def evaluate_pair(
    session: requests.Session,
    question: str,
    answer: str,
    ground_truth: str,
) -> dict:
    """Call POST /graphs/{graph_id}/evaluate and return RAGAS scores."""
    if not GRAPH_ID:
        print("ERROR: GRAPH_ID environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    url = f"{KGB_BASE_URL}/graphs/{GRAPH_ID}/evaluate"
    payload = {
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "metrics": ["faithfulness", "answer_relevance", "context_precision", "context_recall"],
    }

    resp = session.post(url, json=payload, headers=get_headers(), timeout=120)
    resp.raise_for_status()
    body = resp.json()

    scores = body.get("scores", {})
    return {
        "faithfulness": scores.get("faithfulness"),
        "answer_relevance": scores.get("answer_relevance"),
        "context_precision": scores.get("context_precision"),
        "context_recall": scores.get("context_recall"),
        "overall": body.get("overall"),
        "metrics_computed": body.get("metrics_computed", []),
        "is_grounded": body.get("is_grounded", False),
        "warnings": body.get("warnings", []),
    }


def safe_mean(values: list[float | None]) -> float | None:
    """Mean of non-None values."""
    clean = [v for v in values if v is not None]
    return statistics.mean(clean) if clean else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Oraclous Benchmark 2: RAGAS Retrieval Quality")
    print("=" * 60)

    if not GRAPH_ID:
        print("ERROR: Set GRAPH_ID to the UUID of your target graph.", file=sys.stderr)
        sys.exit(1)

    # Step 1: Load articles
    articles = load_articles()
    print(f"Loaded {len(articles)} articles from dataset.")

    # Step 2: Generate or load Q&A pairs
    if SKIP_QA_GEN:
        pairs = load_qa_pairs()
        if not pairs:
            print("ERROR: No cached Q&A pairs found. Remove SKIP_QA_GEN=1 to generate.", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded {len(pairs)} Q&A pairs from cache.")
    else:
        pairs = load_qa_pairs()
        if len(pairs) == len(articles):
            print(f"Loaded {len(pairs)} Q&A pairs from cache.")
        else:
            pairs = generate_qa_pairs(articles)
            save_qa_pairs(pairs)

    # Step 3: For each Q&A pair, chat then evaluate
    print(f"\nRunning chat + evaluate for {len(pairs)} Q&A pairs ...")
    session = requests.Session()
    per_question_results: list[dict] = []

    for i, pair in enumerate(pairs, 1):
        question = pair["question"]
        ground_truth = pair["ground_truth"]

        # Chat query
        try:
            t_chat = time.perf_counter()
            chat_result = chat_query(session, question)
            chat_elapsed = time.perf_counter() - t_chat
        except Exception as exc:
            print(f"  [{i:3d}/100] CHAT ERROR: {exc}")
            per_question_results.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "error": f"chat: {exc}",
                }
            )
            continue

        # Evaluate
        try:
            t_eval = time.perf_counter()
            eval_result = evaluate_pair(session, question, chat_result["answer"], ground_truth)
            eval_elapsed = time.perf_counter() - t_eval
        except Exception as exc:
            print(f"  [{i:3d}/100] EVAL ERROR: {exc}")
            per_question_results.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer": chat_result.get("answer", ""),
                    "contexts": chat_result.get("contexts", []),
                    "error": f"eval: {exc}",
                }
            )
            continue

        row = {
            "question": question,
            "ground_truth": ground_truth,
            "answer": chat_result["answer"],
            "contexts": chat_result["contexts"],
            "is_grounded": chat_result["is_grounded"],
            "chat_elapsed_s": round(chat_elapsed, 3),
            "eval_elapsed_s": round(eval_elapsed, 3),
            **eval_result,
        }
        per_question_results.append(row)

        faithfulness_str = (
            f"{eval_result['faithfulness']:.3f}"
            if eval_result["faithfulness"] is not None
            else "n/a"
        )
        ar_str = (
            f"{eval_result['answer_relevance']:.3f}"
            if eval_result["answer_relevance"] is not None
            else "n/a"
        )
        print(
            f"  [{i:3d}/100] faith={faithfulness_str}  ar={ar_str}  "
            f"chat={chat_elapsed:.1f}s  eval={eval_elapsed:.1f}s"
        )

    # Step 4: Aggregate
    ok_results = [r for r in per_question_results if "error" not in r]
    faithfulness_scores = [r.get("faithfulness") for r in ok_results]
    ar_scores = [r.get("answer_relevance") for r in ok_results]
    cp_scores = [r.get("context_precision") for r in ok_results]
    cr_scores = [r.get("context_recall") for r in ok_results]

    mean_faithfulness = safe_mean(faithfulness_scores)
    mean_ar = safe_mean(ar_scores)
    mean_cp = safe_mean(cp_scores)
    mean_cr = safe_mean(cr_scores)

    aggregate = {
        "total_pairs": len(pairs),
        "evaluated": len(ok_results),
        "errors": len(per_question_results) - len(ok_results),
        "mean_faithfulness": round(mean_faithfulness, 4) if mean_faithfulness is not None else None,
        "mean_answer_relevance": round(mean_ar, 4) if mean_ar is not None else None,
        "mean_context_precision": round(mean_cp, 4) if mean_cp is not None else None,
        "mean_context_recall": round(mean_cr, 4) if mean_cr is not None else None,
        "target_faithfulness": TARGET_FAITHFULNESS,
        "target_answer_relevance": TARGET_ANSWER_RELEVANCE,
        "faithfulness_target_met": (mean_faithfulness is not None and mean_faithfulness >= TARGET_FAITHFULNESS),
        "answer_relevance_target_met": (mean_ar is not None and mean_ar >= TARGET_ANSWER_RELEVANCE),
    }

    output = {
        "benchmark": "ragas_retrieval_quality",
        "graph_id": GRAPH_ID,
        "aggregate": aggregate,
        "per_question": per_question_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "ragas.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    # Step 5: Print summary
    print("\n" + "=" * 60)
    print("RESULTS — RAGAS Retrieval Quality")
    print("=" * 60)
    print(f"  Evaluated pairs:      {len(ok_results)}/{len(pairs)}")
    print(f"  Faithfulness:         {mean_faithfulness:.4f}" if mean_faithfulness is not None else "  Faithfulness:         n/a")
    print(f"  Answer relevance:     {mean_ar:.4f}" if mean_ar is not None else "  Answer relevance:     n/a")
    print(f"  Context precision:    {mean_cp:.4f}" if mean_cp is not None else "  Context precision:    n/a")
    print(f"  Context recall:       {mean_cr:.4f}" if mean_cr is not None else "  Context recall:       n/a")
    print()

    for metric, actual, target, key in [
        ("faithfulness", mean_faithfulness, TARGET_FAITHFULNESS, "faithfulness_target_met"),
        ("answer_relevance", mean_ar, TARGET_ANSWER_RELEVANCE, "answer_relevance_target_met"),
    ]:
        if actual is not None:
            result_str = "PASS" if aggregate[key] else "FAIL"
            print(f"  TARGET {metric} >={target}: {result_str} ({actual:.4f})")
        else:
            print(f"  TARGET {metric} >={target}: SKIP (no scores computed)")

    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
