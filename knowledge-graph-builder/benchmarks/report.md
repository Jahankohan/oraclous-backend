# Oraclous Benchmark Report

> **Status:** Template — run all 4 scripts and fill in the "Actual" column.
> See `README.md` for reproduction steps.

---

## Machine Spec

| Field | Value |
|-------|-------|
| Machine | TBD (e.g., MacBook Pro M3 Max, 36 GB RAM) |
| OS | TBD (e.g., macOS 15.x) |
| Docker | TBD (e.g., Docker Desktop 4.x, 8 CPUs / 16 GB RAM allocated) |
| Neo4j | TBD (e.g., Neo4j 5.x, community edition, default heap) |
| Date | TBD |
| Oraclous commit | TBD (run `git rev-parse HEAD`) |

---

## Benchmark 1: Ingestion Throughput

**Dataset:** 100 Wikipedia article summaries (fetched via Wikipedia REST API)
**Method:** Sequential ingestion via `POST /api/v1/graphs/{graph_id}/ingest`, one doc at a time
**Mode:** `incremental`

| Metric | Target | Actual |
|--------|--------|--------|
| Throughput (docs/min) | >20 | TBD — run `bench_ingestion.py` |
| Latency mean (s/doc) | — | TBD |
| Latency P50 (s/doc) | — | TBD |
| Latency P95 (s/doc) | — | TBD |
| Latency P99 (s/doc) | — | TBD |
| Total elapsed (s) | — | TBD |
| Successful docs | — | TBD/100 |

**Target met:** TBD (PASS / FAIL)

**Notes:**
- Background job timing: the API responds immediately after job creation; ingestion
  continues asynchronously. The elapsed time per doc measured here is the HTTP round-trip
  to start the job, not the time for full entity extraction.
- Entity extraction counts are available in the job status endpoint
  (`GET /api/v1/graphs/{graph_id}/jobs/{job_id}`) after each job completes.

---

## Benchmark 2: RAGAS Retrieval Quality

**Dataset:** 100 Q&A pairs generated from the 100 Wikipedia articles via Claude Haiku
**Method:**
1. `POST /api/v1/graphs/{graph_id}/chat` — get system answer + contexts
2. `POST /api/v1/graphs/{graph_id}/evaluate` — compute RAGAS scores

| Metric | Target | Actual |
|--------|--------|--------|
| Faithfulness (mean) | >0.85 | TBD — run `bench_ragas.py` |
| Answer Relevance (mean) | >0.80 | TBD |
| Context Precision (mean) | — | TBD |
| Context Recall (mean) | — | TBD |
| Evaluated pairs | — | TBD/100 |
| Errors | — | TBD |

**Faithfulness target met:** TBD (PASS / FAIL)
**Answer Relevance target met:** TBD (PASS / FAIL)

### RAGAS metric definitions

- **Faithfulness** — fraction of claims in the system answer that are supported by
  retrieved context. Measures hallucination rate. Score of 1.0 = fully grounded.
- **Answer Relevance** — how well the answer addresses the question (independent of
  whether it is factually correct). Measures response coherence.
- **Context Precision** — fraction of retrieved context chunks that are actually
  relevant to answering the question. Measures retrieval precision.
- **Context Recall** — fraction of ground-truth facts that appear in the retrieved
  context. Measures retrieval recall. Requires `ground_truth`.

---

## Benchmark 3: Chat Latency

**Dataset:** 100 hardcoded questions (mix of factual, multi-hop, and summary queries)
**Method:** 100 sequential `POST /api/v1/graphs/{graph_id}/chat` calls per retriever type

| Retriever | P50 (s) | P95 (s) | P99 (s) | Mean (s) | Success |
|-----------|---------|---------|---------|----------|---------|
| vector | TBD | TBD | TBD | TBD | TBD/100 |
| vector_cypher | TBD | TBD | TBD | TBD | TBD/100 |
| hybrid | TBD | TBD | TBD | TBD | TBD/100 |
| hybrid_cypher | TBD | TBD | TBD | TBD | TBD/100 |
| text2cypher | TBD | TBD | TBD | TBD | TBD/100 |

**Target — vector P95 <5s:** TBD (PASS / FAIL)

**Notes:**
- `vector` — pure vector similarity search; fastest retriever
- `vector_cypher` — vector search + graph traversal (default mode); best quality/speed tradeoff
- `hybrid` — vector + full-text search; requires full-text indexes
- `hybrid_cypher` — hybrid search + graph traversal; most thorough, slowest
- `text2cypher` — natural language to Cypher; accuracy depends on schema size

---

## Benchmark 4: Federation Overhead

**Dataset:** Same 10 questions across 1, 2, and 5 graphs (same Wikipedia content)
**Method:** `POST /api/v1/federation/query` (fallback: sequential single-graph chat)
**Repetitions:** 10 per scenario

| Scenario | P50 (s) | P95 (s) | P99 (s) | Mean (s) |
|----------|---------|---------|---------|----------|
| 1 graph (baseline) | TBD | TBD | TBD | TBD |
| 2 graphs | TBD | TBD | TBD | TBD |
| 5 graphs | TBD | TBD | TBD | TBD |

**Overhead per additional graph (P95):** TBD s/graph

**Notes:**
- Overhead = (5-graph P95 minus 1-graph P95) / 4
- If the federation endpoint is not yet deployed, the script falls back to sequential
  single-graph chat calls. Sequential results measure raw cumulative latency, not
  true parallel federation overhead.

---

## Methodology

### Corpus

100 Wikipedia article summaries fetched from the Wikipedia REST API
(`https://en.wikipedia.org/api/rest_v1/page/summary/{title}`).
Titles are hardcoded in `bench_ingestion.py` for reproducibility.
Wikipedia content is in the public domain (CC BY-SA); no licensing concerns.

### Auth

All API calls use a Bearer token from `ORACLOUS_API_KEY`.
No credentials are hardcoded in any script.

### Timing

- Ingestion latency: measures HTTP round-trip time to submit the job (not full extraction time)
- Chat latency: measures full HTTP round-trip including retrieval + LLM generation
- All timings use `time.perf_counter()` for sub-millisecond precision

### Reproducibility

- Article list: 100 fixed titles (hardcoded in `bench_ingestion.py`)
- Latency questions: 100 fixed questions (hardcoded in `bench_latency.py`)
- Federation queries: 10 fixed questions (hardcoded in `bench_federation.py`)
- Q&A pairs: LLM-generated (may vary between runs); cached in `datasets/qa_pairs_100.json`

### Competitive context

| Competitor | Published metric |
|------------|-----------------|
| Zep | 18.5% DMR improvement over full-context baseline |
| FalkorDB | Sub-50ms query latency (graph operations) |
| MS GraphRAG | Comprehensiveness: 72% win rate vs naive RAG |
| Oraclous | TBD — run scripts |

---

Generated by TASK-027. Re-run all 4 scripts to refresh numbers.
