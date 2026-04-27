# Oraclous Benchmark Suite

Reproducible benchmarks for the Oraclous knowledge-graph-builder service.
All scripts run against the live Docker Compose stack — they call the HTTP API only.

## Benchmarks

| # | Script | What it measures | Target |
|---|--------|-----------------|--------|
| 1 | `bench_ingestion.py` | Ingestion throughput (docs/min, P95 latency) | >20 docs/min |
| 2 | `bench_ragas.py` | Retrieval quality via RAGAS metrics | faithfulness >0.85, answer_relevance >0.80 |
| 3 | `bench_latency.py` | Chat P50/P95/P99 per retriever type | P95 <5s (vector) |
| 4 | `bench_federation.py` | Cross-graph federation overhead | Overhead per graph documented |

---

## Prerequisites

### 1. Docker Compose stack running

```bash
cd /path/to/oraclous-data-studio
docker compose up -d
```

Wait until all services are healthy:

```bash
docker compose ps
# knowledge-graph-builder should be "Up (healthy)"
```

Service ports:
- Auth service:              `http://localhost:8000`
- Credential broker:         `http://localhost:8002`
- Knowledge-graph-builder:   `http://localhost:8003`

### 2. Environment variables

```bash
export ORACLOUS_API_KEY="<your-bearer-token>"
export ANTHROPIC_API_KEY="<your-anthropic-key>"   # Only needed for bench_ragas.py
```

#### How to obtain ORACLOUS_API_KEY

Option A — Use your user JWT (short-lived):
```bash
curl -s -X POST http://localhost:8000/login/ \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "yourpassword"}' \
  | jq -r '.access_token'
```

Option B — Create a service account (long-lived):
```bash
# Create a graph first, then create a service account scoped to it:
curl -s -X POST http://localhost:8003/api/v1/graphs/<graph_id>/service-accounts \
  -H "Authorization: Bearer <your-jwt>" \
  -H "Content-Type: application/json" \
  -d '{"name": "benchmark-sa", "permission": "write"}' \
  | jq -r '.api_key'
```

### 3. Python dependencies

The benchmark scripts use only the Python standard library and `requests`.
Install `requests` if not already available:

```bash
pip install requests
```

---

## How to create the test graph

Before running any benchmark, create a knowledge graph to ingest into:

```bash
# Create a graph
GRAPH_ID=$(curl -s -X POST http://localhost:8003/api/v1/graphs \
  -H "Authorization: Bearer $ORACLOUS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "benchmark-graph", "description": "Benchmark dataset: 100 Wikipedia articles"}' \
  | jq -r '.id')

echo "GRAPH_ID=$GRAPH_ID"
export GRAPH_ID
```

For Benchmark 4 (federation), create five graphs with the same content:

```bash
for i in 1 2 3 4 5; do
  ID=$(curl -s -X POST http://localhost:8003/api/v1/graphs \
    -H "Authorization: Bearer $ORACLOUS_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"benchmark-fed-$i\", \"description\": \"Federation benchmark graph $i\"}" \
    | jq -r '.id')
  echo "Graph $i: $ID"
done
# Set GRAPH_IDS=<id1>,<id2>,<id3>,<id4>,<id5>
```

---

## How to run each script

Run all scripts from the `knowledge-graph-builder/` directory:

```bash
cd knowledge-graph-builder
```

### Benchmark 1: Ingestion throughput

```bash
ORACLOUS_API_KEY=$ORACLOUS_API_KEY \
GRAPH_ID=$GRAPH_ID \
python benchmarks/scripts/bench_ingestion.py
```

This will:
1. Fetch 100 Wikipedia article summaries (saved to `benchmarks/datasets/wikipedia_100.jsonl`)
2. Ingest them one at a time into the graph
3. Write results to `benchmarks/results/ingestion.json`
4. Print PASS/FAIL against the >20 docs/min target

### Benchmark 2: RAGAS retrieval quality

Requires Benchmark 1 to have completed (articles must be ingested).

```bash
ORACLOUS_API_KEY=$ORACLOUS_API_KEY \
GRAPH_ID=$GRAPH_ID \
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
python benchmarks/scripts/bench_ragas.py
```

To skip Q&A pair generation (use cached `qa_pairs_100.json`):

```bash
SKIP_QA_GEN=1 \
ORACLOUS_API_KEY=$ORACLOUS_API_KEY \
GRAPH_ID=$GRAPH_ID \
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
python benchmarks/scripts/bench_ragas.py
```

### Benchmark 3: Chat latency

```bash
ORACLOUS_API_KEY=$ORACLOUS_API_KEY \
GRAPH_ID=$GRAPH_ID \
python benchmarks/scripts/bench_latency.py
```

To test specific retriever types only:

```bash
RETRIEVER_TYPES=vector,vector_cypher \
ORACLOUS_API_KEY=$ORACLOUS_API_KEY \
GRAPH_ID=$GRAPH_ID \
python benchmarks/scripts/bench_latency.py
```

Available retriever types: `vector`, `vector_cypher`, `hybrid`, `hybrid_cypher`, `text2cypher`

### Benchmark 4: Federation overhead

Requires 5 pre-populated graphs. Populate each with the same dataset by running
Benchmark 1 five times with different `GRAPH_ID` values.

```bash
GRAPH_IDS="<id1>,<id2>,<id3>,<id4>,<id5>" \
ORACLOUS_API_KEY=$ORACLOUS_API_KEY \
python benchmarks/scripts/bench_federation.py
```

If the federation endpoint is not available, the script automatically falls back
to sequential single-graph chat queries to provide a comparable baseline.

---

## How to regenerate datasets

### Re-fetch Wikipedia articles

Delete the cached file and re-run Benchmark 1:

```bash
rm benchmarks/datasets/wikipedia_100.jsonl
ORACLOUS_API_KEY=$ORACLOUS_API_KEY GRAPH_ID=$GRAPH_ID python benchmarks/scripts/bench_ingestion.py
```

### Re-generate Q&A pairs

Delete the cached file and re-run Benchmark 2 without `SKIP_QA_GEN`:

```bash
rm benchmarks/datasets/qa_pairs_100.json
ORACLOUS_API_KEY=$ORACLOUS_API_KEY GRAPH_ID=$GRAPH_ID ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
python benchmarks/scripts/bench_ragas.py
```

Note: Q&A pairs are generated by an LLM and may differ slightly between runs.
The `wikipedia_100.jsonl` articles are always fetched from the same fixed list of
100 titles, so the knowledge base itself is reproducible.

---

## Expected runtimes

These are rough estimates on a MacBook Pro M3 with the Docker stack running locally:

| Benchmark | Steps | Estimated time |
|-----------|-------|----------------|
| 1. Ingestion | Fetch 100 articles + ingest 100 docs | 5–20 minutes (depends on Neo4j throughput) |
| 2. RAGAS | Generate 100 Q&A pairs + 100 chat + 100 evaluate | 30–60 minutes |
| 3. Latency | 100 queries × 5 retriever types | 10–30 minutes |
| 4. Federation | 10 queries × 10 reps × 3 scenarios | 20–40 minutes |
| **Total** | | **~1–2.5 hours** |

### Running all benchmarks end-to-end

```bash
export ORACLOUS_API_KEY="..."
export GRAPH_ID="..."
export ANTHROPIC_API_KEY="..."
export GRAPH_IDS="<id1>,<id2>,<id3>,<id4>,<id5>"

python benchmarks/scripts/bench_ingestion.py
python benchmarks/scripts/bench_ragas.py
python benchmarks/scripts/bench_latency.py
python benchmarks/scripts/bench_federation.py
```

Results are written to `benchmarks/results/*.json`. See `benchmarks/report.md` for
the published results template.

---

## File layout

```
benchmarks/
├── README.md                  ← This file
├── datasets/
│   ├── wikipedia_100.jsonl    ← 100 Wikipedia article summaries (generated by bench_ingestion.py)
│   └── qa_pairs_100.json      ← 100 Q&A pairs for RAGAS (generated by bench_ragas.py)
├── scripts/
│   ├── bench_ingestion.py     ← Benchmark 1: ingestion throughput
│   ├── bench_ragas.py         ← Benchmark 2: retrieval quality (RAGAS)
│   ├── bench_latency.py       ← Benchmark 3: chat latency P50/P95/P99
│   └── bench_federation.py    ← Benchmark 4: federation overhead
├── results/                   ← JSON results (written by scripts, gitignored)
│   ├── ingestion.json
│   ├── ragas.json
│   ├── latency.json
│   └── federation.json
└── report.md                  ← Published results template
```

## Notes

- Scripts use only the HTTP API — no direct Neo4j or database access.
- All auth is via Bearer token from `ORACLOUS_API_KEY` env var — never hardcoded.
- Wikipedia articles are fetched from the Wikipedia REST API using a fixed list of 100 titles.
  The list is hardcoded in `bench_ingestion.py` for reproducibility.
- Q&A pairs are generated by Claude Haiku (`claude-haiku-4-5`) via the Anthropic API.
  Regenerating them may produce slightly different questions.
- `results/` JSON files are gitignored — copy notable results into `report.md` manually.
