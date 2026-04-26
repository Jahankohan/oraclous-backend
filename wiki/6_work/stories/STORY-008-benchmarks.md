---
id: STORY-008
title: "Publish benchmarks: ingestion throughput, retrieval quality (RAGAS), chat latency, federation overhead"
type: research
layer: cross-cutting
reporter: reza
status: ready
priority: high
created: 2026-04-26
updated: 2026-04-26
wiki_refs: []
tasks: []
decisions: []
---

# STORY-008: Benchmarks & Evaluation

## Summary

Zero performance or quality numbers are published for Oraclous. Competitors publish:
MS GraphRAG (comprehensiveness/diversity scores), FalkorDB (3.4x improvement claim),
Zep (18.5% DMR accuracy improvement). Without benchmarks, all feature parity claims
are theoretical. This story creates a reproducible benchmark suite covering ingestion,
retrieval quality, chat latency, and federation overhead.

## Problem Statement

- No benchmark exists for any system property (ingestion speed, RAGAS scores, P95 latency)
- Cannot make competitive claims without numbers to back them
- The RAGAS evaluation endpoint exists but has never been run on a published dataset
- Community detection quality (Leiden vs Louvain) cannot be measured without test harness

## Goals

- [ ] Create `benchmarks/` directory with reproducible scripts and documented methodology
- [ ] Ingestion benchmark: 100 Wikipedia articles, measure docs/min, entity count/doc, incremental skip rate
- [ ] Retrieval quality: run RAGAS on curated 100 Q&A pairs (MS GraphRAG podcast dataset or Wikipedia)
- [ ] Chat latency: 100 sequential queries, measure P50/P95/P99 and first-token streaming time
- [ ] Federation benchmark: same query across 1/2/5 graphs, measure overhead vs baseline
- [ ] Community quality: coherence ratio (intra vs inter similarity), global query accuracy (20 thematic questions, human-rated)

## Non-Goals

- Automated CI benchmark runs (manual execution for initial publication)
- Comparison against other platforms (measure Oraclous only; comparisons in separate analysis doc)
- Benchmark infrastructure (no custom benchmark framework; shell scripts + Python are sufficient)

## Acceptance Criteria

- [ ] `benchmarks/README.md` documents exact reproduction steps from zero (clone → run → results)
- [ ] Ingestion benchmark runs to completion and produces a results JSON with all target metrics
- [ ] RAGAS benchmark runs against published test dataset; results match or exceed targets in spec
- [ ] Chat latency numbers exist for P50, P95, P99 on default docker-compose setup
- [ ] `benchmarks/report.md` published with all numbers and methodology notes

## Open Questions

| # | Question | Owner | Status |
|---|----------|-------|--------|
| 1 | Use MS GraphRAG podcast dataset or curated Wikipedia Q&A? | reza | open |
| 2 | Human rating for global query accuracy: Reza + one reviewer, or automated? | reza | open |

## Context & Background

- Full technical spec: `ORACLOUS_DEEPENING_ROADMAP.md` § 12 (Benchmarks, pp. 822-889)
- Existing RAGAS endpoint: `app/api/v1/endpoints/evaluation.py`
- Targets defined in spec: >20 docs/min ingestion, faithfulness >0.85, P95 latency <5s
- Estimated effort: 1-2 weeks
