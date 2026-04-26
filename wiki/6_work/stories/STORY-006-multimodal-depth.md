---
id: STORY-006
title: "Deepen multimodal ingestion: OCR fallback, diagram understanding, CSV/JSON/Markdown"
type: feature
layer: knowledge-graph
reporter: reza
status: ready
priority: medium
created: 2026-04-26
updated: 2026-04-26
wiki_refs: ["layer-1-knowledge-graph"]
tasks: []
decisions: []
---

# STORY-006: Multimodal Depth Enhancement

## Summary

The platform accepts PDFs, DOCX, and images but is shallow: scanned PDFs with no
extractable text fail silently, diagrams are treated as generic images (no structured
extraction), and common data formats (CSV, JSON, Markdown, HTML) are not handled.
Cognee supports 38+ types. This story adds OCR fallback (highest priority), diagram
understanding (medium), and structured data format extractors (medium).

## Problem Statement

- Scanned PDFs: PyMuPDF extracts <100 chars → falls back to nothing (no OCR)
- Architecture diagrams, UML, flowcharts: vision LLM returns raw description, not structured nodes/edges
- CSV, JSON, Markdown, HTML: not dispatched at all; ingest fails or is silently skipped
- Cross-modal linking between text mentions and diagram figures: not implemented

## Goals

- [ ] Add Tesseract OCR fallback in `pdf_extractor.py` when PyMuPDF yields <100 chars
- [ ] Add `Dockerfile` entry for `tesseract-ocr` system package
- [ ] Add diagram detection + structured extraction prompt in `vision_extractor.py` (nodes/edges JSON output)
- [ ] Add CSV/TSV extractor: column headers → schema, rows → entities
- [ ] Add JSON/JSONL extractor: schema inference → entity extraction
- [ ] Add Markdown extractor: header hierarchy → document structure graph
- [ ] Wire new file types into `document_processor.py` dispatch table

## Non-Goals

- Cross-modal linking (text entity ↔ figure entity) — follow-up story
- Audio/video transcription (separate infrastructure concern)
- HTML extractor (lower priority, defer to follow-up)
- Achieving Cognee's 38-type parity (scope is the four formats above)

## Acceptance Criteria

- [ ] Scanned PDF with 0 extractable text chars triggers Tesseract; result contains entities from the scanned content
- [ ] Architecture diagram image produces structured JSON with `nodes` and `edges` arrays, not a text description
- [ ] CSV file with 5 columns and 100 rows produces 100 `__Entity__` nodes with correct properties
- [ ] Markdown file produces nodes for each header with `CONTAINS` hierarchy edges
- [ ] All existing multimodal tests continue passing

## Open Questions

| # | Question | Owner | Status |
|---|----------|-------|--------|
| 1 | Diagram type detection: heuristic (image aspect ratio, color variance) or LLM classification call? | engineering | open |

## Context & Background

- Full technical spec: `ORACLOUS_DEEPENING_ROADMAP.md` § 10 (Multimodal Depth, pp. 678-753)
- Current impl: `app/services/pdf_extractor.py`, `app/services/vision_extractor.py`, `app/services/document_processor.py`
- New dependency: `pytesseract` + system `tesseract-ocr`
- Estimated effort: 2 weeks (OCR + diagram); 1 week each for CSV/JSON/MD
