---
id: STORY-011
title: "L2 P0: LLM Gateway service — provider-agnostic LLM routing for the BSP Executor"
type: architecture
layer: harnessing
reporter: reza
status: ready
priority: critical
created: 2026-04-26
updated: 2026-04-26
wiki_refs: ["layer-2-harnessing-platform"]
tasks: []
decisions: ["ADR-014"]
---

# STORY-011: L2 LLM Gateway (P0)

## Summary

The LLM Gateway is a separate service (not a library) that sits between the BSP Executor
and all LLM providers. Agents never hold provider credentials. The gateway handles
provider routing, fallback chains, per-tenant rate limits/token budgets, prompt cache
management, and audit emission. Without it, the BSP Executor has nowhere to send LLM
calls. This must be built in parallel with STORY-010.

## Problem Statement

- No LLM Gateway service exists; `llm_service.py` calls LangChain providers directly
- Per-tenant token budgets are impossible without a gateway
- LLM provider swap (Anthropic → Bedrock) currently requires code changes
- All LLM calls are invisible to the audit trail

## Goals

- [ ] Build LLM Gateway as a standalone FastAPI service (separate container, not part of knowledge-graph-builder)
- [ ] Accept requests from BSP Executor via internal REST (not agent-facing)
- [ ] Implement provider routing: Anthropic, OpenAI, Azure OpenAI, Bedrock, Ollama (OpenAI-compatible fallback)
- [ ] Implement fallback chains: primary → fallback → fallback2, transparent to caller
- [ ] Implement per-tenant rate limits and token budget enforcement (reject with 429 before LLM call)
- [ ] Implement prompt cache management: `cache_control` on static segments (system prompt); no caching on graph context
- [ ] Emit LLM call audit records to audit sidecar on every call
- [ ] Support SSE streaming pass-through to BSP Executor

## Non-Goals

- Agent-facing LLM API (agents do not call the gateway directly; only BSP Executor does)
- LiteLLM as a replacement for the gateway (LiteLLM may be used internally as routing implementation)
- External runtime LLM calls (Claude Code / Paperclip bring their own LLM; gateway is invisible to them)

## Acceptance Criteria

- [ ] `LLM_PROVIDER=anthropic` → `LLM_PROVIDER=openai` config change, gateway restart only → same BSP Executor request routes to new provider; no Executor code change
- [ ] Request exceeding per-tenant token budget returns 429 before any LLM API call is made
- [ ] Every LLM call produces an audit event in the sidecar (verifiable via L1 AuditEvent nodes)
- [ ] Primary provider failure triggers fallback chain transparently; BSP Executor does not know which provider handled the request
- [ ] Streaming response: first token arrives at Executor within 500ms of gateway receiving request (on same machine)

## Open Questions

| # | Question | Owner | Status |
|---|----------|-------|--------|
| 1 | LiteLLM as internal routing layer vs raw provider SDKs? | engineering | open |
| 2 | Token budget: per-tenant per-month or per-tenant per-day? | reza | open |

## Context & Background

- Full technical spec: `wiki/1_architecture/layer-2-harnessing-platform.md` § LLM Gateway
- ADR-014: LLM Gateway supersedes ADR-012's "L2 does not call LLMs directly"
- Existing problem: `llm_service.py` and `evaluation_service.py` call providers directly — these must be migrated to LLM Gateway once it exists
- Estimated effort: 1-2 weeks (MVP with 2 providers + fallback)
