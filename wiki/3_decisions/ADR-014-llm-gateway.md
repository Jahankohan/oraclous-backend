---
id: ADR-014
title: "L2 runtime calls LLMs exclusively via an LLM Gateway service; L3 training framework uses a pluggable adapter interface"
date: 2026-04-25
status: draft
story: ""
supersedes: "ADR-012"
superseded_by: ""
layer: harnessing
authors: [cto, solution-architect, reza]
---

# ADR-014: LLM Gateway and Training Framework Abstraction

## Status

`draft` — 2026-04-25 · awaiting Reza acceptance

## Context

ADR-012 was written when Layer 2 was defined as "framework-not-runtime" — a design in
which agent execution happened inside the customer's chosen runtime (Paperclip, Claude
Code, LangGraph) and L2 provided only graph-native tools. Under that design, "L2 does
not call the LLM directly" was correct: the runtime was the customer's, and L2 had no
execution engine of its own.

ADR-006 reversed this: Layer 2 is the Oraclous Harnessing Platform — a BSP agent
execution engine built and operated by Oraclous. The BSP Executor runs the
plan → execute → apply → check-termination loop; each super-step requires an LLM call.
"L2 does not call the LLM" is now structurally incorrect. ADR-012 must be superseded.

The L3 training framework adapter decision in ADR-012 remains valid and is carried
forward here unchanged.

## Decision

### Layer 2: LLM Gateway

**The L2 BSP Executor calls LLMs exclusively through the LLM Gateway service.**
Agents running on the Oraclous Harnessing Platform never hold LLM provider credentials.
The LLM Gateway:

1. Accepts requests from the BSP Executor (internal, not agent-facing)
2. Routes to the configured LLM provider (Anthropic, OpenAI, Azure OpenAI, Bedrock,
   Ollama, or any OpenAI-compatible endpoint)
3. Enforces per-tenant rate limits and token budgets
4. Emits LLM call audit records to the audit sidecar
5. Returns structured responses; streaming is supported via server-sent events

**Provider selection is a deployment-time configuration, not a code change.**
The gateway is provider-agnostic at its API boundary — the BSP Executor makes the
same internal call regardless of which LLM provider is configured downstream.

**External runtimes connecting via MCP are unaffected.** A Claude Code session or
Paperclip agent connecting to L2 via MCP brings its own LLM. The gateway applies
only to agents running natively on the Oraclous runtime.

### Layer 3: Training Framework Abstraction

*(Carried forward from ADR-012 — decision unchanged.)*

**Layer 3 Training Agents use a pluggable adapter interface** for training framework
calls. The interface exposes:
- `start_training_run(config: TrainingConfig) → RunHandle`
- `get_run_status(handle: RunHandle) → RunStatus`
- `cancel_run(handle: RunHandle) → None`

Each supported framework (Axolotl, TRL, Unsloth) is an adapter implementing this
interface. Framework selection is a deployment-time configuration, not a code change.
Stock configuration ships with Axolotl as the default adapter.

## Rationale

- The BSP Executor is an LLM-calling component. Agents that run inside the Oraclous
  runtime must have a provider-agnostic path to LLM calls — that is what the gateway
  provides. Without it, every provider swap is a code change.
- Centralising LLM calls in one service makes rate limiting, audit, and credential
  isolation implementable in one place. The alternative — each agent holding its own
  provider credentials — violates the Data Ownership founding principle and makes
  per-tenant token budgets impossible to enforce.
- The gateway is a service, not a library. It runs as a separate container. The BSP
  Executor calls it over a local network. This preserves layer isolation (L2 runtime
  calls L2 gateway; no L1 involvement in LLM routing).

## Alternatives Considered

| Alternative | Why Rejected |
|---|---|
| Agents hold provider credentials directly | Violates Data Ownership founding principle; makes per-tenant rate limiting impossible; credential rotation requires touching every agent config. |
| LiteLLM as the gateway | LiteLLM is a library, not a gateway service in the architectural sense. It can be used as the routing implementation *inside* the gateway service, but it is not the gateway itself. Naming it the gateway would confuse the integration surface. |
| One gateway instance per LLM provider | Agents would need to know which gateway to call based on provider. Routing logic leaks into agent code. One gateway with internal routing is the correct boundary. |
| Defer gateway to Phase 2 | The BSP Executor cannot function without LLM calls. Deferring means P0 agents cannot execute. The gateway is a P0 component alongside the BSP Executor. |

## Consequences

**Positive:**
- Swapping LLM provider is a config change (`LLM_PROVIDER=bedrock`, restart gateway).
  No agent code changes.
- Per-tenant token budgets and rate limits are enforceable at the gateway level.
- All LLM calls appear in the audit trail (gateway emits to audit sidecar).
- Agents running on the Oraclous runtime have zero LLM provider awareness.

**Negative / Trade-offs:**
- The gateway is a P0 component that must be built before agents can run. This is
  engineering investment, but unavoidable given the BSP Executor requires LLM calls.
- The gateway introduces a local network hop on every LLM call. Latency impact is
  negligible on cloud deployments (sub-millisecond local routing vs. hundreds of
  milliseconds for the LLM call itself).

**Neutral:**
- External runtimes (Claude Code, Paperclip) connecting via MCP are unaffected. They
  use their own LLM configurations. The gateway is invisible to them.

## Validation

- Deploy the Oraclous runtime with `LLM_PROVIDER=anthropic`; run an agent task; it
  completes → change to `LLM_PROVIDER=openai`; restart gateway only; same agent task
  completes → no agent code change required.
- Inspect the audit log after an agent run: every LLM call has a corresponding audit
  event from the gateway.
- Configure a per-tenant token budget in the gateway; send requests exceeding the
  budget → gateway rejects with 429 before the LLM call is made.

## Related

- ADR-006 — L2 is the Oraclous Harnessing Platform (decision that made this ADR necessary)
- ADR-012 — superseded by this ADR
- ADR-011 — Audit trail; gateway emits to the same audit sidecar
- ADR-005 — L1 is sole persistence; gateway does not write to L1 (audit sidecar does)
- [wiki/1_architecture/layer-2-harnessing-platform.md](../1_architecture/layer-2-harnessing-platform.md) — LLM Gateway component spec
