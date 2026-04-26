---
id: ADR-012
title: "Layer 2 is LLM-provider-agnostic by design; Layer 3 training framework abstraction uses a pluggable adapter interface"
date: 2026-04-25
status: superseded
story: ""
supersedes: ""
superseded_by: "ADR-014"
layer: cross-cutting
authors: [cto, solution-architect]
---

# ADR-012: Provider Abstraction (LLM and Training Framework)

## Status

`draft` — 2026-04-25 · awaiting Reza acceptance

## Context

The SA raised that the architecture does not document how agents (L2) and the FTOps
loop (L3) remain provider-agnostic. Specifically: (1) Layer 2's LLM provider coupling —
if agents call the LLM directly, swapping providers requires changing agent code;
(2) Layer 3's training framework coupling — the FTOps loop references specific frameworks
(Axolotl, TRL, Unsloth) without documenting the abstraction layer that makes them
interchangeable.

The CTO found that Layer 2 explicitly states "Not opinionated about the LLM" and "Agents
execute in the customer's runtime of choice" (source: layer-2-graph-native-agent-platform.md),
but the mechanism is not specified. The training framework gap is a genuine documentation
gap: the FTOps documents reference specific frameworks without specifying the interface
that makes them swappable.

## Decision

### Layer 2: LLM Provider Abstraction

**Layer 2 does not call the LLM directly.** Agent execution happens in the customer's
chosen runtime (Paperclip, Claude Code, CrewAI, LangGraph, etc.). The runtime is
responsible for LLM calls. Layer 2 provides graph-native integration capabilities via
MCP tools and platform adapters — none of which are LLM-provider-specific.

The `.oraclous.md` skill file format is provider-agnostic: it describes tools and
graph access patterns, not which LLM to use. A skill file used in a Paperclip runtime
and the same skill file used in a Claude Code session produce the same graph operations.

**LLM provider is a runtime configuration concern, not a Layer 2 concern.**

### Layer 3: Training Framework Abstraction

**Layer 3 Training Agents use a pluggable adapter interface** for training framework
calls. The interface exposes:
- `start_training_run(config: TrainingConfig) → RunHandle`
- `get_run_status(handle: RunHandle) → RunStatus`
- `cancel_run(handle: RunHandle) → None`

Each supported framework (Axolotl, TRL, Unsloth) is an adapter implementing this
interface. The Training Agent calls the interface; the adapter translates to the
framework's native API. Framework selection is a deployment-time configuration,
not a code change.

**Stock configuration ships with Axolotl as the default adapter.** Alternative adapters
(TRL, Unsloth) are included in the reference implementation and selectable via
environment configuration.

## Rationale

- The "L2 does not call the LLM" design is already implicit in the framework-not-runtime
  decision (ADR-006). This ADR makes it explicit to close the documentation gap.
- The adapter pattern for L3 training frameworks follows the same logic as the runtime
  abstraction at L2: Oraclous should complement customer tool choices, not constrain them.
- Documenting the adapter interface before implementation prevents the pattern of
  writing framework-specific code and "abstracting later."

## Alternatives Considered

| Alternative | Why Rejected |
|---|---|
| L2 provides a thin LLM client with provider adapters (like LiteLLM) | Adds a dependency on LLM routing infrastructure; duplicates what every major runtime already provides. The framework-not-runtime decision (ADR-006) makes this unnecessary. |
| Single training framework (Axolotl only, others via PRs) | Violates the No Vendor Lock-in founding principle. Customer fine-tuning environments often have an existing framework; Oraclous must integrate, not require migration. |
| Training framework abstraction deferred to post-Phase 1 | The first Training Agent implementation will encode framework assumptions into its logic. Retrofitting an adapter interface after the fact is substantially harder than building to it from the start. |

## Consequences

**Positive:**
- Customers who run Paperclip, Claude Code, or CrewAI as their agent runtime can adopt
  L2 without switching runtimes or LLM providers.
- FTOps Training Agents can target any framework with a conforming adapter; swapping
  Axolotl for TRL is a config change, not a code change.

**Negative / Trade-offs:**
- The adapter interface must be kept stable. A breaking change to `TrainingConfig` or
  `RunHandle` forces updates to all adapters. Interface versioning is required from Day 1.
- For L2: customers who want a runtime bundled with Oraclous must bring their own.
  This is by design (ADR-006), but it means Oraclous must document recommended runtimes
  clearly.

**Neutral:**
- This ADR does not specify LLM provider selection within the customer's runtime. That
  is the customer's configuration responsibility. Oraclous documentation should include
  quickstart guides for the most common runtimes (Paperclip, Claude Code, LangGraph).

## Validation

- Run an L2 skill file via Claude Code runtime and via a LangGraph runtime against the
  same graph → both produce identical graph writes. No L2 code changes required.
- Configure Training Agent to use TRL adapter; run a training job → completes without
  modifying Training Agent code.
- Change `TRAINING_FRAMEWORK=unsloth` in deployment config; restart Training Agent →
  next training run uses Unsloth without code change.

## Related

- ADR-006 — L2 is framework-not-runtime (LLM provider abstraction follows from this)
- ADR-008 — Three deployment models (each deployment model uses any MCP-compatible runtime)
- Concern LLM-PROVIDER-COUPLING (closed by this ADR)
