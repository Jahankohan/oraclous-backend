# Layer 2: Oraclous Harnessing Platform

**Authority:** Constrained by ADR-006. Changes require a new ADR.

---

## What Layer 2 Is

Layer 2 is the Oraclous Harnessing Platform — a graph-native agent runtime built to
run on top of the L1 Neo4j knowledge graph. It executes agents, coordinates them through
graph writes, enforces scope, handles HITL, and exposes all capabilities as MCP tools
and a REST API.

External platforms (LangGraph, CrewAI, Paperclip, Claude Code) connect via MCP (ADR-007).
No custom adapters are built. The Oraclous runtime is the first-class path.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  INTEGRATION SURFACES  (all P0 or P1)                               │
│  MCP Server (P0) · REST API (P1) · Python SDK (P1) · Skill Files   │
├───────────────────────────┬─────────────────────────────────────────┤
│  AGENT RUNTIME            │  COORDINATION LAYER                     │
│  BSP Executor (P0)        │  APOC Trigger Listener (P0)             │
│  Tool Dispatcher (P0)     │  AgentTask Queue in L1 (P0)             │
│  Context Assembly (P0)    │  Redis Streams relay (P2, scale-only)   │
├───────────────────────────┼─────────────────────────────────────────┤
│  PLATFORM SERVICES        │  SECURITY LAYER (server-side at L1 API) │
│  LLM Gateway (P0)         │  Scope Enforcer — JWT-rewrite (P0)      │
│  HITL Gate Engine (P1)    │  JWT Validator — in-process (P0)        │
│  Checkpointer → L1 (P0)   │  Audit Emitter Sidecar (P0)            │
│  Memory Store → L1 (P1)   │  Credential Resolver (P0)              │
├───────────────────────────┴─────────────────────────────────────────┤
│  LAYER 1: Neo4j Knowledge Graph                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Agent Runtime (BSP Executor)

**What it is:** The core execution engine. Implements the Bulk Synchronous Parallel
(Pregel) model: plan → execute → apply → check termination. Each super-step is one
LLM interaction. Context assembly (system prompt + injected graph context + message
history management, token budget, prompt caching segments) is part of the BSP Executor —
not a separate component.

**What it does:**
- Maintains named **channels** with typed values and reducer functions
- Nodes subscribe to channels; a node executes when its subscribed channel is updated
- Parallel nodes execute concurrently within a super-step; results are merged by the channel's reducer (deterministic — reducer is declared at channel registration, not at write time; write conflicts within a super-step are resolved by the reducer, never by last-write-wins)
- Loops are natively supported (unlike DAG-only executors)
- Every super-step writes a checkpoint to L1 (via Checkpointer)
- **Does not start accepting tasks until Audit Emitter Sidecar health check passes** (startup ordering contract — see §7)

**LLM call loop per node:**
1. Assemble context: system prompt + injected graph context + message history; apply token budget; mark cacheable segments
2. Call LLM Gateway with tool definitions
3. If response = `tool_call`: dispatch to Tool Dispatcher, inject result into context, repeat
4. If response = `content`: write output to output channel → trigger downstream nodes

**State machine (ACP-compliant, 7 states):**
```
created → in-progress → completed
                      → failed
                      → awaiting  →  in-progress (resume via POST /runs/{id})
                                  →  cancelled
                      → cancelling → cancelled
```

---

### 2. Tool Dispatcher

**What it does:**
- Static registry: tools registered at agent startup from skill file declarations, cross-validated against the agent service account's ACL grants (see §8 Skill Files — registration rejects if skill file declares a tool not in the ACL)
- Dynamic discovery (P2): embed task description → semantic search over tool vector index → load relevant schemas into context
- Schema validation (Pydantic) on all tool call arguments before dispatch; validation failure → retry with corrected args once, then fail the super-step
- Every tool call emits an OTEL trace span automatically

---

### 3. Coordination Layer (Graph-Native Event System)

**How agent wake-up works:**

```
Agent A writes :Finding node to L1
    ↓
APOC afterAsync trigger fires (non-blocking, post-commit)
One trigger per event type (not per tenant — see below)
    ↓
Trigger Cypher filters by node label + graph_id internally
Writes :AgentTask node to L1 coordination partition
    ↓  (below 500 tasks/hour: Celery worker polls AgentTask)
    ↓  (above 500 tasks/hour: relay to Redis Streams — see §ADR note)
Agent B worker claims :AgentTask, executes, writes output nodes
    ↓
APOC trigger fires again → wakes downstream agents
```

**APOC trigger design — one trigger per event type, not per tenant:**

The trigger filters `graph_id` internally. This avoids O(tenants × types) trigger
proliferation on every Neo4j write transaction.

```cypher
-- One trigger for all tenants, all Finding creations
CALL apoc.trigger.install(
  'neo4j',
  'finding-created',                         -- one global trigger, not per-tenant
  "UNWIND $createdNodes AS n
   WHERE 'Finding' IN labels(n) AND n.status = 'new'
   CREATE (:AgentTask {
     id: apoc.create.uuid(),
     graph_id: n.graph_id,                   -- scoped from the node itself
     agent_type: 'coverage-analyzer',
     trigger_node_id: n.id,
     trigger_label: 'Finding',
     status: 'pending',
     priority: 1,
     created_at: timestamp()
   })",
  {phase: 'afterAsync'}
)
```

**Coordination data model:**
```cypher
(:AgentTask {
  id: uuid,
  graph_id: string,           // tenant scoping (enforced by Scope Enforcer on reads)
  agent_type: string,         // routing key
  trigger_node_id: string,
  trigger_label: string,
  status: 'pending' | 'claimed' | 'running' | 'done' | 'failed',
  priority: int,
  created_at: epoch,
  claimed_at: epoch,
  worker_id: string,
  retry_count: int,
  dead_letter: bool           // true if trigger fired but AgentTask write failed; alerts ops
})-[:TRIGGERED_BY]->(:Finding | :Proposal | ...)
```

**APOC trigger failure handling:** If the `CREATE (:AgentTask ...)` inside the trigger
fails (e.g., transaction conflict), the trigger writes a `dead_letter: true` AgentTask
node and alerts ops. Events are never silently dropped.

**Redis Streams and ADR-005:** AgentTask nodes in L1 are the **authoritative coordination
record**. Redis Streams (when used above the scale threshold) is an ephemeral delivery
channel only — it holds no state that is not already in L1. Redis Streams is not the
store of record; L1 is. This makes Redis Streams a transport, not a second persistence
layer, and ADR-005 is not violated.

---

### 4. LLM Gateway

**What it is:** A gateway service (not a library) between the agent runtime and all LLM
providers. The runtime calls one unified API. Agents never hold provider credentials.

> **ADR-014** supersedes ADR-012 on this point. ADR-012 stated "L2 does not call the LLM
> directly" — written before L2 was confirmed as a first-party runtime. ADR-014 formalizes
> the LLM Gateway as the correct mechanism: agents call LLMs through the gateway, never
> directly, and never hold provider credentials.

**What it does:**
- Provider routing: capability-based (vision, function-calling, long-context, cost tier)
- Fallback chains: primary → fallback → fallback2, transparent to agents
- Rate limit management: per-provider token bucket, exponential backoff + jitter at gateway
- Prompt cache management: injects `cache_control` headers on static segments (system prompt); dynamic segments (graph context) are never marked cacheable
- Credential isolation: resolves provider API keys from Credential Broker per-task; credentials never reach agent code
- Observability: latency, token usage, cost, cache hit/miss per (agent_id, model, graph_id)

**Interface:**
```python
async def complete(
    messages: List[Message],
    tools: Optional[List[ToolDef]],
    model: str,
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> AsyncIterator[CompletionChunk] | CompletionResult

async def embed(texts: List[str], model: str) -> List[List[float]]

def estimate_tokens(messages: List[Message], model: str) -> int
# Note: uses provider-specific tokeniser where available; tiktoken fallback otherwise
```

**Supported providers (Phase 1):** Anthropic, OpenAI.
**Phase 2:** Mistral, Google Gemini, self-hosted (Ollama, vLLM).

---

### 5. Checkpointer

**What it does:** Persists full agent execution state after every super-step.
Enables: crash recovery, HITL pause/resume, time-travel debugging.

**Storage:** L1 (Neo4j). Checkpoint reads and writes go through the same L1 API
security middleware (Scope Enforcer, JWT Validator) as all other L1 operations.
The Checkpointer service account has WRITE access only to `:Checkpoint` nodes scoped
to its authorised `graph_id` — enforced at the database ACL level, not just at the
application layer.

```cypher
(:Checkpoint {
  id: uuid,
  thread_id: string,         // agent run identifier
  graph_id: string,          // tenant (enforced by Scope Enforcer)
  step: int,                 // super-step number
  channel_values: bytes,     // MsgPack-serialized; max ~50MB (Neo4j 64MB limit)
  created_at: epoch,
  ttl_expires_at: epoch      // set to created_at + 30 days; GC job cleans expired checkpoints
})
```

**Crash recovery protocol:** A Celery beat task runs every 60s. It queries for
`:AgentRun` nodes with `status = 'in-progress'` and `last_heartbeat < now - 120s`.
These are orphaned runs. The beat task sets their status to `failed` and emits an
alert. Resumption from the last valid checkpoint is manual (operator triggers via
`POST /runs/{run_id}` with `resume_from_checkpoint: true`).

**Channel write-conflict semantics:** When two parallel nodes write to the same channel
in the same super-step, the channel's declared reducer function resolves the conflict.
Reducers are declared at channel registration time (e.g., `append_reducer` for message
lists, `latest_wins_reducer` for scalar values). Undefined reducer = registration error
at agent startup.

---

### 6. HITL Gate Engine

**Dependencies:** Checkpointer (P0) must be operational. `POST /runs/{run_id}` REST
endpoint must be deployed. HITL is not testable or deliverable without both.

**What it does:** Pauses agent execution at declared interrupt points, stores the
pending decision in L1, notifies the reviewer, and resumes from checkpoint on response.

**HITL review record in L1:**
```cypher
(:HITLReview {
  id: uuid,
  run_id: string,           // links to the paused AgentRun
  graph_id: string,
  agent_type: string,
  action_proposed: string,  // tool name + serialised args
  reviewer_role: string,    // role required to resolve
  status: 'pending' | 'resolved' | 'timed_out',
  created_at: epoch,
  timeout_at: epoch,        // created_at + configurable timeout (default 24h)
  resolution: string        // 'approved' | 'modified' | 'rejected' | 'supplemented'
})-[:REQUIRES_REVIEW_ON]->(:AgentRun)
```

**Reviewer notification:** On `status = 'awaiting'`, the platform emits a webhook to
the configured `hitl_webhook_url` for the tenant (stored in L1 as a graph node on the
`:Graph`). If no webhook is configured, the pending review is surfaced via
`GET /graphs/{graph_id}/reviews/pending`.

**Timeout handling:** The Celery beat task checks for `:HITLReview` nodes where
`timeout_at < now AND status = 'pending'`. On timeout: sets `status = 'timed_out'`,
sets the parent run to `failed`, emits an alert. No auto-approval. Timeout window is
configurable per tenant, default 24 hours.

**HITL Policy Engine:** evaluates `(tool_name, args, agent_context)` → `allow | deny | interrupt`.
Policy nodes in L1:
```cypher
(:HITLPolicy {
  graph_id: string,
  agent_type: string,       // '*' = applies to all agents in tenant
  tool_name: string,        // '*' = applies to all tools
  condition: string,        // Cypher expression evaluated against {tool_name, args, agent_context}
  action: string            // 'allow' | 'deny' | 'interrupt'
})
```
Policy evaluation is a Cypher query against these nodes at each tool call. Policies are
indexed by (graph_id, agent_type, tool_name) for performance.

---

### 7. Security Layer (Server-Side at L1 API Boundary)

**Invariant:** All enforcement is server-side. No security property depends on the
caller using the SDK, MCP server, or any specific client. REST, MCP, direct Bolt, and
native runtime calls all traverse the same middleware stack.

**Components:**

**JWT Validator (in-process):**
Validates JWT signature, expiry, and issuer in-process (not via remote auth-service
call — removes per-request remote call latency). Extracts `tenant_id`, `graph_id`,
`agent_id`, `scopes[]` from token claims. The existing remote-call approach in
`auth_service.py` is replaced by in-process validation using the auth-service's public
key (fetched at startup + rotated on JWKS endpoint change).

**Scope Enforcer (Cypher rewrite — not parameter rejection):**
The Scope Enforcer does **not** reject queries that supply a `graph_id` parameter.
It **unconditionally rewrites** every Cypher query to inject the JWT-bound `graph_id`
as a server-controlled parameter before execution. Agent-supplied `graph_id` values in
the parameter bag are ignored and overwritten. This eliminates tenant-ID forgery by
construction — an agent cannot supply a `graph_id` that differs from its JWT claim
because the enforcer overwrites it before the query reaches Neo4j.

```
# What agent submits:
MATCH (f:Finding {graph_id: $gid}) RETURN f   -- $gid = "attacker-tenant"

# What Scope Enforcer executes:
MATCH (f:Finding {graph_id: $__enforced_graph_id}) RETURN f
                               -- $__enforced_graph_id bound to JWT claim, not agent param
```

**Credential Resolver:**
Agents never hold LLM provider keys or object store credentials. Per-task short-lived
credentials are resolved from the Credential Broker service
(`oraclous-data-studio/credential-broker-service/`) at task start. The broker is
already implemented and enforces `graph_id` scoping on all issued credentials.

**Audit Emitter Sidecar — startup ordering contract:**
1. Sidecar starts and registers with a health endpoint (`GET /audit-sidecar/health`)
2. BSP Executor performs a health check against the sidecar endpoint before accepting
   any task. Health check runs every 5s; executor rejects new tasks if sidecar is
   unhealthy for > 10s.
3. Event channel: the Sidecar listens on an APOC `afterAsync` trigger on the
   agent-writable L1 partition (separate from the coordination trigger). Every write
   to a non-audit L1 partition emits an event; the sidecar appends an immutable,
   hash-chained audit record to the audit partition.
4. If the sidecar becomes unhealthy mid-run: the BSP Executor's health check fires;
   the executor sets the current run to `status = 'failed'` with
   `failure_reason = 'audit_sidecar_unavailable'` and does not resume until sidecar
   health is restored.

---

### 8. Integration Surfaces

#### MCP Server (P0 — coequal with the REST API, not a later addition)

ADR-007 declares MCP first-class. The MCP Server is P0 alongside the BSP Executor.
The P0 runtime does not run without an MCP surface.

- Exposes all L2 capabilities as MCP tools (ADR-007)
- Both client and server: L2 agents call external MCP tools; external runtimes call L2 via MCP
- Implements MCP stdio and HTTP+SSE transports
- Every L2 REST endpoint has an MCP tool equivalent. No capability is MCP-only or REST-only.
- MCP requests pass through the same L1 API security middleware as REST requests

#### REST API (P1)

ACP-compliant lifecycle endpoints:
```
POST   /runs                      Create run (sync | async | stream)
GET    /runs/{run_id}             Poll status
POST   /runs/{run_id}             Resume awaiting run (HITL response body)
POST   /runs/{run_id}/cancel      Cancel (202 Accepted; async)
GET    /agents                    List registered agent types
GET    /agents/{agent_type}       Agent card (capabilities, ACL grants, skill file)
POST   /tasks                     Manually enqueue an AgentTask
GET    /tasks/{task_id}           Task status
GET    /graphs/{graph_id}/runs    All runs for a tenant
GET    /graphs/{graph_id}/reviews/pending   Pending HITL reviews
POST   /graphs/{graph_id}/hitl-policies     Register/update HITL policy node
```

#### Memory Store — L1 node schema and API (P1)

`ctx.memory` is a cross-thread, cross-run key-value store backed by L1. Memory nodes
are scoped by `(agent_id, graph_id)` and are bitemporal (all writes carry `valid_time`
and `transaction_time` per Commitment 6).

```cypher
(:AgentMemory {
  id: uuid,
  agent_id: string,
  graph_id: string,         // Scope Enforcer enforces this
  key: string,
  value: string,            // JSON-serialised
  valid_time: epoch,        // when this value became true
  transaction_time: epoch   // when it was written
})
```

API surface (`ctx.memory`):
```python
await ctx.memory.set(key, value)           # writes new bitemporal record
await ctx.memory.get(key)                  # reads latest valid_time record
await ctx.memory.get_at(key, valid_time)   # time-travel read
await ctx.memory.delete(key)               # logical delete (sets valid_time end)
await ctx.memory.list()                    # all keys in (agent_id, graph_id) namespace
```

Memory reads by default are scoped to the calling agent's `(agent_id, graph_id)`.
An agent cannot read another agent's memory unless the owning agent has written a
shared memory record with `scope = 'tenant'` (readable by all agents in the same
`graph_id`).

#### Python SDK (`oraclous`) — P1

```python
from oraclous import agent, tool, AgentContext

@agent(
    name="coverage-analyzer",
    reads=["Finding", "KGNode"],
    writes=["CoverageGap"],
    tools=["graph.query", "graph.write", "llm.complete"],
)
async def coverage_analyzer(ctx: AgentContext) -> None:
    findings = await ctx.graph.query(
        "MATCH (f:Finding {status: 'new'}) RETURN f"
        # graph_id is injected server-side by Scope Enforcer; do not pass it as param
    )
    for finding in findings:
        gap = await ctx.llm.complete(analyze_prompt(finding))
        await ctx.graph.write(CoverageGap(
            source_finding_id=finding.id,
            gap_type=gap.type,
            confidence=gap.confidence
        ))
```

**Context object:**
- `ctx.graph` — `GraphClient`: bitemporal write API; `graph_id` enforced server-side
- `ctx.llm` — LLM Gateway client; no provider credentials in context
- `ctx.hitl` — `await ctx.hitl.interrupt(action, payload, reviewer_role)`
- `ctx.scope` — parent scope for inheritance enforcement
- `ctx.memory` — cross-thread key-value store (see Memory Store above)
- `ctx.delegate` — `await ctx.delegate("agent-name", input=payload)` — spawns child agent with inherited scope (P2; **L3 hierarchical agents cannot begin until this is implemented**)
- `ctx.stream` — `ctx.stream.emit(partial)` — surface intermediate results
- `ctx.graph.query_at(valid_time=dt)` — bitemporal time-travel query (P2; requires L1 bitemporal infrastructure to be complete)

**Lifecycle hooks:**
```python
@on_pause    # called synchronously before checkpoint write; exception aborts the pause
@on_resume   # called after checkpoint restore, before first super-step
@on_cancel   # cleanup; runs in a new transaction after cancellation is confirmed
@on_error    # receives (exception, retry_count); returns 'retry' | 'fail' | 'escalate_hitl'
```

#### Skill Files (`.oraclous.md`) — P1

Declarative agent definition. One file per agent. The runtime reads skill files to
register agents, grant tool access, declare scope, and configure HITL policies.

**Registration validation:** At registration time, the runtime cross-checks the skill
file's `reads`, `writes`, and `tools` declarations against the agent service account's
ACL grants (ADR-004). If the skill file declares wider scope than the ACL grants,
registration is rejected. Scope enforcement at runtime uses the ACL-granted scope, not
the skill-file-stated scope. Skill files are declarations of intent; ACLs are the
enforcement boundary.

**Format:**
```yaml
---
name: coverage-analyzer
version: 1.0.0
reads: [Finding, KGNode]
writes: [CoverageGap]
tools:
  - graph.query
  - graph.write
  - llm.complete
scope: inherit          # child of parent; cannot exceed parent ACL grants
hitl:
  - tool: graph.write
    when: confidence < 0.7
    reviewer_role: data-curator
---
## Goal
Analyze new Finding nodes and produce CoverageGap nodes.
## Constraints
- Do not write CoverageGap nodes with confidence < 0.4
- Rate limit: max 100 graph writes per run
- Timeout: 300s per task
```

Skill file format versioning is governed by semver (ADR to be written — open item).

#### TypeScript SDK (`@oraclous/sdk`) — P2

Same surface as Python SDK. After Python SDK is stable and used by at least one L3 agent.

---

### 9. Observability (P2 for agent spans; P0 baseline from existing OTEL setup)

Existing OTEL setup in `oraclous-data-studio/` (Jaeger/Tempo export) covers L1 graph
operation traces. That setup is preserved and extended — not replaced.

**L2-specific spans added when components are built:**
- Per-tool-call span: tool name, args hash, duration, result status
- Per-LLM-call span: model, token count, cost estimate, cache hit/miss, latency
- Per-graph-write span: node type, `graph_id`, write duration
- Per-super-step span: step number, parallel node count, total duration
- HITL interrupt event: run_id, reviewer_role, timeout_at

Agent observability tooling (Arize Phoenix or equivalent) is evaluated as an ADR decision
once L2 produces agent traces. The existing Jaeger backend is the default until then.

---

## Implementation Priority

| Component | Priority | Notes |
|-----------|----------|-------|
| L1 API security middleware: JWT Validator (in-process) + Scope Enforcer (Cypher rewrite) | P0 | Replaces existing remote JWT call + string-injection in multi_tenant_components.py |
| Audit Emitter Sidecar + health check contract with BSP Executor | P0 | BSP Executor must not start without sidecar health confirmed |
| Neo4j Checkpointer | P0 | Needed for any stateful execution |
| BSP Executor (core loop + context assembly) | P0 | The runtime |
| Tool Dispatcher (static registry + Pydantic validation) | P0 | Required for tool-using agents |
| APOC Trigger Listener + AgentTask queue (single parameterized triggers per type) | P0 | Graph-native coordination |
| LLM Gateway (Anthropic + OpenAI, Phase 1) | P0 | Agents cannot call LLMs without it |
| MCP Server (L2 capabilities exposed as MCP tools) | P0 | ADR-007 requires MCP coequal with runtime |
| Python SDK (`@agent`, `@tool`, `AgentContext`) | P1 | L3 agent authors need this |
| REST API (lifecycle endpoints + HITL resume) | P1 | HITL requires both Checkpointer (P0) and this |
| HITL Gate Engine + HITLReview L1 schema + webhook notification | P1 | Depends on: Checkpointer + REST resume endpoint |
| Skill File Parser + Registry (with ACL cross-validation at registration) | P1 | L3 agents are defined by skill files |
| Memory Store (AgentMemory L1 schema + ctx.memory API) | P1 | Required for cross-run agent state |
| L3 hierarchical agent work gate | — | **Cannot begin until ctx.delegate (P2) is implemented and tested** |
| TypeScript SDK | P2 | After Python SDK stable |
| LLM Gateway multi-provider (Mistral, Gemini, Ollama, vLLM) | P2 | Start with Anthropic + OpenAI |
| Dynamic Tool Discovery (vector index) | P2 | Start with static registration |
| Redis Streams relay (above 500 tasks/hour scale threshold) | P2 | AgentTask polling is sufficient below threshold |
| ctx.delegate — sub-agent delegation | P2 | Gate for L3 hierarchical agents |
| ctx.graph.query_at — bitemporal time-travel | P2 | Requires L1 bitemporal infrastructure to be complete |
| Observability (L2 agent spans, OTEL extension) | P2 | Basic logging sufficient at P0/P1 |
| Checkpoint GC job (TTL cleanup) | P2 | Run as Celery beat task |

---

## Open ADRs Required Before Implementation

1. **~~ADR-014: LLM Gateway~~** — written (2026-04-25). Supersedes ADR-012.
2. **L2 API versioning policy** — skill file format semver, SDK deprecation windows.
3. **Redis Streams as L2 infrastructure (scale threshold ADR)** — declares Redis Streams as a required infrastructure component above 500 tasks/hour, added to docker-compose + Helm. ADR-005 compliance is maintained because AgentTask L1 nodes remain authoritative; Redis Streams is ephemeral relay.
