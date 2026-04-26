## CTO Evidence: Layer 2 Harnessing Platform Spec (2026-04-25)

**Source spec:** `wiki/1_architecture/layer-2-harnessing-platform.md`
**Governing ADR:** ADR-006
**Date of review:** 2026-04-25

---

### 1. Agent Runtime (BSP Executor)

Existing code: Nothing in `oraclous-data-studio/` implements a BSP/Pregel execution model. The closest thing is `app/services/task_executor.py`, which is a generic `AsyncTaskExecutor` that wraps arbitrary async functions in a fresh Celery event loop. It has no concept of channels, super-steps, node subscriptions, or deterministic merge. `app/orchestrators/graph_orchestrator.py` exists but is a RAG query orchestrator (search mode routing), not an agent execution engine.

KB evidence: ADR-006 formally decides that L2 is a graph-native agent runtime and that the BSP model maps naturally to the Neo4j substrate. The graphify index confirms Layer 2 as a god node with 9 connections (`raw-knowledge/docs/platform-promise/02-architecture/layer-2-graph-native-agent-platform.md`). Architectural Commitment 8 ("Every Agent Action Is a Graph Write") is extracted from `raw-knowledge/docs/platform-promise/12-architectural-commitments/README.md` and is load-bearing — the BSP loop's channel writes must go to L1.

Gaps:
- The BSP Executor is entirely unbuilt. No code, no skeleton, no prototype.
- The spec defines the ACP-compliant state machine (`created → in-progress → awaiting → ...`) but no ADR formalises the ACP compliance requirement or the version of ACP being targeted. If ACP evolves, the spec has no change-management hook.
- The spec does not specify how the channel reducer functions handle write conflicts when two parallel nodes write to the same channel in the same super-step. This is a correctness gap, not a style gap — deterministic merge semantics need to be specified before implementation starts.
- The per-super-step checkpoint (written to L1 after every LLM interaction) will be high-frequency. The spec defers write-batching as "an implementation concern," but given ADR-005's decision that L1 is the sole persistence layer, write latency under concurrent agent runs is an unsolved throughput question that no document addresses.

---

### 2. Tool Dispatcher

Existing code: Nothing in `oraclous-data-studio/` implements a Tool Dispatcher as described. The existing `LLMService` (`app/services/llm_service.py`) calls LangChain-wrapped LLM providers directly for graph transformation (KG extraction), not for agent tool dispatch. No static tool registry, no dynamic tool discovery via vector index, no schema validation layer before dispatch, no automatic trace span per tool call.

KB evidence: ADR-009 establishes that L2 exposes capabilities as MCP tools and that L3 agents consume them. The graphify index confirms `Graph Primitives as Agent Tools (Traversal, Analysis, Prediction, Write)` as a node exposed by Layer 2. The Architectural Commitments doc lists "Provenance Everywhere" — every tool call must have a traceable source path.

Gaps:
- Dynamic tool discovery (embed task description → semantic search → load relevant schemas) requires a vector index over tool schemas. The spec lists this as P2. No document specifies which Neo4j index holds the tool embeddings, how tool schemas are versioned in that index, or what happens when a tool schema changes after embeddings are cached.
- The spec says "Schema validation on all tool call arguments before dispatch" but does not specify the schema format (JSON Schema? Pydantic? something else). The Python SDK example uses Pydantic (`GraphQueryInput`), but the spec does not confirm Pydantic as the canonical schema layer for tool definitions.
- No document addresses tool call retry policy within a single super-step (e.g., schema validation fails → retry vs. fail the step vs. escalate to HITL).

---

### 3. Coordination Layer (Graph-Native Event System)

Existing code: The codebase uses Celery for background jobs (`app/services/background_jobs.py`, `app/tasks/community_tasks.py`, `app/tasks/ontology_tasks.py`). These are one-way ingestion and analysis tasks, not agent coordination. APOC is present in the codebase only for entity resolution utilities (`app/components/entity_resolver.py`) — `apoc.create.relationship`, `apoc.text.levenshteinSimilarity`, and `apoc.cypher.doIt`. There are no APOC trigger registrations, no `afterAsync` trigger patterns, no AgentTask node creation logic, and no Redis Streams consumer code. The `app/services/task_database.py` file exists but is for tracking Celery task state in PostgreSQL, not for graph-native agent coordination.

KB evidence: ADR-006 defines graph-native coordination as a core L2 capability. ADR-005 mandates that coordination state (agent status, job queues, task assignments) be stored as graph nodes in L1. The graphify index lists `Every Agent Action Is a Graph Write` as an architectural commitment.

Gaps:
- The spec's APOC trigger registration example uses `CALL apoc.trigger.install(...)` with `graph_id` interpolated into the trigger name (`'finding-created-{graph_id}'`). This creates one APOC trigger per tenant per trigger type. No document addresses: (a) the maximum number of registered APOC triggers before Neo4j performance degrades, (b) the lifecycle management of these triggers when a tenant is deleted or suspended, (c) the APOC version compatibility matrix for `apoc.trigger.install` vs. the older `apoc.trigger.add` API.
- The scale threshold (500 tasks/hour → Redis Streams) is asserted without a measurement basis. No document records how this number was derived. This is a gap in the open ADR item flagged in the spec itself.
- The spec states Redis is "already in oraclous-data-studio stack." This is accurate for Celery's broker, but the spec proposes Redis Streams as an additional use (event relay), which is a different Redis data structure with different consumer group semantics. No ADR distinguishes the two uses, and the open ADR item in the spec acknowledges Redis Streams must be declared as a required infrastructure component. This ADR has not been drafted.
- No document addresses what happens if the APOC trigger fires but the AgentTask write fails (e.g., Neo4j transaction conflict). A partial trigger execution would produce no AgentTask and silently lose the coordination event. The spec has no dead-letter or retry mechanism described for this failure mode.

---

### 4. LLM Gateway

Existing code: A direct contradiction exists here. The spec defines an LLM Gateway as "a gateway service (not a library)" with a unified `async def complete(...)` interface — agents never hold provider API keys and never call providers directly. What is built in `oraclous-data-studio/` is the opposite: `app/services/llm_service.py` initializes LangChain provider clients directly (`ChatOpenAI`, `ChatAnthropic`, `ChatGoogleGenerativeAI`) inside the knowledge-graph-builder service. Credentials are retrieved from the credential broker per call, but the LLM client is constructed inline in the service, not routed through a separate gateway process. `app/services/evaluation_service.py` also constructs its own `ChatOpenAI` and `OpenAIEmbeddings` directly using LangChain.

KB evidence: ADR-012 decides that "Layer 2 does not call the LLM directly" and that "LLM provider is a runtime configuration concern, not a Layer 2 concern." However, that ADR was written in the context of L2 being a framework-not-runtime — the customer's chosen runtime (Paperclip, Claude Code, LangGraph) handles LLM calls. The new spec (post-ADR-006 pivot to Oraclous as its own runtime) introduces an LLM Gateway as a first-party L2 component. This creates a tension: ADR-012 says L2 does not call LLMs directly, but the runtime spec requires a gateway so the BSP Executor can call LLMs on behalf of agents.

Gaps:
- ADR-012 needs to be reconciled with the new runtime model. As written, ADR-012 says the runtime handles LLM calls — but the Oraclous runtime IS now L2. The LLM Gateway resolves this, but the ADR contradiction is unresolved and no new ADR supersedes ADR-012 on this point.
- The existing `llm_service.py` uses LangChain wrappers for all providers. The spec's gateway interface (`async def complete(messages, tools, model, temperature, max_tokens, stream)`) is a raw completion API — not a LangChain-wrapped one. If the gateway is built, `llm_service.py` becomes a duplicate path. No document specifies the migration plan or whether `llm_service.py` is deprecated once the gateway exists.
- The spec defines `estimate_tokens(messages, model)` as a synchronous method on the gateway interface. Token counting for models like Claude and Gemini requires provider-specific tokenisers, some of which are not publicly available or require API calls. No document addresses how the gateway implements accurate token estimation for non-OpenAI models.
- Prompt cache management (`inject cache_control headers on cacheable segments, track hit rate`) is listed as a gateway responsibility. No document specifies which message segments are marked cacheable by default, what the cache key strategy is, or how the gateway handles cache invalidation when an agent's system prompt changes mid-run.

---

### 5. Checkpointer

Existing code: The codebase has a `checkpoint_version_id` column in `app/models/graph.py` (a PostgreSQL SQLAlchemy model) and a `create_checkpoint: bool` field in `app/schemas/graph_schemas.py`. These are versioning checkpoints for graph snapshots (implemented in `app/services/versioning_service.py` and `app/services/snapshot_service.py`), not agent execution state checkpoints. They are a different concept using the same word. The `versioning_service.py` implements zero-copy snapshots anchored by `captured_at` timestamps — this is L1 graph versioning, not L2 agent state persistence.

KB evidence: ADR-005 mandates that L2's memory and state writes go to L1. The spec's Checkpointer design (`:Checkpoint` node with `thread_id`, `step`, `channel_values` as MsgPack bytes) is consistent with ADR-005. No existing ADR or graphify article describes a custom Neo4j checkpointer implementation. The LangGraph project has a `neo4j-checkpoint` community implementation; the spec does not reference it or state whether it was evaluated.

Gaps:
- `channel_values: bytes` stored as MsgPack in a Neo4j node property raises a practical constraint: Neo4j has a default property size limit of 64MB for byte arrays. For a large agent with many channels and long message histories, this limit could be hit. No document addresses this.
- The spec says "crash recovery" is enabled by the checkpointer, but does not specify the recovery protocol: who detects that a worker died mid-super-step, how the `in-progress` run is identified as orphaned, what the timeout threshold is, and who triggers the resume from the last checkpoint. This is a liveness gap.
- The `:Checkpoint` node schema does not include a `ttl` or `expires_at` field. Completed runs will accumulate checkpoint nodes in L1 indefinitely. No document addresses checkpoint garbage collection.

---

### 6. HITL Gate Engine

Existing code: Nothing in `oraclous-data-studio/` implements HITL. The existing `app/api/v1/endpoints/webhooks.py` exists but handles external data source webhooks (connector callbacks), not human-in-the-loop review flows.

KB evidence: The graphify index confirms "HITL Gates Per-Agent" as a concept extracted from `raw-knowledge/docs/platform-promise/04-agent-team/README.md`. ADR-006 lists "HITL gates via checkpoint pause/resume" as a core L2 capability. ADR-010 acknowledges that full per-operation permission enforcement (the strongest defense against confused-deputy attacks) requires Phase 3; HITL is part of the interim mitigation strategy.

Gaps:
- The spec defines four interrupt types (Approve / Modify / Reject with feedback / Supplement) but does not specify the data model for the HITL review record in L1. Specifically: where is the pending decision stored before the human responds? The spec says "the runtime writes a checkpoint with `status = 'awaiting'`" — but the checkpoint only stores agent execution state, not the human-facing description of what decision is needed (`what action is proposed`, `what reviewer_role is required`, `what the deadline is`). A separate `:HITLReview` node or equivalent is implied but not specified.
- The HITL Policy Engine (`evaluate (tool_name, args, agent_context) → allow | deny | interrupt`) is described as configurable per tool, per agent type, per graph_id, stored as graph nodes. No document specifies the Cypher schema for these policy nodes or how policy evaluation is implemented (rule engine? sequential scan? index?). At high throughput, a policy scan on every tool call could be a hot path.
- The spec does not specify reviewer notification. When an agent writes a checkpoint with `status = 'awaiting'`, how does the human reviewer know there is a pending decision? No push notification, webhook, or polling endpoint is described beyond the REST API's `GET /runs/{run_id}` endpoint, which requires the reviewer to already know the run_id.
- Timeout handling for HITL is absent from the spec. If a reviewer does not respond within a configurable window, the spec does not state whether the run should be cancelled, the decision should be auto-approved, or the interrupt should escalate.

---

### 7. Security Layer (Server-Side at L1 API Boundary)

Existing code: Partial implementation exists. JWT validation is performed by delegating to the auth-service via HTTP (`app/services/auth_service.py`: `GET {AUTH_SERVICE_URL}/me`). This is a remote validation call per request, not an in-process JWT signature check — adding latency to every authenticated request. ReBAC is implemented in `app/services/rebac_service.py` (Phase A: `CAN_ACCESS` model; Phase B: ORA-48 `HAS_ROLE/HAS_PERMISSION/INHERITS_FROM` model). Agent service accounts with scoped credentials exist in `app/services/service_account_service.py`. The credential broker service (`oraclous-data-studio/credential-broker-service/`) is implemented and called from the LLM service. OpenTelemetry is instrumented (`app/core/telemetry.py`). The graph_id filter is applied in `app/components/multi_tenant_components.py` and `app/services/retriever_factory.py` via a string-injection approach (`_inject_graph_id_filter`).

KB evidence: The graphify index explicitly flags "Confirmed Vulnerability: String Interpolation in multi_tenant_components.py" (source: `raw-knowledge/SECURITY_THREAT_MODEL.md`). ADR-011 mandates a structurally separate audit log partition with an audit-writer sidecar. ADR-010 establishes scope inheritance as a platform-enforced invariant. The graphify community "Access Control & Multi-Tenancy" references "APOC Procedure Abuse Risk" as a known threat (source: `raw-knowledge/docs/platform-promise/15-security/graph-and-query-security.md`).

Gaps:
- The **Scope Enforcer** as described in the spec (injects mandatory `graph_id` filter into all Cypher queries server-side, rejects queries missing `graph_id`) does not exist as middleware. The current implementation injects `graph_id` as a parameter or string filter in individual service methods, not at a middleware layer. A query that bypasses `multi_tenant_components.py` (e.g., a raw Cypher call in a new service) would not be caught. This contradicts the spec's "Critical: All enforcement is server-side."
- The **string interpolation vulnerability** in `multi_tenant_components.py` is flagged in the security threat model (`raw-knowledge/SECURITY_THREAT_MODEL.md`). The spec describes parameterized Cypher as an architecture rule. The vulnerability means the current code violates an architecture rule that the spec claims is enforced.
- The **Audit Emitter Sidecar** (ADR-011) does not exist. No sidecar process, no append-only audit partition, no hash-chaining implementation. ADR-011 is draft status — the mechanism is decided but nothing is built.
- The **JWT Validator** makes a remote HTTP call to the auth-service per request. The spec implies in-process JWT validation (extracting `tenant_id`, `graph_id`, `agent_id`, `scopes[]` from token claims). The current architecture has no JWT claim extraction in the knowledge-graph-builder service — user data is returned by the remote `GET /me` call. This means `graph_id` and `scopes[]` are not extracted from the token itself; they are returned by the auth service, which is a different trust model.
- The graphify index references "Authentication Mechanisms (OAuth, JWT, mTLS, SPIFFE)" (source: `raw-knowledge/docs/platform-promise/15-security/access-control.md`) and "SPIFFE/SPIRE — Cryptographic Agent Identity Framework" as relevant. The spec does not mention mTLS or SPIFFE for inter-service agent identity. This is a gap between the security research KB and the spec: the research identifies SPIFFE as a relevant mitigation for multi-agent identity threats, but the spec uses only JWT.

---

### 8. Integration Surfaces — MCP Server

Existing code: An MCP server exists at `app/mcp/server.py` using `mcp.server.fastmcp.FastMCP`. It supports both stdio and SSE transports. It exposes graph operations (CRUD, ingestion, chat) by proxying to the Oraclous REST API via HTTP, and exposes low-level node inspection tools via direct Cypher. The server uses a single `ORACLOUS_API_KEY` Bearer token for all operations — it does not extract `graph_id` or `agent_id` from per-request context.

KB evidence: ADR-007 mandates that every capability is available as both REST and MCP, and that no capability is MCP-only or REST-only. The graphify index confirms MCP Client and Server as a Layer 2 node connected to the platform framework.

Gaps:
- The existing MCP server is a standalone process that calls the REST API — it is not integrated with the L2 runtime described in the spec. The spec describes the MCP server as exposing L2 capabilities (BSP executor operations, HITL, checkpointing, agent lifecycle). The current server exposes L1 graph operations only.
- The spec states the MCP server implements "MCP stdio and HTTP+SSE transports" — the existing implementation does both, which is consistent. However, the existing server bypasses the Scope Enforcer for the low-level Cypher tools (they query Neo4j directly with no graph_id scoping), which violates ADR-007's parity requirement and the security invariant that all enforcement is server-side.
- The spec asserts "No capability is MCP-only or REST-only" (ADR-007). The existing MCP server has low-level node inspection tools (`search_nodes`, `get_node`, `get_neighbors`) that "are not yet exposed as REST endpoints" — this is explicitly acknowledged in the server's docstring. This is an existing ADR-007 violation that predates the L2 spec.
- No MCP tool versioning strategy exists. The spec's open ADR item (#1) calls for a skill file format versioning ADR. The same versioning question applies to MCP tool definitions — if a tool schema changes, existing clients using the old schema will break silently.

---

### 9. Integration Surfaces — REST API

Existing code: A REST API exists under `app/api/v1/` with endpoints for: graphs, memories, connectors, federation, permissions, service accounts, evaluation, chat, webhooks, multimodal, code graphs, health. None of these endpoints match the agent lifecycle endpoints specified in the spec (`POST /runs`, `GET /runs/{run_id}`, `POST /runs/{run_id}` for HITL resume, `GET /agents`, `POST /tasks`, etc.).

KB evidence: ADR-007 mandates REST + MCP parity. The spec's REST API section describes ACP-compliant lifecycle endpoints that are entirely new surface area — they do not exist in any current endpoint file.

Gaps:
- The existing REST API serves L1 graph operations (ingestion, retrieval, KG management). The L2 lifecycle endpoints are net-new. No overlap exists.
- The spec lists management APIs for HITL policy management, APOC trigger registry view, and AgentTask queue management. These require L1 nodes that do not yet exist (policy nodes, trigger registry, AgentTask nodes). There is a dependency chain: these REST endpoints cannot be built until the underlying L1 data model is defined and created.
- Observability APIs ("run metrics, token usage, cost per (agent, tenant, time range)") require that the LLM Gateway and Checkpointer emit structured data that can be queried. Without those components, the observability endpoints have no data source.

---

### 10. Integration Surfaces — Python SDK

Existing code: No Python SDK (`oraclous` package with `@agent`, `@tool`, `AgentContext`, `GraphClient` decorators) exists. The existing codebase is a FastAPI service, not a distributable SDK package. There is no `pyproject.toml` in a location that would indicate an SDK build target for external consumption.

KB evidence: The graphify index confirms "Agent Skill File (.oraclous.md)" and "Declarative Skill Files (.oraclous.md)" as extracted nodes from `raw-knowledge/docs/platform-promise/04-agent-team/README.md`. The skill file format is an existing concept in the knowledge base. ADR-009 references "declarative agent skill files (.oraclous.md)" as the mechanism for specifying which L2 capabilities each L3 agent uses.

Gaps:
- The entire Python SDK is unbuilt. The `AgentContext` object described in the spec (`ctx.graph`, `ctx.llm`, `ctx.hitl`, `ctx.scope`, `ctx.memory`, `ctx.delegate`, `ctx.stream`) has no implementation anywhere.
- The spec's `ctx.graph.query_at(valid_time=dt)` (bitemporal time-travel query) depends on L1 bitemporal infrastructure. The deepening phase roadmap is adding true bitemporal tracking. The SDK method is specified before the L1 bitemporal infrastructure is complete — the method cannot be implemented until L1 bitemporal writes are stable.
- The `ctx.delegate(...)` sub-agent delegation method depends on the Coordination Layer (APOC triggers + AgentTask queue), which is also unbuilt. The SDK interface is specified ahead of multiple infrastructure dependencies.
- Lifecycle hooks (`@on_pause`, `@on_resume`, `@on_cancel`, `@on_error`) require integration with the Checkpointer and BSP Executor. No document specifies the hook invocation contract: are hooks called synchronously before/after the checkpoint write? Can a hook abort a checkpoint? What happens if a hook raises an exception?

---

### 11. Integration Surfaces — Skill Files (.oraclous.md)

Existing code: No skill file parser or registry exists in `oraclous-data-studio/`. The `.oraclous.md` format exists conceptually (referenced in the graphify KB and in the spec), but no runtime code reads, validates, or registers agent definitions from skill files.

KB evidence: "Agent Skill File (.oraclous.md)" is extracted from `raw-knowledge/docs/platform-promise/04-agent-team/README.md` and listed as a god node connection in the FTOps community. ADR-009 uses skill files as the boundary-enforcement mechanism between L2 capability grants and L3 agent definitions. The spec notes that skill files are "the customer's escape-velocity artifact: portable, platform-agnostic, version-controlled." Open ADR item #1 in the spec calls for a format versioning ADR — this has not been drafted.

Gaps:
- The skill file format versioning ADR (identified as open in the spec) is unwritten. The format shown in the spec is Option B ("shared core + optional platform hints"), but no ADR closes this choice. The spec itself says this is an open item.
- The spec does not define what "scope: inherit" means at the infrastructure level — specifically, how the runtime resolves the parent scope when a skill file declares inheritance. This requires a call stack or delegation chain that is not specified anywhere.
- The `success_criterion` field in the skill file format is a natural language string ("CoverageGap nodes written for all new Finding nodes in scope"). No document specifies how the runtime evaluates this criterion. Is it evaluated by an LLM judge? By a Cypher query? By a counter? If by LLM judge, which model? This is undefined.
- The spec does not address skill file discovery: how does the runtime find and register `.oraclous.md` files? From a directory? From a registry in L1? From an API call at agent startup? No document specifies the registration protocol.

---

### 12. Observability

Existing code: OpenTelemetry is implemented in `app/core/telemetry.py` with both span and metric providers. The `neo4j_client.py` uses an OTEL tracer. FastAPI is instrumented via `instrument_fastapi(app)`. This covers L1 graph operation tracing. The existing OTEL setup exports to Jaeger (via OTLP gRPC or HTTP/protobuf).

KB evidence: The spec states "All instrumentation is automatic — no developer code required." The existing OTEL setup requires explicit tracer calls in each service (`_tracer = otel_trace.get_tracer(...)` in `neo4j_client.py`). This is not zero-developer-code. The graphify index includes "Arize Phoenix — Agent Observability and Tracing" as a referenced tool (source: `raw-knowledge/SECURITY_TESTING_FRAMEWORKS.md`), suggesting the KB has evaluated agent-specific observability tooling.

Gaps:
- The existing OTEL setup covers L1 graph operations and HTTP request tracing. It does not cover agent-specific spans: per-LLM-call cost, cache hit/miss, per-tool-call duration, per-super-step count, HITL interrupt events. These require the LLM Gateway and BSP Executor to emit spans — neither exists yet.
- The spec states cost tracking "per (agent, model, tenant)" as a built-in metric. Token cost tracking requires knowing the per-token price for each model at the time of the call. No document specifies how pricing data is maintained or updated when providers change pricing. This is an operational dependency with no owner.
- The spec mentions Arize Phoenix in the knowledge base as a candidate for agent observability. No ADR evaluates whether to use Arize Phoenix, build a custom OTEL pipeline, or both. The existing code commits to Jaeger/Tempo as the OTEL backend — if Arize Phoenix is adopted for agent traces, it represents a second observability backend with no documented integration plan.

---

## Cross-Cutting Flags

### Contradiction: LLM Gateway vs. ADR-012

**ADR-012** states: "Layer 2 does not call the LLM directly. Agent execution happens in the customer's chosen runtime." The spec introduces an LLM Gateway as a first-party L2 service that calls LLM providers on behalf of agents. These are contradictory. ADR-012 was written before ADR-006 confirmed Oraclous as its own runtime (not just a framework). ADR-012 needs to be superseded or amended. Until it is, the LLM Gateway has no governing ADR.

### Contradiction: Scope Enforcer vs. existing multi_tenant_components.py

**Architecture Rule** (stated in both CLAUDE.md and the spec): "All async code: `neo4j_client.async_driver`; all Celery tasks: task-scoped sync NullPool." The `multi_tenant_components.py` uses a sync `Driver` and Neo4j GraphRAG's sync `Retriever` base class. The string-interpolation vulnerability flagged in `raw-knowledge/SECURITY_THREAT_MODEL.md` is in this file. The spec requires all enforcement to be server-side middleware. The existing implementation is per-service, per-call, and has a known vulnerability. These are concrete contradictions between spec and existing code.

### Gap: No ADR for Agent Identity (SPIFFE/SPIRE)

The security research KB ("SPIFFE/SPIRE — Cryptographic Agent Identity Framework," source: `raw-knowledge/AGENTIC_AI_SECURITY_RESEARCH.md`) identifies cryptographic agent identity as a relevant mitigation for multi-agent impersonation threats (Threat 7 in the threat model). The spec uses JWT-based agent identity only. No ADR evaluates SPIFFE/SPIRE or documents why JWT is sufficient for agent-to-agent authentication in a multi-agent runtime. This gap exists between the security research layer and the spec.

### Gap: No document addresses MCP server security for agent-generated tool results

The graphify community "Access Control & Multi-Tenancy" includes "Tool Poisoning (MCP Server Security)" and "MCPTox Benchmark (AAAI 2026)" as extracted nodes (source: `raw-knowledge/docs/platform-promise/15-security/agent-security.md`). The spec describes the MCP server as both a client (agents call external MCP tools) and a server (external runtimes call L2 via MCP). The existing MCP server in `app/mcp/server.py` has no validation of tool results from external MCP servers before injecting them into agent context. No document specifies how tool poisoning is mitigated in the inbound MCP client path.

### Gap: Redis declared as optional but spec assumes it for scale

The spec states Redis is "already in oraclous-data-studio stack." Redis is present as Celery's broker. However, the spec's Coordination Layer uses Redis Streams (a different usage pattern requiring consumer groups, stream IDs, and acknowledgment). ADR-008's deployment models do not list Redis Streams as a required infrastructure component. The open ADR item in the spec acknowledges this must be resolved when the scale threshold is crossed. Until that ADR is written, any deployment below 500 tasks/hour has no Redis Streams dependency, but the codebase transition point is undocumented.
