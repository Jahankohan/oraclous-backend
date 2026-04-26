# Solution Architect — Concerns Register

Maintained by the Solution Architect. Open concerns are blocking or significant
findings that have not yet been resolved by an ADR or explicit Reza decision.

A concern cannot be closed by "we'll handle it later."

---

## SA Review: Layer 2 Harnessing Platform Spec (2026-04-25)

---

CONCERN: HITL-PRIORITY-ORDERING
Type: ordering
Severity: blocking
Finding: The HITL Gate Engine is listed P1, but the spec states that HITL pause states are checkpointed execution states — meaning HITL depends structurally on the Checkpointer (P0). This dependency is correctly identified in the spec's text but not in the priority table: HITL is placed P1 alongside the REST API and MCP Server, while the Checkpointer it depends on is P0. This is harmless — except the REST API is also P1 and is the delivery mechanism for HITL responses. Without the REST endpoint `POST /runs/{run_id}` (resume), there is no way to deliver a human response to a paused agent. HITL at P1 with its resume path also at P1 creates a circular dependency at P1: HITL requires the resume endpoint; the resume endpoint is only meaningful with HITL. Neither is blocked individually, but together they cannot be tested in isolation. If either slips to P2, the other loses all value. The priority table should either couple them explicitly or declare the HITL-resume REST endpoint a dependency of HITL.
Evidence: Spec §6 HITL Gate Engine: "Implementation: HITL pause states are checkpointed execution states — no separate mechanism is needed." Priority table lists HITL as P1 and REST API (lifecycle endpoints) as P1, both dependent on Checkpointer (P0). The resume path POST /runs/{run_id} is inside the P1 REST block, not explicitly broken out.
Resolution: The resume endpoint must be explicitly listed as a prerequisite for HITL to be testable. Either elevate `POST /runs/{run_id}` (resume) to a required dependency under the HITL P1 item, or add an explicit note: "HITL is only deliverable when Checkpointer (P0) + REST resume endpoint are present."

---

CONCERN: AUDIT-EMITTER-SIDECAR-NOT-P0
Type: ordering
Severity: blocking
Finding: The Audit Emitter Sidecar is listed as P0, which is correct per ADR-011. However, the BSP Executor (also P0) begins writing agent actions to L1 before the audit sidecar is structurally guaranteed to be listening. The spec does not define the startup ordering contract between the BSP Executor and the Audit Emitter Sidecar. If the sidecar starts after the executor, there is a window where agent writes occur with no audit record — violating ADR-011's validation criterion: "Kill the audit-writer sidecar mid-agent-run → audit gap is detected and alerted; agent execution is paused." The spec states the sidecar "listens to agent write events," but defines no event delivery mechanism — APOC trigger? Kafka-style log? Direct call? — and no failure mode when the sidecar is absent.
Evidence: ADR-011 §Decision: "A failure in the audit-writer sidecar means audit records are not written. This must be treated as a critical failure (alert + agent pause), not a silent degradation." The L2 spec §7 Security Layer lists "Audit Emitter Sidecar" as a component but does not specify the event channel it listens on, the startup dependency relationship with the BSP Executor, or what "agent pause" means in implementation terms.
Resolution: The spec must define: (1) the event channel the sidecar uses to observe writes (APOC trigger? Neo4j change-data-capture? direct HTTP?), (2) the startup ordering guarantee (sidecar must be confirmed healthy before BSP Executor accepts any task), and (3) the "agent pause" mechanism when the sidecar is unresponsive — which component detects absence and which component halts execution.

---

CONCERN: SCOPE-ENFORCER-NOT-AT-L1-API-BOUNDARY
Type: security
Severity: blocking
Finding: The spec places the Scope Enforcer in the "Security Layer (server-side at L1 API boundary)" box in the architecture diagram, which is correct per ADR-010. However, the Python SDK exposes `ctx.graph.query(cypher, params)` where the agent author supplies raw Cypher. The spec states that scope enforcement injects mandatory `graph_id` filter server-side and rejects queries missing `graph_id`. But the SDK's `graph_query` tool also calls `ctx.graph.query(input.cypher, input.params)` — meaning the agent can supply arbitrary Cypher containing its own `graph_id` parameter, which the Scope Enforcer then validates against. If the Scope Enforcer compares the `graph_id` in the JWT against a `graph_id` in the Cypher string, an agent could supply `graph_id: $gid` (which it controls at runtime) and cause parameter substitution to vary from its authorized graph. The spec does not state whether the server-side enforcer extracts and validates the `graph_id` from query parameters or whether it rewrites the query to inject the authorized `graph_id` unconditionally.
Evidence: Spec §7: "Injects mandatory `graph_id` filter into all Cypher queries server-side. A query missing `graph_id` is rejected, not passed through." SDK example: `ctx.graph.query("MATCH (f:Finding {graph_id: $gid}) RETURN f", params={"gid": ctx.graph_id})` — the agent sets `gid` from `ctx.graph_id`, which is populated from the JWT. But the SDK does not prevent an agent from supplying `params={"gid": "another-tenant-id"}`. Whether the server-side enforcer catches this is not specified.
Resolution: The spec must explicitly state that the Scope Enforcer does not rely on the `graph_id` value in query parameters submitted by agents. It must unconditionally rewrite or augment Cypher to bind `graph_id` to the value extracted from the JWT, regardless of what the agent's parameter bag contains. "Reject if missing" is not sufficient; "overwrite with JWT value" is required.

---

CONCERN: APOC-TRIGGER-MULTI-TENANT-PROLIFERATION
Type: structural
Severity: significant
Finding: The coordination layer registers one APOC trigger per `(graph_id, trigger_type)` pair. The registration example shows `'finding-created-{graph_id}'` as the trigger name. In a multi-tenant deployment, this creates one trigger per tenant per event type. APOC stores triggers in Neo4j's system database and evaluates them on every write transaction. At even modest scale (100 tenants × 5 trigger types = 500 triggers), every write transaction in the database must be evaluated against 500 trigger conditions regardless of which tenant's graph was written to. This is an O(tenants × trigger_types) overhead on every write, imposed at the database layer where it cannot be sharded. The spec does not address this.
Evidence: Spec §3 Coordination Layer: trigger name `'finding-created-{graph_id}'` implies per-tenant trigger registration. No mention of trigger count limits, consolidation strategy, or namespace partitioning. APOC trigger documentation: triggers are evaluated per-database (not per-subgraph), so all tenant triggers fire in the same evaluation pass per transaction.
Resolution: The spec must either: (a) define a single parameterized trigger per event type that filters internally by `graph_id` (reducing N×M triggers to N triggers), or (b) explicitly acknowledge the O(tenants × types) cost and define the scale limit above which this becomes a structural problem, or (c) commit to using Redis Streams as the primary coordination path (rather than APOC triggers) for multi-tenant deployments, with APOC triggers only in single-tenant mode.

---

CONCERN: MEMORY-STORE-NO-INTERFACE-SPEC
Type: interface
Severity: significant
Finding: The `ctx.memory` object is listed in the SDK as "cross-thread key-value store (namespace = (agent_id, graph_id))" but has no interface specification in the spec. The architecture diagram shows "Memory Store (→ L1)" in the Platform Services box, implying it is a component. ADR-005 mandates that all persistent data goes to L1. But the spec never defines: what operations `ctx.memory` exposes, what L1 node structure backs it, whether memory reads are scoped by the Scope Enforcer, whether memory is bitemporal, or whether an agent can read another agent's memory if both share the same `graph_id`. This is an undefined interface attached to a critical surface (the agent context object).
Evidence: Spec §8 Python SDK: "`ctx.memory` — cross-thread key-value store (namespace = (agent_id, graph_id))". Architecture diagram: "Memory Store (→ L1)". No further specification anywhere in the document. ADR-005 mandates writes to L1 but does not specify the memory node structure.
Resolution: The spec must define: (1) the L1 node schema backing `ctx.memory`, (2) the read/write API surface (`get`, `set`, `delete`, `list`? something else?), (3) whether reads are scoped through the same Scope Enforcer middleware as graph queries or are a separate trusted path, (4) whether memory is readable across agents within the same tenant, and (5) whether memory nodes are bitemporal (required by ADR-005 logic, since L1 is the sole store and L1 has bitemporal tracking).

---

CONCERN: MCP-SERVER-PRIORITY-ADR007-VIOLATION
Type: ordering
Severity: significant
Finding: ADR-007 declares MCP as a first-class interface coequal with REST — "No capability is MCP-only or REST-only." The validation criterion in ADR-007 is: "call every REST endpoint via its MCP equivalent — if any gap exists, this ADR has been violated." However, the priority table places the MCP Server at P1 while REST API lifecycle endpoints are also P1. The problem is not the P1 designation in isolation, but that the spec defines a P0 operational gap: the BSP Executor (P0), Tool Dispatcher (P0), and APOC Trigger Listener (P0) are all operational before any MCP surface exists. During the P0→P1 window, the system can run agents but exposes zero MCP capability. This directly violates ADR-007's "MCP is first-class" commitment — the first-class interface will not exist while the core runtime is already operational.
Evidence: ADR-007: "MCP is the primary integration protocol for agent-to-agent and tool-to-tool communication. It is not an optional adapter — it is the first-class interface." Priority table: BSP Executor P0, MCP Server P1. The P0 runtime will produce observable outputs (L1 writes, APOC triggers) with no MCP surface to consume them.
Resolution: Either (a) the MCP Server must be elevated to P0 alongside the REST API (which also has lifecycle utility even before HITL), or (b) ADR-007 must be amended with a startup exception that explicitly permits the P0→P1 window as a controlled development gap (not a production state). The spec cannot treat MCP as first-class and simultaneously list it as a P1 capability that comes after the runtime is running.

---

CONCERN: SKILL-FILE-SCOPE-ENFORCEMENT-GAP
Type: security
Severity: significant
Finding: The skill file format (§8 Skill Files) declares `reads`, `writes`, and `tools` as agent-declared fields. The spec states these are read by the runtime to "register agents, configure tool access, declare scope, set HITL policies." ADR-010 mandates that scope inheritance is platform-enforced, not trusted to skill files. But the spec does not state who validates the skill file's `reads`/`writes` declarations against the agent service account's actual ACL grants (ADR-004). If the runtime reads the skill file and registers the agent with the declared scope without cross-checking against the service account's ACL, the skill file becomes a self-declaration of scope — which ADR-010 explicitly rejects as bypassable.
Evidence: ADR-010: "Application-layer enforcement of scope (e.g., in agent skill files) can be bypassed by a compromised or malicious agent. Platform-level enforcement cannot." Spec §8: "The runtime reads skill files to: register agents, configure tool access, declare scope, set HITL policies." No mention of ACL cross-validation at registration time. Spec §7: Scope Enforcer validates "every requested operation against the agent's declared scope" — but "declared" here is ambiguous: is it the skill file's declaration or the service account's ACL?
Resolution: The spec must explicitly state that skill file `reads`/`writes`/`tools` declarations are validated at registration time against the agent service account's ACL grants (ADR-004). If the skill file declares a tool not granted to the service account, registration is rejected — not deferred to runtime enforcement. "Declared scope" in the Scope Enforcer must mean the ACL-granted scope, not the skill-file-stated scope.

---

CONCERN: CHECKPOINTER-TENANT-ISOLATION-UNSPECIFIED
Type: security
Severity: significant
Finding: The Checkpointer schema defines `thread_id` and `graph_id` fields on Checkpoint nodes. ADR-005 mandates L1 as the sole persistence layer, which means checkpoints live in the same Neo4j instance as all tenants' knowledge graphs. The spec does not state whether checkpoint nodes are subject to the same Scope Enforcer middleware that enforces `graph_id` isolation on graph queries. If checkpoint reads/writes bypass the Scope Enforcer (e.g., because they are internal runtime calls, not SDK calls), a compromised runtime component could read or write checkpoints across tenant boundaries.
Evidence: Spec §5 Checkpointer: "Storage: L1 (Neo4j) as the sole persistence layer (ADR-005)." Schema includes `graph_id: string // tenant`. No statement that checkpoint read/write operations are subject to the L1 API security middleware. The spec distinguishes "server-side at L1 API boundary" (§7) from the Checkpointer (§5), suggesting the Checkpointer may have a different path to L1.
Resolution: The spec must explicitly state that Checkpointer reads and writes to L1 go through the same L1 API security middleware (Scope Enforcer, JWT Validator) as all other L1 writes. Alternatively, if the Checkpointer has a privileged internal path, that path must be documented and the isolation guarantee must be restated with specificity (e.g., "Checkpointer service account has WRITE access only to Checkpoint nodes and is ACL-restricted by `graph_id` at the database level").

---

CONCERN: SUB-AGENT-DELEGATION-P2-BUT-L3-REQUIRES-IT
Type: ordering
Severity: significant
Finding: Sub-agent delegation (`ctx.delegate`) is listed P2 — "Required for L3 hierarchical agents." If L3 FTOps agents are hierarchical (which the spec asserts by naming this feature as "Required for L3"), then L3 cannot deliver its hierarchical multi-agent patterns until P2 is complete. The spec's priority table has no L3 agent row — it is not an L2 concern — but the dependency is structural: no P2 delegation means no L3 hierarchical patterns. This is not an internal L2 ordering problem; it is an L2→L3 dependency that is nowhere called out as a gate. If L3 work begins before `ctx.delegate` is implemented, L3 agents will be architected around a capability that does not yet exist.
Evidence: Priority table: "Sub-agent delegation — P2 — Required for L3 hierarchical agents." The CLAUDE.md project context states L3 FTOps Wave 1 includes agents in "Structural, Coverage & Gap, Community & Hierarchy analysis families." Hierarchy families imply delegating to sub-agents. The current phase targets FTOps agent build after Deepening phase.
Resolution: The spec must add an explicit L3 gate: "L3 hierarchical agent work cannot begin until `ctx.delegate` (P2) is implemented and tested." This should appear in the Open Items section or as an explicit dependency row in the priority table, not be inferred from a parenthetical "Required for L3" note.

---

CONCERN: REDIS-STREAMS-ADR005-CONFLICT
Type: structural
Severity: significant
Finding: The spec defers a Redis Streams ADR to the "Open Items" section, noting it becomes required above 500 tasks/hour. ADR-005 mandates L1 as the sole persistence layer. Redis Streams is a persistent data store (not an in-memory cache in this usage — stream entries are durable until acknowledged and trimmed). If Redis Streams carries AgentTask events that are not also written to L1, then at the moment the scale threshold is crossed, a second persistent store comes into existence, violating ADR-005. The spec acknowledges "ADR needed when the scale threshold is crossed" but does not pre-resolve the conflict with ADR-005.
Evidence: ADR-005: "A future implementation that introduces a second persistent store (Redis for memory, separate log DB, etc.) violates this ADR and must either be justified by a superseding ADR or reverted." Spec §3 Coordination Layer: "≥ ~500 tasks/hour: relay through Redis Streams." Spec Open Items: "Redis Streams as L2 infrastructure component — ADR needed when the scale threshold is crossed."
Resolution: The spec must pre-resolve this conflict now rather than deferring it. Options: (a) declare that AgentTask nodes in L1 remain the authoritative record and Redis Streams is a delivery channel only (messages are ephemeral relay, not authoritative state), meaning ADR-005 is not violated because Redis is not the store of record; or (b) acknowledge this will require a superseding ADR when crossed. Option (a) is architecturally correct if specified now; option (b) leaves ADR-005 in an unresolved conflict state. The spec cannot simultaneously claim ADR-005 compliance and leave this unresolved.

---

CONCERN: CONTEXT-MANAGER-PROMPT-MANAGER-UNDEFINED
Type: interface
Severity: minor
Finding: "Context Manager + Prompt Manager" is listed as a P1 component with the note "Required for coherent multi-turn agents." These components appear in no other section of the spec — no description, no interface, no responsibilities. The BSP Executor (§1) describes assembling context ("system prompt + injected graph context + message history"), which is precisely what a Context Manager and Prompt Manager would do. Either these are the same thing (in which case the P1 row is redundant and confusing) or they are separate components with undefined responsibilities (in which case P1 items have no spec).
Evidence: Priority table row: "Context Manager + Prompt Manager — P1 — Required for coherent multi-turn agents." Spec §1 BSP Executor LLM call loop step 1: "Assemble context (system prompt + injected graph context + message history)" — this is context management, performed inside the BSP Executor with no delegation to a separate component.
Resolution: Either (a) remove "Context Manager + Prompt Manager" from the priority table and fold context assembly into the BSP Executor spec (where it already lives), or (b) add a dedicated section specifying what Context Manager and Prompt Manager are, what interface they expose, and what is left to the BSP Executor vs. delegated to them.

---

## Open Concerns

### CONCERN: AGENT-SCOPE-CONTAINMENT
**Type:** security
**Severity:** significant
**Status:** open (partially mitigated)
**Raised:** 2026-04-25
**Resolved:** —
**Resolved by:** ADR-010 (partial — Phase 3 gap remains)

**Finding:** Layer 2 orchestrates agents but the architecture provides no structural mechanism to prevent a sub-agent from exceeding the scope of its parent — the classic confused-deputy problem.

**Evidence:** The vision states "flexibility for integration" as cross-cutting. Flexibility and strict scope containment are in direct tension. An agent that can compose two permitted operations to produce an unpermitted outcome defeats multi-tenant isolation without violating any individual permission. The vision names "governance" as cross-cutting but does not specify the structural mechanism.

**Resolution required:** Per-operation permissions (fresh permission check per LLM request) — committed to Phase 3. Until Phase 3: scope inheritance + agent service account ACLs (ADR-010) provide structural mitigation. The confused-deputy attack vector is partially open until Phase 3 delivery. **This concern will close when Phase 3 per-operation permissions are delivered.**

---

## Closed Concerns

### CONCERN: L1-SCOPE-CONTRADICTION
**Type:** structural
**Severity:** blocking
**Status:** closed
**Raised:** 2026-04-25
**Resolved:** 2026-04-25
**Resolved by:** ADR-005

**Finding:** Layer 1 is described as both "lean, single-purpose" and as "supports agent coordination" — these are two distinct responsibilities, not one.

**Resolution:** ADR-005 establishes that L1 is the sole persistence layer. "Agent coordination" means coordination state (job state, agent activity, routing signals) is stored as graph nodes/edges — the same substrate, not a second system. L1 remains single-purpose: persistent graph. Coordination is a use pattern of the graph, not a second responsibility.

---

### CONCERN: L2-L3-EVALUATION-OVERLAP
**Type:** structural
**Severity:** blocking
**Status:** closed
**Raised:** 2026-04-25
**Resolved:** 2026-04-25
**Resolved by:** ADR-009

**Finding:** Layer 2 claims "evaluation" and Layer 3 claims "fine-tuning lifecycle" — model evaluation is a core step of every fine-tuning lifecycle, making these responsibilities structurally overlapping.

**Resolution:** ADR-009 establishes the primitive/lifecycle distinction. L2 provides evaluation *framework primitives* (RAGAS integration, test set generation, metrics APIs). L3 Evaluation Agents (Stage 7) own the evaluation *lifecycle stage* and call L2 primitives to do their work. "Evaluation" in L2 always means primitives; evaluation stage is L3.

---

### CONCERN: L1-L2-MEMORY-BOUNDARY
**Type:** structural
**Severity:** blocking
**Status:** closed
**Raised:** 2026-04-25
**Resolved:** 2026-04-25
**Resolved by:** ADR-005

**Finding:** Layer 1 holds the "knowledge base" and Layer 2 holds "memory" — two persistent stores with undefined boundary.

**Resolution:** ADR-005 establishes that all persistent state, including L2 agent memory, is stored in L1 as graph writes. L2 does not own a persistent store — it owns the interface through which agents read and write L1. There is one persistent store. The consistency problem dissolves.

---

### CONCERN: L3-ARTIFACT-ISOLATION
**Type:** security
**Severity:** blocking
**Status:** closed
**Raised:** 2026-04-25
**Resolved:** 2026-04-25
**Resolved by:** ADR-013

**Finding:** The architecture states no mechanism by which fine-tuned model artifacts produced by Layer 3 are tenant-isolated.

**Resolution:** ADR-013 specifies: artifacts stored in customer-owned object storage (S3-compatible); path namespace scoped by `graph_id`; object store ACLs enforced by the credential broker service. The graph (L1) is the artifact registry (metadata, provenance, lineage); binary bytes live in customer infrastructure. Tenant isolation is structural: a second tenant's credentials cannot access the first tenant's path prefix.

---

### CONCERN: AUDIT-TRAIL-WRITEABLE
**Type:** security
**Severity:** significant
**Status:** closed
**Raised:** 2026-04-25
**Resolved:** 2026-04-25
**Resolved by:** ADR-011

**Finding:** If the audit trail is stored in Layer 1 and Layer 2/3 agents have write access to Layer 1, agents can modify or overwrite audit records.

**Resolution:** ADR-011 specifies: audit log partition in L1 with infrastructure-level ACLs that grant write access only to a dedicated audit-writer sidecar service. Agent service accounts have no write or delete grants on the audit partition. Hash-chaining provides tamper-evidence. The sidecar listens to agent write events and appends immutable records independently of agent transactions.

---

### CONCERN: LLM-PROVIDER-COUPLING
**Type:** integration
**Severity:** significant
**Status:** closed
**Raised:** 2026-04-25
**Resolved:** 2026-04-25
**Resolved by:** ADR-012

**Finding:** Layer 2 requires LLM API calls for orchestration, but the architecture specifies no provider abstraction.

**Resolution:** ADR-012 establishes that L2 does not call the LLM directly — agent execution happens in the customer's runtime (ADR-006), which owns the LLM provider selection. L2's skill file format is provider-agnostic. For L3 training framework coupling: a pluggable adapter interface (`start_training_run`, `get_run_status`, `cancel_run`) makes Axolotl/TRL/Unsloth interchangeable; framework selection is deployment-time config.

---

### CONCERN: L2-HARNESSING-PLATFORM-BOUNDARY
**Type:** integration
**Severity:** significant
**Status:** closed
**Raised:** 2026-04-25
**Resolved:** 2026-04-25
**Resolved by:** ADR-006

**Finding:** Layer 2 is named "Harnessing Platform (name TBD)" — the boundary with existing harnessing platforms (LangGraph, CrewAI, AutoGen) has not been defined.

**Resolution:** ADR-006 establishes that L2 is the Oraclous Harnessing Platform — a graph-native agent runtime built for the L1 graph substrate. It owns the execution loop, inter-agent coordination via APOC triggers, HITL, LLM gateway, and all integration surfaces (MCP, REST, SDK). External platforms connect via MCP; no custom adapters are built.

---

### CONCERN: MCP-SURFACE-UNDEFINED
**Type:** integration
**Severity:** significant
**Status:** closed
**Raised:** 2026-04-25
**Resolved:** 2026-04-25
**Resolved by:** ADR-007

**Finding:** The vision does not mention MCP. If Layer 2 does not expose its graph primitives via a contract-based interface, the platform cannot be used by external agents without custom integration.

**Resolution:** ADR-007 formalizes Architectural Commitment 3 (MCP-First). Every capability is exposed as both REST and MCP. MCP is the primary integration protocol for agent-to-agent communication. No capability is MCP-only or REST-only.

---

### CONCERN: INDEPENDENT-LAYERS-CLAIM
**Type:** conflict
**Severity:** significant
**Status:** closed
**Raised:** 2026-04-25
**Resolved:** 2026-04-25
**Resolved by:** ADR-008

**Finding:** The architecture implies layers are independently deployable, but L2 requires L1 and L3 requires L1+L2 — making the claim false for Layers 2 and 3.

**Resolution:** ADR-008 defines the three valid deployment models (Integration Fabric = L1+MCP; Custom AI Orchestration = L1+L2; End-to-End FTOps = L1+L2+L3). Layers are independent *product boundaries with progressive onramp*, not independent runtimes. The phrase "independently deployable" is removed from the architecture.

---

### CONCERN: L2-SCOPE-TOO-BROAD
**Type:** structural
**Severity:** minor
**Status:** closed
**Raised:** 2026-04-25
**Resolved:** 2026-04-25
**Resolved by:** CTO challenge (no ADR required)

**Finding:** Layer 2's responsibility list (memory, orchestration, evaluation, monitoring, feedback loops, observability) is six distinct functional areas that risk creating a monolith.

**Resolution:** CTO evidence: all six functions share a common architectural substrate — the graph. Memory is graph reads/writes. Orchestration signals are graph nodes. Monitoring state is graph writes. Feedback records are graph nodes. Observability queries are graph traversals. L2 is lean because it delegates all complexity to L1 (the graph substrate) and L3 (the agent team). The six functions are not six subsystems; they are six read/write patterns on one substrate. No ADR required.

---

## Format Reference

```
### CONCERN: [label]
**Type:** structural | security | integration | conflict
**Severity:** blocking | significant | minor
**Status:** open | closed
**Raised:** YYYY-MM-DD
**Resolved:** YYYY-MM-DD (if closed)
**Resolved by:** ADR-XXX | explicit-reza-decision (if closed)

**Finding:** [one specific, falsifiable statement]

**Evidence:** [what in the architecture leads to this concern]

**Resolution required:** [what must be true to close this]
```
