---
id: ADR-011
title: "Audit trail immutability is enforced via a structurally write-protected append-only log partition, separate from the agent-writable graph"
date: 2026-04-25
status: draft
story: ""
supersedes: ""
superseded_by: ""
layer: cross-cutting
authors: [cto, solution-architect]
---

# ADR-011: Audit Trail Immutability Mechanism

## Status

`draft` — 2026-04-25 · awaiting Reza acceptance

## Context

The SA raised that audit logs are described as "immutable, tamper-evident, cryptographically
signed" (source: raw-knowledge/docs/platform-promise/15-security/audit-and-compliance.md),
but agent actions are written to the same graph substrate (Commitment 8: Every Agent Action
Is a Graph Write). The SA found no structural mechanism that prevents a compromised agent
from overwriting or deleting its own audit trail — particularly since the agent's write
path to the graph is the same substrate that holds the audit records.

The CTO confirmed this as a genuine gap: the architecture documents assert immutability
but do not specify the mechanism by which it is enforced while the graph is also the
agent's write target.

## Decision

The audit log partition is **structurally separate from the agent-writable graph**. Concretely:

1. **Audit log = append-only Neo4j partition** with write access restricted at the
   infrastructure level to a dedicated audit-writer service account. No agent service
   account (ADR-004) has `WRITE` or `DELETE` grants on this partition.

2. **Agent actions write to the main graph partition.** The audit-writer service (a
   sidecar process, not an agent) listens to agent write events and appends immutable
   records to the audit log partition. Agents cannot invoke the audit writer directly.

3. **Cryptographic signing** is the audit-writer's responsibility: each appended record
   is hash-chained to the previous, producing a tamper-evident log. The hash chain is
   verifiable by the compliance tooling without granting agents any read access to other
   tenants' audit trails.

4. **Tenant isolation**: audit records carry `graph_id` and are scoped by the same ReBAC
   rules that govern the main graph. An agent cannot read another tenant's audit records.

## Rationale

- Keeping audit records in the same graph substrate (Neo4j) simplifies operations:
  one backup, one query layer, one topology. The structural protection comes from
  partition-level ACLs, not a separate database.
- The audit-writer sidecar pattern is established practice in append-only audit systems
  (e.g., Kafka compacted topics with producer-only ACLs, AWS CloudTrail separation from
  the resources it audits).
- Hash-chaining provides tamper-evidence without requiring a separate immutable store
  (e.g., WORM object storage), which would complicate the self-hosted deployment model
  (Founding Principle 4: Self-Hosted).

## Alternatives Considered

| Alternative | Why Rejected |
|---|---|
| Separate WORM object store (S3 Object Lock, Glacier) for audit records | Adds a second storage system to every self-hosted deployment; increases operational complexity. Acceptable as an optional cold-archive export, not as the primary store. |
| Trust agent service account ACLs to prevent audit overwrites | ACLs are admin-configured, but a compromised admin account would defeat this. Structural separation (dedicated audit-writer service account, not exposed to agents) provides a stronger guarantee. |
| Write audit records inside each agent's transaction | An agent that aborts its transaction also aborts the audit record. The sidecar pattern ensures audit records are written even when agent transactions fail or are rolled back. |

## Consequences

**Positive:**
- A compromised agent cannot delete or modify its own audit trail — it has no write
  access to the audit partition.
- Audit records survive agent transaction failures (sidecar writes independently).
- Hash chain provides tamper-evidence verifiable by compliance tooling.

**Negative / Trade-offs:**
- The audit-writer sidecar is a new infrastructure component. It must be included in
  the self-hosted docker-compose and Helm charts.
- A failure in the audit-writer sidecar means audit records are not written. This must
  be treated as a critical failure (alert + agent pause), not a silent degradation.

**Neutral:**
- Cold-archive export (e.g., to WORM object storage) is an optional compliance feature
  that can be added later; this ADR does not preclude it.

## Validation

- Attempt to write a `DELETE` Cypher query to the audit log partition using an agent
  service account → platform rejects it (infrastructure ACL).
- Modify an audit record directly in Neo4j using the audit-writer service account →
  hash chain verification fails, alerting compliance tooling.
- Kill the audit-writer sidecar mid-agent-run → audit gap is detected and alerted;
  agent execution is paused.

## Related

- ADR-004 — Agent Service Accounts (defines the ACL model that audit partition ACLs extend)
- ADR-005 — L1 is the sole persistence layer (audit partition is part of L1)
- Commitment 8 — Every Agent Action Is a Graph Write
- Concern AUDIT-TRAIL-WRITEABLE (closed by this ADR)
