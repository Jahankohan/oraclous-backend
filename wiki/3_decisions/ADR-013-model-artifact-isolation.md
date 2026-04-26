---
id: ADR-013
title: "Model checkpoints are stored in customer-owned object storage; tenant isolation is enforced via path namespace and object store ACLs"
date: 2026-04-25
status: draft
story: ""
supersedes: ""
superseded_by: ""
layer: ftops
authors: [cto, solution-architect]
---

# ADR-013: Model Artifact Storage and Tenant Isolation

## Status

`draft` — 2026-04-25 · awaiting Reza acceptance

## Context

The SA raised that Layer 3 produces model artifacts (checkpoints, evaluation results,
LoRA adapters, final merged models) but the architecture does not document where these
are stored or how tenant isolation is enforced for them. Binary model artifacts cannot
be stored as graph nodes (too large). Yet L1 is the sole persistence layer (ADR-005),
which would appear to require everything to live in the graph.

The CTO found a genuine gap: the architecture documents describe artifact production
(training, evaluation, merging) without specifying the storage substrate or how
Founding Principles 3 (Data Ownership) and 4 (Self-Hosted) apply to binary artifacts.

## Decision

**Model artifacts are stored in customer-owned object storage** (S3-compatible: AWS S3,
MinIO, Cloudflare R2, or any S3-compatible self-hosted store). Oraclous does not operate
artifact storage on behalf of customers.

**The graph (L1) is the artifact registry, not the artifact store.** Every artifact is
registered as a graph node with metadata (path, size, hash, version, tenant `graph_id`,
provenance edges to the training run that produced it). The binary bytes live in the
object store; the graph node is the index and access control point.

**Tenant isolation is enforced at two levels:**
1. **Path namespace**: all artifacts for a tenant are written to a prefix scoped by
   `graph_id` (e.g., `s3://bucket/tenants/{graph_id}/checkpoints/`). No cross-prefix
   writes are issued by the Training Agent.
2. **Object store ACLs**: the customer configures their object store such that the
   Training Agent's service credentials have read/write access only to the tenant's
   prefix. Oraclous provides reference IAM policies for AWS; equivalent policies for
   MinIO and R2 are documented.

**The credential broker service** (existing: `oraclous-data-studio/credential-broker-service/`)
supplies the object store credentials to the Training Agent at job start. The broker
enforces that the issued credentials are scoped to the requesting tenant's prefix.
Agents cannot request credentials for a different tenant's prefix.

## Rationale

- Binary model artifacts (multi-GB checkpoints) cannot be stored in Neo4j. Using
  customer-owned object storage is the only approach consistent with Founding Principle 3
  (Data Ownership) and Principle 4 (Self-Hosted): customer data never leaves customer
  infrastructure.
- The graph-as-registry pattern (metadata in graph, bytes in object store) is an
  established pattern (DVC, MLflow Artifacts, Hugging Face Hub architecture). It keeps
  L1 as the sole source of truth for *what exists and where*, without requiring L1 to
  store the bytes.
- The existing credential broker service already solves the credential scoping problem
  for other services. Extending it to object store credentials is consistent with the
  existing architecture.
- Path namespace isolation + object store ACLs provides defense-in-depth: a bug in the
  Training Agent (wrong prefix) is caught by the object store ACL; a misconfigured ACL
  is caught by the path namespace convention.

## Alternatives Considered

| Alternative | Why Rejected |
|---|---|
| Oraclous-operated shared artifact store (multi-tenant S3 bucket) | Violates Data Ownership (Principle 3) and Self-Hosted (Principle 4). Customer model weights would leave customer infrastructure. |
| Store artifacts in Neo4j as binary properties | Neo4j is not designed for binary blob storage. Multi-GB checkpoints would make Neo4j performance unacceptable. |
| One object store bucket per tenant, customer-managed | Adds operational overhead (tenant provisioning complexity). Path namespace within a shared customer bucket achieves the same isolation with simpler setup for the customer. |
| Deferred — specify storage in Phase 2 | First Training Agent implementation will write to disk or a hardcoded path. Retrofitting tenant isolation is more expensive than building it from the start. |

## Consequences

**Positive:**
- Customer model weights never leave customer infrastructure — Founding Principle 3
  satisfied unconditionally.
- The graph provides a complete, queryable registry of all artifacts (provenance, version
  lineage, training run links) without storing binary data.
- Credential broker extends an existing pattern; no new service required.

**Negative / Trade-offs:**
- Customers must provision and configure object storage before running FTOps jobs. This
  is a deployment prerequisite that must be documented clearly in the quickstart guide.
- Oraclous must provide and maintain reference IAM/ACL policies for at least AWS S3,
  MinIO, and Cloudflare R2. These policies must be kept current with provider API changes.

**Neutral:**
- The artifact graph nodes enable future features: automatic artifact comparison,
  regression detection across training runs, and artifact lineage visualization — all
  without accessing the binary bytes.

## Validation

- Configure a test tenant with a MinIO bucket and a path-scoped credential. Run a
  training job → checkpoint appears at the correct path prefix, inaccessible from
  a second tenant's credential.
- Attempt to request object store credentials for a different tenant's `graph_id`
  via the credential broker → broker rejects the request.
- Delete the object store artifact; query the graph → artifact node remains (tombstone)
  with a `missing_artifact` flag — the registry reflects reality.

## Related

- ADR-004 — Agent Service Accounts (credential broker is part of this ecosystem)
- ADR-005 — L1 is the sole persistence layer (graph as registry; bytes in object store)
- Founding Principle 3 — Data Ownership
- Founding Principle 4 — Self-Hosted
- Concern L3-ARTIFACT-ISOLATION (closed by this ADR)
