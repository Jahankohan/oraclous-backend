# Security Findings Register

This file is append-only. Every entry is a Security Architect finding from a code-level review.
Format: one block per finding, ordered by review date then severity.

---

## SECURITY REVIEW: TASK-005

**Branch:** `agent/STORY-002/TASK-005-bitemporal-models`
**Reviewed:** 2026-04-26
**Reviewer:** Security Architect
**Files reviewed:**
- `knowledge-graph-builder/app/schemas/graph_schemas.py`
- `knowledge-graph-builder/app/components/multi_tenant_components.py`
- `knowledge-graph-builder/app/services/pipeline_service.py`
- `knowledge-graph-builder/app/services/background_jobs.py`

---

### Finding 1

```
SECURITY REVIEW: TASK-005
Finding: VULNERABILITY
Severity: high
Location: knowledge-graph-builder/app/services/background_jobs.py:2055–2064 (run_bitemporal_migration_v1)
Description: The entity migration Cypher `MATCH (e:__Entity__) WHERE e.event_time IS NULL ...`
  has no graph_id filter. It operates across ALL tenants in the database — every __Entity__ node
  regardless of which tenant owns it. This is a cross-tenant data modification executed by a
  single admin-triggered task. Any party with the ability to trigger this Celery task (e.g., via
  the beat schedule or a misconfigured Celery route) causes data to be written across every
  tenant simultaneously. The relationship migration at line 2071 has the same defect.
Exploit path: An attacker with Celery task-dispatch access (e.g., via a compromised broker
  credential or an exposed Celery Flower UI) triggers run_bitemporal_migration_v1. The task
  rewrites ingestion_source = 'pre-migration' and event_time on every __Entity__ node in Neo4j,
  destroying provenance data for all tenants. In a multi-tenant production deployment this is a
  bulk data-integrity attack requiring no per-tenant authentication. If the task were ever added
  to beat_schedule it would execute automatically on every worker restart.
Required fix: Restructure the migration to be graph-scoped. Dispatch one Celery subtask per
  known graph_id. At minimum add a WHERE clause:

    MATCH (e:__Entity__)
    WHERE e.event_time IS NULL AND e.graph_id IN $authorized_graph_ids

  The full fix: introduce a fan-out task that fetches all distinct graph_ids from Neo4j and
  dispatches run_bitemporal_migration_v1 once per graph with a bound parameter, plus remove the
  global MATCH from the single-task form. The same fix must be applied to the relationship
  migration query (line 2071–2079).
```

---

### Finding 2

```
SECURITY REVIEW: TASK-005
Finding: VULNERABILITY
Severity: medium
Location: knowledge-graph-builder/app/components/multi_tenant_components.py:221–249 (MultiTenantKGWriter.run)
Description: ingestion_source is written with setdefault — meaning a caller can pre-set
  ingestion_source in node.properties before handing the graph to the writer, and setdefault
  will NOT overwrite it. The LLM extractor (LLMEntityRelationExtractor) assembles node
  properties from untrusted LLM JSON output. If an adversary can influence the LLM output (via
  prompt injection in the ingested document), they can embed an arbitrary ingestion_source value
  that the writer will preserve unchanged. There is no length bound or character validation on
  the stored string at any point in the pipeline. An unbounded or control-character-containing
  ingestion_source reaches Neo4j verbatim.
Exploit path: An adversary crafts a document containing a prompt-injection payload that causes
  the LLM extractor to output a node with a property {"ingestion_source": "<script>...</script>"}
  or a 1MB string. setdefault preserves the attacker-controlled value. It is stored in Neo4j.
  Downstream queries returning ingestion_source surface the value — a stored XSS vector if
  rendered in a frontend without escaping, or a log-injection vector if logged at info/debug
  level (logger.info calls in pipeline_service.py include entity properties).
Required fix: Sanitize and bound ingestion_source before setdefault. Add to MultiTenantKGWriter:

    _MAX_INGESTION_SOURCE_LEN = 512

    @staticmethod
    def _sanitize_source(raw: str | None) -> str | None:
        if raw is None:
            return None
        cleaned = raw.replace("\x00", "").strip()[:_MAX_INGESTION_SOURCE_LEN]
        return cleaned or None

  In run(), replace the setdefault call with:
    # Strip any caller/LLM-provided ingestion_source; apply sanitized writer value.
    node.properties.pop("ingestion_source", None)
    safe = self._sanitize_source(self.ingestion_source)
    if safe is not None:
        node.properties["ingestion_source"] = safe

  Apply the same logic to relationships.
```

---

### Finding 3

```
SECURITY REVIEW: TASK-005
Finding: VULNERABILITY
Severity: medium
Location: knowledge-graph-builder/app/schemas/graph_schemas.py:71–74, 119–122
Description: The new ingestion_source field in EntityNodeProperties and RelationshipProperties
  is declared as `str | None` with no max_length constraint. event_time, event_time_end, and
  ingestion_time have no range bounds. Nothing prevents a client from submitting:
  - ingestion_source = a multi-megabyte string (Neo4j property storage DoS)
  - ingestion_time = datetime(0001, 1, 1) (breaks any "ingested this week" temporal filter)
  - event_time = datetime(9999, 12, 31) (poisons temporal range queries)
  These fields flow from the Pydantic model into the Neo4j write path unchanged.
Exploit path: A client with ingestion API access sends a request body containing
  ingestion_source set to 1MB. Pydantic accepts it. The pipeline writes it to Neo4j. Property
  storage bloat across many nodes accumulates. Separately, a client sets ingestion_time to a
  value in year 9999; a bitemporal query for "facts ingested last month" never returns this
  entity. Both are denial-of-service or data-corruption vectors available to any authenticated
  API user.
Required fix:
  1. Add max_length=512 to ingestion_source in both models:
       ingestion_source: str | None = Field(None, max_length=512)
  2. Add a field_validator to reject implausible ingestion_time values (server-set field;
     reject if caller provides a value outside [2020-01-01, now+1h]):
       @field_validator("ingestion_time", mode="before")
       @classmethod
       def bound_ingestion_time(cls, v):
           if isinstance(v, datetime):
               _min = datetime(2020, 1, 1, tzinfo=UTC)
               _max = datetime.now(UTC) + timedelta(hours=1)
               if v < _min or v > _max:
                   return None
           return v
  3. Long-term: ingestion_time should be removed from the API request schema entirely and
     set only server-side in MultiTenantKGWriter. This is tracked as an architectural gap.
```

---

### Finding 4

```
SECURITY REVIEW: TASK-005
Finding: OBSERVATION
Severity: informational
Location: knowledge-graph-builder/app/components/multi_tenant_components.py:237–246 (MultiTenantKGWriter.run)
Description: graph_id is written using .update() (unconditional overwrite), while
  ingestion_source uses setdefault (preserve-existing). The asymmetry is intentional and
  correct: graph_id must always be the writer's value to enforce tenant isolation. If a future
  developer changes .update() to setdefault() for graph_id, tenant isolation would be broken —
  a caller could inject an arbitrary graph_id through node.properties. The current code is
  correct but lacks a comment explaining why the asymmetry is intentional.
Exploit path: Not currently exploitable. Risk is a future regression if the asymmetry is
  "fixed" without understanding the security rationale.
Required fix: Add an inline comment:
    # graph_id: unconditional overwrite via update() — this IS the tenant isolation boundary.
    # Never change to setdefault(); a caller-supplied graph_id in node.properties would bypass
    # multi-tenancy. See security-findings.md Finding 4, TASK-005 review.
    node.properties.update({
        "graph_id": self.graph_id,
        ...
    })
```

---

### Finding 5

```
SECURITY REVIEW: TASK-005
Finding: PASS
Severity: informational
Location: knowledge-graph-builder/app/components/multi_tenant_components.py (MultiTenantKGWriter.run)
Description: The new ingestion_time and ingestion_source writes are not Cypher injection
  vectors. Both values are injected into the Python-side node.properties dict before
  base_writer.run(graph) is called. The base Neo4j GraphRAG Neo4jWriter serialises node
  properties via the driver's parameterized query interface — not via string interpolation.
  ingestion_time is datetime.now(UTC), a server-generated Python datetime object.
  ingestion_source is a Python string passed through the driver's parameter binding.
  Neither creates a new Cypher string-construction path.
  The pre-existing string-injection vulnerability in _inject_graph_id_filter (retriever path)
  was NOT touched by TASK-005 and was NOT worsened. It remains deferred to TASK-012.
Exploit path: None for the new bitemporal writes.
Required fix: None. Pre-existing _inject_graph_id_filter injection risk remains and is tracked
  in the pre-existing findings register entry (fix when TASK-012 Scope Enforcer is implemented).
```

---

### Finding 6

```
SECURITY REVIEW: TASK-005
Finding: PASS
Severity: informational
Location: knowledge-graph-builder/app/services/pipeline_service.py:817
Description: kg_writer.ingestion_source = source where source is the document path string
  (e.g., "job_<uuid>"). This value does not contain connection strings or tokens. The
  database connector sync task (sync_database_connector) passes connector_id as the
  ingestion_source identifier — connector_id is a UUID set by the system when the connector
  was created, not a connection string. Database credentials are stored in the credential
  broker (credential_service.get_user_credentials) and are never copied into connector_id
  or any other Neo4j-stored field. No credential leakage via ingestion_source is present in
  the TASK-005 changes or in the existing connector architecture.
Exploit path: None for the specific credential-leakage concern raised in the review mandate.
Required fix: None. The broader ingestion_source sanitization fix (Finding 2) still applies.
```

---

### Finding 7

```
SECURITY REVIEW: TASK-005
Finding: VULNERABILITY
Severity: low
Location: knowledge-graph-builder/app/services/background_jobs.py:2043–2051 (run_bitemporal_migration_v1 guard)
Description: The migration guard (check for Migration node, then run migration) uses two
  separate session.run() calls in the same session but without explicit transaction semantics
  that prevent concurrent execution. A TOCTOU (time-of-check/time-of-use) race exists: two
  concurrent invocations of this task both pass the guard check before either creates the
  Migration node, then both execute the migration queries. The result is idempotent (SET is
  safe to apply twice for these scalar properties), so this does not corrupt data, but it
  causes unnecessary double write-load on large deployments and means the entities_updated /
  rels_updated counters in the Migration node will be set to whichever invocation finishes last.
Exploit path: Low-severity. An operator triggering the task twice rapidly (or two Celery workers
  racing) causes double migration execution. No tenant isolation bypass or data corruption.
Required fix: Combine guard and migration atomically using MERGE with ON CREATE:
    MERGE (m:Migration {id: $migration_id})
    ON CREATE SET m.in_progress = true, m.started_at = datetime()
    ON MATCH SET m = m  -- no-op for existing
    RETURN m.done AS already_done, m.in_progress AS in_progress
  Then check already_done before proceeding. This makes the guard atomic at the Neo4j
  transaction level. Alternatively, rely on Celery task locking (e.g., redis SETNX) to
  prevent concurrent dispatch.
```

---

## VERDICT: TASK-005

**Status: BLOCKED**

**Reason:** Finding 1 is a high-severity cross-tenant data modification. The bitemporal
migration task operates without any `graph_id` scope, touching every tenant's entity and
relationship nodes in a single query. This is a tenant isolation violation per Security
Architect mandate (tenant isolation bypass = always blocking).

**Required before merge:**
1. **Finding 1 (high — blocking):** Scope `run_bitemporal_migration_v1` to individual
   graph_ids. Must be fixed; does not require Reza acknowledgment — it is an unambiguous
   tenant isolation violation.

**Required acknowledgment by Reza (medium — not blocking pending acknowledgment):**
2. **Finding 2 (medium):** Add ingestion_source sanitization in `MultiTenantKGWriter`.
3. **Finding 3 (medium):** Add max_length and field_validator bounds to bitemporal fields
   in `graph_schemas.py`.

**Non-blocking (no action required to unblock merge):**
4. Finding 4 (informational): Add comment explaining graph_id .update() asymmetry.
5. Finding 5 (informational): PASS — no Cypher injection introduced.
6. Finding 6 (informational): PASS — no credential leakage via ingestion_source.
7. Finding 7 (low): Fix TOCTOU race in migration guard (preferred but not blocking).

---

## RE-REVIEW: TASK-005

**Re-reviewed:** 2026-04-26
**Reviewer:** Security Architect
**Trigger:** Backend-developer agent log entry claims three fix commits applied.

---

```
RE-REVIEW: TASK-005
Finding: 1
Original severity: high
Status: UNRESOLVED
Notes: run_bitemporal_migration_v1 does not exist anywhere in background_jobs.py on the
  branch. The TASK-005 branch (agent/STORY-002/TASK-005-bitemporal-models) contains only
  two commits: the initial monorepo commit (530a9c0) and a docs-only commit (5d965e3) adding
  the wiki task file. background_jobs.py is not tracked by this branch at all. Inspecting
  the on-disk file: zero occurrences of "bitemporal", "run_bitemporal_migration",
  "_run_bitemporal_migration_for_graph", or any Migration guard node. The entire migration
  implementation is absent. Finding 1 remains open — no fan-out per graph_id exists because
  no migration task exists at all.
```

```
RE-REVIEW: TASK-005
Finding: 2
Original severity: medium
Status: UNRESOLVED
Notes: multi_tenant_components.py, MultiTenantKGWriter.run() — lines 292-293 and 311-312
  still use setdefault("ingestion_source", self.ingestion_source) on both entity nodes and
  relationships. No pop() call is present. No _sanitize_source() static method exists
  anywhere in the file. The Agent Log entry claiming "pop/sanitize ingestion_source;
  _sanitize_source() helper" does not match the on-disk file content. Finding 2 remains
  open — the LLM-controlled prompt-injection path is unchanged.
```

```
RE-REVIEW: TASK-005
Finding: 3
Original severity: medium
Status: UNRESOLVED
Notes: graph_schemas.py — RelationshipProperties.ingestion_source (line 71) is declared
  as Field(None, description="...") with no max_length argument. EntityNodeProperties.ingestion_source
  (line 119) is identical: no max_length. No bound_ingestion_time field_validator exists —
  the only field_validators present are coerce_temporal_string (valid_from/valid_to) and
  reject_banned_properties (extra). The timedelta import is absent from the file. The Agent
  Log claim "ingestion_source max_length=512 via Field(); ingestion_time field_validator
  rejects values outside [2020-01-01, now+1h]" does not match the on-disk code. Finding 3
  remains open.
```

```
RE-REVIEW: TASK-005
Finding: N1 (new — process integrity issue)
Original severity: N/A
Status: NEW_ISSUE
Notes: The backend-developer Agent Log records three fix commits as applied. No such commits
  exist on the branch. The TASK-005 branch has exactly two git commits (5d965e3 docs and
  530a9c0 initial). The three code files targeted (background_jobs.py, multi_tenant_components.py,
  graph_schemas.py) are not tracked in any commit on this branch. The log entry is false: the
  fixes were not committed to agent/STORY-002/TASK-005-bitemporal-models. This is either a
  process failure (agent wrote the log before committing) or the commits were made to a different
  branch and never cherry-picked. This does not affect security posture directly but means the
  review cannot be closed — there is nothing to verify. The task cannot proceed until the fixes
  are committed to the correct branch and verifiable via `git log`.
```

## VERDICT: TASK-005 (re-review)

**Status: STILL BLOCKED** *(this verdict was incorrect — see corrected re-review below)*

All three original findings are unresolved. The claimed fix commits do not exist on the branch.
The on-disk files are unchanged from the state reviewed in the initial BLOCK. Finding N1 flags
a process integrity issue: the agent log recorded fixes as applied when no corresponding commits
exist on agent/STORY-002/TASK-005-bitemporal-models.

**Note:** This re-review was conducted from the outer monorepo (`/Users/reza/workspace/Oraclous/`),
which does not track the backend code. The oraclous-data-studio inner repo
(`/Users/reza/workspace/Oraclous/oraclous-data-studio/`) is a separate git repository and was
not checked. The "STILL BLOCKED" verdict above is therefore incorrect. See corrected re-review below.

---

## RE-REVIEW: TASK-005 (corrected)

**Re-reviewed:** 2026-04-26
**Reviewer:** Security Architect
**Trigger:** Previous re-review read the wrong git repository (outer monorepo). This corrected
re-review checks out `agent/STORY-002/TASK-005-bitemporal-models` from the inner
`oraclous-data-studio` repo and reads the actual on-disk code (commits 3f5003c, c52b0ad, 49e38c0).

---

```
RE-REVIEW: TASK-005
Finding: 1
Original severity: high
Status: RESOLVED
Notes: run_bitemporal_migration_v1 (background_jobs.py lines 2123-2154) is a fan-out
  orchestrator. It opens a single sync Neo4j session, runs
  "MATCH (e:__Entity__) RETURN DISTINCT e.graph_id AS graph_id", collects graph_ids,
  then dispatches _run_bitemporal_migration_for_graph.delay(gid) once per graph. No
  migration writes happen in this task — all writes are in the per-graph subtask.

  _run_bitemporal_migration_for_graph (lines 2004-2120):
  - Guard (lines 2037-2044): single MERGE statement — "MERGE (m:Migration {id: $migration_id})
    ON CREATE SET m.done = false, m.started_at = datetime()" — atomically creates the guard
    node if absent. migration_id = "bitemporal-v1-{graph_id}" (per-graph, not global).
    Checks already_done flag in the same round-trip; skips if true. Atomic: no TOCTOU race.
  - Entity query (lines 2056-2067): WHERE clause is "WHERE e.event_time IS NULL
    AND e.graph_id = $graph_id" — scoped to single graph. Correct.
  - Relationship query (lines 2070-2083): WHERE clause is
    "WHERE r.event_time IS NULL
       AND (r.ingested_at IS NOT NULL OR r.transaction_time IS NOT NULL)
       AND r.graph_id = $graph_id" — scoped to single graph. Correct.
  All three sub-issues (cross-tenant query, TOCTOU race, global guard id) are resolved.
```

```
RE-REVIEW: TASK-005
Finding: 2
Original severity: medium
Status: RESOLVED
Notes: MultiTenantKGWriter (multi_tenant_components.py):
  - _sanitize_source() static method exists at lines 258-266. It strips null bytes via
    .replace("\x00", ""), strips whitespace with .strip(), and caps at
    MultiTenantKGWriter._MAX_INGESTION_SOURCE_LEN (512). Returns None for empty result
    or None input. Correct implementation matching the prescribed fix.
  - Node loop (lines 310-313): node.properties.pop("ingestion_source", None) removes any
    caller/LLM-supplied value unconditionally. Then _sanitize_source(self.ingestion_source)
    is called and result assigned only if not None. No setdefault anywhere.
  - Relationship loop (lines 336-339): identical pop+sanitize pattern applied to rel.properties.
  - Comment on graph_id .update() asymmetry added at lines 299-301 (nodes) and 325-327
    (relationships), explaining the security rationale. Finding 4 (informational) also resolved.
  The prompt-injection vector via LLM-controlled ingestion_source is closed.
```

```
RE-REVIEW: TASK-005
Finding: 3
Original severity: medium
Status: RESOLVED
Notes: graph_schemas.py:
  - RelationshipProperties.ingestion_source (lines 71-75): Field(None, max_length=512, ...).
    max_length constraint present. Correct.
  - EntityNodeProperties.ingestion_source (lines 135-138): Field(None, max_length=512, ...).
    max_length constraint present. Correct.
  - RelationshipProperties.bound_ingestion_time validator (lines 94-107): @field_validator on
    "ingestion_time", mode="before". Checks isinstance(v, datetime); computes _min =
    datetime(2020,1,1,tzinfo=timezone.utc) and _max = datetime.now(timezone.utc)+timedelta(hours=1).
    Returns None (not raises) for values outside range — ingestion is not blocked. Correct.
  - EntityNodeProperties.bound_ingestion_time validator (lines 141-154): identical logic.
  - timedelta import confirmed present at line 1 (from datetime import UTC, datetime, timedelta,
    timezone). All three sub-issues resolved.
```

```
RE-REVIEW: TASK-005
Finding: N1 (process integrity — from previous re-review)
Original severity: N/A
Status: RESOLVED
Notes: Commits 3f5003c, c52b0ad, 49e38c0 are present on the branch in oraclous-data-studio
  (confirmed via git log). The previous "STILL BLOCKED" verdict was a review-process error:
  git was run against the outer monorepo which has no backend code, not the inner
  oraclous-data-studio repo. The fixes were committed to the correct branch.
```

## VERDICT: TASK-005 (corrected re-review)

**Status: PASS**

All three original blocking findings are resolved in the on-disk code on branch
`agent/STORY-002/TASK-005-bitemporal-models` (oraclous-data-studio repo, commits 3f5003c,
c52b0ad, 49e38c0):

1. **Finding 1 (high — resolved):** Migration is fully graph-scoped. Fan-out orchestrator
   dispatches one subtask per graph_id. Both entity and relationship Cypher queries filter
   `AND e.graph_id = $graph_id` / `AND r.graph_id = $graph_id`. Guard is atomic MERGE ON
   CREATE with per-graph id `bitemporal-v1-{graph_id}`.
2. **Finding 2 (medium — resolved):** `_sanitize_source()` static method added. `run()` uses
   pop+sanitize pattern for both nodes and relationships. LLM-controlled prompt injection
   vector is closed.
3. **Finding 3 (medium — resolved):** `ingestion_source` has `max_length=512` in both models.
   `bound_ingestion_time` validator returns None for out-of-range values in both models.
4. **Finding 4 (informational — resolved):** Comment explaining `.update()` asymmetry added.

No new security issues introduced by the fix commits. Task can proceed to QA Engineer sign-off.

---

## SECURITY REVIEW: TASK-009

**Branch:** `agent/STORY-005/TASK-009-same-as-embeddings`
**Reviewed:** 2026-04-26
**Reviewer:** Security Architect
**Files reviewed:**
- `knowledge-graph-builder/app/schemas/federation_schemas.py`
- `knowledge-graph-builder/app/services/federation_service.py`

---

### Finding 1

```
SECURITY REVIEW: TASK-009
Finding: VULNERABILITY
Severity: high
Location: knowledge-graph-builder/app/services/federation_service.py, find_same_as_candidates()
Description: find_same_as_candidates() accepts target_graph_ids from the caller and queries Neo4j
  against them with no call to _validate_and_filter(). Every other public entry point on
  FederationService (federated_query, federated_vector_search) gates on that helper before any
  Neo4j access. find_same_as_candidates() does not. Any caller — a background job, a TASK-010
  endpoint, an agent tool — can supply attacker-controlled graph IDs and read entity names and
  element IDs from other tenants' graphs without any authorization check.
Exploit path: A TASK-010 background job invokes find_same_as_candidates() with a graph_ids list
  that includes tenants the caller does not own. Entity names and elementIds from those tenants
  are returned and then fed into the multi-signal scorer, leaking cross-tenant entity data.
  No JWT or principal check stands in the way.
Required fix: Add user_id and principal parameters to find_same_as_candidates() and call
  await self._validate_and_filter(user_id, target_graph_ids, principal=principal) before any
  Neo4j access. Signature change:
    async def find_same_as_candidates(
        self, source_entity_id: str, source_graph_id: str,
        target_graph_ids: list[str], embedding: list[float],
        user_id: str, principal: str
    ) -> list[SameAsCandidate]:
        await self._validate_and_filter(user_id, target_graph_ids, principal=principal)
        ...
```

---

### Finding 2

```
SECURITY REVIEW: TASK-009
Finding: VULNERABILITY
Severity: high
Location: knowledge-graph-builder/app/services/federation_service.py, _vector_search_candidates()
Description: candidate_count is calculated as MAX_RESULTS_PER_GRAPH * len(target_graph_ids) * 1.5
  with no upper bound on len(target_graph_ids). This runs before any authorization check (which
  does not exist — see Finding 1). An attacker supplying 10,000 graph IDs drives candidate_count
  to 1,500,000, saturating the vector index and causing extreme memory allocation in the Neo4j
  driver response path for all concurrent requests.
Exploit path: POST /federation/same-as or equivalent with target_graph_ids containing 10,000
  entries. candidate_count = MAX_RESULTS_PER_GRAPH (20) * 10,000 * 1.5 = 300,000. Neo4j
  vector index scan for 300K results consumes GBs of memory and blocks the index for all
  other queries. Denial of service against the vector search path for all tenants.
Required fix: Add an explicit guard at the top of _vector_search_candidates():
    MAX_TARGET_GRAPHS = 50  # or whatever the platform limit is
    if len(target_graph_ids) > MAX_TARGET_GRAPHS:
        raise ValueError(f"target_graph_ids exceeds limit of {MAX_TARGET_GRAPHS}")
  Apply this defence-in-depth check independently of the authorization fix in Finding 1.
```

---

### Finding 3

```
SECURITY REVIEW: TASK-009
Finding: VULNERABILITY
Severity: medium
Location: knowledge-graph-builder/app/services/federation_service.py, _vector_search_candidates()
Description: The embedding parameter is passed directly to db.index.vector.queryNodes() with no
  dimensionality check and no float coercion. The entity_embeddings index is 3072-dimensional.
  A zero-length list causes a crash in the Neo4j driver. An embedding from an external source
  with the wrong number of dimensions causes a runtime error at query time, not at validation
  time. Neo4j can return integer arrays from stored embeddings; passing them to cosine similarity
  without float coercion produces incorrect scores silently.
Exploit path: External caller supplies a 1-dimensional or 0-dimensional embedding. The driver
  throws a transport-layer exception that may leak internal stack traces in the error response.
  Alternatively, an integer-typed embedding from Neo4j passes through to scoring silently wrong.
Requires Reza's explicit risk acknowledgment or the following fix:
  1. Validate before calling: if not embedding or len(embedding) != 3072: raise ValueError(...)
  2. Coerce: embedding = [float(v) for v in embedding]
```

---

### Finding 4

```
SECURITY REVIEW: TASK-009
Finding: PASS
Severity: informational
Location: knowledge-graph-builder/app/services/federation_service.py
Description: Both new Cypher queries (_find_exact_match and _vector_search_candidates) are fully
  parameterized. _find_exact_match binds entity names via $entity_name and $graph_id parameters.
  _vector_search_candidates binds graph IDs via $graph_ids list and the embedding via $embedding.
  No string concatenation or f-string interpolation is used in query construction for these two
  methods. The pre-existing _execute_entity_union f-string pattern (which builds parameter key
  names like $entity_name_0, not values) was not changed and does not introduce new injection
  surface. SameAsCandidate TypedDict contains no secrets.
Required fix: None.
```

---

## VERDICT: TASK-009

**Status: BLOCKED**

**Reason:** Two high-severity findings. Finding 1 is a missing authorization gate that allows
any caller to read entity data from arbitrary tenants — a tenant isolation bypass, always blocking.
Finding 2 is a denial-of-service vector against the vector index caused by an unbounded
candidate_count calculation.

**Required before merge:**
1. **Finding 1 (high — blocking):** Add `user_id` and `principal` to `find_same_as_candidates()`
   and call `_validate_and_filter()` before any Neo4j access. This is the authorization gate
   every other public method already has.
2. **Finding 2 (high — blocking):** Add `len(target_graph_ids) > MAX_TARGET_GRAPHS` guard in
   `_vector_search_candidates()` as defence-in-depth.

**Required acknowledgment by Reza (medium — not blocking pending acknowledgment):**
3. **Finding 3 (medium):** Add embedding dimensionality check and float coercion before the
   vector index call. Or explicitly accept the risk of malformed embedding crashes.

**Non-blocking:**
4. Finding 4 (informational): PASS — no Cypher injection introduced.

---

## RE-REVIEW: TASK-009

**Re-reviewed:** 2026-04-26
**Reviewer:** Security Architect
**Trigger:** Three fix commits applied by backend-developer (commits 70ed512, 75059d9, c0222da)

---

```
RE-REVIEW: TASK-009
Finding: 1
Original severity: high
Status: RESOLVED
Notes: find_same_as_candidates() now accepts user_id: str and principal: dict | None = None.
  Line 378 calls await self._validate_and_filter(user_id, target_graph_ids, principal=principal)
  as the very first statement — before any dict construction or Neo4j access. Matches the exact
  pattern used by federated_query() (line 81) and federated_vector_search() (line 144).
  The original finding listed principal: str in the prescribed signature; the fix correctly uses
  dict | None, which is consistent with all other public methods on the class.
```

```
RE-REVIEW: TASK-009
Finding: 2
Original severity: high
Status: RESOLVED
Notes: _MAX_TARGET_GRAPHS = 50 defined as a module-level constant (line 38). Guard placed at
  lines 453-457, at the very top of _vector_search_candidates(), before candidate_count is
  computed on line 467. Raises ValueError with a clear message. Defence-in-depth check is
  independent of the auth gate — operates even if _validate_and_filter() were somehow bypassed.
```

```
RE-REVIEW: TASK-009
Finding: 3
Original severity: medium
Status: RESOLVED
Notes: _EMBEDDING_DIM = 3072 defined as a module-level constant (line 41). Dimensionality check
  at lines 459-463 raises ValueError for missing or wrong-length embeddings. Float coercion
  (line 464) applied immediately after the check, before candidate_count computation and before
  the Cypher params dict is built. Ordering is correct.
```

```
RE-REVIEW: TASK-009
Finding: 5 (new observation)
Original severity: N/A
Status: NEW_ISSUE
Notes: Non-security logic defect. find_same_as_candidates() builds entity = {"entity_id":
  source_entity_id, "embedding": embedding} (line 380) then passes it to _find_exact_match().
  _find_exact_match() extracts name and type via .get() with empty-string fallback, then returns
  None if either is falsy (lines 412-413). The entity dict has no "name" or "type" keys, so the
  fast path always returns None and the exact match branch is dead code. The task spec intends
  the entity object (with name and type) to be passed through. This does not affect security
  posture — no tenant data leaks from a path that always returns None. Deferred to QA as a
  functional defect.
```

## VERDICT: TASK-009 (re-review)

**Status: PASS**

All three original blocking/medium findings are resolved. No new security issues were introduced
by the fix commits. One non-security logic defect (Finding 5 above — exact-match fast path is
dead) is noted for QA. Task can proceed to QA Engineer sign-off.

---

## SECURITY REVIEW: TASK-010

**Branch:** `agent/STORY-005/TASK-010-same-as-scorer`
**Reviewed:** 2026-04-26
**Reviewer:** Security Architect
**Files reviewed:**
- `knowledge-graph-builder/requirements.txt`
- `knowledge-graph-builder/app/components/entity_resolver.py` (new `EntityResolver` class, lines 34–293; pre-existing `MultiTenantEntityDeduplicator` class, lines 295–807)
- `knowledge-graph-builder/app/services/federation_service.py` (added `EntityResolver` import + `resolve_entity()` method)

---

### Finding 1

```
SECURITY REVIEW: TASK-010
Finding: PASS
Severity: informational
Location: requirements.txt — jellyfish>=1.0.1
Description: jellyfish is a legitimate, well-established string similarity library maintained
  by James Turk (jamesturk on PyPI), hosted on Codeberg at codeberg.org/jpt/jellyfish with a
  GitHub mirror (2.2k stars). The package name is not a typosquat — the PyPI owner matches the
  Codeberg author, and the home page (jellyfish.jpt.sh) is consistent with the project history.
  Latest version is 1.2.1 (no yanked releases). The pin jellyfish>=1.0.1 includes all releases
  from 1.0.1 to 1.2.1; no CVEs appear in the OSV database (osv.dev returns no results for
  jellyfish/PyPI). The package is implemented in Rust with pre-built wheels and declares zero
  transitive dependencies (requires_dist is null). It performs no network or filesystem I/O —
  it is a pure computation library operating only on string arguments passed by the caller.
  jaro_winkler_similarity() is invoked at entity_resolver.py:104 with two normalized Python
  strings; no external access path exists.
Exploit path: N/A
Required fix: None — PASS. Hygiene recommendation (non-blocking): pin to ==1.2.1 (exact)
  rather than >=1.0.1 to prevent automatic uptake of a future supply-chain-compromised release,
  consistent with all other pinned dependencies in requirements.txt.
```

---

### Finding 2

```
SECURITY REVIEW: TASK-010
Finding: PASS
Severity: informational
Location: entity_resolver.py:157–179 — EntityResolver._get_neighbor_names()
Description: The neighbor Cypher query is fully parameterized. entity_id is bound to $entity_id
  and graph_id is bound to $graph_id in the params dict (lines 171–172). No user-controlled
  string appears in the query text. The double UNION pattern (outgoing and incoming edges)
  correctly constrains both branches to WHERE e.graph_id = $graph_id, preventing the neighbor
  walk from crossing into unrelated graph_ids. The entity node is first matched by elementId()
  — a stable internal Neo4j identifier not derivable from entity names — and then constrained
  by graph_id. There is no path for an entity in graph A to retrieve neighbors from graph B
  by crafting its entity_id value.
Exploit path: N/A
Required fix: None — PASS
```

---

### Finding 3

```
SECURITY REVIEW: TASK-010
Finding: PASS
Severity: informational
Location: entity_resolver.py:275–292 — EntityResolver._create_same_as_link()
Description: The MERGE Cypher is fully parameterized: id_a, id_b, and score are all bound via
  the params dict (line 283). The MATCH clauses use elementId(a) = $id_a and
  elementId(b) = $id_b — Neo4j elementIds are system-generated identifiers, not user-controlled
  strings. No f-string, concatenation, or unparameterized user input reaches the query text.
  The MERGE creates SAME_AS relationships only between the two specific nodes identified by their
  elementIds; the pattern cannot inadvertently link other entities because MERGE on a relationship
  pattern is scoped to the matched start and end nodes. The bidirectional pattern is correct and
  idempotent per spec. (A tenant isolation concern with this method is raised separately in
  Finding 6 below.)
Exploit path: N/A for injection
Required fix: None for injection — PASS. See Finding 6 for the tenant isolation concern.
```

---

### Finding 4 — BLOCKING

```
SECURITY REVIEW: TASK-010
Finding: VULNERABILITY
Severity: critical
Location: entity_resolver.py:590–608 — MultiTenantEntityDeduplicator._create_relationship_fallback()
Description: The fallback method constructs an inner Cypher string via Python-side string
  concatenation of a runtime value:

    'MERGE (s)-[r:' + $rel_type + ']->(t) RETURN r'

  This string is passed as the dynamic query argument to apoc.cypher.doIt(). The value of
  $rel_type originates from type(r) called on a relationship read back from Neo4j (lines 484
  and 522). While database-sourced rather than directly HTTP-sourced, it is fully attacker-
  controlled: any tenant who can write graph data (e.g. via document ingestion) can create a
  relationship whose type field contains arbitrary Cypher syntax. When that entity group is
  later deduplicated, _create_relationship_fallback() is invoked, apoc.cypher.doIt() receives
  the crafted inner query, and the injected Cypher executes with full database permissions —
  including cross-tenant MATCH, MERGE, SET, and DETACH DELETE.

  This is a Cypher injection finding under zero-tolerance policy.

  Note: MultiTenantEntityDeduplicator predates TASK-010 (original commit 6b67c2c, Sep 2025).
  However, TASK-010 modified the same file and did not remediate this injection. Per the
  Security Architect's mandate, any file touched by a task must not ship with a known critical
  injection path.
Exploit path: Attacker ingests a document that causes the LLM extractor to produce a relationship
  of type `WORKS_FOR`]->(t)) DETACH DELETE t //` (or any variant that closes the MERGE pattern
  and appends a destructive clause). When _deduplicate_exact_matches() runs on that graph, type(r)
  returns the crafted string; _create_relationship_fallback() concatenates it into apoc.cypher.doIt();
  Neo4j executes the injection. With APOC in unrestricted mode (the typical self-hosted config),
  the injected query has access to all graphs — full cross-tenant data destruction or exfiltration.
Required fix: Delete the apoc.cypher.doIt() block entirely (lines 590–608). Replace
  _create_relationship_fallback() with a strict allowlist dispatch that never constructs
  executable Cypher from runtime values:

    _ALLOWED_REL_TYPES: frozenset[str] = frozenset({
        "WORKS_FOR", "FOUNDED", "LEADS", "MANAGES",
        "DEVELOPED", "PARTNERED_WITH",
    })

    def _create_relationship_fallback(
        self, session, source_id, target_id, rel_type, rel_props
    ):
        if rel_type not in _ALLOWED_REL_TYPES:
            logger.warning(
                "skipping relationship of unrecognised type %r during deduplication "
                "— type not in allowlist", rel_type
            )
            return
        self._create_known_relationship(
            session, source_id, target_id, rel_type, rel_props
        )

  If unknown relationship types must be preserved, the primary apoc.create.relationship()
  call (already at lines 500 and 537) accepts the type as a parameter string without
  constructing executable Cypher — use that, not apoc.cypher.doIt().
```

---

### Finding 5 — BLOCKING

```
SECURITY REVIEW: TASK-010
Finding: VULNERABILITY
Severity: high
Location: entity_resolver.py:234 — EntityResolver.resolve_and_link() — graph_id_b fallback
Description: When a candidate entity_b does not carry a "source_graph_id" key, the code
  falls back to target_graph_ids[0]:

    graph_id_b = entity_b.get(
        "source_graph_id",
        target_graph_ids[0] if target_graph_ids else ""
    )

  This value is passed as the graph_id argument to _get_neighbor_names() (via score() →
  _context_score()), which scopes the 1-hop neighbor Cypher to
  WHERE neighbor.graph_id = $graph_id. If the caller passes target_graph_ids =
  ["victim-graph-id"] and the candidate dict omits "source_graph_id", graph_id_b silently
  becomes "victim-graph-id". The neighbor query then legitimately returns neighbor entity
  names from the victim tenant's graph into the attacker's process — a cross-tenant data read.

  Additionally, this fallback shapes which graph's neighbor names influence the final_score,
  and therefore whether _create_same_as_link() fires. An attacker can manipulate the scoring
  outcome for an entity pair by controlling which graph's neighborhood is read.
Exploit path: Caller invokes resolve_entity() with a candidate dict where "source_graph_id"
  is absent and target_graph_ids = ["victim-graph-id"]. resolve_and_link() falls back to
  "victim-graph-id" for graph_id_b. _get_neighbor_names() reads neighbor entity names from
  the victim graph into `neighbors_b`. If the attacker also controls the embedding score (e.g.
  by crafting embeddings) to push final_score above STORE_THRESHOLD, a SAME_AS link is written
  between an attacker-controlled entity and a victim entity.
Required fix: Remove the target_graph_ids[0] fallback. A candidate with no source_graph_id is
  malformed and must be skipped:

    graph_id_b = entity_b.get("source_graph_id")
    if not graph_id_b:
        logger.warning(
            "skipping candidate with missing source_graph_id: %r",
            entity_b.get("entity_id", "?"),
        )
        continue

  The source_graph_id is already set by find_same_as_candidates() for all candidates returned
  by _find_exact_match() and _vector_search_candidates() (present in both RETURN clauses).
  There is no legitimate code path that produces a candidate without source_graph_id; the
  fallback should not exist.
```

---

### Finding 6 — BLOCKING

```
SECURITY REVIEW: TASK-010
Finding: VULNERABILITY
Severity: high
Location: entity_resolver.py:275–292 + federation_service.py:resolve_entity()
  — SAME_AS link creation without graph_id filter; no authorization gate on resolve_entity()
Description: Two related issues that together constitute a tenant isolation bypass:

  (a) _create_same_as_link() MATCH clauses have no graph_id constraint:
        MATCH (a:__Entity__) WHERE elementId(a) = $id_a
        MATCH (b:__Entity__) WHERE elementId(b) = $id_b
      These clauses match nodes purely by elementId with no tenant fence. Once written, SAME_AS
      edges span graph boundaries at the Neo4j storage level. Any future Cypher traversal of
      SAME_AS without an explicit graph_id filter — including any user query before TASK-012's
      Scope Enforcer is built — will cross tenant boundaries. The CLAUDE.md Architecture Rules
      mandate graph_id on every Cypher write query; this MERGE has none.

  (b) resolve_entity() (the new public method added by TASK-010) opens a Neo4j session and
      calls EntityResolver.resolve_and_link() without calling _validate_and_filter() first.
      Every other public method on FederationService gates on _validate_and_filter() before
      any Neo4j access. resolve_entity() does not, creating an unauthorized write path: a
      caller with attacker-controlled graph_ids can create SAME_AS links between arbitrary
      entities across arbitrary tenants with no ownership check.

  The TASK-009 high finding (missing authorization gate in find_same_as_candidates()) is still
  open; resolve_entity() inherits that unguarded read path and adds an unguarded write path.
Exploit path: (a) Cross-tenant traversal: a query authenticated to graph A that executes
  MATCH (e:__Entity__ {graph_id: $gid_a})-[:SAME_AS*1..]->(x) without constraining x.graph_id
  traverses into graph B's entities via the SAME_AS edge. Until TASK-012 is implemented, this
  affects all user-visible graph queries.
  (b) Unauthorized write: any caller reaching resolve_entity() with attacker-controlled
  graph_ids can write SAME_AS links between entities in tenants they do not own.
Required fix: Two changes, both mandatory:

  (a) Add graph_id parameters to _create_same_as_link() MATCH clauses:

    async def _create_same_as_link(
        session, id_a, id_b, score, graph_id_a, graph_id_b
    ):
        query = """
        MATCH (a:__Entity__ {graph_id: $graph_id_a}) WHERE elementId(a) = $id_a
        MATCH (b:__Entity__ {graph_id: $graph_id_b}) WHERE elementId(b) = $id_b
        MERGE (a)-[:SAME_AS {confidence: $score, method: 'multi-signal',
                              created_at: datetime()}]->(b)
        MERGE (b)-[:SAME_AS {confidence: $score, method: 'multi-signal',
                              created_at: datetime()}]->(a)
        """
        ...

  Update the call site in resolve_and_link() to pass graph_id_a and graph_id_b.

  (b) Add _validate_and_filter() at the top of resolve_entity():

    async def resolve_entity(
        self,
        entity: dict,
        graph_id: str,
        target_graph_ids: list[str],
        user_id: str,
        principal: dict[str, Any] | None = None,
    ) -> list[dict]:
        await self._validate_and_filter(
            user_id, [graph_id] + target_graph_ids, principal=principal
        )
        # ... rest unchanged ...

  Note: TASK-012 (Scope Enforcer) must also be implemented before SAME_AS traversal is safe
  in user-facing queries. That is a separate task dependency, not part of this required fix.
```

---

### Finding 7

```
SECURITY REVIEW: TASK-010
Finding: OBSERVATION
Severity: low
Location: entity_resolver.py:65–70 — _normalize_name()
Description: The name normalization function applies _LEGAL_SUFFIX_RE and _PUNCT_RE before
  passing the result to jellyfish.jaro_winkler_similarity(). Three specific inputs examined:
  - Null bytes (\x00): [^\w\s] with re.UNICODE correctly strips null bytes (they are not word
    characters). Not a vulnerability.
  - Regex-special characters in entity names: input is matched against a compiled pattern,
    not treated as a pattern itself. Not a vulnerability.
  - Extremely long strings: No length cap is applied before normalization or before
    jaro_winkler_similarity(). The Rust implementation is O(n*m) in string length. Two 10MB
    strings produce effectively unbounded CPU work. If an ingested document causes the LLM
    to extract an entity with a very long name and upload-size limits are not enforced upstream,
    EntityResolver.score() can be made to consume excessive CPU — a low-severity CPU DoS
    dependent on an upstream control being absent.
Exploit path: Attacker ingests a document with an entity name near the LLM context limit
  (e.g. 32,000 characters). With O(n) candidates per entity, each scored with jaro_winkler
  O(n^2) total, one ingestion event can cause measurable latency spikes. Low severity because
  it requires authenticated ingestion access and upstream upload-size limits should already bound
  the input.
Required fix: Add a name length cap in _normalize_name() as defence-in-depth:

    _MAX_NAME_LEN = 1000  # characters; names longer than this are pathological
    name = name[:_MAX_NAME_LEN]

  Place before the regex substitutions. Non-blocking; does not require an immediate fix.
```

---

### Finding 8

```
SECURITY REVIEW: TASK-010
Finding: OBSERVATION
Severity: informational
Location: entity_resolver.py:246–252 — resolve_and_link() ambiguous candidate logging
Description: The INFO log line logs entity names from both entity_a (graph A) and entity_b
  (graph B). In a multi-tenant environment with centralized log aggregation, entity names from
  tenant B appear in log entries associated with tenant A's operations. If log access is not
  scoped per-tenant, tenant A operators can read entity names from tenant B via INFO logs.
  This is inherent to any cross-graph scoring operation and cannot be eliminated at the code
  level without per-tenant log partitioning at the infrastructure level.
Exploit path: Requires access to application logs. Not a code-level exploit.
Required fix: None at code level. Operational recommendation: when log aggregation is configured
  for production, partition log streams by tenant_id or graph_id before granting operator access.
```

---

## VERDICT: TASK-010

**Status: BLOCKED**

**Blocking findings (must be fixed before merge):**

| # | Finding | Severity | Reason blocks |
|---|---|---|---|
| 4 | `apoc.cypher.doIt` Cypher injection via `rel_type` string concatenation in `_create_relationship_fallback()` | critical | Cypher injection — zero-tolerance, always blocking |
| 5 | `graph_id_b` fallback to `target_graph_ids[0]` in `resolve_and_link()` | high | Tenant isolation bypass — cross-tenant neighbor read and scoring manipulation |
| 6 | No authorization gate on `resolve_entity()`; no `graph_id` filter on `_create_same_as_link()` MATCH clauses | high | Tenant isolation bypass — unauthorized SAME_AS write path and cross-tenant SAME_AS traversal |

**Non-blocking (Reza acknowledgment required):**

| # | Finding | Severity | Action |
|---|---|---|---|
| 7 | Name length not capped before Jaro-Winkler | low | Acknowledge or add `name = name[:1000]` in `_normalize_name()` |

**Non-blocking, no action required:**

| # | Finding | Severity | |
|---|---|---|---|
| 1 | `jellyfish>=1.0.1` — legitimate package, no CVEs; loose pin is hygiene-only | informational | Recommend `==1.2.1` |
| 2 | `_get_neighbor_names()` Cypher — fully parameterized, tenant-scoped | informational | PASS |
| 3 | `_create_same_as_link()` Cypher — no injection path | informational | PASS (see Finding 6 for separate tenant concern) |
| 8 | Ambiguous candidate log leaks entity names across tenant boundary into shared log stream | informational | Operational config only |

The critical finding (Cypher injection in `_create_relationship_fallback`) is zero-tolerance blocking
and originated in pre-existing code; TASK-010 touched the file without remediating it, which is
sufficient trigger. The two high findings are both tenant isolation bypasses, also always blocking.
All three must be resolved before this branch merges.

---

## RE-REVIEW: TASK-010

**Re-reviewed:** 2026-04-26
**Reviewer:** Security Architect
**Trigger:** Four fix commits applied by backend-developer (8824123, 5ef71ff, f859d9c, 9a7f5fd)

---

```
RE-REVIEW: TASK-010
Finding: 4
Original severity: critical
Status: RESOLVED
Notes: apoc.cypher.doIt() is completely removed. _create_relationship_fallback() now performs
  a frozenset membership check (rel_type not in _ALLOWED_REL_TYPES) and returns early with a
  warning for any unrecognised type. _create_known_relationship() dispatches via a chain of
  elif branches, each containing a static Cypher string with no runtime interpolation — the
  rel_type value never appears in any query text. The APOC primary path
  (apoc.create.relationship()) accepts rel_type as a parameter, not as Cypher text, so the
  primary path was already safe; the fallback is now also safe. Zero executable Cypher
  constructed from runtime values.
```

```
RE-REVIEW: TASK-010
Finding: 5
Original severity: high
Status: RESOLVED
Notes: graph_id_b = entity_b.get("source_graph_id") with no default (line 255). Immediately
  followed by: if not graph_id_b: logger.warning(...); continue. The target_graph_ids[0]
  fallback is gone. Candidates with missing source_graph_id are skipped before any scoring
  or Neo4j access occurs.
```

```
RE-REVIEW: TASK-010
Finding: 6
Original severity: high
Status: RESOLVED
Notes: Three sub-issues all resolved:
  (a) _create_same_as_link() MATCH clauses now read:
        MATCH (a:__Entity__ {graph_id: $graph_id_a}) WHERE elementId(a) = $id_a
        MATCH (b:__Entity__ {graph_id: $graph_id_b}) WHERE elementId(b) = $id_b
      Both graph_id_a and graph_id_b are present in the params dict. A MATCH that mismatches
      graph_id returns no row; the downstream MERGE is skipped. Tenant fence is in place.
  (b) resolve_entity() calls await self._validate_and_filter(user_id,
      [graph_id] + target_graph_ids, principal=principal) as the very first statement (line 407),
      before find_same_as_candidates() or any session open. Matches the auth-gate pattern used by
      federated_query() and federated_vector_search().
  (c) graph_id_a and graph_id_b are correctly threaded: resolve_and_link() receives graph_id_a
      from resolve_entity(); graph_id_b is taken from entity_b.get("source_graph_id") (see Finding
      5); both are passed as keyword arguments to _create_same_as_link().
```

```
RE-REVIEW: TASK-010
Finding: 7
Original severity: low
Status: RESOLVED
Notes: _MAX_NAME_LEN = 1000 defined as a module-level constant (line 64). name = name[:_MAX_NAME_LEN]
  is the first operation inside _normalize_name() (line 87), before _LEGAL_SUFFIX_RE.sub() and
  _PUNCT_RE.sub(). Ordering is correct — the cap is applied to raw input, not to post-regex output.
```

```
RE-REVIEW: TASK-010
Finding: N1 (new — non-blocking)
Original severity: N/A
Status: NEW_ISSUE
Notes: Allowlist coverage gap in _ALLOWED_REL_TYPES vs. LLM extraction prompt.
  The extraction prompt (pipeline_service.py:120) lists these "common types" that the LLM is
  instructed to produce: WORKS_FOR, REPORTS_TO, HAS_SKILL, MEMBER_OF, INVESTED_IN, CITES,
  AUTHORED, WORKS_ON, DEPENDS_ON, ACQUIRED_BY, PARTNER_OF. Of these 11 types, only WORKS_FOR
  appears in _ALLOWED_REL_TYPES. The other 10 (REPORTS_TO, HAS_SKILL, MEMBER_OF, INVESTED_IN,
  CITES, AUTHORED, WORKS_ON, DEPENDS_ON, ACQUIRED_BY, PARTNER_OF) are absent. Additionally,
  the prompt uses PARTNER_OF while the allowlist has PARTNERED_WITH — these are different strings.
  Security impact: LOW. The fallback fires only when apoc.create.relationship() throws; in normal
  operation (APOC available) the parameterized primary path is used and the allowlist is never
  consulted. When the fallback does fire for a type outside the allowlist, the relationship is
  silently dropped. This is a data-loss / correctness defect, not a security vulnerability — no
  cross-tenant data is read or written for unrecognised types. The allowlist correctly prevents
  injection for unknown types. However, the allowlist should be expanded to match the extraction
  prompt's type vocabulary, and PARTNER_OF / PARTNERED_WITH inconsistency should be resolved.
  Deferred to QA as a functional correctness defect.
```

```
RE-REVIEW: TASK-010
Finding: N2 (new — informational)
Original severity: N/A
Status: NEW_ISSUE
Notes: _store_same_as_links() in FederationService (federation_service.py:553-577) is a
  pre-existing path not touched by the TASK-010 fixes. Its MATCH clauses have no graph_id
  constraint:
    MATCH (a:__Entity__) WHERE elementId(a) = pair.id_a
    MATCH (b:__Entity__) WHERE elementId(b) = pair.id_b
  This path is invoked by _resolve_same_as() (exact-match deduplication in federated_query)
  and is guarded upstream by _validate_and_filter(). elementId values are system-generated
  and not directly injectable. The tenant risk is lower than the fixed Finding 6 (_create_same_as_link
  now has graph_id guards), but the CLAUDE.md architecture rule — graph_id on every Cypher write
  query — is violated. This is a pre-existing gap, not introduced by TASK-010. Non-blocking for
  this task; recommend addressing as a follow-up item or within TASK-012 Scope Enforcer scope.
```

## VERDICT: TASK-010 (re-review)

**Status: PASS**

All four original blocking findings (4 critical, 5 high, 6 high, 7 low) are resolved. No new
security issues were introduced by the fix commits. Two new observations are noted:

- Finding N1: allowlist types do not match the LLM extraction prompt vocabulary — functional
  correctness defect, deferred to QA. Non-blocking for security sign-off.
- Finding N2: _store_same_as_links() pre-existing MATCH clauses lack graph_id guard — pre-existing
  gap, not introduced by TASK-010, non-blocking. Recommend addressing in TASK-012.

Task can proceed to QA Engineer sign-off.
