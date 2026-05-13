# Retired: MCP-as-product-integration-surface (May 2026)

**Status:** abandoned 2026-05-13.
**What this used to be:** the assessment substrate (`/api/v1/assessments/*`,
the `app/mcp/` server + tools, `app/services/assessment_service.py`,
`app/scripts/seed_assessment_catalog.py`, the `:AssessmentRun` / `:Module` /
`:Finding` / `:Conflict` / `:Deliverable` node-label family, and the SSE
event broker) — built across SPRINT-001 and SPRINT-002 as the canonical
example of "let an external Claude Code orchestrator drive Oraclous via
MCP."

**Why it's gone:** the operator (Reza) declared the entire approach a bad
idea after a live fresh-user test session on 2026-05-13. **This was a
shitty decision** — both the original architectural choice to ship a
bespoke schema-per-use-case substrate, and the time spent building it
out before the design was stress-tested against real workflow.

Direct quote from the moment of the call:

> "I just decided to drop whatever we have done for using remote Oraclous
> with local claude, it is a shity idea."

And, earlier in the same session — the critique that should have prevented
the substrate from ever being shipped:

> "I was against the custom logic, api, endpoints, models, tables,
> schemas, etc. due to the fact that for any new use case we shouldn't be
> defining the things. That proved that oraclous is not scaleable, in the
> age of agentic ai why we have to be repeating ourselfs?"

## Why the test session forced the call

A single fresh-user run from a clean local Docker stack hit, in order:

1. **TASK-090** — every assessment / onboarding REST route was mounted at
   `/api/v1/api/v1/...` because both `main.py` and `app/api/v1/router.py`
   added a `/api/v1` prefix. Operator had to discover the workaround.
2. **TASK-091** — the MCP server used a static `ORACLOUS_API_KEY` env var
   for all tool calls, ignoring the per-session Bearer header from the
   SSE handshake. Worked only single-tenant; broke "share with a friend."
3. **TASK-092** — every MCP tool wrapper exposed `body: dict[str, Any]`,
   so the published JSON schema told the client nothing about the body
   shape. The orchestrator had to grep the codebase to figure out how to
   call `create_run`. Wouldn't work for any client without source access.
4. **TASK-093** — the MCP container didn't initialize its Neo4j async
   driver; first iteration of the fix put `connect()` in FastMCP's
   lifespan, which fires *per SSE session*, so the docker healthcheck
   tore the pool down every 30 seconds and tool calls failed
   intermittently. Second iteration moved init to process scope.
5. **TASK-094** — the catalog seed script reads module prompts from
   `~/.claude/skills/...` on the operator's host. The Docker container
   has no such directory. `docker compose down -v` wipes the catalog
   and there's no portable re-seed path.
6. **Stale-session symptom** — after the MCP container restarted (each
   fix above triggered one), the Claude Code MCP client kept sending
   tool calls against a session ID the server no longer recognized,
   returning `-32602 INVALID_PARAMS` for every call. Confused diagnosis
   for ~30 minutes.
7. **Template-slug guesswork** — even with typed schemas (TASK-092), the
   orchestrator had to *guess* what template slugs were valid because
   templates were seeded data, not runtime-discoverable. It guessed
   `generic-ai-adoption`; the actual seeded slugs were `assess-v1` and
   `eurail-report-v1`.

Every one of these was a symptom of the same root cause: a closed,
bespoke substrate where every gap becomes a tax on the operator. None of
them happen in a platform where agents define their own workflows + data
shapes + tools at runtime.

## What's been removed from the codebase

This commit deletes:

- `app/mcp/` (the entire MCP server, including the legacy graph CRUD
  tools — the whole pattern is rejected, not just the assessment slice)
- `app/api/v1/endpoints/assessments.py`, `assessments_reads.py`,
  `assessments_sse.py`
- `app/services/assessment_service.py`, `blob_cas_service.py`,
  `assessment_event_broker.py`
- `app/schemas/assessment_schemas.py`
- `app/scripts/seed_assessment_catalog.py`,
  `backfill_assessment_run.py`, `dump_run_to_jsonl.py`,
  `live_rerun_simulation.py`
- `app/db/assessment_schema_init.py`
- `app/cypher/migrations/2026-05-11_assessment_schema.cypher`
- All assessment + MCP test files under `tests/unit/`,
  `tests/integration/`, `tests/qa/`, `tests/scripts/`, `tests/cypher/`
- The `knowledge-graph-mcp` service in `docker-compose.yml`
- The `ensure_assessment_schema` startup hook in `app/main.py`
- The three assessment router includes in `app/api/v1/router.py`

What's intentionally kept:

- Everything else in the L1 KG product (graph CRUD, chat, agents,
  communities, federation, memories, multimodal, code-knowledge-graph,
  permissions, webhooks, service-accounts, etc.)
- `auth-service` is untouched on `develop` — the `home_graph_id` JWT
  work, the onboarding bootstrap endpoint, and the `:Module` →
  `:CodeModule` rename all live on **unmerged feature branches**
  (`agent/STORY-027/TASK-08X-*` and `TASK-09X-*`). They are not on
  develop. If you want any of that infrastructure back later — for a
  different use case — cherry-pick from those branches.

## What this does NOT mean

It does not mean Oraclous itself is dead. The L1 multi-tenant Neo4j
substrate, ReBAC, auth, frontend (visual-flow), agent CRUD, chat — all
of it is intact. What's dead is the **specific product story** of "fresh
user installs Claude Code, points it at oraclous.dev via MCP, and runs
assessments through it." That story is in the bin.

## Open question for whoever picks this up next

The redesign principle Reza has been pushing — agent-defined runtime
structures, generic primitives instead of bespoke per-use-case schemas —
should be written up as a real ADR before any new substrate work
restarts. Until that ADR lands, the codebase is in a clean state:
nothing extending the dead pattern is committed.

The unmerged feature branches `agent/STORY-027/TASK-089` through
`TASK-093` contain four real bug-fixes that have nothing to do with
the substrate (doubled-prefix routing, per-request MCP auth, typed
schemas, process-scope DB init). If MCP-as-integration-surface ever
gets revisited, those branches are the starting point — not develop.
