# Oraclous — Claude Code Context

## What this project is

Oraclous is an open-source, self-hosted FTOps (Fine-Tuning DevOps) platform: a team of specialist agents that automates the full fine-tuning lifecycle over a graph-native knowledge substrate.

**Three-layer architecture:**
- Layer 1: Knowledge Graph — multi-tenant Neo4j substrate, ReBAC, zero-copy versioning, federation
- Layer 2: Harnessing Platform — graph-native agent runtime (BSP executor, APOC-triggered coordination, LLM gateway, HITL, MCP server/client, Python+TS SDK, skill files)
- Layer 3: FTOps — 16 specialist agents delivering the end-to-end fine-tuning loop

## Single Source of Truth

**Everything about this project lives in the wiki at `/Users/reza/workspace/Oraclous/wiki/`.**

Start every session by reading `/Users/reza/workspace/Oraclous/wiki/INDEX.md` if you need project context.

The wiki is maintained by the `/wiki` skill. To use it:
```
/wiki story "<title>"         — create a new requirement
/wiki task "<title>"          — create a task
/wiki split STORY-XXX         — decompose a story into tasks
/wiki adr "<title>"           — record an architecture decision
/wiki board                   — current work board
/wiki validate                — check wiki for conflicts
/wiki ingest <path>           — synthesize a doc into wiki pages
/wiki query "<question>"      — answer a question from the wiki
```

## Hierarchy of Truth

When content conflicts, this order governs (higher wins):
1. `wiki/0_foundation/` — immutable, highest authority (at `/Users/reza/workspace/Oraclous/wiki/0_foundation/`)
2. `wiki/1_architecture/` — structural
3. `wiki/3_decisions/` — permanent record
4. `wiki/2_product/` and `wiki/4_agents/`
5. `wiki/6_work/` — stories and tasks

## Agent Team

Agents are added incrementally as the architecture demands them. See the wiki at `/Users/reza/workspace/Oraclous/wiki/4_agents/README.md`.

**Current agents:**
- `solution-architect` — evaluates architecture, detects conflicts, authors ADRs. First agent; all others are defined after SA validates the architecture.

Invoke via `/sa evaluate` (Challenge Mode) or `/sa review <story>` (Review Mode).

## Two Repositories — One Rule

| Repo | Path | Contains |
|------|------|----------|
| **Wiki (monorepo)** | `/Users/reza/workspace/Oraclous/` | `wiki/` — the single source of truth |
| **Code (this repo)** | `/Users/reza/workspace/Oraclous/oraclous-data-studio/` | All backend code |

**Wiki files** (`wiki/6_work/tasks/*.md`, `wiki/4_agents/security-findings.md`, etc.) live in the **outer monorepo** at `/Users/reza/workspace/Oraclous/wiki/`. Edit them there. Do **not** look for a `wiki/` directory inside this repo — it does not exist.

**All git operations** (checkout, commit, branch, log) for code work happen inside `oraclous-data-studio/`. Never run `git` commands from the outer directory for code changes.

## Codebase Layout

```
knowledge-graph-builder/        ← Main backend service (Python, FastAPI, Neo4j, Celery)
auth-service/                   ← Auth service
credential-broker-service/      ← Credential broker
```

## Founding Principles (Never Override)

1. **Open Source** — all layers are open source; no black boxes
2. **No Vendor Lock-in** — all artifacts portable (HF formats, standard JSONL, Neo4j export)
3. **Data Ownership** — customer data never leaves customer infrastructure
4. **Self-Hosted** — ships as docker-compose + Helm; no egress to Oraclous-operated SaaS

## Architecture Rules (Backend)

- `graph_id` on every Cypher query — no cross-tenant queries ever
- Parameterized Cypher everywhere — never string-interpolate user input
- Fail-closed security — deny by default
- All async code: `neo4j_client.async_driver`; all Celery tasks: task-scoped sync NullPool
- One service per major functionality — no duplicate services, no `enhanced_*` files

## Agent Git Workflow

Every task execution by an AI agent must follow this convention:

- **Branch from `develop`** using the exact branch name in the task's `branch:` frontmatter field
- **Branch naming:** `agent/STORY-{NNN}/TASK-{NNN}-{short-slug}`
- **Never push to `develop` or `main` directly** — all work goes through a PR
- **Open a PR** when the task moves to `in-review`; PR title = `TASK-XXX: <title>`
- **One agent, one branch** — do not commit to another agent's branch
- **Dependency handling:** if `blocked_by` lists a task not yet merged, branch from that task's branch and rebase onto `develop` after it merges
- **One commit per concern** — each commit covers one logical change; if the message needs "and", split into two commits
- **No AI attribution** — never add `Co-Authored-By: Claude` or any AI/tool signature to commit messages

Full convention: `/Users/reza/workspace/Oraclous/wiki/5_research/agent-git-workflow.md`

## Agent Coordination Protocol

Every task has a `status:` field (frontmatter) and an `## Agent Log` section.
**You must update both whenever your state changes.**

- On start: set `status: in-progress`, append a start entry to `## Agent Log`
- On finish: set `status: in-review`, append a finish entry with files changed and decisions made
- Security and QA agents append their verdict to the task file's `## Agent Log`

**Wiki task file location:** `/Users/reza/workspace/Oraclous/wiki/6_work/tasks/TASK-XXX-*.md`

Full rules: `/Users/reza/workspace/Oraclous/wiki/6_work/AGENT_PROTOCOL.md`

## Key Blocking Gates

No story is done without:
1. **Security Architect sign-off** — any security-touching change
2. **QA Engineer sign-off** — all quality gates must pass

## Current Phase

**Deepening phase** (April–July 2026):
- Leiden hierarchical communities (replacing Louvain)
- True bitemporal tracking
- Federation SAME_AS semantic matching
- Frontend MVP (4 core screens)
- Published benchmarks

Then: FTOps agent build (Wave 1: Structural, Coverage & Gap, Community & Hierarchy analysis families).
