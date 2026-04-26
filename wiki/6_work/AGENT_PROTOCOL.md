# Agent Coordination Protocol

Every agent working on a task in this project follows this protocol.
The task file is the single coordination artifact — all state, decisions, and handoffs live there.

---

## Nested Git Repository Warning

This project contains **two separate git repositories:**

| Repo | Path | What it contains |
|------|------|-----------------|
| Outer (Oraclous monorepo) | `/Users/reza/workspace/Oraclous/` | `wiki/`, docs, this protocol |
| Inner (backend code) | `/Users/reza/workspace/Oraclous/oraclous-data-studio/` | All Python backend code |

**All backend code changes and git operations (checkout, commit, log, diff) must be run from inside `oraclous-data-studio/`.** Running `git log` from the outer repo will show only 1–2 monorepo commits and none of the backend history. This has caused false "STILL BLOCKED" security review verdicts.

When asked to review backend code on a branch named `agent/STORY-XXX/TASK-XXX-*`:
1. `cd /Users/reza/workspace/Oraclous/oraclous-data-studio`
2. `git checkout agent/STORY-XXX/TASK-XXX-*`
3. Read files from within this directory
4. When done, restore the branch you found (`git checkout <original-branch>`)

Wiki files (task files, security findings) live in the **outer** repo and are edited there.

---

## Status Lifecycle

```
open → in-progress → in-review → done
                  ↘ (if security blocks) → in-review (fix applied) → in-review → done
```

| Status | Meaning | Who sets it |
|--------|---------|-------------|
| `open` | Not started; prerequisites may be unmet | Created this way |
| `in-progress` | Agent actively implementing | Implementing agent on start |
| `in-review` | Implementation complete; awaiting Security + QA | Implementing agent on finish |
| `done` | Security cleared + QA signed off | QA Engineer on sign-off |

**Rule:** Update the `status:` field in the task file frontmatter immediately when state changes.
Do not leave a task `open` while you are implementing it.

---

## Agent Log

Every task file has an `## Agent Log` section at the bottom. This is the running record of who
did what and why. Append to it — never delete entries.

### Format

```markdown
## Agent Log

### YYYY-MM-DD — [agent-slug] — [status transition or action]

[What was done, decided, or found. Be specific. Link to files changed or findings doc.]
```

### Required entries

| Trigger | Who writes it | What to record |
|---------|---------------|----------------|
| Starting work | Implementing agent | "Starting implementation. Branch: `branch-name`." |
| Finishing implementation | Implementing agent | Summary of changes made, files touched, any decisions taken during impl |
| Security review complete | Security Architect | Verdict (PASS / BLOCKED), findings reference, what is required before merge |
| Fix applied | Implementing agent | Which findings were fixed and how |
| QA sign-off | QA Engineer | Test count, pass count, regressions, APPROVED or REJECTED |

---

## Security Review Trigger

Security Architect review is **required** for any task that:
- Modifies a Cypher write path (`MERGE`, `CREATE`, `SET`)
- Adds or changes an authentication or authorization check
- Handles file uploads, URL inputs, or external data ingestion
- Adds a new dependency to `requirements.txt`
- Changes credential handling, token generation, or key management
- Modifies the Scope Enforcer, ReBAC service, or service account service
- Adds a new API endpoint

When your task moves to `in-review`, the orchestrator will trigger Security Architect review
if any of those conditions apply. You do not need to request it yourself.

Security findings are written to `wiki/4_agents/security-findings.md` (append-only).
A summary entry is also added to the task file's `## Agent Log`.

### What happens after a security finding

| Severity | Effect |
|----------|--------|
| `critical` or `high` | Task stays `in-review`. Fix is required before QA is triggered. |
| `medium` | Fix required or Reza explicitly accepts risk. Task stays `in-review` until resolved. |
| `low` or `informational` | Noted in log. Does not block. |

---

## QA Trigger

QA Engineer review is triggered when:
1. All implementation tasks for a story are `in-review`
2. Security Architect has cleared all blocking findings for those tasks

QA writes test files and a sign-off entry to the task file. The task moves to `done` only
after a QA sign-off entry appears in the `## Agent Log`.

---

## Handoff Rules

- **Do not assume the next agent knows your context.** Write log entries as if they will be read
  six months later by someone who was not in the session.
- **Record decisions, not just actions.** If you chose approach A over approach B, write why.
- **Record blockers.** If you hit something unexpected, write it in the log before stopping.
- **Never edit another agent's log entry.** Only append.

---

## Finding another agent's task

If you need context on a task you did not implement:
1. Read the task file — the `## Agent Log` has the history
2. Check `wiki/4_agents/security-findings.md` for any security review of that task
3. Check the branch and PR if you need to see the code

---

## Example: complete lifecycle of a task

```
## Agent Log

### 2026-04-26 — backend-developer — open → in-progress

Starting implementation. Branch: `agent/STORY-002/TASK-005-bitemporal-models`.
Working in `oraclous-data-studio/knowledge-graph-builder/`.

### 2026-04-26 — backend-developer — in-progress → in-review

Implementation complete. Three commits:
- graph_schemas.py: added four bitemporal fields to entity and relationship models
- multi_tenant_components.py: writer now sets ingestion_time and ingestion_source
- background_jobs.py: added run_bitemporal_migration_v1 task with idempotency guard

Decision: used Celery beat one-off (not startup migration) to avoid blocking startup on large graphs.

### 2026-04-26 — security-architect — security review

BLOCKED. See wiki/4_agents/security-findings.md — TASK-005.
Blocking: migration has no graph_id scope (cross-tenant write). Fix before QA.
Medium (fix required): ingestion_source sanitization, Pydantic field bounds.

### 2026-04-26 — backend-developer — fix applied

Fixed all blocking and medium findings:
- background_jobs.py: fan-out migration per graph_id; atomic guard via MERGE ON CREATE
- multi_tenant_components.py: pop/sanitize ingestion_source; _sanitize_source() helper
- graph_schemas.py: max_length=512 on ingestion_source; field_validator on ingestion_time

### 2026-04-26 — security-architect — re-review

PASS. All findings resolved. QA can proceed.

### 2026-04-26 — qa-engineer — QA sign-off

Tests written: 9. Tests passing: 9. Regressions: none.
Status: APPROVED. Task moves to done.
```
