# PR Review Guideline — Docker Isolation Protocol

**Status:** Active — Required for all backend PRs
**Approved via:** [ORA-267](/ORA/issues/ORA-267)
**Runbook:** `knowledge-base/operations/docker-agent-coordination.md`
**Enforced by:** Backend Lead Developer (ORA-273)
**Effective:** 2026-04-11

---

## Purpose

Multiple agents running Docker simultaneously on the same host can collide when using the default Compose project namespace or the `:latest` image tag. This guideline ensures every PR review checks for compliance with the Docker multi-agent isolation protocol.

---

## Mandatory PR Checklist Item

Add the following check to **every** backend PR review, regardless of whether the PR appears to touch Docker directly (CI scripts, Dockerfiles, compose files, and task descriptions must all be checked):

```
**Docker isolation protocol followed?**
- [ ] `docker compose` calls use `COMPOSE_PROJECT_NAME=oraclous-{ORA-XXX}` (not bare `docker compose`)
- [ ] No `:latest` image tags — must use `IMAGE_TAG={branch-slug}-{git-sha}`
```

---

## Hard-Block Rejection Criteria

**REJECT the PR immediately** (do not approve, do not request minor changes — block it) if:

| Violation | Description |
|---|---|
| Missing `COMPOSE_PROJECT_NAME` | Any `docker compose up`, `docker compose down`, `docker compose build`, or equivalent call that does not prefix `COMPOSE_PROJECT_NAME=oraclous-{ORA-XXX}` |
| `:latest` tag used | Any Dockerfile `FROM image:latest`, compose file `image: service:latest`, or `docker build` that does not set `IMAGE_TAG={branch-slug}-{git-sha}` |

### Where to Look

- `Dockerfile` and `docker-compose.yml` / `docker-compose.override.yml`
- CI workflow files (`.github/workflows/`)
- Task PR descriptions referencing Docker commands
- CI run logs linked from the PR

---

## Correct Usage Examples

```bash
# Correct — project-scoped compose
COMPOSE_PROJECT_NAME=oraclous-273 docker compose up --build -d
COMPOSE_PROJECT_NAME=oraclous-273 docker compose down -v --remove-orphans

# Correct — deterministic image tag
IMAGE_TAG=feature-phase3-sr-backend-ora-273-5c366ef docker compose build
```

---

## Review Table Row

Include this row in the Architecture Compliance table of every review:

```markdown
| Docker isolation protocol | PASS/FAIL | COMPOSE_PROJECT_NAME + IMAGE_TAG used correctly? |
```

---

## References

- Runbook: `knowledge-base/operations/docker-agent-coordination.md`
- Approval issue: [ORA-267](/ORA/issues/ORA-267)
- This guideline: [ORA-273](/ORA/issues/ORA-273)
