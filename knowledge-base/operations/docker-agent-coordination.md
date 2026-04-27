# Docker Multi-Agent Isolation Protocol

**Status:** Active — Approved via ORA-267 (2026-04-11)
**Scope:** All backend agents, all CI/CD workflows, all PR reviews
**Owner:** DevOps SRE Specialist (ORA-265)

---

## Problem

Multiple agents running Docker simultaneously on the same host share the default Compose project namespace (`oraclous`). Without isolation, agents can:

- Overwrite each other's containers, networks, and volumes
- Produce false-positive or false-negative test results by sharing runtime state
- Contaminate each other's images via the `:latest` tag (overwritten mid-run)

---

## Required Protocol

### Layer 1 — Compose Project Isolation

All `docker compose` calls **MUST** include a unique project name tied to the issue identifier. Never run bare `docker compose` without `COMPOSE_PROJECT_NAME`.

```bash
export COMPOSE_PROJECT_NAME=oraclous-ORA-XXX   # e.g. oraclous-ORA-265
docker compose up --build -d
```

Docker Compose uses this value to prefix all container names, network names, and volume names. This is the primary isolation boundary — stacks from different agents cannot touch each other's resources.

### Layer 2 — Image Tag Isolation

All local image builds **MUST** use a deterministic tag derived from the current branch and commit SHA. Never build or push `:latest`.

```bash
export IMAGE_TAG="$(git rev-parse --abbrev-ref HEAD | tr '/' '-')-$(git rev-parse --short HEAD)"
# e.g. feature-phase1-devops-ora-265-docker-agent-isolation-a3f8c12
docker compose build
```

The `docker-compose.yml` uses `${IMAGE_TAG:-dev}` on every locally-built service `image:` field. Without `IMAGE_TAG` set, it falls back to `dev` (safe for solo local development, not for multi-agent environments).

**Reza's key principle:** Each image must point only to changes on the same branch — no collisions across branches running locally.

### Layer 3 — Dynamic Port Allocation

When running multiple stacks concurrently on the same host, use the `docker-compose.agent.yml` override to replace all fixed host ports with `"0:PORT"` (OS-assigned ephemeral port):

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.agent.yml \
  up --build -d
```

Find the assigned port for a service after startup:

```bash
docker compose port knowledge-graph-builder 8000
# => 0.0.0.0:52341
```

---

## Full Agent Startup Sequence

Before running **any** `docker compose` command on this repo, set both required variables:

```bash
# 1. Set project isolation (mandatory)
export COMPOSE_PROJECT_NAME=oraclous-ORA-XXX

# 2. Set image tag (mandatory in multi-agent environments)
export IMAGE_TAG="$(git rev-parse --abbrev-ref HEAD | tr '/' '-')-$(git rev-parse --short HEAD)"

# 3. Start stack (add -f docker-compose.agent.yml if port conflicts are possible)
docker compose -f docker-compose.yml -f docker-compose.agent.yml up --build -d

# 4. Verify health
docker compose ps
```

---

## Example `.env.agent` Template

Create this file at the repo root before running Compose. Do not commit it.

```dotenv
# .env.agent — copy this, fill in your issue ID, do NOT commit
COMPOSE_PROJECT_NAME=oraclous-ORA-XXX
IMAGE_TAG=feature-branch-slug-$(git rev-parse --short HEAD)
```

Then start with:

```bash
set -a && source .env.agent && set +a
docker compose -f docker-compose.yml -f docker-compose.agent.yml up --build -d
```

---

## Automated Cleanup (Script-Enforced)

When branch work is complete, run the cleanup script to tear down the isolated stack and remove tagged images. Cleanup is **mandatory** before abandoning or merging a branch.

```bash
./scripts/agent-docker-cleanup.sh ORA-XXX
```

What the script does:

1. Derives `COMPOSE_PROJECT_NAME=oraclous-ORA-XXX` from the issue identifier
2. Runs `docker compose -p oraclous-ORA-XXX down --volumes --remove-orphans`
3. Removes locally-built images tagged with this issue's branch slug

Dry-run mode (preview without executing):

```bash
./scripts/agent-docker-cleanup.sh ORA-XXX --dry-run
```

**Do not skip cleanup.** Orphaned stacks accumulate disk usage, dangling networks, and stale volumes that interfere with future runs.

---

## PR Review Checklist (Mandatory)

The Backend Lead Developer **MUST reject** any PR where:

- [ ] An agent ran `docker compose` without `COMPOSE_PROJECT_NAME=oraclous-{ORA-XXX}`
- [ ] An agent built or referenced a `:latest` image tag
- [ ] Task descriptions or CI logs show bare `docker compose up` calls
- [ ] The cleanup script was not run before branch merge

Add this question to every PR review:

> **Docker isolation protocol followed?** (`COMPOSE_PROJECT_NAME` + `IMAGE_TAG` set? Cleanup script run?)

---

## Rationale

Approved by Reza (ORA-267). Without isolation, concurrent agent workstreams interfere with each other's Docker environments, leading to non-reproducible failures and contaminated test results.

Explicit `container_name:` fields were removed from `docker-compose.yml` to allow `COMPOSE_PROJECT_NAME` to take effect. Hardcoded container names bypass project namespacing and cause `docker: Error response from daemon: Conflict` errors when two stacks are live.

---

*Created by DevOps SRE Specialist — [ORA-265](/ORA/issues/ORA-265). Protocol designed in [ORA-264](/ORA/issues/ORA-264).*
