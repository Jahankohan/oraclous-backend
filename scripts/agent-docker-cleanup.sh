#!/usr/bin/env bash
# agent-docker-cleanup.sh
#
# Tears down the Compose stack for a given Oraclous issue/agent workstream.
# Run this after your branch work is complete or the Docker environment is no
# longer needed.
#
# Usage:
#   ./scripts/agent-docker-cleanup.sh ORA-265
#   ./scripts/agent-docker-cleanup.sh ORA-265 --dry-run
#
# The script uses the issue identifier to derive COMPOSE_PROJECT_NAME so that
# only containers/networks/volumes belonging to that stack are removed —
# other concurrently-running agent stacks are not affected.

set -euo pipefail

ISSUE_ID="${1:-}"
DRY_RUN="${2:-}"

if [[ -z "$ISSUE_ID" ]]; then
  echo "Usage: $0 <issue-identifier> [--dry-run]" >&2
  echo "  e.g. $0 ORA-265" >&2
  exit 1
fi

# Normalise to lowercase for Docker project name compatibility
PROJECT_NAME="oraclous-$(echo "$ISSUE_ID" | tr '[:upper:]' '[:lower:]')"

echo "=> Compose project: $PROJECT_NAME"

if [[ "$DRY_RUN" == "--dry-run" ]]; then
  echo "[dry-run] Would run: docker compose -p $PROJECT_NAME down --volumes --remove-orphans"
  exit 0
fi

# Change to repo root (script may be called from any working directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=> Tearing down $PROJECT_NAME ..."
docker compose -p "$PROJECT_NAME" down --volumes --remove-orphans

echo "=> Removing tagged images for this stack (IMAGE_TAG prefix: ${ISSUE_ID,,}) ..."
# Remove any locally-built images that carry this issue's tag prefix to free disk space.
docker images --format '{{.Repository}}:{{.Tag}}' \
  | grep ":.*$(echo "$ISSUE_ID" | tr '[:upper:]' '[:lower:]')" \
  | xargs -r docker rmi --force || true

echo "=> Cleanup complete for $PROJECT_NAME"
