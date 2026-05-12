"""
Assessment substrate schema runner (TASK-067, STORY-026).

Reads and executes the declarative Cypher migration that lands the assessment
substrate schema: constraints, indexes, and the catalog graph anchors. Designed
to be invoked from the FastAPI lifespan (matching the existing
`ensure_code_schema`, `ensure_memory_indexes`, etc. pattern) AND as a
standalone CLI for local-dev / CI smoke runs.

All statements are idempotent (`CREATE CONSTRAINT IF NOT EXISTS`,
`CREATE INDEX IF NOT EXISTS`, `MERGE`). Re-running is safe.

Usage from FastAPI startup:

    from app.db.assessment_schema_init import ensure_assessment_schema

    await ensure_assessment_schema(neo4j_client.async_driver)

Usage as a script:

    python -m app.db.assessment_schema_init
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

# Path to the canonical `.cypher` migration. The Python entrypoint is a thin
# runner — the migration file is the source of truth.
_MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent
    / "cypher"
    / "migrations"
    / "2026-05-11_assessment_schema.cypher"
)


def _split_statements(cypher: str) -> list[str]:
    """
    Split a multi-statement Cypher file on bare `;` terminators.

    Rules:
      - Drop line comments starting with `//`.
      - Drop blank lines.
      - A statement ends at a line whose trimmed text ends with `;`.
      - The terminating `;` is removed from each statement (Neo4j drivers
        reject a trailing semicolon when running a single statement).

    Block comments (`/* … */`) are not used in the migration file, so we
    intentionally do not parse them.
    """
    statements: list[str] = []
    current: list[str] = []
    for raw_line in cypher.splitlines():
        # Strip line comments. Naive — the migration file does not contain
        # string literals with `//` inside, so this is safe.
        stripped = raw_line.split("//", 1)[0].rstrip()
        if not stripped.strip():
            if current:
                # Preserve blank lines inside a statement so error messages keep
                # readable line numbers.
                current.append("")
            continue
        current.append(stripped)
        if stripped.endswith(";"):
            # Reassemble the statement and trim the terminating ';'
            stmt = "\n".join(current).rstrip()
            assert stmt.endswith(";")
            stmt = stmt[:-1].strip()
            if stmt:
                statements.append(stmt)
            current = []
    if current and "\n".join(current).strip():
        # Trailing statement without a terminating semicolon — treat as one
        # final statement.
        statements.append("\n".join(current).strip())
    return statements


def load_migration_statements(
    migration_path: Path | None = None,
) -> list[str]:
    """Load the assessment-schema migration file and return executable statements."""
    path = migration_path or _MIGRATION_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Assessment-schema migration file not found at {path}. "
            "Expected the file shipped with TASK-067."
        )
    return _split_statements(path.read_text(encoding="utf-8"))


async def ensure_assessment_schema(async_driver: Any) -> None:
    """
    Apply the assessment-substrate Cypher migration. Idempotent.

    All statements use `IF NOT EXISTS` or `MERGE`, so this is safe to call
    on every application startup (matches the pattern used by
    `ensure_code_schema`, `ensure_memory_indexes`, and
    `rebac_service.initialize_schema_full`).

    Args:
        async_driver: Active `neo4j.AsyncDriver`. The caller is responsible
            for connection lifecycle.
    """
    from app.core.config import settings

    statements = load_migration_statements()
    if not statements:
        logger.warning("Assessment-schema migration contained zero statements")
        return

    applied = 0
    skipped: list[tuple[str, str]] = []
    for stmt in statements:
        try:
            await async_driver.execute_query(stmt, database_=settings.NEO4J_DATABASE)
            applied += 1
        except Exception as exc:  # pragma: no cover — defensive only
            # Constraints and indexes use `IF NOT EXISTS`, so collisions are
            # not expected. Log and continue so a single bad statement does
            # not block the rest of the migration during startup.
            preview = stmt.splitlines()[0][:80]
            logger.warning(
                "Assessment-schema statement warning (statement starting %r): %s",
                preview,
                exc,
            )
            skipped.append((preview, str(exc)))

    logger.info(
        "Assessment-schema migration applied %d statement(s) (skipped=%d)",
        applied,
        len(skipped),
    )


async def _main() -> None:
    """Standalone CLI entrypoint — opens its own Neo4j connection."""
    from app.core.neo4j_client import neo4j_client

    logger.info("Connecting to Neo4j for standalone assessment-schema migration")
    await neo4j_client.connect()
    try:
        await ensure_assessment_schema(neo4j_client.async_driver)
        logger.info("Assessment-schema migration complete")
    finally:
        await neo4j_client.disconnect()


if __name__ == "__main__":
    asyncio.run(_main())
