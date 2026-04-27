#!/usr/bin/env python3
"""
Migration Backfill — Incremental KG Updates (ORA-60)

Sets required provenance fields on existing Document and Chunk nodes that were
ingested before the incremental KG update feature was added.

Run once after deploying the ORA-60 changes:

    python scripts/backfill_incremental_kg.py

Environment variables required (same as the app):
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
"""

import os
import sys

from neo4j import GraphDatabase


def get_driver():
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    username = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")
    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()
    return driver


def run_backfill(driver) -> None:
    with driver.session() as session:
        # ── Step 1: Mark all Documents without contentHash as LEGACY_UNKNOWN ─────
        # LEGACY_UNKNOWN ensures the first re-ingestion will always run in full/upsert
        # mode (hash will never match a real SHA256) and then correctly set the hash.
        result = session.run(
            """
            MATCH (d:Document)
            WHERE d.contentHash IS NULL
            SET d.contentHash    = 'LEGACY_UNKNOWN',
                d.lastIngestedAt = datetime(),
                d.ingestMode     = 'legacy'
            RETURN count(d) AS updated
            """
        )
        record = result.single()
        docs_updated = int(record["updated"]) if record else 0
        print(f"[Step 1] Documents backfilled: {docs_updated}")

        # ── Step 2: Assign synthetic jobId to Chunks without one ─────────────────
        # Uses elementId() as a stable, unique suffix — avoids generating UUIDs in Cypher.
        result = session.run(
            """
            MATCH (c:Chunk)
            WHERE c.jobId IS NULL
            SET c.jobId      = 'legacy-' + elementId(c),
                c.ingestedAt = datetime()
            RETURN count(c) AS updated
            """
        )
        record = result.single()
        chunks_updated = int(record["updated"]) if record else 0
        print(f"[Step 2] Chunks backfilled:   {chunks_updated}")

        # ── Step 3: Create indexes for delta detection performance ─────────────────
        indexes = [
            ("doc_content_hash",   "Document",   "contentHash"),
            ("chunk_job_id",       "Chunk",      "jobId"),
            ("chunk_content_hash", "Chunk",      "contentHash"),
            ("entity_job_id",      "__Entity__", "lastJobId"),
        ]
        for idx_name, label, prop in indexes:
            try:
                session.run(
                    f"CREATE INDEX {idx_name} IF NOT EXISTS FOR (n:{label}) ON (n.{prop})"
                )
                print(f"[Step 3] Index '{idx_name}' created (or already exists)")
            except Exception as exc:
                print(f"[Step 3] WARNING: Could not create index '{idx_name}': {exc}")

    print("\nBackfill complete.")
    print("  → All existing Documents and Chunks have provenance fields.")
    print("  → Indexes for delta detection are in place.")
    print("  → First re-ingestion of any legacy document will proceed as 'full' mode")
    print("    (LEGACY_UNKNOWN hash never matches a real SHA256), then set the correct hash.")


def main() -> None:
    print("Incremental KG Updates — Backfill Migration")
    print("=" * 50)
    driver = get_driver()
    try:
        run_backfill(driver)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
