#!/usr/bin/env python
"""
Dump an assessment run's :Finding rows back to a JSONL file.

STORY-026 — SPRINT-001 QA helper (TASK-072).

Counterpart to :mod:`app.scripts.backfill_assessment_run`. Given a tenant
``graph_id`` + ``run_id``, walks every :Finding in the tenant graph (filtered
to that run), joins each finding to its :Source via the :CITES edge, and
emits one JSON object per line — sorted by ``finding_id`` so the resulting
file is stable across runs and diff-friendly.

Why this exists
---------------

QA needs an automated, machine-verifiable round-trip proof: take the source
``evidence/evidence.jsonl``, push it through the backfill, dump it back, and
diff. If anything is lost (a property dropped, a transformation that doesn't
invert cleanly, a finding that didn't write), the diff surfaces it.

The dump output shape mirrors the legacy ``evidence.jsonl`` shape as closely
as possible so a textual diff against the source is meaningful. Field
ordering and presence are documented at the top of every dumped row; see
:func:`_finding_record_to_dict` for the mapping. The differences QA expects
between source and dump are enumerated in
``tests/qa/test_eurail_backfill_roundtrip.py`` (the round-trip test
docstring).

Usage
-----

::

    python -m app.scripts.dump_run_to_jsonl \\
        --graph-id eurail-tenant-graph \\
        --run-id   run-<uuid> \\
        --output   /tmp/eurail-roundtrip.jsonl \\
        --neo4j-uri bolt://localhost:7687 \\
        --neo4j-user neo4j --neo4j-password test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("dump_run_to_jsonl")


async def dump_run_findings(
    *,
    graph_id: str,
    run_id: str,
    output: Path,
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    driver: Any | None = None,
) -> int:
    """Walk :Finding rows in the tenant graph and write one JSON object per line.

    Returns the number of records written.

    The query joins each finding to:

    * The producing :ModuleRun → :Module → catalog slug (for the ``module``
      column in the legacy shape).
    * The cited :Source (if any) → URL / name / type / dates (for the
      legacy ``source`` block).

    Findings are sorted by ``finding_id`` ascending — deterministic regardless
    of write order, suitable for byte-comparison via ``diff``.
    """
    own_driver = False
    if driver is None:
        from neo4j import AsyncGraphDatabase

        if not (neo4j_uri and neo4j_user and neo4j_password):
            raise ValueError(
                "dump_run_findings requires either a driver or neo4j_uri+user+password"
            )
        driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        own_driver = True

    try:
        # Pull every :Finding for this run, joined to the producing module
        # slug (via :PRODUCED <- :ModuleRun -> :Module) and to its cited
        # :Source (via :CITES — optional). The catalog source lives in a
        # different graph_id; we look it up by id only (no cross-graph
        # filter; ADR-018 §Tenancy).
        result = await driver.execute_query(
            """
            MATCH (f:Finding:__Platform__ {graph_id: $graph_id, run_id: $run_id})
            OPTIONAL MATCH (mr:ModuleRun:__Platform__)-[:PRODUCED]->(f)
            OPTIONAL MATCH (m:Module:__Platform__ {module_id: mr.module_id})
            OPTIONAL MATCH (f)-[c:CITES]->(s:Source:__Platform__)
            RETURN
                f.finding_id            AS finding_id,
                f.claim                 AS claim,
                f.raw                   AS raw,
                f.label                 AS label,
                f.confidence            AS confidence,
                f.dimensions            AS dimensions,
                f.ai_adoption_relevance AS ai_adoption_relevance,
                f.notes                 AS notes,
                f.superseded_by         AS superseded_by,
                m.slug                  AS module_slug,
                s.source_id             AS source_id,
                s.type                  AS source_type,
                s.url_normalized        AS source_url,
                s.name                  AS source_name,
                s.publication_date      AS source_publication_date,
                s.fetch_date            AS source_fetch_date,
                s.language              AS source_language,
                c.quote                 AS source_quote
            ORDER BY f.finding_id ASC
            """,
            {"graph_id": graph_id, "run_id": run_id},
        )

        written = 0
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            for rec in result.records:
                row = _finding_record_to_dict(dict(rec))
                f.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
                f.write("\n")
                written += 1
        logger.info("dumped %d finding(s) to %s", written, output)
        return written
    finally:
        if own_driver:
            await driver.close()


def _finding_record_to_dict(rec: dict[str, Any]) -> dict[str, Any]:
    """Translate one Neo4j record into a legacy-shape JSON object.

    The fields emitted mirror ``evidence.jsonl`` so the dump can be
    diff'd against the source. Fields that don't apply (e.g. no source
    block) are omitted rather than set to ``null`` to keep the dump
    compact and the diff readable.
    """
    out: dict[str, Any] = {
        "id": rec["finding_id"],
        "module": rec.get("module_slug"),
        "claim": rec.get("claim"),
        "raw": rec.get("raw"),
        "label": rec.get("label"),
        "confidence": rec.get("confidence"),
        "dimensions": rec.get("dimensions") or [],
        "ai_adoption_relevance": rec.get("ai_adoption_relevance"),
        "notes": rec.get("notes"),
    }
    if rec.get("superseded_by"):
        out["superseded_by"] = rec["superseded_by"]
    # Source block (omit entirely if no source).
    source_block: dict[str, Any] = {}
    for src_key, dump_key in (
        ("source_type", "type"),
        ("source_url", "url"),
        ("source_name", "name"),
        ("source_publication_date", "publication_date"),
        ("source_fetch_date", "fetch_date"),
        ("source_language", "language"),
    ):
        v = rec.get(src_key)
        if v is not None:
            source_block[dump_key] = v
    if rec.get("source_id"):
        source_block["source_id"] = rec["source_id"]
    if source_block:
        out["source"] = source_block
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="dump_run_to_jsonl",
        description=(
            "Dump :Finding nodes for an assessment run back to a JSONL file. "
            "Round-trip companion to backfill_assessment_run.py."
        ),
    )
    p.add_argument("--graph-id", required=True, help="Tenant graph_id.")
    p.add_argument("--run-id", required=True, help="AssessmentRun id.")
    p.add_argument(
        "--output",
        required=True,
        help="Path to write the JSONL dump.",
    )
    p.add_argument(
        "--neo4j-uri",
        default="bolt://localhost:7687",
        help="Neo4j URI (default: bolt://localhost:7687).",
    )
    p.add_argument("--neo4j-user", default="neo4j")
    p.add_argument("--neo4j-password", default="password")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    try:
        n = asyncio.run(
            dump_run_findings(
                graph_id=args.graph_id,
                run_id=args.run_id,
                output=Path(args.output),
                neo4j_uri=args.neo4j_uri,
                neo4j_user=args.neo4j_user,
                neo4j_password=args.neo4j_password,
            )
        )
    except Exception as exc:  # pragma: no cover — top-level safety net
        logger.exception("dump failed: %s", exc)
        return 2
    print(f"wrote {n} finding(s) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
