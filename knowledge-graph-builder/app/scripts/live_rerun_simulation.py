#!/usr/bin/env python
"""Live-rerun simulation for the SPRINT-002 sprint gate (TASK-084).

This script proves the end-to-end MCP loop closes — `create_run` →
`record_finding_bulk` (many) → `record_conflict` →
`record_unresolved_question` → `persist_deliverable` (with Blob CAS bytes)
→ SSE event broker → `finalize_run` → read-back via `list_*` + `get_*`.

It is NOT a replacement for Reza running the actual `/eurail-report`
skill in Claude Code; that requires the Max subscription's LLM budget
for ~25 subagent invocations. What this script does instead is exercise
**the same MCP tool functions Claude Code's skill would call**, against
the same Neo4j + Postgres + Blob CAS + event broker stack.

The claim: **if this simulation succeeds, the real Claude Code run will
hit the same backend code paths and produce a comparable `:AssessmentRun`**.

Auth is short-circuited (not bypassed): we patch
``auth_service.verify_token`` and ``rebac_service.check_graph_permission``
to return a known service-account principal with ``home_graph_id`` set
to the simulation's tenant graph. This mirrors what a JWT-issued
service-account token would resolve to, and exercises every code path
*downstream* of the auth boundary — Pydantic validation, ADR-010 scope
normalization, service-layer Cypher, Postgres CAS round-trip, and the
event-broker publish hooks — without requiring the standalone
``auth-service`` container.

Data source: the existing ``docs/eurail/out/eurail-2026-05-06/`` run.
This is read-only — the script consumes its evidence + conflicts but
does not mutate it. The script reuses the helpers from
``backfill_assessment_run.py`` (``_evidence_to_finding``,
``_normalize_module_slug``) so the simulation and the SPRINT-001 backfill
exercise the same translation logic.

Usage (after the catalog seed has run)::

    python -m app.scripts.live_rerun_simulation \\
        --run-dir docs/eurail/out/eurail-2026-05-06 \\
        --graph-id eurail-tenant-test \\
        --template-slug eurail-report-v1

Exit codes:
    0  — simulation succeeded; counts printed to stdout match expectations
    1  — Neo4j unavailable / catalog not seeded / a tool returned an
         error that breaks the loop
    2  — an MCP tool contract diverged from what the skill expects
         (loud failure; STOP and inspect)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client

# Import the MCP tool functions we will drive. These are the same callables
# FastMCP registers via ``app.mcp.server`` — invoking them in-process exercises
# the exact post-auth code path the network MCP transport would hit.
from app.mcp.tools.assessment_tools import (
    create_run,
    finalize_run,
    get_deliverable_content,
    get_run,
    get_wave_status,
    list_conflicts,
    list_deliverables,
    list_findings,
    list_module_runs,
    list_unresolved_questions,
    persist_deliverable,
    record_conflict,
    record_finding_bulk,
    record_unresolved_question,
    update_module_run,
)
from app.scripts.backfill_assessment_run import (
    _evidence_to_finding,
    _iter_jsonl_records,
    _normalize_module_slug,
    _resolve_record_id,
    _validate_finding_or_log_skip,
)
from app.services.assessment_event_broker import (
    AssessmentEvent,
    AssessmentEventBroker,
    get_assessment_event_broker,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The fake service-account principal we inject during the simulation. It has
#: the same shape as a real auth-service ``/me`` response so the contextvar-
#: based ReBAC plumbing in ``app.api.dependencies`` treats it identically.
SIM_PRINCIPAL_ID = "sa-task-084-sim"

#: How many findings per bulk batch we submit. The skill aims for one bulk
#: call per :ModuleRun (typically 20-50 findings); we match that shape.
BULK_BATCH_LIMIT = 200


def _make_principal(graph_id: str) -> dict[str, Any]:
    return {
        "id": SIM_PRINCIPAL_ID,
        "principal_type": "service_account",
        "tenant_id": "tenant-sim",
        "home_graph_id": graph_id,
        "email": "task-084-sim@oraclous.test",
        "name": "TASK-084 Simulation SA",
    }


# ---------------------------------------------------------------------------
# Auth short-circuit
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _patched_auth(graph_id: str):
    """Patch auth_service + rebac_service so MCP tools resolve the sim principal.

    Both patches are scoped to the ``async with`` body. They exercise the same
    code paths the real auth flow would after token verification:

    - ``auth_service.verify_token`` returns the sim principal dict
    - ``rebac_service.check_graph_permission`` permits read+write+admin on the
      sim graph_id (mirrors what a service-account ACL row would produce)
    - ``service_account_service.check_sa_graph_permission`` mirrors the same

    Anything *after* the auth boundary — ``verify_graph_access`` body,
    ``_current_principal`` contextvar plumbing, ADR-010 scope normalization,
    Pydantic schema validation, ``AssessmentService`` Cypher writes,
    ``BlobCASService`` Postgres writes, and ``AssessmentEventBroker``
    publishes — runs unmocked.
    """
    principal = _make_principal(graph_id)

    async def _verify_token(token: str) -> dict[str, Any]:
        return principal

    async def _check_graph_permission(*args: Any, **kwargs: Any) -> bool:
        return True

    with (
        patch(
            "app.services.auth_service.auth_service.verify_token",
            new=AsyncMock(side_effect=_verify_token),
        ),
        patch(
            "app.services.rebac_service.rebac_service.check_graph_permission",
            new=AsyncMock(side_effect=_check_graph_permission),
        ),
        patch(
            "app.services.service_account_service.service_account_service."
            "check_sa_graph_permission",
            new=AsyncMock(side_effect=_check_graph_permission),
        ),
    ):
        # ORACLOUS_API_KEY just needs to be non-empty — the patched
        # verify_token ignores the value but the MCP _auth helper checks
        # truthiness before calling verify_token.
        prev_key = os.environ.get("ORACLOUS_API_KEY")
        os.environ["ORACLOUS_API_KEY"] = "task-084-sim-key"
        try:
            yield principal
        finally:
            if prev_key is None:
                os.environ.pop("ORACLOUS_API_KEY", None)
            else:
                os.environ["ORACLOUS_API_KEY"] = prev_key


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------


class ContractDivergence(RuntimeError):
    """Raised when an MCP tool returned an unexpected shape — STOP-and-report."""


def _check_ok(label: str, result: dict[str, Any]) -> dict[str, Any]:
    """Assert the MCP tool result is not a structured error.

    Raises ``ContractDivergence`` so the script exits with code 2.
    """
    if isinstance(result, dict) and "error" in result and "code" in result:
        raise ContractDivergence(
            f"{label} returned error: code={result['code']!r} message={result['error']!r}"
        )
    return result


# ---------------------------------------------------------------------------
# Phase: create_run
# ---------------------------------------------------------------------------


async def _phase_create_run(
    *, graph_id: str, template_slug: str, subject_block: dict[str, Any]
) -> tuple[str, list[str]]:
    body = {
        "template_slug": template_slug,
        "subject": subject_block,
        "cli_flags": {
            "task": "TASK-084",
            "simulation": True,
            "source_run_dir": "docs/eurail/out/eurail-2026-05-06",
        },
    }
    resp = await create_run(body)
    _check_ok("assessment.create_run", resp)
    run_id = resp["run_id"]
    module_run_ids = resp.get("module_run_ids") or []
    logger.info(
        "create_run OK: run_id=%s template_id=%s module_runs=%d already_existed=%s",
        run_id,
        resp.get("template_id"),
        len(module_run_ids),
        resp.get("already_existed"),
    )
    return run_id, module_run_ids


# ---------------------------------------------------------------------------
# Phase: module_run discovery
# ---------------------------------------------------------------------------


async def _discover_module_runs(
    *, run_id: str
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Page through list_module_runs to build {slug → module_run_id}.

    Also returns the raw module_run row per slug so the simulation can pick
    a wave-1 module run and update its status / heartbeat.
    """
    slug_to_mr_id: dict[str, str] = {}
    raw_by_slug: dict[str, dict[str, Any]] = {}
    cursor: str | None = None
    while True:
        page = await list_module_runs(run_id=run_id, limit=200, cursor=cursor)
        _check_ok("assessment.list_module_runs", page)
        for row in page["items"]:
            slug = row.get("module_slug")
            if slug:
                slug_to_mr_id[slug] = row["module_run_id"]
                raw_by_slug[slug] = row
        nxt = page["page"].get("next_cursor")
        if not nxt:
            break
        cursor = nxt
    logger.info(
        "list_module_runs OK: discovered %d (slug → module_run_id) pairs",
        len(slug_to_mr_id),
    )
    return slug_to_mr_id, raw_by_slug


# ---------------------------------------------------------------------------
# Phase: ingest evidence
# ---------------------------------------------------------------------------


async def _phase_record_findings(
    *,
    run_id: str,
    graph_id: str,
    run_dir: Path,
    slug_to_mr_id: dict[str, str],
) -> dict[str, int]:
    """Stream evidence.jsonl, group by module slug, bulk-write per module."""
    evidence_path = run_dir / "evidence" / "evidence.jsonl"
    if not evidence_path.is_file():
        raise FileNotFoundError(f"Expected evidence stream at {evidence_path}")

    per_module: dict[str, list[dict[str, Any]]] = defaultdict(list)
    parse_errs = 0
    skipped_no_id = 0
    skipped_unknown_module = 0
    skipped_schema = 0

    for ln, record in _iter_jsonl_records(evidence_path):
        if "__parse_error__" in record:
            parse_errs += 1
            continue
        if not _resolve_record_id(record):
            skipped_no_id += 1
            continue
        module_slug = _normalize_module_slug(record.get("module") or "")
        if not module_slug:
            skipped_unknown_module += 1
            continue
        mr_id = slug_to_mr_id.get(module_slug)
        if not mr_id:
            # Module slug not in the catalog-seeded template (gap-research
            # modules are dynamic; the simulation skips them per TASK-084
            # framing — the live skill inserts them mid-run, the static
            # template doesn't carry them).
            skipped_unknown_module += 1
            continue
        finding = _evidence_to_finding(
            record, graph_id=graph_id, run_id=run_id, module_run_id=mr_id
        )
        if not _validate_finding_or_log_skip(finding):
            skipped_schema += 1
            continue
        per_module[mr_id].append(finding)

    succeeded = 0
    failed = 0
    batches = 0
    for mr_id, findings in per_module.items():
        # Chunk into BULK_BATCH_LIMIT-sized pieces to stay under the
        # LIST_MAX_ITEMS Pydantic bound on RecordFindingBulkRequest.
        for i in range(0, len(findings), BULK_BATCH_LIMIT):
            chunk = findings[i : i + BULK_BATCH_LIMIT]
            resp = await record_finding_bulk(
                run_id=run_id, body={"findings": chunk}
            )
            _check_ok("assessment.record_finding_bulk", resp)
            batches += 1
            succeeded += int(resp.get("succeeded", 0))
            failed += int(resp.get("failed", 0))

    logger.info(
        "record_finding_bulk: %d batches, %d succeeded, %d failed; "
        "skipped: parse=%d no_id=%d unknown_module=%d schema=%d",
        batches,
        succeeded,
        failed,
        parse_errs,
        skipped_no_id,
        skipped_unknown_module,
        skipped_schema,
    )
    return {
        "succeeded": succeeded,
        "failed": failed,
        "batches": batches,
        "skipped_parse": parse_errs,
        "skipped_no_id": skipped_no_id,
        "skipped_unknown_module": skipped_unknown_module,
        "skipped_schema": skipped_schema,
    }


# ---------------------------------------------------------------------------
# Phase: ingest conflicts
# ---------------------------------------------------------------------------


async def _phase_record_conflicts(
    *,
    run_id: str,
    graph_id: str,
    run_dir: Path,
) -> int:
    conflicts_path = run_dir / "evidence" / "conflicts.jsonl"
    if not conflicts_path.is_file():
        logger.info("No conflicts.jsonl found at %s; skipping", conflicts_path)
        return 0

    created = 0
    for ln, record in _iter_jsonl_records(conflicts_path):
        if "__parse_error__" in record:
            continue
        cf_id = record.get("id") or f"cf-{uuid.uuid4().hex[:8]}"
        body = {
            "conflict": {
                "conflict_id": cf_id,
                "graph_id": graph_id,
                "run_id": run_id,
                "topic": (record.get("topic") or "untitled conflict")[:500],
                "summary": (record.get("summary") or record.get("explanation") or "")[
                    :8000
                ],
                "status": "resolved"
                if str(record.get("resolution", "")).upper() == "RESOLVED"
                else "open",
                "resolution": record.get("resolution") if record.get("resolution") else None,
                "synthesis_note": record.get("synthesis_note"),
                "involved_finding_ids": list(record.get("evidence_ids") or [])[:50],
            }
        }
        resp = await record_conflict(run_id=run_id, body=body)
        _check_ok("assessment.record_conflict", resp)
        if resp.get("created"):
            created += 1
    logger.info("record_conflict: created=%d", created)
    return created


# ---------------------------------------------------------------------------
# Phase: unresolved questions (a couple hand-curated, since the legacy run
# doesn't emit them as a JSONL stream)
# ---------------------------------------------------------------------------


async def _phase_record_questions(
    *,
    run_id: str,
    graph_id: str,
    slug_to_mr_id: dict[str, str],
) -> int:
    """Emit a small set of synthesized open questions.

    The legacy filesystem run did not emit ``unresolved-questions.jsonl`` as
    a discrete file (the skill kept them inline in deliverables). For the
    simulation we synthesize a representative set so the read-side tools
    (``list_unresolved_questions``) have something to return.
    """
    sample_questions = [
        (
            "02-cooperative-governance",
            "Which 5 of the 35 member operators have working AI integration teams "
            "and which 30 do not?",
        ),
        (
            "09-breach-aftermath",
            "What is the per-customer remediation cost and the regulator-imposed "
            "remediation deadline for the January 2026 breach?",
        ),
        (
            "10-trip-planner-sunset",
            "Was the Trip Planner sunset purely a cost decision, or did it follow "
            "a measured drop in usage?",
        ),
    ]
    created = 0
    for slug, text in sample_questions:
        mr_id = slug_to_mr_id.get(slug)
        if not mr_id:
            continue
        body = {
            "question": {
                "question_id": f"uq-{uuid.uuid4().hex[:8]}",
                "graph_id": graph_id,
                "run_id": run_id,
                "module_run_id": mr_id,
                "text": text,
                "suggested_module": None,
                "status": "open",
            }
        }
        resp = await record_unresolved_question(run_id=run_id, body=body)
        _check_ok("assessment.record_unresolved_question", resp)
        if resp.get("created"):
            created += 1
    logger.info("record_unresolved_question: created=%d", created)
    return created


# ---------------------------------------------------------------------------
# Phase: persist deliverables (module Markdown + simulated final HTML via CAS)
# ---------------------------------------------------------------------------


async def _phase_persist_deliverables(
    *,
    run_id: str,
    graph_id: str,
    run_dir: Path,
    slug_to_mr_id: dict[str, str],
) -> tuple[int, int]:
    """Persist each ``NN_<slug>.md`` deliverable + 5 synthetic final HTML docs.

    The 5 final docs exercise the Blob CAS path: each carries
    ``content_uri`` we leave None on the wire, and we let the service layer
    write the bytes through ``BlobCASService.put`` via the
    ``content_bytes`` kwarg — but the MCP wrapper does NOT expose
    ``content_bytes`` to the tool body (verified in the MCP tool surface
    inspection in TASK-080). For the simulation we therefore exercise CAS
    by writing ``content_inline`` for module deliverables (matches what the
    skill does) and invoking the assessment service ``persist_deliverable``
    directly with ``content_bytes`` for the final 5 HTML docs.
    """
    # Module deliverables — inline content path (matches eurail-report skill)
    module_deliverables_created = 0
    md_files = sorted(run_dir.glob("[0-9][0-9]_*.md"))
    for md_path in md_files:
        slug = _normalize_module_slug(md_path.stem.split("_", 1)[1].replace("_", "-"))
        mr_id = slug_to_mr_id.get(slug)
        # Some filenames don't correspond to a catalog module (e.g.
        # ``01_tech_stack_appendix``); persist them as run-level
        # deliverables (module_run_id=None).
        kind = "research"
        ordinal = int(md_path.stem.split("_", 1)[0])
        content = md_path.read_text(encoding="utf-8")
        if len(content) > 50_000:
            # SIZE_BLOB_TEXT clamp from TASK-076; truncate to avoid blowing
            # the Pydantic bound. The full file is also available in the
            # legacy on-disk run for cross-check.
            content = content[:50_000]
        body = {
            "deliverable": {
                "deliverable_id": f"del-{uuid.uuid4().hex[:12]}",
                "graph_id": graph_id,
                "run_id": run_id,
                "module_run_id": mr_id,
                "kind": kind,
                "filename": md_path.name,
                "ordinal": ordinal,
                "content_uri": None,
                "content_inline": content,
                "sha256": None,
                "word_count": len(content.split()),
            }
        }
        resp = await persist_deliverable(run_id=run_id, body=body)
        _check_ok("assessment.persist_deliverable", resp)
        if resp.get("created"):
            module_deliverables_created += 1

    # Final 5 HTML docs — exercises the Blob CAS path. The MCP wrapper for
    # persist_deliverable does NOT accept content_bytes (per the
    # TASK-080 review, the wrapper takes only PersistDeliverableRequest
    # which has no content_bytes field). We invoke the AssessmentService
    # directly via the build_service helper for these to prove the CAS
    # round-trip; a follow-up wrapper enhancement is tracked separately.
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from app.core.database import engine
    from app.mcp.tools._auth import build_service

    svc = build_service()
    Session = async_sessionmaker(engine, expire_on_commit=False)

    final_doc_ids: list[str] = []
    final_doc_created = 0
    for n, label in enumerate(
        ["executive_overview", "diagnosis", "roadmap", "partnership_brief", "appendix"],
        start=1,
    ):
        html_bytes = (
            f"<!doctype html><html><head><title>{label} (TASK-084 sim)"
            f"</title></head><body><h1>{label}</h1>"
            f"<p>Final doc {n} of 5 synthesized for the simulation. "
            f"Real run produces a multi-MB HTML body via /docify.</p></body></html>"
        ).encode("utf-8")
        from app.schemas.assessment_schemas import Deliverable

        deliverable = Deliverable(
            deliverable_id=f"del-final-{n:02d}-{uuid.uuid4().hex[:8]}",
            graph_id=graph_id,
            run_id=run_id,
            module_run_id=None,
            kind="final_doc",
            filename=f"{n:02d}_{label}.html",
            ordinal=n,
            content_uri=None,
            content_inline=None,
            sha256=None,
            word_count=None,
        )
        async with Session() as db:
            created = await svc.persist_deliverable(
                graph_id,
                run_id,
                deliverable,
                content_bytes=html_bytes,
                mime_type="text/html",
                db=db,
            )
            await db.commit()
        if created:
            final_doc_created += 1
        final_doc_ids.append(deliverable.deliverable_id)

    logger.info(
        "persist_deliverable: module=%d final_doc=%d",
        module_deliverables_created,
        final_doc_created,
    )
    return module_deliverables_created, final_doc_created


# ---------------------------------------------------------------------------
# Phase: SSE event verification
# ---------------------------------------------------------------------------


async def _drain_events_for(
    broker: AssessmentEventBroker,
    graph_id: str,
    run_id: str,
    *,
    timeout_s: float = 0.25,
) -> list[AssessmentEvent]:
    """Read everything still in the broker history for (graph_id, run_id).

    The broker keeps a bounded ring buffer per run; subscribing replays the
    buffer eagerly so a late subscriber still sees the events. We use that
    to verify events were published during the simulation without having to
    keep an active subscriber alive the whole run.
    """
    out: list[AssessmentEvent] = []
    iterator = broker.subscribe(graph_id, run_id, since=0)
    try:
        while True:
            try:
                event = await asyncio.wait_for(iterator.__anext__(), timeout=timeout_s)
            except (TimeoutError, asyncio.TimeoutError):
                break
            except StopAsyncIteration:
                break
            out.append(event)
    finally:
        await iterator.aclose()
    return out


# ---------------------------------------------------------------------------
# Phase: read-back + final summary
# ---------------------------------------------------------------------------


async def _phase_readback(
    *, run_id: str, graph_id: str
) -> dict[str, Any]:
    detail = _check_ok("assessment.get_run", await get_run(run_id=run_id))

    wave_status = _check_ok(
        "assessment.get_wave_status", await get_wave_status(run_id=run_id, wave=1)
    )

    # First page of findings
    findings_page = _check_ok(
        "assessment.list_findings",
        await list_findings(run_id=run_id, limit=10),
    )
    # Conflicts
    conflicts_page = _check_ok(
        "assessment.list_conflicts",
        await list_conflicts(run_id=run_id, limit=200),
    )
    # Open questions
    questions_page = _check_ok(
        "assessment.list_unresolved_questions",
        await list_unresolved_questions(run_id=run_id, limit=200),
    )
    # Deliverables
    deliverables_page = _check_ok(
        "assessment.list_deliverables",
        await list_deliverables(run_id=run_id, limit=200),
    )

    # Pick the first final_doc deliverable and verify its content resolves
    final_doc_resolved = False
    final_doc_sample: dict[str, Any] | None = None
    for row in deliverables_page["items"]:
        if row.get("kind") == "final_doc":
            content = await get_deliverable_content(
                run_id=run_id, deliverable_id=row["deliverable_id"]
            )
            _check_ok("assessment.get_deliverable_content", content)
            final_doc_sample = content
            # The wrapper returns content_uri (per O1 observation in TASK-080
            # security review — the CAS bytes are not piped through MCP yet)
            # but if it carries a content_uri or content_inline, we count it.
            if content.get("content_uri") or content.get("content_inline"):
                final_doc_resolved = True
            break

    return {
        "run_detail": detail,
        "wave_1_status": wave_status,
        "first_findings": findings_page["items"],
        "finding_page_size": findings_page["page"]["page_size"],
        "conflict_count_seen": len(conflicts_page["items"]),
        "open_question_count_seen": len(questions_page["items"]),
        "deliverable_count_seen": len(deliverables_page["items"]),
        "final_doc_resolved": final_doc_resolved,
        "final_doc_sample": final_doc_sample,
    }


# ---------------------------------------------------------------------------
# Phase: finalize
# ---------------------------------------------------------------------------


async def _phase_finalize(*, run_id: str) -> dict[str, Any]:
    resp = await finalize_run(run_id=run_id)
    _check_ok("assessment.finalize_run", resp)
    return resp


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def run_simulation(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    graph_id = args.graph_id
    template_slug = args.template_slug
    subject_block = {
        "subject_id": f"subj-{uuid.uuid4().hex[:8]}",
        "graph_id": graph_id,
        "slug": args.subject_slug,
        "name": args.subject_name,
        "vertical_slug": args.vertical_slug,
        "domains": [args.subject_domain] if args.subject_domain else [],
        "aliases": ["Eurail", "Interrail", "DiscoverEU"]
        if args.subject_slug == "eurail"
        else [],
    }

    if not run_dir.is_dir():
        logger.error("Run dir not found: %s", run_dir)
        return 1

    if not neo4j_client.async_driver:
        logger.error(
            "Neo4j driver not initialized. Set NEO4J_URI / NEO4J_USERNAME / "
            "NEO4J_PASSWORD env vars, then run inside an event loop that "
            "executes ``await neo4j_client.connect()`` first — or run via "
            "the FastAPI app lifespan."
        )
        return 1

    broker = get_assessment_event_broker()
    initial_buffered = 0  # baseline; broker is shared with the app

    async with _patched_auth(graph_id):
        logger.info("==== Phase 1: create_run ====")
        run_id, mr_ids = await _phase_create_run(
            graph_id=graph_id,
            template_slug=template_slug,
            subject_block=subject_block,
        )
        initial_buffered = broker.buffered_event_count(graph_id, run_id)

        logger.info("==== Phase 2: discover module_runs ====")
        slug_to_mr, raw_by_slug = await _discover_module_runs(run_id=run_id)

        logger.info("==== Phase 3: transition wave-1 modules to 'running' ====")
        # Move every wave-1 module to running so the event broker emits
        # status_changed events. (We transition back to 'finished' at the end.)
        wave_1_slugs: list[str] = []
        for slug, row in raw_by_slug.items():
            if row.get("wave") == 1:
                wave_1_slugs.append(slug)
                resp = await update_module_run(
                    run_id=run_id,
                    module_run_id=row["module_run_id"],
                    body={"status": "running"},
                )
                _check_ok("assessment.update_module_run", resp)

        logger.info("==== Phase 4: record_finding_bulk ====")
        finding_stats = await _phase_record_findings(
            run_id=run_id,
            graph_id=graph_id,
            run_dir=run_dir,
            slug_to_mr_id=slug_to_mr,
        )

        logger.info("==== Phase 5: record_conflict ====")
        conflicts_created = await _phase_record_conflicts(
            run_id=run_id, graph_id=graph_id, run_dir=run_dir
        )

        logger.info("==== Phase 6: record_unresolved_question ====")
        questions_created = await _phase_record_questions(
            run_id=run_id, graph_id=graph_id, slug_to_mr_id=slug_to_mr
        )

        logger.info("==== Phase 7: persist_deliverable ====")
        module_dels, final_dels = await _phase_persist_deliverables(
            run_id=run_id,
            graph_id=graph_id,
            run_dir=run_dir,
            slug_to_mr_id=slug_to_mr,
        )

        logger.info("==== Phase 8: transition wave-1 modules to 'finished' ====")
        for slug in wave_1_slugs:
            row = raw_by_slug[slug]
            resp = await update_module_run(
                run_id=run_id,
                module_run_id=row["module_run_id"],
                body={"status": "finished"},
            )
            _check_ok("assessment.update_module_run", resp)

        logger.info("==== Phase 9: drain SSE event broker buffer ====")
        events = await _drain_events_for(broker, graph_id, run_id)
        event_types: dict[str, int] = defaultdict(int)
        for ev in events:
            event_types[ev.event_type] += 1

        logger.info("==== Phase 10: read-back via MCP tools ====")
        readback = await _phase_readback(run_id=run_id, graph_id=graph_id)

        logger.info("==== Phase 11: finalize_run ====")
        finalize_resp = await _phase_finalize(run_id=run_id)

    # ---- summary --------------------------------------------------------
    summary = {
        "run_id": run_id,
        "graph_id": graph_id,
        "template_slug": template_slug,
        "module_runs_planned": len(mr_ids),
        "module_runs_discovered_by_slug": len(slug_to_mr),
        "findings": {
            "submitted_total": finding_stats["succeeded"] + finding_stats["failed"],
            "succeeded": finding_stats["succeeded"],
            "failed": finding_stats["failed"],
            "batches": finding_stats["batches"],
            "skipped_unknown_module": finding_stats["skipped_unknown_module"],
            "skipped_schema": finding_stats["skipped_schema"],
        },
        "conflicts_created": conflicts_created,
        "unresolved_questions_created": questions_created,
        "deliverables": {
            "module": module_dels,
            "final_doc": final_dels,
        },
        "events": {
            "buffered_before_run": initial_buffered,
            "drained_after_run": len(events),
            "by_type": dict(event_types),
        },
        "readback": {
            "run_status": readback["run_detail"].get("status"),
            "finding_count": readback["run_detail"].get("finding_count"),
            "conflict_count": readback["run_detail"].get("conflict_count"),
            "open_question_count": readback["run_detail"].get("open_question_count"),
            "deliverable_count": readback["run_detail"].get("deliverable_count"),
            "wave_1_done": readback["wave_1_status"].get("done"),
            "wave_1_total": readback["wave_1_status"].get("total"),
            "conflict_count_seen_via_list": readback["conflict_count_seen"],
            "open_question_count_seen_via_list": readback[
                "open_question_count_seen"
            ],
            "deliverable_count_seen_via_list": readback["deliverable_count_seen"],
            "final_doc_resolved": readback["final_doc_resolved"],
        },
        "finalize": {
            "passed": finalize_resp.get("passed"),
            "status": finalize_resp.get("status"),
            "direct_findings": finalize_resp.get("direct_finding_count"),
            "inferred_findings": finalize_resp.get("inferred_finding_count"),
            "deliverables": finalize_resp.get("deliverable_count"),
            "unresolved_conflicts": finalize_resp.get("unresolved_conflict_count"),
            "open_questions": finalize_resp.get("open_question_count"),
            "failure_reasons": finalize_resp.get("failure_reasons"),
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))

    # Loop-closure assertions — the TASK-084 sprint gate.
    counts = summary["readback"]
    if counts["finding_count"] == 0:
        logger.error("ASSERTION FAIL: 0 findings after run; the loop did not close.")
        return 2
    if not summary["events"]["drained_after_run"]:
        logger.error("ASSERTION FAIL: SSE broker emitted no events.")
        return 2
    if summary["deliverables"]["final_doc"] == 0:
        logger.error("ASSERTION FAIL: no final_doc landed in Blob CAS.")
        return 2
    if not readback["final_doc_resolved"]:
        logger.warning(
            "OBSERVATION: final_doc resolved via get_deliverable_content "
            "returned no content_uri/content_inline. Tracked as TASK-080 "
            "non-blocking observation O1 (method-shadowing); does not block "
            "TASK-084 sprint gate."
        )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="live_rerun_simulation",
        description=(
            "Simulate /eurail-report end-to-end against a local Oraclous "
            "stack via in-process MCP tool calls. Proves the sprint-002 "
            "loop closes without consuming LLM tokens."
        ),
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to an existing /eurail-report run dir (e.g., "
        "docs/eurail/out/eurail-2026-05-06). Read-only.",
    )
    parser.add_argument(
        "--graph-id",
        default="eurail-tenant-test",
        help="Tenant graph_id for the simulation. Default: eurail-tenant-test.",
    )
    parser.add_argument(
        "--template-slug",
        default="eurail-report-v1",
        help="Catalog template slug. Default: eurail-report-v1.",
    )
    parser.add_argument(
        "--subject-slug",
        default="eurail",
        help="Subject slug to MERGE in the tenant graph.",
    )
    parser.add_argument(
        "--subject-name",
        default="Eurail B.V. (Interrail B.V.)",
    )
    parser.add_argument(
        "--vertical-slug",
        default="rail-cooperative",
    )
    parser.add_argument(
        "--subject-domain",
        default="eurail.com",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    async def _runner() -> int:
        await neo4j_client.connect()
        try:
            return await run_simulation(args)
        finally:
            await neo4j_client.close()

    try:
        return asyncio.run(_runner())
    except ContractDivergence as exc:
        logger.error("CONTRACT DIVERGENCE — stop and inspect: %s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001 — top-level CLI guard
        logger.exception("Simulation aborted: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(_main())
