#!/usr/bin/env python
"""
Backfill an existing /assess or /eurail-report run into the assessment substrate.

STORY-026 — SPRINT-001 validation gate (TASK-071).

The script loads a completed filesystem run (the legacy output layout under
``docs/eurail/out/<run>/``) into the new Neo4j-backed schema via the REST API
landed by TASK-069. It is the round-trip-lossless proof point for the
schema + write API. If this can't load the existing Eurail 2026-05-06 run
end-to-end, the schema is wrong.

What it loads
-------------

Per the run-directory layout used by the ``/assess`` and ``/eurail-report``
Claude Code skills:

1. ``evidence/evidence.jsonl`` — one JSON object per line, ``ev-*`` ids,
   grouped by ``module`` slug.
2. ``evidence/conflicts.jsonl`` — one JSON object per line, ``cf-*`` ids,
   referencing ``evidence_ids`` from the evidence file.
3. ``NN_*.md`` — numbered module deliverables at the run root.
4. ``ddN_*.md`` — deep-dive deliverables at the run root.
5. ``gap-NN_*.md`` — gap-research deliverables at the run root.
6. (optional) ``final/*.html`` / ``final/*.pdf`` / ``final/*.md`` — the 5-doc
   final set produced by ``docify`` after a run, when present.

How it loads
------------

The script is REST-API-first per the task spec. For each entity it calls the
endpoints landed by TASK-069::

    POST   /api/v1/api/v1/assessments/runs                            → create_run
    PATCH  /api/v1/api/v1/assessments/runs/{run}/module-runs/{mr}     → update_module_run
    POST   /api/v1/api/v1/assessments/runs/{run}/findings:bulk        → record_finding_bulk
    POST   /api/v1/api/v1/assessments/runs/{run}/conflicts            → record_conflict
    POST   /api/v1/api/v1/assessments/runs/{run}/deliverables         → persist_deliverable
    POST   /api/v1/api/v1/assessments/runs/{run}/deliverables:bulk-final → persist_final_docs
    POST   /api/v1/api/v1/assessments/runs/{run}:finalize             → finalize_run

The double-``/api/v1`` prefix is intentional — the assessment router is
mounted under ``/api/v1`` inside an outer ``/api/v1``-prefixed router (see
``app/api/v1/router.py``). Tests under ``tests/integration/test_assessments_endpoints.py``
use the same ``/api/v1/api/v1/assessments`` base.

Gap-research modules
--------------------

The Eurail 2026-05-06 run's ``evidence.jsonl`` references gap-research module
slugs (``gap-silent-operators``, ``gap-01-b2b``, ``gap-05-silent-pilots``)
that are **not** part of the static ``eurail-report-v1`` catalog seeded by
TASK-070 — gap-research is a dynamic wave inserted mid-run by the orchestrator.
The TASK-069 REST API only exposes ``update_module_run`` (not "create"), so
the script needs a small bypass for these ad-hoc ``:ModuleRun`` rows:

* When ``--neo4j-uri/--neo4j-user/--neo4j-password`` are supplied, the script
  inserts ad-hoc ``:Module`` + ``:ModuleRun`` rows directly via Cypher,
  scoped to the same tenant graph. The ``:Module`` is anchored to the
  catalog graph with ``kind='gap-research', wave=2``. This is the *only*
  direct-graph write the script makes; all findings/conflicts/deliverables
  for these modules still go through the REST API.
* Without Neo4j creds the script logs a warning and skips gap-research
  findings, surfacing the skipped count in the final summary.

Lossless validation per STORY-026 §Verification requires Neo4j creds.

Slug resolution
---------------

The JSONL ``module`` field is matched against ``:Module.slug`` in the
catalog. Resolution is case-insensitive and trim-aware. A few legacy slugs
in the Eurail run don't exactly match the catalog (e.g. ``cust-journey`` →
``customer-journey``). A small static alias map (LEGACY_SLUG_ALIASES) handles
these. Unknown slugs are logged + skipped with a counter — the script never
silently drops records.

Edge cases / robustness
-----------------------

* JSONL lines that fail to parse are logged with line number and counted —
  the script keeps going. The Eurail 2026-05-06 run has one such line (two
  objects concatenated); the script handles it via ``raw_decode`` so even
  malformed input round-trips losslessly.
* Confidence values in the legacy JSONL are strings ("HIGH"/"MEDIUM"/"LOW");
  the new schema requires a float in [0.0, 1.0]. The script maps via
  CONFIDENCE_STR_TO_FLOAT.
* Label values include ``None`` and one ``"ASSUMPTION"`` outlier. The script
  normalizes via LABEL_NORMALIZE (default → ``DIRECT``, ``ASSUMPTION`` →
  ``INFERRED``).
* Conflict records use ``resolution`` (string label) where the new schema
  uses ``status`` (open / resolved / accepted_open). The script maps via
  CONFLICT_RESOLUTION_TO_STATUS.
* Gap-research rows in the Eurail run use ``gap_id`` instead of ``id`` and
  carry ``finding`` / ``missing`` / ``searched`` instead of ``claim`` / ``raw``.
  ``_resolve_record_id`` + ``_evidence_to_finding`` fold these into the new
  schema shape.

Idempotency
-----------

Every REST call MERGEs on a natural id, so re-running the script is safe:
already-persisted findings come back with ``already_existed=true`` in the
bulk response and are counted under ``replayed``.

Usage
-----

::

    python -m app.scripts.backfill_assessment_run \\
        --run-dir /Users/reza/workspace/Oraclous/docs/eurail/out/eurail-2026-05-06 \\
        --graph-id eurail-tenant-graph \\
        --template-slug eurail-report-v1 \\
        --api-base http://localhost:8000/api/v1 \\
        --token <service-account JWT> \\
        --neo4j-uri bolt://localhost:7687 \\
        --neo4j-user neo4j --neo4j-password test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import httpx

logger = logging.getLogger("backfill_assessment_run")


# =============================================================================
# Constants & normalization tables
# =============================================================================

DEFAULT_TEMPLATE_SLUG = "eurail-report-v1"
DEFAULT_API_BASE = "http://localhost:8000/api/v1"
DEFAULT_RUN_DIR = "/Users/reza/workspace/Oraclous/docs/eurail/out/eurail-2026-05-06"

#: Threshold above which deliverable content is referenced by ``content_uri``
#: instead of stored inline as ``content_inline``. The schema docstring says
#: < 50 KB inline; we conservatively use 50 000 chars to stay well under any
#: Neo4j string-property soft limit.
INLINE_CONTENT_MAX_CHARS = 50_000

#: Bulk POST chunk size. The bulk endpoint already does per-record writes
#: server-side; this chunk keeps each HTTP request body manageable.
BULK_FINDING_CHUNK = 100

#: Confidence is a string in legacy evidence.jsonl; the new schema wants a
#: float in [0.0, 1.0]. The mapping reflects what each label meant in the
#: source pipeline (eyeballed from a sample — feedback welcome from QA).
CONFIDENCE_STR_TO_FLOAT: dict[str, float] = {
    "HIGH": 0.9,
    "MEDIUM": 0.6,
    "LOW": 0.3,
}

#: The schema constrains label to {DIRECT, INFERRED, CONTRADICTION}. The
#: Eurail run has 18 nulls + 1 ASSUMPTION outlier; we map deterministically.
LABEL_NORMALIZE: dict[Optional[str], str] = {
    None: "DIRECT",
    "": "DIRECT",
    "DIRECT": "DIRECT",
    "INFERRED": "INFERRED",
    "CONTRADICTION": "CONTRADICTION",
    "ASSUMPTION": "INFERRED",
}

#: Conflict resolution → status. Legacy uses uppercase labels; new schema
#: uses lowercase enum status.
CONFLICT_RESOLUTION_TO_STATUS: dict[Optional[str], str] = {
    None: "open",
    "": "open",
    "OPEN": "open",
    "RESOLVED": "resolved",
    "ACCEPTED_OPEN": "accepted_open",
    "ACCEPTED-OPEN": "accepted_open",
}

#: Module-slug aliases used by the legacy Eurail run that don't exactly
#: match the catalog slugs seeded by TASK-070. Direction: legacy → canonical.
LEGACY_SLUG_ALIASES: dict[str, str] = {
    "cust-journey": "customer-journey",
}

#: Match for the leading ordinal on filenames like ``01_eurail_today.md``,
#: ``21_adversarial_redline.md``, ``ddN_*.md``, ``gap-NN_*.md``.
RE_NUMBERED_MD = re.compile(r"^(\d{1,3})_.*\.md$")
RE_DD_MD = re.compile(r"^dd(\d{1,3})_.*\.md$")
RE_GAP_MD = re.compile(r"^gap-(\d{1,3})_.*\.md$")


# =============================================================================
# Stats container
# =============================================================================


@dataclass
class BackfillStats:
    """Summary numbers the script prints at the end (and tests assert on)."""

    findings_total: int = 0
    findings_written: int = 0
    findings_replayed: int = 0
    findings_skipped_unknown_module: int = 0
    findings_skipped_malformed: int = 0
    findings_failed: int = 0

    conflicts_total: int = 0
    conflicts_written: int = 0
    conflicts_replayed: int = 0
    conflicts_skipped_malformed: int = 0
    conflicts_failed: int = 0

    deliverables_total: int = 0
    deliverables_written: int = 0
    deliverables_replayed: int = 0
    deliverables_failed: int = 0

    final_docs_total: int = 0
    final_docs_written: int = 0

    module_runs_by_status: Counter = field(default_factory=Counter)

    jsonl_parse_errors: int = 0

    skipped_modules: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["module_runs_by_status"] = dict(self.module_runs_by_status)
        d["skipped_modules"] = dict(self.skipped_modules)
        return d


# =============================================================================
# JSONL streaming
# =============================================================================


def _iter_jsonl_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Stream records from a JSONL file.

    Yields ``(line_number, record)`` tuples. Handles the case where one
    physical line contains multiple concatenated JSON objects (which the
    Eurail run's ``evidence.jsonl`` has on line 247 due to a writer bug in
    the legacy pipeline) by feeding ``json.JSONDecoder.raw_decode`` until
    the line is exhausted.

    Malformed objects are emitted as ``{"__parse_error__": <msg>}`` so the
    caller can count them without aborting the stream.
    """
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        for ln, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            pos = 0
            while pos < len(stripped):
                try:
                    obj, end = decoder.raw_decode(stripped, pos)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "JSONL parse error in %s line %d col %d: %s",
                        path,
                        ln,
                        pos,
                        exc.msg,
                    )
                    yield (ln, {"__parse_error__": exc.msg})
                    break
                yield (ln, obj)
                pos = end
                # Skip whitespace between concatenated objects.
                while pos < len(stripped) and stripped[pos] in " \t":
                    pos += 1


# =============================================================================
# Value normalization
# =============================================================================


def _normalize_module_slug(raw_slug: str) -> str:
    """Lowercase + trim + alias-translate a legacy module slug."""
    s = (raw_slug or "").strip().lower()
    return LEGACY_SLUG_ALIASES.get(s, s)


def _normalize_confidence(raw: Any) -> float:
    """Coerce confidence to a float in [0.0, 1.0].

    Strings map via ``CONFIDENCE_STR_TO_FLOAT``; numerics are clamped;
    None / unknown defaults to 0.0 (signals "no opinion" downstream).
    """
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        v = float(raw)
        return max(0.0, min(1.0, v))
    if isinstance(raw, str):
        upper = raw.strip().upper()
        if upper in CONFIDENCE_STR_TO_FLOAT:
            return CONFIDENCE_STR_TO_FLOAT[upper]
        # Try parse a stringified float ("0.7", "1.0").
        try:
            v = float(upper)
            return max(0.0, min(1.0, v))
        except ValueError:
            return 0.0
    return 0.0


def _normalize_label(raw: Any) -> str:
    """Map legacy label to schema enum {DIRECT, INFERRED, CONTRADICTION}."""
    key = raw if raw is None else str(raw).strip().upper()
    if key == "":
        key = None
    return LABEL_NORMALIZE.get(key, "DIRECT")


def _normalize_conflict_status(raw: Any) -> str:
    """Map legacy ``resolution`` field to schema ``status`` enum."""
    key = raw if raw is None else str(raw).strip().upper()
    return CONFLICT_RESOLUTION_TO_STATUS.get(key, "open")


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _resolve_record_id(record: dict[str, Any]) -> Optional[str]:
    """Pick the natural id off a legacy JSONL record.

    The Eurail 2026-05-06 run has three id shapes:
    * 596 records: ``id`` (e.g. ``ev-company-001``)
    * 5 records:   ``gap_id`` (e.g. ``gap-b2b-volume-001``) — gap-research
      summary rows that the legacy pipeline emitted with a different key
    """
    return record.get("id") or record.get("gap_id") or record.get("finding_id")


# =============================================================================
# REST client — thin wrapper around httpx
# =============================================================================


class AssessmentApiClient:
    """REST client speaking the TASK-069 assessment endpoints.

    Path prefix is the FULL prefix including the double-``/api/v1``. The
    constructor takes the ``--api-base`` value (e.g.
    ``http://localhost:8000/api/v1``) and appends ``/api/v1/assessments``
    so callers can use bare resource paths like ``/runs/{run_id}/...``.

    Tests can inject a pre-built ``httpx.AsyncClient`` (typically with an
    ``ASGITransport`` pointing at the in-process FastAPI app) so the same
    code path serves both production and integration tests.
    """

    def __init__(
        self,
        base_url: str,
        token: Optional[str],
        timeout: float = 30.0,
        *,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        # The router is mounted under /api/v1 in main.py AND the assessments
        # router itself has its own prefix=/api/v1 in v1/router.py — so the
        # effective absolute path is /api/v1/api/v1/assessments/...
        # (verified by tests/integration/test_assessments_endpoints.py).
        if base_url.endswith("/"):
            base_url = base_url[:-1]
        self._url_root = f"{base_url}/api/v1/assessments"
        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if http_client is not None:
            self._owns_client = False
            self._client = http_client
            if token:
                self._client.headers["Authorization"] = f"Bearer {token}"
        else:
            self._owns_client = True
            self._client = httpx.AsyncClient(
                timeout=timeout, headers=headers, follow_redirects=True
            )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "AssessmentApiClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def create_run(self, body: dict[str, Any]) -> dict[str, Any]:
        r = await self._client.post(f"{self._url_root}/runs", json=body)
        r.raise_for_status()
        return r.json()

    async def update_module_run(
        self, run_id: str, module_run_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        r = await self._client.patch(
            f"{self._url_root}/runs/{run_id}/module-runs/{module_run_id}",
            json=body,
        )
        r.raise_for_status()
        return r.json()

    async def record_findings_bulk(
        self, run_id: str, findings: list[dict[str, Any]]
    ) -> dict[str, Any]:
        r = await self._client.post(
            f"{self._url_root}/runs/{run_id}/findings:bulk",
            json={"findings": findings},
        )
        # 207 is multi-status — surface anyway.
        if r.status_code not in (200, 207):
            r.raise_for_status()
        return r.json()

    async def record_conflict(
        self, run_id: str, conflict: dict[str, Any]
    ) -> dict[str, Any]:
        r = await self._client.post(
            f"{self._url_root}/runs/{run_id}/conflicts",
            json={"conflict": conflict},
        )
        r.raise_for_status()
        return r.json()

    async def persist_deliverable(
        self, run_id: str, deliverable: dict[str, Any]
    ) -> dict[str, Any]:
        r = await self._client.post(
            f"{self._url_root}/runs/{run_id}/deliverables",
            json={"deliverable": deliverable},
        )
        r.raise_for_status()
        return r.json()

    async def persist_final_docs(
        self, run_id: str, deliverables: list[dict[str, Any]]
    ) -> dict[str, Any]:
        r = await self._client.post(
            f"{self._url_root}/runs/{run_id}/deliverables:bulk-final",
            json={"deliverables": deliverables},
        )
        if r.status_code not in (200, 207):
            r.raise_for_status()
        return r.json()

    async def finalize_run(self, run_id: str) -> dict[str, Any]:
        r = await self._client.post(f"{self._url_root}/runs/{run_id}:finalize")
        r.raise_for_status()
        return r.json()


# =============================================================================
# Gap-research module insertion (direct Neo4j write — admin path)
# =============================================================================


async def _insert_gap_module_and_module_run(
    *,
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    catalog_graph_id: str,
    tenant_graph_id: str,
    template_id: str,
    run_id: str,
    module_slug: str,
    driver: Optional[Any] = None,
) -> tuple[str, str]:
    """Insert an ad-hoc gap-research :Module + :ModuleRun via direct Cypher.

    Returns ``(module_id, module_run_id)``.

    Why this exists: TASK-069's REST API exposes ``update_module_run`` only —
    no ``create_module_run`` — because the catalog-seeded ModuleRuns are
    pre-created by ``create_run``. Gap-research modules are dynamically
    inserted mid-run per STORY-026 §Coordination Model and need a way in.
    This direct-Cypher helper is the script's only non-REST path.

    The inserted :Module lives in the catalog graph with
    ``kind='gap-research', wave=2`` (matching the wave layout in ADR-018);
    the :ModuleRun lives in the tenant graph and is wired to the existing
    :AssessmentRun via ``HAS_MODULE_RUN``. Both nodes carry the platform
    marker ``:__Platform__`` so they participate in the same ReBAC/audit
    machinery as the seeded catalog.
    """
    module_id = f"{template_id}__{module_slug}"
    module_run_id = _new_id("mr")

    own_driver = False
    if driver is None:
        from neo4j import AsyncGraphDatabase

        if not (neo4j_uri and neo4j_user and neo4j_password):
            raise ValueError(
                "_insert_gap_module_and_module_run requires either a driver or "
                "neo4j_uri+user+password"
            )
        driver = AsyncGraphDatabase.driver(
            neo4j_uri, auth=(neo4j_user, neo4j_password)
        )
        own_driver = True
    try:
        # 1. MERGE the Module in the catalog graph.
        await driver.execute_query(
            """
            MERGE (m:Module:__Platform__ {module_id: $module_id})
            ON CREATE SET
                m.graph_id    = $catalog_graph_id,
                m.template_id = $template_id,
                m.slug        = $slug,
                m.name        = $name,
                m.wave        = 2,
                m.ordinal     = 0,
                m.kind        = 'gap-research'
            WITH m
            MATCH (t:AssessmentTemplate:__Platform__ {template_id: $template_id})
            MERGE (t)-[:HAS_MODULE]->(m)
            """,
            {
                "module_id": module_id,
                "catalog_graph_id": catalog_graph_id,
                "template_id": template_id,
                "slug": module_slug,
                "name": module_slug.replace("-", " ").title(),
            },
        )
        # 2. MERGE the ModuleRun in the tenant graph + wire to the run.
        await driver.execute_query(
            """
            MERGE (mr:ModuleRun:__Platform__ {module_run_id: $module_run_id})
            ON CREATE SET
                mr.graph_id       = $tenant_graph_id,
                mr.run_id         = $run_id,
                mr.module_id      = $module_id,
                mr.wave           = 2,
                mr.status         = 'planned',
                mr.evidence_count = 0
            WITH mr
            MATCH (r:AssessmentRun:__Platform__ {graph_id: $tenant_graph_id, run_id: $run_id})
            MERGE (r)-[:HAS_MODULE_RUN]->(mr)
            """,
            {
                "module_run_id": module_run_id,
                "tenant_graph_id": tenant_graph_id,
                "run_id": run_id,
                "module_id": module_id,
            },
        )
    finally:
        if own_driver:
            await driver.close()
    return module_id, module_run_id


async def _fetch_module_run_id_to_slug(
    *,
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    tenant_graph_id: str,
    run_id: str,
    driver: Optional[Any] = None,
) -> dict[str, str]:
    """Read the (module_run_id → module slug) mapping for a run.

    Used after ``create_run`` so the script knows which pre-created module
    run to PATCH for a given JSONL ``module`` slug. The catalog ``:Module``
    rows live in a different graph from the tenant ``:ModuleRun`` rows; the
    join is by ``module_id``.
    """
    own_driver = False
    if driver is None:
        from neo4j import AsyncGraphDatabase

        if not (neo4j_uri and neo4j_user and neo4j_password):
            raise ValueError(
                "_fetch_module_run_id_to_slug requires either a driver or "
                "neo4j_uri+user+password"
            )
        driver = AsyncGraphDatabase.driver(
            neo4j_uri, auth=(neo4j_user, neo4j_password)
        )
        own_driver = True
    try:
        result = await driver.execute_query(
            """
            MATCH (r:AssessmentRun:__Platform__ {graph_id: $tenant_graph_id, run_id: $run_id})
                  -[:HAS_MODULE_RUN]->(mr:ModuleRun:__Platform__)
            MATCH (m:Module:__Platform__ {module_id: mr.module_id})
            RETURN mr.module_run_id AS mr_id, m.slug AS slug
            """,
            {"tenant_graph_id": tenant_graph_id, "run_id": run_id},
        )
        return {rec["mr_id"]: rec["slug"] for rec in result.records}
    finally:
        if own_driver:
            await driver.close()


# =============================================================================
# Finding / Conflict / Deliverable mappers
# =============================================================================


def _evidence_to_finding(
    record: dict[str, Any],
    *,
    graph_id: str,
    run_id: str,
    module_run_id: str,
) -> dict[str, Any]:
    """Translate a legacy ``evidence.jsonl`` record into the new Finding schema."""
    source_block = record.get("source") or {}
    # The legacy file has no `source_id`; derive a deterministic id from the
    # source URL (or name fallback). The service MERGEs :Source in the
    # catalog graph by source_id so this is the dedup key.
    source_id = None
    if isinstance(source_block, dict):
        url = source_block.get("url") or source_block.get("url_normalized")
        name = source_block.get("name")
        if url:
            # UUID5 over the URL → stable id without smuggling the raw URL
            # into the id string (clean and short).
            source_id = f"src-{uuid.uuid5(uuid.NAMESPACE_URL, str(url)).hex}"
        elif name:
            source_id = f"src-{uuid.uuid5(uuid.NAMESPACE_OID, str(name)).hex}"

    finding_id = _resolve_record_id(record)
    if not finding_id:
        # Caller should pre-check; this is a defensive fallback.
        finding_id = _new_id("ev")

    # Some gap-research records use a richer flat shape (gap_id / missing /
    # searched / finding / recommend_ask) instead of the canonical
    # {id, claim, raw}. Fold the closest free-text field into ``claim``.
    claim = (
        record.get("claim")
        or record.get("finding")
        or record.get("missing")
        or ""
    )

    return {
        "finding_id": finding_id,
        "graph_id": graph_id,
        "run_id": run_id,
        "module_run_id": module_run_id,
        "claim": claim,
        "raw": record.get("raw") or record.get("searched"),
        "label": _normalize_label(record.get("label")),
        "confidence": _normalize_confidence(record.get("confidence")),
        "dimensions": record.get("dimensions") or [],
        "ai_adoption_relevance": None,  # legacy stored prose; new schema is float
        "notes": (
            record.get("ai_adoption_relevance")  # preserve legacy prose in notes
            if isinstance(record.get("ai_adoption_relevance"), str)
            else record.get("notes")
        ),
        "superseded_by": None,
        "source_id": source_id,
        "source_quote": record.get("raw"),
        "source_locator": None,
    }


def _conflict_to_payload(
    record: dict[str, Any], *, graph_id: str, run_id: str
) -> dict[str, Any]:
    """Translate a legacy ``conflicts.jsonl`` record into the new Conflict schema."""
    return {
        "conflict_id": record["id"],
        "graph_id": graph_id,
        "run_id": run_id,
        "topic": record.get("topic", "") or "",
        "summary": record.get("summary", "") or "",
        "status": _normalize_conflict_status(record.get("resolution")),
        "resolution": record.get("explanation"),
        "synthesis_note": record.get("synthesis_note"),
        "involved_finding_ids": list(record.get("evidence_ids") or []),
    }


def _deliverable_from_md_file(
    path: Path,
    *,
    graph_id: str,
    run_id: str,
    module_run_id: Optional[str],
    ordinal: int,
) -> dict[str, Any]:
    """Build a Deliverable payload from a markdown file on disk."""
    raw = path.read_text(encoding="utf-8")
    size = len(raw)
    inline = raw if size <= INLINE_CONTENT_MAX_CHARS else None
    return {
        "deliverable_id": f"del-{uuid.uuid5(uuid.NAMESPACE_OID, f'{run_id}::{path.name}').hex}",
        "graph_id": graph_id,
        "run_id": run_id,
        "module_run_id": module_run_id,
        "kind": "module-md",
        "filename": path.name,
        "ordinal": ordinal,
        "content_uri": path.as_uri(),
        "content_inline": inline,
        "sha256": None,
        "word_count": len(raw.split()),
    }


def _final_doc_from_file(
    path: Path,
    *,
    graph_id: str,
    run_id: str,
    ordinal: int,
) -> dict[str, Any]:
    """Build a final-doc Deliverable payload (HTML/PDF/MD) from disk."""
    suffix = path.suffix.lower()
    kind = {
        ".html": "final-html",
        ".htm": "final-html",
        ".pdf": "final-pdf",
        ".md": "final-md",
    }.get(suffix, "final-md")
    if kind == "final-pdf":
        # PDFs are binary — never inline them. content_uri only for SPRINT-1.
        raw_text = ""
        inline = None
        word_count = None
    else:
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        inline = raw_text if len(raw_text) <= INLINE_CONTENT_MAX_CHARS else None
        word_count = len(raw_text.split())
    return {
        "deliverable_id": f"final-{uuid.uuid5(uuid.NAMESPACE_OID, f'{run_id}::{path.name}').hex}",
        "graph_id": graph_id,
        "run_id": run_id,
        "module_run_id": None,
        "kind": kind,
        "filename": path.name,
        "ordinal": ordinal,
        "content_uri": path.as_uri(),
        "content_inline": inline,
        "sha256": None,
        "word_count": word_count,
    }


# =============================================================================
# File walker
# =============================================================================


def _walk_module_md_files(run_dir: Path) -> list[tuple[int, str, Path]]:
    """Return ``(ordinal, kind_hint, path)`` for every NN_/ddN_/gap-NN_ file.

    ``kind_hint`` is one of ``"numbered"``, ``"deep-dive"``, ``"gap"`` and is
    used by the caller to decide which module_run to attach the deliverable
    to (deep-dives and gap- files use a different anchor than NN_ files,
    which line up with the seeded catalog modules by ordinal).
    """
    out: list[tuple[int, str, Path]] = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_file() or child.suffix.lower() != ".md":
            continue
        name = child.name
        m = RE_GAP_MD.match(name)
        if m:
            out.append((int(m.group(1)), "gap", child))
            continue
        m = RE_DD_MD.match(name)
        if m:
            out.append((int(m.group(1)), "deep-dive", child))
            continue
        m = RE_NUMBERED_MD.match(name)
        if m:
            out.append((int(m.group(1)), "numbered", child))
            continue
    return out


def _walk_final_doc_files(run_dir: Path) -> list[Path]:
    final_dir = run_dir / "final"
    if not final_dir.is_dir():
        return []
    return sorted(
        p
        for p in final_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".html", ".htm", ".pdf", ".md")
    )


# =============================================================================
# Orchestration — the main loader
# =============================================================================


@dataclass
class BackfillConfig:
    run_dir: Path
    graph_id: str
    template_slug: str
    api_base: str
    token: Optional[str]
    subject_slug: str = "eurail"
    subject_name: str = "Eurail B.V."
    vertical_slug: str = "rail-cooperative"
    # Optional direct-graph creds for ad-hoc gap-research ModuleRuns and
    # the post-create_run module_run_id→slug join (REST doesn't expose it).
    neo4j_uri: Optional[str] = None
    neo4j_user: Optional[str] = None
    neo4j_password: Optional[str] = None
    catalog_graph_id: str = "__assessments_catalog__"


async def run_backfill(
    cfg: BackfillConfig,
    *,
    http_client: Optional[httpx.AsyncClient] = None,
    neo4j_driver: Optional[Any] = None,
) -> BackfillStats:
    """Drive the full backfill end-to-end. Returns the stats counter.

    ``http_client`` and ``neo4j_driver``, when supplied, let tests swap in an
    in-process ASGI client and a shared Neo4j driver — production CLI runs
    leave both ``None`` and the script builds them from the CLI flags.
    """
    stats = BackfillStats()
    if not cfg.run_dir.is_dir():
        raise FileNotFoundError(
            f"--run-dir does not exist or is not a directory: {cfg.run_dir}"
        )

    have_neo4j = bool(neo4j_driver) or bool(
        cfg.neo4j_uri and cfg.neo4j_user and cfg.neo4j_password
    )
    if not have_neo4j:
        logger.warning(
            "No --neo4j-* creds supplied; gap-research findings will be SKIPPED "
            "and the run will not be fully lossless. Pass --neo4j-uri/--neo4j-user/"
            "--neo4j-password to enable ad-hoc gap-research ModuleRun insertion."
        )

    async with AssessmentApiClient(
        cfg.api_base, cfg.token, http_client=http_client
    ) as api:
        # ── 1. POST /runs → captures run_id + pre-created module_run_ids
        create_body: dict[str, Any] = {
            "template_slug": cfg.template_slug,
            "subject": {
                "subject_id": _new_id("subj"),
                "graph_id": cfg.graph_id,
                "slug": cfg.subject_slug,
                "name": cfg.subject_name,
                "vertical_slug": cfg.vertical_slug,
                "domains": [],
                "aliases": [],
            },
            "cli_flags": {
                "backfill": True,
                "source_run_dir": str(cfg.run_dir),
                "skill": "eurail-report",
                "skill_version": "1",
            },
        }
        create_resp = await api.create_run(create_body)
        run_id: str = create_resp["run_id"]
        template_id: str = create_resp["template_id"]
        already_existed = bool(create_resp.get("already_existed"))
        logger.info(
            "create_run: run_id=%s template_id=%s module_runs=%d already_existed=%s",
            run_id,
            template_id,
            len(create_resp["module_run_ids"]),
            already_existed,
        )

        # ── 2. Build the (module_slug → module_run_id) mapping.
        slug_to_module_run_id: dict[str, str] = {}
        if have_neo4j:
            id_to_slug = await _fetch_module_run_id_to_slug(
                neo4j_uri=cfg.neo4j_uri,
                neo4j_user=cfg.neo4j_user,
                neo4j_password=cfg.neo4j_password,
                tenant_graph_id=cfg.graph_id,
                run_id=run_id,
                driver=neo4j_driver,
            )
            for mr_id, slug in id_to_slug.items():
                slug_to_module_run_id[slug] = mr_id
            logger.info(
                "resolved %d (module_slug → module_run_id) pairs from Neo4j",
                len(slug_to_module_run_id),
            )

        # ── 3. Stream evidence.jsonl, group by module.
        evidence_path = cfg.run_dir / "evidence" / "evidence.jsonl"
        per_module_findings: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if evidence_path.is_file():
            for ln, record in _iter_jsonl_records(evidence_path):
                if "__parse_error__" in record:
                    stats.jsonl_parse_errors += 1
                    continue
                if not isinstance(record, dict) or not _resolve_record_id(record):
                    stats.findings_skipped_malformed += 1
                    logger.warning(
                        "evidence.jsonl line %d: no id/gap_id/finding_id field, skipping",
                        ln,
                    )
                    continue
                stats.findings_total += 1
                module_slug = _normalize_module_slug(record.get("module") or "")
                if not module_slug:
                    stats.findings_skipped_unknown_module += 1
                    stats.skipped_modules[""] += 1
                    continue
                per_module_findings[module_slug].append(record)
        else:
            logger.warning("evidence.jsonl not found at %s", evidence_path)

        # ── 4. For each module slug with findings, ensure a ModuleRun exists,
        #      PATCH it to 'running', bulk-write findings, then PATCH 'finished'.
        for module_slug, records in per_module_findings.items():
            module_run_id = slug_to_module_run_id.get(module_slug)

            if module_run_id is None:
                # Unknown to the seeded catalog. If it looks gap-research,
                # try the direct-graph insert path; otherwise log + skip all.
                if module_slug.startswith("gap-") and have_neo4j:
                    try:
                        _, mr_id = await _insert_gap_module_and_module_run(
                            neo4j_uri=cfg.neo4j_uri,
                            neo4j_user=cfg.neo4j_user,
                            neo4j_password=cfg.neo4j_password,
                            catalog_graph_id=cfg.catalog_graph_id,
                            tenant_graph_id=cfg.graph_id,
                            template_id=template_id,
                            run_id=run_id,
                            module_slug=module_slug,
                            driver=neo4j_driver,
                        )
                        slug_to_module_run_id[module_slug] = mr_id
                        module_run_id = mr_id
                        logger.info(
                            "inserted gap-research ModuleRun: slug=%s module_run_id=%s",
                            module_slug,
                            mr_id,
                        )
                    except Exception as exc:  # pragma: no cover — diagnostic
                        logger.error(
                            "failed to insert ad-hoc gap ModuleRun for slug=%s: %s",
                            module_slug,
                            exc,
                        )
                        stats.skipped_modules[module_slug] += len(records)
                        stats.findings_skipped_unknown_module += len(records)
                        continue
                else:
                    reason = (
                        "gap-research without --neo4j-* creds"
                        if module_slug.startswith("gap-")
                        else "unknown slug (not in catalog seed)"
                    )
                    logger.warning(
                        "skipping %d finding(s) for module slug=%r — %s",
                        len(records),
                        module_slug,
                        reason,
                    )
                    stats.skipped_modules[module_slug] += len(records)
                    stats.findings_skipped_unknown_module += len(records)
                    continue

            # PATCH → running. We use the WALL CLOCK on the server side; the
            # script doesn't fabricate a timestamp because the data comes from
            # a historic batch, not a live execution.
            await api.update_module_run(run_id, module_run_id, {"status": "running"})

            # Bulk-write the findings in chunks.
            wrote = 0
            replayed = 0
            failed = 0
            for chunk_start in range(0, len(records), BULK_FINDING_CHUNK):
                chunk = records[chunk_start : chunk_start + BULK_FINDING_CHUNK]
                payload = [
                    _evidence_to_finding(
                        rec,
                        graph_id=cfg.graph_id,
                        run_id=run_id,
                        module_run_id=module_run_id,
                    )
                    for rec in chunk
                ]
                try:
                    bulk_resp = await api.record_findings_bulk(run_id, payload)
                except httpx.HTTPStatusError as exc:
                    logger.error(
                        "bulk findings POST failed for module=%s (chunk %d): %s",
                        module_slug,
                        chunk_start,
                        exc,
                    )
                    failed += len(chunk)
                    continue

                for res in bulk_resp.get("results", []):
                    if res.get("success"):
                        if res.get("already_existed"):
                            replayed += 1
                        else:
                            wrote += 1
                    else:
                        failed += 1
                        logger.warning(
                            "finding %s failed: %s", res.get("id"), res.get("error")
                        )
            stats.findings_written += wrote
            stats.findings_replayed += replayed
            stats.findings_failed += failed

            # PATCH → finished + evidence_count.
            await api.update_module_run(
                run_id,
                module_run_id,
                {"status": "finished", "evidence_count": wrote + replayed},
            )

        # ── 5. Stream conflicts.jsonl → POST one at a time.
        conflicts_path = cfg.run_dir / "evidence" / "conflicts.jsonl"
        if conflicts_path.is_file():
            for ln, record in _iter_jsonl_records(conflicts_path):
                if "__parse_error__" in record:
                    stats.jsonl_parse_errors += 1
                    continue
                if not isinstance(record, dict) or "id" not in record:
                    stats.conflicts_skipped_malformed += 1
                    logger.warning(
                        "conflicts.jsonl line %d: missing 'id', skipping", ln
                    )
                    continue
                stats.conflicts_total += 1
                payload = _conflict_to_payload(
                    record, graph_id=cfg.graph_id, run_id=run_id
                )
                try:
                    cresp = await api.record_conflict(run_id, payload)
                    if cresp.get("created"):
                        stats.conflicts_written += 1
                    else:
                        stats.conflicts_replayed += 1
                except httpx.HTTPStatusError as exc:
                    stats.conflicts_failed += 1
                    logger.error(
                        "conflict %s POST failed: %s", record.get("id"), exc
                    )

        # ── 6. Walk markdown deliverables (NN_/ddN_/gap-NN_) → POST each.
        files = _walk_module_md_files(cfg.run_dir)
        stats.deliverables_total = len(files)
        for ordinal, kind_hint, path in files:
            # Attach to a module_run when we can. For NN_ files we have no
            # reliable mapping (filename ordinal != module ordinal across
            # waves in this run), so we leave module_run_id=None and the
            # deliverable hangs off the run itself via HAS_DELIVERABLE.
            module_run_id = None
            if kind_hint == "gap":
                # gap-NN_X.md — if we successfully inserted a corresponding
                # gap-X module run, link to that; otherwise to no module_run.
                stem = path.stem  # e.g. "gap-01_b2b_segment"
                # match "gap-01-b2b" style slugs registered via direct insert.
                tail = stem.replace("_", "-")  # "gap-01-b2b-segment"
                # Try progressively shorter prefixes to find a registered slug.
                candidates: list[str] = []
                parts = tail.split("-")
                for i in range(len(parts), 1, -1):
                    candidates.append("-".join(parts[:i]))
                for c in candidates:
                    if c in slug_to_module_run_id:
                        module_run_id = slug_to_module_run_id[c]
                        break

            deliverable = _deliverable_from_md_file(
                path,
                graph_id=cfg.graph_id,
                run_id=run_id,
                module_run_id=module_run_id,
                ordinal=ordinal,
            )
            try:
                resp = await api.persist_deliverable(run_id, deliverable)
                if resp.get("created"):
                    stats.deliverables_written += 1
                else:
                    stats.deliverables_replayed += 1
            except httpx.HTTPStatusError as exc:
                stats.deliverables_failed += 1
                logger.error(
                    "deliverable POST failed for %s: %s", path.name, exc
                )

        # ── 7. final/ → bulk-final.
        final_files = _walk_final_doc_files(cfg.run_dir)
        stats.final_docs_total = len(final_files)
        if final_files:
            payload = [
                _final_doc_from_file(
                    p, graph_id=cfg.graph_id, run_id=run_id, ordinal=i
                )
                for i, p in enumerate(final_files)
            ]
            try:
                bulk_resp = await api.persist_final_docs(run_id, payload)
                for res in bulk_resp.get("results", []):
                    if res.get("success"):
                        # Count both new + replayed under written for the
                        # "the docs are present" summary.
                        stats.final_docs_written += 1
            except httpx.HTTPStatusError as exc:
                logger.error("final docs bulk POST failed: %s", exc)

        # ── 8. Module-run status rollup (for the summary print).
        if have_neo4j:
            own_driver = False
            driver = neo4j_driver
            if driver is None:
                from neo4j import AsyncGraphDatabase

                driver = AsyncGraphDatabase.driver(
                    cfg.neo4j_uri,
                    auth=(cfg.neo4j_user, cfg.neo4j_password),
                )
                own_driver = True
            try:
                rs = await driver.execute_query(
                    """
                    MATCH (r:AssessmentRun:__Platform__ {graph_id: $gid, run_id: $rid})
                          -[:HAS_MODULE_RUN]->(mr:ModuleRun:__Platform__)
                    RETURN mr.status AS status, count(*) AS cnt
                    """,
                    {"gid": cfg.graph_id, "rid": run_id},
                )
                for rec in rs.records:
                    stats.module_runs_by_status[rec["status"]] = rec["cnt"]
            finally:
                if own_driver:
                    await driver.close()

        # ── 9. Finalize.
        finalize_resp = await api.finalize_run(run_id)
        logger.info(
            "finalize_run: passed=%s status=%s direct=%d inferred=%d deliverables=%d",
            finalize_resp.get("passed"),
            finalize_resp.get("status"),
            finalize_resp.get("direct_finding_count"),
            finalize_resp.get("inferred_finding_count"),
            finalize_resp.get("deliverable_count"),
        )

    # Stash the run_id on the stats so callers (and tests) can pick it up.
    stats.__dict__["run_id"] = run_id  # type: ignore[attr-defined]
    return stats


# =============================================================================
# CLI
# =============================================================================


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="backfill_assessment_run",
        description=(
            "Backfill a completed /assess or /eurail-report run from disk "
            "into the assessment substrate via REST."
        ),
    )
    p.add_argument(
        "--run-dir",
        default=DEFAULT_RUN_DIR,
        help=f"Path to the run directory (default: {DEFAULT_RUN_DIR}).",
    )
    p.add_argument(
        "--graph-id",
        required=True,
        help="Tenant graph_id the run should land in (required).",
    )
    p.add_argument(
        "--template-slug",
        default=DEFAULT_TEMPLATE_SLUG,
        help=f"Assessment template slug (default: {DEFAULT_TEMPLATE_SLUG}).",
    )
    p.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help=f"REST API base URL (default: {DEFAULT_API_BASE}).",
    )
    p.add_argument(
        "--token",
        default=None,
        help="Service-account JWT. May be omitted if the API has auth disabled in dev.",
    )
    p.add_argument("--subject-slug", default="eurail")
    p.add_argument("--subject-name", default="Eurail B.V.")
    p.add_argument("--vertical-slug", default="rail-cooperative")
    p.add_argument("--neo4j-uri", default=None)
    p.add_argument("--neo4j-user", default=None)
    p.add_argument("--neo4j-password", default=None)
    p.add_argument("--catalog-graph-id", default="__assessments_catalog__")
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p.parse_args(argv)


def _format_summary(stats: BackfillStats) -> str:
    return (
        "\n=== Backfill summary ===\n"
        f"run_id:                              {getattr(stats, 'run_id', '?')}\n"
        f"findings_total:                      {stats.findings_total}\n"
        f"  written (new):                     {stats.findings_written}\n"
        f"  replayed (already existed):        {stats.findings_replayed}\n"
        f"  skipped_unknown_module:            {stats.findings_skipped_unknown_module}\n"
        f"  skipped_malformed:                 {stats.findings_skipped_malformed}\n"
        f"  failed:                            {stats.findings_failed}\n"
        f"conflicts_total:                     {stats.conflicts_total}\n"
        f"  written:                           {stats.conflicts_written}\n"
        f"  replayed:                          {stats.conflicts_replayed}\n"
        f"  skipped_malformed:                 {stats.conflicts_skipped_malformed}\n"
        f"  failed:                            {stats.conflicts_failed}\n"
        f"deliverables_total:                  {stats.deliverables_total}\n"
        f"  written:                           {stats.deliverables_written}\n"
        f"  replayed:                          {stats.deliverables_replayed}\n"
        f"  failed:                            {stats.deliverables_failed}\n"
        f"final_docs_total:                    {stats.final_docs_total}\n"
        f"  written:                           {stats.final_docs_written}\n"
        f"jsonl_parse_errors:                  {stats.jsonl_parse_errors}\n"
        f"module_runs_by_status:               {dict(stats.module_runs_by_status)}\n"
        f"skipped_modules (slug → count):      {dict(stats.skipped_modules)}\n"
    )


def _main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    cfg = BackfillConfig(
        run_dir=Path(args.run_dir),
        graph_id=args.graph_id,
        template_slug=args.template_slug,
        api_base=args.api_base,
        token=args.token,
        subject_slug=args.subject_slug,
        subject_name=args.subject_name,
        vertical_slug=args.vertical_slug,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        catalog_graph_id=args.catalog_graph_id,
    )
    try:
        stats = asyncio.run(run_backfill(cfg))
    except Exception as exc:  # pragma: no cover — top-level safety net
        logger.exception("backfill failed: %s", exc)
        return 2
    print(_format_summary(stats))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
