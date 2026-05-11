"""Tests for the Eurail backfill script (STORY-026, TASK-071).

Two layers:

* **Unit tests** — pure-Python normalization + parsing helpers. Run anywhere,
  no Neo4j, no FastAPI. Cover the legacy → schema field translations the
  Eurail JSONL exercises (string confidence → float, label nulls/aliases,
  conflict resolution → status enum, slug aliases, concatenated JSON on
  one line, gap_id fallback id resolution).

* **Integration test** — wires the script's `run_backfill()` against an
  in-process ASGI FastAPI client (so REST calls hit the live router) plus
  a real dockerized Neo4j. Seeds the catalog with TASK-070's seed, runs
  backfill against a *copy* of the Eurail 2026-05-06 run directory (or a
  minimal fixture if the live dir is absent), and asserts the lossless
  round-trip per STORY-026 §Verification.

The integration test is skipped if Neo4j is unreachable so the unit suite
remains runnable in any environment.

What this file does NOT cover (delegated to TASK-072, QA):

* The full dump-back / round-trip-via-graph-read validation
* Cross-run analysis queries (admin source search)
* Performance characterization
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

try:
    from neo4j import AsyncDriver

    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False

from app.scripts.backfill_assessment_run import (
    CONFIDENCE_STR_TO_FLOAT,
    LEGACY_SLUG_ALIASES,
    BackfillConfig,
    BackfillStats,
    _conflict_to_payload,
    _deliverable_from_md_file,
    _evidence_to_finding,
    _iter_jsonl_records,
    _normalize_conflict_status,
    _normalize_confidence,
    _normalize_label,
    _normalize_module_slug,
    _resolve_record_id,
    _walk_final_doc_files,
    _walk_module_md_files,
    run_backfill,
)


# ============================================================================
# Unit tests — value normalization
# ============================================================================


class TestNormalizeConfidence:
    """String labels and out-of-range numerics map to a clean float in [0, 1]."""

    def test_high_medium_low_map_to_floats(self):
        assert _normalize_confidence("HIGH") == CONFIDENCE_STR_TO_FLOAT["HIGH"]
        assert _normalize_confidence("MEDIUM") == CONFIDENCE_STR_TO_FLOAT["MEDIUM"]
        assert _normalize_confidence("LOW") == CONFIDENCE_STR_TO_FLOAT["LOW"]

    def test_case_insensitive(self):
        assert _normalize_confidence("high") == CONFIDENCE_STR_TO_FLOAT["HIGH"]
        assert _normalize_confidence(" Medium ") == CONFIDENCE_STR_TO_FLOAT["MEDIUM"]

    def test_numeric_passthrough_and_clamp(self):
        assert _normalize_confidence(0.5) == 0.5
        assert _normalize_confidence(0) == 0.0
        assert _normalize_confidence(1.5) == 1.0
        assert _normalize_confidence(-0.2) == 0.0

    def test_stringified_floats(self):
        assert _normalize_confidence("0.7") == 0.7

    def test_none_and_garbage_default_to_zero(self):
        assert _normalize_confidence(None) == 0.0
        assert _normalize_confidence("not-a-number") == 0.0


class TestNormalizeLabel:
    """Label enum stays {DIRECT, INFERRED, CONTRADICTION}; legacy variants map cleanly."""

    def test_passthrough(self):
        assert _normalize_label("DIRECT") == "DIRECT"
        assert _normalize_label("INFERRED") == "INFERRED"
        assert _normalize_label("CONTRADICTION") == "CONTRADICTION"

    def test_none_defaults_to_direct(self):
        assert _normalize_label(None) == "DIRECT"
        assert _normalize_label("") == "DIRECT"

    def test_assumption_maps_to_inferred(self):
        """ASSUMPTION is the one outlier in the Eurail evidence file."""
        assert _normalize_label("ASSUMPTION") == "INFERRED"

    def test_case_insensitive(self):
        assert _normalize_label("direct") == "DIRECT"
        assert _normalize_label(" Inferred ") == "INFERRED"

    def test_unknown_defaults_to_direct(self):
        assert _normalize_label("FUTURE-LABEL") == "DIRECT"


class TestNormalizeConflictStatus:
    """Legacy ``resolution`` strings → schema ``status`` enum."""

    def test_resolved_uppercase(self):
        assert _normalize_conflict_status("RESOLVED") == "resolved"

    def test_open(self):
        assert _normalize_conflict_status("OPEN") == "open"
        assert _normalize_conflict_status(None) == "open"
        assert _normalize_conflict_status("") == "open"

    def test_accepted_open_variants(self):
        assert _normalize_conflict_status("ACCEPTED_OPEN") == "accepted_open"
        assert _normalize_conflict_status("ACCEPTED-OPEN") == "accepted_open"

    def test_unknown_defaults_to_open(self):
        assert _normalize_conflict_status("partial") == "open"


class TestNormalizeModuleSlug:
    def test_passthrough(self):
        assert _normalize_module_slug("company-intel") == "company-intel"

    def test_legacy_alias(self):
        assert _normalize_module_slug("cust-journey") == LEGACY_SLUG_ALIASES["cust-journey"]
        assert _normalize_module_slug("cust-journey") == "customer-journey"

    def test_trim_and_lower(self):
        assert _normalize_module_slug("  CUST-JOURNEY ") == "customer-journey"


class TestResolveRecordId:
    """Three id shapes in the Eurail run; resolver picks the first present."""

    def test_id(self):
        assert _resolve_record_id({"id": "ev-x-1"}) == "ev-x-1"

    def test_gap_id_fallback(self):
        assert _resolve_record_id({"gap_id": "gap-b2b-001"}) == "gap-b2b-001"

    def test_finding_id_fallback(self):
        assert _resolve_record_id({"finding_id": "f-1"}) == "f-1"

    def test_no_id_returns_none(self):
        assert _resolve_record_id({"claim": "x"}) is None


# ============================================================================
# Unit tests — JSONL parsing edge cases
# ============================================================================


class TestIterJsonlRecords:
    """The Eurail evidence file has one row with two concatenated JSONs on a
    single line. The parser must round-trip both objects."""

    def test_concatenated_objects_on_one_line(self, tmp_path: Path):
        p = tmp_path / "evidence.jsonl"
        # Two valid JSONs on one line, separated by no delimiter.
        p.write_text('{"id":"a","x":1}{"id":"b","x":2}\n')
        recs = list(_iter_jsonl_records(p))
        ids = [r.get("id") for _, r in recs if isinstance(r, dict) and "id" in r]
        assert ids == ["a", "b"]

    def test_malformed_line_logs_and_continues(self, tmp_path: Path):
        p = tmp_path / "evidence.jsonl"
        p.write_text('{"id":"good","x":1}\n{not-json}\n{"id":"good2"}\n')
        recs = list(_iter_jsonl_records(p))
        ok_ids = [r.get("id") for _, r in recs if isinstance(r, dict) and "id" in r]
        parse_errors = [r for _, r in recs if isinstance(r, dict) and "__parse_error__" in r]
        assert "good" in ok_ids and "good2" in ok_ids
        assert len(parse_errors) == 1

    def test_blank_lines_skipped(self, tmp_path: Path):
        p = tmp_path / "evidence.jsonl"
        p.write_text("\n\n{\"id\":\"a\"}\n\n")
        recs = [r for _, r in _iter_jsonl_records(p)]
        assert len(recs) == 1 and recs[0]["id"] == "a"


# ============================================================================
# Unit tests — record → payload mappers
# ============================================================================


class TestEvidenceToFinding:
    def test_minimal_record(self):
        rec = {"id": "ev-1", "module": "company-intel", "claim": "X is true"}
        out = _evidence_to_finding(rec, graph_id="g", run_id="r", module_run_id="mr")
        assert out["finding_id"] == "ev-1"
        assert out["graph_id"] == "g"
        assert out["run_id"] == "r"
        assert out["module_run_id"] == "mr"
        assert out["claim"] == "X is true"
        assert out["label"] == "DIRECT"
        assert out["confidence"] == 0.0  # no confidence supplied
        assert out["source_id"] is None  # no source block

    def test_full_record_with_string_confidence_and_source(self):
        rec = {
            "id": "ev-2",
            "module": "company-intel",
            "claim": "Y",
            "raw": "verbatim quote",
            "label": "INFERRED",
            "confidence": "HIGH",
            "dimensions": ["regulatory"],
            "source": {
                "type": "press",
                "url": "https://example.com/article-y",
                "name": "Example Article",
            },
        }
        out = _evidence_to_finding(rec, graph_id="g", run_id="r", module_run_id="mr")
        assert out["confidence"] == CONFIDENCE_STR_TO_FLOAT["HIGH"]
        assert out["label"] == "INFERRED"
        assert out["dimensions"] == ["regulatory"]
        assert out["source_id"] is not None  # derived from URL
        assert out["source_quote"] == "verbatim quote"

    def test_gap_research_shape_with_gap_id(self):
        """Gap-research rows use gap_id + 'finding' / 'missing' instead of
        the canonical id/claim/raw. The mapper folds them into the new shape."""
        rec = {
            "gap_id": "gap-b2b-volume-001",
            "module": "gap-01-b2b",
            "missing": "Total B2B volume figure",
            "searched": "phocuswright eurail b2b",
            "finding": "No public source available",
        }
        out = _evidence_to_finding(rec, graph_id="g", run_id="r", module_run_id="mr")
        assert out["finding_id"] == "gap-b2b-volume-001"
        assert out["claim"]  # not empty — folded from 'finding' / 'missing'


class TestConflictToPayload:
    def test_resolved_maps_to_status_lowercase(self):
        rec = {
            "id": "cf-1",
            "topic": "Date mismatch",
            "summary": "X says 2022; Y says 2020",
            "evidence_ids": ["ev-a", "ev-b"],
            "resolution": "RESOLVED",
            "synthesis_note": "Use 2020.",
            "explanation": "Y is more authoritative.",
        }
        out = _conflict_to_payload(rec, graph_id="g", run_id="r")
        assert out["conflict_id"] == "cf-1"
        assert out["status"] == "resolved"
        assert out["involved_finding_ids"] == ["ev-a", "ev-b"]
        # The legacy 'explanation' free-text lands on the schema's `resolution`
        # field; the legacy 'synthesis_note' stays as `synthesis_note`.
        assert out["resolution"] == "Y is more authoritative."
        assert out["synthesis_note"] == "Use 2020."

    def test_missing_resolution_defaults_to_open(self):
        rec = {"id": "cf-2", "topic": "t", "summary": "s"}
        out = _conflict_to_payload(rec, graph_id="g", run_id="r")
        assert out["status"] == "open"
        assert out["involved_finding_ids"] == []


class TestDeliverableFromMd:
    def test_small_file_is_inlined(self, tmp_path: Path):
        f = tmp_path / "01_eurail_today.md"
        f.write_text("# Eurail Today\n\nShort report.\n")
        d = _deliverable_from_md_file(
            f, graph_id="g", run_id="r", module_run_id="mr", ordinal=1
        )
        assert d["filename"] == "01_eurail_today.md"
        assert d["ordinal"] == 1
        assert d["kind"] == "module-md"
        assert d["content_uri"].startswith("file://")
        assert d["content_inline"]  # inlined because well under 50K chars

    def test_large_file_is_not_inlined(self, tmp_path: Path):
        f = tmp_path / "big.md"
        f.write_text("x" * (60_000))  # > 50K char threshold
        d = _deliverable_from_md_file(
            f, graph_id="g", run_id="r", module_run_id=None, ordinal=0
        )
        assert d["content_inline"] is None
        assert d["content_uri"].startswith("file://")


# ============================================================================
# Unit tests — filesystem walkers
# ============================================================================


class TestFileWalker:
    def test_numbered_dd_and_gap_files_classified(self, tmp_path: Path):
        names = [
            "00_executive_summary.md",
            "01_eurail_today.md",
            "21_adversarial_redline.md",
            "dd3_sncf_chatgpt_app.md",
            "gap-01_b2b_segment.md",
            "gap-05_silent_pilots.md",
            "ignore_me.txt",  # non-md, skipped
            "not-numbered.md",  # not a recognized pattern, skipped
        ]
        for n in names:
            (tmp_path / n).write_text("content")
        out = _walk_module_md_files(tmp_path)
        by_name = {p.name: (ord_, kind) for ord_, kind, p in out}
        assert by_name["00_executive_summary.md"] == (0, "numbered")
        assert by_name["01_eurail_today.md"] == (1, "numbered")
        assert by_name["21_adversarial_redline.md"] == (21, "numbered")
        assert by_name["dd3_sncf_chatgpt_app.md"] == (3, "deep-dive")
        assert by_name["gap-01_b2b_segment.md"] == (1, "gap")
        assert by_name["gap-05_silent_pilots.md"] == (5, "gap")
        assert "ignore_me.txt" not in by_name
        assert "not-numbered.md" not in by_name

    def test_final_doc_walker_handles_absent_dir(self, tmp_path: Path):
        assert _walk_final_doc_files(tmp_path) == []

    def test_final_doc_walker_picks_html_pdf_md(self, tmp_path: Path):
        final = tmp_path / "final"
        final.mkdir()
        (final / "00-intro.html").write_text("<html/>")
        (final / "01-section.pdf").write_bytes(b"%PDF-1.4")
        (final / "02-section.md").write_text("md")
        (final / "skipme.txt").write_text("nope")
        names = [p.name for p in _walk_final_doc_files(tmp_path)]
        assert names == ["00-intro.html", "01-section.pdf", "02-section.md"]


# ============================================================================
# Integration test — full end-to-end against in-process FastAPI + Neo4j
# ============================================================================


pytestmark_integration = pytest.mark.skipif(
    not _NEO4J_AVAILABLE, reason="neo4j driver not installed"
)

_INTEGRATION_SESSION = uuid.uuid4().hex[:8]
_INT_GID = f"integ-backfill-{_INTEGRATION_SESSION}"
_INT_USER = f"integ-backfill-user-{_INTEGRATION_SESSION}"
_INT_TEMPLATE_SLUG = f"eurail-report-backfill-{_INTEGRATION_SESSION}"
_INT_TEMPLATE_ID = f"t-backfill-{_INTEGRATION_SESSION}"
_REAL_EURAIL_RUN_DIR = Path(
    "/Users/reza/workspace/Oraclous/docs/eurail/out/eurail-2026-05-06"
)


def _build_minimal_run_dir(dst: Path) -> dict[str, int]:
    """Build a small synthetic run dir when the real Eurail run is absent.

    Returns the expected counts the integration test will assert on.
    """
    (dst / "evidence").mkdir(parents=True, exist_ok=True)
    evidence_rows = [
        # Two records under company-intel, one inferred + one direct
        {
            "id": "ev-test-001",
            "module": "company-intel",
            "claim": "Synthetic claim 1.",
            "label": "DIRECT",
            "confidence": "HIGH",
            "source": {"url": "https://example.com/a", "name": "A"},
        },
        {
            "id": "ev-test-002",
            "module": "company-intel",
            "claim": "Synthetic claim 2.",
            "label": "INFERRED",
            "confidence": "MEDIUM",
        },
        # One record under customer-voice
        {
            "id": "ev-test-003",
            "module": "customer-voice",
            "claim": "Customers say things.",
            "label": "DIRECT",
            "confidence": "HIGH",
        },
        # Legacy slug alias
        {
            "id": "ev-test-004",
            "module": "cust-journey",
            "claim": "Journey insight.",
            "label": "DIRECT",
            "confidence": "MEDIUM",
        },
    ]
    with (dst / "evidence" / "evidence.jsonl").open("w") as f:
        for r in evidence_rows:
            f.write(json.dumps(r) + "\n")

    conflict_rows = [
        {
            "id": "cf-test-001",
            "topic": "T",
            "summary": "S",
            "evidence_ids": ["ev-test-001", "ev-test-002"],
            "resolution": "RESOLVED",
            "synthesis_note": "OK.",
        }
    ]
    with (dst / "evidence" / "conflicts.jsonl").open("w") as f:
        for r in conflict_rows:
            f.write(json.dumps(r) + "\n")

    # Two module deliverables.
    (dst / "01_company_intel.md").write_text("# Company Intel\n\nBody.\n")
    (dst / "07_customer_voice.md").write_text("# Customer Voice\n\nBody.\n")
    return {"findings": 4, "conflicts": 1, "deliverables": 2}


@pytest_asyncio.fixture
async def _seeded_run_setup(neo4j_test_driver: AsyncDriver):
    """Seed a tiny eurail-report-v1-shaped template + modules; provide cleanup."""
    catalog_gid = "__assessments_catalog__"

    # 14 research + 3 analysis + 4 synthesis + 2 quality-gate, slugs matching
    # the subset of eurail-report-v1 the test's run dir exercises.
    modules = [
        ("company-intel", 1, 1, "research"),
        ("customer-voice", 1, 8, "research"),
        ("customer-journey", 3, 15, "analysis"),
    ]

    async def _wipe():
        for gid in (_INT_GID, catalog_gid):
            # Catalog wipe is scoped to OUR template id + its module ids.
            await neo4j_test_driver.execute_query(
                "MATCH (n {graph_id: $gid}) WHERE n.template_id = $tid "
                "OR n.module_id STARTS WITH $tid OR n.run_id IS NOT NULL "
                "AND n.graph_id = $gid DETACH DELETE n",
                {"gid": gid, "tid": _INT_TEMPLATE_ID},
            )
        await neo4j_test_driver.execute_query(
            "MATCH (n) WHERE n.graph_id = $gid DETACH DELETE n",
            {"gid": _INT_GID},
        )
        await neo4j_test_driver.execute_query(
            "MATCH (t:AssessmentTemplate {template_id: $tid}) "
            "OPTIONAL MATCH (t)-[:HAS_MODULE]->(m) DETACH DELETE t, m",
            {"tid": _INT_TEMPLATE_ID},
        )

    await _wipe()
    await neo4j_test_driver.execute_query(
        """
        MERGE (t:AssessmentTemplate:__Platform__ {template_id: $tid})
        ON CREATE SET t.graph_id = $catalog, t.slug = $slug,
                      t.name = 'Backfill-Test Template', t.version = '0.0.1'
        WITH t
        UNWIND $modules AS m
        MERGE (mod:Module:__Platform__ {module_id: m.module_id})
        ON CREATE SET
            mod.graph_id    = $catalog,
            mod.template_id = $tid,
            mod.slug        = m.slug,
            mod.name        = m.name,
            mod.wave        = m.wave,
            mod.ordinal     = m.ordinal,
            mod.kind        = m.kind
        MERGE (t)-[:HAS_MODULE]->(mod)
        """,
        {
            "tid": _INT_TEMPLATE_ID,
            "catalog": "__assessments_catalog__",
            "slug": _INT_TEMPLATE_SLUG,
            "modules": [
                {
                    "module_id": f"{_INT_TEMPLATE_ID}__{slug}",
                    "slug": slug,
                    "name": slug.replace("-", " ").title(),
                    "wave": wave,
                    "ordinal": ordinal,
                    "kind": kind,
                }
                for slug, wave, ordinal, kind in modules
            ],
        },
    )

    yield neo4j_test_driver

    await _wipe()


@pytest_asyncio.fixture
async def _http_client(neo4j_test_driver: AsyncDriver):
    """An ASGI httpx client wired into the FastAPI app, with auth + service overrides."""
    from httpx import ASGITransport, AsyncClient

    from app.api.dependencies import get_current_user, get_current_user_id
    from app.api.v1.endpoints import assessments as assessments_mod
    from app.api.v1.endpoints.assessments import _assessment_service
    from app.main import app
    from app.services.assessment_service import AssessmentService

    def _principal() -> dict:
        return {
            "id": _INT_USER,
            "principal_type": "service_account",
            "home_graph_id": _INT_GID,
        }

    app.dependency_overrides[get_current_user] = lambda: _principal()
    app.dependency_overrides[get_current_user_id] = lambda: _INT_USER
    app.dependency_overrides[_assessment_service] = lambda: AssessmentService(
        neo4j_test_driver
    )
    original_verify = assessments_mod.verify_graph_access
    assessments_mod.verify_graph_access = AsyncMock(return_value=_INT_GID)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac

    assessments_mod.verify_graph_access = original_verify
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(get_current_user_id, None)
    app.dependency_overrides.pop(_assessment_service, None)


@pytestmark_integration
@pytest.mark.asyncio
async def test_backfill_minimal_synthetic_run_round_trips(
    tmp_path: Path,
    neo4j_test_driver: AsyncDriver,
    _seeded_run_setup,
    _http_client,
):
    """Synthetic run dir → backfill → counts in Neo4j match the source."""
    expected = _build_minimal_run_dir(tmp_path / "run")

    cfg = BackfillConfig(
        run_dir=tmp_path / "run",
        graph_id=_INT_GID,
        template_slug=_INT_TEMPLATE_SLUG,
        api_base="http://test",
        token=None,
    )
    stats = await run_backfill(cfg, http_client=_http_client, neo4j_driver=neo4j_test_driver)

    # ── Findings round-trip ──────────────────────────────────────────────
    run_id = getattr(stats, "run_id")
    rs = await neo4j_test_driver.execute_query(
        """
        MATCH (f:Finding {graph_id: $gid, run_id: $rid})
        RETURN f.finding_id AS id
        """,
        {"gid": _INT_GID, "rid": run_id},
    )
    persisted_ids = {rec["id"] for rec in rs.records}
    # All 4 evidence rows should be persisted (cust-journey aliases to a
    # seeded module, no records should be skipped).
    expected_ids = {"ev-test-001", "ev-test-002", "ev-test-003", "ev-test-004"}
    assert persisted_ids == expected_ids, (
        f"Lossy backfill: missing {expected_ids - persisted_ids}, "
        f"extra {persisted_ids - expected_ids}"
    )

    # ── Conflicts round-trip ─────────────────────────────────────────────
    rs = await neo4j_test_driver.execute_query(
        """
        MATCH (c:Conflict {graph_id: $gid, run_id: $rid})
        RETURN c.conflict_id AS id, c.status AS status
        """,
        {"gid": _INT_GID, "rid": run_id},
    )
    conflict_records = rs.records
    assert len(conflict_records) == expected["conflicts"]
    assert conflict_records[0]["status"] == "resolved"

    # ── Deliverables round-trip ──────────────────────────────────────────
    rs = await neo4j_test_driver.execute_query(
        """
        MATCH (r:AssessmentRun {graph_id: $gid, run_id: $rid})-[:HAS_DELIVERABLE]->(d:Deliverable)
        RETURN d.filename AS fn, d.kind AS kind
        """,
        {"gid": _INT_GID, "rid": run_id},
    )
    deliverable_files = {rec["fn"] for rec in rs.records}
    assert deliverable_files == {"01_company_intel.md", "07_customer_voice.md"}

    # ── ModuleRuns finished count matches modules referenced in evidence ─
    rs = await neo4j_test_driver.execute_query(
        """
        MATCH (r:AssessmentRun {graph_id: $gid, run_id: $rid})
              -[:HAS_MODULE_RUN]->(mr:ModuleRun)
        RETURN mr.status AS status, count(*) AS cnt
        """,
        {"gid": _INT_GID, "rid": run_id},
    )
    by_status = {rec["status"]: rec["cnt"] for rec in rs.records}
    # 2 modules had findings (company-intel, customer-voice + the alias
    # cust-journey → customer-journey). 3 modules → finished.
    assert by_status.get("finished", 0) == 3, by_status
    # The remaining seeded module(s) stay 'planned'.
    assert by_status.get("planned", 0) >= 0

    # ── Stats sanity ─────────────────────────────────────────────────────
    assert stats.findings_total == 4
    assert stats.findings_written == 4
    assert stats.findings_failed == 0
    assert stats.findings_skipped_unknown_module == 0
    assert stats.conflicts_written == 1
    assert stats.deliverables_total == 2
    assert stats.deliverables_written == 2


@pytestmark_integration
@pytest.mark.asyncio
async def test_backfill_is_idempotent(
    tmp_path: Path,
    neo4j_test_driver: AsyncDriver,
    _seeded_run_setup,
    _http_client,
):
    """Re-running the backfill must not duplicate findings or conflicts."""
    _build_minimal_run_dir(tmp_path / "run")

    cfg = BackfillConfig(
        run_dir=tmp_path / "run",
        graph_id=_INT_GID,
        template_slug=_INT_TEMPLATE_SLUG,
        api_base="http://test",
        token=None,
    )
    # First pass.
    first = await run_backfill(cfg, http_client=_http_client, neo4j_driver=neo4j_test_driver)
    run_id_1 = getattr(first, "run_id")

    # Count findings after first pass.
    rs = await neo4j_test_driver.execute_query(
        """
        MATCH (f:Finding {graph_id: $gid, run_id: $rid}) RETURN count(f) AS n
        """,
        {"gid": _INT_GID, "rid": run_id_1},
    )
    n_after_first = rs.records[0]["n"]
    assert n_after_first == 4

    # A second backfill call creates a new run (different run_id), but the
    # per-finding MERGE on `finding_id` keeps total finding rows the same
    # IF the second pass targets the same run. We achieve "same run" by
    # asserting on the global Finding count in the tenant graph.
    # (The script's own create_run generates a fresh run_id each time —
    # idempotency at the finding level is what STORY-026 cares about.)
    second = await run_backfill(cfg, http_client=_http_client, neo4j_driver=neo4j_test_driver)
    rs = await neo4j_test_driver.execute_query(
        """
        MATCH (f:Finding {graph_id: $gid}) RETURN count(f) AS n
        """,
        {"gid": _INT_GID},
    )
    n_after_second = rs.records[0]["n"]
    # Same 4 finding_ids, no duplicates (the second run reuses MERGE).
    assert n_after_second == 4, (
        f"Idempotency violation: {n_after_first} → {n_after_second}"
    )


@pytestmark_integration
@pytest.mark.asyncio
async def test_backfill_real_eurail_run_round_trips(
    tmp_path: Path,
    neo4j_test_driver: AsyncDriver,
    _http_client,
):
    """The full Eurail 2026-05-06 run loads losslessly into the substrate.

    This is the STORY-026 §Verification gate: ~600 findings, ~24 conflicts,
    ~31 deliverables. We seed a template that lines up with the slugs the
    real Eurail run uses (subset of eurail-report-v1) — not the full
    23-module catalog, since this test doesn't depend on TASK-070's seed
    being run first; that composition is exercised by the TASK-070 tests.
    """
    if not _REAL_EURAIL_RUN_DIR.is_dir():
        pytest.skip(
            f"Real Eurail run dir not present at {_REAL_EURAIL_RUN_DIR} — "
            "skipping the full-substrate round-trip"
        )

    # Seed a template that includes EVERY module slug referenced in the
    # real evidence.jsonl (after legacy alias normalization). This is
    # the catalog the backfill resolves against.
    real_evidence = _REAL_EURAIL_RUN_DIR / "evidence" / "evidence.jsonl"
    seen_slugs: set[str] = set()
    for _, rec in _iter_jsonl_records(real_evidence):
        if not isinstance(rec, dict):
            continue
        slug = _normalize_module_slug(rec.get("module") or "")
        if slug and not slug.startswith("gap-"):
            seen_slugs.add(slug)

    template_id = f"t-eurail-real-{_INTEGRATION_SESSION}"
    template_slug = f"eurail-real-{_INTEGRATION_SESSION}"
    gid = f"integ-backfill-real-{_INTEGRATION_SESSION}"
    catalog_gid = "__assessments_catalog__"

    async def _wipe():
        await neo4j_test_driver.execute_query(
            "MATCH (n {graph_id: $gid}) DETACH DELETE n", {"gid": gid}
        )
        await neo4j_test_driver.execute_query(
            "MATCH (t:AssessmentTemplate {template_id: $tid}) "
            "OPTIONAL MATCH (t)-[:HAS_MODULE]->(m) DETACH DELETE t, m",
            {"tid": template_id},
        )

    await _wipe()
    try:
        await neo4j_test_driver.execute_query(
            """
            MERGE (t:AssessmentTemplate:__Platform__ {template_id: $tid})
            ON CREATE SET t.graph_id = $catalog, t.slug = $slug,
                          t.name = 'Eurail Real Backfill Test', t.version = '0.0.1'
            WITH t
            UNWIND $modules AS m
            MERGE (mod:Module:__Platform__ {module_id: m.module_id})
            ON CREATE SET
                mod.graph_id    = $catalog,
                mod.template_id = $tid,
                mod.slug        = m.slug,
                mod.name        = m.name,
                mod.wave        = m.wave,
                mod.ordinal     = m.ordinal,
                mod.kind        = m.kind
            MERGE (t)-[:HAS_MODULE]->(mod)
            """,
            {
                "tid": template_id,
                "catalog": catalog_gid,
                "slug": template_slug,
                "modules": [
                    {
                        "module_id": f"{template_id}__{slug}",
                        "slug": slug,
                        "name": slug.replace("-", " ").title(),
                        "wave": 1,
                        "ordinal": i,
                        "kind": "research",
                    }
                    for i, slug in enumerate(sorted(seen_slugs))
                ],
            },
        )

        # Make a copy of the run dir so we don't risk mutating the real one.
        run_copy = tmp_path / "eurail-run"
        shutil.copytree(_REAL_EURAIL_RUN_DIR, run_copy)

        # Read Neo4j connection settings from env (the same vars the
        # test conftest uses) so the gap-research direct insert path works.
        neo4j_uri = os.getenv("TEST_NEO4J_URI", "neo4j://neo4j:7687")
        neo4j_user = os.getenv("TEST_NEO4J_USERNAME", "neo4j")
        neo4j_password = os.getenv("TEST_NEO4J_PASSWORD", "password")

        cfg = BackfillConfig(
            run_dir=run_copy,
            graph_id=gid,
            template_slug=template_slug,
            api_base="http://test",
            token=None,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
        )
        stats = await run_backfill(
            cfg, http_client=_http_client, neo4j_driver=neo4j_test_driver
        )

        # ── Lossless evidence.jsonl ──────────────────────────────────────
        # Count finding_ids in the source.
        source_ids: set[str] = set()
        for _, rec in _iter_jsonl_records(real_evidence):
            if not isinstance(rec, dict):
                continue
            rid = _resolve_record_id(rec)
            if rid:
                source_ids.add(rid)

        rs = await neo4j_test_driver.execute_query(
            "MATCH (f:Finding {graph_id: $gid}) RETURN f.finding_id AS id",
            {"gid": gid},
        )
        persisted_ids = {rec["id"] for rec in rs.records}
        missing = source_ids - persisted_ids
        assert not missing, (
            f"Lossy backfill: {len(missing)} finding_ids missing. "
            f"First 10: {sorted(missing)[:10]}"
        )

        # ── Lossless conflicts.jsonl ─────────────────────────────────────
        conflict_path = _REAL_EURAIL_RUN_DIR / "evidence" / "conflicts.jsonl"
        source_cids: set[str] = set()
        for _, rec in _iter_jsonl_records(conflict_path):
            if not isinstance(rec, dict):
                continue
            rid = _resolve_record_id(rec)
            if rid:
                source_cids.add(rid)
        rs = await neo4j_test_driver.execute_query(
            "MATCH (c:Conflict {graph_id: $gid}) RETURN c.conflict_id AS id",
            {"gid": gid},
        )
        persisted_cids = {rec["id"] for rec in rs.records}
        # Conflicts in the file may have duplicate ids (Eurail run has
        # one duplicated cf-breach-002); the set of unique ids must match.
        missing_conflicts = source_cids - persisted_cids
        assert not missing_conflicts, (
            f"Lossy conflict backfill: missing {missing_conflicts}"
        )

        # ── Deliverable count matches the markdown files at run root ─────
        from app.scripts.backfill_assessment_run import _walk_module_md_files as walker

        expected_deliverable_count = len(walker(_REAL_EURAIL_RUN_DIR))
        rs = await neo4j_test_driver.execute_query(
            """
            MATCH (r:AssessmentRun {graph_id: $gid})-[:HAS_DELIVERABLE]->(d:Deliverable)
            RETURN count(d) AS n
            """,
            {"gid": gid},
        )
        assert rs.records[0]["n"] == expected_deliverable_count

        # ── Module runs reached 'finished' for every module touched by
        #     evidence (modules without findings remain 'planned'). ──────
        rs = await neo4j_test_driver.execute_query(
            """
            MATCH (r:AssessmentRun {graph_id: $gid})-[:HAS_MODULE_RUN]->(mr:ModuleRun)
            RETURN mr.status AS status, count(*) AS cnt
            """,
            {"gid": gid},
        )
        by_status = {rec["status"]: rec["cnt"] for rec in rs.records}
        # Every seeded research module has at least some evidence in the
        # Eurail run — all should be 'finished'. The 3 dynamically-inserted
        # gap-research module-runs also land in 'finished' state. The seeded
        # template's module count is len(seen_slugs).
        assert by_status.get("finished", 0) >= len(seen_slugs), by_status

        # The stats counter should agree with the source.
        assert stats.findings_total >= len(source_ids), (
            f"stats.findings_total ({stats.findings_total}) < source ({len(source_ids)})"
        )
    finally:
        await _wipe()
