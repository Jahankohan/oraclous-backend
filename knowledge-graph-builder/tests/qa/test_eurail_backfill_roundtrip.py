"""QA validation — Eurail backfill round-trip (STORY-026, TASK-072).

This is the **SPRINT-001 exit gate** for the assessment substrate. The premise
of STORY-026 is that the existing Eurail 2026-05-06 run can be losslessly
reloaded through the new schema. If anything goes missing on the way through
— a finding ID, a conflict, a deliverable, a property — the schema is wrong
and Sprint 1 cannot close.

This test suite verifies the round-trip *independently* of TASK-071's own
tests. It writes against the live REST API (in-process ASGI), runs the
backfill against the real Eurail run dir, dumps the graph back to JSONL
via :mod:`app.scripts.dump_run_to_jsonl`, and asserts what was loaded
matches what was dumped — modulo a small, documented set of expected
differences (see :class:`TestEurailBackfillRoundtrip`'s docstring below).

Layout (per the task spec):

a. **Count-parity tests** against the source ``evidence/evidence.jsonl`` /
   ``conflicts.jsonl`` / on-disk markdown files.
b. **Round-trip dump test** — backfill, dump, diff.
c. **Property-level spot checks** on a random sample of 10 findings,
   5 conflicts, 5 deliverables.
d. The regression run is performed *outside* this file by re-running the
   full pytest suite; this file's job is the round-trip proof.

Run requirements:

- Real Neo4j (the dockerized one this repo's ``tests/conftest.py`` already
  uses; ``oraclous-data-studio-neo4j-1``).
- The real Eurail run dir at
  ``/Users/reza/workspace/Oraclous/docs/eurail/out/eurail-2026-05-06/``.
  If absent, the suite skips with a clear reason — *but does not silently
  pass*.

The QA fixture does NOT depend on TASK-070's catalog seed. We seed an
ad-hoc template that covers every module slug the real evidence file uses
(after legacy-alias normalization), so this file is hermetic w.r.t. TASK-070.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

try:
    from neo4j import AsyncDriver

    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False

from app.scripts.backfill_assessment_run import (
    BackfillConfig,
    _iter_jsonl_records,
    _normalize_module_slug,
    _resolve_record_id,
    _walk_module_md_files,
    run_backfill,
)
from app.scripts.dump_run_to_jsonl import dump_run_findings

# ============================================================================
# Constants — the Eurail 2026-05-06 baseline numbers, derived from the source
# ============================================================================

_REAL_EURAIL_RUN_DIR = Path(
    "/Users/reza/workspace/Oraclous/docs/eurail/out/eurail-2026-05-06"
)

#: Lines in evidence.jsonl. The QA reference is what `wc -l` returns.
EXPECTED_SOURCE_EVIDENCE_LINES = 600

#: After parsing (one row at line 247 is two concatenated JSON objects), the
#: source yields 601 records. The backfill MERGEs on natural id, so both end
#: up in the graph as distinct rows — they have distinct ids
#: (ev-cust-voice-056, ev-disruption-001) and belong to different modules.
EXPECTED_SOURCE_EVIDENCE_RECORDS = 601

#: Lines in conflicts.jsonl. 24 source rows; cf-breach-002 appears twice,
#: so the unique id set has 23 entries.
EXPECTED_SOURCE_CONFLICT_LINES = 24
EXPECTED_UNIQUE_CONFLICT_IDS = 23

#: Markdown deliverables at run root: 25 NN_* + 1 dd3_* + 5 gap-NN_* = 31.
EXPECTED_DELIVERABLE_FILES = 31

#: No `final/` subdir in this baseline run.
EXPECTED_FINAL_DOCS = 0


# ============================================================================
# Module-level pytest skip — Neo4j driver must be importable
# ============================================================================

pytestmark = pytest.mark.skipif(
    not _NEO4J_AVAILABLE, reason="neo4j driver not installed"
)


# ============================================================================
# Session state — one randomized session id keeps the fixture rows isolated
# from any parallel test that might be using __assessments_catalog__.
# ============================================================================

_SESSION = uuid.uuid4().hex[:8]
_TEMPLATE_ID = f"t-qa-roundtrip-{_SESSION}"
_TEMPLATE_SLUG = f"qa-roundtrip-{_SESSION}"
_TENANT_GID = f"qa-roundtrip-{_SESSION}"
_USER_ID = f"qa-roundtrip-user-{_SESSION}"
_CATALOG_GID = "__assessments_catalog__"


# ============================================================================
# Helpers — source-file inspection (read once per test; cheap on a 600-row
# file, so we don't cache module-wide).
# ============================================================================


def _read_source_evidence_ids() -> list[str]:
    """Return the ordered list of finding ids from the source file."""
    ids: list[str] = []
    path = _REAL_EURAIL_RUN_DIR / "evidence" / "evidence.jsonl"
    for _, rec in _iter_jsonl_records(path):
        if not isinstance(rec, dict):
            continue
        rid = _resolve_record_id(rec)
        if rid:
            ids.append(rid)
    return ids


def _read_source_evidence_by_id() -> dict[str, dict[str, Any]]:
    """Index source records by their natural id."""
    by_id: dict[str, dict[str, Any]] = {}
    path = _REAL_EURAIL_RUN_DIR / "evidence" / "evidence.jsonl"
    for _, rec in _iter_jsonl_records(path):
        if not isinstance(rec, dict):
            continue
        rid = _resolve_record_id(rec)
        if rid:
            by_id[rid] = rec
    return by_id


def _read_source_conflict_ids() -> list[str]:
    ids: list[str] = []
    path = _REAL_EURAIL_RUN_DIR / "evidence" / "conflicts.jsonl"
    for _, rec in _iter_jsonl_records(path):
        if not isinstance(rec, dict):
            continue
        cid = rec.get("id")
        if cid:
            ids.append(cid)
    return ids


def _read_source_conflicts_by_id() -> dict[str, dict[str, Any]]:
    """Index by id; on duplicate id, last write wins (matches MERGE semantics
    in the graph, where the second MERGE on the same id is a no-op since the
    node already exists with the first record's properties)."""
    by_id: dict[str, dict[str, Any]] = {}
    path = _REAL_EURAIL_RUN_DIR / "evidence" / "conflicts.jsonl"
    for _, rec in _iter_jsonl_records(path):
        if not isinstance(rec, dict):
            continue
        cid = rec.get("id")
        if cid and cid not in by_id:
            # Keep the FIRST occurrence — that's what MERGE preserves.
            by_id[cid] = rec
    return by_id


def _count_unique_source_urls() -> int:
    urls: set[str] = set()
    path = _REAL_EURAIL_RUN_DIR / "evidence" / "evidence.jsonl"
    for _, rec in _iter_jsonl_records(path):
        if not isinstance(rec, dict):
            continue
        src = rec.get("source") or {}
        if isinstance(src, dict):
            u = src.get("url") or src.get("url_normalized")
            if u:
                urls.add(u)
    return len(urls)


# ============================================================================
# Fixtures — Neo4j wipe + template seed + ASGI client
# ============================================================================


@pytest_asyncio.fixture
async def _wipe_qa_state(neo4j_test_driver: AsyncDriver):
    """Wipe everything the QA test owns, before and after."""

    async def _wipe():
        # Tenant graph
        await neo4j_test_driver.execute_query(
            "MATCH (n) WHERE n.graph_id = $gid DETACH DELETE n",
            {"gid": _TENANT_GID},
        )
        # The seed-test catalog rows we own (template + its modules)
        await neo4j_test_driver.execute_query(
            "MATCH (t:AssessmentTemplate {template_id: $tid}) "
            "OPTIONAL MATCH (t)-[:HAS_MODULE]->(m) DETACH DELETE t, m",
            {"tid": _TEMPLATE_ID},
        )
        # Any catalog :Module rows we may have inserted as gap-research
        # (their module_id starts with the test's template_id).
        await neo4j_test_driver.execute_query(
            "MATCH (m:Module) WHERE m.module_id STARTS WITH $tid DETACH DELETE m",
            {"tid": _TEMPLATE_ID},
        )

    await _wipe()
    yield
    await _wipe()


@pytest_asyncio.fixture
async def _seeded_template(neo4j_test_driver: AsyncDriver, _wipe_qa_state):
    """Seed an ad-hoc template that covers every module slug in the real run.

    We compute the slug set from the source file rather than hard-coding —
    if the run dir evolves (more modules, fewer modules), the seed adjusts.
    Gap-research slugs (``gap-*``) are NOT seeded; the backfill creates them
    on the fly via direct-Cypher (the documented gap-research path).
    """
    if not _REAL_EURAIL_RUN_DIR.is_dir():
        pytest.skip(f"Real Eurail run dir not present at {_REAL_EURAIL_RUN_DIR}")

    seen_non_gap: set[str] = set()
    for _, rec in _iter_jsonl_records(_REAL_EURAIL_RUN_DIR / "evidence" / "evidence.jsonl"):
        if not isinstance(rec, dict):
            continue
        slug = _normalize_module_slug(rec.get("module") or "")
        if slug and not slug.startswith("gap-"):
            seen_non_gap.add(slug)

    modules_payload = [
        {
            "module_id": f"{_TEMPLATE_ID}__{slug}",
            "slug": slug,
            "name": slug.replace("-", " ").title(),
            "wave": 1,
            "ordinal": i,
            "kind": "research",
        }
        for i, slug in enumerate(sorted(seen_non_gap))
    ]

    await neo4j_test_driver.execute_query(
        """
        MERGE (t:AssessmentTemplate:__Platform__ {template_id: $tid})
        ON CREATE SET t.graph_id = $catalog, t.slug = $slug,
                      t.name = 'QA Roundtrip Template', t.version = '0.0.1'
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
            "tid": _TEMPLATE_ID,
            "catalog": _CATALOG_GID,
            "slug": _TEMPLATE_SLUG,
            "modules": modules_payload,
        },
    )
    yield {"seeded_non_gap_slugs": seen_non_gap}


@pytest_asyncio.fixture
async def _asgi_client(neo4j_test_driver: AsyncDriver):
    """An in-process ASGI client wired to the FastAPI app + Neo4j driver.

    Mirrors TASK-071's integration test's client setup so the same auth
    overrides apply — the QA test exercises the REAL REST router code
    (not a mock), but skips the JWT plumbing.
    """
    from httpx import ASGITransport, AsyncClient

    from app.api.dependencies import get_current_user, get_current_user_id
    from app.api.v1.endpoints import assessments as assessments_mod
    from app.api.v1.endpoints.assessments import _assessment_service
    from app.main import app
    from app.services.assessment_service import AssessmentService

    def _principal() -> dict[str, str]:
        return {
            "id": _USER_ID,
            "principal_type": "service_account",
            "home_graph_id": _TENANT_GID,
        }

    app.dependency_overrides[get_current_user] = lambda: _principal()
    app.dependency_overrides[get_current_user_id] = lambda: _USER_ID
    app.dependency_overrides[_assessment_service] = lambda: AssessmentService(
        neo4j_test_driver
    )
    original_verify = assessments_mod.verify_graph_access
    assessments_mod.verify_graph_access = AsyncMock(return_value=_TENANT_GID)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac

    assessments_mod.verify_graph_access = original_verify
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(get_current_user_id, None)
    app.dependency_overrides.pop(_assessment_service, None)


@pytest_asyncio.fixture
async def _loaded_run(
    tmp_path: Path,
    neo4j_test_driver: AsyncDriver,
    _seeded_template,
    _asgi_client,
):
    """Run the backfill against a copy of the real Eurail run and yield the
    resulting (cfg, stats) tuple so each test can re-use the loaded state.

    Scoped per-test-function so each test gets a clean run — but the
    backfill is expensive (~600 finding writes), so within a single test
    we lean on parameterization to avoid re-loading.
    """
    if not _REAL_EURAIL_RUN_DIR.is_dir():
        pytest.skip(f"Real Eurail run dir not present at {_REAL_EURAIL_RUN_DIR}")

    # Copy the run dir so the backfill doesn't mutate the source-of-truth.
    run_copy = tmp_path / "eurail-run"
    shutil.copytree(_REAL_EURAIL_RUN_DIR, run_copy)

    neo4j_uri = os.getenv("TEST_NEO4J_URI", "neo4j://neo4j:7687")
    neo4j_user = os.getenv("TEST_NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("TEST_NEO4J_PASSWORD", "password")

    cfg = BackfillConfig(
        run_dir=run_copy,
        graph_id=_TENANT_GID,
        template_slug=_TEMPLATE_SLUG,
        # The backfill script's REST client appends `/api/v1/assessments`
        # to `--api-base`. The FastAPI app double-prefixes the routes
        # (outer `/api/v1` in `main.py` + inner `/api/v1` in
        # `api/v1/router.py` for the assessments subrouter) so the live
        # route is `/api/v1/api/v1/assessments/...`. Passing `http://test`
        # alone resolves to `/api/v1/assessments/...` and 404s — TASK-071's
        # own integration tests have this bug and never actually reached
        # the live router. We thread one `/api/v1` into the base so the
        # script's `_url_root` lands on the real route, mirroring
        # production CLI (`--api-base http://localhost:8000/api/v1`).
        api_base="http://test/api/v1",
        token=None,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
    )
    stats = await run_backfill(
        cfg, http_client=_asgi_client, neo4j_driver=neo4j_test_driver
    )
    yield cfg, stats


# ============================================================================
# (a) Count-parity tests
# ============================================================================


class TestCountParity:
    """The graph's counts must match the source JSONL / filesystem.

    Each assertion is independently meaningful: a failure points exactly
    at which axis of the round-trip lost data.
    """

    @pytest.mark.asyncio
    async def test_source_evidence_has_expected_lines(self):
        """Source baseline — `wc -l` says 600. Catches silent file change."""
        path = _REAL_EURAIL_RUN_DIR / "evidence" / "evidence.jsonl"
        if not path.is_file():
            pytest.skip(f"Real Eurail run dir absent at {_REAL_EURAIL_RUN_DIR}")
        with path.open() as f:
            n = sum(1 for _ in f)
        assert n == EXPECTED_SOURCE_EVIDENCE_LINES, (
            f"Source file changed: evidence.jsonl now has {n} lines, "
            f"expected {EXPECTED_SOURCE_EVIDENCE_LINES}"
        )

    @pytest.mark.asyncio
    async def test_source_evidence_yields_601_records_after_concat_decode(self):
        """The doubled-line at line 247 means 600 lines yield 601 records."""
        if not _REAL_EURAIL_RUN_DIR.is_dir():
            pytest.skip(f"Real Eurail run dir absent at {_REAL_EURAIL_RUN_DIR}")
        ids = _read_source_evidence_ids()
        assert len(ids) == EXPECTED_SOURCE_EVIDENCE_RECORDS, (
            f"Concat-decode yielded {len(ids)} records, "
            f"expected {EXPECTED_SOURCE_EVIDENCE_RECORDS}. "
            f"If this drifts the line-247 fix may have regressed."
        )

    @pytest.mark.asyncio
    async def test_source_conflicts_have_24_lines_23_unique_ids(self):
        """cf-breach-002 is duplicated in the source; MERGE will dedupe."""
        path = _REAL_EURAIL_RUN_DIR / "evidence" / "conflicts.jsonl"
        if not path.is_file():
            pytest.skip(f"Real Eurail run dir absent at {_REAL_EURAIL_RUN_DIR}")
        with path.open() as f:
            lines = sum(1 for _ in f)
        ids = _read_source_conflict_ids()
        assert lines == EXPECTED_SOURCE_CONFLICT_LINES, (
            f"Conflicts.jsonl line count drift: {lines} vs "
            f"{EXPECTED_SOURCE_CONFLICT_LINES}"
        )
        assert len(set(ids)) == EXPECTED_UNIQUE_CONFLICT_IDS, (
            f"Unique conflict ids drift: {len(set(ids))} vs "
            f"{EXPECTED_UNIQUE_CONFLICT_IDS}. Duplicates were "
            f"{[i for i, c in Counter(ids).items() if c > 1]}."
        )

    @pytest.mark.asyncio
    async def test_run_dir_has_31_deliverables_zero_final_docs(self):
        """25 numbered + 1 deep-dive + 5 gap = 31 module-md; no final/."""
        if not _REAL_EURAIL_RUN_DIR.is_dir():
            pytest.skip(f"Real Eurail run dir absent at {_REAL_EURAIL_RUN_DIR}")
        files = _walk_module_md_files(_REAL_EURAIL_RUN_DIR)
        kinds = Counter(k for _, k, _ in files)
        assert len(files) == EXPECTED_DELIVERABLE_FILES, (
            f"Deliverable file count drift: {len(files)} vs "
            f"{EXPECTED_DELIVERABLE_FILES}. Breakdown: {dict(kinds)}"
        )
        assert kinds["numbered"] == 25
        assert kinds["deep-dive"] == 1
        assert kinds["gap"] == 5
        assert not (_REAL_EURAIL_RUN_DIR / "final").is_dir(), (
            "Baseline assumed no `final/` dir; if one appears, update "
            "EXPECTED_FINAL_DOCS and rerun."
        )

    @pytest.mark.asyncio
    async def test_findings_count_in_graph_matches_source(
        self, neo4j_test_driver: AsyncDriver, _loaded_run
    ):
        """After backfill: graph Finding count == source records (601)."""
        cfg, stats = _loaded_run
        rs = await neo4j_test_driver.execute_query(
            "MATCH (f:Finding {graph_id: $gid, run_id: $rid}) RETURN count(f) AS n",
            {"gid": cfg.graph_id, "rid": getattr(stats, "run_id")},
        )
        n = rs.records[0]["n"]
        assert n == EXPECTED_SOURCE_EVIDENCE_RECORDS, (
            f"Graph has {n} :Finding rows for this run, expected "
            f"{EXPECTED_SOURCE_EVIDENCE_RECORDS}. "
            f"Stats: written={stats.findings_written} replayed={stats.findings_replayed} "
            f"skipped_unknown={stats.findings_skipped_unknown_module} "
            f"failed={stats.findings_failed}"
        )

    @pytest.mark.asyncio
    async def test_conflicts_count_in_graph_matches_unique_source(
        self, neo4j_test_driver: AsyncDriver, _loaded_run
    ):
        """After backfill: graph Conflict count == unique source ids (23)."""
        cfg, stats = _loaded_run
        rs = await neo4j_test_driver.execute_query(
            "MATCH (c:Conflict {graph_id: $gid, run_id: $rid}) "
            "RETURN count(c) AS n",
            {"gid": cfg.graph_id, "rid": getattr(stats, "run_id")},
        )
        n = rs.records[0]["n"]
        assert n == EXPECTED_UNIQUE_CONFLICT_IDS, (
            f"Graph has {n} :Conflict rows; expected "
            f"{EXPECTED_UNIQUE_CONFLICT_IDS} (24 source lines, 23 unique ids "
            f"with cf-breach-002 duplicated)"
        )

    @pytest.mark.asyncio
    async def test_deliverables_count_in_graph_matches_files(
        self, neo4j_test_driver: AsyncDriver, _loaded_run
    ):
        """After backfill: graph :Deliverable count == file count (31)."""
        cfg, stats = _loaded_run
        rs = await neo4j_test_driver.execute_query(
            """
            MATCH (r:AssessmentRun {graph_id: $gid, run_id: $rid})
                  -[:HAS_DELIVERABLE]->(d:Deliverable)
            RETURN count(d) AS n
            """,
            {"gid": cfg.graph_id, "rid": getattr(stats, "run_id")},
        )
        n = rs.records[0]["n"]
        assert n == EXPECTED_DELIVERABLE_FILES, (
            f"Graph has {n} :Deliverable rows; expected "
            f"{EXPECTED_DELIVERABLE_FILES} (25 numbered + 1 dd + 5 gap)"
        )

    @pytest.mark.asyncio
    async def test_source_catalog_deduped_against_source_urls(
        self, neo4j_test_driver: AsyncDriver, _loaded_run
    ):
        """The catalog `:Source` count equals unique source URLs in the file.

        The backfill MERGEs :Source rows by `source_id` (which is
        UUID5(NAMESPACE_URL, url) deterministically), so two findings citing
        the same URL share a single :Source row in the catalog graph.

        We assert the catalog has *at least* the expected unique-URL count
        (the catalog may have rows from other tests' previous runs; we
        cannot assume an empty catalog). The strict ID-set assertion is
        delegated to the property-level test below.
        """
        cfg, stats = _loaded_run
        expected_url_count = _count_unique_source_urls()
        run_id = getattr(stats, "run_id")
        rs = await neo4j_test_driver.execute_query(
            """
            MATCH (f:Finding {graph_id: $gid, run_id: $rid})-[:CITES]->(s:Source)
            RETURN count(DISTINCT s.source_id) AS n
            """,
            {"gid": cfg.graph_id, "rid": run_id},
        )
        n = rs.records[0]["n"]
        # The number of distinct :Source nodes cited by this run's findings
        # should equal the number of unique URLs in the source file. Any
        # ambient catalog rows from other tests are excluded by going
        # through the :CITES edge from this run's findings.
        assert n == expected_url_count, (
            f"Distinct cited :Source count is {n}, expected "
            f"{expected_url_count}. Some source urls didn't dedupe, or some "
            f"findings cited no source when they should have."
        )


# ============================================================================
# (b) Round-trip dump test
# ============================================================================


class TestRoundTripDump:
    """Backfill → dump → diff.

    What we assert: for every finding_id in the source, the dumped row
    matches the source on the load-bearing fields. The dumped form is NOT
    byte-identical to the source — these are the documented expected
    differences:

    * **Field ordering** — the dump emits keys sorted alphabetically; the
      source preserves insertion order. We normalize by parsing both as
      JSON and comparing dicts.
    * **Confidence type** — source: string ``"HIGH"|"MEDIUM"|"LOW"``;
      dump: float in [0, 1]. The mapping is one-way deterministic
      (CONFIDENCE_STR_TO_FLOAT). We compare the *mapped* value, not the
      raw form.
    * **Label normalization** — source: nullable string with ``None`` /
      ``"ASSUMPTION"`` outliers; dump: schema enum ``{DIRECT, INFERRED,
      CONTRADICTION}``. ``None`` → ``DIRECT``; ``ASSUMPTION`` →
      ``INFERRED``. We compare *normalized* values on both sides.
    * **Module aliasing** — source uses ``cust-journey`` for 34 rows;
      dump emits ``customer-journey``. The alias is in
      LEGACY_SLUG_ALIASES.
    * **Source block keys** — source includes ``publication_date`` and
      ``fetch_date`` directly on the source dict; dump preserves these
      via the catalog :Source columns. The committed assessment_service
      does not currently populate these catalog properties from the
      backfill (it MERGEs with bare ``source_id`` only), so the dump's
      source block is sparse — present as a key only when the catalog
      already had richer source rows from a prior process. We mark this
      as a known limitation and do not block on it (see the test below).
    * **ai_adoption_relevance vs notes** — the legacy file has a free-text
      ``ai_adoption_relevance`` string; the new schema's
      ``ai_adoption_relevance`` is a float, and the prose is preserved
      under ``notes``. The dump emits the prose back under ``notes``.
    * **The 5 gap-research rows** use ``gap_id`` in the source; the dump
      emits them under ``id`` (the schema's natural column). Compare via
      ``_resolve_record_id``.
    * **superseded_by** — only emitted when non-null.

    The test parses both ends, applies these documented normalizations,
    and compares the resulting dicts.
    """

    @pytest.mark.asyncio
    async def test_every_source_finding_id_round_trips(
        self,
        tmp_path: Path,
        neo4j_test_driver: AsyncDriver,
        _loaded_run,
    ):
        """The set of finding_ids in the dump == the set in the source."""
        cfg, stats = _loaded_run
        dump_path = tmp_path / "dump.jsonl"
        run_id = getattr(stats, "run_id")
        n_written = await dump_run_findings(
            graph_id=cfg.graph_id,
            run_id=run_id,
            output=dump_path,
            driver=neo4j_test_driver,
        )

        dumped_ids: set[str] = set()
        with dump_path.open() as f:
            for line in f:
                rec = json.loads(line)
                rid = rec.get("id") or rec.get("gap_id") or rec.get("finding_id")
                if rid:
                    dumped_ids.add(rid)

        source_ids = set(_read_source_evidence_ids())
        missing = source_ids - dumped_ids
        extra = dumped_ids - source_ids
        assert n_written == EXPECTED_SOURCE_EVIDENCE_RECORDS, (
            f"Dump wrote {n_written} rows; expected "
            f"{EXPECTED_SOURCE_EVIDENCE_RECORDS}"
        )
        assert not missing, (
            f"Lossy backfill: {len(missing)} finding_ids dropped on the "
            f"round-trip. First 10: {sorted(missing)[:10]}"
        )
        assert not extra, (
            f"Spurious findings in dump that aren't in source: "
            f"{sorted(extra)[:10]}"
        )

    @pytest.mark.asyncio
    async def test_dumped_load_bearing_fields_match_source(
        self,
        tmp_path: Path,
        neo4j_test_driver: AsyncDriver,
        _loaded_run,
    ):
        """For every source finding, claim+dimensions round-trip identically.

        These two fields are the closest to "pure data" in the schema —
        ``claim`` is the free-text fact statement; ``dimensions`` is a
        list of dimension slugs. Neither is transformed by the backfill.
        If they round-trip cleanly for all 601 records, the schema is
        carrying the load-bearing content losslessly.

        Confidence + label normalization is checked in a separate test
        that knows about the mapping.
        """
        from app.scripts.backfill_assessment_run import (
            _normalize_confidence,
            _normalize_label,
        )

        cfg, stats = _loaded_run
        dump_path = tmp_path / "dump.jsonl"
        run_id = getattr(stats, "run_id")
        await dump_run_findings(
            graph_id=cfg.graph_id,
            run_id=run_id,
            output=dump_path,
            driver=neo4j_test_driver,
        )

        dumped: dict[str, dict[str, Any]] = {}
        with dump_path.open() as f:
            for line in f:
                rec = json.loads(line)
                rid = rec.get("id")
                if rid:
                    dumped[rid] = rec

        source = _read_source_evidence_by_id()

        mismatches: list[str] = []
        for rid, src in source.items():
            dmp = dumped.get(rid)
            if dmp is None:
                mismatches.append(f"{rid}: missing from dump")
                continue
            # claim — gap-research rows fold "finding"/"missing" → claim.
            expected_claim = (
                src.get("claim")
                or src.get("finding")
                or src.get("missing")
                or ""
            )
            if (dmp.get("claim") or "") != expected_claim:
                mismatches.append(
                    f"{rid}: claim mismatch — src={expected_claim!r} "
                    f"dump={dmp.get('claim')!r}"
                )
                continue
            # dimensions — should match exactly (list of strings).
            src_dims = src.get("dimensions") or []
            dump_dims = dmp.get("dimensions") or []
            if list(src_dims) != list(dump_dims):
                mismatches.append(
                    f"{rid}: dimensions — src={src_dims} dump={dump_dims}"
                )
                continue
            # label — apply the same normalization to both sides.
            src_label = _normalize_label(src.get("label"))
            if dmp.get("label") != src_label:
                mismatches.append(
                    f"{rid}: label — src(norm)={src_label} dump={dmp.get('label')}"
                )
                continue
            # confidence — same normalization; allow tiny float epsilon.
            src_conf = _normalize_confidence(src.get("confidence"))
            dmp_conf = dmp.get("confidence")
            if dmp_conf is None or abs(dmp_conf - src_conf) > 1e-9:
                mismatches.append(
                    f"{rid}: confidence — src(norm)={src_conf} dump={dmp_conf}"
                )
                continue

        assert not mismatches, (
            f"{len(mismatches)} property mismatches detected. First 10:\n"
            + "\n".join(mismatches[:10])
        )

    @pytest.mark.asyncio
    async def test_dump_filename_is_diff_friendly_and_sorted(
        self,
        tmp_path: Path,
        neo4j_test_driver: AsyncDriver,
        _loaded_run,
    ):
        """The dump is sorted by finding_id; lines are independent.

        Sortedness is a property-of-the-dumper invariant. If it breaks,
        the byte-diff goal is lost and the round-trip story doesn't
        compose with future tooling (e.g., periodic backfill drift
        checks). Cheap to verify; high value.
        """
        cfg, stats = _loaded_run
        dump_path = tmp_path / "dump.jsonl"
        run_id = getattr(stats, "run_id")
        await dump_run_findings(
            graph_id=cfg.graph_id,
            run_id=run_id,
            output=dump_path,
            driver=neo4j_test_driver,
        )
        ids_in_order: list[str] = []
        with dump_path.open() as f:
            for line in f:
                rec = json.loads(line)
                rid = rec.get("id")
                if rid:
                    ids_in_order.append(rid)
        assert ids_in_order == sorted(ids_in_order), (
            "Dump is not sorted by finding_id ASC; "
            f"first 5 actual={ids_in_order[:5]} "
            f"first 5 sorted={sorted(ids_in_order)[:5]}"
        )


# ============================================================================
# (c) Property-level spot checks — random sampling
# ============================================================================

# Deterministic seed so a failure in CI is reproducible — and so successive
# runs of this test on the same data hit the same sample. Picks a 10/5/5
# subset per the task spec.
_RNG_SEED = 0xEU2A1L  # arbitrary mnemonic; doesn't matter as long as it's pinned


class TestPropertyLevelSpotChecks:
    """Deep-equal a sample of findings, conflicts, deliverables against source."""

    @pytest.mark.asyncio
    async def test_ten_random_findings_match_source(
        self, neo4j_test_driver: AsyncDriver, _loaded_run
    ):
        """Per the task spec: 10 random findings; deep equality on
        ``claim``, ``raw``, ``label``, ``confidence``, ``dimensions``,
        ``ai_adoption_relevance``, ``notes``."""
        from app.scripts.backfill_assessment_run import (
            _normalize_confidence,
            _normalize_label,
        )

        cfg, stats = _loaded_run
        run_id = getattr(stats, "run_id")

        source = _read_source_evidence_by_id()
        all_ids = sorted(source.keys())
        random.seed(_RNG_SEED)
        sample = random.sample(all_ids, 10)

        rs = await neo4j_test_driver.execute_query(
            """
            MATCH (f:Finding {graph_id: $gid, run_id: $rid})
            WHERE f.finding_id IN $ids
            RETURN
                f.finding_id            AS id,
                f.claim                 AS claim,
                f.raw                   AS raw,
                f.label                 AS label,
                f.confidence            AS confidence,
                f.dimensions            AS dimensions,
                f.ai_adoption_relevance AS ai_adoption_relevance,
                f.notes                 AS notes
            """,
            {"gid": cfg.graph_id, "rid": run_id, "ids": sample},
        )
        in_graph = {rec["id"]: dict(rec) for rec in rs.records}

        for rid in sample:
            assert rid in in_graph, f"finding {rid!r} missing from graph"
            src = source[rid]
            g = in_graph[rid]

            expected_claim = (
                src.get("claim")
                or src.get("finding")
                or src.get("missing")
                or ""
            )
            assert (g.get("claim") or "") == expected_claim, (
                f"{rid}: claim mismatch"
            )

            # raw — backfill sets to record.get('raw') or record.get('searched');
            # for canonical evidence records it's just 'raw'; for gap-research
            # the 'searched' folds in.
            expected_raw = src.get("raw") if src.get("raw") is not None else src.get("searched")
            assert g.get("raw") == expected_raw, (
                f"{rid}: raw mismatch — src={expected_raw!r} g={g.get('raw')!r}"
            )

            assert g.get("label") == _normalize_label(src.get("label")), (
                f"{rid}: label — src={src.get('label')} g={g.get('label')}"
            )
            assert abs(g.get("confidence", 0.0) - _normalize_confidence(src.get("confidence"))) < 1e-9, (
                f"{rid}: confidence — src={src.get('confidence')!r} "
                f"g={g.get('confidence')}"
            )
            assert list(g.get("dimensions") or []) == list(src.get("dimensions") or []), (
                f"{rid}: dimensions mismatch"
            )

            # ai_adoption_relevance — the legacy file's value is free-text prose;
            # the backfill stores it in `notes` and sets `ai_adoption_relevance`
            # (schema is float, not used yet) to None. Verify by checking notes.
            src_aar = src.get("ai_adoption_relevance")
            if isinstance(src_aar, str):
                # Legacy prose lands under `notes`.
                assert g.get("notes") == src_aar, (
                    f"{rid}: notes (folded ai_adoption_relevance prose) "
                    f"mismatch — src={src_aar!r} g={g.get('notes')!r}"
                )
                # Schema column should be None (we don't have a numeric value).
                assert g.get("ai_adoption_relevance") is None, (
                    f"{rid}: schema ai_adoption_relevance unexpectedly set"
                )
            else:
                # No prose → notes preserves the explicit notes field
                # (or None).
                assert g.get("notes") == src.get("notes"), (
                    f"{rid}: notes mismatch — src={src.get('notes')!r} "
                    f"g={g.get('notes')!r}"
                )

    @pytest.mark.asyncio
    async def test_five_random_conflicts_match_source(
        self, neo4j_test_driver: AsyncDriver, _loaded_run
    ):
        """5 random conflicts: topic, summary, status, involved finding ids."""
        cfg, stats = _loaded_run
        run_id = getattr(stats, "run_id")

        source = _read_source_conflicts_by_id()
        all_ids = sorted(source.keys())
        random.seed(_RNG_SEED + 1)
        sample = random.sample(all_ids, 5)

        rs = await neo4j_test_driver.execute_query(
            """
            MATCH (c:Conflict {graph_id: $gid, run_id: $rid})
            WHERE c.conflict_id IN $ids
            OPTIONAL MATCH (c)-[:INVOLVES]->(f:Finding)
            RETURN
                c.conflict_id      AS id,
                c.topic            AS topic,
                c.summary          AS summary,
                c.status           AS status,
                c.resolution       AS resolution,
                c.synthesis_note   AS synthesis_note,
                collect(DISTINCT f.finding_id) AS involved
            """,
            {"gid": cfg.graph_id, "rid": run_id, "ids": sample},
        )
        in_graph = {rec["id"]: dict(rec) for rec in rs.records}

        from app.scripts.backfill_assessment_run import _normalize_conflict_status

        for cid in sample:
            assert cid in in_graph, f"conflict {cid!r} missing from graph"
            src = source[cid]
            g = in_graph[cid]

            assert g["topic"] == src.get("topic", ""), (
                f"{cid}: topic — src={src.get('topic')!r} g={g['topic']!r}"
            )
            assert g["summary"] == src.get("summary", ""), (
                f"{cid}: summary mismatch"
            )
            assert g["status"] == _normalize_conflict_status(src.get("resolution")), (
                f"{cid}: status mismatch"
            )
            assert g["resolution"] == src.get("explanation"), (
                f"{cid}: resolution (legacy 'explanation') mismatch"
            )
            assert g["synthesis_note"] == src.get("synthesis_note"), (
                f"{cid}: synthesis_note mismatch"
            )
            # INVOLVES edges should reference every evidence_id from the
            # source — modulo evidence ids that weren't loaded (none expected
            # in a successful round-trip).
            expected_involved = set(src.get("evidence_ids") or [])
            graph_involved = set(i for i in (g["involved"] or []) if i is not None)
            missing_involved = expected_involved - graph_involved
            assert not missing_involved, (
                f"{cid}: missing INVOLVES edges to {missing_involved}"
            )

    @pytest.mark.asyncio
    async def test_five_random_deliverables_match_source_files(
        self, neo4j_test_driver: AsyncDriver, _loaded_run
    ):
        """5 random deliverables: content_inline or content_uri matches disk."""
        from app.scripts.backfill_assessment_run import INLINE_CONTENT_MAX_CHARS

        cfg, stats = _loaded_run
        run_id = getattr(stats, "run_id")
        files = _walk_module_md_files(_REAL_EURAIL_RUN_DIR)
        # Pick 5 distinct filenames across the three kinds.
        random.seed(_RNG_SEED + 2)
        sample = random.sample(files, 5)

        for ordinal, kind_hint, path in sample:
            rs = await neo4j_test_driver.execute_query(
                """
                MATCH (r:AssessmentRun {graph_id: $gid, run_id: $rid})
                      -[:HAS_DELIVERABLE]->(d:Deliverable {filename: $fn})
                RETURN
                    d.filename       AS filename,
                    d.kind           AS kind,
                    d.ordinal        AS ordinal,
                    d.content_uri    AS uri,
                    d.content_inline AS inline,
                    d.word_count     AS word_count
                """,
                {"gid": cfg.graph_id, "rid": run_id, "fn": path.name},
            )
            assert rs.records, f"deliverable {path.name!r} not in graph"
            d = dict(rs.records[0])

            disk = path.read_text(encoding="utf-8")
            assert d["filename"] == path.name
            assert d["ordinal"] == ordinal
            # The kind is always "module-md" for this run — `kind_hint` is
            # an internal walker classification not stored in the graph.
            assert d["kind"] == "module-md", (
                f"{path.name}: expected module-md kind, got {d['kind']!r}"
            )
            if len(disk) <= INLINE_CONTENT_MAX_CHARS:
                assert d["inline"] == disk, (
                    f"{path.name}: inline content drift "
                    f"(disk={len(disk)} chars, inline={len(d['inline'] or '')} chars)"
                )
            else:
                # Large file — uri set, inline None.
                assert d["inline"] is None
                assert d["uri"].endswith(path.name), (
                    f"{path.name}: content_uri does not point at the file"
                )


# ============================================================================
# Idempotency — a second backfill of the same source should not duplicate
# rows in the tenant graph.
# ============================================================================


class TestIdempotency:
    """Replay safety: re-running the backfill is a no-op for finding rows.

    The script generates a fresh `run_id` per call, so the tenant graph
    accumulates module_run rows across replays. But the :Finding MERGE
    keys are stable, and TASK-073's high-severity fix scopes them per
    `graph_id`. Replaying the SAME source against the SAME tenant should
    leave the finding count flat.
    """

    @pytest.mark.asyncio
    async def test_second_backfill_does_not_duplicate_findings(
        self,
        tmp_path: Path,
        neo4j_test_driver: AsyncDriver,
        _seeded_template,
        _asgi_client,
    ):
        """Re-run the backfill; the tenant's finding count stays at 601."""
        if not _REAL_EURAIL_RUN_DIR.is_dir():
            pytest.skip(f"Real Eurail run dir absent at {_REAL_EURAIL_RUN_DIR}")

        neo4j_uri = os.getenv("TEST_NEO4J_URI", "neo4j://neo4j:7687")
        neo4j_user = os.getenv("TEST_NEO4J_USERNAME", "neo4j")
        neo4j_password = os.getenv("TEST_NEO4J_PASSWORD", "password")

        run_copy = tmp_path / "eurail-run"
        shutil.copytree(_REAL_EURAIL_RUN_DIR, run_copy)
        cfg = BackfillConfig(
            run_dir=run_copy,
            graph_id=_TENANT_GID,
            template_slug=_TEMPLATE_SLUG,
            # See `_loaded_run` for why `/api/v1` is threaded into the base.
            api_base="http://test/api/v1",
            token=None,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
        )
        first = await run_backfill(
            cfg, http_client=_asgi_client, neo4j_driver=neo4j_test_driver
        )
        rs1 = await neo4j_test_driver.execute_query(
            "MATCH (f:Finding {graph_id: $gid}) RETURN count(f) AS n",
            {"gid": cfg.graph_id},
        )
        n1 = rs1.records[0]["n"]
        assert n1 == EXPECTED_SOURCE_EVIDENCE_RECORDS, (
            f"First pass: {n1} findings; expected {EXPECTED_SOURCE_EVIDENCE_RECORDS}"
        )

        # Second pass — same source, fresh run_id, but findings MERGE.
        second = await run_backfill(
            cfg, http_client=_asgi_client, neo4j_driver=neo4j_test_driver
        )
        rs2 = await neo4j_test_driver.execute_query(
            "MATCH (f:Finding {graph_id: $gid}) RETURN count(f) AS n",
            {"gid": cfg.graph_id},
        )
        n2 = rs2.records[0]["n"]
        assert n2 == n1, (
            f"Idempotency violation: tenant finding count {n1} → {n2} after "
            f"replay. First run_id={getattr(first, 'run_id')}, "
            f"second run_id={getattr(second, 'run_id')}."
        )
