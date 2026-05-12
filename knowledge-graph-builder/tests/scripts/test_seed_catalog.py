"""Tests for the assessment-catalog seed (STORY-026, TASK-070).

Two test layers:

* Unit tests on :mod:`app.scripts.seed_assessment_catalog` that exercise the
  static module inventory and the prompt-loading helpers against a tmp_path
  fixture. They do not require Neo4j.
* Integration tests that run the seed against the dockerized Neo4j instance
  (the same target as ``tests/integration/*``) and assert the expected node /
  edge counts from the task's Definition of Done.

The integration tests are skipped if Neo4j isn't reachable so the unit tests
remain runnable in any environment.
"""

from __future__ import annotations

import uuid
from collections import Counter
from pathlib import Path

import pytest
import pytest_asyncio

try:
    from neo4j import AsyncDriver

    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False

from app.scripts.seed_assessment_catalog import (
    ASSESS_V1_MODULES,
    CATALOG_GRAPH_ID,
    EURAIL_REPORT_V1_MODULES,
    SEED_CREATED_BY,
    TEMPLATES,
    ModuleSpec,
    _read_prompt,
    _skills_root,
    seed_assessment_catalog,
)

# ---------------------------------------------------------------------------
# Unit tests — module inventory shape
# ---------------------------------------------------------------------------


class TestModuleInventory:
    """The static module specs encode the task's Definition-of-Done counts."""

    def test_two_templates_seeded(self):
        ids = {t.template_id for t in TEMPLATES}
        assert ids == {"assess-v1", "eurail-report-v1"}

    def test_assess_v1_has_21_modules(self):
        """assess-v1 ships 21 module prompts: 11 research + 3 analysis +
        5 synthesis + 2 quality-gate.

        Larger than the task spec's bare "11 modules" which only referenced
        the research wave; documented in the task's Notes table.
        """
        assert len(ASSESS_V1_MODULES) == 21
        kinds = Counter(m.kind for m in ASSESS_V1_MODULES)
        assert kinds["research"] == 11
        assert kinds["analysis"] == 3
        assert kinds["synthesis"] == 5
        assert kinds["quality-gate"] == 2

    def test_eurail_report_v1_has_23_modules(self):
        """eurail-report-v1 ships 23 module prompts: 14 research + 3 analysis +
        4 synthesis + 2 quality-gate.

        Larger than the task spec's "14 modules" — same reasoning as above.
        """
        assert len(EURAIL_REPORT_V1_MODULES) == 23
        kinds = Counter(m.kind for m in EURAIL_REPORT_V1_MODULES)
        assert kinds["research"] == 14
        assert kinds["analysis"] == 3
        assert kinds["synthesis"] == 4
        assert kinds["quality-gate"] == 2

    def test_module_ids_are_globally_unique(self):
        """Module IDs are ``<template>__<slug>`` so they are unique across
        templates even when slugs overlap (e.g. ``company-intel`` exists in
        both templates)."""
        all_ids = [m.module_id for t in TEMPLATES for m in t.modules]
        assert len(all_ids) == len(set(all_ids))
        assert len(all_ids) == 44

    def test_overlapping_slugs_are_preserved_as_separate_rows(self):
        """The task spec requires modules with the same slug across templates
        to remain as separate :Module rows scoped by template_id."""
        assess_slugs = {m.slug for m in ASSESS_V1_MODULES}
        eurail_slugs = {m.slug for m in EURAIL_REPORT_V1_MODULES}
        shared = assess_slugs & eurail_slugs
        # Sanity check — there really is overlap, otherwise the dedup
        # decision in Notes is moot.
        assert "company-intel" in shared
        assert "customer-journey" in shared
        assert "adversarial-redline" in shared
        # Each shared slug appears once per template → 2 distinct module_ids
        for slug in shared:
            ids = [m.module_id for t in TEMPLATES for m in t.modules if m.slug == slug]
            assert len(ids) == 2, f"Slug {slug!r} did not produce 2 rows"
            assert len(set(ids)) == 2

    def test_every_module_has_required_metadata(self):
        for t in TEMPLATES:
            for m in t.modules:
                assert m.template_id == t.template_id
                assert m.slug
                assert m.name
                assert m.wave in (1, 2, 3, 4, 5)
                assert m.ordinal >= 1
                assert m.kind in (
                    "research",
                    "gap-research",
                    "analysis",
                    "synthesis",
                    "quality-gate",
                )
                assert m.deliverable_filename
                assert m.prompt_relpath.endswith(".md")


# ---------------------------------------------------------------------------
# Unit tests — prompt loading
# ---------------------------------------------------------------------------


class TestPromptLoading:
    def test_read_prompt_loads_file_content(self, tmp_path: Path):
        """The prompt loader reads the skill file content verbatim."""
        spec = ModuleSpec(
            template_id="assess-v1",
            slug="company-intel",
            name="Company Intel",
            wave=1,
            ordinal=1,
            kind="research",
            prompt_relpath="assess/modules/research/01-company-intel.md",
            deliverable_filename="01_company_intel.md",
            required_outputs=("deliverable_md", "evidence_jsonl"),
        )
        target = tmp_path / "assess" / "modules" / "research" / "01-company-intel.md"
        target.parent.mkdir(parents=True)
        target.write_text("# Mock prompt\n\nDo research.\n", encoding="utf-8")

        out = _read_prompt(spec, tmp_path)
        assert "Mock prompt" in out
        assert "Do research." in out

    def test_read_prompt_raises_on_missing_file(self, tmp_path: Path):
        spec = ModuleSpec(
            template_id="assess-v1",
            slug="company-intel",
            name="Company Intel",
            wave=1,
            ordinal=1,
            kind="research",
            prompt_relpath="assess/modules/research/01-company-intel.md",
            deliverable_filename="01_company_intel.md",
        )
        with pytest.raises(FileNotFoundError):
            _read_prompt(spec, tmp_path)

    def test_skills_root_honours_env_override(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("ORACLOUS_SKILLS_ROOT", str(tmp_path))
        assert _skills_root() == tmp_path

    def test_all_real_skill_files_resolve(self):
        """Sanity: every module spec points at an actual ``.md`` file under
        the live ``~/.claude/skills/`` tree.

        Skipped if the developer doesn't have the skills installed (CI does
        not — those runs should set ORACLOUS_SKILLS_ROOT to a fixture)."""
        root = _skills_root()
        if not root.is_dir():
            pytest.skip(f"Skills root {root} not present")
        for t in TEMPLATES:
            for m in t.modules:
                path = root / m.prompt_relpath
                assert path.is_file(), (
                    f"Missing skill file for {m.template_id}/{m.slug}: {path}"
                )


# ---------------------------------------------------------------------------
# Integration tests — real Neo4j
# ---------------------------------------------------------------------------


pytestmark_integration = pytest.mark.skipif(
    not _NEO4J_AVAILABLE, reason="neo4j driver not installed"
)


@pytest_asyncio.fixture
async def _seed_fixture(tmp_path: Path, monkeypatch, neo4j_test_driver):
    """Build a fixture skills tree mirroring the real layout, point the seed
    at it via env var, and ensure the catalog graph is clean before and after.

    Tests can call ``seed_assessment_catalog(driver)`` directly — the
    skills_root resolves from the env var the fixture sets.
    """
    # ── 1. Build a minimal skills tree with the exact paths each ModuleSpec
    #      expects. Content can be a single deterministic line per file —
    #      tests assert on persistence, not on the prompt body.
    for t in TEMPLATES:
        for m in t.modules:
            target = tmp_path / m.prompt_relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                f"# {m.template_id} :: {m.slug}\n\nFixture prompt body.\n",
                encoding="utf-8",
            )
    monkeypatch.setenv("ORACLOUS_SKILLS_ROOT", str(tmp_path))

    async def _wipe():
        # Delete only the catalog rows the seed writes. Other catalog content
        # (Registry items, the catalog graph anchor itself) stays intact.
        await neo4j_test_driver.execute_query(
            """
            MATCH (a:Agent:__Platform__ {graph_id: $gid})
            WHERE a.created_by = $created_by
            DETACH DELETE a
            """,
            {"gid": CATALOG_GRAPH_ID, "created_by": SEED_CREATED_BY},
        )
        await neo4j_test_driver.execute_query(
            """
            MATCH (m:Module:__Platform__ {graph_id: $gid}) DETACH DELETE m
            """,
            {"gid": CATALOG_GRAPH_ID},
        )
        await neo4j_test_driver.execute_query(
            """
            MATCH (t:AssessmentTemplate:__Platform__ {graph_id: $gid}) DETACH DELETE t
            """,
            {"gid": CATALOG_GRAPH_ID},
        )

    await _wipe()
    yield neo4j_test_driver
    await _wipe()


@pytestmark_integration
class TestSeedIntegration:
    """End-to-end: run the seed, verify counts, re-run, verify idempotency."""

    @pytest.mark.asyncio
    async def test_seed_creates_two_templates(self, _seed_fixture):
        counts = await seed_assessment_catalog(_seed_fixture)
        assert counts.templates == 2

    @pytest.mark.asyncio
    async def test_seed_creates_44_modules(self, _seed_fixture):
        counts = await seed_assessment_catalog(_seed_fixture)
        # 21 (assess-v1) + 23 (eurail-report-v1)
        assert counts.modules == 44

    @pytest.mark.asyncio
    async def test_seed_creates_44_agents(self, _seed_fixture):
        """One :Agent per :Module — no slug-level dedup because the prompts
        differ between templates (see Notes / Decisions Made on TASK-070)."""
        counts = await seed_assessment_catalog(_seed_fixture)
        assert counts.agents == 44

    @pytest.mark.asyncio
    async def test_seed_creates_44_has_module_edges(self, _seed_fixture):
        counts = await seed_assessment_catalog(_seed_fixture)
        assert counts.has_module_edges == 44

    @pytest.mark.asyncio
    async def test_seed_creates_44_executed_by_edges(self, _seed_fixture):
        counts = await seed_assessment_catalog(_seed_fixture)
        assert counts.executed_by_edges == 44

    @pytest.mark.asyncio
    async def test_eurail_template_has_23_modules_total(self, _seed_fixture):
        """The literal DoD wording ``MATCH (t {template_id: 'eurail-report-v1'})
        -[:HAS_MODULE]->(:Module) RETURN count(m)`` returns the **full** module
        catalog. The original spec text said "returns 14" but that only
        referenced the research wave (14 of 23 modules)."""
        await seed_assessment_catalog(_seed_fixture)
        res = await _seed_fixture.execute_query(
            """
            MATCH (t:AssessmentTemplate:__Platform__ {template_id: 'eurail-report-v1'})
                  -[:HAS_MODULE]->(m:Module:__Platform__)
            RETURN count(m) AS c
            """,
        )
        assert int(res.records[0]["c"]) == 23

    @pytest.mark.asyncio
    async def test_eurail_template_has_14_research_modules(self, _seed_fixture):
        """Explicit assertion against the original DoD line referring to the
        research-wave module count."""
        await seed_assessment_catalog(_seed_fixture)
        res = await _seed_fixture.execute_query(
            """
            MATCH (t:AssessmentTemplate:__Platform__ {template_id: 'eurail-report-v1'})
                  -[:HAS_MODULE]->(m:Module:__Platform__ {kind: 'research'})
            RETURN count(m) AS c
            """,
        )
        assert int(res.records[0]["c"]) == 14

    @pytest.mark.asyncio
    async def test_every_module_has_one_executor_agent(self, _seed_fixture):
        await seed_assessment_catalog(_seed_fixture)
        res = await _seed_fixture.execute_query(
            """
            MATCH (m:Module:__Platform__ {graph_id: $gid})
            OPTIONAL MATCH (m)-[r:EXECUTED_BY]->(:Agent:__Platform__)
            WITH m, count(r) AS n_edges
            RETURN min(n_edges) AS lo, max(n_edges) AS hi
            """,
            {"gid": CATALOG_GRAPH_ID},
        )
        row = res.records[0]
        assert int(row["lo"]) == 1
        assert int(row["hi"]) == 1

    @pytest.mark.asyncio
    async def test_every_catalog_node_carries_platform_marker(self, _seed_fixture):
        await seed_assessment_catalog(_seed_fixture)
        # Confirm every seeded node has both its primary label AND :__Platform__
        for primary in ("AssessmentTemplate", "Module", "Agent"):
            res = await _seed_fixture.execute_query(
                f"""
                MATCH (n:{primary} {{graph_id: $gid}})
                WHERE NOT n:__Platform__
                RETURN count(n) AS c
                """,
                {"gid": CATALOG_GRAPH_ID},
            )
            bad = int(res.records[0]["c"])
            assert bad == 0, f"{primary} rows missing :__Platform__: {bad}"

    @pytest.mark.asyncio
    async def test_seed_is_idempotent(self, _seed_fixture):
        """Re-running the seed must not duplicate nodes or edges."""
        first = await seed_assessment_catalog(_seed_fixture)
        second = await seed_assessment_catalog(_seed_fixture)
        assert first == second
        # Sanity — counts match the inventory after the second run too.
        assert second.templates == 2
        assert second.modules == 44
        assert second.agents == 44
        assert second.has_module_edges == 44
        assert second.executed_by_edges == 44

    @pytest.mark.asyncio
    async def test_seed_updates_system_prompt_on_rerun(self, _seed_fixture, tmp_path):
        """If a skill file is edited between runs, the :Module.system_prompt
        and the executor :Agent.system_prompt should both update — the seed
        doubles as a prompt-refresh tool."""
        await seed_assessment_catalog(_seed_fixture)

        # Pick one module, rewrite its fixture prompt, re-seed.
        target = ASSESS_V1_MODULES[0]
        new_text = f"# Updated prompt — {uuid.uuid4().hex}"
        path = tmp_path / target.prompt_relpath
        path.write_text(new_text, encoding="utf-8")

        await seed_assessment_catalog(_seed_fixture)

        res = await _seed_fixture.execute_query(
            """
            MATCH (m:Module:__Platform__ {module_id: $mid})
            OPTIONAL MATCH (m)-[:EXECUTED_BY]->(a:Agent:__Platform__)
            RETURN m.system_prompt AS module_prompt,
                   a.system_prompt AS agent_prompt
            """,
            {"mid": target.module_id},
        )
        row = res.records[0]
        assert row["module_prompt"] == new_text
        assert row["agent_prompt"] == new_text

    @pytest.mark.asyncio
    async def test_executor_agent_has_expected_shape(self, _seed_fixture):
        """Seeded :Agent rows look identical to ``AgentService.create_agent``
        output (same labels, same property keys) so the existing agent CRUD
        treats them as ordinary Graph-Native Agents."""
        await seed_assessment_catalog(_seed_fixture)
        sample = ASSESS_V1_MODULES[0]
        res = await _seed_fixture.execute_query(
            """
            MATCH (a:Agent:__Platform__ {agent_id: $aid})
            RETURN a.graph_id              AS graph_id,
                   a.reasoning_mode        AS reasoning_mode,
                   a.retriever_strategy    AS retriever_strategy,
                   a.retriever_hop_depth   AS retriever_hop_depth,
                   a.retriever_max_results AS retriever_max_results,
                   a.tools                 AS tools,
                   a.created_by            AS created_by,
                   a.deactivated_at        AS deactivated_at
            """,
            {"aid": sample.module_id},
        )
        row = res.records[0]
        assert row["graph_id"] == CATALOG_GRAPH_ID
        # Research-wave module → research reasoning mode.
        assert row["reasoning_mode"] == "research"
        assert row["retriever_strategy"] == "hybrid"
        assert int(row["retriever_hop_depth"]) == 2
        assert int(row["retriever_max_results"]) == 20
        # tools is stored as a JSON string (matches AgentService.create_agent)
        assert row["tools"] == '["graph_search"]'
        assert row["created_by"] == SEED_CREATED_BY
        assert row["deactivated_at"] is None
