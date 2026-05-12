#!/usr/bin/env python
"""
Seed the assessment catalog (STORY-026, TASK-070).

Populates the catalog graph ``__assessments_catalog__`` with the two assessment
templates Oraclous currently ships:

* ``assess-v1`` — the vertical-agnostic 6-wave assessment pipeline (21 modules)
* ``eurail-report-v1`` — the hardcoded Eurail variant (23 modules)

For every module across both templates the script:

1. ``MERGE``\\ s a ``:Module:__Platform__`` row carrying ``module_id``,
   ``template_id``, ``slug``, ``name``, ``wave``, ``ordinal``, ``kind``,
   ``deliverable_filename``, ``required_outputs`` and the **full system prompt**
   read at seed time from the corresponding ``.md`` file under
   ``~/.claude/skills/<skill>/modules/<wave>/``.
2. Creates the ``(:AssessmentTemplate)-[:HAS_MODULE]->(:Module)`` edge.
3. ``MERGE``\\ s a matching ``:Agent:__Platform__`` row in the same catalog
   graph using a deterministic ``agent_id`` (``<template>__<module-slug>``) so
   re-running the seed is idempotent. The node shape (labels and property keys)
   matches what :class:`app.services.agent_service.AgentService.create_agent`
   writes — so agent CRUD, the executor, and the API treat these as ordinary
   Graph-Native Agents.
4. Creates the ``(:Module)-[:EXECUTED_BY]->(:Agent)`` edge.

All ``MERGE``\\ es use natural IDs and ``ON CREATE`` / ``ON MATCH`` clauses so
re-running the seed is a no-op for unchanged rows and a content update when a
prompt file has been edited.

Usage
-----

As a module (the canonical entry point)::

    python -m app.scripts.seed_assessment_catalog

Programmatically (e.g., from a test or future bootstrap routine)::

    from app.scripts.seed_assessment_catalog import seed_assessment_catalog

    counts = await seed_assessment_catalog(neo4j_client.async_driver)

The function returns a :class:`SeedCounts` dataclass with the per-entity totals
the tests assert on.

Dependencies
------------

This script assumes the assessment-schema migration (TASK-067) has been applied
— constraints on ``:AssessmentTemplate.template_id``, ``:Module.module_id``
and the ``__assessments_catalog__`` graph anchor must exist. If they do not,
the ``MERGE``\\ es still work but uniqueness is not guaranteed; the standalone
``_main`` entrypoint therefore applies the schema migration first.

The script does **not** seed:

- ``:Subject`` rows (created per-run by ``create_run``)
- Registry items (STORY-028 work — separate catalog graph ``__registry__``)
- Any tenant-graph content
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATALOG_GRAPH_ID = "__assessments_catalog__"

#: System user that authored the platform-curated catalog rows.
SEED_CREATED_BY = "system:assessment-catalog-seed"

#: Default skills root. Overridable via ``ORACLOUS_SKILLS_ROOT`` so tests can
#: point at a fixture tree without touching the real ``~/.claude/skills/``.
DEFAULT_SKILLS_ROOT = Path.home() / ".claude" / "skills"


ModuleKind = Literal[
    "research", "gap-research", "analysis", "synthesis", "quality-gate"
]
ReasoningMode = Literal["direct", "research", "analytical", "conversational"]


# ---------------------------------------------------------------------------
# Module inventory — single source of truth for what the seed should create.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModuleSpec:
    """Static metadata for one catalog module.

    The ``system_prompt`` is **not** stored here — it's read from the skill's
    ``.md`` file at seed time so the skill file remains the source of truth.
    """

    template_id: str
    slug: str
    name: str
    wave: int
    ordinal: int
    kind: ModuleKind
    #: Path of the module prompt relative to its skill root, e.g.
    #: ``"assess/modules/research/01-company-intel.md"``.
    prompt_relpath: str
    #: Filename of the deliverable this module's executor writes,
    #: e.g. ``"01_company_intel.md"``. Used by the orchestrator (Claude Code)
    #: when fanning out subagents.
    deliverable_filename: str
    #: Other outputs the executor must produce besides the deliverable —
    #: typically ``["evidence_jsonl"]`` for research modules.
    required_outputs: tuple[str, ...] = ()

    @property
    def module_id(self) -> str:
        """Deterministic identifier — natural key for ``MERGE``."""
        return f"{self.template_id}__{self.slug}"


# Map module kind → reasoning mode. Research and gap-research modules query
# the web and need the research executor; analysis/synthesis are reasoning-only;
# quality gates audit prior outputs and use the analytical executor.
_REASONING_FOR_KIND: dict[ModuleKind, ReasoningMode] = {
    "research": "research",
    "gap-research": "research",
    "analysis": "analytical",
    "synthesis": "analytical",
    "quality-gate": "analytical",
}


def _required_outputs_for_kind(kind: ModuleKind) -> tuple[str, ...]:
    """Default `required_outputs` from module kind.

    Research modules emit both a markdown deliverable and a JSONL evidence file.
    Other kinds emit only the deliverable.
    """
    if kind in ("research", "gap-research"):
        return ("deliverable_md", "evidence_jsonl")
    return ("deliverable_md",)


def _r(
    template: str, slug: str, name: str, ordinal: int, deliverable: str
) -> ModuleSpec:
    """Build a research-wave spec (wave=1, kind=research)."""
    skill_root = "assess" if template == "assess-v1" else "eurail-report"
    return ModuleSpec(
        template_id=template,
        slug=slug,
        name=name,
        wave=1,
        ordinal=ordinal,
        kind="research",
        prompt_relpath=f"{skill_root}/modules/research/{ordinal:02d}-{slug}.md",
        deliverable_filename=deliverable,
        required_outputs=_required_outputs_for_kind("research"),
    )


def _a(
    template: str, slug: str, name: str, ordinal: int, deliverable: str
) -> ModuleSpec:
    """Build an analysis-wave spec (wave=3, kind=analysis)."""
    skill_root = "assess" if template == "assess-v1" else "eurail-report"
    return ModuleSpec(
        template_id=template,
        slug=slug,
        name=name,
        wave=3,
        ordinal=ordinal,
        kind="analysis",
        prompt_relpath=f"{skill_root}/modules/analysis/{ordinal:02d}-{slug}.md",
        deliverable_filename=deliverable,
        required_outputs=_required_outputs_for_kind("analysis"),
    )


def _s(
    template: str, slug: str, name: str, ordinal: int, deliverable: str
) -> ModuleSpec:
    """Build a synthesis-wave spec (wave=4, kind=synthesis)."""
    skill_root = "assess" if template == "assess-v1" else "eurail-report"
    return ModuleSpec(
        template_id=template,
        slug=slug,
        name=name,
        wave=4,
        ordinal=ordinal,
        kind="synthesis",
        prompt_relpath=f"{skill_root}/modules/synthesis/{ordinal:02d}-{slug}.md",
        deliverable_filename=deliverable,
        required_outputs=_required_outputs_for_kind("synthesis"),
    )


def _g(
    template: str, slug: str, name: str, ordinal: int, deliverable: str
) -> ModuleSpec:
    """Build a quality-gate-wave spec (wave=5, kind=quality-gate)."""
    skill_root = "assess" if template == "assess-v1" else "eurail-report"
    return ModuleSpec(
        template_id=template,
        slug=slug,
        name=name,
        wave=5,
        ordinal=ordinal,
        kind="quality-gate",
        prompt_relpath=f"{skill_root}/modules/gates/{ordinal:02d}-{slug}.md",
        deliverable_filename=deliverable,
        required_outputs=_required_outputs_for_kind("quality-gate"),
    )


# ---------------------------------------------------------------------------
# assess-v1 — 21 modules
# ---------------------------------------------------------------------------

ASSESS_V1_MODULES: tuple[ModuleSpec, ...] = (
    # Wave 1 — Research (11 modules, ordinals 01-11)
    _r("assess-v1", "company-intel", "Company Intel", 1, "01_company_intel.md"),
    _r(
        "assess-v1",
        "tech-stack-forensics",
        "Tech Stack Forensics",
        2,
        "02_tech_stack_forensics.md",
    ),
    _r("assess-v1", "industry-market", "Industry & Market", 3, "03_industry_market.md"),
    _r(
        "assess-v1",
        "competitor-ai-benchmark",
        "Competitor AI Benchmark",
        4,
        "04_competitor_ai_benchmark.md",
    ),
    _r(
        "assess-v1",
        "mcp-agent-surface",
        "MCP Agent Surface",
        5,
        "05_mcp_agent_surface.md",
    ),
    _r(
        "assess-v1",
        "third-party-surrogates",
        "Third-Party Surrogates",
        6,
        "06_third_party_surrogates.md",
    ),
    _r("assess-v1", "customer-voice", "Customer Voice", 7, "07_customer_voice.md"),
    _r(
        "assess-v1",
        "distribution-disruption",
        "Distribution Disruption",
        8,
        "08_distribution_disruption.md",
    ),
    _r(
        "assess-v1",
        "regulatory-landscape",
        "Regulatory Landscape",
        9,
        "09_regulatory_landscape.md",
    ),
    _r("assess-v1", "geo-aeo-state", "GEO/AEO State", 10, "10_geo_aeo_state.md"),
    _r(
        "assess-v1",
        "breach-incident-history",
        "Breach & Incident History",
        11,
        "11_breach_incident_history.md",
    ),
    # Wave 3 — Analysis (3 modules, ordinals 12-14)
    _a(
        "assess-v1",
        "customer-journey",
        "Customer Journey",
        12,
        "12_customer_journey.md",
    ),
    _a(
        "assess-v1",
        "ai-maturity-scorecard",
        "AI Maturity Scorecard",
        13,
        "13_ai_maturity_scorecard.md",
    ),
    _a(
        "assess-v1",
        "structural-advantage",
        "Structural Advantage",
        14,
        "14_structural_advantage.md",
    ),
    # Wave 4 — Synthesis (5 modules, ordinals 15-19)
    _s(
        "assess-v1",
        "adoption-scenario",
        "Adoption Scenario",
        15,
        "15_adoption_scenario.md",
    ),
    _s(
        "assess-v1",
        "inaction-scenario",
        "Inaction Scenario",
        16,
        "16_inaction_scenario.md",
    ),
    _s(
        "assess-v1",
        "opportunities",
        "Opportunities (Four-Pass Scorer)",
        17,
        "17_opportunities.md",
    ),
    _s(
        "assess-v1",
        "vendor-fit-roadmap",
        "Vendor Fit Roadmap",
        18,
        "18_vendor_fit_roadmap.md",
    ),
    _s("assess-v1", "three-asks", "Three Asks", 19, "19_three_asks.md"),
    # Wave 5 — Quality Gates (2 modules, ordinals 20-21)
    _g(
        "assess-v1",
        "adversarial-redline",
        "Adversarial Redline",
        20,
        "20_adversarial_redline.md",
    ),
    _g("assess-v1", "report-editor", "Report Editor", 21, "00_executive_summary.md"),
)


# ---------------------------------------------------------------------------
# eurail-report-v1 — 23 modules
# ---------------------------------------------------------------------------

EURAIL_REPORT_V1_MODULES: tuple[ModuleSpec, ...] = (
    # Wave 1 — Research (14 modules, ordinals 01-14)
    _r("eurail-report-v1", "company-intel", "Company Intel", 1, "01_eurail_today.md"),
    _r(
        "eurail-report-v1",
        "cooperative-governance",
        "Cooperative Governance",
        2,
        "02_cooperative_governance.md",
    ),
    _r(
        "eurail-report-v1",
        "tech-stack-forensics",
        "Tech Stack Forensics",
        3,
        "01_tech_stack_appendix.md",
    ),
    _r(
        "eurail-report-v1",
        "industry-market",
        "Industry & Market",
        4,
        "03_industry_position.md",
    ),
    _r(
        "eurail-report-v1",
        "competitor-ai-benchmark",
        "Competitor AI Benchmark",
        5,
        "04_competitor_ai_benchmark.md",
    ),
    _r("eurail-report-v1", "mcp-ecosystem", "MCP Ecosystem", 6, "05_mcp_ecosystem.md"),
    _r(
        "eurail-report-v1",
        "third-party-surrogates",
        "Third-Party Surrogates",
        7,
        "06_third_party_surrogates.md",
    ),
    _r(
        "eurail-report-v1",
        "customer-voice",
        "Customer Voice",
        8,
        "07_customer_voice.md",
    ),
    _r(
        "eurail-report-v1",
        "disruption-resilience",
        "Disruption Resilience",
        9,
        "08_disruption_resilience.md",
    ),
    _r(
        "eurail-report-v1",
        "breach-aftermath",
        "Breach Aftermath",
        10,
        "09_breach_aftermath.md",
    ),
    _r(
        "eurail-report-v1",
        "trip-planner-sunset",
        "Trip Planner Sunset",
        11,
        "10_trip_planner_sunset.md",
    ),
    _r(
        "eurail-report-v1",
        "distribution-disruption",
        "Distribution Disruption",
        12,
        "11_distribution_disruption.md",
    ),
    _r(
        "eurail-report-v1",
        "regulatory-landscape",
        "Regulatory Landscape",
        13,
        "12_regulatory_landscape.md",
    ),
    _r("eurail-report-v1", "geo-aeo-state", "GEO/AEO State", 14, "13_geo_aeo_state.md"),
    # Wave 3 — Analysis (3 modules, ordinals 15-17)
    _a(
        "eurail-report-v1",
        "customer-journey",
        "Customer Journey",
        15,
        "14_customer_journey_map.md",
    ),
    _a(
        "eurail-report-v1",
        "ai-maturity-scorer",
        "AI Maturity Scorer",
        16,
        "15_ai_maturity_scorecard.md",
    ),
    _a(
        "eurail-report-v1",
        "federation-moat",
        "Federation Moat",
        17,
        "16_federation_moat.md",
    ),
    # Wave 4 — Synthesis (4 modules, ordinals 18-21)
    _s(
        "eurail-report-v1",
        "adoption-scenario",
        "Adoption Scenario",
        18,
        "17_adoption_scenario.md",
    ),
    _s(
        "eurail-report-v1",
        "inaction-scenario",
        "Inaction Scenario",
        19,
        "18_inaction_scenario.md",
    ),
    _s(
        "eurail-report-v1",
        "oraclous-fit-roadmap",
        "Oraclous Fit Roadmap",
        20,
        "19_oraclous_fit_roadmap.md",
    ),
    _s("eurail-report-v1", "three-asks", "Three Asks", 21, "20_three_asks.md"),
    # Wave 5 — Quality Gates (2 modules, ordinals 22-23)
    _g(
        "eurail-report-v1",
        "adversarial-redline",
        "Adversarial Redline",
        22,
        "21_adversarial_redline.md",
    ),
    _g(
        "eurail-report-v1",
        "report-editor",
        "Report Editor",
        23,
        "00_executive_summary.md",
    ),
)


# ---------------------------------------------------------------------------
# Template-level metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    name: str
    version: str
    vertical_slug: str | None
    description: str
    modules: tuple[ModuleSpec, ...]


TEMPLATES: tuple[TemplateSpec, ...] = (
    TemplateSpec(
        template_id="assess-v1",
        name="Generic AI Adoption Assessment",
        version="1",
        vertical_slug=None,
        description=(
            "Industry-agnostic six-wave AI-adoption assessment pipeline. "
            "Composes with vertical-<slug> skills for industry-specific lenses, "
            "docify for the 5-doc HTML+PDF set, and forge for skill persistence."
        ),
        modules=ASSESS_V1_MODULES,
    ),
    TemplateSpec(
        template_id="eurail-report-v1",
        name="Eurail AI Adoption Report",
        version="1",
        vertical_slug="rail-cooperative",
        description=(
            "Purpose-built AI adoption report for Eurail B.V. — 23 modules across "
            "research, deep-dive, analysis, synthesis, and quality-gate waves. "
            "Hardcoded to Eurail's specifics (35+ rail operators, Jan 2026 breach, "
            "May 2026 Trip Planner sunset, 1.2M passes/year B.V. structure, 65/100 "
            "GEO baseline). The vertical-rail-cooperative reference implementation."
        ),
        modules=EURAIL_REPORT_V1_MODULES,
    ),
)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _skills_root() -> Path:
    """Return the configured skills root.

    Tests override via ``ORACLOUS_SKILLS_ROOT`` so the seed can run against a
    fixture tree without depending on the developer's real ``~/.claude/skills``.
    """
    env = os.environ.get("ORACLOUS_SKILLS_ROOT")
    if env:
        return Path(env).expanduser()
    return DEFAULT_SKILLS_ROOT


def _read_prompt(spec: ModuleSpec, skills_root: Path) -> str:
    """Load the system prompt for a module from its skill file.

    Raises:
        FileNotFoundError: If the skill file is missing — seed must not
            silently substitute an empty prompt.
    """
    path = skills_root / spec.prompt_relpath
    if not path.is_file():
        raise FileNotFoundError(
            f"Module prompt not found for {spec.template_id}/{spec.slug}: "
            f"expected file at {path}"
        )
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Cypher writers
# ---------------------------------------------------------------------------


_MERGE_TEMPLATE = """
MERGE (t:AssessmentTemplate {template_id: $template_id})
ON CREATE SET
    t:__Platform__,
    t.graph_id      = $graph_id,
    t.name          = $name,
    t.version       = $version,
    t.vertical_slug = $vertical_slug,
    t.description   = $description,
    t.created_at    = $now
ON MATCH SET
    t:__Platform__,
    t.name          = $name,
    t.version       = $version,
    t.vertical_slug = $vertical_slug,
    t.description   = $description
RETURN t.template_id AS template_id
"""

_MERGE_MODULE = """
MATCH (t:AssessmentTemplate {template_id: $template_id})
MERGE (m:Module {module_id: $module_id})
ON CREATE SET
    m:__Platform__,
    m.graph_id             = $graph_id,
    m.template_id          = $template_id,
    m.slug                 = $slug,
    m.name                 = $name,
    m.display_name         = $display_name,
    m.wave                 = $wave,
    m.ordinal              = $ordinal,
    m.kind                 = $kind,
    m.system_prompt        = $system_prompt,
    m.required_outputs     = $required_outputs,
    m.deliverable_filename = $deliverable_filename,
    m.created_at           = $now
ON MATCH SET
    m:__Platform__,
    m.slug                 = $slug,
    m.name                 = $name,
    m.display_name         = $display_name,
    m.wave                 = $wave,
    m.ordinal              = $ordinal,
    m.kind                 = $kind,
    m.system_prompt        = $system_prompt,
    m.required_outputs     = $required_outputs,
    m.deliverable_filename = $deliverable_filename
MERGE (t)-[:HAS_MODULE]->(m)
RETURN m.module_id AS module_id
"""

# Mirrors the node shape produced by ``AgentService.create_agent`` exactly
# (same primary + reserved labels, same property keys) so existing agent CRUD,
# the executor, and the API treat seeded agents as ordinary Graph-Native
# Agents. The only differences are:
#   * MERGE on agent_id with a deterministic ID for idempotency
#   * created_by set to SEED_CREATED_BY rather than a real user UUID
_MERGE_AGENT_AND_LINK = """
MATCH (m:Module {module_id: $module_id})
MERGE (a:Agent {agent_id: $agent_id})
ON CREATE SET
    a:__Platform__,
    a.graph_id              = $graph_id,
    a.name                  = $name,
    a.description           = $description,
    a.system_prompt         = $system_prompt,
    a.reasoning_mode        = $reasoning_mode,
    a.retriever_strategy    = $retriever_strategy,
    a.retriever_hop_depth   = $retriever_hop_depth,
    a.retriever_max_results = $retriever_max_results,
    a.tools                 = $tools,
    a.llm_config_id         = $llm_config_id,
    a.created_by            = $created_by,
    a.created_at            = $now,
    a.deactivated_at        = null
ON MATCH SET
    a:__Platform__,
    a.name                  = $name,
    a.description           = $description,
    a.system_prompt         = $system_prompt,
    a.reasoning_mode        = $reasoning_mode,
    a.tools                 = $tools
MERGE (m)-[:EXECUTED_BY]->(a)
RETURN a.agent_id AS agent_id
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class SeedCounts:
    """Per-entity totals returned by the seed run.

    Tests assert on these to confirm idempotency and module-inventory counts.
    """

    templates: int = 0
    modules: int = 0
    agents: int = 0
    has_module_edges: int = 0
    executed_by_edges: int = 0


async def seed_assessment_catalog(
    async_driver: Any,
    *,
    skills_root: Path | None = None,
) -> SeedCounts:
    """Seed the assessment catalog. Idempotent — safe to re-run.

    Args:
        async_driver: Active :class:`neo4j.AsyncDriver`. The caller owns its
            connection lifecycle (matches the pattern of
            :func:`app.db.assessment_schema_init.ensure_assessment_schema`).
        skills_root: Override for the skills root directory. Defaults to
            ``~/.claude/skills`` (or ``$ORACLOUS_SKILLS_ROOT`` if set).

    Returns:
        :class:`SeedCounts` with the post-seed totals for templates, modules,
        agents, and the two edge types.
    """
    from app.core.config import settings

    root = skills_root or _skills_root()
    now = int(time.time())

    logger.info(
        "Seeding assessment catalog from %s into graph_id=%s",
        root,
        CATALOG_GRAPH_ID,
    )

    # ── 1. Templates ────────────────────────────────────────────────────────
    for template in TEMPLATES:
        await async_driver.execute_query(
            _MERGE_TEMPLATE,
            {
                "template_id": template.template_id,
                "graph_id": CATALOG_GRAPH_ID,
                "name": template.name,
                "version": template.version,
                "vertical_slug": template.vertical_slug,
                "description": template.description,
                "now": now,
            },
            database_=settings.NEO4J_DATABASE,
        )
        logger.info("MERGEd :AssessmentTemplate %s", template.template_id)

    # ── 2. Modules + 3. Agents + 4. Edges ───────────────────────────────────
    for template in TEMPLATES:
        for spec in template.modules:
            prompt = _read_prompt(spec, root)

            # Module `name` is the clean, human-readable label (e.g. "Company
            # Intel") — no more `template_id /` qualification. The colliding
            # `module_unique` constraint on `(:Module, graph_id, name)` that
            # forced TASK-070 to synthesise qualified names has been removed
            # in TASK-075: the code-parser's `:Module` label was renamed to
            # `:CodeModule` per ADR-015 so the two namespaces no longer share
            # a constraint. Uniqueness for assessment modules is enforced by
            # `module_id_unique` on `module_id` (TASK-067 migration).
            await async_driver.execute_query(
                _MERGE_MODULE,
                {
                    "template_id": spec.template_id,
                    "graph_id": CATALOG_GRAPH_ID,
                    "module_id": spec.module_id,
                    "slug": spec.slug,
                    "name": spec.name,
                    "display_name": spec.name,
                    "wave": spec.wave,
                    "ordinal": spec.ordinal,
                    "kind": spec.kind,
                    "system_prompt": prompt,
                    "required_outputs": list(spec.required_outputs),
                    "deliverable_filename": spec.deliverable_filename,
                    "now": now,
                },
                database_=settings.NEO4J_DATABASE,
            )

            agent_id = spec.module_id  # one agent per module; same natural key
            # Agent `name` keeps the `template_id /` prefix so that when a
            # user lists all platform-managed agents across all assessment
            # templates, the names disambiguate between overlapping slugs
            # (e.g. `assess-v1 / Company Intel` vs `eurail-report-v1 /
            # Company Intel`). The module `name` doesn't need this because
            # modules are always read in the context of a parent template.
            agent_name = f"{spec.template_id} / {spec.name}"
            agent_description = (
                f"Module-executor agent for {spec.name} "
                f"(wave {spec.wave}, kind {spec.kind}) in {spec.template_id}."
            )
            await async_driver.execute_query(
                _MERGE_AGENT_AND_LINK,
                {
                    "module_id": spec.module_id,
                    "agent_id": agent_id,
                    "graph_id": CATALOG_GRAPH_ID,
                    "name": agent_name,
                    "description": agent_description,
                    "system_prompt": prompt,
                    "reasoning_mode": _REASONING_FOR_KIND[spec.kind],
                    # Module executors don't use the in-graph retriever — they
                    # do their own external research via the orchestrator.
                    # The retriever-config keys are required by the agent
                    # node shape (see AgentService.create_agent); use the
                    # AgentCreate defaults verbatim.
                    "retriever_strategy": "hybrid",
                    "retriever_hop_depth": 2,
                    "retriever_max_results": 20,
                    "tools": json.dumps(["graph_search"]),
                    "llm_config_id": None,
                    "created_by": SEED_CREATED_BY,
                    "now": now,
                },
                database_=settings.NEO4J_DATABASE,
            )

    # ── 5. Verification counts ──────────────────────────────────────────────
    counts = await _read_counts(async_driver)
    logger.info(
        "Assessment catalog seeded: %d template(s), %d module(s), "
        "%d agent(s), %d HAS_MODULE edge(s), %d EXECUTED_BY edge(s)",
        counts.templates,
        counts.modules,
        counts.agents,
        counts.has_module_edges,
        counts.executed_by_edges,
    )
    return counts


async def _read_counts(async_driver: Any) -> SeedCounts:
    """Read post-seed totals from the catalog graph."""
    from app.core.config import settings

    counts = SeedCounts()

    async def _scalar(query: str) -> int:
        res = await async_driver.execute_query(
            query,
            {"graph_id": CATALOG_GRAPH_ID},
            database_=settings.NEO4J_DATABASE,
        )
        # neo4j.AsyncDriver.execute_query returns an EagerResult; the records
        # attribute is a list of neo4j.Record. Each is dict-like and yields the
        # first column under its alias.
        if not res.records:
            return 0
        return int(res.records[0][0])

    counts.templates = await _scalar(
        "MATCH (t:AssessmentTemplate:__Platform__ {graph_id: $graph_id}) RETURN count(t)"
    )
    counts.modules = await _scalar(
        "MATCH (m:Module:__Platform__ {graph_id: $graph_id}) RETURN count(m)"
    )
    counts.agents = await _scalar(
        "MATCH (a:Agent:__Platform__ {graph_id: $graph_id}) RETURN count(a)"
    )
    counts.has_module_edges = await _scalar(
        "MATCH (:AssessmentTemplate:__Platform__ {graph_id: $graph_id})"
        "-[r:HAS_MODULE]->(:Module:__Platform__ {graph_id: $graph_id}) "
        "RETURN count(r)"
    )
    counts.executed_by_edges = await _scalar(
        "MATCH (:Module:__Platform__ {graph_id: $graph_id})"
        "-[r:EXECUTED_BY]->(:Agent:__Platform__ {graph_id: $graph_id}) "
        "RETURN count(r)"
    )
    return counts


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


async def _main() -> None:
    """Standalone CLI: applies the assessment schema, then runs the seed."""
    from app.core.neo4j_client import neo4j_client
    from app.db.assessment_schema_init import ensure_assessment_schema

    logger.info("Connecting to Neo4j for standalone assessment-catalog seed")
    await neo4j_client.connect()
    try:
        await ensure_assessment_schema(neo4j_client.async_driver)
        counts = await seed_assessment_catalog(neo4j_client.async_driver)
        logger.info(
            "Seed complete — templates=%d modules=%d agents=%d "
            "HAS_MODULE=%d EXECUTED_BY=%d",
            counts.templates,
            counts.modules,
            counts.agents,
            counts.has_module_edges,
            counts.executed_by_edges,
        )
    finally:
        await neo4j_client.disconnect()


if __name__ == "__main__":
    asyncio.run(_main())
