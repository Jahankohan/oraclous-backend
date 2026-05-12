"""Bounds tests for ``app/schemas/assessment_schemas.py`` (TASK-076).

Closes the MEDIUM security finding from TASK-073 — "no ``max_length`` on any
string/list/dict field; property-bloat DoS, same class as TASK-005 Finding 3".

For every entity / request-wrapper model in ``assessment_schemas`` we submit a
payload that just exceeds the field's declared ``max_length`` and assert that
Pydantic raises ``ValidationError``. A separate "happy-path" test for each
entity confirms a real-sized payload validates cleanly, so the bounds aren't
also rejecting legitimate input. The Eurail 2026-05-06 backfill is the
field-realism ground truth — see the module docstring of
``assessment_schemas.py`` for the observed maxes.

The tests deliberately exercise the Pydantic boundary directly (no FastAPI
TestClient, no Neo4j) because the boundary IS the defense — once a field
slips past Pydantic into the service layer, the bound is no longer enforced.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.assessment_schemas import (
    LIST_MAX_IDS,
    LIST_MAX_ITEMS,
    LIST_MAX_TAGS,
    SIZE_BLOB_TEXT,
    SIZE_CLAIM,
    SIZE_DATE,
    SIZE_ENUM,
    SIZE_FILENAME,
    SIZE_HASH,
    SIZE_ID,
    SIZE_LANG,
    SIZE_LONG_TEXT,
    SIZE_NAME,
    SIZE_SHORT_TEXT,
    SIZE_SLUG,
    SIZE_URL,
    SIZE_VERSION,
    AssessmentRun,
    AssessmentTemplate,
    Conflict,
    CreateRunRequest,
    Deliverable,
    Finding,
    Module,
    ModuleRun,
    PersistFinalDocsRequest,
    RecordFindingBulkRequest,
    RegistryItem,
    Source,
    Subject,
    UnresolvedQuestion,
    UpdateModuleRunRequest,
)

# =============================================================================
# Helpers — build a "just-barely-too-big" oversize string / list.
# We use a single character repeated rather than random bytes so tests stay
# deterministic and the failure messages are easy to read in the assertion.
# =============================================================================


def _too_long(bound: int) -> str:
    """A string one character longer than ``bound``."""
    return "x" * (bound + 1)


def _too_many(bound: int, item: str = "id") -> list[str]:
    """A list one element longer than ``bound``."""
    return [item] * (bound + 1)


# =============================================================================
# Minimal valid kwargs builders — used both for the happy-path sanity tests
# and as the base for the oversize tests (the test then overrides one field).
# =============================================================================


def _ok_finding(**over) -> dict:
    base = dict(
        finding_id="ev-1",
        graph_id="g1",
        run_id="r1",
        module_run_id="mr1",
        claim="A claim.",
    )
    base.update(over)
    return base


def _ok_conflict(**over) -> dict:
    base = dict(
        conflict_id="cf-1",
        graph_id="g1",
        run_id="r1",
        topic="topic",
        summary="summary",
    )
    base.update(over)
    return base


def _ok_question(**over) -> dict:
    base = dict(
        question_id="uq-1",
        graph_id="g1",
        run_id="r1",
        module_run_id="mr1",
        text="A question.",
    )
    base.update(over)
    return base


def _ok_deliverable(**over) -> dict:
    base = dict(
        deliverable_id="del-1",
        graph_id="g1",
        run_id="r1",
        kind="module-md",
        filename="01_intro.md",
    )
    base.update(over)
    return base


def _ok_run(**over) -> dict:
    base = dict(
        run_id="r1",
        graph_id="g1",
        template_id="tpl-1",
        subject_id="subj-1",
    )
    base.update(over)
    return base


def _ok_module_run(**over) -> dict:
    base = dict(
        module_run_id="mr1",
        graph_id="g1",
        run_id="r1",
        module_id="mod-1",
        wave=1,
    )
    base.update(over)
    return base


def _ok_subject(**over) -> dict:
    base = dict(
        subject_id="subj-1",
        graph_id="g1",
        slug="eurail",
        name="Eurail B.V.",
    )
    base.update(over)
    return base


def _ok_source(**over) -> dict:
    base = dict(source_id="src-1")
    base.update(over)
    return base


def _ok_template(**over) -> dict:
    base = dict(
        template_id="tpl-1",
        slug="eurail-report-v1",
        name="Eurail Report v1",
        version="1.0.0",
    )
    base.update(over)
    return base


def _ok_module(**over) -> dict:
    base = dict(
        module_id="mod-1",
        template_id="tpl-1",
        slug="customer-journey",
        name="Customer Journey",
        wave=1,
        ordinal=0,
        kind="research",
    )
    base.update(over)
    return base


def _ok_registry(**over) -> dict:
    base = dict(
        item_id="ri-1",
        graph_id="g1",
        kind="skill",
        slug="custom-skill",
        owner_user_id="user-1",
        name="Custom Skill",
    )
    base.update(over)
    return base


# =============================================================================
# Catalog-layer string-bound tests
# =============================================================================


class TestAssessmentTemplateBounds:
    @pytest.mark.parametrize(
        ("field", "bound"),
        [
            ("template_id", SIZE_ID),
            ("slug", SIZE_SLUG),
            ("name", SIZE_NAME),
            ("version", SIZE_VERSION),
            ("vertical_slug", SIZE_SLUG),
            ("description", SIZE_SHORT_TEXT),
        ],
    )
    def test_oversize_string_rejected(self, field, bound):
        with pytest.raises(ValidationError):
            AssessmentTemplate(**_ok_template(**{field: _too_long(bound)}))

    def test_happy_path_at_realistic_size(self):
        # Realistic — well under every bound. Must validate.
        AssessmentTemplate(**_ok_template(description="x" * 500))


class TestModuleBounds:
    @pytest.mark.parametrize(
        ("field", "bound"),
        [
            ("module_id", SIZE_ID),
            ("template_id", SIZE_ID),
            ("slug", SIZE_SLUG),
            ("name", SIZE_NAME),
            ("agent_id", SIZE_ID),
            ("description", SIZE_SHORT_TEXT),
        ],
    )
    def test_oversize_string_rejected(self, field, bound):
        with pytest.raises(ValidationError):
            Module(**_ok_module(**{field: _too_long(bound)}))


class TestSourceBounds:
    @pytest.mark.parametrize(
        ("field", "bound"),
        [
            ("source_id", SIZE_ID),
            ("type", SIZE_ENUM),
            ("url_normalized", SIZE_URL),
            ("name", SIZE_NAME),
            ("publication_date", SIZE_DATE),
            ("fetch_date", SIZE_DATE),
            ("language", SIZE_LANG),
        ],
    )
    def test_oversize_string_rejected(self, field, bound):
        with pytest.raises(ValidationError):
            Source(**_ok_source(**{field: _too_long(bound)}))

    def test_happy_path_eurail_max(self):
        # Largest values seen in Eurail's evidence.jsonl: url 173, name 117.
        Source(
            source_id="src-1",
            url_normalized="https://example.com/" + "a" * 150,
            name="x" * 117,
        )


# =============================================================================
# Run-layer string-bound tests
# =============================================================================


class TestSubjectBounds:
    @pytest.mark.parametrize(
        ("field", "bound"),
        [
            ("subject_id", SIZE_ID),
            ("graph_id", SIZE_ID),
            ("slug", SIZE_SLUG),
            ("name", SIZE_NAME),
            ("vertical_slug", SIZE_SLUG),
        ],
    )
    def test_oversize_string_rejected(self, field, bound):
        with pytest.raises(ValidationError):
            Subject(**_ok_subject(**{field: _too_long(bound)}))

    def test_oversize_list_domains_rejected(self):
        with pytest.raises(ValidationError):
            Subject(**_ok_subject(domains=_too_many(LIST_MAX_TAGS)))

    def test_oversize_list_aliases_rejected(self):
        with pytest.raises(ValidationError):
            Subject(**_ok_subject(aliases=_too_many(LIST_MAX_TAGS)))


class TestAssessmentRunBounds:
    @pytest.mark.parametrize(
        ("field", "bound"),
        [
            ("run_id", SIZE_ID),
            ("graph_id", SIZE_ID),
            ("template_id", SIZE_ID),
            ("subject_id", SIZE_ID),
            ("failure_reason", SIZE_SHORT_TEXT),
        ],
    )
    def test_oversize_string_rejected(self, field, bound):
        with pytest.raises(ValidationError):
            AssessmentRun(**_ok_run(**{field: _too_long(bound)}))


class TestModuleRunBounds:
    @pytest.mark.parametrize(
        ("field", "bound"),
        [
            ("module_run_id", SIZE_ID),
            ("graph_id", SIZE_ID),
            ("run_id", SIZE_ID),
            ("module_id", SIZE_ID),
            ("deliverable_path", SIZE_URL),
            ("failure_reason", SIZE_SHORT_TEXT),
        ],
    )
    def test_oversize_string_rejected(self, field, bound):
        with pytest.raises(ValidationError):
            ModuleRun(**_ok_module_run(**{field: _too_long(bound)}))


class TestFindingBounds:
    @pytest.mark.parametrize(
        ("field", "bound"),
        [
            ("finding_id", SIZE_ID),
            ("graph_id", SIZE_ID),
            ("run_id", SIZE_ID),
            ("module_run_id", SIZE_ID),
            ("claim", SIZE_CLAIM),
            ("raw", SIZE_BLOB_TEXT),
            ("notes", SIZE_LONG_TEXT),
            ("superseded_by", SIZE_ID),
            ("source_id", SIZE_ID),
            ("source_quote", SIZE_CLAIM),
            ("source_locator", SIZE_SHORT_TEXT),
        ],
    )
    def test_oversize_string_rejected(self, field, bound):
        with pytest.raises(ValidationError):
            Finding(**_ok_finding(**{field: _too_long(bound)}))

    def test_oversize_dimensions_rejected(self):
        with pytest.raises(ValidationError):
            Finding(**_ok_finding(dimensions=_too_many(LIST_MAX_TAGS, "dim")))

    def test_happy_path_eurail_realistic(self):
        # Largest sizes observed in Eurail evidence.jsonl: claim 509, raw 775,
        # notes 275. Confirm Finding accepts those comfortably.
        Finding(
            **_ok_finding(
                claim="x" * 600,
                raw="x" * 800,
                notes="x" * 300,
                dimensions=["regulatory", "tech-maturity", "governance", "moat"],
            )
        )


class TestConflictBounds:
    @pytest.mark.parametrize(
        ("field", "bound"),
        [
            ("conflict_id", SIZE_ID),
            ("graph_id", SIZE_ID),
            ("run_id", SIZE_ID),
            ("topic", SIZE_SHORT_TEXT),
            ("summary", SIZE_LONG_TEXT),
            ("resolution", SIZE_LONG_TEXT),
            ("synthesis_note", SIZE_LONG_TEXT),
        ],
    )
    def test_oversize_string_rejected(self, field, bound):
        with pytest.raises(ValidationError):
            Conflict(**_ok_conflict(**{field: _too_long(bound)}))

    def test_oversize_evidence_ids_rejected(self):
        with pytest.raises(ValidationError):
            Conflict(
                **_ok_conflict(involved_finding_ids=_too_many(LIST_MAX_IDS, "ev-x"))
            )

    def test_happy_path_eurail_realistic(self):
        # Largest sizes observed in conflicts.jsonl: explanation 757,
        # synthesis_note 607, summary 379.
        Conflict(
            **_ok_conflict(
                summary="x" * 400,
                resolution="x" * 800,
                synthesis_note="x" * 700,
                involved_finding_ids=[f"ev-{i}" for i in range(8)],
            )
        )


class TestDeliverableBounds:
    @pytest.mark.parametrize(
        ("field", "bound"),
        [
            ("deliverable_id", SIZE_ID),
            ("graph_id", SIZE_ID),
            ("run_id", SIZE_ID),
            ("module_run_id", SIZE_ID),
            ("filename", SIZE_FILENAME),
            ("content_uri", SIZE_URL),
            ("content_inline", SIZE_BLOB_TEXT),
            ("sha256", SIZE_HASH),
        ],
    )
    def test_oversize_string_rejected(self, field, bound):
        with pytest.raises(ValidationError):
            Deliverable(**_ok_deliverable(**{field: _too_long(bound)}))

    def test_happy_path_eurail_largest_deliverable(self):
        # Largest Eurail module-md (22_evidence_derived_strategy.md) is
        # 43_876 chars on disk; well below the 65_536 SIZE_BLOB_TEXT bound.
        Deliverable(
            **_ok_deliverable(content_inline="x" * 45_000),
        )


class TestUnresolvedQuestionBounds:
    @pytest.mark.parametrize(
        ("field", "bound"),
        [
            ("question_id", SIZE_ID),
            ("graph_id", SIZE_ID),
            ("run_id", SIZE_ID),
            ("module_run_id", SIZE_ID),
            ("text", SIZE_LONG_TEXT),
            ("suggested_module", SIZE_SLUG),
        ],
    )
    def test_oversize_string_rejected(self, field, bound):
        with pytest.raises(ValidationError):
            UnresolvedQuestion(**_ok_question(**{field: _too_long(bound)}))


# =============================================================================
# Registry-layer string-bound tests
# =============================================================================


class TestRegistryItemBounds:
    @pytest.mark.parametrize(
        ("field", "bound"),
        [
            ("item_id", SIZE_ID),
            ("graph_id", SIZE_ID),
            ("slug", SIZE_SLUG),
            ("version", SIZE_VERSION),
            ("owner_user_id", SIZE_ID),
            ("name", SIZE_NAME),
            ("description", SIZE_SHORT_TEXT),
            ("content_uri", SIZE_URL),
            ("sha256", SIZE_HASH),
        ],
    )
    def test_oversize_string_rejected(self, field, bound):
        with pytest.raises(ValidationError):
            RegistryItem(**_ok_registry(**{field: _too_long(bound)}))


# =============================================================================
# Request/response wrapper bound tests
# =============================================================================


class TestCreateRunRequestBounds:
    def test_oversize_template_slug_rejected(self):
        with pytest.raises(ValidationError):
            CreateRunRequest(
                template_slug=_too_long(SIZE_SLUG),
                subject=Subject(**_ok_subject()),
            )

    def test_oversize_run_id_rejected(self):
        with pytest.raises(ValidationError):
            CreateRunRequest(
                template_slug="eurail-report-v1",
                subject=Subject(**_ok_subject()),
                run_id=_too_long(SIZE_ID),
            )


class TestUpdateModuleRunRequestBounds:
    def test_oversize_deliverable_path_rejected(self):
        with pytest.raises(ValidationError):
            UpdateModuleRunRequest(deliverable_path=_too_long(SIZE_URL))

    def test_oversize_failure_reason_rejected(self):
        with pytest.raises(ValidationError):
            UpdateModuleRunRequest(failure_reason=_too_long(SIZE_SHORT_TEXT))


class TestRecordFindingBulkRequestBounds:
    def test_oversize_findings_list_rejected(self):
        # The list cap is LIST_MAX_ITEMS findings.
        valid_finding = Finding(**_ok_finding())
        with pytest.raises(ValidationError):
            RecordFindingBulkRequest(findings=[valid_finding] * (LIST_MAX_ITEMS + 1))

    def test_at_cap_accepts(self):
        # Boundary check — exactly LIST_MAX_ITEMS items must be accepted.
        valid_finding = Finding(**_ok_finding())
        RecordFindingBulkRequest(findings=[valid_finding] * LIST_MAX_ITEMS)


class TestPersistFinalDocsRequestBounds:
    def test_oversize_deliverables_list_rejected(self):
        valid_deliverable = Deliverable(**_ok_deliverable())
        with pytest.raises(ValidationError):
            PersistFinalDocsRequest(
                deliverables=[valid_deliverable] * (LIST_MAX_ITEMS + 1)
            )


# =============================================================================
# Boundary tests — exactly-at-cap must accept; one-over must reject.
# Provides confidence the bound is exclusive on the over side.
# =============================================================================


class TestBoundaryExactness:
    def test_claim_at_cap_accepts(self):
        Finding(**_ok_finding(claim="x" * SIZE_CLAIM))

    def test_claim_one_over_rejects(self):
        with pytest.raises(ValidationError):
            Finding(**_ok_finding(claim="x" * (SIZE_CLAIM + 1)))

    def test_content_inline_at_cap_accepts(self):
        Deliverable(**_ok_deliverable(content_inline="x" * SIZE_BLOB_TEXT))

    def test_content_inline_one_over_rejects(self):
        with pytest.raises(ValidationError):
            Deliverable(**_ok_deliverable(content_inline="x" * (SIZE_BLOB_TEXT + 1)))

    def test_dimensions_at_cap_accepts(self):
        Finding(**_ok_finding(dimensions=["d"] * LIST_MAX_TAGS))

    def test_dimensions_one_over_rejects(self):
        with pytest.raises(ValidationError):
            Finding(**_ok_finding(dimensions=["d"] * (LIST_MAX_TAGS + 1)))

    def test_involved_finding_ids_at_cap_accepts(self):
        Conflict(**_ok_conflict(involved_finding_ids=["ev"] * LIST_MAX_IDS))

    def test_involved_finding_ids_one_over_rejects(self):
        with pytest.raises(ValidationError):
            Conflict(**_ok_conflict(involved_finding_ids=["ev"] * (LIST_MAX_IDS + 1)))
