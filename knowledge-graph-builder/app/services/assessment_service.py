"""
Assessment substrate write-path service (STORY-026, TASK-068).

`AssessmentService` is the application-layer write surface for the assessment
substrate. It backs the REST endpoints (TASK-069) and the MCP wrappers
(SPRINT-002). Read methods land in SPRINT-002 — this file is write-only.

Architectural rules honored (`oraclous-data-studio/CLAUDE.md`):

1.  `graph_id` is a required parameter on every public method. Every Cypher
    statement filters by it. Cross-tenant queries are impossible.
2.  All Cypher is parameterized. No f-string interpolation of user input.
3.  Every platform-managed `MERGE` adds the `:__Platform__` marker (ADR-015).
4.  The catalog graph anchor uses `:Graph:__Rebac__` per the existing pattern
    in `rebac_service.py` and `graph_node_service.py`.
5.  Async driver (`neo4j_client.async_driver`) for all methods.
6.  Idempotency: every write `MERGE`s on the natural id (run_id, finding_id,
    etc.). Replay is safe.

Tenancy notes (ADR-018 §Tenancy concession):

-   `:AssessmentRun`, `:ModuleRun`, `:Finding`, `:Conflict`, `:Deliverable`,
    `:UnresolvedQuestion`, `:Subject` live in the customer's tenant graph.
-   `:AssessmentTemplate`, `:Module`, `:Source` live in the catalog graph
    `__assessments_catalog__`.
-   `(finding)-[:CITES]->(source)` crosses the namespace boundary by
    `source_id` join in the API layer — never by Cypher MATCH across graphs.
    This service therefore MERGEs the `:Source` in the catalog graph and
    creates the `[:CITES]` edge as a separate statement scoped to the
    tenant graph; the edge resolves the source by id at read time.

Registry tenancy (ADR-019):

-   `private` RegistryItem writes go to the owner's tenant graph.
-   `curated` and `public` RegistryItem writes go to `__registry__`.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from neo4j import AsyncDriver

from app.core.logging import get_logger
from app.schemas.assessment_schemas import (
    ASSESSMENTS_CATALOG_GRAPH_ID,
    REGISTRY_CATALOG_GRAPH_ID,
    BulkItemResult,
    BulkResponse,
    Conflict,
    ConflictRow,
    CreateRunRequest,
    CreateRunResponse,
    Deliverable,
    DeliverableRow,
    FinalizeRunResponse,
    Finding,
    FindingRow,
    FindingSearchRow,
    ModuleDefinition,
    ModuleRunRow,
    RegistryItem,
    RegistryItemContent,
    RegistryItemRow,
    RegistryKind,
    RegistryVisibility,
    RunDetail,
    RunStatus,
    RunSummary,
    Source,
    Subject,
    UnresolvedQuestion,
    UnresolvedQuestionRow,
    UpdateModuleRunRequest,
    WaveModuleStatus,
    WaveStatusResponse,
)
from app.services.assessment_event_broker import (
    AssessmentEventBroker,
    get_assessment_event_broker,
)
from app.services.blob_cas_service import (
    BlobCASService,
    InvalidBlobURIError,
    is_blob_uri,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)


# =============================================================================
# Typed errors — surfaced to the REST layer with structured HTTP status codes.
# =============================================================================


class RegistryOwnershipError(PermissionError):
    """Raised when a RegistryItem write fails the ownership check (ADR-019).

    Per TASK-073 Finding 1 (TASK-069): public/yanked items in `__registry__`
    are owner-writable only. A non-owner attempting to overwrite a public item
    triggers this error. The endpoint layer translates this to HTTP 403.

    Curated items (admin-gated at the endpoint) are exempt from the ownership
    check because platform admins legitimately update items owned by others.
    """


# =============================================================================
# Citation-coverage gate thresholds — STORY-026 §Acceptance Criteria.
# Tuned for the Eurail backfill (~600 direct findings, 23 module deliverables,
# 5 final docs). A run that fails the gate moves to `status='failed'`.
# =============================================================================

MIN_DIRECT_FINDINGS_FOR_PASS = 1
MIN_DELIVERABLES_FOR_PASS = 1


# =============================================================================
# Helpers
# =============================================================================


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id(prefix: str) -> str:
    """Generate a UUID4-based id with a short type prefix for readability."""
    return f"{prefix}-{uuid.uuid4().hex}"


def _to_neo_primitive(value: Any) -> Any:
    """Coerce Python values to Neo4j-compatible primitives.

    Lists, strings, numbers, bools, None, and ISO datetimes pass through.
    Dicts get serialized to JSON so they can land as a single property value
    (Neo4j 5 supports nested maps on properties but not on every storage
    engine; using JSON keeps the migration target broad).
    """
    if isinstance(value, datetime):
        # Strip tz to UTC ISO; Neo4j datetime() accepts the string.
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()
    if isinstance(value, dict):
        import json

        return json.dumps(value, separators=(",", ":"))
    return value


# =============================================================================
# Service
# =============================================================================


class AssessmentService:
    """Write-path service for assessment runs, findings, conflicts, deliverables.

    The constructor takes an `AsyncDriver` so unit tests can pass a mock.
    Production callers (REST endpoints in TASK-069) will inject
    `neo4j_client.async_driver` via FastAPI Depends().

    Optionally takes a `BlobCASService` for deliverable content storage
    (TASK-082). When not supplied, the service falls back to constructing
    its own stateless instance — `BlobCASService` carries no state and the
    Postgres `AsyncSession` is passed in per-call by the endpoint layer.
    """

    def __init__(
        self,
        driver: AsyncDriver,
        blob_cas: BlobCASService | None = None,
        event_broker: AssessmentEventBroker | None = None,
    ):
        self._driver = driver
        self._blob_cas = blob_cas or BlobCASService()
        # TASK-081: fire-and-forget event broker for SSE tail_run subscribers.
        # `publish()` is sync + non-blocking — drops events into per-subscriber
        # bounded queues — so a slow SSE client can never stall a write.
        # Defaults to the process-wide singleton so callers that don't supply
        # one still exercise the publish path; tests inject their own.
        self._events = event_broker or get_assessment_event_broker()

    def _publish_event(
        self,
        graph_id: str,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Fire-and-forget event publish — never raises, never blocks.

        Wrapped in a try/except so a broker malfunction (e.g. an overflowing
        queue) can never roll back a Neo4j write that has already committed.
        Called *after* every successful write Cypher commit.
        """
        try:
            # The broker's EventType Literal is enforced at the broker
            # boundary; this wrapper accepts plain `str` so service-layer
            # call sites stay readable.
            self._events.publish(graph_id, run_id, event_type, payload)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "assessment_service: event publish failed graph_id=%s run_id=%s "
                "type=%s err=%s",
                graph_id,
                run_id,
                event_type,
                exc,
            )

    # =========================================================================
    # create_run
    # =========================================================================

    async def create_run(
        self,
        graph_id: str,
        request: CreateRunRequest,
        created_by: str | None = None,
    ) -> CreateRunResponse:
        """Bootstrap an `:AssessmentRun` + all `:ModuleRun` rows in `planned`.

        Reads the template + modules from the catalog graph; writes the run
        and `:ModuleRun`s in the tenant graph. MERGEs the `:Subject` in the
        tenant graph (deduped by `slug`).

        Idempotent on `request.run_id` when the caller supplies one — replay
        returns the existing run with `already_existed=True`.
        """
        if not graph_id:
            raise ValueError("graph_id is required")

        # 1. Resolve template + modules from the catalog graph.
        template_row = await self._fetch_template_by_slug(request.template_slug)
        if template_row is None:
            raise ValueError(
                f"Assessment template not found: slug={request.template_slug!r} "
                f"(catalog graph_id={ASSESSMENTS_CATALOG_GRAPH_ID!r})"
            )
        template_id: str = template_row["template_id"]

        modules = await self._fetch_modules_for_template(template_id)
        if not modules:
            raise ValueError(
                f"Template {request.template_slug!r} has no :Module rows in the catalog graph"
            )

        # 2. Idempotency probe — if the caller supplied a run_id and it already
        #    exists in the tenant graph, short-circuit and return that run.
        run_id = request.run_id or _new_id("run")
        existing = await self._fetch_existing_run(graph_id, run_id)
        if existing is not None:
            module_run_ids = await self._fetch_module_run_ids(graph_id, run_id)
            logger.info(
                "create_run: idempotent replay graph_id=%s run_id=%s", graph_id, run_id
            )
            return CreateRunResponse(
                run_id=run_id,
                template_id=existing["template_id"],
                subject_id=existing["subject_id"],
                module_run_ids=module_run_ids,
                status=existing["status"],
                already_existed=True,
            )

        # 3. MERGE the Subject in the tenant graph (deduped by slug).
        subject_id = await self._merge_subject_tenant(graph_id, request.subject)

        # 4. CREATE the AssessmentRun.
        started_at = _utcnow()
        run_props = {
            "run_id": run_id,
            "graph_id": graph_id,
            "template_id": template_id,
            "subject_id": subject_id,
            "status": "planned",
            "started_at": started_at.isoformat(),
            "orchestrator_last_seen": started_at.isoformat(),
            "cli_flags": _to_neo_primitive(request.cli_flags),
            "created_by": created_by or "",
        }
        # MERGE the :AssessmentRun. Per TASK-073 Finding 1 (cross-tenant MERGE
        # collision): MERGE on `run_id` alone risks matching a foreign tenant's
        # run because the migration declares a single-property UNIQUE constraint
        # on `run_id`. The `WITH r WHERE r.graph_id = $graph_id` filter rejects
        # any cross-tenant match before we touch the row. RuntimeError surfaces
        # the collision to the caller (vs. silently creating orphan rows).
        run_merge_result = await self._driver.execute_query(
            """
            MERGE (r:AssessmentRun:__Platform__ {run_id: $run_id})
            ON CREATE SET
                r.graph_id              = $graph_id,
                r.template_id           = $template_id,
                r.subject_id            = $subject_id,
                r.status                = $status,
                r.started_at            = datetime($started_at),
                r.orchestrator_last_seen = datetime($orchestrator_last_seen),
                r.cli_flags             = $cli_flags,
                r.created_by            = $created_by
            WITH r
            WHERE r.graph_id = $graph_id
            RETURN r.run_id AS id
            """,
            {
                "run_id": run_props["run_id"],
                "graph_id": run_props["graph_id"],
                "template_id": run_props["template_id"],
                "subject_id": run_props["subject_id"],
                "status": run_props["status"],
                "started_at": run_props["started_at"],
                "orchestrator_last_seen": run_props["orchestrator_last_seen"],
                "cli_flags": run_props["cli_flags"],
                "created_by": run_props["created_by"],
            },
        )
        if not run_merge_result.records:
            raise RuntimeError(
                f"AssessmentRun MERGE matched a foreign-tenant row for "
                f"run_id={run_id!r} (graph_id={graph_id!r}). Refusing to "
                f"attach :ModuleRun children to a cross-tenant run."
            )

        # 5. CREATE one :ModuleRun per :Module, all in status='planned'.
        module_run_ids: list[str] = []
        for mod in modules:
            module_run_id = _new_id("mr")
            module_run_ids.append(module_run_id)
            # Per TASK-073 Finding 1: MERGE on `module_run_id` alone could match
            # a foreign tenant's :ModuleRun. WHERE-after-MERGE filters that out
            # before we attach the [:HAS_MODULE_RUN] edge.
            await self._driver.execute_query(
                """
                MERGE (mr:ModuleRun:__Platform__ {module_run_id: $module_run_id})
                ON CREATE SET
                    mr.graph_id        = $graph_id,
                    mr.run_id          = $run_id,
                    mr.module_id       = $module_id,
                    mr.wave            = $wave,
                    mr.status          = 'planned',
                    mr.evidence_count  = 0
                WITH mr
                WHERE mr.graph_id = $graph_id
                MATCH (r:AssessmentRun:__Platform__ {graph_id: $graph_id, run_id: $run_id})
                MERGE (r)-[:HAS_MODULE_RUN]->(mr)
                """,
                {
                    "module_run_id": module_run_id,
                    "graph_id": graph_id,
                    "run_id": run_id,
                    "module_id": mod["module_id"],
                    "wave": mod["wave"],
                },
            )

        logger.info(
            "create_run: graph_id=%s run_id=%s template=%s modules=%d",
            graph_id,
            run_id,
            request.template_slug,
            len(module_run_ids),
        )
        return CreateRunResponse(
            run_id=run_id,
            template_id=template_id,
            subject_id=subject_id,
            module_run_ids=module_run_ids,
            status="planned",
            already_existed=False,
        )

    async def _fetch_template_by_slug(self, slug: str) -> dict[str, Any] | None:
        result = await self._driver.execute_query(
            """
            MATCH (t:AssessmentTemplate:__Platform__ {slug: $slug})
            RETURN t.template_id AS template_id, t.slug AS slug
            LIMIT 1
            """,
            {"slug": slug},
        )
        if not result.records:
            return None
        rec = result.records[0]
        return {"template_id": rec["template_id"], "slug": rec["slug"]}

    async def _fetch_modules_for_template(
        self, template_id: str
    ) -> list[dict[str, Any]]:
        result = await self._driver.execute_query(
            """
            MATCH (t:AssessmentTemplate:__Platform__ {template_id: $template_id})
                  -[:HAS_MODULE]->(m:Module:__Platform__)
            RETURN m.module_id AS module_id, m.slug AS slug,
                   m.wave AS wave, m.ordinal AS ordinal, m.kind AS kind
            ORDER BY m.wave ASC, m.ordinal ASC
            """,
            {"template_id": template_id},
        )
        return [
            {
                "module_id": rec["module_id"],
                "slug": rec["slug"],
                "wave": rec["wave"],
                "ordinal": rec["ordinal"],
                "kind": rec["kind"],
            }
            for rec in result.records
        ]

    async def _fetch_existing_run(
        self, graph_id: str, run_id: str
    ) -> dict[str, Any] | None:
        result = await self._driver.execute_query(
            """
            MATCH (r:AssessmentRun:__Platform__ {graph_id: $graph_id, run_id: $run_id})
            RETURN r.template_id AS template_id, r.subject_id AS subject_id, r.status AS status
            LIMIT 1
            """,
            {"graph_id": graph_id, "run_id": run_id},
        )
        if not result.records:
            return None
        rec = result.records[0]
        return {
            "template_id": rec["template_id"],
            "subject_id": rec["subject_id"],
            "status": rec["status"],
        }

    async def _fetch_module_run_ids(self, graph_id: str, run_id: str) -> list[str]:
        result = await self._driver.execute_query(
            """
            MATCH (r:AssessmentRun:__Platform__ {graph_id: $graph_id, run_id: $run_id})
                  -[:HAS_MODULE_RUN]->(mr:ModuleRun:__Platform__)
            RETURN mr.module_run_id AS module_run_id
            ORDER BY mr.wave ASC, mr.module_run_id ASC
            """,
            {"graph_id": graph_id, "run_id": run_id},
        )
        return [rec["module_run_id"] for rec in result.records]

    async def _merge_subject_tenant(self, graph_id: str, subject: Subject) -> str:
        """MERGE the subject in the tenant graph, deduped by `slug`.

        The subject.subject_id is treated as a *suggested* id — if a subject
        with the same `slug` already exists in the tenant graph we reuse its
        id, otherwise we create with the supplied id (or generate one).
        """
        subject_id = subject.subject_id or _new_id("subj")
        result = await self._driver.execute_query(
            """
            MERGE (s:Subject:__Platform__ {graph_id: $graph_id, slug: $slug})
            ON CREATE SET
                s.subject_id    = $subject_id,
                s.name          = $name,
                s.vertical_slug = $vertical_slug,
                s.domains       = $domains,
                s.aliases       = $aliases
            ON MATCH SET
                s.name          = $name,
                s.vertical_slug = coalesce($vertical_slug, s.vertical_slug),
                s.domains       = $domains,
                s.aliases       = $aliases
            RETURN s.subject_id AS subject_id
            """,
            {
                "graph_id": graph_id,
                "slug": subject.slug,
                "subject_id": subject_id,
                "name": subject.name,
                "vertical_slug": subject.vertical_slug,
                "domains": subject.domains,
                "aliases": subject.aliases,
            },
        )
        return result.records[0]["subject_id"]

    # =========================================================================
    # update_module_run
    # =========================================================================

    async def update_module_run(
        self,
        graph_id: str,
        run_id: str,
        module_run_id: str,
        update: UpdateModuleRunRequest,
    ) -> bool:
        """State-transition + heartbeat update for a `:ModuleRun`.

        Allowed transitions (NOT enforced server-side in SPRINT-001 to keep
        the surface flexible during orchestrator iteration; orchestrator-side
        contract per STORY-026 §Coordination Model):

            planned   → running, failed, cancelled
            running   → finished, failed
            finished  → (terminal)
            failed    → (terminal; orchestrator may insert a NEW :ModuleRun for retry)
            cancelled → (terminal)

        Returns True iff the row existed and was updated.
        """
        if not graph_id:
            raise ValueError("graph_id is required")
        if not run_id or not module_run_id:
            raise ValueError("run_id and module_run_id are required")

        set_clauses: list[str] = []
        params: dict[str, Any] = {
            "graph_id": graph_id,
            "run_id": run_id,
            "module_run_id": module_run_id,
        }

        if update.status is not None:
            set_clauses.append("mr.status = $status")
            params["status"] = update.status
        if update.started_at is not None:
            set_clauses.append("mr.started_at = datetime($started_at)")
            params["started_at"] = update.started_at.isoformat()
        if update.finished_at is not None:
            set_clauses.append("mr.finished_at = datetime($finished_at)")
            params["finished_at"] = update.finished_at.isoformat()
        if update.last_heartbeat_at is not None:
            set_clauses.append("mr.last_heartbeat_at = datetime($last_heartbeat_at)")
            params["last_heartbeat_at"] = update.last_heartbeat_at.isoformat()
        if update.evidence_count is not None:
            set_clauses.append("mr.evidence_count = $evidence_count")
            params["evidence_count"] = update.evidence_count
        if update.deliverable_path is not None:
            set_clauses.append("mr.deliverable_path = $deliverable_path")
            params["deliverable_path"] = update.deliverable_path
        if update.failure_reason is not None:
            set_clauses.append("mr.failure_reason = $failure_reason")
            params["failure_reason"] = update.failure_reason

        if not set_clauses:
            # No-op update — still verify the row exists so callers get a clear
            # signal if they're operating on a deleted/missing module_run.
            result = await self._driver.execute_query(
                """
                MATCH (mr:ModuleRun:__Platform__ {
                    graph_id: $graph_id, run_id: $run_id, module_run_id: $module_run_id
                })
                RETURN mr.module_run_id AS id
                """,
                params,
            )
            return bool(result.records)

        # TASK-081: read the prev status before the SET so the status-changed
        # event payload can carry both prev and new. The probe is a separate
        # query because the SET-and-return-prev pattern is awkward in Cypher
        # and we need the row anyway for cross-tenant guard.
        prev_status: str | None = None
        if update.status is not None:
            probe = await self._driver.execute_query(
                """
                MATCH (mr:ModuleRun:__Platform__ {
                    graph_id: $graph_id, run_id: $run_id, module_run_id: $module_run_id
                })
                RETURN mr.status AS status
                LIMIT 1
                """,
                {
                    "graph_id": graph_id,
                    "run_id": run_id,
                    "module_run_id": module_run_id,
                },
            )
            if probe.records:
                prev_status = probe.records[0]["status"]

        cypher = f"""
            MATCH (mr:ModuleRun:__Platform__ {{
                graph_id: $graph_id, run_id: $run_id, module_run_id: $module_run_id
            }})
            SET {", ".join(set_clauses)}
            RETURN mr.module_run_id AS id
            """
        result = await self._driver.execute_query(cypher, params)
        updated = bool(result.records)

        if updated:
            # TASK-081: publish post-commit. Status changes carry prev/new; a
            # heartbeat-only update emits `module_run.heartbeat` (broker
            # throttles to 1 per 30s per module_run_id).
            if update.status is not None and update.status != prev_status:
                self._publish_event(
                    graph_id,
                    run_id,
                    "module_run.status_changed",
                    {
                        "module_run_id": module_run_id,
                        "prev_status": prev_status,
                        "new_status": update.status,
                        "at": _utcnow().isoformat(),
                    },
                )
            if update.last_heartbeat_at is not None:
                self._publish_event(
                    graph_id,
                    run_id,
                    "module_run.heartbeat",
                    {
                        "module_run_id": module_run_id,
                        "last_heartbeat_at": update.last_heartbeat_at.isoformat(),
                    },
                )

        return updated

    async def heartbeat_run(self, graph_id: str, run_id: str) -> bool:
        """Update `:AssessmentRun.orchestrator_last_seen` to now.

        Resumability per STORY-026 §Acceptance Criteria. The orchestrator
        calls this every 60s; the Celery reset agent uses it to detect
        orphaned runs.
        """
        if not graph_id or not run_id:
            raise ValueError("graph_id and run_id are required")
        now = _utcnow().isoformat()
        result = await self._driver.execute_query(
            """
            MATCH (r:AssessmentRun:__Platform__ {graph_id: $graph_id, run_id: $run_id})
            SET r.orchestrator_last_seen = datetime($now)
            RETURN r.run_id AS id
            """,
            {"graph_id": graph_id, "run_id": run_id, "now": now},
        )
        return bool(result.records)

    # =========================================================================
    # record_finding_bulk
    # =========================================================================

    async def record_finding_bulk(
        self,
        graph_id: str,
        run_id: str,
        module_run_id: str,
        findings: list[Finding],
    ) -> BulkResponse:
        """Write a batch of `:Finding` rows under one `:ModuleRun`.

        Per-record success/failure semantics (STORY-026 Open Question #4
        resolution): a malformed finding among 50 valid ones does not roll
        back the batch. The response carries one `BulkItemResult` per input
        in the same order so the caller can correlate by index or by id.

        Each finding is MERGEd on `finding_id` for idempotent replay. The
        `:Source` (if `source_id` is supplied) is MERGEd in the catalog
        graph and the `[:CITES]` edge created in the tenant graph by
        `source_id` reference (no cross-graph MATCH; see ADR-018 §Tenancy).
        """
        if not graph_id:
            raise ValueError("graph_id is required")
        if not run_id or not module_run_id:
            raise ValueError("run_id and module_run_id are required")

        # Verify parent ModuleRun exists. Per-finding writes that would
        # silently create orphan rows are a footgun.
        parent_check = await self._driver.execute_query(
            """
            MATCH (mr:ModuleRun:__Platform__ {
                graph_id: $graph_id, run_id: $run_id, module_run_id: $module_run_id
            })
            RETURN mr.module_run_id AS id
            LIMIT 1
            """,
            {
                "graph_id": graph_id,
                "run_id": run_id,
                "module_run_id": module_run_id,
            },
        )
        if not parent_check.records:
            raise ValueError(
                f"ModuleRun not found: graph_id={graph_id!r} run_id={run_id!r} "
                f"module_run_id={module_run_id!r}"
            )

        results: list[BulkItemResult] = []
        succeeded = 0
        failed = 0

        for finding in findings:
            try:
                already_existed = await self._record_one_finding(
                    graph_id, run_id, module_run_id, finding
                )
                results.append(
                    BulkItemResult(
                        id=finding.finding_id,
                        success=True,
                        already_existed=already_existed,
                    )
                )
                succeeded += 1
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    "record_finding_bulk: write failed graph_id=%s run_id=%s "
                    "finding_id=%s err=%s",
                    graph_id,
                    run_id,
                    finding.finding_id,
                    exc,
                )
                results.append(
                    BulkItemResult(
                        id=finding.finding_id,
                        success=False,
                        error=str(exc),
                    )
                )
                failed += 1

        # Refresh the parent's evidence_count to match what's persisted.
        await self._refresh_evidence_count(graph_id, run_id, module_run_id)

        # TASK-081: emit a single delta event for the whole batch (NOT one
        # event per finding — keeps SSE traffic bounded for large batches).
        # `finding_count_delta` counts only newly-created findings; idempotent
        # replay yields delta=0.
        new_count = sum(1 for r in results if r.success and not r.already_existed)
        if new_count:
            self._publish_event(
                graph_id,
                run_id,
                "finding.recorded",
                {
                    "module_run_id": module_run_id,
                    "finding_count_delta": new_count,
                },
            )

        return BulkResponse(
            total=len(findings),
            succeeded=succeeded,
            failed=failed,
            results=results,
        )

    async def _record_one_finding(
        self,
        graph_id: str,
        run_id: str,
        module_run_id: str,
        finding: Finding,
    ) -> bool:
        """Persist a single :Finding. Returns True if the row already existed."""
        # 1. MERGE the Finding in the tenant graph.
        # Per TASK-073 Finding 1: WHERE-after-MERGE filters out foreign-tenant
        # rows so a cross-tenant `finding_id` collision can never produce a
        # [:PRODUCED] edge from our :ModuleRun to a foreign-tenant :Finding,
        # nor overwrite that foreign-tenant row's properties.
        result = await self._driver.execute_query(
            """
            MERGE (f:Finding:__Platform__ {finding_id: $finding_id})
            ON CREATE SET
                f.graph_id              = $graph_id,
                f.run_id                = $run_id,
                f.module_run_id         = $module_run_id,
                f.claim                 = $claim,
                f.raw                   = $raw,
                f.label                 = $label,
                f.confidence            = $confidence,
                f.dimensions            = $dimensions,
                f.ai_adoption_relevance = $ai_adoption_relevance,
                f.notes                 = $notes,
                f.superseded_by         = $superseded_by,
                f._created              = true
            ON MATCH SET
                f._created              = false
            WITH f
            WHERE f.graph_id = $graph_id
            MATCH (mr:ModuleRun:__Platform__ {
                graph_id: $graph_id, run_id: $run_id, module_run_id: $module_run_id
            })
            MERGE (mr)-[:PRODUCED]->(f)
            RETURN f._created AS created
            """,
            {
                "finding_id": finding.finding_id,
                "graph_id": graph_id,
                "run_id": run_id,
                "module_run_id": module_run_id,
                "claim": finding.claim,
                "raw": finding.raw,
                "label": finding.label,
                "confidence": finding.confidence,
                "dimensions": finding.dimensions,
                "ai_adoption_relevance": finding.ai_adoption_relevance,
                "notes": finding.notes,
                "superseded_by": finding.superseded_by,
            },
        )
        if not result.records:
            # Per TASK-073 Finding 1: an empty result here also indicates that
            # MERGE matched a foreign-tenant :Finding (filtered out by the
            # post-MERGE WHERE clause). Either way the write is rejected.
            raise RuntimeError(
                f"Finding write returned no records for finding_id={finding.finding_id!r} "
                f"(parent module_run missing OR cross-tenant finding_id collision)"
            )
        already_existed = not bool(result.records[0]["created"])

        # 2. If the finding cites a source, MERGE the :Source in the catalog
        # graph and create the [:CITES] edge in the tenant graph.
        if finding.source_id:
            # TASK-077: thread the full Source payload through when present.
            # When the caller supplies a nested `finding.source`, we write the
            # rich fields (url_normalized / name / type / publication_date /
            # fetch_date / language) into the catalog row on first observation
            # so cross-run search by URL works. When only `source_id` is
            # supplied (legacy callers), the catalog row keeps id-only fields.
            if finding.source is not None:
                if (
                    finding.source.source_id
                    and finding.source.source_id != finding.source_id
                ):
                    raise ValueError(
                        f"finding.source.source_id={finding.source.source_id!r} "
                        f"does not match finding.source_id={finding.source_id!r}"
                    )
                await self._merge_source_catalog(
                    finding.source_id,
                    type=finding.source.type,
                    url_normalized=finding.source.url_normalized,
                    name=finding.source.name,
                    publication_date=finding.source.publication_date,
                    fetch_date=finding.source.fetch_date,
                    language=finding.source.language,
                )
            else:
                await self._merge_source_catalog(finding.source_id)
            await self._driver.execute_query(
                """
                MATCH (f:Finding:__Platform__ {finding_id: $finding_id})
                // :Source lives in the catalog graph; we lookup by id only
                // (no graph_id filter on the source side — the catalog
                // graph_id is implicit by the source_id namespace).
                MATCH (s:Source:__Platform__ {source_id: $source_id})
                MERGE (f)-[c:CITES]->(s)
                ON CREATE SET c.quote = $quote, c.locator = $locator
                ON MATCH SET
                    c.quote   = coalesce($quote, c.quote),
                    c.locator = coalesce($locator, c.locator)
                """,
                {
                    "finding_id": finding.finding_id,
                    "source_id": finding.source_id,
                    "quote": finding.source_quote,
                    "locator": finding.source_locator,
                },
            )

        return already_existed

    async def _merge_source_catalog(self, source_id: str, **props: Any) -> None:
        """MERGE a :Source in the catalog graph (`__assessments_catalog__`).

        ON CREATE SET — writes every Source field on first observation.
        ON MATCH SET — preserves identity-bearing fields (url/name/type/lang)
        once populated; refreshes `fetch_date` (most-recent wins) and
        `publication_date` (newer-wins by ISO-8601 lex compare).
        """
        await self._driver.execute_query(
            """
            MERGE (s:Source:__Platform__ {source_id: $source_id})
            ON CREATE SET
                s.graph_id         = $catalog_graph_id,
                s.type             = $type,
                s.url_normalized   = $url_normalized,
                s.name             = $name,
                s.publication_date = $publication_date,
                s.fetch_date       = $fetch_date,
                s.language         = $language
            ON MATCH SET
                s.url_normalized   = coalesce(s.url_normalized, $url_normalized),
                s.name             = coalesce(s.name, $name),
                s.type             = coalesce(s.type, $type),
                s.language         = coalesce(s.language, $language),
                s.fetch_date       = CASE
                    WHEN $fetch_date IS NULL THEN s.fetch_date
                    ELSE $fetch_date
                END,
                s.publication_date = CASE
                    WHEN $publication_date IS NULL THEN s.publication_date
                    WHEN s.publication_date IS NULL THEN $publication_date
                    WHEN $publication_date > s.publication_date THEN $publication_date
                    ELSE s.publication_date
                END
            """,
            {
                "source_id": source_id,
                "catalog_graph_id": ASSESSMENTS_CATALOG_GRAPH_ID,
                "type": props.get("type"),
                "url_normalized": props.get("url_normalized"),
                "name": props.get("name"),
                "publication_date": props.get("publication_date"),
                "fetch_date": props.get("fetch_date"),
                "language": props.get("language"),
            },
        )

    async def _refresh_evidence_count(
        self, graph_id: str, run_id: str, module_run_id: str
    ) -> None:
        await self._driver.execute_query(
            """
            MATCH (mr:ModuleRun:__Platform__ {
                graph_id: $graph_id, run_id: $run_id, module_run_id: $module_run_id
            })-[:PRODUCED]->(f:Finding:__Platform__)
            WITH mr, count(f) AS cnt
            SET mr.evidence_count = cnt
            """,
            {
                "graph_id": graph_id,
                "run_id": run_id,
                "module_run_id": module_run_id,
            },
        )

    # =========================================================================
    # record_conflict
    # =========================================================================

    async def record_conflict(
        self, graph_id: str, run_id: str, conflict: Conflict
    ) -> bool:
        """MERGE a :Conflict + [:INVOLVES] edges to the participating findings.

        Returns True iff the conflict node was newly created.
        """
        if not graph_id:
            raise ValueError("graph_id is required")
        if conflict.run_id != run_id:
            raise ValueError(
                f"conflict.run_id={conflict.run_id!r} does not match run_id={run_id!r}"
            )

        # Per TASK-073 Finding 1: pre-MERGE existence probe + WHERE-after-MERGE.
        # MERGE on `conflict_id` alone can match a foreign-tenant :Conflict and
        # the ON MATCH SET clause would overwrite that tenant's topic/summary/
        # status. The WHERE-after-MERGE guard prevents the ON MATCH SET on
        # foreign rows because the MATCH/UPDATE pipeline is filtered before
        # the SET runs. We split into a probe + scoped writes so we can keep
        # the existing return contract.
        existing_probe = await self._driver.execute_query(
            """
            MATCH (c:Conflict:__Platform__ {conflict_id: $conflict_id})
            RETURN c.graph_id AS graph_id
            LIMIT 1
            """,
            {"conflict_id": conflict.conflict_id},
        )
        if existing_probe.records:
            other_gid = existing_probe.records[0]["graph_id"]
            if other_gid != graph_id:
                raise RuntimeError(
                    f"Conflict {conflict.conflict_id!r} already exists in a "
                    f"different tenant (graph_id={other_gid!r}); refusing to "
                    f"merge cross-tenant"
                )

        result = await self._driver.execute_query(
            """
            MERGE (c:Conflict:__Platform__ {conflict_id: $conflict_id})
            ON CREATE SET
                c.graph_id        = $graph_id,
                c.run_id          = $run_id,
                c.topic           = $topic,
                c.summary         = $summary,
                c.status          = $status,
                c.resolution      = $resolution,
                c.synthesis_note  = $synthesis_note,
                c._created        = true
            ON MATCH SET
                c.topic           = CASE WHEN c.graph_id = $graph_id THEN $topic ELSE c.topic END,
                c.summary         = CASE WHEN c.graph_id = $graph_id THEN $summary ELSE c.summary END,
                c.status          = CASE WHEN c.graph_id = $graph_id THEN $status ELSE c.status END,
                c.resolution      = CASE WHEN c.graph_id = $graph_id THEN coalesce($resolution, c.resolution) ELSE c.resolution END,
                c.synthesis_note  = CASE WHEN c.graph_id = $graph_id THEN coalesce($synthesis_note, c.synthesis_note) ELSE c.synthesis_note END,
                c._created        = false
            WITH c
            WHERE c.graph_id = $graph_id
            RETURN c._created AS created
            """,
            {
                "conflict_id": conflict.conflict_id,
                "graph_id": graph_id,
                "run_id": run_id,
                "topic": conflict.topic,
                "summary": conflict.summary,
                "status": conflict.status,
                "resolution": conflict.resolution,
                "synthesis_note": conflict.synthesis_note,
            },
        )
        if not result.records:
            # Cross-tenant collision — should have been caught by the probe
            # above, but defense-in-depth: if a concurrent write inserted the
            # row between probe and MERGE, the WHERE filters it out here too.
            raise RuntimeError(
                f"Conflict {conflict.conflict_id!r} MERGE rejected due to "
                f"cross-tenant id collision (graph_id={graph_id!r})"
            )
        created = bool(result.records[0]["created"])

        # Wire [:INVOLVES] edges to each participating finding (within the
        # same tenant graph; conflicts do not span tenants).
        for finding_id in conflict.involved_finding_ids:
            await self._driver.execute_query(
                """
                MATCH (c:Conflict:__Platform__ {conflict_id: $conflict_id})
                MATCH (f:Finding:__Platform__ {
                    graph_id: $graph_id, finding_id: $finding_id
                })
                MERGE (c)-[:INVOLVES]->(f)
                """,
                {
                    "conflict_id": conflict.conflict_id,
                    "graph_id": graph_id,
                    "finding_id": finding_id,
                },
            )

        # TASK-081: emit on every conflict record (created OR updated) — the
        # status / topic change matters to subscribers even on re-record.
        self._publish_event(
            graph_id,
            run_id,
            "conflict.recorded",
            {
                "conflict_id": conflict.conflict_id,
                "summary": conflict.summary,
                "status": conflict.status,
                "created": created,
            },
        )

        return created

    # =========================================================================
    # record_unresolved_question
    # =========================================================================

    async def record_unresolved_question(
        self,
        graph_id: str,
        run_id: str,
        module_run_id: str,
        question: UnresolvedQuestion,
    ) -> bool:
        """MERGE an :UnresolvedQuestion and [:RAISED] edge from the ModuleRun.

        Returns True iff newly created.
        """
        if not graph_id:
            raise ValueError("graph_id is required")
        if question.run_id != run_id or question.module_run_id != module_run_id:
            raise ValueError(
                "question.run_id / question.module_run_id must match the args"
            )

        # Per TASK-073 Finding 1: pre-MERGE probe + race-safe CASE-guarded
        # ON MATCH SET. MERGE on `question_id` alone could match a foreign-
        # tenant :UnresolvedQuestion and overwrite its caller-controlled text/
        # status. The probe rejects the obvious case; the CASE guards inside
        # ON MATCH SET close the narrow race window between probe and MERGE.
        existing_probe = await self._driver.execute_query(
            """
            MATCH (q:UnresolvedQuestion:__Platform__ {question_id: $question_id})
            RETURN q.graph_id AS graph_id
            LIMIT 1
            """,
            {"question_id": question.question_id},
        )
        if existing_probe.records:
            other_gid = existing_probe.records[0]["graph_id"]
            if other_gid != graph_id:
                raise RuntimeError(
                    f"UnresolvedQuestion {question.question_id!r} already exists in a "
                    f"different tenant (graph_id={other_gid!r}); refusing to "
                    f"merge cross-tenant"
                )

        result = await self._driver.execute_query(
            """
            MERGE (q:UnresolvedQuestion:__Platform__ {question_id: $question_id})
            ON CREATE SET
                q.graph_id          = $graph_id,
                q.run_id            = $run_id,
                q.module_run_id     = $module_run_id,
                q.text              = $text,
                q.suggested_module  = $suggested_module,
                q.status            = $status,
                q._created          = true
            ON MATCH SET
                q.text              = CASE WHEN q.graph_id = $graph_id THEN $text ELSE q.text END,
                q.suggested_module  = CASE WHEN q.graph_id = $graph_id THEN coalesce($suggested_module, q.suggested_module) ELSE q.suggested_module END,
                q.status            = CASE WHEN q.graph_id = $graph_id THEN $status ELSE q.status END,
                q._created          = false
            WITH q
            WHERE q.graph_id = $graph_id
            MATCH (mr:ModuleRun:__Platform__ {
                graph_id: $graph_id, run_id: $run_id, module_run_id: $module_run_id
            })
            MERGE (mr)-[:RAISED]->(q)
            RETURN q._created AS created
            """,
            {
                "question_id": question.question_id,
                "graph_id": graph_id,
                "run_id": run_id,
                "module_run_id": module_run_id,
                "text": question.text,
                "suggested_module": question.suggested_module,
                "status": question.status,
            },
        )
        created = bool(result.records and result.records[0]["created"])

        # TASK-081: emit only on first observation. Re-record (idempotent
        # replay) is silent.
        if created:
            self._publish_event(
                graph_id,
                run_id,
                "unresolved_question.raised",
                {
                    "question_id": question.question_id,
                    "module_run_id": module_run_id,
                    "suggested_module": question.suggested_module,
                },
            )

        return created

    # =========================================================================
    # persist_deliverable / persist_final_docs
    # =========================================================================

    async def persist_deliverable(
        self,
        graph_id: str,
        run_id: str,
        deliverable: Deliverable,
        *,
        content_bytes: bytes | None = None,
        mime_type: str | None = None,
        db: AsyncSession | None = None,
    ) -> bool:
        """MERGE a :Deliverable and its edges to the run (+ module_run if set).

        SPRINT-001 stored `content_uri` as an opaque string and `content_inline`
        as a property for small payloads. SPRINT-002 (TASK-082) introduces the
        Postgres-backed CAS: callers pass `content_bytes` + `mime_type`, the
        service writes the bytes via `BlobCASService.put()`, and the resulting
        canonical `blob://sha256/<hex>` URI is written to `:Deliverable.content_uri`
        (along with the computed `sha256`). Inline content via
        `Deliverable.content_inline` stays supported for sub-50KB markdown.

        Args:
            graph_id, run_id, deliverable: as before.
            content_bytes: optional raw payload to store via the CAS. When
                supplied, `db` and `mime_type` must also be supplied;
                `deliverable.content_uri` is computed from the bytes and
                any caller-supplied value on `deliverable` is ignored. Any
                caller-supplied `deliverable.sha256` must match the digest
                or it is overridden with a warning.
            mime_type: required when `content_bytes` is supplied. Stored
                in the `blob_cas.mime_type` column for `get_deliverable_content`.
            db: Postgres `AsyncSession` for the CAS write. Required when
                `content_bytes` is supplied; the caller commits the session.

        Returns True iff newly created.
        """
        if not graph_id:
            raise ValueError("graph_id is required")
        if deliverable.run_id != run_id:
            raise ValueError(
                f"deliverable.run_id={deliverable.run_id!r} != run_id={run_id!r}"
            )

        # --- CAS write (TASK-082) ---------------------------------------------
        # When the caller passes raw bytes, persist via the CAS first, then
        # overwrite `content_uri` + `sha256` on the deliverable Pydantic copy
        # we hand to Neo4j. The Pydantic model is frozen by ConfigDict(extra=
        # 'forbid') but field reassignment is allowed; we use `model_copy` to
        # avoid mutating the caller's object.
        if content_bytes is not None:
            if db is None:
                raise ValueError(
                    "db (AsyncSession) is required when content_bytes is supplied"
                )
            if not mime_type:
                raise ValueError("mime_type is required when content_bytes is supplied")
            cas_result = await self._blob_cas.put(
                db, graph_id, content_bytes, mime_type
            )
            cas_sha256 = cas_result["sha256"]
            cas_uri = cas_result["content_uri"]
            # If the caller pre-computed a sha256, verify it matches.
            if deliverable.sha256 and deliverable.sha256 != cas_sha256:
                logger.warning(
                    "persist_deliverable: caller-supplied sha256=%s does not "
                    "match computed sha256=%s; using computed value",
                    deliverable.sha256,
                    cas_sha256,
                )
            deliverable = deliverable.model_copy(
                update={
                    "content_uri": cas_uri,
                    "sha256": cas_sha256,
                }
            )
        elif mime_type is not None or db is not None:
            # Both halves of the CAS contract must travel together.
            raise ValueError(
                "mime_type / db were supplied without content_bytes — "
                "either provide all three or none"
            )

        # Per TASK-073 Finding 1: pre-MERGE probe + race-safe CASE-guarded
        # ON MATCH SET. MERGE on `deliverable_id` alone could match a foreign-
        # tenant :Deliverable and overwrite its kind/filename/content fields.
        existing_probe = await self._driver.execute_query(
            """
            MATCH (d:Deliverable:__Platform__ {deliverable_id: $deliverable_id})
            RETURN d.graph_id AS graph_id
            LIMIT 1
            """,
            {"deliverable_id": deliverable.deliverable_id},
        )
        if existing_probe.records:
            other_gid = existing_probe.records[0]["graph_id"]
            if other_gid != graph_id:
                raise RuntimeError(
                    f"Deliverable {deliverable.deliverable_id!r} already exists in a "
                    f"different tenant (graph_id={other_gid!r}); refusing to "
                    f"merge cross-tenant"
                )

        result = await self._driver.execute_query(
            """
            MERGE (d:Deliverable:__Platform__ {deliverable_id: $deliverable_id})
            ON CREATE SET
                d.graph_id        = $graph_id,
                d.run_id          = $run_id,
                d.module_run_id   = $module_run_id,
                d.kind            = $kind,
                d.filename        = $filename,
                d.ordinal         = $ordinal,
                d.content_uri     = $content_uri,
                d.content_inline  = $content_inline,
                d.sha256          = $sha256,
                d.word_count      = $word_count,
                d._created        = true
            ON MATCH SET
                d.kind            = CASE WHEN d.graph_id = $graph_id THEN $kind ELSE d.kind END,
                d.filename        = CASE WHEN d.graph_id = $graph_id THEN $filename ELSE d.filename END,
                d.ordinal         = CASE WHEN d.graph_id = $graph_id THEN $ordinal ELSE d.ordinal END,
                d.content_uri     = CASE WHEN d.graph_id = $graph_id THEN coalesce($content_uri, d.content_uri) ELSE d.content_uri END,
                d.content_inline  = CASE WHEN d.graph_id = $graph_id THEN coalesce($content_inline, d.content_inline) ELSE d.content_inline END,
                d.sha256          = CASE WHEN d.graph_id = $graph_id THEN coalesce($sha256, d.sha256) ELSE d.sha256 END,
                d.word_count      = CASE WHEN d.graph_id = $graph_id THEN coalesce($word_count, d.word_count) ELSE d.word_count END,
                d._created        = false
            WITH d
            WHERE d.graph_id = $graph_id
            MATCH (r:AssessmentRun:__Platform__ {graph_id: $graph_id, run_id: $run_id})
            MERGE (r)-[:HAS_DELIVERABLE]->(d)
            RETURN d._created AS created
            """,
            {
                "deliverable_id": deliverable.deliverable_id,
                "graph_id": graph_id,
                "run_id": run_id,
                "module_run_id": deliverable.module_run_id,
                "kind": deliverable.kind,
                "filename": deliverable.filename,
                "ordinal": deliverable.ordinal,
                "content_uri": deliverable.content_uri,
                "content_inline": deliverable.content_inline,
                "sha256": deliverable.sha256,
                "word_count": deliverable.word_count,
            },
        )
        created = bool(result.records and result.records[0]["created"])

        # Wire the optional [:PRODUCED_BY] edge from deliverable to module_run.
        if deliverable.module_run_id:
            await self._driver.execute_query(
                """
                MATCH (d:Deliverable:__Platform__ {deliverable_id: $deliverable_id})
                MATCH (mr:ModuleRun:__Platform__ {
                    graph_id: $graph_id,
                    run_id: $run_id,
                    module_run_id: $module_run_id
                })
                MERGE (mr)-[:PRODUCED_DELIVERABLE]->(d)
                """,
                {
                    "deliverable_id": deliverable.deliverable_id,
                    "graph_id": graph_id,
                    "run_id": run_id,
                    "module_run_id": deliverable.module_run_id,
                },
            )

        # TASK-081: emit on every persist (including overwrites) — frontend
        # cares about content changes, not just first observation.
        self._publish_event(
            graph_id,
            run_id,
            "deliverable.persisted",
            {
                "deliverable_id": deliverable.deliverable_id,
                "kind": deliverable.kind,
                "filename": deliverable.filename,
                "module_run_id": deliverable.module_run_id,
                "created": created,
            },
        )

        return created

    async def persist_final_docs(
        self,
        graph_id: str,
        run_id: str,
        deliverables: list[Deliverable],
        *,
        contents: list[tuple[bytes, str] | None] | None = None,
        db: AsyncSession | None = None,
    ) -> BulkResponse:
        """Bulk-persist a final 5-doc set (intro + 4 sections, typically).

        Same per-record success/failure shape as `record_finding_bulk`.

        TASK-082: pass `contents` as a list of `(content_bytes, mime_type)`
        tuples (or `None` for deliverables that already carry `content_uri` /
        `content_inline`). When supplied, the list MUST be the same length as
        `deliverables`. Each non-`None` entry triggers a CAS write before the
        Neo4j MERGE; `db` is required in that case.
        """
        if not graph_id:
            raise ValueError("graph_id is required")
        if contents is not None and len(contents) != len(deliverables):
            raise ValueError(
                f"contents length ({len(contents)}) must match deliverables "
                f"length ({len(deliverables)})"
            )
        results: list[BulkItemResult] = []
        succeeded = 0
        failed = 0
        for idx, d in enumerate(deliverables):
            try:
                cas_payload = contents[idx] if contents is not None else None
                if cas_payload is not None:
                    cb, mt = cas_payload
                    created = await self.persist_deliverable(
                        graph_id,
                        run_id,
                        d,
                        content_bytes=cb,
                        mime_type=mt,
                        db=db,
                    )
                else:
                    created = await self.persist_deliverable(graph_id, run_id, d)
                results.append(
                    BulkItemResult(
                        id=d.deliverable_id,
                        success=True,
                        already_existed=not created,
                    )
                )
                succeeded += 1
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    "persist_final_docs: write failed run_id=%s deliverable_id=%s err=%s",
                    run_id,
                    d.deliverable_id,
                    exc,
                )
                results.append(
                    BulkItemResult(id=d.deliverable_id, success=False, error=str(exc))
                )
                failed += 1
        return BulkResponse(
            total=len(deliverables),
            succeeded=succeeded,
            failed=failed,
            results=results,
        )

    # =========================================================================
    # get_deliverable_content (TASK-082 — TASK-079 picks this up at the endpoint)
    # =========================================================================

    async def get_deliverable_content(
        self,
        db: AsyncSession,
        graph_id: str,
        run_id: str,
        deliverable_id: str,
    ) -> dict[str, Any] | None:
        """Resolve a deliverable's payload, preferring CAS over inline.

        Returns a dict with the resolved bytes/text + metadata, or `None`
        when the deliverable does not exist in this tenant. Shape:

            {
                "deliverable_id": str,
                "filename": str,
                "kind": str,
                "content_uri": str | None,
                "sha256": str | None,
                "content_bytes": bytes | None,  # populated for CAS-backed
                "content_text": str | None,     # populated for inline
                "mime_type": str | None,        # populated for CAS-backed
                "size_bytes": int | None,
            }

        Resolution order:
          1. If `content_uri` is a CAS URI (`blob://sha256/<hex>`), look it
             up via `BlobCASService.get()`. Cross-tenant existence is masked
             (returns `None`).
          2. Else if `content_inline` is non-null, return it as text.
          3. Else return the deliverable row with both `content_*` fields
             `None` — caller decides what to do (likely 404).

        This method is the resolution surface TASK-079 wires into the GET
        deliverable endpoint. It lives on the service so endpoints stay thin.
        """
        if not graph_id:
            raise ValueError("graph_id is required")
        if not run_id or not deliverable_id:
            raise ValueError("run_id and deliverable_id are required")

        # 1. Fetch the deliverable metadata from Neo4j, scoped by tenant.
        meta_result = await self._driver.execute_query(
            """
            MATCH (d:Deliverable:__Platform__ {
                graph_id: $graph_id,
                run_id: $run_id,
                deliverable_id: $deliverable_id
            })
            RETURN d.deliverable_id  AS deliverable_id,
                   d.filename        AS filename,
                   d.kind            AS kind,
                   d.content_uri     AS content_uri,
                   d.content_inline  AS content_inline,
                   d.sha256          AS sha256
            LIMIT 1
            """,
            {
                "graph_id": graph_id,
                "run_id": run_id,
                "deliverable_id": deliverable_id,
            },
        )
        if not meta_result.records:
            return None
        rec = meta_result.records[0]
        out: dict[str, Any] = {
            "deliverable_id": rec["deliverable_id"],
            "filename": rec["filename"],
            "kind": rec["kind"],
            "content_uri": rec["content_uri"],
            "sha256": rec["sha256"],
            "content_bytes": None,
            "content_text": None,
            "mime_type": None,
            "size_bytes": None,
        }

        # 2. Prefer CAS resolution when the URI is shaped right.
        content_uri = rec["content_uri"]
        if is_blob_uri(content_uri):
            try:
                cas = await self._blob_cas.get(db, graph_id, content_uri)
            except InvalidBlobURIError:
                cas = None
            if cas is not None:
                out["content_bytes"] = cas["content_bytes"]
                out["mime_type"] = cas["mime_type"]
                out["size_bytes"] = cas["size_bytes"]
                return out
            # CAS row missing or cross-tenant — fall through to inline.

        # 3. Inline fallback.
        inline = rec["content_inline"]
        if inline is not None:
            out["content_text"] = inline
            out["size_bytes"] = len(inline.encode("utf-8"))
        return out

    # =========================================================================
    # finalize_run
    # =========================================================================

    async def finalize_run(self, graph_id: str, run_id: str) -> FinalizeRunResponse:
        """Server-side citation-coverage gate (STORY-026 §Critical Cypher).

        Counts the persisted artifacts and decides whether the run passes:

        - At least `MIN_DIRECT_FINDINGS_FOR_PASS` :Finding{label:'DIRECT'} rows
          attached to non-failed :ModuleRuns
        - At least `MIN_DELIVERABLES_FOR_PASS` :Deliverable rows

        Sets `:AssessmentRun.status` to `finished` on pass, `failed` on fail,
        and writes `finished_at` in both cases.
        """
        if not graph_id or not run_id:
            raise ValueError("graph_id and run_id are required")

        # 1. Counts. Per STORY-026 acceptance criteria, findings under FAILED
        # module_runs are excluded from the gate (but kept for audit).
        counts_result = await self._driver.execute_query(
            """
            MATCH (r:AssessmentRun:__Platform__ {graph_id: $graph_id, run_id: $run_id})
            OPTIONAL MATCH (r)-[:HAS_MODULE_RUN]->(mr:ModuleRun:__Platform__)
                           -[:PRODUCED]->(f:Finding:__Platform__)
                           WHERE mr.status <> 'failed'
            WITH r,
                 sum(CASE WHEN f.label = 'DIRECT' THEN 1 ELSE 0 END) AS direct_count,
                 sum(CASE WHEN f.label = 'INFERRED' THEN 1 ELSE 0 END) AS inferred_count
            OPTIONAL MATCH (r)-[:HAS_DELIVERABLE]->(d:Deliverable:__Platform__)
            WITH r, direct_count, inferred_count, count(d) AS deliverable_count
            OPTIONAL MATCH (c:Conflict:__Platform__ {graph_id: $graph_id, run_id: $run_id})
                           WHERE c.status = 'open'
            WITH r, direct_count, inferred_count, deliverable_count,
                 count(c) AS unresolved_conflict_count
            OPTIONAL MATCH (q:UnresolvedQuestion:__Platform__ {
                              graph_id: $graph_id, run_id: $run_id})
                           WHERE q.status = 'open'
            RETURN
                direct_count            AS direct_count,
                inferred_count          AS inferred_count,
                deliverable_count       AS deliverable_count,
                unresolved_conflict_count AS unresolved_conflict_count,
                count(q)                AS open_question_count
            """,
            {"graph_id": graph_id, "run_id": run_id},
        )
        if not counts_result.records:
            raise ValueError(
                f"AssessmentRun not found: graph_id={graph_id!r} run_id={run_id!r}"
            )
        rec = counts_result.records[0]
        direct = int(rec["direct_count"] or 0)
        inferred = int(rec["inferred_count"] or 0)
        deliverable = int(rec["deliverable_count"] or 0)
        unresolved_conflicts = int(rec["unresolved_conflict_count"] or 0)
        open_questions = int(rec["open_question_count"] or 0)

        # 2. Gate decision.
        failure_reasons: list[str] = []
        if direct < MIN_DIRECT_FINDINGS_FOR_PASS:
            failure_reasons.append(
                f"insufficient_direct_findings (have {direct}, "
                f"need {MIN_DIRECT_FINDINGS_FOR_PASS})"
            )
        if deliverable < MIN_DELIVERABLES_FOR_PASS:
            failure_reasons.append(
                f"insufficient_deliverables (have {deliverable}, "
                f"need {MIN_DELIVERABLES_FOR_PASS})"
            )
        passed = not failure_reasons
        final_status: RunStatus = "finished" if passed else "failed"
        finished_at = _utcnow()

        # 3. Persist final state on the run.
        await self._driver.execute_query(
            """
            MATCH (r:AssessmentRun:__Platform__ {graph_id: $graph_id, run_id: $run_id})
            SET r.status         = $status,
                r.finished_at    = datetime($finished_at),
                r.failure_reason = $failure_reason
            """,
            {
                "graph_id": graph_id,
                "run_id": run_id,
                "status": final_status,
                "finished_at": finished_at.isoformat(),
                "failure_reason": "; ".join(failure_reasons)
                if failure_reasons
                else None,
            },
        )

        logger.info(
            "finalize_run: graph_id=%s run_id=%s passed=%s direct=%d inferred=%d "
            "deliverable=%d unresolved_conflicts=%d open_questions=%d",
            graph_id,
            run_id,
            passed,
            direct,
            inferred,
            deliverable,
            unresolved_conflicts,
            open_questions,
        )

        # TASK-081: terminal event — subscribers should expect the stream to
        # quiesce after this. The SSE endpoint keeps the connection open so
        # clients can reconnect-with-cursor and still drain replay; closing
        # is a client decision.
        self._publish_event(
            graph_id,
            run_id,
            "run.finalized",
            {
                "run_id": run_id,
                "status": final_status,
                "passed": passed,
                "evidence_count_direct": direct,
                "deliverable_count": deliverable,
                "finished_at": finished_at.isoformat(),
            },
        )

        return FinalizeRunResponse(
            run_id=run_id,
            passed=passed,
            status=final_status,
            finished_at=finished_at,
            direct_finding_count=direct,
            inferred_finding_count=inferred,
            deliverable_count=deliverable,
            unresolved_conflict_count=unresolved_conflicts,
            open_question_count=open_questions,
            failure_reasons=failure_reasons,
        )

    # =========================================================================
    # Registry (ADR-019) — write-side only; reads land in SPRINT-002.
    # =========================================================================

    @staticmethod
    def _registry_target_graph_id(
        item: RegistryItem, owner_tenant_graph_id: str | None
    ) -> str:
        """Decide which graph the RegistryItem write lands in (ADR-019).

        - private  → owner's tenant graph_id (caller must supply `owner_tenant_graph_id`)
        - public   → __registry__
        - curated  → __registry__
        - yanked   → wherever the item already lives (treated like public for new writes)
        """
        if item.visibility == "private":
            if not owner_tenant_graph_id:
                raise ValueError(
                    "owner_tenant_graph_id is required for private RegistryItem writes"
                )
            return owner_tenant_graph_id
        return REGISTRY_CATALOG_GRAPH_ID

    async def persist_registry_item(
        self,
        item: RegistryItem,
        owner_tenant_graph_id: str | None = None,
    ) -> bool:
        """Write a RegistryItem to its visibility-appropriate graph (ADR-019).

        Returns True iff newly created.

        Per TASK-073 Finding 1 (TASK-069): public/yanked items are owner-only
        on UPDATE — a non-owner attempting to overwrite an existing item
        raises :class:`RegistryOwnershipError`. Curated writes (admin-gated at
        the endpoint) are exempt, since platform admins legitimately update
        items owned by others. Per TASK-073 Finding 1 (TASK-068): the MERGE
        is scoped by `(graph_id, item_id)` semantics — a cross-tenant item_id
        collision raises a clear error rather than silently overwriting.
        """
        target_graph_id = self._registry_target_graph_id(item, owner_tenant_graph_id)
        # The item's `graph_id` field must match the visibility-resolved target.
        if item.graph_id != target_graph_id:
            raise ValueError(
                f"RegistryItem.graph_id={item.graph_id!r} does not match "
                f"visibility-resolved target={target_graph_id!r} "
                f"(visibility={item.visibility!r})"
            )

        # Pre-MERGE existence + ownership + tenancy probe.
        # Per TASK-073 Finding 1 (TASK-069): the endpoint's `verify_graph_access`
        # gates "can write to the catalog" but not "can write *this* item".
        # ADR-019 mandates owner-only writes for `public` and `yanked`. A
        # non-owner attempting to update an existing item is rejected with
        # `RegistryOwnershipError`, which the endpoint maps to HTTP 403.
        existing_probe = await self._driver.execute_query(
            """
            MATCH (ri:RegistryItem:__Platform__ {item_id: $item_id})
            RETURN ri.graph_id      AS graph_id,
                   ri.owner_user_id AS owner_user_id,
                   ri.visibility    AS visibility
            LIMIT 1
            """,
            {"item_id": item.item_id},
        )
        if existing_probe.records:
            rec = existing_probe.records[0]
            existing_graph_id = rec["graph_id"]
            existing_owner = rec["owner_user_id"]
            existing_visibility = rec["visibility"]

            # Tenant-isolation guard (TASK-073 Finding 1, TASK-068): the
            # existing row must live in the same graph we're writing to.
            # Otherwise this is a cross-tenant id collision attempt.
            if existing_graph_id != target_graph_id:
                raise RuntimeError(
                    f"RegistryItem {item.item_id!r} already exists in a "
                    f"different graph (existing graph_id={existing_graph_id!r}, "
                    f"target={target_graph_id!r}); refusing to merge cross-tenant"
                )

            # Ownership guard (TASK-073 Finding 1, TASK-069): public/yanked
            # writes are owner-only. Curated writes are admin-gated at the
            # endpoint and may legitimately update non-owned items.
            if item.visibility in ("public", "yanked") or existing_visibility in (
                "public",
                "yanked",
            ):
                # Only the owner of the existing item may update it.
                if existing_owner and existing_owner != item.owner_user_id:
                    raise RegistryOwnershipError(
                        f"RegistryItem {item.item_id!r} is owned by another "
                        f"user (existing owner_user_id={existing_owner!r}, "
                        f"caller={item.owner_user_id!r}); only the owner may "
                        f"update a public/yanked item (curated writes require "
                        f"platform admin)"
                    )

        result = await self._driver.execute_query(
            """
            MERGE (ri:RegistryItem:__Platform__ {item_id: $item_id})
            ON CREATE SET
                ri.graph_id      = $graph_id,
                ri.kind          = $kind,
                ri.slug          = $slug,
                ri.version       = $version,
                ri.visibility    = $visibility,
                ri.owner_user_id = $owner_user_id,
                ri.name          = $name,
                ri.description   = $description,
                ri.content_uri   = $content_uri,
                ri.sha256        = $sha256,
                ri.created_at    = datetime($created_at),
                ri.yanked_at     = $yanked_at,
                ri._created      = true
            ON MATCH SET
                ri.name          = CASE WHEN ri.graph_id = $graph_id THEN $name ELSE ri.name END,
                ri.description   = CASE WHEN ri.graph_id = $graph_id THEN coalesce($description, ri.description) ELSE ri.description END,
                ri.content_uri   = CASE WHEN ri.graph_id = $graph_id THEN coalesce($content_uri, ri.content_uri) ELSE ri.content_uri END,
                ri.sha256        = CASE WHEN ri.graph_id = $graph_id THEN coalesce($sha256, ri.sha256) ELSE ri.sha256 END,
                ri.visibility    = CASE WHEN ri.graph_id = $graph_id THEN $visibility ELSE ri.visibility END,
                ri.yanked_at     = CASE WHEN ri.graph_id = $graph_id THEN $yanked_at ELSE ri.yanked_at END,
                ri._created      = false
            WITH ri
            WHERE ri.graph_id = $graph_id
            RETURN ri._created AS created
            """,
            {
                "item_id": item.item_id,
                "graph_id": target_graph_id,
                "kind": item.kind,
                "slug": item.slug,
                "version": item.version,
                "visibility": item.visibility,
                "owner_user_id": item.owner_user_id,
                "name": item.name,
                "description": item.description,
                "content_uri": item.content_uri,
                "sha256": item.sha256,
                "created_at": (item.created_at or _utcnow()).isoformat(),
                "yanked_at": item.yanked_at.isoformat() if item.yanked_at else None,
            },
        )
        if not result.records:
            # The probe should have caught the cross-tenant case, but a
            # concurrent write could still race us here.
            raise RuntimeError(
                f"RegistryItem {item.item_id!r} MERGE rejected due to "
                f"cross-tenant collision (target graph_id={target_graph_id!r})"
            )
        return bool(result.records[0]["created"])

    # =========================================================================
    # READ PATH (TASK-079, SPRINT-002)
    # =========================================================================
    #
    # Read methods follow the same `graph_id`-first rule as the write methods.
    # Every Cypher statement is scoped to `graph_id` (the JWT-derived tenant)
    # so cross-tenant queries are impossible — even with a malformed cursor
    # carrying a foreign `run_id`, the tenant filter rules out the foreign row.
    #
    # Catalog hydration (`:Source`, `:Module`) is performed at the application
    # layer per ADR-018 §Tenancy: the per-tenant Cypher returns ids, then a
    # second Cypher fetches the catalog rows by id and we zip them in Python.
    # This keeps each query single-tenant from Neo4j's perspective.
    #
    # Pagination uses opaque offsets at the service boundary; the endpoint
    # layer is responsible for encoding/decoding the opaque cursor string.

    @staticmethod
    def _neo_datetime(value: Any) -> datetime | None:
        """Coerce a Neo4j datetime / ISO string to a tz-aware Python datetime."""
        if value is None:
            return None
        to_native = getattr(value, "to_native", None)
        if callable(to_native):
            native = to_native()
            if native.tzinfo is None:
                native = native.replace(tzinfo=UTC)
            return native
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=UTC)
            return value
        if isinstance(value, str):
            try:
                normalized = (
                    value.replace("Z", "+00:00") if value.endswith("Z") else value
                )
                parsed = datetime.fromisoformat(normalized)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                return parsed
            except ValueError:
                return None
        return None

    @staticmethod
    def _cli_flags(value: Any) -> dict[str, Any] | None:
        """Decode `cli_flags` from whatever shape the write path landed it in."""
        if value is None or value == "":
            return None
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                import json as _json

                return _json.loads(value)
            except (ValueError, TypeError):
                return {"_raw": value}
        return None

    # ─── list_runs ─────────────────────────────────────────────────────────

    async def list_runs(
        self,
        graph_id: str,
        *,
        status: RunStatus | None = None,
        subject_slug: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[RunSummary], bool]:
        """Paginated runs list for the tenant.

        Returns `(rows, has_more)`. `has_more` is computed by fetching
        `limit + 1` rows and trimming; the endpoint layer translates this
        into an opaque cursor.
        """
        if not graph_id:
            raise ValueError("graph_id is required")
        params: dict[str, Any] = {
            "graph_id": graph_id,
            "limit": limit + 1,
            "offset": offset,
        }
        filters = ["r.graph_id = $graph_id"]
        if status is not None:
            filters.append("r.status = $status")
            params["status"] = status

        subject_join = ""
        if subject_slug is not None:
            subject_join = (
                "MATCH (s:Subject:__Platform__ {graph_id: $graph_id, slug: $subject_slug})"
                " WHERE r.subject_id = s.subject_id\n"
            )
            params["subject_slug"] = subject_slug

        cypher = (
            "MATCH (r:AssessmentRun:__Platform__)\n"
            f"WHERE {' AND '.join(filters)}\n"
            f"{subject_join}"
            "OPTIONAL MATCH (r)-[:HAS_MODULE_RUN]->(mr:ModuleRun:__Platform__)\n"
            "WITH r,\n"
            "     count(mr) AS module_run_total,\n"
            "     sum(CASE WHEN mr.status = 'finished' THEN 1 ELSE 0 END) AS module_run_done,\n"
            "     sum(CASE WHEN mr.status = 'failed'   THEN 1 ELSE 0 END) AS module_run_failed\n"
            "OPTIONAL MATCH (subj:Subject:__Platform__ {graph_id: $graph_id, subject_id: r.subject_id})\n"
            "RETURN\n"
            "    r.run_id                AS run_id,\n"
            "    r.template_id           AS template_id,\n"
            "    r.subject_id            AS subject_id,\n"
            "    r.status                AS status,\n"
            "    r.started_at            AS started_at,\n"
            "    r.finished_at           AS finished_at,\n"
            "    r.orchestrator_last_seen AS orchestrator_last_seen,\n"
            "    subj.slug               AS subject_slug,\n"
            "    subj.name               AS subject_name,\n"
            "    module_run_total        AS module_run_total,\n"
            "    module_run_done         AS module_run_done,\n"
            "    module_run_failed       AS module_run_failed\n"
            "ORDER BY r.started_at DESC, r.run_id ASC\n"
            "SKIP $offset LIMIT $limit\n"
        )
        result = await self._driver.execute_query(cypher, params)
        recs = result.records
        has_more = len(recs) > limit
        page_recs = recs[:limit]

        template_ids = list(
            {rec["template_id"] for rec in page_recs if rec["template_id"]}
        )
        template_slugs = await self._fetch_template_slugs(template_ids)

        rows: list[RunSummary] = []
        for rec in page_recs:
            rows.append(
                RunSummary(
                    run_id=rec["run_id"],
                    template_id=rec["template_id"],
                    template_slug=template_slugs.get(rec["template_id"]),
                    subject_id=rec["subject_id"],
                    subject_slug=rec["subject_slug"],
                    subject_name=rec["subject_name"],
                    status=rec["status"],
                    started_at=self._neo_datetime(rec["started_at"]),
                    finished_at=self._neo_datetime(rec["finished_at"]),
                    orchestrator_last_seen=self._neo_datetime(
                        rec["orchestrator_last_seen"]
                    ),
                    module_run_total=int(rec["module_run_total"] or 0),
                    module_run_done=int(rec["module_run_done"] or 0),
                    module_run_failed=int(rec["module_run_failed"] or 0),
                )
            )
        return rows, has_more

    async def _fetch_template_slugs(self, template_ids: list[str]) -> dict[str, str]:
        """Batch-fetch `template_id → slug` from the catalog graph."""
        if not template_ids:
            return {}
        result = await self._driver.execute_query(
            """
            MATCH (t:AssessmentTemplate:__Platform__)
            WHERE t.template_id IN $ids
            RETURN t.template_id AS template_id, t.slug AS slug
            """,
            {"ids": template_ids},
        )
        return {
            rec["template_id"]: rec["slug"] for rec in result.records if rec["slug"]
        }

    # ─── get_run_detail ────────────────────────────────────────────────────

    async def get_run_detail(self, graph_id: str, run_id: str) -> RunDetail | None:
        """Single-run rollup for the run-detail page.

        Returns `None` when the run does not exist in this tenant. Does NOT
        distinguish "not in this tenant" from "does not exist" — endpoint
        translates both to 404 so we cannot leak cross-tenant existence.
        """
        if not graph_id or not run_id:
            raise ValueError("graph_id and run_id are required")
        result = await self._driver.execute_query(
            """
            MATCH (r:AssessmentRun:__Platform__ {graph_id: $graph_id, run_id: $run_id})
            OPTIONAL MATCH (r)-[:HAS_MODULE_RUN]->(mr:ModuleRun:__Platform__)
            WITH r,
                 count(mr) AS module_run_total,
                 sum(CASE WHEN mr.status = 'finished' THEN 1 ELSE 0 END) AS module_run_done,
                 sum(CASE WHEN mr.status = 'failed'   THEN 1 ELSE 0 END) AS module_run_failed
            OPTIONAL MATCH (r)-[:HAS_MODULE_RUN]->(mr2:ModuleRun:__Platform__)
                  -[:PRODUCED]->(f:Finding:__Platform__)
                  WHERE mr2.status <> 'failed'
            WITH r, module_run_total, module_run_done, module_run_failed,
                 count(f) AS finding_count
            OPTIONAL MATCH (c:Conflict:__Platform__ {graph_id: $graph_id, run_id: $run_id})
            WITH r, module_run_total, module_run_done, module_run_failed,
                 finding_count, count(c) AS conflict_count
            OPTIONAL MATCH (q:UnresolvedQuestion:__Platform__ {
                graph_id: $graph_id, run_id: $run_id, status: 'open'
            })
            WITH r, module_run_total, module_run_done, module_run_failed,
                 finding_count, conflict_count, count(q) AS open_question_count
            OPTIONAL MATCH (r)-[:HAS_DELIVERABLE]->(d:Deliverable:__Platform__)
            WITH r, module_run_total, module_run_done, module_run_failed,
                 finding_count, conflict_count, open_question_count,
                 count(d) AS deliverable_count
            OPTIONAL MATCH (subj:Subject:__Platform__ {
                graph_id: $graph_id, subject_id: r.subject_id
            })
            RETURN
                r.run_id                AS run_id,
                r.graph_id              AS graph_id,
                r.template_id           AS template_id,
                r.subject_id            AS subject_id,
                r.status                AS status,
                r.started_at            AS started_at,
                r.finished_at           AS finished_at,
                r.orchestrator_last_seen AS orchestrator_last_seen,
                r.cli_flags             AS cli_flags,
                r.failure_reason        AS failure_reason,
                subj.slug               AS subject_slug,
                subj.name               AS subject_name,
                module_run_total        AS module_run_total,
                module_run_done         AS module_run_done,
                module_run_failed       AS module_run_failed,
                finding_count           AS finding_count,
                conflict_count          AS conflict_count,
                open_question_count     AS open_question_count,
                deliverable_count       AS deliverable_count
            """,
            {"graph_id": graph_id, "run_id": run_id},
        )
        if not result.records:
            return None
        rec = result.records[0]
        template_slugs = await self._fetch_template_slugs([rec["template_id"]])
        return RunDetail(
            run_id=rec["run_id"],
            graph_id=rec["graph_id"],
            template_id=rec["template_id"],
            template_slug=template_slugs.get(rec["template_id"]),
            subject_id=rec["subject_id"],
            subject_slug=rec["subject_slug"],
            subject_name=rec["subject_name"],
            status=rec["status"],
            started_at=self._neo_datetime(rec["started_at"]),
            finished_at=self._neo_datetime(rec["finished_at"]),
            orchestrator_last_seen=self._neo_datetime(rec["orchestrator_last_seen"]),
            cli_flags=self._cli_flags(rec["cli_flags"]),
            failure_reason=rec["failure_reason"],
            module_run_total=int(rec["module_run_total"] or 0),
            module_run_done=int(rec["module_run_done"] or 0),
            module_run_failed=int(rec["module_run_failed"] or 0),
            finding_count=int(rec["finding_count"] or 0),
            conflict_count=int(rec["conflict_count"] or 0),
            open_question_count=int(rec["open_question_count"] or 0),
            deliverable_count=int(rec["deliverable_count"] or 0),
        )

    # ─── get_wave_status ───────────────────────────────────────────────────

    async def get_wave_status(
        self, graph_id: str, run_id: str, wave: int
    ) -> WaveStatusResponse | None:
        """Per-wave done/failed/total + per-module status list."""
        if not graph_id or not run_id:
            raise ValueError("graph_id and run_id are required")
        if wave < 1:
            raise ValueError("wave must be >= 1")

        run_check = await self._driver.execute_query(
            """
            MATCH (r:AssessmentRun:__Platform__ {graph_id: $graph_id, run_id: $run_id})
            RETURN r.run_id AS id
            LIMIT 1
            """,
            {"graph_id": graph_id, "run_id": run_id},
        )
        if not run_check.records:
            return None

        rows_result = await self._driver.execute_query(
            """
            MATCH (mr:ModuleRun:__Platform__ {graph_id: $graph_id, run_id: $run_id, wave: $wave})
            RETURN
                mr.module_run_id      AS module_run_id,
                mr.module_id          AS module_id,
                mr.status             AS status,
                mr.started_at         AS started_at,
                mr.finished_at        AS finished_at,
                mr.last_heartbeat_at  AS last_heartbeat_at,
                mr.evidence_count     AS evidence_count,
                mr.failure_reason     AS failure_reason
            ORDER BY mr.module_run_id ASC
            """,
            {"graph_id": graph_id, "run_id": run_id, "wave": wave},
        )
        recs = rows_result.records

        module_ids = list({rec["module_id"] for rec in recs if rec["module_id"]})
        module_info = await self._fetch_modules_by_ids(module_ids)

        modules: list[WaveModuleStatus] = []
        counters: dict[str, int] = {
            "planned": 0,
            "running": 0,
            "finished": 0,
            "failed": 0,
            "cancelled": 0,
        }
        for rec in recs:
            status = rec["status"]
            if status in counters:
                counters[status] += 1
            info = module_info.get(rec["module_id"], {})
            modules.append(
                WaveModuleStatus(
                    module_run_id=rec["module_run_id"],
                    module_id=rec["module_id"],
                    module_slug=info.get("slug"),
                    module_name=info.get("name"),
                    module_kind=info.get("kind"),
                    status=status,
                    started_at=self._neo_datetime(rec["started_at"]),
                    finished_at=self._neo_datetime(rec["finished_at"]),
                    last_heartbeat_at=self._neo_datetime(rec["last_heartbeat_at"]),
                    evidence_count=int(rec["evidence_count"] or 0),
                    failure_reason=rec["failure_reason"],
                )
            )

        return WaveStatusResponse(
            run_id=run_id,
            wave=wave,
            total=len(modules),
            done=counters["finished"],
            failed=counters["failed"],
            running=counters["running"],
            planned=counters["planned"],
            cancelled=counters["cancelled"],
            modules=modules,
        )

    async def _fetch_modules_by_ids(
        self, module_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Catalog-graph lookup of `:Module` rows by id (ADR-018 §Tenancy)."""
        if not module_ids:
            return {}
        result = await self._driver.execute_query(
            """
            MATCH (m:Module:__Platform__)
            WHERE m.module_id IN $ids
            RETURN
                m.module_id   AS module_id,
                m.slug        AS slug,
                m.name        AS name,
                m.wave        AS wave,
                m.kind        AS kind,
                m.ordinal     AS ordinal,
                m.agent_id    AS agent_id,
                m.description AS description,
                m.template_id AS template_id
            """,
            {"ids": module_ids},
        )
        out: dict[str, dict[str, Any]] = {}
        for rec in result.records:
            mid = rec["module_id"]
            if not mid:
                continue
            out[mid] = {
                "slug": rec["slug"],
                "name": rec["name"],
                "wave": rec["wave"],
                "kind": rec["kind"],
                "ordinal": rec["ordinal"],
                "agent_id": rec["agent_id"],
                "description": rec["description"],
                "template_id": rec["template_id"],
            }
        return out

    # ─── list_module_runs ──────────────────────────────────────────────────

    async def list_module_runs(
        self,
        graph_id: str,
        run_id: str,
        *,
        status: RunStatus | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[ModuleRunRow], bool] | None:
        """All `:ModuleRun` rows for a run, joined to `:Module` (catalog-hydrated)."""
        if not graph_id or not run_id:
            raise ValueError("graph_id and run_id are required")

        run_check = await self._driver.execute_query(
            """
            MATCH (r:AssessmentRun:__Platform__ {graph_id: $graph_id, run_id: $run_id})
            RETURN r.run_id AS id LIMIT 1
            """,
            {"graph_id": graph_id, "run_id": run_id},
        )
        if not run_check.records:
            return None

        filters = ["mr.graph_id = $graph_id", "mr.run_id = $run_id"]
        params: dict[str, Any] = {
            "graph_id": graph_id,
            "run_id": run_id,
            "offset": offset,
            "limit": limit + 1,
        }
        if status is not None:
            filters.append("mr.status = $status")
            params["status"] = status

        cypher = (
            "MATCH (mr:ModuleRun:__Platform__)\n"
            f"WHERE {' AND '.join(filters)}\n"
            "RETURN\n"
            "    mr.module_run_id      AS module_run_id,\n"
            "    mr.run_id             AS run_id,\n"
            "    mr.module_id          AS module_id,\n"
            "    mr.wave               AS wave,\n"
            "    mr.status             AS status,\n"
            "    mr.started_at         AS started_at,\n"
            "    mr.finished_at        AS finished_at,\n"
            "    mr.last_heartbeat_at  AS last_heartbeat_at,\n"
            "    mr.evidence_count     AS evidence_count,\n"
            "    mr.deliverable_path   AS deliverable_path,\n"
            "    mr.failure_reason     AS failure_reason\n"
            "ORDER BY mr.wave ASC, mr.module_run_id ASC\n"
            "SKIP $offset LIMIT $limit\n"
        )
        result = await self._driver.execute_query(cypher, params)
        recs = result.records
        has_more = len(recs) > limit
        page = recs[:limit]

        module_ids = list({rec["module_id"] for rec in page if rec["module_id"]})
        module_info = await self._fetch_modules_by_ids(module_ids)

        rows: list[ModuleRunRow] = []
        for rec in page:
            info = module_info.get(rec["module_id"], {})
            rows.append(
                ModuleRunRow(
                    module_run_id=rec["module_run_id"],
                    run_id=rec["run_id"],
                    module_id=rec["module_id"],
                    module_slug=info.get("slug"),
                    module_name=info.get("name"),
                    module_kind=info.get("kind"),
                    module_wave=info.get("wave"),
                    module_agent_id=info.get("agent_id"),
                    wave=int(rec["wave"]),
                    status=rec["status"],
                    started_at=self._neo_datetime(rec["started_at"]),
                    finished_at=self._neo_datetime(rec["finished_at"]),
                    last_heartbeat_at=self._neo_datetime(rec["last_heartbeat_at"]),
                    evidence_count=int(rec["evidence_count"] or 0),
                    deliverable_path=rec["deliverable_path"],
                    failure_reason=rec["failure_reason"],
                )
            )
        return rows, has_more

    # ─── list_findings ─────────────────────────────────────────────────────

    async def list_findings(
        self,
        graph_id: str,
        run_id: str,
        *,
        module_slug: str | None = None,
        dimension: str | None = None,
        label: str | None = None,
        min_confidence: float | None = None,
        source_type: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[FindingRow], bool] | None:
        """Findings table with optional filters + `:Source` hydration.

        ADR-018 §Tenancy: per-tenant Cypher pulls findings + denormalized
        `source_id`; a second Cypher fetches catalog `:Source` rows by id.
        Each query stays single-tenant.

        The `source_type` filter is applied AFTER catalog hydration; it
        cannot push down because :Source lives in a separate graph partition.
        """
        if not graph_id or not run_id:
            raise ValueError("graph_id and run_id are required")

        run_check = await self._driver.execute_query(
            """
            MATCH (r:AssessmentRun:__Platform__ {graph_id: $graph_id, run_id: $run_id})
            RETURN r.run_id AS id LIMIT 1
            """,
            {"graph_id": graph_id, "run_id": run_id},
        )
        if not run_check.records:
            return None

        filters = ["f.graph_id = $graph_id", "f.run_id = $run_id"]
        params: dict[str, Any] = {
            "graph_id": graph_id,
            "run_id": run_id,
            "offset": offset,
            "limit": limit + 1,
        }
        if label is not None:
            filters.append("f.label = $label")
            params["label"] = label
        if min_confidence is not None:
            filters.append("f.confidence >= $min_confidence")
            params["min_confidence"] = float(min_confidence)
        if dimension is not None:
            filters.append("$dimension IN f.dimensions")
            params["dimension"] = dimension

        module_slug_join = ""
        if module_slug is not None:
            module_id_lookup = await self._driver.execute_query(
                """
                MATCH (m:Module:__Platform__ {slug: $slug})
                RETURN m.module_id AS module_id
                """,
                {"slug": module_slug},
            )
            module_ids = [r["module_id"] for r in module_id_lookup.records]
            if not module_ids:
                return [], False
            params["module_ids"] = module_ids
            module_slug_join = (
                "MATCH (mr:ModuleRun:__Platform__ {graph_id: $graph_id})"
                " WHERE mr.module_run_id = f.module_run_id"
                " AND mr.module_id IN $module_ids\n"
            )

        cypher = (
            "MATCH (f:Finding:__Platform__)\n"
            f"WHERE {' AND '.join(filters)}\n"
            f"{module_slug_join}"
            "OPTIONAL MATCH (f)-[c:CITES]->(:Source)\n"
            "RETURN\n"
            "    f.finding_id           AS finding_id,\n"
            "    f.run_id               AS run_id,\n"
            "    f.module_run_id        AS module_run_id,\n"
            "    f.claim                AS claim,\n"
            "    f.raw                  AS raw,\n"
            "    f.label                AS label,\n"
            "    f.confidence           AS confidence,\n"
            "    f.dimensions           AS dimensions,\n"
            "    f.ai_adoption_relevance AS ai_adoption_relevance,\n"
            "    f.notes                AS notes,\n"
            "    f.superseded_by        AS superseded_by,\n"
            "    f.source_id            AS source_id,\n"
            "    c.quote                AS source_quote,\n"
            "    c.locator              AS source_locator\n"
            "ORDER BY f.confidence DESC, f.finding_id ASC\n"
            "SKIP $offset LIMIT $limit\n"
        )
        result = await self._driver.execute_query(cypher, params)
        recs = result.records
        has_more = len(recs) > limit
        page = recs[:limit]

        module_run_ids = list(
            {rec["module_run_id"] for rec in page if rec["module_run_id"]}
        )
        module_run_to_module = await self._fetch_module_run_to_module_map(
            graph_id, module_run_ids
        )
        module_lookup = await self._fetch_modules_by_ids(
            list(set(module_run_to_module.values()))
        )

        source_ids = list({rec["source_id"] for rec in page if rec["source_id"]})
        sources = await self._fetch_sources_by_ids(source_ids)

        rows: list[FindingRow] = []
        for rec in page:
            module_id = module_run_to_module.get(rec["module_run_id"])
            module_meta = module_lookup.get(module_id, {}) if module_id else {}
            src = sources.get(rec["source_id"]) if rec["source_id"] else None
            if source_type is not None:
                if src is None or src.type != source_type:
                    continue
            rows.append(
                FindingRow(
                    finding_id=rec["finding_id"],
                    run_id=rec["run_id"],
                    module_run_id=rec["module_run_id"],
                    module_slug=module_meta.get("slug"),
                    module_name=module_meta.get("name"),
                    claim=rec["claim"],
                    raw=rec["raw"],
                    label=rec["label"],
                    confidence=float(rec["confidence"] or 0.0),
                    dimensions=list(rec["dimensions"] or []),
                    ai_adoption_relevance=(
                        float(rec["ai_adoption_relevance"])
                        if rec["ai_adoption_relevance"] is not None
                        else None
                    ),
                    notes=rec["notes"],
                    superseded_by=rec["superseded_by"],
                    source_id=rec["source_id"],
                    source_quote=rec["source_quote"],
                    source_locator=rec["source_locator"],
                    source=src,
                )
            )
        return rows, has_more

    async def _fetch_module_run_to_module_map(
        self, graph_id: str, module_run_ids: list[str]
    ) -> dict[str, str]:
        """`module_run_id → module_id` lookup, single-tenant."""
        if not module_run_ids:
            return {}
        result = await self._driver.execute_query(
            """
            MATCH (mr:ModuleRun:__Platform__)
            WHERE mr.graph_id = $graph_id AND mr.module_run_id IN $ids
            RETURN mr.module_run_id AS module_run_id, mr.module_id AS module_id
            """,
            {"graph_id": graph_id, "ids": module_run_ids},
        )
        return {
            rec["module_run_id"]: rec["module_id"]
            for rec in result.records
            if rec["module_id"]
        }

    async def _fetch_sources_by_ids(self, source_ids: list[str]) -> dict[str, Source]:
        """Catalog-graph lookup of `:Source` rows by id (ADR-018 §Tenancy)."""
        if not source_ids:
            return {}
        result = await self._driver.execute_query(
            """
            MATCH (s:Source:__Platform__)
            WHERE s.source_id IN $ids
            RETURN
                s.source_id        AS source_id,
                s.type             AS type,
                s.url_normalized   AS url_normalized,
                s.name             AS name,
                s.publication_date AS publication_date,
                s.fetch_date       AS fetch_date,
                s.language         AS language
            """,
            {"ids": source_ids},
        )
        out: dict[str, Source] = {}
        for rec in result.records:
            out[rec["source_id"]] = Source(
                source_id=rec["source_id"],
                type=rec["type"],
                url_normalized=rec["url_normalized"],
                name=rec["name"],
                publication_date=rec["publication_date"],
                fetch_date=rec["fetch_date"],
                language=rec["language"],
            )
        return out

    # ─── list_conflicts ────────────────────────────────────────────────────

    async def list_conflicts(
        self,
        graph_id: str,
        run_id: str,
        *,
        status: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[ConflictRow], bool] | None:
        if not graph_id or not run_id:
            raise ValueError("graph_id and run_id are required")

        run_check = await self._driver.execute_query(
            """
            MATCH (r:AssessmentRun:__Platform__ {graph_id: $graph_id, run_id: $run_id})
            RETURN r.run_id AS id LIMIT 1
            """,
            {"graph_id": graph_id, "run_id": run_id},
        )
        if not run_check.records:
            return None

        filters = ["c.graph_id = $graph_id", "c.run_id = $run_id"]
        params: dict[str, Any] = {
            "graph_id": graph_id,
            "run_id": run_id,
            "offset": offset,
            "limit": limit + 1,
        }
        if status is not None:
            filters.append("c.status = $status")
            params["status"] = status

        cypher = (
            "MATCH (c:Conflict:__Platform__)\n"
            f"WHERE {' AND '.join(filters)}\n"
            "OPTIONAL MATCH (c)-[:INVOLVES]->(f:Finding:__Platform__)\n"
            "WITH c, collect(f.finding_id) AS involved_finding_ids\n"
            "RETURN\n"
            "    c.conflict_id     AS conflict_id,\n"
            "    c.run_id          AS run_id,\n"
            "    c.topic           AS topic,\n"
            "    c.summary         AS summary,\n"
            "    c.status          AS status,\n"
            "    c.resolution      AS resolution,\n"
            "    c.synthesis_note  AS synthesis_note,\n"
            "    involved_finding_ids AS involved_finding_ids\n"
            "ORDER BY c.conflict_id ASC\n"
            "SKIP $offset LIMIT $limit\n"
        )
        result = await self._driver.execute_query(cypher, params)
        recs = result.records
        has_more = len(recs) > limit
        page = recs[:limit]

        rows: list[ConflictRow] = []
        for rec in page:
            rows.append(
                ConflictRow(
                    conflict_id=rec["conflict_id"],
                    run_id=rec["run_id"],
                    topic=rec["topic"],
                    summary=rec["summary"],
                    status=rec["status"],
                    resolution=rec["resolution"],
                    synthesis_note=rec["synthesis_note"],
                    involved_finding_ids=[
                        fid for fid in (rec["involved_finding_ids"] or []) if fid
                    ],
                )
            )
        return rows, has_more

    # ─── list_unresolved_questions ─────────────────────────────────────────

    async def list_unresolved_questions(
        self,
        graph_id: str,
        run_id: str,
        *,
        status: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[UnresolvedQuestionRow], bool] | None:
        if not graph_id or not run_id:
            raise ValueError("graph_id and run_id are required")

        run_check = await self._driver.execute_query(
            """
            MATCH (r:AssessmentRun:__Platform__ {graph_id: $graph_id, run_id: $run_id})
            RETURN r.run_id AS id LIMIT 1
            """,
            {"graph_id": graph_id, "run_id": run_id},
        )
        if not run_check.records:
            return None

        filters = ["q.graph_id = $graph_id", "q.run_id = $run_id"]
        params: dict[str, Any] = {
            "graph_id": graph_id,
            "run_id": run_id,
            "offset": offset,
            "limit": limit + 1,
        }
        if status is not None:
            filters.append("q.status = $status")
            params["status"] = status

        cypher = (
            "MATCH (q:UnresolvedQuestion:__Platform__)\n"
            f"WHERE {' AND '.join(filters)}\n"
            "RETURN\n"
            "    q.question_id      AS question_id,\n"
            "    q.run_id           AS run_id,\n"
            "    q.module_run_id    AS module_run_id,\n"
            "    q.text             AS text,\n"
            "    q.suggested_module AS suggested_module,\n"
            "    q.status           AS status\n"
            "ORDER BY q.question_id ASC\n"
            "SKIP $offset LIMIT $limit\n"
        )
        result = await self._driver.execute_query(cypher, params)
        recs = result.records
        has_more = len(recs) > limit
        page = recs[:limit]

        module_run_ids = list(
            {rec["module_run_id"] for rec in page if rec["module_run_id"]}
        )
        module_run_to_module = await self._fetch_module_run_to_module_map(
            graph_id, module_run_ids
        )
        module_lookup = await self._fetch_modules_by_ids(
            list(set(module_run_to_module.values()))
        )

        rows: list[UnresolvedQuestionRow] = []
        for rec in page:
            module_id = module_run_to_module.get(rec["module_run_id"])
            module_meta = module_lookup.get(module_id, {}) if module_id else {}
            rows.append(
                UnresolvedQuestionRow(
                    question_id=rec["question_id"],
                    run_id=rec["run_id"],
                    module_run_id=rec["module_run_id"],
                    module_slug=module_meta.get("slug"),
                    text=rec["text"],
                    suggested_module=rec["suggested_module"],
                    status=rec["status"],
                )
            )
        return rows, has_more

    # ─── list_deliverables ─────────────────────────────────────────────────

    async def list_deliverables(
        self,
        graph_id: str,
        run_id: str,
        *,
        kind: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[DeliverableRow], bool] | None:
        """Deliverable metadata for a run; content fetched via a separate endpoint."""
        if not graph_id or not run_id:
            raise ValueError("graph_id and run_id are required")

        run_check = await self._driver.execute_query(
            """
            MATCH (r:AssessmentRun:__Platform__ {graph_id: $graph_id, run_id: $run_id})
            RETURN r.run_id AS id LIMIT 1
            """,
            {"graph_id": graph_id, "run_id": run_id},
        )
        if not run_check.records:
            return None

        filters = ["d.graph_id = $graph_id", "d.run_id = $run_id"]
        params: dict[str, Any] = {
            "graph_id": graph_id,
            "run_id": run_id,
            "offset": offset,
            "limit": limit + 1,
        }
        if kind is not None:
            filters.append("d.kind = $kind")
            params["kind"] = kind

        cypher = (
            "MATCH (d:Deliverable:__Platform__)\n"
            f"WHERE {' AND '.join(filters)}\n"
            "RETURN\n"
            "    d.deliverable_id  AS deliverable_id,\n"
            "    d.run_id          AS run_id,\n"
            "    d.module_run_id   AS module_run_id,\n"
            "    d.kind            AS kind,\n"
            "    d.filename        AS filename,\n"
            "    d.ordinal         AS ordinal,\n"
            "    d.content_uri     AS content_uri,\n"
            "    d.sha256          AS sha256,\n"
            "    d.word_count      AS word_count,\n"
            "    (d.content_inline IS NOT NULL AND d.content_inline <> '') AS has_inline\n"
            "ORDER BY d.ordinal ASC, d.deliverable_id ASC\n"
            "SKIP $offset LIMIT $limit\n"
        )
        result = await self._driver.execute_query(cypher, params)
        recs = result.records
        has_more = len(recs) > limit
        page = recs[:limit]

        rows: list[DeliverableRow] = []
        for rec in page:
            rows.append(
                DeliverableRow(
                    deliverable_id=rec["deliverable_id"],
                    run_id=rec["run_id"],
                    module_run_id=rec["module_run_id"],
                    kind=rec["kind"],
                    filename=rec["filename"],
                    ordinal=int(rec["ordinal"] or 0),
                    content_uri=rec["content_uri"],
                    sha256=rec["sha256"],
                    word_count=(
                        int(rec["word_count"])
                        if rec["word_count"] is not None
                        else None
                    ),
                    has_inline_content=bool(rec["has_inline"]),
                )
            )
        return rows, has_more

    # ─── list_template_modules ─────────────────────────────────────────────

    async def list_template_modules(
        self, template_slug: str
    ) -> tuple[dict[str, Any], list[ModuleDefinition]] | None:
        """Template introspection for the UI's pre-run rendering."""
        if not template_slug:
            raise ValueError("template_slug is required")

        result = await self._driver.execute_query(
            """
            MATCH (t:AssessmentTemplate:__Platform__ {slug: $slug})
            OPTIONAL MATCH (t)-[:HAS_MODULE]->(m:Module:__Platform__)
            WITH t, m
            ORDER BY m.wave ASC, m.ordinal ASC
            RETURN
                t.template_id   AS template_id,
                t.slug          AS template_slug,
                t.name          AS template_name,
                t.version       AS template_version,
                m.module_id     AS module_id,
                m.slug          AS module_slug,
                m.name          AS module_name,
                m.wave          AS wave,
                m.ordinal       AS ordinal,
                m.kind          AS kind,
                m.agent_id      AS agent_id,
                m.description   AS description
            """,
            {"slug": template_slug},
        )
        if not result.records:
            return None

        first = result.records[0]
        meta = {
            "template_id": first["template_id"],
            "template_slug": first["template_slug"],
            "template_name": first["template_name"],
            "template_version": first["template_version"],
        }
        modules: list[ModuleDefinition] = []
        for rec in result.records:
            if rec["module_id"] is None:
                continue
            modules.append(
                ModuleDefinition(
                    module_id=rec["module_id"],
                    template_id=first["template_id"],
                    slug=rec["module_slug"],
                    name=rec["module_name"],
                    wave=int(rec["wave"] or 1),
                    ordinal=int(rec["ordinal"] or 0),
                    kind=rec["kind"],
                    agent_id=rec["agent_id"],
                    description=rec["description"],
                )
            )
        return meta, modules

    # ─── Registry reads (ADR-019) ──────────────────────────────────────────

    @staticmethod
    def _resolve_visibility_targets(
        visibility: RegistryVisibility | None,
    ) -> set[str]:
        """ADR-019: default listing → curated+public+caller-owned-private."""
        if visibility is None:
            return {"curated", "public", "private"}
        return {visibility}

    async def list_registry_items(
        self,
        *,
        caller_user_id: str,
        caller_graph_id: str,
        kind: RegistryKind,
        owner_user_id: str | None = None,
        visibility: RegistryVisibility | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[RegistryItemRow], bool]:
        """List registry items per ADR-019 visibility rules.

        - `curated`/`public`/`yanked`: from `__registry__`.
        - `private`: from caller's tenant graph, caller-owned only.
        - default (None): curated + public + caller-owned private.
        """
        if not caller_user_id:
            raise ValueError("caller_user_id is required")
        if not caller_graph_id:
            raise ValueError("caller_graph_id is required")

        targets = self._resolve_visibility_targets(visibility)
        all_rows: list[dict[str, Any]] = []

        if {"curated", "public", "yanked"} & targets:
            catalog_visibilities = [
                v for v in ("curated", "public", "yanked") if v in targets
            ]
            cat_filters = [
                "ri.graph_id = $catalog",
                "ri.kind = $kind",
                "ri.visibility IN $visibilities",
            ]
            cat_params: dict[str, Any] = {
                "catalog": REGISTRY_CATALOG_GRAPH_ID,
                "kind": kind,
                "visibilities": catalog_visibilities,
            }
            if owner_user_id is not None:
                cat_filters.append("ri.owner_user_id = $owner")
                cat_params["owner"] = owner_user_id
            cat_cypher = (
                "MATCH (ri:RegistryItem:__Platform__)\n"
                f"WHERE {' AND '.join(cat_filters)}\n"
                "RETURN ri AS ri\n"
            )
            cat_result = await self._driver.execute_query(cat_cypher, cat_params)
            for rec in cat_result.records:
                all_rows.append(dict(rec["ri"].items()))

        if "private" in targets:
            if owner_user_id is not None and owner_user_id != caller_user_id:
                pass  # cannot see someone else's private
            else:
                priv_cypher = (
                    "MATCH (ri:RegistryItem:__Platform__)\n"
                    "WHERE ri.graph_id = $graph_id"
                    " AND ri.kind = $kind"
                    " AND ri.visibility = 'private'"
                    " AND ri.owner_user_id = $owner\n"
                    "RETURN ri AS ri\n"
                )
                priv_result = await self._driver.execute_query(
                    priv_cypher,
                    {
                        "graph_id": caller_graph_id,
                        "kind": kind,
                        "owner": caller_user_id,
                    },
                )
                for rec in priv_result.records:
                    all_rows.append(dict(rec["ri"].items()))

        all_rows.sort(
            key=lambda r: (
                r.get("kind") or "",
                r.get("slug") or "",
                r.get("version") or "",
                r.get("item_id") or "",
            )
        )
        page = all_rows[offset : offset + limit + 1]
        has_more = len(page) > limit
        page = page[:limit]

        rows = [self._registry_item_row_from_dict(r) for r in page]
        return rows, has_more

    def _registry_item_row_from_dict(self, r: dict[str, Any]) -> RegistryItemRow:
        return RegistryItemRow(
            item_id=r["item_id"],
            graph_id=r["graph_id"],
            kind=r["kind"],
            slug=r["slug"],
            version=r.get("version") or "0.1.0",
            visibility=r["visibility"],
            owner_user_id=r["owner_user_id"],
            name=r["name"],
            description=r.get("description"),
            content_uri=r.get("content_uri"),
            sha256=r.get("sha256"),
            created_at=self._neo_datetime(r.get("created_at")),
            yanked_at=self._neo_datetime(r.get("yanked_at")),
        )

    async def get_registry_item(
        self,
        *,
        caller_user_id: str,
        caller_graph_id: str,
        kind: RegistryKind,
        slug: str,
        version: str | None = None,
    ) -> RegistryItemRow | None:
        """Resolve `<kind>/<slug>[@version]` to a single RegistryItem.

        Returns `None` for "not found OR private-owned-by-someone-else".
        The 404/403 collapse is deliberate per ADR-019 — a non-owner must
        not distinguish "your private slug exists" from "no such slug".
        """
        if not slug:
            raise ValueError("slug is required")

        # Catalog first (curated/public/yanked).
        cypher_cat = (
            "MATCH (ri:RegistryItem:__Platform__ {graph_id: $catalog, kind: $kind, slug: $slug})\n"
            + ("WHERE ri.version = $version\n" if version else "")
            + "RETURN ri AS ri\n"
            "ORDER BY ri.version DESC\n"
            "LIMIT 1\n"
        )
        params_cat: dict[str, Any] = {
            "catalog": REGISTRY_CATALOG_GRAPH_ID,
            "kind": kind,
            "slug": slug,
        }
        if version:
            params_cat["version"] = version
        cat_result = await self._driver.execute_query(cypher_cat, params_cat)
        if cat_result.records:
            return self._registry_item_row_from_dict(
                dict(cat_result.records[0]["ri"].items())
            )

        # Then private (caller's tenant, caller-owned).
        cypher_priv = (
            "MATCH (ri:RegistryItem:__Platform__ {graph_id: $graph_id, kind: $kind, slug: $slug, visibility: 'private'})\n"
            "WHERE ri.owner_user_id = $owner\n"
            + ("AND ri.version = $version\n" if version else "")
            + "RETURN ri AS ri\n"
            "ORDER BY ri.version DESC\n"
            "LIMIT 1\n"
        )
        params_priv: dict[str, Any] = {
            "graph_id": caller_graph_id,
            "kind": kind,
            "slug": slug,
            "owner": caller_user_id,
        }
        if version:
            params_priv["version"] = version
        priv_result = await self._driver.execute_query(cypher_priv, params_priv)
        if priv_result.records:
            return self._registry_item_row_from_dict(
                dict(priv_result.records[0]["ri"].items())
            )
        return None

    async def get_registry_item_content(
        self,
        *,
        caller_user_id: str,
        caller_graph_id: str,
        kind: RegistryKind,
        slug: str,
        version: str,
    ) -> RegistryItemContent | None:
        """Content payload for a specific RegistryItem version (ADR-019 gated)."""
        item_row = await self.get_registry_item(
            caller_user_id=caller_user_id,
            caller_graph_id=caller_graph_id,
            kind=kind,
            slug=slug,
            version=version,
        )
        if item_row is None:
            return None
        content_result = await self._driver.execute_query(
            """
            MATCH (ri:RegistryItem:__Platform__ {item_id: $item_id})
            RETURN ri.content_inline AS content_inline, ri.sha256 AS sha256
            LIMIT 1
            """,
            {"item_id": item_row.item_id},
        )
        content_inline: str | None = None
        if content_result.records:
            content_inline = content_result.records[0]["content_inline"]
        content_type = (
            "text/markdown" if kind in ("skill", "agent") else "application/json"
        )
        return RegistryItemContent(
            item_id=item_row.item_id,
            kind=item_row.kind,
            slug=item_row.slug,
            version=item_row.version,
            content_type=content_type,
            content_inline=content_inline,
            content_uri=item_row.content_uri,
            sha256=item_row.sha256,
        )

    # ─── Admin: cross-run findings:search ──────────────────────────────────

    async def search_findings_admin(
        self,
        *,
        source_url: str | None = None,
        dimension: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[FindingSearchRow], bool]:
        """Admin-only cross-run findings search.

        Endpoint gates this behind
        `verify_graph_access('__assessments_catalog__', 'admin', user_id)`.
        The query crosses graph_id by design; each row carries `graph_id`
        so admins can trace origin.
        """
        source_id_filter: list[str] | None = None
        if source_url:
            src_result = await self._driver.execute_query(
                """
                MATCH (s:Source:__Platform__ {url_normalized: $url})
                RETURN s.source_id AS source_id
                """,
                {"url": source_url},
            )
            source_id_filter = [r["source_id"] for r in src_result.records]
            if not source_id_filter:
                return [], False

        filters: list[str] = []
        params: dict[str, Any] = {"offset": offset, "limit": limit + 1}
        if source_id_filter is not None:
            filters.append("f.source_id IN $source_ids")
            params["source_ids"] = source_id_filter
        if dimension is not None:
            filters.append("$dimension IN f.dimensions")
            params["dimension"] = dimension
        if not filters:
            # Force at least one filter so unfiltered admin queries don't
            # `LIMIT-only` the entire :Finding graph.
            filters.append("f.source_id IS NOT NULL")

        cypher = (
            "MATCH (f:Finding:__Platform__)\n"
            f"WHERE {' AND '.join(filters)}\n"
            "RETURN\n"
            "    f.finding_id     AS finding_id,\n"
            "    f.graph_id       AS graph_id,\n"
            "    f.run_id         AS run_id,\n"
            "    f.module_run_id  AS module_run_id,\n"
            "    f.claim          AS claim,\n"
            "    f.label          AS label,\n"
            "    f.confidence     AS confidence,\n"
            "    f.dimensions     AS dimensions,\n"
            "    f.source_id      AS source_id\n"
            "ORDER BY f.confidence DESC, f.finding_id ASC\n"
            "SKIP $offset LIMIT $limit\n"
        )
        result = await self._driver.execute_query(cypher, params)
        recs = result.records
        has_more = len(recs) > limit
        page = recs[:limit]

        source_ids = list({rec["source_id"] for rec in page if rec["source_id"]})
        sources = await self._fetch_sources_by_ids(source_ids)

        rows: list[FindingSearchRow] = [
            FindingSearchRow(
                finding_id=rec["finding_id"],
                graph_id=rec["graph_id"],
                run_id=rec["run_id"],
                module_run_id=rec["module_run_id"],
                claim=rec["claim"],
                label=rec["label"],
                confidence=float(rec["confidence"] or 0.0),
                dimensions=list(rec["dimensions"] or []),
                source_id=rec["source_id"],
                source=sources.get(rec["source_id"]) if rec["source_id"] else None,
            )
            for rec in page
        ]
        return rows, has_more
