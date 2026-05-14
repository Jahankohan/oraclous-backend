"""Structured-data ingestion (STORY-9).

Pre-STORY-9 the only ingest path was text → chunk → LLM-extract → graph,
which is wasteful and lossy for inputs that already have explicit
structure (JSON records, CSV rows, RDB tables). STORY-9 adds a
structured ingest path: every JSON record becomes one entity with
typed properties and optional relationships to other entities via
foreign-key-style fields.

What this service does NOT do (intentional scope cuts):
- Run the LLM extractor — the structure is already known, no need.
- Auto-infer relationships beyond the explicit ``relationships`` mapping.
- Chunk and embed long text fields — caller can do that via the
  existing /ingest endpoint when needed (or via an
  agent-driven flow planned for STORY-10).

Field mapping rules (applied to every record):
- ``id_field`` (default ``"id"``) identifies the entity. Required.
- Primitive values (str, int, float, bool) → entity properties verbatim.
- ``None`` → property dropped.
- Lists of primitives → entity property as Neo4j array.
- Lists of dicts or nested dicts → JSON-stringified property (callers
  needing rich child nodes should issue a follow-up
  ``ingest-records`` call for those).
- ``relationships`` list maps a record field to an outgoing edge:
  ``{"from_field": "company_id", "to_label": "Company",
  "rel_type": "ABOUT", "to_id_field": "id"}`` — for each record's
  ``company_id`` value, MERGE a Company node (id = that value) and
  create the typed edge. ``to_id_field`` is the property on the
  target node that identifies it (default ``"id"``).

All Cypher is parameterized. Tenant isolation enforced via
``graph_id`` set on every entity and via the MATCH clauses.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client

logger = get_logger(__name__)


@dataclass(slots=True)
class RelationshipMapping:
    """One field-to-edge rule."""

    from_field: str
    to_label: str
    rel_type: str
    to_id_field: str = "id"


@dataclass(slots=True)
class IngestReport:
    """Outcome of one ingest_records call."""

    label: str
    records_processed: int = 0
    entities_created_or_updated: int = 0
    relationships_created: int = 0
    related_entities_created: int = 0
    skipped: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


class StructuredIngestService:
    """Idempotent JSON-record → graph ingest."""

    async def ingest_records(
        self,
        graph_id: str,
        records: list[dict[str, Any]],
        label: str,
        id_field: str = "id",
        relationships: list[RelationshipMapping] | None = None,
    ) -> IngestReport:
        """Ingest a list of JSON records into the graph.

        Args:
            graph_id: Tenant graph id (set on every entity).
            records: List of JSON-decoded dicts. Each dict becomes one entity.
            label: Neo4j label for the resulting entities (e.g.
                ``"EvidenceRecord"``). Must be alphanumeric + underscore.
            id_field: Name of the dict key that holds the entity id.
                Records without this key are skipped.
            relationships: Optional list of edge mappings. For each
                mapping, the record's value at ``from_field`` becomes
                the target entity's id (under ``to_label``).
        """
        start = time.time()
        report = IngestReport(label=label)
        if not _is_safe_label(label):
            raise ValueError(
                f"Invalid label {label!r}: only alphanumeric characters and "
                "underscore allowed"
            )

        relationships = list(relationships or [])
        for rmap in relationships:
            if not _is_safe_label(rmap.to_label):
                raise ValueError(
                    f"Invalid relationship to_label {rmap.to_label!r}: only "
                    "alphanumeric characters and underscore allowed"
                )
            if not _is_safe_rel_type(rmap.rel_type):
                raise ValueError(
                    f"Invalid rel_type {rmap.rel_type!r}: must match ^[A-Z][A-Z0-9_]*$"
                )

        # Pre-flight: validate every record has the id field and a
        # plausible value. Better to fail fast than half-ingest.
        for rec in records:
            if not isinstance(rec, dict):
                report.skipped += 1
                _bump(report.skip_reasons, "not_a_dict")
                continue
            entity_id = rec.get(id_field)
            if entity_id is None or not isinstance(entity_id, str | int):
                report.skipped += 1
                _bump(report.skip_reasons, "missing_or_invalid_id")
                continue

            # Build the property map: primitives + array-of-primitives
            # passed verbatim; complex types JSON-stringified.
            props = _record_to_properties(rec, id_field=id_field)

            try:
                await self._merge_entity(
                    graph_id=graph_id,
                    label=label,
                    entity_id=str(entity_id),
                    properties=props,
                )
                report.entities_created_or_updated += 1
            except Exception as exc:
                logger.warning(
                    "structured_ingest: failed to merge entity %s/%s: %s",
                    label,
                    entity_id,
                    exc,
                )
                report.skipped += 1
                _bump(report.skip_reasons, "merge_failed")
                continue

            # Process relationships
            for rmap in relationships:
                raw_target = rec.get(rmap.from_field)
                if raw_target is None:
                    continue
                # Allow a list of target ids — emit one edge per element.
                target_ids = (
                    raw_target if isinstance(raw_target, list) else [raw_target]
                )
                for tid in target_ids:
                    if not isinstance(tid, str | int) or tid == "":
                        continue
                    try:
                        created = await self._upsert_relationship(
                            graph_id=graph_id,
                            from_label=label,
                            from_id=str(entity_id),
                            to_label=rmap.to_label,
                            to_id_field=rmap.to_id_field,
                            to_id=str(tid),
                            rel_type=rmap.rel_type,
                        )
                        report.relationships_created += 1
                        if created:
                            report.related_entities_created += 1
                    except Exception as exc:
                        logger.warning(
                            "structured_ingest: failed to link %s/%s -[%s]-> %s/%s: %s",
                            label,
                            entity_id,
                            rmap.rel_type,
                            rmap.to_label,
                            tid,
                            exc,
                        )

            report.records_processed += 1

        report.elapsed_seconds = round(time.time() - start, 2)
        logger.info(
            "structured_ingest complete graph_id=%s label=%s "
            "records=%d entities=%d rels=%d skipped=%d elapsed=%.2fs",
            graph_id,
            label,
            report.records_processed,
            report.entities_created_or_updated,
            report.relationships_created,
            report.skipped,
            report.elapsed_seconds,
        )
        return report

    # ── private ──────────────────────────────────────────────────────────────

    async def _merge_entity(
        self,
        graph_id: str,
        label: str,
        entity_id: str,
        properties: dict[str, Any],
    ) -> None:
        """MERGE entity by (graph_id, id). SET property map.

        Label name has been validated by ``_is_safe_label`` — safe for
        f-string interpolation. graph_id and id passed as parameters.
        """
        # Strip ``id`` from properties so we don't overwrite the MERGE key
        # with a different value.
        props = {k: v for k, v in properties.items() if k != "id"}
        await neo4j_client.execute_query(
            (
                f"MERGE (e:`{label}`:__Entity__ {{id: $id, graph_id: $gid}}) "
                "SET e += $props, "
                "    e.updated_at = datetime(), "
                "    e.ingestion_source = 'structured_ingest'"
            ),
            {"id": entity_id, "gid": graph_id, "props": props},
        )

    async def _upsert_relationship(
        self,
        graph_id: str,
        from_label: str,
        from_id: str,
        to_label: str,
        to_id_field: str,
        to_id: str,
        rel_type: str,
    ) -> bool:
        """MERGE target entity if absent; MERGE typed edge from→to.

        Returns True when the target entity was newly created (so
        callers can count related-entity creations).

        Labels and rel_type pre-validated by ``_is_safe_label`` /
        ``_is_safe_rel_type``. Parameterized for everything else.
        """
        # Check whether the target exists BEFORE the MERGE so we can
        # accurately report related_entities_created.
        existence = await neo4j_client.execute_query(
            (
                f"MATCH (t:`{to_label}` {{`{to_id_field}`: $to_id, graph_id: $gid}}) "
                "RETURN count(t) AS n"
            ),
            {"to_id": to_id, "gid": graph_id},
        )
        existed_before = (existence[0]["n"] if existence else 0) > 0

        await neo4j_client.execute_query(
            (
                f"MATCH (f:`{from_label}` {{id: $from_id, graph_id: $gid}}) "
                f"MERGE (t:`{to_label}`:__Entity__ {{`{to_id_field}`: $to_id, graph_id: $gid}}) "
                f"ON CREATE SET t.created_at = datetime(), "
                "             t.ingestion_source = 'structured_ingest' "
                f"MERGE (f)-[r:`{rel_type}` {{graph_id: $gid}}]->(t) "
                "ON CREATE SET r.count = 1, r.first_seen = datetime() "
                "ON MATCH SET r.count = coalesce(r.count, 1) + 1, "
                "             r.last_seen = datetime()"
            ),
            {
                "from_id": from_id,
                "to_id": to_id,
                "gid": graph_id,
            },
        )
        return not existed_before


# ── helpers ─────────────────────────────────────────────────────────────────


def _is_safe_label(label: Any) -> bool:
    """Neo4j label: alphanumeric + underscore. Reject everything else
    so we never interpolate user input into Cypher uncontrolled."""
    return isinstance(label, str) and bool(label) and label.replace("_", "").isalnum()


def _is_safe_rel_type(rel_type: Any) -> bool:
    """Neo4j rel type: ALL_CAPS_WITH_UNDERSCORES — same pattern that
    ExtractedRelationship uses."""
    import re as _re

    return isinstance(rel_type, str) and bool(_re.match(r"^[A-Z][A-Z0-9_]*$", rel_type))


def _record_to_properties(record: dict[str, Any], id_field: str) -> dict[str, Any]:
    """Project a JSON record to Neo4j-acceptable property values.

    Primitives + arrays-of-primitives pass through unchanged. Nested
    dicts and arrays-of-dicts are JSON-stringified so they're still
    queryable as strings. None values are dropped.
    """
    out: dict[str, Any] = {}
    for key, value in record.items():
        if value is None:
            continue
        if isinstance(value, str | int | float | bool):
            out[key] = value
            continue
        if isinstance(value, list):
            # All primitives → keep as array. Anything else → JSON-stringify.
            if all(isinstance(x, str | int | float | bool) for x in value):
                out[key] = value
            else:
                out[key] = json.dumps(value, default=str)
            continue
        # Nested dict or other complex type → JSON-stringify
        try:
            out[key] = json.dumps(value, default=str)
        except (TypeError, ValueError):
            # Last-resort: stringify
            out[key] = str(value)
    return out


def _bump(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


# Module-level singleton — service is stateless.
structured_ingest_service = StructuredIngestService()
