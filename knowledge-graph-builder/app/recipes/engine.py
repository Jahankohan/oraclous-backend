"""Recipe execution engine — the run-time stage of concern-driven ingestion.

TASK-223 / STORY-034 / ADR-022.

The engine is the **mechanical, deterministic** half of the recipe pipeline. A
recipe (TASK-220) is *data* — it is authored once, at design time, by the
`data-specialist` agent. This engine *interprets* that recipe over a
`StructuralRepresentation` (TASK-222) and projects it into the unified graph
model (TASK-221): a `:Source` node, container nodes, and `:__Entity__` nodes
with recipe-supplied domain labels.

Run-time guarantees:

* **No LLM, no agent.** Structured projection (node / edge / property / skip) is
  fully deterministic. `text_extraction` — the only rule kind that needs a
  model — is **out of scope for TASK-223** and is recognised, skipped, and
  reported as a warning (see ``_apply_text_extraction``).
* **Injection-immune.** A recipe is agent-authored data; every recipe-supplied
  label / relationship type / property key passes a strict safe-identifier
  allowlist (recipe-spec §5.5) before it is built into a Cypher string. An
  unsafe identifier is a *rejected recipe*, not a sanitised one.
* **Tenant-scoped.** `graph_id` is stamped on every node and edge written and
  is part of every identity hash — no cross-tenant write is possible.
* **Idempotent.** Identity is a deterministic hash; every write is a `MERGE` on
  that identity. Re-running ``execute`` yields the same graph — no duplicates.
* **Batched.** Writes are `UNWIND`-batched (~500 rows) — no per-row round trip.

The engine writes with the **sync** Neo4j driver — it runs inside a Celery task
(see ``recipe_execution_task`` in ``app/tasks/recipe_tasks.py``), following the
same NullPool, task-scoped pattern as ``app/services/row_transformer.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import resources
from typing import Any

import jsonschema

from app.core.logging import get_logger
from app.recipes.primitives.interface import StructuralRepresentation, StructuralUnit

logger = get_logger(__name__)

_BATCH_SIZE = 500

# Strict Cypher-safe identifier — letters/underscore start, the (?!__) guard
# rejects the ADR-015 reserved __wrapped__ namespace. Mirrors recipe.schema.json
# `safe_identifier` and the schema_mapper / row_transformer validation pattern.
_SAFE_IDENTIFIER = re.compile(r"^(?!__)[A-Za-z_][A-Za-z0-9_]*$")

# ADR-015 reserved namespace — a recipe declares *domain* labels only.
_RESERVED_LABELS = frozenset(
    {"__Platform__", "__Entity__", "__KGBuilder__", "__Rebac__", "__System__"}
)

# Container labels are platform-owned; a recipe must never declare one of these
# as a domain label (unified-graph-model.md §3).
_CONTAINER_LABELS = frozenset({"Source", "Table", "Sheet", "File", "Chunk"})

# unit_kind → container label, for the materialized structure layer.
_CONTAINER_KIND_TO_LABEL: dict[str, str] = {
    "table": "Table",
    "sheet": "Sheet",
    "file": "File",
    "chunk": "Chunk",
}

# Kinds that carry a real per-record payload a `node` rule projects per row.
_RECORD_KINDS = frozenset({"record", "row"})


class RecipeValidationError(ValueError):
    """The recipe is invalid — schema-invalid, or an unsafe / reserved identifier.

    A recipe failing validation is *refused*. The engine never coerces or
    sanitises a recipe (recipe-spec §5.5).
    """


@dataclass
class ExecutionResult:
    """Outcome of one ``RecipeExecutionEngine.execute`` run."""

    recipe_id: str
    recipe_version: int
    graph_id: str
    source_id: str
    containers_written: int = 0
    nodes_written: int = 0
    edges_written: int = 0
    properties_written: int = 0
    units_skipped: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "recipe_version": self.recipe_version,
            "graph_id": self.graph_id,
            "source_id": self.source_id,
            "containers_written": self.containers_written,
            "nodes_written": self.nodes_written,
            "edges_written": self.edges_written,
            "properties_written": self.properties_written,
            "units_skipped": self.units_skipped,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Identifier safety (recipe-spec §5.5)
# ---------------------------------------------------------------------------


def _is_safe_identifier(value: Any) -> bool:
    """True if *value* is a Cypher-safe identifier and not a reserved label."""
    if not isinstance(value, str):
        return False
    if not _SAFE_IDENTIFIER.match(value):
        return False
    if value in _RESERVED_LABELS:
        return False
    return True


# ---------------------------------------------------------------------------
# Identity normalization — bounded to the recipe-schema's fixed enum
# ---------------------------------------------------------------------------


def _normalize_identity(value: Any, ops: list[str]) -> str:
    """Apply the recipe's `identity.normalize` ops to a single identity value.

    Only the three ops the recipe JSON Schema allows are honoured — `casefold`,
    `trim`, `collapse_whitespace` — so a recipe cannot define an arbitrary
    normalization that over-collapses two distinct entities (unified-graph-model
    §4, SA finding 4).
    """
    text = "" if value is None else str(value)
    for op in ops:
        if op == "casefold":
            text = text.casefold()
        elif op == "trim":
            text = text.strip()
        elif op == "collapse_whitespace":
            text = re.sub(r"\s+", " ", text)
    return text


def _deterministic_id(graph_id: str, label: str, identity_key: str) -> str:
    """A deterministic hash of (graph_id, domain label, normalized identity).

    This is the entity `id` — decoupled from any storage primary key
    (unified-graph-model §4). Re-running the recipe re-derives the same id, so
    the MERGE is idempotent.
    """
    payload = f"{graph_id}|{label}|{identity_key}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _source_id(graph_id: str, source_descriptor: str) -> str:
    """Deterministic `:Source` id — hash of (graph_id, source descriptor)."""
    payload = f"{graph_id}|source|{source_descriptor}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _container_id(graph_id: str, unit_id: str) -> str:
    """Deterministic container id — hash of (graph_id, unit path in the source)."""
    payload = f"{graph_id}|container|{unit_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Value coercion (Neo4j-compatible types)
# ---------------------------------------------------------------------------


def _coerce_value(value: Any) -> Any:
    """Coerce a value to a Neo4j-storable type.

    Neo4j supports bool / int / float / str / list / dict / None. A datetime is
    serialized to ISO-8601; anything else is stringified.
    """
    if isinstance(value, (bool, int, float, str, type(None))):
        return value
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_coerce_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _coerce_value(v) for k, v in value.items()}
    return str(value)


def _chunks(items: list, size: int):
    """Yield fixed-size chunks of *items*."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


class RecipeExecutionEngine:
    """Interpret a recipe over a `StructuralRepresentation` and write the graph.

    Stateless — one instance can execute many recipes. The Neo4j *driver* is
    passed per ``execute`` call so the engine stays agnostic of how the driver
    is created (FastAPI vs. Celery worker).
    """

    SUPPORTED_FORMAT_VERSIONS = frozenset({"0.2"})

    def __init__(self) -> None:
        self._schema = self._load_schema()
        self._validator = jsonschema.Draft202012Validator(self._schema)

    # -- public API --------------------------------------------------------

    def execute(
        self,
        recipe: dict[str, Any],
        representation: StructuralRepresentation,
        graph_id: str,
        driver: Any,
    ) -> ExecutionResult:
        """Project *representation* into the graph for tenant *graph_id*.

        Steps:
          1. Validate the recipe against the JSON Schema and the §5.5 identifier
             safety rules — an invalid recipe raises `RecipeValidationError`.
          2. Materialize the structure layer — a `:Source` node plus a container
             node per container-kind unit, with `PART_OF` edges.
          3. For each structural unit, apply the *first* matching mapping rule
             (first-match-wins, recipe-spec §5.6).
          4. Return an `ExecutionResult` with counts and warnings.

        The whole run is idempotent — re-calling ``execute`` produces the same
        graph (every write is a `MERGE` on a deterministic identity).
        """
        self._validate_recipe(recipe)

        recipe_id: str = recipe["id"]
        recipe_version: int = recipe["version"]
        ingestion_time = datetime.now(UTC).isoformat()

        # -- 2. structure layer -------------------------------------------
        source_descriptor = (
            f"{representation.source_type}:{representation.shape_signature}"
        )
        source_id = _source_id(graph_id, source_descriptor)
        result = ExecutionResult(
            recipe_id=recipe_id,
            recipe_version=recipe_version,
            graph_id=graph_id,
            source_id=source_id,
        )

        meta = {
            "graph_id": graph_id,
            "ingestion_source": source_id,
            "recipe_id": recipe_id,
            "recipe_version": recipe_version,
            "ingestion_time": ingestion_time,
        }

        # Per-run lookup — each unit's record payload by id — so property
        # rules can read the value travelling with the node a unit produced.
        payload_cache = {
            u.unit_id: self._resolve_unit_payload(u) for u in representation.units
        }

        self._write_source(driver, source_id, representation, meta)
        unit_to_container = self._write_containers(
            driver, representation, source_id, meta, result
        )

        # -- 3. mapping rules ---------------------------------------------
        node_rules = {
            rule["id"]: rule
            for rule in recipe["mappings"]
            if rule["project_to"] == "node"
        }
        defaults = recipe.get("defaults", {})

        # First pass — node projection. A `node` rule projects one entity per
        # record-bearing unit it matches; a node rule matching a container/
        # source unit projects a single node for that unit itself.
        # node_index maps (rule_id, unit_id) -> entity id, so edge / property
        # rules can resolve the node a unit produced.
        node_index: dict[tuple[str, str], str] = {}
        for unit in representation.units:
            rule = self._first_match(recipe["mappings"], unit)
            if rule is None:
                result.units_skipped += 1
                continue
            kind = rule["project_to"]
            if kind == "node":
                entity_id = self._project_node(
                    driver,
                    rule,
                    unit,
                    graph_id,
                    meta,
                    defaults,
                    unit_to_container,
                    source_id,
                    node_index,
                    result,
                )
                if entity_id is not None:
                    node_index[(rule["id"], unit.unit_id)] = entity_id
            elif kind == "skip":
                result.units_skipped += 1
            # property / edge / text_extraction handled in the second pass —
            # they depend on node_index being fully populated.

        # Second pass — property, edge, text_extraction.
        for unit in representation.units:
            rule = self._first_match(recipe["mappings"], unit)
            if rule is None:
                continue
            kind = rule["project_to"]
            if kind == "property":
                self._apply_property(
                    driver,
                    rule,
                    unit,
                    graph_id,
                    node_rules,
                    node_index,
                    payload_cache,
                    result,
                )
            elif kind == "edge":
                self._apply_edge(
                    driver,
                    rule,
                    unit,
                    graph_id,
                    meta,
                    defaults,
                    node_rules,
                    node_index,
                    representation,
                    payload_cache,
                    result,
                )
            elif kind == "text_extraction":
                self._apply_text_extraction(rule, unit, result)

        logger.info(
            "recipe_engine: executed %s v%s for graph %s — "
            "%d containers, %d nodes, %d edges, %d skipped",
            recipe_id,
            recipe_version,
            graph_id,
            result.containers_written,
            result.nodes_written,
            result.edges_written,
            result.units_skipped,
        )
        return result

    # -- validation --------------------------------------------------------

    @staticmethod
    def _load_schema() -> dict[str, Any]:
        """Load the recipe JSON Schema bundled with the package."""
        text = (
            resources.files("app.recipes")
            .joinpath("recipe.schema.json")
            .read_text(encoding="utf-8")
        )
        return json.loads(text)

    def _validate_recipe(self, recipe: dict[str, Any]) -> None:
        """Validate *recipe* — schema, supported format version, §5.5 safety.

        Raises `RecipeValidationError` on any failure. The engine refuses an
        invalid recipe; it never coerces it.
        """
        if not isinstance(recipe, dict):
            raise RecipeValidationError("recipe must be a JSON object")

        errors = sorted(self._validator.iter_errors(recipe), key=lambda e: list(e.path))
        if errors:
            first = errors[0]
            path = "/".join(str(p) for p in first.path) or "<root>"
            raise RecipeValidationError(
                f"recipe failed schema validation at {path}: {first.message}"
            )

        fmt = recipe.get("recipe_format_version")
        if fmt not in self.SUPPORTED_FORMAT_VERSIONS:
            raise RecipeValidationError(
                f"unsupported recipe_format_version {fmt!r}; "
                f"engine supports {sorted(self.SUPPORTED_FORMAT_VERSIONS)}"
            )

        # §5.5 — every recipe-supplied label / relationship type / property key
        # must be a safe Cypher identifier outside the reserved namespace.
        # The schema's `safe_identifier` pattern already enforces this; this
        # is a defence-in-depth re-check before the value reaches a query
        # string, and additionally rejects the platform container labels.
        for rule in recipe["mappings"]:
            self._check_rule_identifiers(rule)

    def _check_rule_identifiers(self, rule: dict[str, Any]) -> None:
        """Reject a rule that carries an unsafe / reserved Cypher identifier."""
        kind = rule["project_to"]
        if kind == "node":
            label = rule["label"]
            if not _is_safe_identifier(label):
                raise RecipeValidationError(
                    f"node rule {rule['id']!r}: unsafe domain label {label!r}"
                )
            if label in _CONTAINER_LABELS:
                raise RecipeValidationError(
                    f"node rule {rule['id']!r}: domain label {label!r} collides "
                    f"with a platform-owned container label"
                )
            for prop in rule.get("properties", []):
                if not _is_safe_identifier(prop["name"]):
                    raise RecipeValidationError(
                        f"node rule {rule['id']!r}: unsafe property key "
                        f"{prop['name']!r}"
                    )
        elif kind == "edge":
            rel_type = rule["type"]
            if not _is_safe_identifier(rel_type):
                raise RecipeValidationError(
                    f"edge rule {rule['id']!r}: unsafe relationship type {rel_type!r}"
                )
            for prop in rule.get("properties", []):
                if not _is_safe_identifier(prop["name"]):
                    raise RecipeValidationError(
                        f"edge rule {rule['id']!r}: unsafe property key "
                        f"{prop['name']!r}"
                    )
        elif kind == "property":
            if not _is_safe_identifier(rule["name"]):
                raise RecipeValidationError(
                    f"property rule {rule['id']!r}: unsafe property key "
                    f"{rule['name']!r}"
                )
        elif kind == "text_extraction":
            edge_type = rule["link_to"]["edge_type"]
            if not _is_safe_identifier(edge_type):
                raise RecipeValidationError(
                    f"text_extraction rule {rule['id']!r}: unsafe edge type "
                    f"{edge_type!r}"
                )

    # -- rule matching (recipe-spec §5.6) ----------------------------------

    @staticmethod
    def _matches(match: dict[str, Any], unit: StructuralUnit) -> bool:
        """True if a rule's `match` clause matches *unit*.

        A `match` constrains `unit_kind` (required) and, optionally, `name` and
        `role`. An omitted constraint matches anything.
        """
        if match["unit_kind"] != unit.kind.value:
            return False
        if "name" in match and match["name"] != unit.name:
            return False
        if "role" in match and match["role"] != unit.role:
            return False
        return True

    def _first_match(
        self, mappings: list[dict[str, Any]], unit: StructuralUnit
    ) -> dict[str, Any] | None:
        """The first mapping rule whose `match` matches *unit* (first-match-wins)."""
        for rule in mappings:
            if self._matches(rule["match"], unit):
                return rule
        return None

    # -- structure layer (unified-graph-model §3, §7) ----------------------

    def _write_source(
        self,
        driver: Any,
        source_id: str,
        representation: StructuralRepresentation,
        meta: dict[str, Any],
    ) -> None:
        """MERGE the `:Source` node for this representation."""
        with driver.session() as session:
            session.run(
                """
                MERGE (s:Source:__KGBuilder__ {graph_id: $graph_id,
                                                source_id: $source_id})
                ON CREATE SET s.source_type      = $source_type,
                              s.shape_signature  = $shape_signature,
                              s.ingestion_source = $source_id,
                              s.provenance       = 'EXTRACTED',
                              s.recipe_id        = $recipe_id,
                              s.recipe_version   = $recipe_version,
                              s.ingestion_time   = $ingestion_time
                """,
                {
                    "graph_id": meta["graph_id"],
                    "source_id": source_id,
                    "source_type": representation.source_type,
                    "shape_signature": representation.shape_signature,
                    "recipe_id": meta["recipe_id"],
                    "recipe_version": meta["recipe_version"],
                    "ingestion_time": meta["ingestion_time"],
                },
            )

    def _write_containers(
        self,
        driver: Any,
        representation: StructuralRepresentation,
        source_id: str,
        meta: dict[str, Any],
        result: ExecutionResult,
    ) -> dict[str, str]:
        """MERGE a container node per container-kind unit; wire `PART_OF` edges.

        Returns a mapping unit_id -> container node id, so entity nodes derived
        from a unit can be wired with `DERIVED_FROM` to their container.
        """
        unit_to_container: dict[str, str] = {}
        # Group container units by their (validated) container label.
        by_label: dict[str, list[dict[str, Any]]] = {}
        for unit in representation.units:
            label = _CONTAINER_KIND_TO_LABEL.get(unit.kind.value)
            if label is None:
                continue
            cid = _container_id(meta["graph_id"], unit.unit_id)
            unit_to_container[unit.unit_id] = cid
            by_label.setdefault(label, []).append(
                {
                    "id": cid,
                    "unit_id": unit.unit_id,
                    "name": unit.name,
                    "parent_unit_id": unit.parent_id,
                }
            )

        for label, rows in by_label.items():
            # `label` is from the fixed _CONTAINER_KIND_TO_LABEL map — not
            # recipe-supplied — so it is safe to build into the query string.
            cypher = f"""
            UNWIND $batch AS row
            MERGE (c:{label}:__KGBuilder__ {{graph_id: $graph_id, id: row.id}})
            ON CREATE SET c.unit_id          = row.unit_id,
                          c.name             = row.name,
                          c.ingestion_source = $source_id,
                          c.provenance       = 'EXTRACTED',
                          c.recipe_id        = $recipe_id,
                          c.recipe_version   = $recipe_version,
                          c.ingestion_time   = $ingestion_time
            WITH c, row
            MATCH (s:Source {{graph_id: $graph_id, source_id: $source_id}})
            MERGE (c)-[:PART_OF {{graph_id: $graph_id}}]->(s)
            """
            for batch in _chunks(rows, _BATCH_SIZE):
                with driver.session() as session:
                    session.run(
                        cypher,
                        {
                            "batch": batch,
                            "graph_id": meta["graph_id"],
                            "source_id": source_id,
                            "recipe_id": meta["recipe_id"],
                            "recipe_version": meta["recipe_version"],
                            "ingestion_time": meta["ingestion_time"],
                        },
                    )
                result.containers_written += len(batch)

        # Wire container → container `PART_OF` where a container nests inside
        # another container (e.g. a chunk inside a file).
        nested = [
            {"child": cid, "parent": unit_to_container[rows["parent_unit_id"]]}
            for label_rows in by_label.values()
            for rows in label_rows
            if rows["parent_unit_id"] in unit_to_container
            for cid in [rows["id"]]
        ]
        if nested:
            with driver.session() as session:
                session.run(
                    """
                    UNWIND $batch AS row
                    MATCH (child:__KGBuilder__ {graph_id: $graph_id, id: row.child})
                    MATCH (parent:__KGBuilder__ {graph_id: $graph_id, id: row.parent})
                    MERGE (child)-[:PART_OF {graph_id: $graph_id}]->(parent)
                    """,
                    {"batch": nested, "graph_id": meta["graph_id"]},
                )
        return unit_to_container

    # -- node projection (recipe-spec §5.1) --------------------------------

    def _resolve_unit_payload(self, unit: StructuralUnit) -> dict[str, Any]:
        """The record's value dict for a record-bearing unit.

        A `record` / `row` unit carries the row dict in ``sample_values[0]``
        (see csv_primitive / json_primitive). A container/source unit has no
        per-record payload — its own `name` / `metadata` are used instead.
        """
        if unit.kind.value in _RECORD_KINDS and unit.sample_values:
            first = unit.sample_values[0]
            if isinstance(first, dict):
                return first
        return {}

    @staticmethod
    def _read_field(payload: dict[str, Any], unit: StructuralUnit, ref: str) -> Any:
        """Resolve a recipe `value_from` / identity `from` reference.

        A reference is one of:
          * ``column:<name>`` / ``field:<name>`` — a key of the record payload;
          * ``name`` — the unit's own name;
          * a bare key — looked up directly in the payload.
        """
        if ref in ("name", "unit:name"):
            return unit.name
        if ":" in ref:
            _, _, key = ref.partition(":")
        else:
            key = ref
        return payload.get(key)

    def _project_node(
        self,
        driver: Any,
        rule: dict[str, Any],
        unit: StructuralUnit,
        graph_id: str,
        meta: dict[str, Any],
        defaults: dict[str, Any],
        unit_to_container: dict[str, str],
        source_id: str,
        node_index: dict[tuple[str, str], str],
        result: ExecutionResult,
    ) -> str | None:
        """MERGE one `:<label>:__Entity__` node for *unit* under *rule*.

        Returns the entity id, or None when the identity key is empty (the unit
        carries no value the recipe's identity rule can key on).
        """
        label: str = rule["label"]
        identity = rule["identity"]
        normalize_ops: list[str] = identity.get("normalize", [])
        payload = self._resolve_unit_payload(unit)

        # Build the normalized identity key from the recipe's `identity.from`.
        parts: list[str] = []
        for ref in identity["from"]:
            raw = self._read_field(payload, unit, ref)
            parts.append(_normalize_identity(raw, normalize_ops))
        identity_key = "|".join(parts)
        if not identity_key.strip("|"):
            result.warnings.append(
                f"node rule {rule['id']!r}: empty identity key for unit "
                f"{unit.unit_id!r} — node skipped"
            )
            result.units_skipped += 1
            return None

        entity_id = _deterministic_id(graph_id, label, identity_key)

        # Recipe-declared properties.
        props: dict[str, Any] = {}
        for prop in rule.get("properties", []):
            value = self._read_field(payload, unit, prop["value_from"])
            if value is not None:
                props[prop["name"]] = _coerce_value(value)

        provenance = rule.get("provenance") or defaults.get("provenance", "EXTRACTED")

        # `label` passed _is_safe_identifier in _check_rule_identifiers — safe
        # to build into the query string.
        cypher = f"""
        UNWIND $batch AS row
        MERGE (e:{label}:__Entity__ {{graph_id: $graph_id, id: row.id}})
        ON CREATE SET e.ingestion_source = $source_id,
                      e.provenance       = $provenance,
                      e.recipe_id        = $recipe_id,
                      e.recipe_version   = $recipe_version,
                      e.ingestion_time   = $ingestion_time,
                      e.identity_key     = row.identity_key
        SET e += row.properties
        """
        param_row: dict[str, Any] = {
            "id": entity_id,
            "identity_key": identity_key,
            "properties": props,
        }
        run_params: dict[str, Any] = {
            "batch": [param_row],
            "graph_id": graph_id,
            "source_id": source_id,
            "provenance": provenance,
            "recipe_id": meta["recipe_id"],
            "recipe_version": meta["recipe_version"],
            "ingestion_time": meta["ingestion_time"],
        }
        with driver.session() as session:
            session.run(cypher, run_params)

            # `confidence` only on INFERRED elements (unified-graph-model §5).
            if provenance == "INFERRED":
                session.run(
                    f"""
                    MATCH (e:{label}:__Entity__ {{graph_id: $graph_id, id: $id}})
                    SET e.confidence = coalesce(e.confidence, $confidence)
                    """,
                    {
                        "graph_id": graph_id,
                        "id": entity_id,
                        "confidence": rule.get("confidence", 0.5),
                    },
                )

            # DERIVED_FROM — every entity links back to where it came from
            # (its container, else the source). unified-graph-model §7.
            container_id = unit_to_container.get(unit.parent_id or "")
            if container_id is not None:
                session.run(
                    """
                    MATCH (e:__Entity__ {graph_id: $graph_id, id: $entity_id})
                    MATCH (c:__KGBuilder__ {graph_id: $graph_id, id: $container_id})
                    MERGE (e)-[:DERIVED_FROM {graph_id: $graph_id}]->(c)
                    """,
                    {
                        "graph_id": graph_id,
                        "entity_id": entity_id,
                        "container_id": container_id,
                    },
                )
            else:
                session.run(
                    """
                    MATCH (e:__Entity__ {graph_id: $graph_id, id: $entity_id})
                    MATCH (s:Source {graph_id: $graph_id, source_id: $source_id})
                    MERGE (e)-[:DERIVED_FROM {graph_id: $graph_id}]->(s)
                    """,
                    {
                        "graph_id": graph_id,
                        "entity_id": entity_id,
                        "source_id": source_id,
                    },
                )

        result.nodes_written += 1
        return entity_id

    # -- property projection (recipe-spec §5.3) ----------------------------

    def _apply_property(
        self,
        driver: Any,
        rule: dict[str, Any],
        unit: StructuralUnit,
        graph_id: str,
        node_rules: dict[str, dict[str, Any]],
        node_index: dict[tuple[str, str], str],
        payload_cache: dict[str, dict[str, Any]],
        result: ExecutionResult,
    ) -> None:
        """SET a property on every node the `on` node-rule produced.

        A `property` rule (recipe-spec §5.3) matches a `column`/`field` unit and
        declares the unit becomes a *property* on an already-mapped node rather
        than its own node. The value is read — per record — from the same record
        payload the `on` node was projected from, and SET on that node. This is
        consistent with the unified model: in a database a column's value
        travels with its row, and the row *is* the entity (unified-graph-model
        §2).
        """
        on_rule_id: str = rule["on"]
        if on_rule_id not in node_rules:
            result.warnings.append(
                f"property rule {rule['id']!r}: `on` references unknown node "
                f"rule {on_rule_id!r} — skipped"
            )
            return

        # value_from is optional in §5.3's shorthand; default to the rule name.
        value_ref = rule.get("value_from") or f"column:{rule['name']}"
        prop_name = rule["name"]

        targets: list[dict[str, Any]] = []
        for (rid, unit_id), entity_id in node_index.items():
            if rid != on_rule_id:
                continue
            payload = payload_cache.get(unit_id, {})
            value = self._read_field(payload, unit, value_ref)
            if value is not None:
                targets.append({"id": entity_id, "value": _coerce_value(value)})

        if not targets:
            return

        # `prop_name` passed _is_safe_identifier — safe to build into the query.
        cypher = f"""
        UNWIND $batch AS row
        MATCH (e:__Entity__ {{graph_id: $graph_id, id: row.id}})
        SET e.{prop_name} = row.value
        """
        for batch in _chunks(targets, _BATCH_SIZE):
            with driver.session() as session:
                session.run(cypher, {"batch": batch, "graph_id": graph_id})
            result.properties_written += len(batch)

    # -- edge projection (recipe-spec §5.2) --------------------------------

    def _resolve_fk_target_edges(
        self,
        rule: dict[str, Any],
        unit: StructuralUnit,
        from_rule_id: str,
        to_rule_id: str,
        node_index: dict[tuple[str, str], str],
        representation: StructuralRepresentation,
        payload_cache: dict[str, dict[str, Any]],
        result: ExecutionResult,
    ) -> list[dict[str, str]]:
        """Resolve `resolve_by: "fk_target"` edges (recipe-spec §5.2).

        *unit* is the foreign-key `column` unit the edge rule matched. It carries
        on its `metadata` (set by the relational primitive):

          * ``fk_target``        — the `unit_id` of the referenced `:Table` unit;
          * ``fk_target_column`` — the referenced column's name (the target row's
            key the FK value points at; defaults to ``id``).

        For each `from` row entity (a `record` unit under the FK column's own
        table), the FK column's value is read from that row and matched against
        the `to` row entities (the `record` units under the target table) whose
        referenced-column value equals it. unified-graph-model §2 — a row is the
        entity, so an FK value resolves a row, not a table.
        """
        fk_target_table: str | None = unit.metadata.get("fk_target")
        if not fk_target_table:
            result.warnings.append(
                f"edge rule {rule['id']!r}: matched unit {unit.unit_id!r} carries "
                f"no `fk_target` metadata — the column is not a resolvable "
                f"foreign key; edge skipped."
            )
            return []
        if not unit.metadata.get("fk_target_present", True):
            result.warnings.append(
                f"edge rule {rule['id']!r}: FK target table for unit "
                f"{unit.unit_id!r} is not present in the representation — "
                f"edge skipped."
            )
            return []

        fk_column_name: str | None = unit.name
        if not fk_column_name:
            result.warnings.append(
                f"edge rule {rule['id']!r}: matched FK unit {unit.unit_id!r} has "
                f"no column name to read the foreign-key value from — skipped."
            )
            return []
        ref_column: str = unit.metadata.get("fk_target_column", "id")
        source_table_id: str | None = unit.parent_id

        # Index the target table's row entities by their referenced-column
        # value. A `record` unit's parent_id is its `:Table` unit id.
        unit_by_id = {u.unit_id: u for u in representation.units}
        to_by_ref_value: dict[Any, str] = {}
        for (rid, target_unit_id), entity_id in node_index.items():
            if rid != to_rule_id:
                continue
            target_unit = unit_by_id.get(target_unit_id)
            if target_unit is None or target_unit.parent_id != fk_target_table:
                continue
            ref_value = payload_cache.get(target_unit_id, {}).get(ref_column)
            if ref_value is not None:
                to_by_ref_value[ref_value] = entity_id

        edges: list[dict[str, str]] = []
        unmatched = 0
        for (rid, source_unit_id), entity_id in node_index.items():
            if rid != from_rule_id:
                continue
            source_unit = unit_by_id.get(source_unit_id)
            # Only rows of the table that *owns* the FK column carry the FK
            # value — guard against a `from` rule matching another table.
            if source_unit is None or (
                source_table_id is not None and source_unit.parent_id != source_table_id
            ):
                continue
            fk_value = payload_cache.get(source_unit_id, {}).get(fk_column_name)
            if fk_value is None:
                continue  # NULL FK — no edge for this row, not an error.
            target_entity = to_by_ref_value.get(fk_value)
            if target_entity is None:
                unmatched += 1
                continue
            edges.append({"from": entity_id, "to": target_entity})

        if unmatched:
            result.warnings.append(
                f"edge rule {rule['id']!r}: {unmatched} foreign-key value(s) on "
                f"{unit.unit_id!r} matched no target row — dangling reference(s) "
                f"skipped."
            )
        return edges

    def _apply_edge(
        self,
        driver: Any,
        rule: dict[str, Any],
        unit: StructuralUnit,
        graph_id: str,
        meta: dict[str, Any],
        defaults: dict[str, Any],
        node_rules: dict[str, dict[str, Any]],
        node_index: dict[tuple[str, str], str],
        representation: StructuralRepresentation,
        payload_cache: dict[str, dict[str, Any]],
        result: ExecutionResult,
    ) -> None:
        """MERGE a typed relationship between two recipe-projected nodes.

        The matched unit is the edge-bearing structural unit (commonly a
        foreign-key `column`). `from.node_rule` / `to.node_rule` name the node
        rules whose nodes the edge connects; `to.resolve_by` says how the target
        node is found:

          * ``self``       — the same node the `from` rule produced for the row;
          * ``identity``   — the target node identified by its own identity;
          * ``fk_target``  — the target row referenced by the foreign key.

        `fk_target` (TASK-235): with the relational primitive now emitting one
        `record` unit per row, an FK column value on a `record` resolves to the
        target table's row entity. The matched FK `column` unit carries the
        target *table* unit id (`metadata.fk_target`) and the referenced
        column (`metadata.fk_target_column`); each source row's FK value is
        matched against the target rows' referenced-column values.
        """
        rel_type: str = rule["type"]
        from_rule_id: str = rule["from"]["node_rule"]
        to_rule_id: str = rule["to"]["node_rule"]
        resolve_by: str = rule["to"]["resolve_by"]
        provenance = rule.get("provenance") or defaults.get("provenance", "EXTRACTED")

        if from_rule_id not in node_rules or to_rule_id not in node_rules:
            result.warnings.append(
                f"edge rule {rule['id']!r}: references unknown node rule "
                f"({from_rule_id!r} / {to_rule_id!r}) — skipped"
            )
            return

        edges: list[dict[str, str]] = []

        if resolve_by == "self":
            # A self-loop is rarely meaningful; supported for completeness.
            for (rid, _unit_id), entity_id in node_index.items():
                if rid == from_rule_id:
                    edges.append({"from": entity_id, "to": entity_id})
        elif resolve_by == "identity":
            # Connect each `from` node to the `to` node sharing the same record
            # payload — both node rules projected from the same record unit.
            to_by_unit = {
                unit_id: eid
                for (rid, unit_id), eid in node_index.items()
                if rid == to_rule_id
            }
            for (rid, unit_id), entity_id in node_index.items():
                if rid != from_rule_id:
                    continue
                target = to_by_unit.get(unit_id)
                if target is not None:
                    edges.append({"from": entity_id, "to": target})
        else:  # fk_target
            edges = self._resolve_fk_target_edges(
                rule,
                unit,
                from_rule_id,
                to_rule_id,
                node_index,
                representation,
                payload_cache,
                result,
            )

        if not edges:
            return

        # `rel_type` passed _is_safe_identifier — safe to build into the query.
        cypher = f"""
        UNWIND $batch AS row
        MATCH (a:__Entity__ {{graph_id: $graph_id, id: row.from}})
        MATCH (b:__Entity__ {{graph_id: $graph_id, id: row.to}})
        MERGE (a)-[r:{rel_type} {{graph_id: $graph_id}}]->(b)
        ON CREATE SET r.ingestion_source = $source_id,
                      r.provenance       = $provenance,
                      r.recipe_id        = $recipe_id,
                      r.recipe_version   = $recipe_version,
                      r.ingestion_time   = $ingestion_time
        """
        for batch in _chunks(edges, _BATCH_SIZE):
            with driver.session() as session:
                session.run(
                    cypher,
                    {
                        "batch": batch,
                        "graph_id": graph_id,
                        "source_id": meta["ingestion_source"],
                        "provenance": provenance,
                        "recipe_id": meta["recipe_id"],
                        "recipe_version": meta["recipe_version"],
                        "ingestion_time": meta["ingestion_time"],
                    },
                )
            result.edges_written += len(batch)

    # -- text extraction (recipe-spec §9) — OUT OF SCOPE for TASK-223 ------

    def _apply_text_extraction(
        self, rule: dict[str, Any], unit: StructuralUnit, result: ExecutionResult
    ) -> None:
        """Recognise a `text_extraction` rule and skip it — deferred.

        TODO(TASK-223 follow-up): `text_extraction` invokes a named, possibly
        LLM-backed text-extraction primitive (recipe-spec §9). That primitive
        does not exist yet — it is a separate deliverable. The engine MUST NOT
        invoke an LLM here. When the primitive lands, this method:
          1. resolves the named primitive,
          2. runs it over the unit's free-text payload (it returns entities and
             relations as *data*; it never writes the graph),
          3. has the deterministic engine MERGE those results — `graph_id`-
             scoped, §5.5-checked — tagged `INFERRED` with a confidence,
          4. links them to the `link_to.node_rule` node via `link_to.edge_type`.

        For TASK-223 the rule is recognised, skipped, and reported as a warning
        so the run is observably incomplete rather than silently wrong.
        """
        result.units_skipped += 1
        result.warnings.append(
            f"text_extraction rule {rule['id']!r} (primitive "
            f"{rule['primitive']!r}) skipped for unit {unit.unit_id!r}: the "
            f"LLM-backed text-extraction primitive is not implemented "
            f"(out of scope for TASK-223)."
        )
        logger.info(
            "recipe_engine: text_extraction rule %s skipped — primitive %s "
            "not implemented (TASK-223 scope)",
            rule["id"],
            rule["primitive"],
        )
