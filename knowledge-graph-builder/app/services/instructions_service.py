"""
Instructions Service — User Context & Instructions System (ORA-8)

Implements:
- InstructionsResolver: reads GraphInstructions from Neo4j, merges with IngestionOverrides
- InstructionsCompiler: converts ResolvedInstructions to an LLM prompt prefix string
- InstructionsService: CRUD for storing/retrieving instructions on the Graph node
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client
from app.schemas.graph_schemas import (
    EntityTypeDefinition,
    ExtractionDensity,
    GraphInstructions,
    GraphInstructionsResponse,
    IngestionOverrides,
    OntologyPatchRequest,
    OntologyResponse,
    OntologySetRequest,
    OntologyValidationMode,
    RelationshipTypeDefinition,
)

logger = get_logger(__name__)


# ==================== RESOLVED INSTRUCTIONS ====================


@dataclass
class ResolvedInstructions:
    """
    Final merged extraction instructions for one ingestion job.
    Not persisted — computed fresh per job from GraphInstructions + IngestionOverrides.
    """

    domain: str | None = None
    extraction_density: ExtractionDensity = ExtractionDensity.BALANCED
    entity_types: list[EntityTypeDefinition] | None = None
    relationship_types: list[RelationshipTypeDefinition] | None = None
    edge_property_fields: list[str] | None = None
    focus_areas: list[str] | None = field(default_factory=list)
    ignore_patterns: list[str] | None = None
    language: str = "en"
    custom_prompt_suffix: str | None = None
    ontology_mode: OntologyValidationMode = OntologyValidationMode.WARN


# ==================== INSTRUCTIONS RESOLVER ====================


class InstructionsResolver:
    """
    Reads GraphInstructions from Neo4j, merges with IngestionOverrides.

    Merge rules (per ORA-5 spec):
    - overrides.override_density  → replaces graph.extraction_density if set
    - overrides.additional_focus  → appended to focus_areas list
    - overrides.extra_entity_types → appended to entity_types if any
    - overrides.schema_evolution_hint → appended to custom_prompt_suffix
    """

    async def resolve(
        self,
        graph_id: str,
        overrides: IngestionOverrides | None = None,
    ) -> ResolvedInstructions:
        """
        Load graph-level instructions from Neo4j and merge with per-job overrides.
        If no instructions are configured, returns defaults (free-form extraction).
        """
        graph_instructions = await self._load_from_neo4j(graph_id)

        if graph_instructions is None:
            # No instructions stored — use defaults and apply any overrides
            resolved = ResolvedInstructions()
        else:
            resolved = ResolvedInstructions(
                domain=graph_instructions.domain,
                extraction_density=graph_instructions.extraction_density,
                entity_types=(
                    list(graph_instructions.entity_types)
                    if graph_instructions.entity_types
                    else None
                ),
                relationship_types=(
                    list(graph_instructions.relationship_types)
                    if graph_instructions.relationship_types
                    else None
                ),
                edge_property_fields=(
                    list(graph_instructions.edge_property_fields)
                    if graph_instructions.edge_property_fields
                    else None
                ),
                focus_areas=(
                    list(graph_instructions.focus_areas)
                    if graph_instructions.focus_areas
                    else []
                ),
                ignore_patterns=(
                    list(graph_instructions.ignore_patterns)
                    if graph_instructions.ignore_patterns
                    else None
                ),
                language=graph_instructions.language or "en",
                custom_prompt_suffix=graph_instructions.custom_prompt_suffix,
                ontology_mode=graph_instructions.ontology_mode,
            )

        if overrides is None:
            return resolved

        # Apply overrides
        if overrides.override_density is not None:
            resolved.extraction_density = overrides.override_density

        if overrides.additional_focus is not None:
            if resolved.focus_areas is None:
                resolved.focus_areas = []
            resolved.focus_areas.append(overrides.additional_focus)

        if overrides.extra_entity_types:
            if resolved.entity_types is None:
                resolved.entity_types = []
            resolved.entity_types.extend(overrides.extra_entity_types)

        if overrides.schema_evolution_hint is not None:
            existing = resolved.custom_prompt_suffix or ""
            resolved.custom_prompt_suffix = (
                f"{existing}\n{overrides.schema_evolution_hint}".strip()
                if existing
                else overrides.schema_evolution_hint
            )

        return resolved

    async def _load_from_neo4j(self, graph_id: str) -> GraphInstructions | None:
        """Read instructions_config JSON from the Graph node."""
        query = """
        MATCH (g:Graph {graph_id: $graph_id})
        RETURN g.instructions_config AS instructions_config
        """
        try:
            records = await neo4j_client.execute_query(query, {"graph_id": graph_id})
            if not records:
                return None
            raw = records[0].get("instructions_config")
            if not raw:
                return None
            data = json.loads(raw) if isinstance(raw, str) else raw
            return GraphInstructions(**data)
        except Exception as e:
            logger.warning(f"Could not load instructions for graph {graph_id}: {e}")
            return None


# ==================== INSTRUCTIONS COMPILER ====================


class InstructionsCompiler:
    """
    Converts ResolvedInstructions into an LLM prompt prefix string.

    Omits blocks when fields are empty/None so the base extraction prompt
    is not cluttered in free-form mode.
    """

    def to_prompt(self, resolved: ResolvedInstructions) -> str:
        """Build and return the prompt prefix. Returns empty string for all-default instructions."""
        blocks: list[str] = []

        if resolved.domain:
            blocks.append(
                f"## Extraction Context\n"
                f"This text is from the domain: {resolved.domain}.\n"
                f"Apply domain-specific extraction conventions for this domain."
            )

        if resolved.entity_types:
            lines = ["## Entity Types", "Extract ONLY entities of the following types:"]
            for et in resolved.entity_types:
                desc = f": {et.description}" if et.description else ""
                examples = (
                    f" Examples: {', '.join(et.examples)}." if et.examples else ""
                )
                props = ""
                if et.properties:
                    prop_list = ", ".join(
                        f"{k} ({v})" for k, v in et.properties.items()
                    )
                    props = f" Properties to capture: {prop_list}."
                lines.append(f"- {et.name}{desc}.{examples}{props}")
            lines.append("Do NOT create entity types not listed above.")
            blocks.append("\n".join(lines))

        if resolved.relationship_types:
            lines = ["## Relationship Types", "Use ONLY these relationship types:"]
            for rt in resolved.relationship_types:
                src = rt.source_type or "Entity"
                tgt = rt.target_type or "Entity"
                edge_props = ""
                # Support both new `properties` field and deprecated `store_as_edge_property`
                stored_props = rt.properties or rt.store_as_edge_property
                if stored_props:
                    edge_props = f"\n  Store {stored_props} as properties ON THIS RELATIONSHIP, not on the nodes."
                lines.append(f"- ({src})-[{rt.name}]->({tgt}){edge_props}")
            blocks.append("\n".join(lines))

        rules = self._build_rules_block(resolved)
        if rules:
            blocks.append(rules)

        if resolved.focus_areas:
            lines = ["## Focus Areas", "Pay special attention to:"]
            for area in resolved.focus_areas:
                lines.append(f"- {area}")
            blocks.append("\n".join(lines))

        return "\n\n".join(blocks)

    def build_schema_block(self, resolved: ResolvedInstructions) -> str:
        """
        Build the plain-text schema block for pre-rendering the `{schema}` placeholder
        in the extraction prompt template.

        Returns an empty string when no ontology is configured (free-form mode).
        """
        if not resolved.entity_types and not resolved.relationship_types:
            return ""

        parts: list[str] = []
        if resolved.entity_types:
            node_types = ", ".join(et.name for et in resolved.entity_types)
            parts.append(f"Node types: {node_types}")
        if resolved.relationship_types:
            rel_types = ", ".join(rt.name for rt in resolved.relationship_types)
            parts.append(f"Relationship types: {rel_types}")
        return "\n".join(parts)

    def _build_rules_block(self, resolved: ResolvedInstructions) -> str:
        density_desc = {
            ExtractionDensity.SPARSE: "sparse = major entities only; prefer precision over recall",
            ExtractionDensity.BALANCED: "balanced = meaningful entities; include supporting entities when relevant",
            ExtractionDensity.DENSE: "dense = all entity mentions; maximize recall",
        }
        lines = [
            "## Extraction Rules",
            f"- Extraction density: {resolved.extraction_density.value}",
            f"  {density_desc[resolved.extraction_density]}",
            f"- Language: {resolved.language}. Normalize entity names to canonical form.",
        ]

        if resolved.edge_property_fields:
            lines.append(
                f"- These properties MUST be stored on relationships, not nodes: {resolved.edge_property_fields}"
            )

        if resolved.ignore_patterns:
            lines.append(f"- Ignore text matching: {resolved.ignore_patterns}")

        if resolved.custom_prompt_suffix:
            lines.append(resolved.custom_prompt_suffix)

        return "\n".join(lines)


# ==================== INSTRUCTIONS SERVICE ====================


class InstructionsService:
    """
    CRUD for graph-level extraction instructions.
    Stores on the Neo4j Graph node (authoritative) and in PostgreSQL schema_config (cache).
    """

    async def set_instructions(
        self, graph_id: str, instructions: GraphInstructions
    ) -> GraphInstructionsResponse:
        """Store or replace GraphInstructions on the Graph node. Increments version."""
        config_json = instructions.model_dump_json()
        now = datetime.now(UTC)

        query = """
        MATCH (g:Graph {graph_id: $graph_id})
        SET g.instructions_config = $config,
            g.instructions_version = COALESCE(g.instructions_version, 0) + 1,
            g.instructions_updated_at = datetime($updated_at)
        RETURN g.instructions_version AS version
        """
        records = await neo4j_client.execute_query(
            query,
            {
                "graph_id": graph_id,
                "config": config_json,
                "updated_at": now.isoformat(),
            },
        )

        version = records[0]["version"] if records else 1
        logger.info(f"Set instructions v{version} for graph {graph_id}")

        return GraphInstructionsResponse(
            graph_id=graph_id,  # type: ignore[arg-type]
            instructions=instructions,
            version=version,
            updated_at=now,
        )

    async def get_instructions(self, graph_id: str) -> GraphInstructionsResponse | None:
        """Fetch GraphInstructions from the Graph node. Returns None if not configured."""
        query = """
        MATCH (g:Graph {graph_id: $graph_id})
        RETURN g.instructions_config AS config,
               g.instructions_version AS version,
               g.instructions_updated_at AS updated_at
        """
        records = await neo4j_client.execute_query(query, {"graph_id": graph_id})
        if not records:
            return None

        row = records[0]
        raw_config = row.get("config")
        if not raw_config:
            return None

        try:
            data = json.loads(raw_config) if isinstance(raw_config, str) else raw_config
            instructions = GraphInstructions(**data)
        except Exception as e:
            logger.error(
                f"Failed to deserialize instructions for graph {graph_id}: {e}"
            )
            return None

        raw_updated_at = row.get("updated_at")
        if hasattr(raw_updated_at, "to_native"):
            updated_at = raw_updated_at.to_native()
        elif isinstance(raw_updated_at, str):
            updated_at = datetime.fromisoformat(raw_updated_at)
        else:
            updated_at = datetime.now(UTC)

        return GraphInstructionsResponse(
            graph_id=graph_id,  # type: ignore[arg-type]
            instructions=instructions,
            version=row.get("version") or 1,
            updated_at=updated_at,
        )

    async def delete_instructions(self, graph_id: str) -> None:
        """Remove all instruction properties from the Graph node."""
        query = """
        MATCH (g:Graph {graph_id: $graph_id})
        REMOVE g.instructions_config, g.instructions_version, g.instructions_updated_at
        """
        await neo4j_client.execute_query(query, {"graph_id": graph_id})
        logger.info(f"Deleted instructions for graph {graph_id}")

    # ==================== ONTOLOGY CRUD ====================

    async def get_ontology(self, graph_id: str) -> OntologyResponse | None:
        """
        Return the ontology section of the graph's instructions.
        Returns None if no instructions are set or no ontology is configured.
        """
        result = await self.get_instructions(graph_id)
        if result is None or not result.instructions.entity_types:
            return None
        inst = result.instructions
        return OntologyResponse(
            graph_id=graph_id,  # type: ignore[arg-type]
            entity_types=inst.entity_types or [],
            relationship_types=inst.relationship_types or [],
            ontology_mode=inst.ontology_mode,
            version=result.version,
            updated_at=result.updated_at,
        )

    async def set_ontology(
        self, graph_id: str, request: OntologySetRequest
    ) -> OntologyResponse:
        """
        Replace the ontology fields on the graph's instructions.
        Creates instructions if none exist; preserves all non-ontology fields.
        Invalidates schema_manager cache.
        """
        existing = await self.get_instructions(graph_id)
        if existing is not None:
            instructions = existing.instructions.model_copy(
                update={
                    "entity_types": request.entity_types,
                    "relationship_types": request.relationship_types or [],
                    "ontology_mode": request.ontology_mode,
                }
            )
        else:
            instructions = GraphInstructions(
                entity_types=request.entity_types,
                relationship_types=request.relationship_types,
                ontology_mode=request.ontology_mode,
            )
        response = await self.set_instructions(graph_id, instructions)
        await self._invalidate_schema_cache(graph_id)
        return OntologyResponse(
            graph_id=graph_id,  # type: ignore[arg-type]
            entity_types=instructions.entity_types or [],
            relationship_types=instructions.relationship_types or [],
            ontology_mode=instructions.ontology_mode,
            version=response.version,
            updated_at=response.updated_at,
        )

    async def patch_ontology(
        self, graph_id: str, patch: OntologyPatchRequest
    ) -> OntologyResponse:
        """
        Merge-update the ontology: add/remove individual types, update mode.
        Raises ValueError if no ontology is configured and nothing to add.
        Invalidates schema_manager cache.
        """
        existing = await self.get_instructions(graph_id)
        if existing is not None:
            instructions = existing.instructions
        else:
            instructions = GraphInstructions()

        entity_types: list[EntityTypeDefinition] = list(instructions.entity_types or [])
        rel_types: list[RelationshipTypeDefinition] = list(
            instructions.relationship_types or []
        )

        if patch.remove_entity_types:
            remove_set = set(patch.remove_entity_types)
            entity_types = [et for et in entity_types if et.name not in remove_set]

        if patch.add_entity_types:
            existing_names = {et.name for et in entity_types}
            for et in patch.add_entity_types:
                if et.name in existing_names:
                    entity_types = [
                        et if e.name == et.name else e for e in entity_types
                    ]
                else:
                    entity_types.append(et)

        if patch.remove_relationship_types:
            remove_set = set(patch.remove_relationship_types)
            rel_types = [rt for rt in rel_types if rt.name not in remove_set]

        if patch.add_relationship_types:
            existing_names = {rt.name for rt in rel_types}
            for rt in patch.add_relationship_types:
                if rt.name in existing_names:
                    rel_types = [rt if r.name == rt.name else r for r in rel_types]
                else:
                    rel_types.append(rt)

        mode = (
            patch.ontology_mode
            if patch.ontology_mode is not None
            else instructions.ontology_mode
        )

        updated = instructions.model_copy(
            update={
                "entity_types": entity_types or None,
                "relationship_types": rel_types or None,
                "ontology_mode": mode,
            }
        )
        response = await self.set_instructions(graph_id, updated)
        await self._invalidate_schema_cache(graph_id)
        return OntologyResponse(
            graph_id=graph_id,  # type: ignore[arg-type]
            entity_types=entity_types,
            relationship_types=rel_types,
            ontology_mode=mode,
            version=response.version,
            updated_at=response.updated_at,
        )

    async def delete_ontology(self, graph_id: str) -> None:
        """
        Clear ontology fields (entity_types, relationship_types, ontology_mode)
        while preserving all other instruction fields.
        Invalidates schema_manager cache.
        """
        existing = await self.get_instructions(graph_id)
        if existing is None:
            return
        updated = existing.instructions.model_copy(
            update={
                "entity_types": None,
                "relationship_types": None,
                "ontology_mode": OntologyValidationMode.WARN,
            }
        )
        await self.set_instructions(graph_id, updated)
        await self._invalidate_schema_cache(graph_id)
        logger.info(f"Deleted ontology for graph {graph_id}")

    async def _invalidate_schema_cache(self, graph_id: str) -> None:
        """Invalidate the schema_manager cache for this graph after ontology changes."""
        try:
            from app.services.schema_service import schema_manager

            schema_manager.clear_cache(graph_id)
        except Exception:
            pass  # Non-critical — cache will expire naturally


# ==================== GLOBAL INSTANCES ====================

instructions_resolver = InstructionsResolver()
instructions_compiler = InstructionsCompiler()
instructions_service = InstructionsService()
