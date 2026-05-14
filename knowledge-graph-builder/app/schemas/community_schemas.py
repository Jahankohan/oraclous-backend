"""
Public-facing community schemas (TASK-050).

The richer internal community shapes live in `graph_schemas.py`
(`CommunityItem`, `CommunityDetailResponse`, etc.).  This module owns the
flat list shape the frontend consumes via `Community[]` in
`oraclous-visual-flow-main/src/lib/api.ts`.
"""

from pydantic import BaseModel, Field


class Community(BaseModel):
    """A single community summary across all registered kinds.

    Field names match the frontend `Community` interface for the entity
    (Leiden) case so the API client mounts without remapping. The added
    ``kind`` and ``member_label`` fields let new clients distinguish
    entity-level from chunk-level communities; older clients can ignore
    them since both default-back to entity behavior.
    """

    community_id: str = Field(..., description="Stable community identifier")
    kind: str = Field(
        default="entity",
        description=(
            "Community kind from the registry. ``entity`` is the Leiden "
            "output over `__Entity__` nodes (hierarchical, multi-level); "
            "``chunk`` is the Louvain output over `:Chunk` nodes (flat). "
            "Discover the full list via GET /communities/kinds."
        ),
    )
    level: int = Field(
        ...,
        description=(
            "Hierarchy level (0 = coarsest, higher integers = finer). "
            "Always 0 for flat kinds where the algorithm has no hierarchy."
        ),
    )
    label: str = Field(
        ...,
        description=(
            "Short human-readable label.  Derived from the first sentence "
            "of `summary` when present, otherwise `Community <short-id>`."
        ),
    )
    size: int = Field(..., ge=0, description="Number of members")
    member_label: str | None = Field(
        default=None,
        description=(
            "Neo4j label of the member nodes of this community "
            "(`__Entity__` for entity communities, `Chunk` for chunk "
            "communities). Lets callers route lookups without consulting "
            "the kinds registry."
        ),
    )
    summary: str | None = Field(
        default=None,
        description="Optional LLM-generated summary of the community's members",
    )
    summary_keywords: list[str] | None = Field(
        default=None,
        description=(
            "Key entities / concepts / topics extracted alongside the "
            "summary. Populated for chunk communities by STORY-4b's "
            "summarize endpoint; entity-Leiden summaries don't have this "
            "field yet."
        ),
    )
    summary_excerpt: str | None = Field(
        default=None,
        description=(
            "Up to ~500 chars from one representative member, quoted "
            "verbatim. Lets agent tools return concrete evidence without "
            "an extra round trip to fetch members."
        ),
    )


class CommunityKindInfo(BaseModel):
    """Registry-facing description of one community kind.

    Returned from ``GET /communities/kinds`` so clients can drive their UI
    off the same source of truth that backend services use, instead of
    hardcoding kind names or labels in JS.
    """

    kind: str = Field(..., description="Stable wire identifier for the kind")
    display_name: str = Field(..., description="Human-readable name for UI labels")
    community_label: str = Field(
        ..., description="Neo4j label of the community node itself"
    )
    member_label: str = Field(..., description="Neo4j label of the member nodes")
    hierarchical: bool = Field(
        ...,
        description=(
            "True when the algorithm produces multiple hierarchy levels "
            "(Leiden); False for flat clusterings (Louvain)."
        ),
    )
    detection_supported: bool = Field(
        ...,
        description=(
            "False when the kind is read-only — i.e., no Celery task is "
            "wired yet and POST /communities/detect returns HTTP 405."
        ),
    )
