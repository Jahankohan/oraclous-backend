"""
Public-facing community schemas (TASK-050).

The richer internal community shapes live in `graph_schemas.py`
(`CommunityItem`, `CommunityDetailResponse`, etc.).  This module owns the
flat list shape the frontend consumes via `Community[]` in
`oraclous-visual-flow-main/src/lib/api.ts`.
"""

from pydantic import BaseModel, Field


class Community(BaseModel):
    """A single Leiden community summary.

    Field names match the frontend `Community` interface exactly so the
    API client mounts without remapping.
    """

    community_id: str = Field(..., description="Stable community identifier")
    level: int = Field(
        ..., description="Hierarchy level (0 = coarsest, higher integers = finer)"
    )
    label: str = Field(
        ...,
        description=(
            "Short human-readable label.  Derived from the first sentence "
            "of `summary` when present, otherwise `Community <short-id>`."
        ),
    )
    size: int = Field(..., ge=0, description="Number of member entities")
    summary: str | None = Field(
        default=None,
        description="Optional LLM-generated summary of the community's members",
    )
