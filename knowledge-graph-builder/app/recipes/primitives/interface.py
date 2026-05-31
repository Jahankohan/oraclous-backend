"""Primitive interface for concern-driven ingestion.

TASK-222 / STORY-034 / ADR-022.

A *primitive* is the deterministic, per-data-type first stage of ingestion. It
mechanically decomposes a data source into a normalized
`StructuralRepresentation` — a flat list of `StructuralUnit`s — with **no LLM and
no concern awareness**. Recipes (TASK-220) are authored and executed against
this representation; the recipe execution engine (TASK-223) and the
recipe-authoring loop (TASK-225) are its consumers.

The `UnitKind` and `role` vocabularies are the *structural-unit vocabulary* that
`docs/recipe-spec.md` §5 references in recipe `match` clauses. TASK-220 owns that
vocabulary; primitives emit units that conform to it.
"""

from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ExtractionMode(str, Enum):
    """Why a primitive is being run."""

    SAMPLE = "sample"
    """Design time — emit the full structure but only bounded example values.
    The data-specialist agent authors a recipe from this output."""

    FULL = "full"
    """Run time — emit the complete representation for the recipe execution
    engine to project mechanically over the whole source."""


class UnitKind(str, Enum):
    """The canonical kinds of structural unit a primitive may emit.

    A recipe rule's ``match.unit_kind`` (recipe-spec §5) is matched against this
    set. It is owned by TASK-220 and extended only deliberately.
    """

    SOURCE = "source"
    TABLE = "table"
    SHEET = "sheet"
    COLUMN = "column"
    RECORD = "record"
    FIELD = "field"
    DOCUMENT = "document"
    CHUNK = "chunk"
    FILE = "file"
    SYMBOL = "symbol"


class StructuralUnit(BaseModel):
    """One structural unit of a source — the atom a recipe rule matches against."""

    kind: UnitKind
    unit_id: str = Field(..., description="Stable, path-like id within the source.")
    name: str | None = Field(
        default=None, description="Human name — a table/column/field/symbol name."
    )
    data_type: str | None = Field(
        default=None, description="Inferred type of the unit's values, if any."
    )
    role: str | None = Field(
        default=None,
        description=(
            "A hint a recipe rule matches on. Common values: 'primary_key', "
            "'foreign_key', 'free_text', 'measure', 'timestamp'."
        ),
    )
    parent_id: str | None = Field(
        default=None,
        description="unit_id of the containing unit — the containment edge.",
    )
    sample_values: list[Any] = Field(
        default_factory=list,
        description="Example values — bounded in SAMPLE mode, complete in FULL mode.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Per-kind extras, e.g. {'fk_target': '<unit_id>'} on a foreign-key "
            "column. Recipes read this for edge resolution (recipe-spec §5.2)."
        ),
    )


class StructuralRepresentation(BaseModel):
    """A primitive's normalized output — deterministic and concern-agnostic."""

    source_type: str = Field(
        ..., description="relational | csv | json | text | code | ..."
    )
    shape_signature: str = Field(
        ...,
        description=(
            "Deterministic descriptor of the source's shape — the recipe "
            "library's lookup key (recipe-spec §4)."
        ),
    )
    mode: ExtractionMode
    units: list[StructuralUnit] = Field(default_factory=list)


@runtime_checkable
class Primitive(Protocol):
    """A deterministic, per-data-type decomposer — one primitive per data type.

    Adding a new data *type* may add a primitive (rare; the set is finite). A
    new *concern* never adds a primitive — that is a recipe (ADR-022).
    """

    source_type: str

    def decompose(self, source: Any, mode: ExtractionMode) -> StructuralRepresentation:
        """Mechanically decompose ``source`` into a `StructuralRepresentation`.

        Deterministic — no LLM, no concern awareness. In ``SAMPLE`` mode, emit
        the full structure with only bounded example values; in ``FULL`` mode,
        emit the complete representation.
        """
        ...
