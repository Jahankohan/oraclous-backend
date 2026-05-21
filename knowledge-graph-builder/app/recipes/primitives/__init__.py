"""Deterministic, per-data-type primitives for concern-driven ingestion (ADR-022).

A primitive mechanically decomposes a data source into a normalized
`StructuralRepresentation`. See `interface.py` for the contract and TASK-222.
"""

from app.recipes.primitives.interface import (
    ExtractionMode,
    Primitive,
    StructuralRepresentation,
    StructuralUnit,
    UnitKind,
)

__all__ = [
    "ExtractionMode",
    "Primitive",
    "StructuralRepresentation",
    "StructuralUnit",
    "UnitKind",
]
