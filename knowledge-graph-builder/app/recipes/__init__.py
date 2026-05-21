"""Concern-driven ingestion recipes (ADR-022) — schema, examples, and primitives."""

from app.recipes.engine import (
    ExecutionResult,
    RecipeExecutionEngine,
    RecipeValidationError,
)

__all__ = [
    "ExecutionResult",
    "RecipeExecutionEngine",
    "RecipeValidationError",
]
