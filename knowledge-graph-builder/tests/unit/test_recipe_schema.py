"""Schema-validation tests for the ingestion recipe format.

TASK-220 / STORY-034 / ADR-022. Validates that the recipe JSON Schema accepts
the bundled example recipes and rejects malformed recipes — including the two
identifier-safety guards (ADR-015 reserved namespace, Cypher-safe identifiers)
and the mandatory INFERRED provenance on text_extraction rules.
"""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

_RECIPES_DIR = Path(__file__).resolve().parents[2] / "app" / "recipes"
_SCHEMA_PATH = _RECIPES_DIR / "recipe.schema.json"
_EXAMPLES_DIR = _RECIPES_DIR / "examples"


def _schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text())


def _validator() -> Draft202012Validator:
    return Draft202012Validator(_schema())


def _errors(recipe: dict) -> list:
    return list(_validator().iter_errors(recipe))


def _org_chart_recipe() -> dict:
    """A fresh, valid baseline recipe to mutate per test."""
    return json.loads((_EXAMPLES_DIR / "org-chart-relational.recipe.json").read_text())


def _freetext_recipe() -> dict:
    return json.loads(
        (_EXAMPLES_DIR / "eurail-evidence-freetext.recipe.json").read_text()
    )


def test_schema_is_itself_valid():
    Draft202012Validator.check_schema(_schema())


@pytest.mark.parametrize(
    "example", sorted(_EXAMPLES_DIR.glob("*.recipe.json")), ids=lambda p: p.name
)
def test_example_recipes_validate(example):
    errors = _errors(json.loads(example.read_text()))
    assert not errors, [e.message for e in errors]


def test_rejects_unknown_top_level_key():
    recipe = _org_chart_recipe()
    recipe["unexpected"] = True
    assert _errors(recipe)


def test_rejects_missing_required_field():
    recipe = _org_chart_recipe()
    del recipe["concern"]
    assert _errors(recipe)


def test_rejects_reserved_namespace_label():
    recipe = _org_chart_recipe()
    recipe["mappings"][0]["label"] = "__Platform__"
    assert _errors(recipe)


def test_rejects_cypher_unsafe_label():
    recipe = _org_chart_recipe()
    recipe["mappings"][0]["label"] = "Bad-Label"
    assert _errors(recipe)


def test_rejects_unknown_project_to():
    recipe = _org_chart_recipe()
    recipe["mappings"][0]["project_to"] = "teleport"
    assert _errors(recipe)


def test_rejects_empty_mappings():
    recipe = _org_chart_recipe()
    recipe["mappings"] = []
    assert _errors(recipe)


def test_text_extraction_must_be_inferred():
    recipe = _freetext_recipe()
    for rule in recipe["mappings"]:
        if rule["project_to"] == "text_extraction":
            rule["provenance"] = "EXTRACTED"
    assert _errors(recipe), "text_extraction with EXTRACTED provenance must be rejected"
