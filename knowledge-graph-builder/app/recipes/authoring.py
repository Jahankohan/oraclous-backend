"""The design-time recipe-authoring loop (TASK-225, STORY-034, ADR-022).

Concern-driven ingestion has a three-step authoring loop:

  1. **sample** — take a sample of a data source. A primitive (TASK-222) is run
     in ``ExtractionMode.SAMPLE`` to produce a `StructuralRepresentation` with
     the full structure but only bounded example values. This module's
     `sample_source` does this step.
  2. **author** — the `data-specialist` agent reads the sample plus a
     natural-language concern and *writes* a recipe. This is **agent
     reasoning, not code**. Nothing in this module generates, infers, or
     drafts a recipe — that step is the agent, and the agent only.
  3. **save** — the agent-authored recipe is stored in the recipe library
     (TASK-224) as a draft version. This module's `save_recipe_draft` does
     this step, a thin wrapper over `RecipeLibrary.store`.

Steps 1 and 3 are deterministic code. Step 2 is the agent. The CLI exposed by
``python -m app.recipes.authoring`` is the seam the `/ds` skill drives: it runs
`sample` (the agent reads the JSON it prints), then the agent authors a
``recipe.json``, then it runs `save`.

The primitive **registry** maps an explicit ``kind`` string to a primitive
instance. The key is a ``kind``, not a `source_type`: `MarkdownPrimitive` and
`DocumentPrimitive` both declare ``source_type = "text"``, so `source_type`
alone cannot disambiguate them — the caller states the kind.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from app.recipes.primitives import (
    CodePrimitive,
    CsvPrimitive,
    DocumentPrimitive,
    JsonPrimitive,
    MarkdownPrimitive,
    Primitive,
    RelationalPrimitive,
)
from app.recipes.primitives.interface import ExtractionMode, StructuralRepresentation

# ---------------------------------------------------------------------------
# Primitive registry — explicit `kind` -> primitive instance
# ---------------------------------------------------------------------------
#
# The key is a deliberate, explicit `kind` (not `source_type`): both the
# markdown and document primitives declare `source_type = "text"`, so the
# registry must distinguish them by an authoritative caller-supplied kind.
PRIMITIVE_REGISTRY: dict[str, Primitive] = {
    "csv": CsvPrimitive(),
    "json": JsonPrimitive(),
    "relational": RelationalPrimitive(),
    "code": CodePrimitive(),
    "markdown": MarkdownPrimitive(),
    "document": DocumentPrimitive(),
}


class UnknownPrimitiveKindError(ValueError):
    """The requested primitive ``kind`` is not in the registry."""


def _get_primitive(kind: str) -> Primitive:
    """Look up the primitive registered for ``kind``.

    Raises `UnknownPrimitiveKindError` — naming the supported kinds — for a
    kind the registry does not know.
    """
    try:
        return PRIMITIVE_REGISTRY[kind]
    except KeyError:
        supported = ", ".join(sorted(PRIMITIVE_REGISTRY))
        raise UnknownPrimitiveKindError(
            f"unknown primitive kind {kind!r}; supported kinds: {supported}"
        ) from None


# ---------------------------------------------------------------------------
# Step 1 — sample
# ---------------------------------------------------------------------------


def sample_source(source: Any, kind: str) -> StructuralRepresentation:
    """Sample a data *source* with the primitive registered for *kind*.

    This is step 1 of the authoring loop: the primitive is run in
    ``ExtractionMode.SAMPLE``, so the returned `StructuralRepresentation`
    carries the source's full structure but only bounded example values. The
    `data-specialist` agent reads this representation to author a recipe — no
    LLM and no concern awareness are involved in producing it.

    Args:
        source: The data source, in whatever form the chosen primitive
            accepts (a CSV/JSON/document file path, a `FileMetadata` for code,
            a `SchemaSnapshot` for relational, markdown text or a path).
        kind: One of the registry keys — ``csv``, ``json``, ``relational``,
            ``code``, ``markdown``, ``document``.

    Returns:
        The SAMPLE-mode `StructuralRepresentation`.

    Raises:
        UnknownPrimitiveKindError: if *kind* is not a registered primitive.
    """
    primitive = _get_primitive(kind)
    return primitive.decompose(source, ExtractionMode.SAMPLE)


# ---------------------------------------------------------------------------
# Step 3 — save
# ---------------------------------------------------------------------------


async def save_recipe_draft(
    recipe: dict[str, Any], graph_id: str, session: Any
) -> dict[str, Any]:
    """Save an agent-authored *recipe* as a draft version (step 3 of the loop).

    A thin wrapper over `RecipeLibrary.store`: the library validates the recipe
    against ``recipe.schema.json``, assigns the next version for the
    ``(id, graph_id)`` pair, and persists it with ``status = "draft"``.
    Validation and versioning live in the library — this function does not
    duplicate them and does not generate or alter the recipe.

    The caller owns the transaction. `RecipeLibrary.store` flushes the new row
    so it is visible within the transaction but does not commit; the CLI's
    `save` command commits explicitly.

    Args:
        recipe: The recipe document the `data-specialist` agent authored.
        graph_id: The authoring tenant.
        session: An open `AsyncSession`.

    Returns:
        The stored recipe document, with the library-assigned ``version`` and
        ``status = "draft"``.

    Raises:
        RecipeValidationError: if the recipe fails schema validation (raised
            from inside `RecipeLibrary.store`).
    """
    # Imported here so importing this module (e.g. for `sample_source` in a
    # hermetic unit test) does not pull in the Postgres-backed library.
    from app.recipes.library import RecipeLibrary

    library = RecipeLibrary(session)
    return await library.store(recipe, graph_id)


# ---------------------------------------------------------------------------
# CLI — the seam the `/ds` skill drives
# ---------------------------------------------------------------------------


def _resolve_postgres_url() -> str:
    """A Postgres URL reachable from the process running the CLI.

    `settings.POSTGRES_URL` uses the Docker-network host ``postgres``. When the
    CLI runs outside the compose network that host is unresolvable, so it is
    rewritten to ``localhost``. ``TEST_POSTGRES_URL`` takes precedence — the
    same override the integration tests honour.
    """
    import os

    from app.core.config import settings

    override = os.getenv("TEST_POSTGRES_URL")
    if override:
        return override
    return settings.POSTGRES_URL.replace("@postgres:", "@localhost:")


def _cmd_sample(args: argparse.Namespace) -> int:
    """`sample` — print a source's SAMPLE-mode `StructuralRepresentation`.

    This is the output the `data-specialist` agent reads to author a recipe.
    """
    representation = sample_source(args.source, args.kind)
    json.dump(representation.model_dump(mode="json"), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


async def _cmd_save_async(args: argparse.Namespace) -> int:
    """`save` — store an agent-authored recipe JSON file as a draft version."""
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )
    from sqlalchemy.pool import NullPool

    with open(args.recipe, encoding="utf-8") as fh:
        recipe = json.load(fh)

    engine = create_async_engine(
        _resolve_postgres_url(), poolclass=NullPool, future=True
    )
    try:
        session_maker = async_sessionmaker(
            bind=engine, class_=AsyncSession, expire_on_commit=False
        )
        async with session_maker() as session:
            stored = await save_recipe_draft(recipe, args.graph_id, session)
            await session.commit()
    finally:
        await engine.dispose()

    json.dump(stored, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _cmd_save(args: argparse.Namespace) -> int:
    """Sync entry point for the `save` command."""
    return asyncio.run(_cmd_save_async(args))


def _build_parser() -> argparse.ArgumentParser:
    """Build the `python -m app.recipes.authoring` argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m app.recipes.authoring",
        description=(
            "Design-time recipe-authoring loop (TASK-225, ADR-022). "
            "`sample` a source for the data-specialist agent to read, then "
            "`save` the recipe the agent authored as a draft."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_p = subparsers.add_parser(
        "sample",
        help="sample a data source and print its StructuralRepresentation as JSON",
    )
    sample_p.add_argument(
        "--kind",
        required=True,
        choices=sorted(PRIMITIVE_REGISTRY),
        help="the primitive kind to decompose the source with",
    )
    sample_p.add_argument(
        "source",
        help="the data source (a file path; markdown also accepts raw text)",
    )
    sample_p.set_defaults(func=_cmd_sample)

    save_p = subparsers.add_parser(
        "save",
        help="store an agent-authored recipe JSON file as a draft version",
    )
    save_p.add_argument(
        "--graph-id",
        required=True,
        help="the authoring tenant the draft is stored under",
    )
    save_p.add_argument(
        "recipe",
        help="path to the agent-authored recipe JSON file",
    )
    save_p.set_defaults(func=_cmd_save)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (UnknownPrimitiveKindError, FileNotFoundError) as exc:
        parser.exit(2, f"error: {exc}\n")
    except Exception as exc:  # noqa: BLE001 — surface any failure as a CLI error
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
