"""Code primitive — wraps the AST stage of `app.services.code_parser_service`.

TASK-222 / STORY-034 / ADR-022.

Deterministically decomposes a set of parsed source files into a
`StructuralRepresentation`:

  * one `FILE` unit per source file;
  * one `SYMBOL` unit per AST symbol (`Module` import marker, `Class`,
    `Function`, `Variable`), with `parent_id` containment — a method's parent
    is its class, a top-level symbol's parent is its file — and `metadata`
    carrying the symbol type plus, where available, the symbol's unresolved
    `calls` and `imports`.

The adapter does NOT modify `code_parser_service`. It calls `parse_file` /
`parse_files_parallel` (AST stage 2 only) and translates the resulting
`RawSymbol` records. Cross-file resolution (stage 3), embeddings (stage 4) and
Neo4j writes (stage 5) are out of scope — a primitive is concern-agnostic.

`decompose` accepts either a single `FileMetadata` or a list of them. Building
`FileMetadata` (the bootstrap stage) is the caller's job; the helper
`file_metadata_from_path` is provided for hermetic single-file use.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from app.recipes.primitives.interface import (
    ExtractionMode,
    StructuralRepresentation,
    StructuralUnit,
    UnitKind,
)
from app.services.code_parser_service import (
    LANGUAGE_EXTENSIONS,
    FileMetadata,
    RawSymbol,
    parse_file,
)

# Bounded number of calls / imports kept on a SYMBOL unit's metadata in
# SAMPLE mode. FULL mode keeps the complete lists.
_SAMPLE_LIMIT = 10


def file_metadata_from_path(
    abs_path: str, repo_root: str | None = None
) -> FileMetadata:
    """Build a `FileMetadata` for a single source file on disk.

    A thin convenience wrapper so callers (and hermetic tests) can decompose
    one file without running the whole repository bootstrap stage.
    """
    p = Path(abs_path)
    rel_path = str(p.relative_to(repo_root)) if repo_root else p.name
    ext = p.suffix.lower()
    language = LANGUAGE_EXTENSIONS.get(ext, "unknown")
    content = p.read_bytes()
    parts = Path(rel_path).parts
    is_test = any(
        part.startswith("test_") or part in ("tests", "test") for part in parts
    )
    return FileMetadata(
        path=rel_path,
        abs_path=str(p),
        language=language,
        size_bytes=len(content),
        content_hash=hashlib.sha256(content).hexdigest(),
        is_test=is_test,
    )


def _file_unit_id(file_path: str) -> str:
    return f"file:{file_path}"


def _symbol_unit_id(symbol: RawSymbol) -> str:
    """Stable, path-like id — file path plus the symbol's qualified name."""
    return f"symbol:{symbol.file_path}:{symbol.qualified_name}:{symbol.start_line}"


class CodePrimitive:
    """Adapter turning parsed source files into a `StructuralRepresentation`."""

    source_type: str = "code"

    def decompose(self, source: Any, mode: ExtractionMode) -> StructuralRepresentation:
        """Decompose one or more source files into structural units.

        Args:
            source: A single `FileMetadata` or a list of `FileMetadata`.
            mode: SAMPLE bounds the per-symbol calls/imports metadata; FULL
                keeps the complete lists.
        """
        file_metas: list[FileMetadata]
        if isinstance(source, FileMetadata):
            file_metas = [source]
        else:
            file_metas = list(source)

        units: list[StructuralUnit] = []

        for meta in file_metas:
            file_id = _file_unit_id(meta.path)
            units.append(
                StructuralUnit(
                    kind=UnitKind.FILE,
                    unit_id=file_id,
                    name=meta.path,
                    data_type=meta.language,
                    metadata={
                        "language": meta.language,
                        "size_bytes": meta.size_bytes,
                        "is_test": meta.is_test,
                    },
                )
            )

            symbols = parse_file(meta)
            # Map a qualified class name → its symbol unit_id, so a method /
            # class-scoped variable's parent_id is its class, not the file.
            class_unit_by_qname: dict[str, str] = {
                s.qualified_name: _symbol_unit_id(s)
                for s in symbols
                if s.symbol_type == "Class"
            }

            for symbol in symbols:
                units.append(
                    self._symbol_unit(symbol, file_id, class_unit_by_qname, mode)
                )

        return StructuralRepresentation(
            source_type=self.source_type,
            shape_signature=self._shape_signature(file_metas),
            mode=mode,
            units=units,
        )

    @staticmethod
    def _symbol_unit(
        symbol: RawSymbol,
        file_id: str,
        class_unit_by_qname: dict[str, str],
        mode: ExtractionMode,
    ) -> StructuralUnit:
        """Translate one `RawSymbol` into a `SYMBOL` structural unit."""
        # Containment: a symbol nested in a class is parented to that class;
        # otherwise it is parented to its file.
        parent_id = file_id
        if symbol.parent_class and symbol.parent_class in class_unit_by_qname:
            parent_id = class_unit_by_qname[symbol.parent_class]

        calls = list(symbol.raw_calls)
        imports = [imp.get("target", "") for imp in symbol.raw_imports]
        if mode == ExtractionMode.SAMPLE:
            calls = calls[:_SAMPLE_LIMIT]
            imports = imports[:_SAMPLE_LIMIT]

        metadata: dict[str, Any] = {
            "symbol_type": symbol.symbol_type,
            "qualified_name": symbol.qualified_name,
            "start_line": symbol.start_line,
            "end_line": symbol.end_line,
        }
        if symbol.is_method:
            metadata["is_method"] = True
        if symbol.is_async:
            metadata["is_async"] = True
        if symbol.is_test:
            metadata["is_test"] = True
        if calls:
            metadata["calls"] = calls
        if imports:
            metadata["imports"] = imports
        if symbol.raw_bases:
            metadata["bases"] = list(symbol.raw_bases)

        # role mirrors the symbol kind so a recipe can match on it.
        role = symbol.symbol_type.lower()

        return StructuralUnit(
            kind=UnitKind.SYMBOL,
            unit_id=_symbol_unit_id(symbol),
            name=symbol.name,
            data_type=symbol.language,
            role=role,
            parent_id=parent_id,
            metadata=metadata,
        )

    @staticmethod
    def _shape_signature(file_metas: list[FileMetadata]) -> str:
        """Deterministic descriptor — sorted file paths + languages.

        Independent of file ordering so the same file set yields the same
        signature (recipe-spec §4 lookup key).
        """
        parts = sorted(f"{m.path}:{m.language}" for m in file_metas)
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
        return f"code({len(file_metas)}files:{digest})"
