"""
Markdown structured extraction service.

Parses Markdown heading structure using Python stdlib (re, pathlib).
No external markdown library required.

Produces a flat sections list and a parent-child hierarchy tree.
"""

import re
from pathlib import Path
from typing import Any

# Matches ATX-style headings: # Heading, ## Heading, …, ###### Heading
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _build_hierarchy(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build a parent-child hierarchy from a flat list of sections.

    Each entry in the returned list has:
        {"heading": str, "level": int, "parent": str | None, "children": [str, ...]}

    Children is a list of heading strings (not full nodes) to keep it simple
    and serialisable.
    """
    if not sections:
        return []

    # Stack holds (level, heading_str) for ancestors
    stack: list[tuple[int, str]] = []
    hierarchy: list[dict[str, Any]] = []

    for section in sections:
        level = section["level"]
        heading = section["heading"]

        # Pop stack entries that are at the same level or deeper
        while stack and stack[-1][0] >= level:
            stack.pop()

        parent = stack[-1][1] if stack else None
        node: dict[str, Any] = {
            "heading": heading,
            "level": level,
            "parent": parent,
            "children": [],
        }
        hierarchy.append(node)
        stack.append((level, heading))

    # Fill in children references
    heading_to_node: dict[str, dict[str, Any]] = {}
    for node in hierarchy:
        # Use heading as key; duplicates get overwritten (acceptable heuristic)
        heading_to_node[node["heading"]] = node

    for node in hierarchy:
        if node["parent"] is not None:
            parent_node = heading_to_node.get(node["parent"])
            if parent_node is not None:
                parent_node["children"].append(node["heading"])

    return hierarchy


def extract_markdown_from_text(text: str, fallback_title: str = "") -> dict[str, Any]:
    """
    Extract structure from a Markdown string.

    Args:
        text: The raw markdown content.
        fallback_title: Used when the document has no H1 heading. Pass the
            filename stem from the caller, or leave empty.

    Returns: see ``extract_markdown``.
    """
    matches = list(_HEADING_RE.finditer(text))

    sections: list[dict[str, Any]] = []
    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        sections.append({"level": level, "heading": heading, "content": content})

    title = next(
        (s["heading"] for s in sections if s["level"] == 1),
        fallback_title,
    )

    hierarchy = _build_hierarchy(sections)

    return {"title": title, "sections": sections, "hierarchy": hierarchy}


def extract_markdown(file_path: str) -> dict[str, Any]:
    """
    Extract structure from a Markdown file on disk.

    Thin wrapper around ``extract_markdown_from_text`` for callers that have a
    path. Callers that already have the markdown content as a string should use
    ``extract_markdown_from_text`` directly to avoid the I/O round-trip.

    Args:
        file_path: Absolute path to the Markdown file.

    Returns:
        {
            "title": str,    # first H1 heading, or filename without extension
            "sections": [
                {"level": 1-6, "heading": "...", "content": "..."},
                ...
            ],
            "hierarchy": [
                {"heading": "...", "level": int, "parent": str|None, "children": [str, ...]},
                ...
            ]
        }
    """
    with open(file_path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    return extract_markdown_from_text(text, fallback_title=Path(file_path).stem)
