"""Grep-based isolation test for the :__Chat__ namespace (STORY-031 / TASK-106).

ADR-020 mandates that every Cypher statement touching a chat-namespaced
node lives in ``app/cypher/chat_queries.py``. Any other module that
references ``:Conversation`` or ``:ChatTurn`` in a Cypher string fails
this test — chat content must never accidentally appear in community
detection, retrieval, traversal, or any other analytics query.

The test scans every ``.py`` file under ``app/`` for Cypher patterns
matching chat-namespaced labels. If a hit lands outside the allowlist,
the test fails with the offending file:line.

Rationale: the isolation requirement is structural, not procedural —
a single forgotten ``WHERE NOT n:__Chat__`` somewhere in retrieval
could silently leak chat data into agent answers. A grep-based CI gate
makes the rule mechanical to enforce.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Files that ARE allowed to reference :Conversation / :ChatTurn.
# Everything else in app/ must avoid those labels.
_ALLOWED_FILES = {
    "app/cypher/chat_queries.py",
}

# Patterns to flag — chat-namespaced Cypher labels.
# Match only when the label appears inside a Cypher node pattern:
# ``(c:Conversation)``, ``(:ChatTurn)``, ``(node_var:Conversation:Other)``,
# etc. The leading ``\(`` plus optional identifier prevents false
# positives in prose (e.g. "The :Conversation node lives in chat...").
_FORBIDDEN_PATTERNS = [
    re.compile(r"\(\s*[A-Za-z_]\w*\s*(?::[A-Za-z_]\w*)*\s*:\s*Conversation\b"),
    re.compile(r"\(\s*[A-Za-z_]\w*\s*(?::[A-Za-z_]\w*)*\s*:\s*ChatTurn\b"),
    re.compile(r"\(\s*:\s*Conversation\b"),
    re.compile(r"\(\s*:\s*ChatTurn\b"),
]


def _app_root() -> Path:
    """Locate the ``app/`` directory at the repo root."""
    here = Path(__file__).resolve()
    # tests/unit/test_chat_namespace_isolation.py -> knowledge-graph-builder/
    return here.parents[2] / "app"


def _is_allowed(path: Path, repo_root: Path) -> bool:
    rel = path.relative_to(repo_root).as_posix()
    return rel in _ALLOWED_FILES


def _strip_comments(line: str) -> str:
    """Return the line with the # comment portion removed.

    A simple split — fine for Python source. Catches the common case
    where someone writes ``# We DON'T use :Conversation`` in a comment.
    """
    if "#" in line:
        return line.split("#", 1)[0]
    return line


@pytest.mark.unit
def test_no_chat_namespace_cypher_outside_allowlist():
    repo_root = _app_root().parent
    app_dir = _app_root()
    offenders: list[tuple[Path, int, str]] = []

    for py_file in app_dir.rglob("*.py"):
        if _is_allowed(py_file, repo_root):
            continue
        try:
            text = py_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_no, raw_line in enumerate(text.splitlines(), start=1):
            line = _strip_comments(raw_line)
            for pattern in _FORBIDDEN_PATTERNS:
                if pattern.search(line):
                    offenders.append(
                        (py_file.relative_to(repo_root), line_no, raw_line.strip())
                    )
                    break

    if offenders:
        msg_lines = [
            "Chat-namespaced Cypher found outside app/cypher/chat_queries.py:",
            "",
        ]
        for path, line_no, src in offenders:
            msg_lines.append(f"  {path}:{line_no}: {src}")
        msg_lines.append("")
        msg_lines.append(
            "Move the offending Cypher into app/cypher/chat_queries.py "
            "or remove the chat-namespaced label."
        )
        pytest.fail("\n".join(msg_lines))


@pytest.mark.unit
def test_chat_queries_module_carries_namespace_on_chat_nodes():
    """Every Cypher node pattern in chat_queries.py that names :Conversation
    or :ChatTurn MUST also carry :__Chat__ — that's the namespace contract.

    Uses the same node-pattern regex as the outer-allowlist test so prose
    references in the module docstring don't trigger false positives.
    """
    chat_q = _app_root() / "cypher" / "chat_queries.py"
    text = chat_q.read_text(encoding="utf-8")

    offending_lines = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = _strip_comments(raw_line)
        if not any(p.search(line) for p in _FORBIDDEN_PATTERNS):
            continue
        if "__Chat__" not in line:
            offending_lines.append((line_no, raw_line.strip()))

    if offending_lines:
        msg = "\n".join(
            [
                "Chat node pattern in chat_queries.py is missing :__Chat__:",
                *(f"  line {n}: {s}" for n, s in offending_lines),
                "",
                "Every :Conversation / :ChatTurn node pattern MUST also "
                "carry :__Chat__ (ADR-020 + ADR-015 reserved namespace).",
            ]
        )
        pytest.fail(msg)


@pytest.mark.unit
@pytest.mark.parametrize(
    "violation",
    [
        '    cypher = "MATCH (c:Conversation) RETURN c"',
        '"""MATCH (:ChatTurn)-[:IN_CONVERSATION]->() RETURN count(*)"""',
        '    q = "MATCH (t:ChatTurn:OtherLabel) RETURN t"',
    ],
)
def test_isolation_test_catches_planted_violations(violation):
    """Synthetic offenders the regex must catch."""
    stripped = _strip_comments(violation)
    assert any(p.search(stripped) for p in _FORBIDDEN_PATTERNS), (
        f"FORBIDDEN_PATTERNS missed a planted violation: {violation!r}"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "benign",
    [
        "    # The :Conversation node is namespaced under :__Chat__",
        '"""The :Conversation node is the user-facing thread."""',
        "    1. Any Cypher pattern matching :Conversation or :ChatTurn",
        '    """A conversation has many ChatTurns."""',
    ],
)
def test_isolation_test_ignores_prose(benign):
    """Comments and docstrings that mention :Conversation in prose must
    NOT trigger the test."""
    stripped = _strip_comments(benign)
    assert all(not p.search(stripped) for p in _FORBIDDEN_PATTERNS), (
        f"FORBIDDEN_PATTERNS incorrectly matched prose: {benign!r}"
    )
