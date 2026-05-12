"""Opaque-cursor pagination codec for the assessment read endpoints.

Used by `GET /assessments/runs`, `GET /assessments/runs/{run_id}/findings`,
`GET /assessments/registry/{kind}`, and the other read endpoints landed in
TASK-079.

Design choices:

-   The cursor is **opaque to clients**: a base64url string. Clients echo it
    back verbatim; they do not parse it. This lets us change the internal
    scheme (offset → keyset → seek) without breaking compatibility.
-   The internal payload is `{"o": <int offset>, "id": <last-natural-id?>}`
    encoded as JSON, then base64url'd. Offset-based pagination is acceptable
    for SPRINT-002: the largest list (findings) is bounded at ~600 rows for
    the Eurail workload. SPRINT-003 may swap to keyset for scale; the cursor
    surface stays unchanged because callers never crack it open.
-   We **do not expose Neo4j internal pagination tokens** (`elementId`,
    skip-with-internal-cursor). The cursor must be reproducible across
    transactions and database restarts, so the natural id is the only valid
    keyset key when we upgrade.
-   Malformed cursors are rejected with `ValueError`; the endpoint layer
    translates this to HTTP 400.

The codec is intentionally tiny and dependency-free. No urlsafe-padding
shenanigans, no version bytes — when SPRINT-003 needs them, add a `v` key to
the payload and ignore unknown versions on decode.
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any

# A sane bound so a poisoned cursor cannot allocate gigabytes of memory.
_MAX_CURSOR_BYTES = 4 * 1024


def encode_cursor(offset: int, last_id: str | None = None) -> str:
    """Encode a pagination cursor to an opaque base64url string.

    Args:
        offset: Zero-based skip count for the *next* page.
        last_id: Optional natural-id keyset hint (forward-looking; SPRINT-001
            ignores it on decode but stores it so we can switch to keyset
            pagination without breaking issued cursors).

    Returns:
        A base64url-encoded string suitable for echoing in a `next_cursor`
        response field and accepting back as a `cursor=` query param.
    """
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset!r}")
    payload: dict[str, Any] = {"o": int(offset)}
    if last_id:
        payload["id"] = last_id
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def decode_cursor(cursor: str | None) -> tuple[int, str | None]:
    """Decode a cursor produced by :func:`encode_cursor`.

    Args:
        cursor: The opaque string returned by the previous page's
            `next_cursor`, or `None` / empty to request the first page.

    Returns:
        Tuple `(offset, last_id)`. When `cursor` is None/empty, returns
        `(0, None)`.

    Raises:
        ValueError: Cursor is malformed (bad base64, bad JSON, missing
            `o` key, negative offset, or oversize).
    """
    if not cursor:
        return 0, None
    if len(cursor) > _MAX_CURSOR_BYTES:
        raise ValueError(f"cursor exceeds {_MAX_CURSOR_BYTES} bytes")
    try:
        cursor_bytes = cursor.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"cursor must be ASCII: {exc}") from exc
    pad = (-len(cursor_bytes)) % 4
    try:
        raw = base64.urlsafe_b64decode(cursor_bytes + b"=" * pad)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"malformed cursor (base64 decode failed): {exc}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"malformed cursor (JSON decode failed): {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("malformed cursor (payload is not a JSON object)")
    if "o" not in payload:
        raise ValueError("malformed cursor (missing 'o' key)")
    offset_raw = payload["o"]
    if not isinstance(offset_raw, int) or isinstance(offset_raw, bool):
        raise ValueError(f"malformed cursor ('o' must be an int, got {offset_raw!r})")
    if offset_raw < 0:
        raise ValueError(f"malformed cursor ('o' must be >= 0, got {offset_raw!r})")
    last_id = payload.get("id")
    if last_id is not None and not isinstance(last_id, str):
        raise ValueError("malformed cursor ('id' must be a string when present)")
    return offset_raw, last_id


def clamp_limit(limit: int | None, default: int, maximum: int) -> int:
    """Coerce a caller-supplied `limit` into the allowed range.

    Returns `default` when `limit` is None, otherwise clamps to `[1, maximum]`.
    """
    if limit is None:
        return default
    if limit < 1:
        return 1
    if limit > maximum:
        return maximum
    return limit
