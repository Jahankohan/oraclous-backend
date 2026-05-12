"""Unit tests for the opaque pagination codec (TASK-079).

Covers `app.api.v1.endpoints._pagination`:
- Round-trip: `decode(encode(offset, last_id))` returns the inputs
- Empty/None cursor → `(0, None)`
- Malformed cursors raise `ValueError` (the endpoint translates to HTTP 400):
  * Non-ASCII
  * Bad base64
  * Bad JSON
  * Missing `o` key
  * Non-int `o`
  * Negative `o`
  * Non-string `id`
  * Oversize
- `clamp_limit` clamps to the requested range
"""

from __future__ import annotations

import pytest

from app.api.v1.endpoints._pagination import (
    clamp_limit,
    decode_cursor,
    encode_cursor,
)


class TestEncodeDecodeRoundTrip:
    def test_offset_only_round_trip(self):
        cursor = encode_cursor(0)
        assert decode_cursor(cursor) == (0, None)

        cursor = encode_cursor(100)
        assert decode_cursor(cursor) == (100, None)

    def test_offset_with_last_id_round_trip(self):
        cursor = encode_cursor(50, "run-abc")
        offset, last_id = decode_cursor(cursor)
        assert offset == 50
        assert last_id == "run-abc"

    def test_large_offset_round_trip(self):
        cursor = encode_cursor(1_000_000, "f-xyz-1234567890")
        assert decode_cursor(cursor) == (1_000_000, "f-xyz-1234567890")

    def test_unicode_last_id_round_trip(self):
        # Unicode chars in last_id should survive base64url
        cursor = encode_cursor(7, "subj-éurail")
        assert decode_cursor(cursor) == (7, "subj-éurail")


class TestEmptyCursor:
    def test_none_returns_zero_offset(self):
        assert decode_cursor(None) == (0, None)

    def test_empty_string_returns_zero_offset(self):
        assert decode_cursor("") == (0, None)


class TestNegativeOffset:
    def test_encode_rejects_negative(self):
        with pytest.raises(ValueError, match="offset must be >= 0"):
            encode_cursor(-1)


class TestMalformedCursors:
    def test_non_ascii_rejected(self):
        with pytest.raises(ValueError, match="ASCII"):
            decode_cursor("café")

    def test_bad_base64_rejected(self):
        # `!!!` is not valid base64url
        with pytest.raises(ValueError, match="malformed cursor"):
            decode_cursor("!!!@@@###")

    def test_bad_json_rejected(self):
        # base64url-encoded plain text "not json"
        import base64

        bad = base64.urlsafe_b64encode(b"not json at all").rstrip(b"=").decode()
        with pytest.raises(ValueError, match="malformed cursor"):
            decode_cursor(bad)

    def test_missing_o_key_rejected(self):
        import base64
        import json

        payload = json.dumps({"id": "foo"}).encode()
        bad = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
        with pytest.raises(ValueError, match="missing 'o' key"):
            decode_cursor(bad)

    def test_non_int_o_rejected(self):
        import base64
        import json

        payload = json.dumps({"o": "not-int"}).encode()
        bad = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
        with pytest.raises(ValueError, match="must be an int"):
            decode_cursor(bad)

    def test_bool_not_accepted_as_int(self):
        import base64
        import json

        payload = json.dumps({"o": True}).encode()
        bad = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
        with pytest.raises(ValueError, match="must be an int"):
            decode_cursor(bad)

    def test_negative_o_rejected(self):
        import base64
        import json

        payload = json.dumps({"o": -5}).encode()
        bad = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
        with pytest.raises(ValueError, match=">= 0"):
            decode_cursor(bad)

    def test_non_string_id_rejected(self):
        import base64
        import json

        payload = json.dumps({"o": 1, "id": 42}).encode()
        bad = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
        with pytest.raises(ValueError, match="'id' must be a string"):
            decode_cursor(bad)

    def test_array_payload_rejected(self):
        import base64
        import json

        payload = json.dumps([1, 2, 3]).encode()
        bad = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
        with pytest.raises(ValueError, match="not a JSON object"):
            decode_cursor(bad)

    def test_oversize_cursor_rejected(self):
        # 5KB cursor — over the 4KB limit
        with pytest.raises(ValueError, match="exceeds"):
            decode_cursor("A" * 5000)


class TestClampLimit:
    def test_none_returns_default(self):
        assert clamp_limit(None, default=50, maximum=200) == 50

    def test_zero_clamps_up_to_one(self):
        assert clamp_limit(0, default=50, maximum=200) == 1

    def test_negative_clamps_up_to_one(self):
        assert clamp_limit(-10, default=50, maximum=200) == 1

    def test_over_maximum_clamps_down(self):
        assert clamp_limit(500, default=50, maximum=200) == 200

    def test_in_range_passes_through(self):
        assert clamp_limit(75, default=50, maximum=200) == 75

    def test_equal_to_maximum_passes_through(self):
        assert clamp_limit(200, default=50, maximum=200) == 200


class TestForwardCompatibility:
    """The cursor format is intentionally opaque; future versions may add keys.

    These tests pin the current encoding so a future change either keeps the
    contract (decode old cursors) or explicitly breaks it (then update the
    test + cut a new client).
    """

    def test_payload_uses_short_keys(self):
        # `o` (offset) and optional `id` are the only keys in v1.
        import base64
        import json

        cursor = encode_cursor(42, "last-id")
        raw = base64.urlsafe_b64decode(cursor + "=" * ((-len(cursor)) % 4))
        payload = json.loads(raw)
        assert set(payload.keys()) <= {"o", "id"}

    def test_id_omitted_when_none(self):
        import base64
        import json

        cursor = encode_cursor(42)
        raw = base64.urlsafe_b64decode(cursor + "=" * ((-len(cursor)) % 4))
        payload = json.loads(raw)
        assert "id" not in payload
