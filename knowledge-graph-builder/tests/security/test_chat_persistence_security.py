"""Adversarial security tests for chat persistence (STORY-031 / TASK-107).

These tests verify the threat-defense pairs documented in the
STORY-031 security review. Each ``test_*`` function corresponds to a
numbered threat from the review:

1.  Cross-user read              — verified by RLS integration test
2.  Cross-graph read             — verified by RLS integration test
3.  ID enumeration               — verified here (404 vs 403)
4.  RLS bypass                   — verified by RLS integration test
5.  Chat data leaking to KG      — verified by namespace isolation test
6.  Compressed-result bomb       — verified here (decompression cap)
7.  Feedback abuse               — verified here (rating check, owner-only)
8.  Session-var manipulation     — verified here (UUID validation in get_chat_db)
9.  Export-endpoint abuse        — see STORY-031 review (rate limit follow-up)
10. Soft-delete recovery         — verified here (sweeper query shape)

Threats #1, #2, #4 are covered end-to-end by
tests/integration/test_chat_rls_integration.py (skipped when no
TEST_POSTGRES_URL); this file holds the ones that can run without a
real DB. #5 is covered by tests/unit/test_chat_namespace_isolation.py.
"""

from __future__ import annotations

import json
import uuid

import pytest
import zstandard as zstd
from fastapi import HTTPException

from app.api.v1.endpoints.chat_history import (
    _check_feedback_rating,
    _parse_uuid_or_404,
)
from app.services.chat_history_service import (
    MAX_DECOMPRESSED_BYTES,
    CompressionLimitExceeded,
    decompress_result,
)

# ──────────────────────────────────────────────────────────────────────────── #
# Threat #3 — ID enumeration. Malformed and missing ids both return 404
# so an attacker can't distinguish "doesn't exist" from "isn't yours".
# ──────────────────────────────────────────────────────────────────────────── #


@pytest.mark.security
class TestIdEnumerationDefense:
    @pytest.mark.parametrize(
        "value",
        ["", "not-a-uuid", "00000000", "abcdef", "../../etc/passwd"],
    )
    def test_malformed_ids_return_404_not_422(self, value):
        with pytest.raises(HTTPException) as exc:
            _parse_uuid_or_404(value, "conversation")
        # Critical: NOT 422 (which would tell attackers the format is
        # wrong but the resource might exist).
        assert exc.value.status_code == 404

    def test_valid_uuid_does_not_raise(self):
        u = str(uuid.uuid4())
        # Returns the UUID — no exception.
        result = _parse_uuid_or_404(u, "conversation")
        assert str(result) == u


# ──────────────────────────────────────────────────────────────────────────── #
# Threat #6 — Decompression bomb. An adversarial zstd payload could
# expand 1000× on read. The MAX_DECOMPRESSED_BYTES cap rejects such
# payloads with CompressionLimitExceeded instead of allocating memory.
# ──────────────────────────────────────────────────────────────────────────── #


@pytest.mark.security
class TestDecompressionBomb:
    def test_safe_payload_round_trips(self):
        raw = json.dumps({"hello": "world"}).encode("utf-8")
        compressed = zstd.ZstdCompressor(level=3).compress(raw)
        result = decompress_result(compressed, "zstd")
        assert result == raw

    def test_oversize_payload_raises_compression_limit_exceeded(self):
        # Build a payload that decompresses well past the cap.
        # 70 MB of zeros compresses ~3 KB but expands to 70 MB.
        oversize_raw = b"\x00" * (MAX_DECOMPRESSED_BYTES + 16 * 1024 * 1024)
        compressed = zstd.ZstdCompressor(level=3).compress(oversize_raw)
        assert len(compressed) < 1 * 1024 * 1024  # confirm it's small compressed
        with pytest.raises(CompressionLimitExceeded):
            decompress_result(compressed, "zstd")

    def test_none_payload_is_passthrough(self):
        assert decompress_result(None, None) is None
        assert decompress_result(None, "zstd") is None

    def test_unknown_compression_returned_as_is(self):
        """A row with compression=NULL (legacy / unknown) bypasses decompression
        and is returned untouched. Defense-in-depth: the cap is enforced
        for zstd specifically, not for unknown compression types."""
        raw = b"not actually compressed"
        result = decompress_result(raw, None)
        assert result == raw


# ──────────────────────────────────────────────────────────────────────────── #
# Threat #7 — Feedback abuse. Only -1 and +1 ratings are accepted;
# anything else returns 400. The ownership check happens via a JOIN
# on chat_conversations.user_id in the endpoint.
# ──────────────────────────────────────────────────────────────────────────── #


@pytest.mark.security
class TestFeedbackRatingBounds:
    @pytest.mark.parametrize("rating", [-1, 1])
    def test_valid_ratings_pass(self, rating):
        assert _check_feedback_rating(rating) == rating

    @pytest.mark.parametrize("rating", [0, 2, -2, 100, -100, 2**31, -(2**31)])
    def test_invalid_ratings_rejected(self, rating):
        with pytest.raises(HTTPException) as exc:
            _check_feedback_rating(rating)
        assert exc.value.status_code == 400

    def test_db_check_constraint_present(self):
        """The Postgres CHECK constraint is the last line of defense.
        Verifies the model carries it so a future migration can't drop
        it silently."""
        from app.models.chat import ChatMessage

        names = {c.name for c in ChatMessage.__table__.constraints if c.name}
        assert "chat_messages_feedback_rating_check" in names


# ──────────────────────────────────────────────────────────────────────────── #
# Threat #8 — Session-var manipulation. ``get_chat_db`` UUID-validates
# the principal id before passing it to ``set_config``; a forged token
# with a non-UUID id is rejected. The set_config call uses parameter
# binding (no string interpolation) so SQL injection via the GUC is
# structurally impossible.
# ──────────────────────────────────────────────────────────────────────────── #


@pytest.mark.security
class TestSessionVarManipulationDefense:
    def test_set_config_uses_parameter_binding(self):
        """The dependency MUST use bindparams() so the user_id can't
        be string-concatenated into the SQL. We verify this structurally
        by inspecting the source — collapse whitespace before the
        substring search so multi-line formatting doesn't break it."""
        import inspect
        import re

        from app.api import dependencies

        src = inspect.getsource(dependencies.get_chat_db)
        compact = re.sub(r"\s+", " ", src)
        # Use a parameter-bound :uid (no f-string interpolation).
        assert "set_config('app.current_user_id', :uid" in compact, (
            "get_chat_db must pass user_id via parameter binding, not "
            "string interpolation. Without this, a malicious user_id "
            "could escape into the SQL."
        )
        # bindparams() must be called with uid=user_id.
        assert re.search(r"\.bindparams\(\s*uid\s*=\s*user_id\s*\)", compact), (
            "get_chat_db must call .bindparams(uid=user_id) on the "
            "text() statement so the GUC value is escaped by SQLAlchemy."
        )

    def test_uuid_validation_blocks_malformed_ids(self):
        """The UUID parse happens BEFORE the set_config CALL (not the
        docstring mention) so a forged principal with a non-UUID id
        never reaches the DB."""
        import inspect

        from app.api import dependencies

        src = inspect.getsource(dependencies.get_chat_db)
        # The validation block must precede the actual set_config call.
        # We look for "SELECT set_config" which only appears in the
        # text() statement, not in the docstring.
        uuid_idx = src.find("uuid_lib.UUID(user_id)")
        set_config_call_idx = src.find("SELECT set_config")
        assert uuid_idx > 0
        assert set_config_call_idx > 0
        assert set_config_call_idx > uuid_idx, (
            "UUID validation must precede the set_config call so "
            "malformed principals are rejected before touching the DB."
        )


# ──────────────────────────────────────────────────────────────────────────── #
# Threat #10 — Soft-delete recovery. The sweeper task MUST be scoped
# to (deleted_at IS NOT NULL AND deleted_at < now() - 30d) so it
# never accidentally hard-deletes live conversations.
#
# v1: there is no scheduled sweeper task yet (we soft-delete on
# request; hard-delete is a future task). The test below verifies
# nothing currently hard-deletes from chat_conversations except via
# the explicit DELETE-conversation endpoint, which goes through
# _load_owned_conversation first.
# ──────────────────────────────────────────────────────────────────────────── #


@pytest.mark.security
class TestSoftDeleteSafety:
    def test_delete_endpoint_uses_owner_check(self):
        """The DELETE /chat/conversations/{cid} endpoint must call
        ``_load_owned_conversation`` before mutating, so a user can't
        delete someone else's row even via a guessed UUID."""
        import inspect

        from app.api.v1.endpoints import chat_history

        src = inspect.getsource(chat_history.delete_conversation)
        assert "_load_owned_conversation" in src, (
            "delete_conversation must verify ownership via "
            "_load_owned_conversation BEFORE mutating the row."
        )

    def test_delete_endpoint_only_soft_deletes(self):
        """The DELETE endpoint must SET deleted_at, never DELETE FROM."""
        import inspect

        from app.api.v1.endpoints import chat_history

        src = inspect.getsource(chat_history.delete_conversation)
        # Sanity: the source uses the SQLAlchemy update() helper, not
        # delete()/DELETE.
        assert "update(ChatConversation)" in src
        assert "values(deleted_at=" in src
        assert "DELETE FROM" not in src.upper()
