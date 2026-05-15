"""Unit tests for ChatHistoryService (STORY-031 / TASK-103).

These cover the side-effect-free logic — primarily the tool-result
compression path and the LLM pricing module. Persistence interactions
(get_or_create_conversation, write_user_message, etc.) are exercised
end-to-end in ``tests/integration/test_chat_history_writes.py``.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest
import zstandard as zstd

from app.services.chat_history_service import (
    MAX_COMPRESSED_BYTES,
    MAX_RAW_BYTES,
    _compress_result,
)
from app.services.llm_pricing import PRICES, estimate_cost_usd


@pytest.mark.unit
class TestCompressResult:
    def test_none_round_trip(self):
        compressed, compression, size, truncated, summary = _compress_result(None)
        assert compressed is None
        assert compression is None
        assert size is None
        assert truncated is False
        assert summary is None

    def test_small_dict_roundtrips(self):
        result = {"answer": "yes", "count": 3}
        compressed, compression, size, truncated, summary = _compress_result(result)
        assert compression == "zstd"
        assert truncated is False
        assert size == len(json.dumps(result).encode("utf-8"))
        assert summary == "2 keys"
        decompressed = zstd.ZstdDecompressor().decompress(compressed)
        assert json.loads(decompressed) == result

    def test_list_summary(self):
        result = [{"id": i} for i in range(7)]
        _, _, _, truncated, summary = _compress_result(result)
        assert truncated is False
        assert summary == "7 items"

    def test_oversized_raw_is_truncated(self):
        """A raw payload over 50 MB is replaced with a placeholder."""
        # Build something just above MAX_RAW_BYTES of JSON.
        big_string = "x" * (MAX_RAW_BYTES + 100)
        result = {"data": big_string}
        compressed, compression, size, truncated, summary = _compress_result(result)
        assert truncated is True
        # The compressed bytes are the placeholder, not the giant string.
        assert compression == "zstd"
        assert size > MAX_RAW_BYTES
        decompressed = zstd.ZstdDecompressor().decompress(compressed)
        payload = json.loads(decompressed)
        assert payload["_truncated"] is True

    def test_oversized_compressed_drops_blob(self):
        """If even the compressed blob exceeds 5 MB, the row stores no blob."""
        # Random-like bytes resist compression. Build a payload whose
        # compressed form is > MAX_COMPRESSED_BYTES but raw is < MAX_RAW_BYTES.
        # 8 MB of random hex string compresses badly but stays inside raw cap.
        import os

        random_bytes = os.urandom(8 * 1024 * 1024)
        result = {"blob": random_bytes.hex()}  # 16 MB string, but raw < 50 MB
        compressed, compression, size, truncated, summary = _compress_result(result)
        # We don't strictly know whether 16 MB of random hex will go above
        # 5 MB compressed; if it does, the result must reflect the drop.
        if compressed is None:
            assert compression is None
            assert truncated is True
            assert size is not None
            assert summary and "truncated" in summary
        else:
            # If by luck compression got us under the cap, the test still
            # holds — but in practice random-hex doesn't compress.
            assert len(compressed) <= MAX_COMPRESSED_BYTES

    def test_default_str_serialization_covers_exotic_types(self):
        """json.dumps(..., default=str) round-trips datetimes, UUIDs, etc.,
        without raising — so the compressor never refuses input."""
        from datetime import datetime
        from uuid import uuid4

        result = {"ts": datetime(2026, 5, 15), "id": uuid4()}
        compressed, compression, _, truncated, _ = _compress_result(result)
        assert compressed is not None
        assert compression == "zstd"
        assert truncated is False
        decompressed = zstd.ZstdDecompressor().decompress(compressed)
        # Round-trip via str — values are strings now, but the
        # JSON structure is preserved.
        payload = json.loads(decompressed)
        assert isinstance(payload["ts"], str)
        assert isinstance(payload["id"], str)


@pytest.mark.unit
class TestLLMPricing:
    def test_known_model_returns_decimal(self):
        cost = estimate_cost_usd(
            "claude-opus-4-7", prompt_tokens=1000, completion_tokens=500
        )
        # 1000 prompt tokens * $15/1K + 500 completion * $75/1K
        # = $15 + $37.50 = $52.50
        assert cost == Decimal("52.500000")

    def test_unknown_model_returns_none(self):
        assert (
            estimate_cost_usd(
                "made-up-model-9000", prompt_tokens=1, completion_tokens=1
            )
            is None
        )

    def test_missing_tokens_returns_none(self):
        assert estimate_cost_usd("claude-opus-4-7", None, 100) is None
        assert estimate_cost_usd("claude-opus-4-7", 100, None) is None

    def test_zero_tokens_returns_zero(self):
        cost = estimate_cost_usd(
            "claude-opus-4-7", prompt_tokens=0, completion_tokens=0
        )
        assert cost == Decimal("0.000000")

    def test_all_pricing_table_models_round_trip(self):
        """Every model in the price table produces a valid cost for sane inputs."""
        for model in PRICES.keys():
            cost = estimate_cost_usd(model, prompt_tokens=100, completion_tokens=50)
            assert isinstance(cost, Decimal), f"{model} returned non-Decimal {cost!r}"
            assert cost >= Decimal("0"), f"{model} produced negative cost {cost}"
