"""Per-model price lookups for chat-history cost accounting (STORY-031).

Prices are in **USD per 1K tokens**. They are intentionally a small
hard-coded table here rather than a config file — the rates change
slowly, and a Python literal keeps cost rollout local to the persistence
write hooks. Unknown models return ``None`` so the chat-history row
stores ``cost_usd = NULL`` rather than a wrong value.

Adding a new model: append a row to ``PRICES`` keyed by the exact
model id used by the executor (e.g. ``"claude-opus-4-7"``).
Update the test in ``tests/unit/test_llm_pricing.py`` if/when one is
added.

Re-rate events: rates here are not historical. If a customer requires
historical accuracy (e.g. for invoicing), the row in chat_messages
captures the prompt/completion token counts so a downstream report can
re-cost retroactively.
"""

from __future__ import annotations

from decimal import Decimal


class _ModelPrice:
    __slots__ = ("prompt_per_1k", "completion_per_1k")

    def __init__(self, prompt_per_1k: str, completion_per_1k: str) -> None:
        self.prompt_per_1k = Decimal(prompt_per_1k)
        self.completion_per_1k = Decimal(completion_per_1k)


# USD per 1K tokens. Update as providers publish new rates.
PRICES: dict[str, _ModelPrice] = {
    # Anthropic Claude — May 2026 published rates.
    "claude-opus-4-7": _ModelPrice("15.00", "75.00"),
    "claude-opus-4-6": _ModelPrice("15.00", "75.00"),
    "claude-sonnet-4-6": _ModelPrice("3.00", "15.00"),
    "claude-haiku-4-5": _ModelPrice("1.00", "5.00"),
    # OpenAI — May 2026 published rates.
    "gpt-4o": _ModelPrice("2.50", "10.00"),
    "gpt-4o-mini": _ModelPrice("0.15", "0.60"),
    "gpt-4-turbo": _ModelPrice("10.00", "30.00"),
}


def estimate_cost_usd(
    model: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> Decimal | None:
    """Return estimated USD cost for a turn, or ``None`` if pricing is unknown.

    ``None`` is returned when:
      * ``model`` is unknown (not in PRICES)
      * Either token count is None (caller didn't supply them)

    The result is a ``Decimal`` with 6 decimal places — matches the
    ``NUMERIC(10,6)`` shape of ``chat_messages.cost_usd``.
    """
    if model is None or prompt_tokens is None or completion_tokens is None:
        return None
    rate = PRICES.get(model)
    if rate is None:
        return None
    cost = (
        rate.prompt_per_1k * Decimal(prompt_tokens)
        + rate.completion_per_1k * Decimal(completion_tokens)
    ) / Decimal(1000)
    # Quantize to 6 decimal places to match the DB column scale.
    return cost.quantize(Decimal("0.000001"))
