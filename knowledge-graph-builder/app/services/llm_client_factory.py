"""Constructs async LLM clients from a resolved LLMConfig dict (STORY-021).

Three providers are supported:
  anthropic    — AsyncAnthropic (native SDK)
  openrouter   — AsyncOpenAI with custom base_url
  azure-openai — AsyncAzureOpenAI with azure_endpoint + api_version
"""

from typing import Any, Union

try:
    from anthropic import AsyncAnthropic

    _ANTHROPIC_AVAILABLE = True
except ImportError:
    AsyncAnthropic = None  # type: ignore[misc,assignment]
    _ANTHROPIC_AVAILABLE = False

from openai import AsyncAzureOpenAI, AsyncOpenAI

LLMClient = Union["AsyncAnthropic", AsyncOpenAI, AsyncAzureOpenAI]


class LLMClientFactory:
    @staticmethod
    def build(config: dict[str, Any]) -> LLMClient:
        """Return the appropriate async LLM client for *config*.

        config keys used: provider, api_key, base_url, api_version
        api_key must be the plaintext key retrieved from credential-broker.
        """
        provider = config["provider"]
        api_key = config["api_key"]

        if provider == "anthropic":
            if not _ANTHROPIC_AVAILABLE:
                raise RuntimeError(
                    "anthropic SDK not installed; add 'anthropic' to requirements.txt"
                )
            return AsyncAnthropic(api_key=api_key)

        if provider == "openrouter":
            base_url = config.get("base_url") or "https://openrouter.ai/api/v1"
            return AsyncOpenAI(api_key=api_key, base_url=base_url)

        if provider == "azure-openai":
            return AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=config["base_url"],
                api_version=config.get("api_version") or "2024-02-01",
            )

        raise ValueError(f"Unknown LLM provider: {provider!r}")
