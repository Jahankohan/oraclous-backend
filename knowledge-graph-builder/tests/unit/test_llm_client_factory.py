"""Unit tests for LLMClientFactory (STORY-021 / TASK-040)."""

import pytest

from app.services.llm_client_factory import LLMClientFactory


def _cfg(provider, base_url=None, api_version=None):
    return {
        "provider": provider,
        "model": "test-model",
        "api_key": "sk-test",
        "base_url": base_url,
        "api_version": api_version,
    }


class TestLLMClientFactory:
    def test_anthropic_returns_async_anthropic(self):
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            pytest.skip("anthropic SDK not installed")
        client = LLMClientFactory.build(_cfg("anthropic"))
        assert isinstance(client, AsyncAnthropic)

    def test_openrouter_returns_async_openai_with_base_url(self):
        from openai import AsyncOpenAI
        client = LLMClientFactory.build(
            _cfg("openrouter", base_url="https://openrouter.ai/api/v1")
        )
        assert isinstance(client, AsyncOpenAI)
        assert "openrouter" in str(client.base_url)

    def test_openrouter_uses_default_base_url_when_none(self):
        from openai import AsyncOpenAI
        client = LLMClientFactory.build(_cfg("openrouter", base_url=None))
        assert isinstance(client, AsyncOpenAI)
        assert "openrouter" in str(client.base_url)

    def test_azure_openai_returns_async_azure_openai(self):
        from openai import AsyncAzureOpenAI
        client = LLMClientFactory.build(
            _cfg(
                "azure-openai",
                base_url="https://my-resource.openai.azure.com/",
                api_version="2024-02-01",
            )
        )
        assert isinstance(client, AsyncAzureOpenAI)

    def test_azure_openai_uses_default_api_version_when_none(self):
        from openai import AsyncAzureOpenAI
        client = LLMClientFactory.build(
            _cfg("azure-openai", base_url="https://resource.openai.azure.com/")
        )
        assert isinstance(client, AsyncAzureOpenAI)

    def test_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMClientFactory.build(_cfg("cohere"))
