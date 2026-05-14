"""Unit tests for AgentExecutor LLM resolution chain (STORY-021 / TASK-040).

Verifies the agent → project → org → env-var fallback chain and the
Anthropic vs OpenAI-compatible _call_llm dispatch.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.agent_executor import AgentExecutor


def _make_agent(llm_config_id=None, org_id="org1"):
    return {
        "agent_id": "a1",
        "graph_id": "g1",
        "name": "Test",
        "system_prompt": "You are helpful.",
        "reasoning_mode": "direct",
        "tools": [],
        "llm_config_id": llm_config_id,
        "org_id": org_id,
    }


def _resolved_config(provider="openrouter"):
    return {
        "config_id": "cfg-1",
        "provider": provider,
        "model": "openai/gpt-4o",
        "base_url": "https://openrouter.ai/api/v1",
        "api_version": None,
        "api_key_ref": "cred-001",
        "deactivated_at": None,
    }


class TestFromNeo4jResolution:
    async def test_uses_resolved_config_when_available(self):
        driver = MagicMock()
        agent = _make_agent()
        mock_llm = MagicMock()

        with (
            patch("app.services.agent_executor.AgentService") as MockSvc,
            patch("app.services.agent_executor.LLMConfigService") as MockCfg,
            patch("app.services.agent_executor.CredentialBrokerClient") as MockBroker,
            patch("app.services.agent_executor.LLMClientFactory") as MockFactory,
        ):
            MockSvc.return_value.get_agent = AsyncMock(return_value=agent)
            MockCfg.return_value.resolve_for_agent = AsyncMock(
                return_value=_resolved_config()
            )
            MockBroker.return_value.retrieve_api_key = AsyncMock(
                return_value="sk-or-live"
            )
            MockFactory.build = MagicMock(return_value=mock_llm)

            executor = await AgentExecutor.from_neo4j(driver, "g1", "a1")

        assert executor._llm is mock_llm
        MockFactory.build.assert_called_once()
        build_args = MockFactory.build.call_args[0][0]
        assert build_args["api_key"] == "sk-or-live"
        assert build_args["provider"] == "openrouter"

    async def test_falls_back_to_env_var_when_no_config(self):
        from openai import AsyncOpenAI

        driver = MagicMock()
        agent = _make_agent()

        with (
            patch("app.services.agent_executor.AgentService") as MockSvc,
            patch("app.services.agent_executor.LLMConfigService") as MockCfg,
            patch("app.services.agent_executor.settings") as mock_settings,
        ):
            MockSvc.return_value.get_agent = AsyncMock(return_value=agent)
            MockCfg.return_value.resolve_for_agent = AsyncMock(return_value=None)
            mock_settings.LLM_API_KEY = "sk-fallback"
            mock_settings.OPENAI_API_KEY = None
            mock_settings.LLM_MODEL = "gpt-4o"
            mock_settings.CREDENTIAL_BROKER_URL = "http://broker:8000"

            executor = await AgentExecutor.from_neo4j(driver, "g1", "a1")

        assert isinstance(executor._llm, AsyncOpenAI)
        assert executor._model == "gpt-4o"

    async def test_openai_api_key_used_as_fallback_when_llm_api_key_unset(self):
        from openai import AsyncOpenAI

        driver = MagicMock()
        agent = _make_agent()

        with (
            patch("app.services.agent_executor.AgentService") as MockSvc,
            patch("app.services.agent_executor.LLMConfigService") as MockCfg,
            patch("app.services.agent_executor.settings") as mock_settings,
        ):
            MockSvc.return_value.get_agent = AsyncMock(return_value=agent)
            MockCfg.return_value.resolve_for_agent = AsyncMock(return_value=None)
            mock_settings.LLM_API_KEY = None
            mock_settings.OPENAI_API_KEY = "sk-openai-key"
            mock_settings.LLM_MODEL = "gpt-4o"
            mock_settings.CREDENTIAL_BROKER_URL = "http://broker:8000"

            executor = await AgentExecutor.from_neo4j(driver, "g1", "a1")

        assert isinstance(executor._llm, AsyncOpenAI)

    async def test_raises_runtime_error_when_no_key_at_any_level(self):
        driver = MagicMock()
        agent = _make_agent()

        with (
            patch("app.services.agent_executor.AgentService") as MockSvc,
            patch("app.services.agent_executor.LLMConfigService") as MockCfg,
            patch("app.services.agent_executor.settings") as mock_settings,
        ):
            MockSvc.return_value.get_agent = AsyncMock(return_value=agent)
            MockCfg.return_value.resolve_for_agent = AsyncMock(return_value=None)
            mock_settings.LLM_API_KEY = None
            mock_settings.OPENAI_API_KEY = None
            mock_settings.LLM_MODEL = "gpt-4o"
            mock_settings.CREDENTIAL_BROKER_URL = "http://broker:8000"

            with pytest.raises(RuntimeError, match="LLM not configured"):
                await AgentExecutor.from_neo4j(driver, "g1", "a1")

    async def test_broker_error_surfaces_as_runtime_error(self):
        from app.services.credential_broker_client import CredentialBrokerError

        driver = MagicMock()
        agent = _make_agent()

        with (
            patch("app.services.agent_executor.AgentService") as MockSvc,
            patch("app.services.agent_executor.LLMConfigService") as MockCfg,
            patch("app.services.agent_executor.CredentialBrokerClient") as MockBroker,
            patch("app.services.agent_executor.settings") as mock_settings,
        ):
            MockSvc.return_value.get_agent = AsyncMock(return_value=agent)
            MockCfg.return_value.resolve_for_agent = AsyncMock(
                return_value=_resolved_config()
            )
            MockBroker.return_value.retrieve_api_key = AsyncMock(
                side_effect=CredentialBrokerError("not found")
            )
            mock_settings.CREDENTIAL_BROKER_URL = "http://broker:8000"

            with pytest.raises(RuntimeError, match="Could not retrieve LLM API key"):
                await AgentExecutor.from_neo4j(driver, "g1", "a1")


class TestCallLlmDispatch:
    def _executor(self, llm, model="gpt-4o"):
        agent = {
            "graph_id": "g1",
            "system_prompt": "You help.",
            "reasoning_mode": "direct",
            "tools": [],
            "llm_config_id": None,
        }
        return AgentExecutor(agent_def=agent, toolkit=MagicMock(), llm=llm, model=model)

    async def test_openai_uses_chat_completions_create(self):
        from openai import AsyncOpenAI

        llm = MagicMock(spec=AsyncOpenAI)
        resp = MagicMock()
        resp.choices[0].message.content = "hello"
        resp.choices[0].message.tool_calls = None
        llm.chat.completions.create = AsyncMock(return_value=resp)
        ex = self._executor(llm)
        result = await ex._call_llm([{"role": "user", "content": "hi"}])
        # _call_llm now returns _LLMResponse(text, tool_calls)
        assert result.text == "hello"
        assert result.tool_calls == []
        llm.chat.completions.create.assert_called_once()

    async def test_anthropic_uses_messages_create(self):
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            pytest.skip("anthropic SDK not installed")

        llm = MagicMock(spec=AsyncAnthropic)
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "anthropic response"
        resp = MagicMock()
        resp.content = [text_block]
        llm.messages.create = AsyncMock(return_value=resp)
        ex = self._executor(llm)
        result = await ex._call_llm([{"role": "user", "content": "hi"}])
        assert result.text == "anthropic response"
        assert result.tool_calls == []
        llm.messages.create.assert_called_once()
        call_kwargs = llm.messages.create.call_args[1]
        assert "system" in call_kwargs
        assert "messages" in call_kwargs
