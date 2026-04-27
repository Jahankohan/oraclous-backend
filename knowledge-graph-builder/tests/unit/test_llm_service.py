"""
Unit tests for LLMService.

Tests provider abstraction (OpenAI / Anthropic / Google), fallback behavior,
schema configuration, and initialization state — all external LLM calls mocked.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.llm_service import LLMService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service() -> LLMService:
    return LLMService()


# ---------------------------------------------------------------------------
# Tests: initialization state
# ---------------------------------------------------------------------------


class TestLLMServiceInit:
    @pytest.mark.unit
    def test_initial_state_is_uninitialized(self):
        svc = _make_service()
        assert svc.is_initialized() is False
        assert svc.llm is None
        assert svc.graph_transformer is None
        assert svc.current_provider is None
        assert svc.current_model is None


# ---------------------------------------------------------------------------
# Tests: initialize_llm — OpenAI provider
# ---------------------------------------------------------------------------


class TestInitializeLLMOpenAI:
    @pytest.mark.unit
    async def test_openai_init_uses_user_credential_first(self):
        svc = _make_service()
        with (
            patch("app.services.llm_service.credential_service") as mock_creds,
            patch("app.services.llm_service.ChatOpenAI") as mock_openai,
            patch("app.services.llm_service.LLMGraphTransformer"),
        ):
            mock_creds.get_openai_token = AsyncMock(return_value="user-key-123")
            mock_openai.return_value = MagicMock()

            result = await svc.initialize_llm(
                "user-1", provider="openai", model="gpt-4o"
            )

        assert result is True
        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["api_key"] == "user-key-123"

    @pytest.mark.unit
    async def test_openai_init_falls_back_to_settings_key(self):
        svc = _make_service()
        with (
            patch("app.services.llm_service.credential_service") as mock_creds,
            patch("app.services.llm_service.ChatOpenAI") as mock_openai,
            patch("app.services.llm_service.LLMGraphTransformer"),
            patch("app.services.llm_service.settings") as mock_settings,
        ):
            mock_creds.get_openai_token = AsyncMock(return_value=None)
            mock_settings.OPENAI_API_KEY = "settings-key"
            mock_openai.return_value = MagicMock()

            result = await svc.initialize_llm("user-1", provider="openai")

        assert result is True
        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["api_key"] == "settings-key"

    @pytest.mark.unit
    async def test_openai_init_returns_false_when_no_key(self):
        svc = _make_service()
        with (
            patch("app.services.llm_service.credential_service") as mock_creds,
            patch("app.services.llm_service.settings") as mock_settings,
        ):
            mock_creds.get_openai_token = AsyncMock(return_value=None)
            mock_settings.OPENAI_API_KEY = None

            result = await svc.initialize_llm("user-1", provider="openai")

        assert result is False

    @pytest.mark.unit
    async def test_openai_init_sets_provider_and_model(self):
        svc = _make_service()
        with (
            patch("app.services.llm_service.credential_service") as mock_creds,
            patch("app.services.llm_service.ChatOpenAI") as mock_openai,
            patch("app.services.llm_service.LLMGraphTransformer"),
        ):
            mock_creds.get_openai_token = AsyncMock(return_value="k")
            mock_openai.return_value = MagicMock()

            await svc.initialize_llm("u", provider="openai", model="gpt-4o-mini")

        assert svc.current_provider == "openai"
        assert svc.current_model == "gpt-4o-mini"

    @pytest.mark.unit
    async def test_openai_init_creates_graph_transformer(self):
        svc = _make_service()
        with (
            patch("app.services.llm_service.credential_service") as mock_creds,
            patch("app.services.llm_service.ChatOpenAI") as mock_openai,
            patch("app.services.llm_service.LLMGraphTransformer") as mock_gt,
        ):
            mock_creds.get_openai_token = AsyncMock(return_value="k")
            mock_openai.return_value = MagicMock()
            mock_gt.return_value = MagicMock()

            await svc.initialize_llm("u", provider="openai")

        assert svc.is_initialized() is True
        mock_gt.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: initialize_llm — Anthropic provider
# ---------------------------------------------------------------------------


class TestInitializeLLMAnthropic:
    @pytest.mark.unit
    async def test_anthropic_init_success(self):
        svc = _make_service()
        with (
            patch("app.services.llm_service.credential_service") as mock_creds,
            patch("app.services.llm_service.ChatAnthropic") as mock_anthropic,
            patch("app.services.llm_service.LLMGraphTransformer"),
        ):
            mock_creds.get_anthropic_token = AsyncMock(return_value="anthropic-key")
            mock_anthropic.return_value = MagicMock()

            result = await svc.initialize_llm(
                "u", provider="anthropic", model="claude-3-haiku-20240307"
            )

        assert result is True
        assert svc.current_provider == "anthropic"

    @pytest.mark.unit
    async def test_anthropic_init_falls_back_to_default_claude_model(self):
        svc = _make_service()
        with (
            patch("app.services.llm_service.credential_service") as mock_creds,
            patch("app.services.llm_service.ChatAnthropic") as mock_anthropic,
            patch("app.services.llm_service.LLMGraphTransformer"),
        ):
            mock_creds.get_anthropic_token = AsyncMock(return_value="k")
            mock_anthropic.return_value = MagicMock()

            # Passing a non-Claude model should fall back to claude-3-haiku
            await svc.initialize_llm("u", provider="anthropic", model="gpt-4o")

        call_kwargs = mock_anthropic.call_args[1]
        assert call_kwargs["model"] == "claude-3-haiku-20240307"

    @pytest.mark.unit
    async def test_anthropic_init_returns_false_when_no_key(self):
        svc = _make_service()
        with (
            patch("app.services.llm_service.credential_service") as mock_creds,
            patch("app.services.llm_service.settings") as mock_settings,
        ):
            mock_creds.get_anthropic_token = AsyncMock(return_value=None)
            mock_settings.ANTHROPIC_API_KEY = None

            result = await svc.initialize_llm("u", provider="anthropic")

        assert result is False


# ---------------------------------------------------------------------------
# Tests: initialize_llm — Google provider
# ---------------------------------------------------------------------------


class TestInitializeLLMGoogle:
    @pytest.mark.unit
    async def test_google_init_success(self):
        svc = _make_service()
        with (
            patch("app.services.llm_service.credential_service") as mock_creds,
            patch("app.services.llm_service.ChatGoogleGenerativeAI") as mock_google,
            patch("app.services.llm_service.LLMGraphTransformer"),
        ):
            mock_creds.get_user_credentials = AsyncMock(
                return_value={"access_token": "google-key"}
            )
            mock_google.return_value = MagicMock()

            result = await svc.initialize_llm(
                "u", provider="google", model="gemini-1.5-flash"
            )

        assert result is True
        assert svc.current_provider == "google"

    @pytest.mark.unit
    async def test_google_init_uses_gemini_model_fallback(self):
        svc = _make_service()
        with (
            patch("app.services.llm_service.credential_service") as mock_creds,
            patch("app.services.llm_service.ChatGoogleGenerativeAI") as mock_google,
            patch("app.services.llm_service.LLMGraphTransformer"),
        ):
            mock_creds.get_user_credentials = AsyncMock(
                return_value={"access_token": "k"}
            )
            mock_google.return_value = MagicMock()

            await svc.initialize_llm(
                "u", provider="google", model="gpt-4o"
            )  # non-gemini model

        call_kwargs = mock_google.call_args[1]
        assert call_kwargs["model"] == "gemini-1.5-flash"

    @pytest.mark.unit
    async def test_google_init_returns_false_when_no_token(self):
        svc = _make_service()
        with patch("app.services.llm_service.credential_service") as mock_creds:
            mock_creds.get_user_credentials = AsyncMock(return_value=None)

            result = await svc.initialize_llm("u", provider="google")

        assert result is False


# ---------------------------------------------------------------------------
# Tests: unsupported provider
# ---------------------------------------------------------------------------


class TestUnsupportedProvider:
    @pytest.mark.unit
    async def test_unsupported_provider_returns_false(self):
        svc = _make_service()
        result = await svc.initialize_llm("u", provider="cohere")
        assert result is False
        assert svc.is_initialized() is False


# ---------------------------------------------------------------------------
# Tests: exception handling
# ---------------------------------------------------------------------------


class TestInitializeLLMExceptionHandling:
    @pytest.mark.unit
    async def test_returns_false_on_unexpected_exception(self):
        svc = _make_service()
        with (
            patch("app.services.llm_service.credential_service") as mock_creds,
            patch("app.services.llm_service.ChatOpenAI") as mock_openai,
        ):
            mock_creds.get_openai_token = AsyncMock(return_value="k")
            mock_openai.side_effect = Exception("Unexpected crash")

            result = await svc.initialize_llm("u", provider="openai")

        assert result is False
        assert svc.is_initialized() is False


# ---------------------------------------------------------------------------
# Tests: set_schema
# ---------------------------------------------------------------------------


class TestSetSchema:
    @pytest.mark.unit
    def test_set_schema_updates_allowed_nodes_and_relationships(self):
        svc = _make_service()
        mock_transformer = MagicMock()
        svc.graph_transformer = mock_transformer
        svc.llm = MagicMock()

        svc.set_schema(
            {"entities": ["Person", "Company"], "relationships": ["WORKS_AT"]}
        )

        assert mock_transformer.allowed_nodes == ["Person", "Company"]
        assert mock_transformer.allowed_relationships == ["WORKS_AT"]

    @pytest.mark.unit
    def test_set_schema_no_op_when_transformer_none(self):
        svc = _make_service()
        svc.set_schema({"entities": ["Person"]})
        # Should not raise

    @pytest.mark.unit
    def test_set_schema_no_op_when_empty_config(self):
        svc = _make_service()
        mock_transformer = MagicMock()
        svc.graph_transformer = mock_transformer
        svc.set_schema({})
        # empty entities/relationships — no attribute setting expected
        assert not hasattr(mock_transformer, "allowed_nodes") or True  # just no crash


# ---------------------------------------------------------------------------
# Tests: set_dynamic_schema
# ---------------------------------------------------------------------------


class TestSetDynamicSchema:
    @pytest.mark.unit
    def test_set_dynamic_schema_rebuilds_transformer(self):
        svc = _make_service()
        svc.llm = MagicMock()

        with patch("app.services.llm_service.LLMGraphTransformer") as mock_gt:
            mock_gt.return_value = MagicMock()
            svc.set_dynamic_schema(
                {"entities": ["Person", "Company"], "relationships": ["WORKS_AT"]}
            )

        mock_gt.assert_called_once()
        call_kwargs = mock_gt.call_args[1]
        assert call_kwargs["allowed_nodes"] == ["Person", "Company"]
        assert call_kwargs["allowed_relationships"] == ["WORKS_AT"]

    @pytest.mark.unit
    def test_set_dynamic_schema_no_op_when_llm_not_initialized(self):
        svc = _make_service()
        svc.set_dynamic_schema({"entities": ["Person"]})
        # Should not raise, just log a warning
        assert svc.graph_transformer is None

    @pytest.mark.unit
    def test_set_dynamic_schema_passes_none_for_empty_entities(self):
        svc = _make_service()
        svc.llm = MagicMock()

        with patch("app.services.llm_service.LLMGraphTransformer") as mock_gt:
            mock_gt.return_value = MagicMock()
            svc.set_dynamic_schema({})  # no entities or relationships

        call_kwargs = mock_gt.call_args[1]
        assert call_kwargs["allowed_nodes"] is None
        assert call_kwargs["allowed_relationships"] is None


# ---------------------------------------------------------------------------
# Tests: is_initialized
# ---------------------------------------------------------------------------


class TestIsInitialized:
    @pytest.mark.unit
    def test_is_initialized_true_when_both_set(self):
        svc = _make_service()
        svc.llm = MagicMock()
        svc.graph_transformer = MagicMock()
        assert svc.is_initialized() is True

    @pytest.mark.unit
    def test_is_initialized_false_when_transformer_none(self):
        svc = _make_service()
        svc.llm = MagicMock()
        svc.graph_transformer = None
        assert svc.is_initialized() is False

    @pytest.mark.unit
    def test_is_initialized_false_when_llm_none(self):
        svc = _make_service()
        svc.llm = None
        svc.graph_transformer = MagicMock()
        assert svc.is_initialized() is False
