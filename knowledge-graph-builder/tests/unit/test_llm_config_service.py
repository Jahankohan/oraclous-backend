"""Unit tests for LLMConfigService (STORY-021 / TASK-040)."""

from unittest.mock import AsyncMock, MagicMock

from app.schemas.llm_config_schemas import LLMConfigCreate, LLMProvider
from app.services.llm_config_service import LLMConfigService


def _make_create(
    provider: str = "openrouter", model: str = "openai/gpt-4o"
) -> LLMConfigCreate:
    return LLMConfigCreate(
        provider=LLMProvider(provider),
        model=model,
        api_key="sk-test-1234abcd",
        base_url="https://openrouter.ai/api/v1" if provider == "openrouter" else None,
    )


def _mock_driver(records=None):
    driver = MagicMock()
    result = MagicMock()
    result.records = records or []
    driver.execute_query = AsyncMock(return_value=result)
    return driver


# ── create_org_config ─────────────────────────────────────────────────────────


class TestCreateOrgConfig:
    async def test_writes_api_key_ref_not_plaintext_key(self):
        driver = _mock_driver()
        svc = LLMConfigService(driver)
        config_id = await svc.create_org_config(
            "org1", "user1", _make_create(), "cred-ref-001"
        )

        assert isinstance(config_id, str) and len(config_id) == 36
        query, params = driver.execute_query.call_args.args
        assert "api_key_ref" in query
        # "api_key_ref" contains the substring "api_key" — check params, not raw query
        assert params["api_key_ref"] == "cred-ref-001"
        assert "api_key" not in params  # plaintext key never passed to Cypher

    async def test_sets_scope_org(self):
        driver = _mock_driver()
        svc = LLMConfigService(driver)
        await svc.create_org_config("org1", "user1", _make_create(), "cref")
        query, _ = driver.execute_query.call_args.args
        assert '"org"' in query

    async def test_graph_id_is_null_for_org_scope(self):
        driver = _mock_driver()
        svc = LLMConfigService(driver)
        await svc.create_org_config("org1", "user1", _make_create(), "cref")
        query, _ = driver.execute_query.call_args.args
        assert "graph_id:       null" in query


# ── create_project_config ─────────────────────────────────────────────────────


class TestCreateProjectConfig:
    async def test_sets_scope_project_and_graph_id(self):
        driver = _mock_driver()
        svc = LLMConfigService(driver)
        config_id = await svc.create_project_config(
            "graph-A", "org1", "user1", _make_create(), "cref"
        )
        assert isinstance(config_id, str)
        query, params = driver.execute_query.call_args.args
        assert '"project"' in query
        assert params["graph_id"] == "graph-A"


# ── list_org_configs ──────────────────────────────────────────────────────────


class TestListOrgConfigs:
    async def test_returns_only_active_configs(self):
        rec1 = MagicMock()
        rec1.__getitem__ = lambda self, k: (
            {"config_id": "c1", "scope": "org"} if k == "c" else None
        )
        driver = MagicMock()
        result = MagicMock()
        result.records = [rec1]
        driver.execute_query = AsyncMock(return_value=result)

        svc = LLMConfigService(driver)
        await svc.list_org_configs("org1")
        query, params = driver.execute_query.call_args.args
        assert "deactivated_at IS NULL" in query
        assert params["org_id"] == "org1"

    async def test_empty_when_no_configs(self):
        driver = _mock_driver(records=[])
        svc = LLMConfigService(driver)
        configs = await svc.list_org_configs("org-empty")
        assert configs == []


# ── deactivate_config ─────────────────────────────────────────────────────────


class TestDeactivateConfig:
    async def test_returns_true_on_success(self):
        rec = MagicMock()
        driver = _mock_driver(records=[rec])
        svc = LLMConfigService(driver)
        result = await svc.deactivate_config("cfg-1", "org1")
        assert result is True
        query, params = driver.execute_query.call_args.args
        assert "deactivated_at" in query
        assert params["org_id"] == "org1"
        assert params["cid"] == "cfg-1"

    async def test_returns_false_when_not_found_or_wrong_org(self):
        driver = _mock_driver(records=[])
        svc = LLMConfigService(driver)
        result = await svc.deactivate_config("cfg-1", "org-B")
        assert result is False


# ── resolve_for_agent ─────────────────────────────────────────────────────────


class TestResolveForAgent:
    def _svc_with_patches(
        self,
        get_config_return=None,
        project_configs=None,
        org_configs=None,
    ):
        driver = MagicMock()
        svc = LLMConfigService(driver)
        svc.get_config = AsyncMock(return_value=get_config_return)
        svc.list_project_configs = AsyncMock(return_value=project_configs or [])
        svc.list_org_configs = AsyncMock(return_value=org_configs or [])
        return svc

    async def test_agent_config_wins_over_project(self):
        agent_cfg = {"config_id": "agent-cfg", "deactivated_at": None}
        project_cfg = {"config_id": "proj-cfg", "deactivated_at": None}
        svc = self._svc_with_patches(
            get_config_return=agent_cfg, project_configs=[project_cfg]
        )
        result = await svc.resolve_for_agent("g", "org", "agent-cfg")
        assert result["config_id"] == "agent-cfg"
        svc.list_project_configs.assert_not_called()

    async def test_project_config_wins_over_org(self):
        project_cfg = {"config_id": "proj-cfg", "deactivated_at": None}
        org_cfg = {"config_id": "org-cfg", "deactivated_at": None}
        svc = self._svc_with_patches(
            project_configs=[project_cfg], org_configs=[org_cfg]
        )
        result = await svc.resolve_for_agent("g", "org", None)
        assert result["config_id"] == "proj-cfg"
        svc.list_org_configs.assert_not_called()

    async def test_org_config_used_when_no_project_config(self):
        org_cfg = {"config_id": "org-cfg", "deactivated_at": None}
        svc = self._svc_with_patches(org_configs=[org_cfg])
        result = await svc.resolve_for_agent("g", "org1", None)
        assert result["config_id"] == "org-cfg"

    async def test_returns_none_when_nothing_found(self):
        svc = self._svc_with_patches()
        result = await svc.resolve_for_agent("g", "org1", None)
        assert result is None

    async def test_skips_deactivated_agent_config_and_falls_through(self):
        deactivated = {"config_id": "dead", "deactivated_at": 12345}
        project_cfg = {"config_id": "proj", "deactivated_at": None}
        svc = self._svc_with_patches(
            get_config_return=deactivated, project_configs=[project_cfg]
        )
        result = await svc.resolve_for_agent("g", "org1", "dead")
        assert result["config_id"] == "proj"
