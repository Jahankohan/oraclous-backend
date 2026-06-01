"""Unit tests for GET /api/v1/tools/manifest (ORA-347).

Fast, no-I/O tests — no Neo4j, no Postgres. The endpoint is purely static.
"""

import pytest

from app.api.v1.endpoints.tools import _MANIFEST
from app.core.config import settings

EXPECTED_TOOL_COUNT = 9
EXPECTED_TOOL_NAMES = {
    "kg.create_graph",
    "kg.ingest",
    "kg.run_recipe",
    "kg.query",
    "kg.ask",
    "kg.store_fact",
    "kg.recall",
    "kg.federate",
    "kg.job_status",
}
REQUIRED_TOOL_FIELDS = {
    "name",
    "description",
    "input_schema",
    "output_schema",
    "idempotent",
    "async",
}


class TestManifestStructure:
    @pytest.mark.unit
    def test_manifest_has_required_top_level_fields(self):
        for field in ("service", "version", "base_url", "tools"):
            assert field in _MANIFEST, f"Missing top-level field: {field}"

    @pytest.mark.unit
    def test_service_name(self):
        assert _MANIFEST["service"] == settings.SERVICE_NAME

    @pytest.mark.unit
    def test_version_semver_format(self):
        parts = _MANIFEST["version"].split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    @pytest.mark.unit
    def test_tool_count_is_nine(self):
        assert len(_MANIFEST["tools"]) == EXPECTED_TOOL_COUNT

    @pytest.mark.unit
    def test_all_expected_tool_names_present(self):
        names = {t["name"] for t in _MANIFEST["tools"]}
        assert names == EXPECTED_TOOL_NAMES

    @pytest.mark.unit
    def test_every_tool_has_required_fields(self):
        for tool in _MANIFEST["tools"]:
            missing = REQUIRED_TOOL_FIELDS - tool.keys()
            assert not missing, f"Tool {tool.get('name')} missing fields: {missing}"

    @pytest.mark.unit
    def test_idempotent_and_async_are_booleans(self):
        for tool in _MANIFEST["tools"]:
            assert isinstance(tool["idempotent"], bool), (
                f"{tool['name']}.idempotent must be bool"
            )
            assert isinstance(tool["async"], bool), f"{tool['name']}.async must be bool"

    @pytest.mark.unit
    def test_async_tools_are_ingest_and_run_recipe(self):
        async_tools = {t["name"] for t in _MANIFEST["tools"] if t["async"]}
        assert async_tools == {"kg.ingest", "kg.run_recipe"}

    @pytest.mark.unit
    def test_input_and_output_schemas_are_dicts(self):
        for tool in _MANIFEST["tools"]:
            assert isinstance(tool["input_schema"], dict), (
                f"{tool['name']}.input_schema must be dict"
            )
            assert isinstance(tool["output_schema"], dict), (
                f"{tool['name']}.output_schema must be dict"
            )

    @pytest.mark.unit
    def test_manifest_is_stable_across_calls(self):
        """_MANIFEST is a module-level singleton; calling _build_manifest() again
        should produce equal content."""
        from app.api.v1.endpoints.tools import _build_manifest

        fresh = _build_manifest()
        assert fresh["service"] == _MANIFEST["service"]
        assert fresh["version"] == _MANIFEST["version"]
        assert len(fresh["tools"]) == len(_MANIFEST["tools"])
