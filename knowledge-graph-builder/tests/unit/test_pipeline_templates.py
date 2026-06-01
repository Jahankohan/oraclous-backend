"""Unit tests for the pipeline template catalogue (ORA-352).

Fast, no-I/O tests — no Neo4j, no Postgres. Templates are purely static.
HTTP endpoint functions are called directly (no TestClient) to avoid app startup.
"""

from __future__ import annotations

import pytest

from app.api.v1.endpoints.pipeline_templates import (
    get_pipeline_template,
    list_pipeline_templates,
)
from app.schemas.pipeline_template_schemas import PIPELINE_TEMPLATES

EXPECTED_TEMPLATE_IDS = {
    "hr-people-graph",
    "research-knowledge-base",
    "legal-document-graph",
    "codebase-architecture-graph",
}

EXPECTED_DOMAINS = {"hr", "research", "legal", "codebase"}

VALID_NODE_TYPES = {
    "create-graph",
    "ingest-file",
    "ingest-url",
    "ingest-text",
    "ingest-db",
    "ingest-api",
    "ingest-records",
    "community-detect",
    "community-summarize",
    "entity-dedup",
    "similarity-build",
    "ontology-apply",
    "ontology-retroactive",
    "graph-output",
    "export-json",
    "sequential-gate",
    "review-gate",
}

VALID_EDGE_TYPES = {"data-flow", "trigger"}


# ---------------------------------------------------------------------------
# Schema-level tests (pure data, no HTTP)
# ---------------------------------------------------------------------------


class TestCatalogueCompleteness:
    @pytest.mark.unit
    def test_all_four_templates_present(self):
        assert set(PIPELINE_TEMPLATES.keys()) == EXPECTED_TEMPLATE_IDS

    @pytest.mark.unit
    def test_domains_are_correct(self):
        domains = {t.domain for t in PIPELINE_TEMPLATES.values()}
        assert domains == EXPECTED_DOMAINS

    @pytest.mark.unit
    def test_every_template_has_at_least_one_param(self):
        for template_id, template in PIPELINE_TEMPLATES.items():
            assert len(template.params) >= 1, f"{template_id} has no params"

    @pytest.mark.unit
    def test_every_template_has_create_graph_node(self):
        for template_id, template in PIPELINE_TEMPLATES.items():
            types = {n.type for n in template.nodes}
            assert "create-graph" in types, f"{template_id} missing create-graph node"

    @pytest.mark.unit
    def test_every_template_has_graph_output_node(self):
        for template_id, template in PIPELINE_TEMPLATES.items():
            types = {n.type for n in template.nodes}
            assert "graph-output" in types, f"{template_id} missing graph-output node"

    @pytest.mark.unit
    def test_every_template_has_at_least_one_edge(self):
        for template_id, template in PIPELINE_TEMPLATES.items():
            assert len(template.edges) >= 1, f"{template_id} has no edges"

    @pytest.mark.unit
    def test_all_node_types_are_valid(self):
        for template_id, template in PIPELINE_TEMPLATES.items():
            for node in template.nodes:
                assert node.type in VALID_NODE_TYPES, (
                    f"{template_id}: unknown node type '{node.type}'"
                )

    @pytest.mark.unit
    def test_all_edge_types_are_valid(self):
        for template_id, template in PIPELINE_TEMPLATES.items():
            for edge in template.edges:
                assert edge.type in VALID_EDGE_TYPES, (
                    f"{template_id}: unknown edge type '{edge.type}'"
                )

    @pytest.mark.unit
    def test_edge_endpoints_reference_existing_nodes(self):
        for template_id, template in PIPELINE_TEMPLATES.items():
            node_ids = {n.id for n in template.nodes}
            for edge in template.edges:
                assert edge.source in node_ids, (
                    f"{template_id}: edge '{edge.id}' references unknown source '{edge.source}'"
                )
                assert edge.target in node_ids, (
                    f"{template_id}: edge '{edge.id}' references unknown target '{edge.target}'"
                )

    @pytest.mark.unit
    def test_node_ids_are_unique_within_template(self):
        for template_id, template in PIPELINE_TEMPLATES.items():
            ids = [n.id for n in template.nodes]
            assert len(ids) == len(set(ids)), f"{template_id}: duplicate node ids"

    @pytest.mark.unit
    def test_edge_ids_are_unique_within_template(self):
        for template_id, template in PIPELINE_TEMPLATES.items():
            ids = [e.id for e in template.edges]
            assert len(ids) == len(set(ids)), f"{template_id}: duplicate edge ids"

    @pytest.mark.unit
    def test_param_keys_are_unique_within_template(self):
        for template_id, template in PIPELINE_TEMPLATES.items():
            keys = [p.key for p in template.params]
            assert len(keys) == len(set(keys)), f"{template_id}: duplicate param keys"

    @pytest.mark.unit
    def test_graph_name_param_in_every_template(self):
        for template_id, template in PIPELINE_TEMPLATES.items():
            keys = {p.key for p in template.params}
            assert "graph_name" in keys, f"{template_id}: missing 'graph_name' param"

    @pytest.mark.unit
    def test_llm_config_id_param_in_every_template(self):
        for template_id, template in PIPELINE_TEMPLATES.items():
            keys = {p.key for p in template.params}
            assert "llm_config_id" in keys, (
                f"{template_id}: missing 'llm_config_id' param"
            )

    @pytest.mark.unit
    def test_create_graph_nodes_use_graph_name_placeholder(self):
        for template_id, template in PIPELINE_TEMPLATES.items():
            for node in template.nodes:
                if node.type == "create-graph":
                    name_value = node.data.get("config", {}).get("name", "")
                    assert "{{graph_name}}" in name_value, (
                        f"{template_id}: create-graph 'name' missing '{{{{graph_name}}}}' placeholder"
                    )

    @pytest.mark.unit
    def test_legal_template_has_review_gate(self):
        legal = PIPELINE_TEMPLATES["legal-document-graph"]
        types = {n.type for n in legal.nodes}
        assert "review-gate" in types

    @pytest.mark.unit
    def test_legal_template_has_reviewer_email_param(self):
        legal = PIPELINE_TEMPLATES["legal-document-graph"]
        keys = {p.key for p in legal.params}
        assert "reviewer_email" in keys

    @pytest.mark.unit
    def test_codebase_template_has_repo_url_param(self):
        codebase = PIPELINE_TEMPLATES["codebase-architecture-graph"]
        keys = {p.key for p in codebase.params}
        assert "repo_url" in keys

    @pytest.mark.unit
    def test_research_template_has_similarity_build_node(self):
        research = PIPELINE_TEMPLATES["research-knowledge-base"]
        types = {n.type for n in research.nodes}
        assert "similarity-build" in types


# ---------------------------------------------------------------------------
# Endpoint function tests (async, direct call — no TestClient / no DB)
# ---------------------------------------------------------------------------


class TestListEndpointFunction:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_returns_all_four_templates(self):
        response = await list_pipeline_templates()
        assert response.total == 4
        assert len(response.templates) == 4

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_summary_omits_nodes_and_edges(self):
        response = await list_pipeline_templates()
        for summary in response.templates:
            assert not hasattr(summary, "nodes") or not hasattr(summary, "edges")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_all_expected_ids_present(self):
        response = await list_pipeline_templates()
        returned_ids = {s.id for s in response.templates}
        assert returned_ids == EXPECTED_TEMPLATE_IDS

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_node_count_positive(self):
        response = await list_pipeline_templates()
        for summary in response.templates:
            assert summary.node_count > 0


class TestDetailEndpointFunction:
    @pytest.mark.unit
    @pytest.mark.asyncio
    @pytest.mark.parametrize("template_id", list(EXPECTED_TEMPLATE_IDS))
    async def test_returns_full_template(self, template_id: str):
        template = await get_pipeline_template(template_id)
        assert template.id == template_id
        assert len(template.nodes) > 0
        assert len(template.edges) > 0
        assert len(template.params) >= 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_unknown_id_raises_404(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await get_pipeline_template("non-existent-template")
        assert exc_info.value.status_code == 404
