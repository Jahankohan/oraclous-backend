"""
QA integration tests — pipeline template catalogue routes (ORA-352 / ORA-388).

Scope: HTTP route behaviour via FastAPI TestClient.
Does not require a running server, Neo4j, or Redis.
Tag: pytest -m qa
"""

from __future__ import annotations

import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.endpoints.pipeline_templates import router

# ---------------------------------------------------------------------------
# App fixture (isolated from main app to avoid DB deps)
# ---------------------------------------------------------------------------

app = FastAPI()
app.include_router(router, prefix="/api/v1")

client = TestClient(app)

EXPECTED_IDS = frozenset(
    {
        "hr-people-graph",
        "research-knowledge-base",
        "legal-document-graph",
        "codebase-architecture-graph",
    }
)
EXPECTED_DOMAINS = frozenset({"hr", "research", "legal", "codebase"})


# ---------------------------------------------------------------------------
# Catalogue endpoint — GET /api/v1/pipeline-templates
# ---------------------------------------------------------------------------


class TestCatalogueEndpoint:
    def test_returns_200(self):
        assert client.get("/api/v1/pipeline-templates").status_code == 200

    def test_no_auth_required(self):
        # Static public data — no Authorization header needed
        assert client.get("/api/v1/pipeline-templates").status_code == 200

    def test_response_shape(self):
        data = client.get("/api/v1/pipeline-templates").json()
        assert "templates" in data
        assert data["total"] == 4
        assert len(data["templates"]) == 4

    def test_all_four_ids_present(self):
        data = client.get("/api/v1/pipeline-templates").json()
        assert {t["id"] for t in data["templates"]} == EXPECTED_IDS

    def test_all_four_domains_present(self):
        data = client.get("/api/v1/pipeline-templates").json()
        assert {t["domain"] for t in data["templates"]} == EXPECTED_DOMAINS

    def test_summary_has_required_fields(self):
        data = client.get("/api/v1/pipeline-templates").json()
        required = {
            "id",
            "name",
            "description",
            "domain",
            "tags",
            "node_count",
            "param_count",
        }
        for t in data["templates"]:
            missing = required - set(t.keys())
            assert not missing, f"Missing fields in summary for {t['id']}: {missing}"

    def test_summary_omits_nodes_and_edges(self):
        data = client.get("/api/v1/pipeline-templates").json()
        for t in data["templates"]:
            assert "nodes" not in t
            assert "edges" not in t

    def test_node_count_positive(self):
        data = client.get("/api/v1/pipeline-templates").json()
        for t in data["templates"]:
            assert t["node_count"] > 0

    def test_param_count_nonnegative(self):
        data = client.get("/api/v1/pipeline-templates").json()
        for t in data["templates"]:
            assert t["param_count"] >= 0


# ---------------------------------------------------------------------------
# Detail endpoint — GET /api/v1/pipeline-templates/{id}
# ---------------------------------------------------------------------------


class TestDetailEndpoint:
    @pytest.mark.parametrize("tid", sorted(EXPECTED_IDS))
    def test_returns_200(self, tid: str):
        assert client.get(f"/api/v1/pipeline-templates/{tid}").status_code == 200

    def test_no_auth_required(self):
        assert (
            client.get("/api/v1/pipeline-templates/hr-people-graph").status_code == 200
        )

    @pytest.mark.parametrize("tid", sorted(EXPECTED_IDS))
    def test_full_template_has_nodes_edges_params(self, tid: str):
        data = client.get(f"/api/v1/pipeline-templates/{tid}").json()
        assert len(data["nodes"]) > 0
        assert len(data["edges"]) > 0
        assert len(data["params"]) > 0

    @pytest.mark.parametrize("tid", sorted(EXPECTED_IDS))
    def test_id_matches_requested(self, tid: str):
        data = client.get(f"/api/v1/pipeline-templates/{tid}").json()
        assert data["id"] == tid

    def test_unknown_id_returns_404(self):
        assert (
            client.get("/api/v1/pipeline-templates/nonexistent-xyz").status_code == 404
        )

    def test_404_no_internal_path_leak(self):
        body = client.get("/api/v1/pipeline-templates/nonexistent-xyz").json()
        assert "nonexistent-xyz" in body.get("detail", "")
        assert "Traceback" not in str(body)
        assert "/Users/" not in str(body)

    @pytest.mark.parametrize("tid", sorted(EXPECTED_IDS))
    def test_create_graph_node_uses_graph_name_placeholder(self, tid: str):
        data = client.get(f"/api/v1/pipeline-templates/{tid}").json()
        node = next((n for n in data["nodes"] if n["type"] == "create-graph"), None)
        assert node is not None
        assert node["data"]["config"]["name"] == "{{graph_name}}"

    def test_legal_template_reviewer_email_in_review_gate(self):
        data = client.get("/api/v1/pipeline-templates/legal-document-graph").json()
        gate = next((n for n in data["nodes"] if n["type"] == "review-gate"), None)
        assert gate is not None
        assert "{{reviewer_email}}" in gate["data"]["config"]["notify_email"]

    def test_codebase_template_repo_url_in_ingest_node(self):
        data = client.get(
            "/api/v1/pipeline-templates/codebase-architecture-graph"
        ).json()
        node = next((n for n in data["nodes"] if n["type"] == "ingest-url"), None)
        assert node is not None
        assert "{{repo_url}}" in node["data"]["config"]["url"]

    def test_edge_types_valid(self):
        valid = {"data-flow", "trigger"}
        for tid in EXPECTED_IDS:
            data = client.get(f"/api/v1/pipeline-templates/{tid}").json()
            for edge in data["edges"]:
                assert edge.get("type") in valid, f"Invalid edge type in {tid}: {edge}"


# ---------------------------------------------------------------------------
# Multi-tenant isolation — static templates must not leak tenant data
# ---------------------------------------------------------------------------


class TestMultiTenantIsolation:
    def test_no_uuids_in_catalogue(self):
        data = client.get("/api/v1/pipeline-templates").json()
        uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
        assert not re.search(uuid_pattern, str(data), re.I), (
            "UUID found in catalogue — possible tenant leak"
        )

    def test_no_real_email_addresses_in_templates(self):
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        for tid in EXPECTED_IDS:
            data = client.get(f"/api/v1/pipeline-templates/{tid}").json()
            emails = re.findall(email_pattern, str(data))
            assert not emails, f"Email address found in {tid}: {emails}"
