"""
Integration tests for POST /api/v1/api/v1/graphs/{graph_id}/evaluate.

Tests the full request → auth → ownership check → EvaluationService → response
pipeline. EvaluationService internals (ChatService, RAGAS) are mocked.

URL: /api/v1/api/v1/graphs/{graph_id}/evaluate
     ^^^^^^^^ main app prefix
              ^^^^^^^^ router prefix in api_router
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.evaluation_schemas import EvaluationScores, RetrievedContextItem

GRAPH_ID = "test-graph-eval-001"
FAKE_USER_ID = "eval-user-42"
BASE_URL = f"/api/v1/api/v1/graphs/{GRAPH_ID}/evaluate"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eval_result(
    answer="Alice is the CEO.",
    faithfulness=0.92,
    answer_relevance=0.87,
    context_precision=0.80,
    context_recall=None,
    is_grounded=True,
    warnings=None,
):
    """Build the dict that EvaluationService.evaluate() returns."""
    scores = EvaluationScores(
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        context_precision=context_precision,
        context_recall=context_recall,
    )
    computed = ["answer_relevance", "context_precision", "faithfulness"]
    if context_recall is not None:
        computed.append("context_recall")

    non_none = [
        v
        for v in [faithfulness, answer_relevance, context_precision, context_recall]
        if v is not None
    ]
    overall = round(sum(non_none) / len(non_none), 4) if non_none else None

    return {
        "answer": answer,
        "retrieved_contexts": [
            RetrievedContextItem(
                node_id="n1",
                node_labels=["Person"],
                content="Alice is CEO of Acme Corp.",
                relevance_score=0.95,
            )
        ],
        "scores": scores,
        "overall": overall,
        "metrics_computed": sorted(computed),
        "is_grounded": is_grounded,
        "warnings": warnings or [],
    }


class _AuthAndEvalPatch:
    """Context manager that patches auth, graph ownership, and EvaluationService."""

    def __init__(
        self, eval_result=None, user_id=FAKE_USER_ID, graph_found=True, eval_raises=None
    ):
        self._eval_result = eval_result or _make_eval_result()
        self._user_id = user_id
        self._graph_found = graph_found
        self._eval_raises = eval_raises

    def __enter__(self):
        self._p_auth = patch("app.api.v1.endpoints.evaluation.auth_service")
        self._p_gs = patch("app.api.v1.endpoints.evaluation.GraphNodeService")
        self._p_svc = patch("app.api.v1.endpoints.evaluation.EvaluationService")

        mock_auth = self._p_auth.start()
        mock_auth.verify_token = AsyncMock(return_value={"id": self._user_id})

        mock_gs_cls = self._p_gs.start()
        if self._graph_found:
            mock_gs_cls.return_value.get_graph.return_value = {"user_id": self._user_id}
        else:
            mock_gs_cls.return_value.get_graph.return_value = None

        mock_svc_cls = self._p_svc.start()
        mock_svc_inst = MagicMock()
        if self._eval_raises:
            mock_svc_inst.evaluate = AsyncMock(side_effect=self._eval_raises)
        else:
            mock_svc_inst.evaluate = AsyncMock(return_value=self._eval_result)
        mock_svc_cls.return_value = mock_svc_inst
        self.mock_svc_inst = mock_svc_inst

        return self

    def __exit__(self, *_):
        self._p_auth.stop()
        self._p_gs.stop()
        self._p_svc.stop()


# ---------------------------------------------------------------------------
# Tests: happy path
# ---------------------------------------------------------------------------


class TestEvaluationEndpointHappyPath:
    @pytest.mark.integration
    @pytest.mark.api
    async def test_evaluate_returns_200_with_scores(self, async_client):
        with _AuthAndEvalPatch():
            response = await async_client.post(
                BASE_URL,
                json={"question": "Who is the CEO of Acme Corp?"},
                headers={"Authorization": "Bearer fake-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["graph_id"] == GRAPH_ID
        assert data["question"] == "Who is the CEO of Acme Corp?"
        assert data["answer"] == "Alice is the CEO."
        assert "scores" in data
        assert data["scores"]["faithfulness"] == 0.92
        assert data["scores"]["answer_relevance"] == 0.87
        assert data["scores"]["context_precision"] == 0.80
        assert data["scores"]["context_recall"] is None
        assert data["is_grounded"] is True
        assert isinstance(data["overall"], float)

    @pytest.mark.integration
    @pytest.mark.api
    async def test_evaluate_with_ground_truth_includes_recall(self, async_client):
        result = _make_eval_result(context_recall=0.75)
        with _AuthAndEvalPatch(eval_result=result) as ctx:
            response = await async_client.post(
                BASE_URL,
                json={
                    "question": "Who is the CEO?",
                    "ground_truth": "Alice is the CEO of Acme Corp.",
                    "metrics": ["faithfulness", "context_recall"],
                },
                headers={"Authorization": "Bearer fake-token"},
            )
            # Verify service received ground_truth
            call_kwargs = ctx.mock_svc_inst.evaluate.call_args.kwargs
            assert call_kwargs.get("ground_truth") == "Alice is the CEO of Acme Corp."

        assert response.status_code == 200
        assert response.json()["scores"]["context_recall"] == 0.75

    @pytest.mark.integration
    @pytest.mark.api
    async def test_evaluate_with_explicit_answer(self, async_client):
        with _AuthAndEvalPatch() as ctx:
            response = await async_client.post(
                BASE_URL,
                json={"question": "Who is the CEO?", "answer": "Bob is the CEO."},
                headers={"Authorization": "Bearer fake-token"},
            )
            call_kwargs = ctx.mock_svc_inst.evaluate.call_args.kwargs
            assert call_kwargs.get("answer") == "Bob is the CEO."

        assert response.status_code == 200

    @pytest.mark.integration
    @pytest.mark.api
    async def test_evaluate_response_includes_retrieved_contexts(self, async_client):
        with _AuthAndEvalPatch():
            response = await async_client.post(
                BASE_URL,
                json={"question": "Who is the CEO?"},
                headers={"Authorization": "Bearer fake-token"},
            )

        data = response.json()
        assert len(data["retrieved_contexts"]) == 1
        ctx = data["retrieved_contexts"][0]
        assert ctx["node_id"] == "n1"
        assert "Alice" in ctx["content"]
        assert ctx["relevance_score"] == 0.95

    @pytest.mark.integration
    @pytest.mark.api
    async def test_evaluate_warnings_propagated(self, async_client):
        result = _make_eval_result(
            warnings=["context_recall skipped: ground_truth not provided."]
        )
        with _AuthAndEvalPatch(eval_result=result):
            response = await async_client.post(
                BASE_URL,
                json={"question": "Who?"},
                headers={"Authorization": "Bearer fake-token"},
            )

        assert response.status_code == 200
        assert any("context_recall skipped" in w for w in response.json()["warnings"])

    @pytest.mark.integration
    @pytest.mark.api
    async def test_metrics_computed_field_returned(self, async_client):
        with _AuthAndEvalPatch():
            response = await async_client.post(
                BASE_URL,
                json={"question": "Who?", "metrics": ["faithfulness"]},
                headers={"Authorization": "Bearer fake-token"},
            )

        data = response.json()
        assert "metrics_computed" in data
        assert isinstance(data["metrics_computed"], list)


# ---------------------------------------------------------------------------
# Tests: authentication and authorization
# ---------------------------------------------------------------------------


class TestEvaluationEndpointAuth:
    @pytest.mark.integration
    @pytest.mark.api
    async def test_missing_auth_header_returns_403(self, async_client):
        response = await async_client.post(
            BASE_URL,
            json={"question": "Who?"},
        )
        assert response.status_code == 403

    @pytest.mark.integration
    @pytest.mark.api
    async def test_graph_not_found_returns_403(self, async_client):
        with _AuthAndEvalPatch(graph_found=False):
            response = await async_client.post(
                BASE_URL,
                json={"question": "Who?"},
                headers={"Authorization": "Bearer fake-token"},
            )

        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]

    @pytest.mark.integration
    @pytest.mark.api
    async def test_cross_tenant_access_denied(self, async_client):
        """A user cannot evaluate another user's graph."""
        with (
            patch("app.api.v1.endpoints.evaluation.auth_service") as mock_auth,
            patch("app.api.v1.endpoints.evaluation.GraphNodeService") as mock_gs_cls,
        ):
            mock_auth.verify_token = AsyncMock(return_value={"id": "user-A"})
            # Graph belongs to user-B
            mock_gs_cls.return_value.get_graph.return_value = {"user_id": "user-B"}

            response = await async_client.post(
                BASE_URL,
                json={"question": "Who?"},
                headers={"Authorization": "Bearer token-for-user-A"},
            )

        assert response.status_code == 403


# ---------------------------------------------------------------------------
# Tests: validation
# ---------------------------------------------------------------------------


class TestEvaluationEndpointValidation:
    @pytest.mark.integration
    @pytest.mark.api
    async def test_empty_question_returns_422(self, async_client):
        with _AuthAndEvalPatch():
            response = await async_client.post(
                BASE_URL,
                json={"question": ""},
                headers={"Authorization": "Bearer fake-token"},
            )

        assert response.status_code == 422

    @pytest.mark.integration
    @pytest.mark.api
    async def test_missing_question_returns_422(self, async_client):
        with _AuthAndEvalPatch():
            response = await async_client.post(
                BASE_URL,
                json={},
                headers={"Authorization": "Bearer fake-token"},
            )

        assert response.status_code == 422

    @pytest.mark.integration
    @pytest.mark.api
    async def test_service_value_error_returns_422(self, async_client):
        with _AuthAndEvalPatch(eval_raises=ValueError("No valid metrics to compute.")):
            response = await async_client.post(
                BASE_URL,
                json={"question": "Who?", "metrics": ["totally_fake"]},
                headers={"Authorization": "Bearer fake-token"},
            )

        assert response.status_code == 422
        assert "No valid metrics" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestEvaluationEndpointErrors:
    @pytest.mark.integration
    @pytest.mark.api
    async def test_service_runtime_error_returns_500(self, async_client):
        with _AuthAndEvalPatch(eval_raises=RuntimeError("RAGAS exploded")):
            response = await async_client.post(
                BASE_URL,
                json={"question": "Who?"},
                headers={"Authorization": "Bearer fake-token"},
            )

        assert response.status_code == 500
        assert "Evaluation failed" in response.json()["detail"]
