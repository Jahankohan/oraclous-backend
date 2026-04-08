"""
Unit tests for EvaluationService.

All external dependencies (ChatService, RAGAS, OpenAI) are mocked.
Tests cover: metric selection, ground_truth handling, RAGAS result mapping,
RAGAS import failure, empty context warnings, and multi-tenancy usage.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.evaluation_schemas import EvaluationScores
from app.services.evaluation_service import (
    SUPPORTED_METRICS,
    EvaluationService,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grounded_result(
    answer="Alice is the CEO.",
    sources=None,
    is_grounded=True,
    confidence=0.9,
):
    result = MagicMock()
    result.answer = answer
    result.is_grounded = is_grounded
    result.confidence = confidence
    result.sources = sources or [
        {
            "node_id": "n1",
            "node_labels": ["Person"],
            "content": "Alice is CEO of Acme.",
            "relevance_score": 0.95,
        },
    ]
    result.retriever_result = None
    return result


def _make_ragas_dataframe(
    faithfulness=0.9, answer_relevancy=0.85, context_precision=0.8, context_recall=None
):
    """Build a minimal pandas-like mock returned by ragas evaluate()."""
    import pandas as pd

    row = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
    }
    if context_recall is not None:
        row["context_recall"] = context_recall
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Patch helper: suppresses ChatService and RAGAS internals
# ---------------------------------------------------------------------------


class _MockEvalCtx:
    """Context manager that patches ChatService + RAGAS for EvaluationService tests."""

    def __init__(self, grounded_result=None, ragas_df=None, ragas_import_error=False):
        self._grounded_result = grounded_result or _make_grounded_result()
        self._ragas_df = ragas_df or _make_ragas_dataframe()
        self._ragas_import_error = ragas_import_error

    def __enter__(self):
        # Patch ChatService
        self._p_chat = patch("app.services.evaluation_service.ChatService")
        mock_chat_cls = self._p_chat.start()
        mock_chat_inst = MagicMock()
        mock_chat_inst.initialize = AsyncMock()
        mock_chat_inst.search = AsyncMock(return_value=self._grounded_result)
        mock_chat_cls.return_value = mock_chat_inst
        self.mock_chat_inst = mock_chat_inst

        # Patch settings
        self._p_cfg = patch("app.services.evaluation_service.settings")
        mock_cfg = self._p_cfg.start()
        mock_cfg.OPENAI_API_KEY = "test-key"

        # Patch _run_ragas to avoid real RAGAS/OpenAI calls
        ragas_df = self._ragas_df

        def _fake_run_ragas(
            self_inner, question, answer, contexts, ground_truth, metric_names
        ):
            if self._ragas_import_error:
                return {name: None for name in SUPPORTED_METRICS}
            row = ragas_df.iloc[0].to_dict()
            key_map = {
                "faithfulness": "faithfulness",
                "answer_relevancy": "answer_relevance",
                "context_precision": "context_precision",
                "context_recall": "context_recall",
            }
            scores = {name: None for name in SUPPORTED_METRICS}
            for ragas_key, our_key in key_map.items():
                if ragas_key in row:
                    scores[our_key] = round(float(row[ragas_key]), 4)
            for name in list(scores.keys()):
                if name not in metric_names:
                    scores[name] = None
            return scores

        self._p_run = patch.object(EvaluationService, "_run_ragas", _fake_run_ragas)
        self._p_run.start()

        return self

    def __exit__(self, *_):
        self._p_chat.stop()
        self._p_cfg.stop()
        self._p_run.stop()


# ---------------------------------------------------------------------------
# Tests: basic evaluation flow
# ---------------------------------------------------------------------------


class TestEvaluationServiceBasic:

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_evaluate_returns_all_fields(self):
        with _MockEvalCtx():
            svc = EvaluationService(graph_id="g1", user_id="u1")
            result = await svc.evaluate(
                question="Who leads Acme?",
                answer=None,
                ground_truth=None,
                metrics=["faithfulness", "answer_relevance", "context_precision"],
            )

        assert result["answer"] == "Alice is the CEO."
        assert result["is_grounded"] is True
        assert isinstance(result["scores"], EvaluationScores)
        assert result["scores"].faithfulness == 0.9
        assert result["scores"].answer_relevance == 0.85
        assert result["scores"].context_precision == 0.8
        assert result["scores"].context_recall is None
        assert isinstance(result["overall"], float)
        assert result["metrics_computed"] == [
            "answer_relevance",
            "context_precision",
            "faithfulness",
        ]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_caller_supplied_answer_is_used(self):
        """When caller provides an answer, the generated answer should NOT be used."""
        with _MockEvalCtx():
            svc = EvaluationService(graph_id="g1", user_id="u1")
            result = await svc.evaluate(
                question="Who leads Acme?",
                answer="Bob is the CEO.",
                ground_truth=None,
                metrics=["faithfulness"],
            )

        assert result["answer"] == "Bob is the CEO."

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_context_recall_included_with_ground_truth(self):
        df = _make_ragas_dataframe(context_recall=0.75)
        with _MockEvalCtx(ragas_df=df):
            svc = EvaluationService(graph_id="g1", user_id="u1")
            result = await svc.evaluate(
                question="Who leads Acme?",
                answer=None,
                ground_truth="Alice is the CEO of Acme Corp.",
                metrics=None,  # all metrics
            )

        assert result["scores"].context_recall == 0.75
        assert "context_recall" in result["metrics_computed"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_context_recall_skipped_without_ground_truth(self):
        with _MockEvalCtx():
            svc = EvaluationService(graph_id="g1", user_id="u1")
            result = await svc.evaluate(
                question="Who leads Acme?",
                answer=None,
                ground_truth=None,
                metrics=None,  # all metrics requested
            )

        assert result["scores"].context_recall is None
        assert "context_recall" not in result["metrics_computed"]
        assert any("context_recall skipped" in w for w in result["warnings"])

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_unknown_metrics_produce_warning(self):
        with _MockEvalCtx():
            svc = EvaluationService(graph_id="g1", user_id="u1")
            result = await svc.evaluate(
                question="Test",
                answer=None,
                ground_truth=None,
                metrics=["faithfulness", "nonexistent_metric"],
            )

        assert any("Unknown metrics ignored" in w for w in result["warnings"])
        assert "faithfulness" in result["metrics_computed"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_all_unknown_metrics_raises_value_error(self):
        with _MockEvalCtx():
            svc = EvaluationService(graph_id="g1", user_id="u1")
            with pytest.raises(ValueError, match="No valid metrics"):
                await svc.evaluate(
                    question="Test",
                    answer=None,
                    ground_truth=None,
                    metrics=["totally_fake"],
                )


# ---------------------------------------------------------------------------
# Tests: context handling
# ---------------------------------------------------------------------------


class TestEvaluationServiceContext:

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_retrieved_contexts_mapped_from_sources(self):
        sources = [
            {
                "node_id": "n1",
                "node_labels": ["Person"],
                "content": "Alice is CEO.",
                "relevance_score": 0.95,
            },
            {
                "node_id": "n2",
                "node_labels": ["Company"],
                "content": "Acme Corp HQ is NYC.",
                "relevance_score": 0.88,
            },
        ]
        grounded = _make_grounded_result(sources=sources)
        with _MockEvalCtx(grounded_result=grounded):
            svc = EvaluationService(graph_id="g1", user_id="u1")
            result = await svc.evaluate("Who leads Acme?", None, None, ["faithfulness"])

        items = result["retrieved_contexts"]
        assert len(items) == 2
        assert items[0].node_id == "n1"
        assert items[1].content == "Acme Corp HQ is NYC."

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_empty_context_produces_warning(self):
        grounded = _make_grounded_result(sources=[])
        with _MockEvalCtx(grounded_result=grounded):
            svc = EvaluationService(graph_id="g1", user_id="u1")
            result = await svc.evaluate("Who leads Acme?", None, None, ["faithfulness"])

        assert any("No context retrieved" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# Tests: RAGAS failure handling
# ---------------------------------------------------------------------------


class TestEvaluationServiceRagasFailure:

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_ragas_import_error_returns_none_scores(self):
        with _MockEvalCtx(ragas_import_error=True):
            svc = EvaluationService(graph_id="g1", user_id="u1")
            result = await svc.evaluate(
                question="Test?",
                answer=None,
                ground_truth=None,
                metrics=["faithfulness"],
            )

        assert result["scores"].faithfulness is None
        assert result["overall"] is None


# ---------------------------------------------------------------------------
# Tests: overall score calculation
# ---------------------------------------------------------------------------


class TestEvaluationServiceOverall:

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_overall_is_mean_of_computed_scores(self):
        df = _make_ragas_dataframe(
            faithfulness=0.8, answer_relevancy=0.6, context_precision=0.7
        )
        with _MockEvalCtx(ragas_df=df):
            svc = EvaluationService(graph_id="g1", user_id="u1")
            result = await svc.evaluate(
                question="Test?",
                answer=None,
                ground_truth=None,
                metrics=["faithfulness", "answer_relevance", "context_precision"],
            )

        expected = round((0.8 + 0.6 + 0.7) / 3, 4)
        assert result["overall"] == expected
