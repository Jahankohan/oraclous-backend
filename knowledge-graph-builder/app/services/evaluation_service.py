"""
Evaluation Service — RAGAS-based quality scoring for KG-grounded chat.

Computes faithfulness, answer relevance, context precision, and (when
ground_truth is provided) context recall for any question/answer pair
against a specific knowledge graph.

RAGAS metrics require an LLM judge; the same OpenAI model used by
ChatService is reused here to keep configuration consistent.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.evaluation_schemas import (
    EvaluationScores,
    RetrievedContextItem,
)
from app.services.chat_service import ChatService
from app.services.retriever_factory import RetrieverType

logger = get_logger(__name__)

# Metrics that require ground_truth to be present.
_RECALL_METRICS = {"context_recall"}

# All supported metric names.
SUPPORTED_METRICS = frozenset(
    {"faithfulness", "answer_relevance", "context_precision", "context_recall"}
)


def _build_ragas_llm():
    """Return a LangchainLLMWrapper around the project's default OpenAI model."""
    # Import here so missing ragas/langchain doesn't break server startup.
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(
        ChatOpenAI(
            model="gpt-4o",
            api_key=settings.OPENAI_API_KEY,
            temperature=0.0,
        )
    )


def _build_ragas_embeddings():
    """Return a LangchainEmbeddingsWrapper for the project's embedding model."""
    from langchain_openai import OpenAIEmbeddings as LCOpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(
        LCOpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=settings.OPENAI_API_KEY,
        )
    )


def _build_metric_instances(
    metric_names: list[str], ragas_llm: Any, ragas_emb: Any
) -> list[Any]:
    """Instantiate and configure RAGAS metric objects."""
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    mapping = {
        "faithfulness": Faithfulness,
        "answer_relevance": AnswerRelevancy,
        "context_precision": ContextPrecision,
        "context_recall": ContextRecall,
    }

    instances = []
    for name in metric_names:
        cls = mapping[name]
        metric = cls()
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_emb
        instances.append(metric)
    return instances


class EvaluationService:
    """
    Runs RAGAS evaluation for a single question against a knowledge graph.

    Usage:
        svc = EvaluationService(graph_id="...", user_id="...")
        result = await svc.evaluate(
            question="Who leads Acme?",
            answer=None,          # auto-generate
            ground_truth="Alice", # optional
            metrics=["faithfulness", "answer_relevance"],
        )
    """

    def __init__(self, graph_id: str, user_id: str):
        self.graph_id = graph_id
        self.user_id = user_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        question: str,
        answer: str | None,
        ground_truth: str | None,
        metrics: list[str] | None,
    ) -> dict[str, Any]:
        """
        Run RAGAS evaluation and return a dict with:
          answer, retrieved_contexts, scores, overall,
          metrics_computed, is_grounded, warnings
        """
        warnings: list[str] = []

        # Resolve which metrics to compute.
        requested = set(metrics) if metrics else set(SUPPORTED_METRICS)
        unknown = requested - SUPPORTED_METRICS
        if unknown:
            warnings.append(f"Unknown metrics ignored: {sorted(unknown)}")
            requested -= unknown

        # Drop context_recall when ground_truth is absent.
        if not ground_truth and "context_recall" in requested:
            warnings.append("context_recall skipped: ground_truth not provided.")
            requested.discard("context_recall")

        if not requested:
            raise ValueError("No valid metrics to compute.")

        # Retrieve context + answer via ChatService.
        chat_svc = ChatService(
            graph_id=self.graph_id,
            retriever_type=RetrieverType.VECTOR_CYPHER,
        )
        await chat_svc.initialize()

        result = await chat_svc.search(
            query_text=question,
            retriever_config={"top_k": 5},
            return_context=True,
        )

        # Use caller-supplied answer if provided, otherwise use generated one.
        evaluated_answer = answer if answer is not None else result.answer

        # Build context strings from retrieved sources.
        context_strings: list[str] = []
        context_items: list[RetrievedContextItem] = []
        for src in result.sources:
            text = src.get("content") or ""
            if text:
                context_strings.append(text)
            context_items.append(
                RetrievedContextItem(
                    node_id=src.get("node_id"),
                    node_labels=src.get("node_labels"),
                    content=text,
                    relevance_score=src.get("relevance_score"),
                )
            )

        if not context_strings:
            warnings.append(
                "No context retrieved from the graph; RAGAS scores may be unreliable."
            )
            context_strings = ["No relevant context found in the knowledge graph."]

        # Run RAGAS synchronously inside the event loop via executor.
        scores_dict = await asyncio.get_event_loop().run_in_executor(
            None,
            self._run_ragas,
            question,
            evaluated_answer,
            context_strings,
            ground_truth,
            sorted(requested),
        )

        # Compute overall as mean of non-None scores.
        computed_values = [v for v in scores_dict.values() if v is not None]
        overall = (
            round(sum(computed_values) / len(computed_values), 4)
            if computed_values
            else None
        )

        return {
            "answer": evaluated_answer,
            "retrieved_contexts": context_items,
            "scores": EvaluationScores(**scores_dict),
            "overall": overall,
            "metrics_computed": sorted(requested),
            "is_grounded": result.is_grounded,
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_ragas(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None,
        metric_names: list[str],
    ) -> dict[str, float | None]:
        """
        Synchronous RAGAS execution (runs in thread pool executor).
        Returns a dict of {metric_name: score_or_None}.
        """
        try:
            from ragas import EvaluationDataset, SingleTurnSample, evaluate
        except ImportError:
            logger.error("ragas package not installed. Run: pip install ragas")
            return {name: None for name in metric_names}

        ragas_llm = _build_ragas_llm()
        ragas_emb = _build_ragas_embeddings()
        metric_instances = _build_metric_instances(metric_names, ragas_llm, ragas_emb)

        sample_kwargs: dict[str, Any] = {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
        }
        if ground_truth is not None:
            sample_kwargs["reference"] = ground_truth

        sample = SingleTurnSample(**sample_kwargs)
        dataset = EvaluationDataset(samples=[sample])

        try:
            ragas_result = evaluate(dataset=dataset, metrics=metric_instances)
        except Exception as exc:
            logger.error(f"RAGAS evaluation failed: {exc}")
            return {name: None for name in metric_names}

        # Map RAGAS result keys back to our canonical metric names.
        ragas_key_map = {
            "faithfulness": "faithfulness",
            "answer_relevancy": "answer_relevance",
            "context_precision": "context_precision",
            "context_recall": "context_recall",
        }
        scores: dict[str, float | None] = {name: None for name in SUPPORTED_METRICS}

        result_dict = (
            ragas_result.to_pandas().iloc[0].to_dict()
            if hasattr(ragas_result, "to_pandas")
            else {}
        )
        for ragas_key, our_key in ragas_key_map.items():
            if ragas_key in result_dict:
                raw = result_dict[ragas_key]
                scores[our_key] = round(float(raw), 4) if raw is not None else None

        # Zero-out metrics not requested.
        for name in list(scores.keys()):
            if name not in metric_names:
                scores[name] = None

        return scores
