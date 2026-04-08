"""
Evaluation API Schemas

Pydantic request/response models for the RAGAS-based evaluation endpoint.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class EvaluationRequest(BaseModel):
    """Request body for POST /graphs/{graph_id}/evaluate"""

    question: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The question to evaluate against the knowledge graph",
    )
    answer: Optional[str] = Field(
        default=None,
        description=(
            "The answer to evaluate. If omitted, the system generates one "
            "using the default chat retriever."
        ),
    )
    ground_truth: Optional[str] = Field(
        default=None,
        description=(
            "Reference answer for context_recall scoring. "
            "When omitted, context_recall is skipped."
        ),
    )
    metrics: Optional[List[str]] = Field(
        default=None,
        description=(
            "Subset of metrics to compute. "
            "Allowed: faithfulness, answer_relevance, context_precision, context_recall. "
            "Defaults to all metrics applicable given the provided inputs."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "Who is the CEO of Acme Corp?",
                "ground_truth": "John Smith is the CEO of Acme Corp.",
                "metrics": ["faithfulness", "answer_relevance", "context_precision"],
            }
        }
    }


class EvaluationScores(BaseModel):
    """RAGAS metric scores (None when a metric was not computed)."""

    faithfulness: Optional[float] = Field(
        default=None,
        description="Fraction of answer claims supported by retrieved context (0–1)",
    )
    answer_relevance: Optional[float] = Field(
        default=None,
        description="How well the answer addresses the question (0–1)",
    )
    context_precision: Optional[float] = Field(
        default=None,
        description="Fraction of retrieved chunks that are relevant (0–1)",
    )
    context_recall: Optional[float] = Field(
        default=None,
        description=(
            "Fraction of ground-truth information captured by retrieved context (0–1). "
            "Requires ground_truth."
        ),
    )


class RetrievedContextItem(BaseModel):
    """A single retrieved context chunk used during evaluation."""

    node_id: Optional[str] = None
    node_labels: Optional[List[str]] = None
    content: str
    relevance_score: Optional[float] = None


class EvaluationResponse(BaseModel):
    """Response body for POST /graphs/{graph_id}/evaluate"""

    graph_id: str
    question: str
    answer: str = Field(description="The answer that was evaluated")
    retrieved_contexts: List[RetrievedContextItem] = Field(
        default_factory=list,
        description="Graph nodes/relationships retrieved during evaluation",
    )
    scores: EvaluationScores
    overall: Optional[float] = Field(
        default=None,
        description="Mean of all computed scores",
    )
    metrics_computed: List[str] = Field(
        default_factory=list,
        description="Names of the metrics that were actually computed",
    )
    is_grounded: bool = Field(
        description="Whether the answer is grounded in the knowledge graph",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal issues encountered during evaluation",
    )
