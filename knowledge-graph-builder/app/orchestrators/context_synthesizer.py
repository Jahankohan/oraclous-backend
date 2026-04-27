import json
from dataclasses import dataclass
from typing import Any


@dataclass
class SynthesisContext:
    """Context for synthesizing results from multiple services"""

    graphrag_result: dict[str, Any]
    analytics_result: dict[str, Any] | None = None
    community_context: dict[str, Any] | None = None
    temporal_context: dict[str, Any] | None = None
    user_context: dict[str, Any] | None = None


class ContextSynthesizer:
    """
    Intelligent fusion of results from multiple services

    RESPONSIBILITIES:
    - Combine GraphRAG with analytics insights
    - Resolve conflicting information between services
    - Create coherent, comprehensive responses
    - Maintain source attribution and confidence scores
    """

    def __init__(self, llm_service):
        self.llm_service = llm_service

    async def synthesize_comprehensive_response(
        self, context: SynthesisContext, original_query: str
    ) -> dict[str, Any]:
        """
        Main synthesis method combining all available context
        """

        # Build synthesis prompt with all available context
        synthesis_prompt = self._build_synthesis_prompt(context, original_query)

        # Generate enhanced response
        enhanced_response = await self.llm_service.generate_response(synthesis_prompt)

        # Calculate confidence and source attribution
        confidence_score = self._calculate_confidence(context)
        source_attribution = self._build_source_attribution(context)

        return {
            "response": enhanced_response,
            "confidence_score": confidence_score,
            "source_attribution": source_attribution,
            "synthesis_metadata": {
                "graphrag_used": bool(context.graphrag_result),
                "analytics_used": bool(context.analytics_result),
                "community_context_used": bool(context.community_context),
                "temporal_context_used": bool(context.temporal_context),
            },
        }

    def _build_synthesis_prompt(self, context: SynthesisContext, query: str) -> str:
        """Build comprehensive synthesis prompt"""

        prompt_parts = [
            f"Original Query: {query}",
            "",
            "Base Information:",
            json.dumps(context.graphrag_result, indent=2),
        ]

        if context.analytics_result:
            prompt_parts.extend(
                ["", "Graph Analytics:", json.dumps(context.analytics_result, indent=2)]
            )

        if context.community_context:
            prompt_parts.extend(
                [
                    "",
                    "Community Context:",
                    json.dumps(context.community_context, indent=2),
                ]
            )

        if context.temporal_context:
            prompt_parts.extend(
                [
                    "",
                    "Temporal Context:",
                    json.dumps(context.temporal_context, indent=2),
                ]
            )

        prompt_parts.extend(
            [
                "",
                "Instructions:",
                "1. Synthesize all available information to provide a comprehensive answer",
                "2. Prioritize information that directly addresses the query",
                "3. Highlight insights from graph analytics and community context",
                "4. Maintain factual accuracy and cite sources",
                "5. If information conflicts, acknowledge the uncertainty",
                "6. Keep the response natural and conversational",
            ]
        )

        return "\n".join(prompt_parts)

    def _calculate_confidence(self, context: SynthesisContext) -> float:
        """Calculate overall confidence score based on available context"""
        base_confidence = 0.5

        # GraphRAG result adds confidence
        if context.graphrag_result:
            base_confidence += 0.2

        # Analytics add significant confidence
        if context.analytics_result:
            base_confidence += 0.2

        # Community context adds confidence
        if context.community_context:
            base_confidence += 0.1

        # Multiple sources increase confidence
        source_count = sum(
            [
                1
                for ctx in [
                    context.graphrag_result,
                    context.analytics_result,
                    context.community_context,
                    context.temporal_context,
                ]
                if ctx
            ]
        )

        if source_count > 2:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _build_source_attribution(self, context: SynthesisContext) -> dict[str, Any]:
        """Build detailed source attribution"""
        attribution = {
            "primary_sources": [],
            "analytics_sources": [],
            "community_sources": [],
            "confidence_by_source": {},
        }

        if context.graphrag_result and "sources" in context.graphrag_result:
            attribution["primary_sources"] = context.graphrag_result["sources"]
            attribution["confidence_by_source"]["graphrag"] = 0.8

        if context.analytics_result:
            attribution["analytics_sources"] = ["comprehensive_graph_analysis"]
            attribution["confidence_by_source"]["analytics"] = 0.9

        if context.community_context:
            attribution["community_sources"] = [
                "community_detection",
                "hierarchical_clustering",
            ]
            attribution["confidence_by_source"]["community"] = 0.7

        return attribution
