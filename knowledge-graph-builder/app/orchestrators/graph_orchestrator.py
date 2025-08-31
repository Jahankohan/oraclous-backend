from enum import Enum
from typing import Dict, Any, List, Optional, Union
from uuid import UUID
from dataclasses import dataclass
from datetime import datetime
import asyncio

from neo4j_graphrag.generation import GraphRAG
from app.components.multi_tenant_retriever import MultiTenantRetriever
from app.components.drift_search import DRIFTRetriever
from app.services.analytics_service import analytics_service
from app.services.chat_service import ChatService
from app.core.dependencies import get_llm, create_multi_tenant_retriever

class SearchMode(Enum):
    """Unified search modes combining all capabilities"""
    FAST = "fast"              # Neo4j GraphRAG only (200-500ms)
    BALANCED = "balanced"      # Neo4j + light analytics (1-2s)  
    COMPREHENSIVE = "comprehensive"  # Neo4j + full analytics (3-8s)
    DRIFT = "drift"            # Microsoft DRIFT methodology
    LEGACY = "legacy"          # Your existing chat modes

@dataclass
class SearchRequest:
    query: str
    graph_id: str
    mode: SearchMode = SearchMode.BALANCED
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    max_tokens: int = 4000
    include_reasoning: bool = True
    custom_filters: Dict[str, Any] = None

@dataclass
class UnifiedSearchResult:
    response: str
    mode_used: SearchMode
    processing_time: float
    graph_id: str
    sources: List[Dict[str, Any]]
    analytics: Optional[Dict[str, Any]] = None
    reasoning_chain: Optional[List[str]] = None
    performance_metrics: Dict[str, float] = None

class GraphOrchestrator:
    """
    Unified orchestration layer coordinating all search modes and services.
    
    DESIGN PRINCIPLES:
    - Single entry point for all graph operations
    - Mode-based performance guarantees
    - Service coordination without tight coupling
    - Preserves existing service strengths
    """
    
    def __init__(self):
        # Service dependencies (inject via FastAPI)
        self.chat_service = None  # Injected
        self.analytics_service = analytics_service
        
    async def search(self, request: SearchRequest) -> UnifiedSearchResult:
        """
        Main orchestration method - routes to appropriate search strategy
        """
        start_time = datetime.now()
        
        try:
            if request.mode == SearchMode.FAST:
                result = await self._fast_search(request)
            elif request.mode == SearchMode.BALANCED:
                result = await self._balanced_search(request)
            elif request.mode == SearchMode.COMPREHENSIVE:
                result = await self._comprehensive_search(request)
            elif request.mode == SearchMode.DRIFT:
                result = await self._drift_search(request)
            elif request.mode == SearchMode.LEGACY:
                result = await self._legacy_search(request)
            else:
                result = await self._balanced_search(request)  # Default
            
            # Add performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            result.performance_metrics = self._calculate_metrics(result, processing_time)
            
            return result
            
        except Exception as e:
            # Graceful fallback to legacy mode
            fallback_result = await self._legacy_search(request)
            fallback_result.mode_used = SearchMode.LEGACY
            fallback_result.performance_metrics = {"fallback": True, "error": str(e)}
            return fallback_result
    
    async def _fast_search(self, request: SearchRequest) -> UnifiedSearchResult:
        """
        Fast mode: Neo4j GraphRAG only (200-500ms target)
        """
        # Pure Neo4j GraphRAG - minimal overhead
        retriever = create_multi_tenant_retriever(request.graph_id)
        rag = GraphRAG(retriever=retriever, llm=await get_llm())
        
        result = await rag.search(request.query)
        
        return UnifiedSearchResult(
            response=result.get("answer", ""),
            mode_used=SearchMode.FAST,
            processing_time=0,  # Will be set in main method
            graph_id=request.graph_id,
            sources=result.get("sources", []),
            analytics=None  # No analytics in fast mode
        )
    
    async def _balanced_search(self, request: SearchRequest) -> UnifiedSearchResult:
        """
        Balanced mode: Neo4j GraphRAG + light analytics (1-2s target)
        """
        # Neo4j GraphRAG foundation
        fast_result = await self._fast_search(request)
        
        # Add light analytics (entity relationships only)
        entities = self._extract_entities_from_result(fast_result)
        if entities:
            light_analytics = await self.analytics_service.get_entity_relationships(
                entities=entities[:5],  # Limit for performance
                graph_id=UUID(request.graph_id)
            )
            fast_result.analytics = {"relationships": light_analytics}
        
        return fast_result
    
    async def _comprehensive_search(self, request: SearchRequest) -> UnifiedSearchResult:
        """
        Comprehensive mode: Neo4j GraphRAG + full analytics (3-8s target)
        """
        # Start Neo4j GraphRAG and analytics in parallel
        graphrag_task = asyncio.create_task(self._fast_search(request))
        
        # Extract entities for analytics (may need preliminary search)
        preliminary_result = await self._fast_search(request)
        entities = self._extract_entities_from_result(preliminary_result)
        
        # Run full analytics if we have entities
        analytics_task = None
        if entities:
            analytics_task = asyncio.create_task(
                self.analytics_service.comprehensive_graph_analysis(
                    entities=entities,
                    graph_id=UUID(request.graph_id)
                )
            )
        
        # Wait for both to complete
        graphrag_result = await graphrag_task
        analytics_result = await analytics_task if analytics_task else None
        
        # Synthesize results
        if analytics_result:
            enhanced_response = await self._synthesize_with_analytics(
                graphrag_response=graphrag_result.response,
                analytics=analytics_result,
                original_query=request.query
            )
            graphrag_result.response = enhanced_response
            graphrag_result.analytics = analytics_result
        
        graphrag_result.mode_used = SearchMode.COMPREHENSIVE
        return graphrag_result
    
    async def _drift_search(self, request: SearchRequest) -> UnifiedSearchResult:
        """
        DRIFT mode: Microsoft DRIFT methodology with Neo4j foundation
        """
        # Use DRIFT retriever component
        from app.components.drift_search import DRIFTRetriever
        
        base_retriever = create_multi_tenant_retriever(request.graph_id)
        drift_retriever = DRIFTRetriever(
            base_retriever=base_retriever,
            community_service=self.analytics_service,
            llm=await get_llm()
        )
        
        rag = GraphRAG(retriever=drift_retriever, llm=await get_llm())
        result = await rag.search(request.query)
        
        return UnifiedSearchResult(
            response=result.get("answer", ""),
            mode_used=SearchMode.DRIFT,
            processing_time=0,  # Will be set in main method
            graph_id=request.graph_id,
            sources=result.get("sources", []),
            analytics=result.get("community_context"),
            reasoning_chain=result.get("follow_up_questions")
        )
    
    async def _legacy_search(self, request: SearchRequest) -> UnifiedSearchResult:
        """
        Legacy mode: Use existing ChatService implementation
        """
        if not self.chat_service:
            # Initialize chat service if needed
            self.chat_service = ChatService()
            await self.chat_service.initialize_for_graph(
                graph_id=UUID(request.graph_id),
                user_id=request.user_id or "system"
            )
        
        # Use existing chat modes (preserve all functionality)
        legacy_result = await self.chat_service.chat_with_graph(
            query=request.query,
            mode="comprehensive",  # Your best existing mode
            graph_id=UUID(request.graph_id),
            conversation_id=request.conversation_id,
            include_history=True,
            max_context_tokens=request.max_tokens,
            include_reasoning_chain=request.include_reasoning
        )
        
        return UnifiedSearchResult(
            response=legacy_result.get("response", ""),
            mode_used=SearchMode.LEGACY,
            processing_time=0,
            graph_id=request.graph_id,
            sources=legacy_result.get("sources", []),
            analytics=legacy_result.get("graph_context"),
            reasoning_chain=legacy_result.get("reasoning_chain")
        )
    
    # Helper Methods
    
    def _extract_entities_from_result(self, result: UnifiedSearchResult) -> List[str]:
        """Extract entity names from search result for analytics"""
        entities = []
        for source in result.sources:
            if "entities" in source:
                entities.extend([e.get("name", "") for e in source["entities"]])
        return list(set(entities))[:10]  # Limit and dedupe
    
    async def _synthesize_with_analytics(
        self, 
        graphrag_response: str, 
        analytics: Dict[str, Any],
        original_query: str
    ) -> str:
        """Synthesize GraphRAG response with analytics insights"""
        
        synthesis_prompt = f"""
        Original query: "{original_query}"
        
        Base response: {graphrag_response}
        
        Additional graph analytics:
        - Communities: {analytics.get('communities', [])}
        - Influential entities: {analytics.get('influential_entities', [])}
        - Key relationships: {analytics.get('relationships', [])}
        
        Enhance the base response by incorporating relevant analytics insights.
        Keep the response natural and focused on answering the original query.
        Only add analytics insights that directly relate to the query.
        """
        
        llm = await get_llm()
        enhanced_response = await llm.ainvoke(synthesis_prompt)
        return enhanced_response.content
    
    def _calculate_metrics(
        self, 
        result: UnifiedSearchResult, 
        processing_time: float
    ) -> Dict[str, float]:
        """Calculate performance and quality metrics"""
        return {
            "processing_time_ms": processing_time * 1000,
            "sources_count": len(result.sources),
            "has_analytics": 1.0 if result.analytics else 0.0,
            "response_length": len(result.response),
            "mode_efficiency": self._get_mode_efficiency(result.mode_used)
        }
    
    def _get_mode_efficiency(self, mode: SearchMode) -> float:
        """Expected efficiency score for each mode"""
        efficiency_map = {
            SearchMode.FAST: 1.0,
            SearchMode.BALANCED: 0.8,
            SearchMode.COMPREHENSIVE: 0.6,
            SearchMode.DRIFT: 0.7,
            SearchMode.LEGACY: 0.5
        }
        return efficiency_map.get(mode, 0.5)
