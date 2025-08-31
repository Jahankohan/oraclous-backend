from app.orchestrators.graph_orchestrator import GraphOrchestrator, SearchRequest, SearchMode
from app.services.chat_service import ChatService
from uuid import UUID
from typing import Dict, Any

class EnhancedChatService:
    """
    Integration layer that combines ChatService with GraphOrchestrator
    
    DESIGN: This doesn't replace ChatService - it enhances it with orchestration
    """
    
    def __init__(self):
        self.orchestrator = GraphOrchestrator()
        self.legacy_chat_service = ChatService()
    
    async def unified_chat(
        self,
        query: str,
        graph_id: UUID,
        mode: str = "auto",  # auto, fast, balanced, comprehensive, drift, legacy
        user_id: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Unified chat interface that can use any search mode
        """
        
        # Auto mode selection
        if mode == "auto":
            from app.orchestrators.performance_manager import PerformanceManager
            pm = PerformanceManager()
            mode = await pm.recommend_search_mode(query, kwargs.get("user_preferences"))
        
        # Map mode to SearchMode enum
        mode_mapping = {
            "fast": SearchMode.FAST,
            "balanced": SearchMode.BALANCED,
            "comprehensive": SearchMode.COMPREHENSIVE,
            "drift": SearchMode.DRIFT,
            "legacy": SearchMode.LEGACY
        }
        
        search_mode = mode_mapping.get(mode, SearchMode.BALANCED)
        
        # Create search request
        request = SearchRequest(
            query=query,
            graph_id=str(graph_id),
            mode=search_mode,
            user_id=user_id,
            conversation_id=kwargs.get("conversation_id"),
            max_tokens=kwargs.get("max_tokens", 4000),
            include_reasoning=kwargs.get("include_reasoning", True)
        )
        
        # Execute unified search
        result = await self.orchestrator.search(request)
        
        # Convert to chat response format
        return {
            "response": result.response,
            "mode": result.mode_used.value,
            "processing_time": result.processing_time,
            "sources": result.sources,
            "analytics": result.analytics,
            "reasoning_chain": result.reasoning_chain,
            "performance_metrics": result.performance_metrics,
            "graph_id": str(graph_id)
        }
    
    # Preserve all existing ChatService methods
    async def chat_with_graph(self, *args, **kwargs):
        """Delegate to original ChatService for backward compatibility"""
        return await self.legacy_chat_service.chat_with_graph(*args, **kwargs)
    
    async def initialize_for_graph(self, *args, **kwargs):
        """Delegate to original ChatService"""
        return await self.legacy_chat_service.initialize_for_graph(*args, **kwargs)
