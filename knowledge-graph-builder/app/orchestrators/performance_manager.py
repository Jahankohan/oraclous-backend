from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio

class QueryComplexity(Enum):
    SIMPLE = "simple"        # Single entity, direct question
    MODERATE = "moderate"    # Multiple entities, relationships
    COMPLEX = "complex"      # Analytics, reasoning, synthesis

@dataclass
class PerformanceProfile:
    """Performance characteristics for different query types"""
    avg_response_time: float = 0.0
    success_rate: float = 1.0
    user_satisfaction: float = 0.8
    resource_usage: float = 0.5
    last_updated: datetime = field(default_factory=datetime.now)

class PerformanceManager:
    """
    Intelligent query routing and performance optimization
    
    RESPONSIBILITIES:
    - Analyze query complexity and route to optimal mode
    - Track performance metrics across different modes
    - Adaptive routing based on current system load
    - SLA enforcement and fallback strategies
    """
    
    def __init__(self):
        self.performance_profiles: Dict[str, PerformanceProfile] = {
            "fast": PerformanceProfile(avg_response_time=0.5, resource_usage=0.2),
            "balanced": PerformanceProfile(avg_response_time=1.5, resource_usage=0.5),
            "comprehensive": PerformanceProfile(avg_response_time=5.0, resource_usage=0.8),
            "drift": PerformanceProfile(avg_response_time=3.0, resource_usage=0.6),
            "legacy": PerformanceProfile(avg_response_time=2.0, resource_usage=0.4)
        }
        
        self.current_load = 0.0
        self.active_queries = 0
        self.max_concurrent_queries = 10
    
    async def recommend_search_mode(
        self,
        query: str,
        user_preferences: Optional[Dict[str, Any]] = None,
        sla_requirements: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Recommend optimal search mode based on query analysis and system state
        """
        
        # Analyze query complexity
        complexity = await self._analyze_query_complexity(query)
        
        # Check system load
        current_load = await self._get_current_system_load()
        
        # Apply user preferences
        preferred_balance = self._get_user_balance_preference(user_preferences)
        
        # Check SLA requirements
        max_response_time = sla_requirements.get("max_response_time", 10.0) if sla_requirements else 10.0
        
        # Decision logic
        if max_response_time < 1.0:
            return "fast"
        elif complexity == QueryComplexity.SIMPLE and current_load < 0.5:
            return "balanced"
        elif complexity == QueryComplexity.COMPLEX and max_response_time > 5.0:
            return "comprehensive"
        elif preferred_balance == "depth":
            return "comprehensive"
        elif preferred_balance == "speed":
            return "fast"
        else:
            return "balanced"  # Default safe choice
    
    async def _analyze_query_complexity(self, query: str) -> QueryComplexity:
        """Analyze query to determine complexity level"""
        
        # Simple heuristics (could be enhanced with ML)
        query_lower = query.lower()
        
        # Complex indicators
        complex_keywords = [
            "analyze", "compare", "relationship", "community", "influence",
            "pattern", "trend", "comprehensive", "deep dive", "correlation"
        ]
        
        # Simple indicators
        simple_keywords = [
            "what is", "who is", "when did", "where is", "define"
        ]
        
        if any(keyword in query_lower for keyword in complex_keywords):
            return QueryComplexity.COMPLEX
        elif any(keyword in query_lower for keyword in simple_keywords):
            return QueryComplexity.SIMPLE
        else:
            return QueryComplexity.MODERATE
    
    async def _get_current_system_load(self) -> float:
        """Get current system load (simplified implementation)"""
        # In production, this would check actual system metrics
        return min(self.active_queries / self.max_concurrent_queries, 1.0)
    
    def _get_user_balance_preference(self, user_preferences: Optional[Dict]) -> str:
        """Extract user preference for speed vs depth balance"""
        if not user_preferences:
            return "balanced"
        
        return user_preferences.get("search_preference", "balanced")
    
    async def track_query_performance(
        self,
        mode: str,
        response_time: float,
        success: bool,
        user_satisfaction: Optional[float] = None
    ):
        """Track performance metrics for continuous improvement"""
        
        if mode in self.performance_profiles:
            profile = self.performance_profiles[mode]
            
            # Update moving average
            alpha = 0.1  # Learning rate
            profile.avg_response_time = (
                (1 - alpha) * profile.avg_response_time + 
                alpha * response_time
            )
            
            # Update success rate
            profile.success_rate = (
                (1 - alpha) * profile.success_rate + 
                alpha * (1.0 if success else 0.0)
            )
            
            # Update user satisfaction if provided
            if user_satisfaction is not None:
                profile.user_satisfaction = (
                    (1 - alpha) * profile.user_satisfaction + 
                    alpha * user_satisfaction
                )
            
            profile.last_updated = datetime.now()
    
    def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get current performance metrics for all modes"""
        return {
            mode: {
                "avg_response_time": profile.avg_response_time,
                "success_rate": profile.success_rate,
                "user_satisfaction": profile.user_satisfaction,
                "resource_usage": profile.resource_usage
            }
            for mode, profile in self.performance_profiles.items()
        }
