import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from app.core.neo4j_client import Neo4jClient
from app.core.exceptions import ServiceError
from app.config.settings import get_settings
from app.services.advanced_graph_analytic import AdvancedGraphAnalytics
from app.services.enhanced_graph_service import EnhancedGraphService
from app.services.enhanced_chat_service import EnhancedChatService
from app.services.entity_resolution import EntityResolution, SchemaLearning

logger = logging.getLogger(__name__)

class AdvancedGraphIntegrationService:
    """Integration service that orchestrates all advanced graph capabilities"""
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
        self.settings = get_settings()
        
        # Initialize all services
        self.analytics = AdvancedGraphAnalytics(neo4j_client)
        self.graph_service = EnhancedGraphService(neo4j_client)
        self.chat_service = EnhancedChatService(neo4j_client)
        self.entity_resolver = EntityResolution(neo4j_client)
        self.schema_learner = SchemaLearning(neo4j_client)
    
    async def initialize_advanced_features(self) -> Dict[str, Any]:
        """Initialize all advanced graph features"""
        try:
            logger.info("Initializing advanced graph features...")
            
            # Check if graph has minimum data
            stats = await self._get_basic_stats()
            
            if stats["entity_count"] < 10:
                return {
                    "status": "insufficient_data",
                    "message": "Need at least 10 entities to enable advanced features",
                    "stats": stats
                }
            
            # Initialize features progressively
            initialization_results = {}
            
            # 1. Schema Learning
            logger.info("Running schema learning...")
            try:
                schema_results = await self.analytics.learn_graph_schema(min_frequency=2)
                initialization_results["schema_learning"] = {
                    "status": "completed",
                    "patterns_discovered": len(schema_results["discovered_patterns"]),
                    "suggestions": len(schema_results["schema_suggestions"])
                }
            except Exception as e:
                logger.error(f"Schema learning failed: {e}")
                initialization_results["schema_learning"] = {"status": "failed", "error": str(e)}
            
            #