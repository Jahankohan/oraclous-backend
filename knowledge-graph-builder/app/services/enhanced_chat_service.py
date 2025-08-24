from typing import Dict, List, Any, Optional
from uuid import UUID
from app.services.advanced_graph_context import (
    advanced_graph_context, 
    GraphContext, 
    ContextType
)
from app.services.llm_service import llm_service
from app.core.logging import get_logger
from datetime import datetime

logger = get_logger(__name__)

class GraphIntelligentChatService:
    """Enhanced chat service with graph-based reasoning and context"""
    
    def __init__(self):
        self.current_graph_id = None
        self.conversation_context = []
        self.graph_schema = None
        self.reasoning_history = []
    
    async def initialize_for_graph(self, graph_id: UUID, schema: Dict[str, Any]) -> bool:
        """Initialize chat service for specific graph with enhanced setup"""
        
        try:
            self.current_graph_id = graph_id
            self.graph_schema = schema
            self.conversation_context = []
            self.reasoning_history = []
            
            # Pre-compute graph statistics for better context
            await self._precompute_graph_statistics()
            
            logger.info(f"Enhanced chat initialized for graph {graph_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced chat: {e}")
            return False
    
    async def chat_with_advanced_reasoning(
        self,
        query: str,
        reasoning_mode: str = "comprehensive",  # comprehensive, focused, exploratory
        max_context_tokens: int = 4000,
        include_reasoning_chain: bool = True
    ) -> Dict[str, Any]:
        """
        Advanced chat with multi-layered graph reasoning
        
        reasoning_mode options:
        - comprehensive: Use all available graph algorithms and context
        - focused: Focus on direct relationships and close neighbors
        - exploratory: Emphasize pathways and community connections
        """
        
        if not self.current_graph_id:
            return {
                "answer": "Error: Chat not initialized for any graph",
                "success": False,
                "reasoning_chain": []
            }
        
        try:
            logger.info(f"Processing advanced query: '{query[:50]}...' in {reasoning_mode} mode")
            
            # Step 1: Generate rich graph context
            start_time = datetime.now()
            
            context_params = self._get_context_parameters(reasoning_mode)
            graph_context = await advanced_graph_context.generate_rich_context(
                query=query,
                graph_id=self.current_graph_id,
                max_context_size=max_context_tokens,
                reasoning_depth=context_params["depth"]
            )
            
            context_time = (datetime.now() - start_time).total_seconds()
            
            # Step 2: Generate grounded response
            response_start = datetime.now()
            
            response_data = await advanced_graph_context.generate_grounded_response(
                query=query,
                context=graph_context
            )
            
            response_time = (datetime.now() - response_start).total_seconds()
            
            # Step 3: Add conversation memory and learning
            conversation_insight = await self._analyze_conversation_pattern(query, response_data)
            
            # Step 4: Enhance response with graph insights
            enhanced_response = await self._enhance_response_with_insights(
                query, response_data, graph_context
            )
            
            # Step 5: Update reasoning history
            self.reasoning_history.append({
                "query": query,
                "reasoning_mode": reasoning_mode,
                "context_types_used": list(graph_context.confidence_scores.keys()),
                "grounding_confidence": sum(graph_context.confidence_scores.values()) / len(graph_context.confidence_scores) if graph_context.confidence_scores else 0,
                "timestamp": datetime.now()
            })
            
            # Prepare final response
            final_response = {
                "answer": enhanced_response["answer"],
                "success": response_data["success"],
                "grounded": response_data.get("grounded", False),
                "reasoning_mode": reasoning_mode,
                
                # Context information
                "context_summary": {
                    "entities_analyzed": len(graph_context.primary_entities),
                    "relationships_found": len(graph_context.relationships),
                    "pathways_discovered": len(graph_context.pathways),
                    "communities_identified": len(graph_context.communities),
                    "influential_nodes": len(graph_context.influential_nodes)
                },
                
                # Performance metrics
                "performance": {
                    "context_generation_time": context_time,
                    "response_generation_time": response_time,
                    "total_time": context_time + response_time
                },
                
                # Reasoning transparency
                "reasoning_chain": graph_context.reasoning_chain if include_reasoning_chain else [],
                "confidence_scores": graph_context.confidence_scores,
                "conversation_insight": conversation_insight,
                
                # Enhanced features
                "related_entities": enhanced_response.get("related_entities", []),
                "suggested_followup": enhanced_response.get("suggested_followup", []),
                "graph_insights": enhanced_response.get("graph_insights", {})
            }
            
            # Add to conversation context for future queries
            self.conversation_context.append({
                "query": query,
                "response": final_response["answer"],
                "entities_discussed": [e["name"] for e in graph_context.primary_entities],
                "timestamp": datetime.now()
            })
            
            return final_response
            
        except Exception as e:
            logger.error(f"Advanced chat processing failed: {e}")
            return {
                "answer": "I encountered an error while analyzing the graph to answer your question.",
                "success": False,
                "error": str(e),
                "reasoning_chain": ["Error in graph analysis pipeline"]
            }
    
    def _get_context_parameters(self, reasoning_mode: str) -> Dict[str, Any]:
        """Get context generation parameters based on reasoning mode"""
        
        if reasoning_mode == "comprehensive":
            return {
                "depth": 3,
                "include_communities": True,
                "include_influence": True,
                "include_temporal": True,
                "max_pathways": 5
            }
        elif reasoning_mode == "focused":
            return {
                "depth": 2,
                "include_communities": False,
                "include_influence": True,
                "include_temporal": False,
                "max_pathways": 2
            }
        elif reasoning_mode == "exploratory":
            return {
                "depth": 4,
                "include_communities": True,
                "include_influence": False,
                "include_temporal": True,
                "max_pathways": 8
            }
        else:
            return {
                "depth": 2,
                "include_communities": True,
                "include_influence": True,
                "include_temporal": False,
                "max_pathways": 3
            }
    
    async def _precompute_graph_statistics(self) -> Dict[str, Any]:
        """Precompute graph statistics for better context generation"""
        
        from app.core.neo4j_client import neo4j_client
        
        stats_query = """
        MATCH (n)
        WHERE n.graph_id = $graph_id
        
        WITH count(n) as node_count
        
        MATCH ()-[r]-()
        WHERE r.graph_id = $graph_id
        
        WITH node_count, count(r) as rel_count
        
        MATCH (n)
        WHERE n.graph_id = $graph_id
        
        WITH node_count, rel_count, 
             collect(DISTINCT labels(n)) as all_labels
        
        RETURN node_count,
               rel_count,
               all_labels,
               node_count * 1.0 / (node_count + rel_count) as entity_ratio
        """
        
        try:
            result = await neo4j_client.execute_query(stats_query, {
                "graph_id": str(self.current_graph_id)
            })
            
            if result:
                stats = result[0]
                self.graph_statistics = {
                    "node_count": stats["node_count"],
                    "relationship_count": stats["rel_count"],
                    "entity_types": [label[0] if label else "Unknown" for label in stats["all_labels"]],
                    "entity_ratio": stats["entity_ratio"],
                    "density": stats["rel_count"] / (stats["node_count"] * (stats["node_count"] - 1)) if stats["node_count"] > 1 else 0
                }
            else:
                self.graph_statistics = {}
                
        except Exception as e:
            logger.warning(f"Failed to compute graph statistics: {e}")
            self.graph_statistics = {}
    
    async def _analyze_conversation_pattern(
        self, 
        query: str, 
        response_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze conversation patterns to improve future responses"""
        
        # Analyze query type
        query_type = await self._classify_query_type(query)
        
        # Analyze conversation flow
        conversation_flow = {
            "query_type": query_type,
            "follows_previous": len(self.conversation_context) > 0,
            "entity_continuity": self._check_entity_continuity(query),
            "complexity_level": self._assess_query_complexity(query)
        }
        
        return conversation_flow
    
    async def _classify_query_type(self, query: str) -> str:
        """Classify the type of query for better processing"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["who is", "what is", "tell me about"]):
            return "entity_description"
        elif any(word in query_lower for word in ["how", "why", "explain"]):
            return "explanation"
        elif any(word in query_lower for word in ["when", "where"]):
            return "factual_lookup"
        elif any(word in query_lower for word in ["relationship", "connected", "related"]):
            return "relationship_analysis"
        elif any(word in query_lower for word in ["find", "search", "list"]):
            return "search_query"
        elif "?" in query:
            return "question"
        else:
            return "general_query"
    
    def _check_entity_continuity(self, query: str) -> bool:
        """Check if query continues discussing entities from previous conversation"""
        
        if not self.conversation_context:
            return False
        
        # Get entities from recent conversation
        recent_entities = []
        for context in self.conversation_context[-3:]:  # Last 3 exchanges
            recent_entities.extend(context.get("entities_discussed", []))
        
        # Check if current query mentions any recent entities
        query_lower = query.lower()
        return any(entity.lower() in query_lower for entity in recent_entities)
    
    def _assess_query_complexity(self, query: str) -> str:
        """Assess the complexity level of the query"""
        
        complexity_indicators = {
            "simple": ["what", "who", "where", "when"],
            "medium": ["how", "why", "explain", "describe"],
            "complex": ["analyze", "compare", "relationship", "pattern", "trend", "implication"]
        }
        
        query_lower = query.lower()
        
        for level, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                return level
        
        # Default based on length and structure
        if len(query.split()) > 15 or query.count("and") > 2:
            return "complex"
        elif len(query.split()) > 8:
            return "medium"
        else:
            return "simple"
    
    async def _enhance_response_with_insights(
        self, 
        query: str, 
        response_data: Dict[str, Any], 
        graph_context: GraphContext
    ) -> Dict[str, Any]:
        """Enhance response with additional graph insights and suggestions"""
        
        enhanced = response_data.copy()
        
        # Add related entities that might be of interest
        if graph_context.neighborhoods:
            related_entities = []
            for neighborhood in graph_context.neighborhoods[:3]:
                for neighbor_info in neighborhood.get("neighbors", []):
                    neighbor = neighbor_info.get("neighbor", {})
                    if neighbor.get("name"):
                        related_entities.append({
                            "name": neighbor["name"],
                            "relationship": neighbor_info.get("relationship", "RELATED"),
                            "labels": neighbor.get("labels", [])
                        })
            
            enhanced["related_entities"] = related_entities[:5]  # Top 5
        
        # Generate follow-up suggestions
        followup_suggestions = await self._generate_followup_suggestions(
            query, graph_context
        )
        enhanced["suggested_followup"] = followup_suggestions
        
        # Add graph insights
        insights = {
            "graph_coverage": f"Analyzed {len(graph_context.primary_entities)} entities with {len(graph_context.relationships)} relationships",
            "reasoning_depth": f"Used {len(graph_context.reasoning_chain)} reasoning steps",
            "confidence_summary": f"Highest confidence in {max(graph_context.confidence_scores, key=graph_context.confidence_scores.get) if graph_context.confidence_scores else 'N/A'} context"
        }
        
        if graph_context.pathways:
            insights["connectivity"] = f"Found {len(graph_context.pathways)} connection pathways between entities"
        
        if graph_context.communities:
            insights["clustering"] = f"Entities span {len(graph_context.communities)} different communities"
        
        enhanced["graph_insights"] = insights
        
        return enhanced
    
    async def _generate_followup_suggestions(
        self, 
        query: str, 
        graph_context: GraphContext
    ) -> List[str]:
        """Generate intelligent follow-up questions based on graph context"""
        
        suggestions = []
        
        # Based on entities found
        if graph_context.primary_entities:
            for entity in graph_context.primary_entities[:2]:
                suggestions.append(f"Tell me more about {entity['name']}")
        
        # Based on relationships discovered
        if graph_context.relationships:
            unique_rel_types = set(r.get("type", "") for r in graph_context.relationships)
            for rel_type in list(unique_rel_types)[:2]:
                suggestions.append(f"What other {rel_type.lower().replace('_', ' ')} relationships exist?")
        
        # Based on pathways
        if graph_context.pathways:
            suggestions.append("How are these entities connected?")
            suggestions.append("What are the shortest paths between key entities?")
        
        # Based on communities
        if graph_context.communities:
            suggestions.append("What other entities are in the same community?")
        
        # Based on influential nodes
        if graph_context.influential_nodes:
            top_influential = graph_context.influential_nodes[0]
            suggestions.append(f"Why is {top_influential.get('name', 'this entity')} so influential in the graph?")
        
        return suggestions[:4]  # Limit to 4 suggestions
    
    async def explain_reasoning(self, query: str) -> Dict[str, Any]:
        """Provide detailed explanation of how the system would reason about a query"""
        
        if not self.current_graph_id:
            return {"error": "No graph initialized"}
        
        # Generate context without full processing
        graph_context = await advanced_graph_context.generate_rich_context(
            query=query,
            graph_id=self.current_graph_id,
            max_context_size=2000,
            reasoning_depth=2
        )
        
        explanation = {
            "query_analysis": {
                "query_type": await self._classify_query_type(query),
                "complexity": self._assess_query_complexity(query),
                "entities_recognized": len(graph_context.primary_entities)
            },
            
            "graph_analysis_steps": graph_context.reasoning_chain,
            
            "context_sources": {
                "direct_matches": len(graph_context.primary_entities),
                "neighborhood_expansion": len(graph_context.neighborhoods),
                "pathway_analysis": len(graph_context.pathways),
                "community_context": len(graph_context.communities),
                "influence_analysis": len(graph_context.influential_nodes),
                "temporal_context": len(graph_context.temporal_context)
            },
            
            "confidence_assessment": graph_context.confidence_scores,
            
            "reasoning_strategy": "The system will use graph algorithms to find relevant entities, analyze their relationships, discover connection patterns, and provide a response grounded only in the discovered context."
        }
        
        return explanation
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation for debugging/analysis"""
        
        return {
            "graph_id": str(self.current_graph_id) if self.current_graph_id else None,
            "conversation_length": len(self.conversation_context),
            "reasoning_history_length": len(self.reasoning_history),
            "recent_entities_discussed": [
                entity for context in self.conversation_context[-3:]
                for entity in context.get("entities_discussed", [])
            ],
            "reasoning_modes_used": list(set(r["reasoning_mode"] for r in self.reasoning_history)),
            "average_grounding_confidence": sum(r["grounding_confidence"] for r in self.reasoning_history) / len(self.reasoning_history) if self.reasoning_history else 0,
            "graph_statistics": getattr(self, 'graph_statistics', {})
        }

# Create singleton instance  
graph_intelligent_chat = GraphIntelligentChatService()