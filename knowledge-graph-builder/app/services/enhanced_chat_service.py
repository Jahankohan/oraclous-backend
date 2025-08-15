import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime

from app.core.neo4j_client import Neo4jClient
from app.core.exceptions import ServiceError
from app.config.settings import get_settings
from app.services.embedding_service import EmbeddingService
from app.services.advanced_graph_analytic import AdvancedGraphAnalytics
from app.utils.llm_clients import LLMClientFactory

logger = logging.getLogger(__name__)

class EnhancedChatService:
    """Enhanced chat service with multi-hop reasoning and graph analytics"""
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
        self.settings = get_settings()
        self.embedding_service = EmbeddingService()
        self.analytics = AdvancedGraphAnalytics(neo4j_client)
        self.llm_factory = LLMClientFactory()
        
    async def chat_with_graph(
        self, 
        message: str, 
        conversation_id: Optional[str] = None,
        reasoning_type: str = "auto",
        max_hops: int = 3,
        include_analytics: bool = True
    ) -> Dict[str, Any]:
        """Enhanced chat with multi-hop reasoning capabilities"""
        try:
            logger.info(f"Processing chat message with reasoning type: {reasoning_type}")
            
            # Analyze the question to determine the best approach
            question_analysis = await self._analyze_question(message)
            
            # Determine reasoning strategy
            if reasoning_type == "auto":
                reasoning_type = self._select_reasoning_strategy(question_analysis)
            
            # Get relevant context using the selected strategy
            context = await self._get_context_for_question(
                message, 
                question_analysis, 
                reasoning_type, 
                max_hops
            )
            
            # Generate answer using LLM with context
            answer = await self._generate_contextual_answer(
                message, 
                context, 
                question_analysis
            )
            
            # Add analytics insights if requested
            if include_analytics:
                analytics_insights = await self._get_analytics_insights_for_question(
                    message, context
                )
                answer["analytics_insights"] = analytics_insights
            
            # Store conversation if ID provided
            if conversation_id:
                await self._store_conversation_turn(
                    conversation_id, message, answer, context
                )
            
            return {
                "answer": answer["response"],
                "reasoning_type": reasoning_type,
                "context_used": context,
                "confidence": answer["confidence"],
                "sources": answer.get("sources", []),
                "reasoning_paths": context.get("reasoning_paths", []),
                "analytics_insights": answer.get("analytics_insights", []),
                "question_analysis": question_analysis
            }
            
        except Exception as e:
            logger.error(f"Chat with graph failed: {e}")
            raise ServiceError(f"Chat processing failed: {e}")
    
    async def _analyze_question(self, question: str) -> Dict[str, Any]:
        """Analyze the question to understand its structure and requirements"""
        try:
            llm = self.llm_factory.get_llm(self.settings.default_llm_model)
            
            analysis_prompt = f"""
            Analyze this question and categorize it:
            
            Question: "{question}"
            
            Determine:
            1. Question type (factual, analytical, comparative, exploratory, summarization)
            2. Required reasoning depth (simple, moderate, complex)
            3. Key entities or concepts mentioned
            4. Whether it needs multi-hop reasoning
            5. Whether it benefits from graph analytics (centrality, communities, etc.)
            6. Expected answer format (short, detailed, list, explanation)
            
            Respond in JSON format:
            {{
                "question_type": "factual|analytical|comparative|exploratory|summarization",
                "reasoning_depth": "simple|moderate|complex",
                "key_entities": ["entity1", "entity2"],
                "needs_multi_hop": true|false,
                "benefits_from_analytics": true|false,
                "expected_format": "short|detailed|list|explanation",
                "complexity_score": 0-10
            }}
            """
            
            response = await asyncio.to_thread(llm.predict, analysis_prompt)
            
            try:
                analysis = json.loads(response)
                # Add detected question patterns
                analysis["patterns"] = self._detect_question_patterns(question)
                return analysis
            except json.JSONDecodeError:
                # Fallback analysis
                return self._fallback_question_analysis(question)
                
        except Exception as e:
            logger.error(f"Question analysis failed: {e}")
            return self._fallback_question_analysis(question)
    
    def _detect_question_patterns(self, question: str) -> List[str]:
        """Detect specific question patterns"""
        patterns = []
        question_lower = question.lower()
        
        # Relationship queries
        if any(word in question_lower for word in ["how", "related", "connected", "relationship"]):
            patterns.append("relationship_query")
        
        # Comparative queries
        if any(word in question_lower for word in ["compare", "versus", "vs", "difference", "similar"]):
            patterns.append("comparative_query")
        
        # Path/journey queries
        if any(word in question_lower for word in ["path", "journey", "from", "to", "through"]):
            patterns.append("path_query")
        
        # Influence/impact queries
        if any(word in question_lower for word in ["influence", "impact", "effect", "cause"]):
            patterns.append("influence_query")
        
        # Community/group queries
        if any(word in question_lower for word in ["group", "cluster", "community", "similar entities"]):
            patterns.append("community_query")
        
        # Centrality/importance queries
        if any(word in question_lower for word in ["important", "central", "key", "main", "primary"]):
            patterns.append("centrality_query")
        
        # Summary queries
        if any(word in question_lower for word in ["summary", "overview", "main points", "key insights"]):
            patterns.append("summary_query")
        
        return patterns
    
    def _fallback_question_analysis(self, question: str) -> Dict[str, Any]:
        """Fallback question analysis when LLM fails"""
        return {
            "question_type": "factual",
            "reasoning_depth": "moderate",
            "key_entities": [],
            "needs_multi_hop": len(question.split()) > 10,
            "benefits_from_analytics": any(word in question.lower() 
                                         for word in ["important", "central", "related", "similar"]),
            "expected_format": "detailed",
            "complexity_score": min(10, len(question.split()) // 3),
            "patterns": self._detect_question_patterns(question)
        }
    
    def _select_reasoning_strategy(self, analysis: Dict[str, Any]) -> str:
        """Select the best reasoning strategy based on question analysis"""
        patterns = analysis.get("patterns", [])
        complexity = analysis.get("complexity_score", 5)
        
        if "path_query" in patterns:
            return "path_based"
        elif "community_query" in patterns:
            return "community_based"
        elif "centrality_query" in patterns:
            return "centrality_based"
        elif "comparative_query" in patterns:
            return "comparative"
        elif complexity > 7 or analysis.get("needs_multi_hop"):
            return "multi_hop"
        else:
            return "semantic"
    
    async def _get_context_for_question(
        self, 
        question: str, 
        analysis: Dict[str, Any], 
        reasoning_type: str,
        max_hops: int
    ) -> Dict[str, Any]:
        """Get relevant context based on reasoning strategy"""
        
        if reasoning_type == "semantic":
            return await self._get_semantic_context(question, analysis)
        elif reasoning_type == "multi_hop":
            return await self._get_multi_hop_context(question, analysis, max_hops)
        elif reasoning_type == "path_based":
            return await self._get_path_based_context(question, analysis)
        elif reasoning_type == "community_based":
            return await self._get_community_context(question, analysis)
        elif reasoning_type == "centrality_based":
            return await self._get_centrality_context(question, analysis)
        elif reasoning_type == "comparative":
            return await self._get_comparative_context(question, analysis)
        else:
            return await self._get_semantic_context(question, analysis)
    
    async def _get_semantic_context(self, question: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get context using semantic similarity"""
        try:
            # Generate question embedding
            question_embedding = await self.embedding_service.generate_query_embedding(question)
            
            # Find similar entities
            entity_query = """
            MATCH (e:Entity)
            WHERE e.embedding IS NOT NULL
            WITH e, gds.similarity.cosine(e.embedding, $queryEmbedding) AS similarity
            WHERE similarity > 0.7
            RETURN e.id as id, e.name as name, e.description as description,
                   similarity, properties(e) as properties
            ORDER BY similarity DESC
            LIMIT 10
            """
            
            entities = self.neo4j.execute_query(entity_query, {
                "queryEmbedding": question_embedding
            })
            
            # Find similar chunks
            chunk_query = """
            MATCH (c:Chunk)
            WHERE c.embedding IS NOT NULL
            WITH c, gds.similarity.cosine(c.embedding, $queryEmbedding) AS similarity
            WHERE similarity > 0.6
            MATCH (d:Document)-[:HAS_CHUNK]->(c)
            RETURN c.text as text, similarity, d.fileName as source
            ORDER BY similarity DESC
            LIMIT 5
            """
            
            chunks = self.neo4j.execute_query(chunk_query, {
                "queryEmbedding": question_embedding
            })
            
            return {
                "type": "semantic",
                "entities": entities,
                "text_chunks": chunks,
                "reasoning_paths": []
            }
            
        except Exception as e:
            logger.error(f"Semantic context retrieval failed: {e}")
            return {"type": "semantic", "entities": [], "text_chunks": [], "reasoning_paths": []}
    
    async def _get_multi_hop_context(self, question: str, analysis: Dict[str, Any], max_hops: int) -> Dict[str, Any]:
        """Get context using multi-hop reasoning"""
        try:
            # First get seed entities using semantic search
            semantic_context = await self._get_semantic_context(question, analysis)
            seed_entities = [e["id"] for e in semantic_context["entities"][:3]]
            
            if not seed_entities:
                return semantic_context
            
            # Perform multi-hop reasoning
            reasoning_result = await self.analytics.perform_multi_hop_reasoning(
                seed_entities, question, max_hops
            )
            
            # Get additional context from reasoning paths
            path_entities = set()
            for path in reasoning_result.get("reasoning_paths", [])[:5]:
                for node in path.get("nodes", []):
                    path_entities.add(node["id"])
            
            # Get detailed information for path entities
            if path_entities:
                entity_details_query = """
                MATCH (e:Entity)
                WHERE e.id IN $entityIds
                RETURN e.id as id, e.name as name, e.description as description,
                       labels(e) as labels, properties(e) as properties
                """
                
                entity_details = self.neo4j.execute_query(entity_details_query, {
                    "entityIds": list(path_entities)
                })
            else:
                entity_details = []
            
            return {
                "type": "multi_hop",
                "entities": semantic_context["entities"] + entity_details,
                "text_chunks": semantic_context["text_chunks"],
                "reasoning_paths": reasoning_result.get("reasoning_paths", []),
                "reasoning_chains": reasoning_result.get("reasoning_chains", []),
                "confidence": reasoning_result.get("confidence", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Multi-hop context retrieval failed: {e}")
            return await self._get_semantic_context(question, analysis)
    
    async def _get_path_based_context(self, question: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get context for path-based queries"""
        try:
            # Extract entities mentioned in the question
            entity_names = await self._extract_entity_names_from_question(question)
            
            if len(entity_names) >= 2:
                # Find paths between mentioned entities
                path_query = """
                MATCH (start:Entity), (end:Entity)
                WHERE start.name IN $startNames AND end.name IN $endNames
                AND start <> end
                MATCH path = shortestPath((start)-[*1..4]-(end))
                WITH path, nodes(path) as pathNodes, relationships(path) as pathRels
                RETURN 
                    [n IN pathNodes | {id: n.id, name: n.name, labels: labels(n)}] as nodes,
                    [r IN pathRels | {type: type(r), properties: properties(r)}] as relationships,
                    length(path) as pathLength
                ORDER BY pathLength
                LIMIT 5
                """
                
                paths = self.neo4j.execute_query(path_query, {
                    "startNames": entity_names[:3],
                    "endNames": entity_names[:3]
                })
                
                # Get all entities from paths
                all_path_entities = set()
                for path in paths:
                    for node in path["nodes"]:
                        all_path_entities.add(node["id"])
                
                return {
                    "type": "path_based",
                    "entities": [],  # Will be filled from paths
                    "text_chunks": [],
                    "paths": paths,
                    "path_entities": list(all_path_entities)
                }
            else:
                # Fall back to semantic search
                return await self._get_semantic_context(question, analysis)
                
        except Exception as e:
            logger.error(f"Path-based context retrieval failed: {e}")
            return await self._get_semantic_context(question, analysis)
    
    async def _get_community_context(self, question: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get context from relevant communities"""
        try:
            # Find relevant entities first
            semantic_context = await self._get_semantic_context(question, analysis)
            relevant_entities = [e["id"] for e in semantic_context["entities"][:5]]
            
            if not relevant_entities:
                return semantic_context
            
            # Find communities containing these entities
            community_query = """
            MATCH (e:Entity)
            WHERE e.id IN $entityIds AND e.community_id IS NOT NULL
            WITH DISTINCT e.community_id as communityId
            MATCH (c:Community {id: communityId})
            MATCH (member:Entity {community_id: communityId})
            RETURN c.id as communityId, c.description as description,
                   c.size as size, collect({id: member.id, name: member.name}) as members
            ORDER BY c.size DESC
            LIMIT 3
            """
            
            communities = self.neo4j.execute_query(community_query, {
                "entityIds": relevant_entities
            })
            
            # Get community members as context
            community_entities = []
            for community in communities:
                community_entities.extend(community["members"][:10])  # Top 10 from each community
            
            return {
                "type": "community_based",
                "entities": semantic_context["entities"] + community_entities,
                "text_chunks": semantic_context["text_chunks"],
                "communities": communities,
                "reasoning_paths": []
            }
            
        except Exception as e:
            logger.error(f"Community context retrieval failed: {e}")
            return await self._get_semantic_context(question, analysis)
    
    async def _get_centrality_context(self, question: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get context based on centrality metrics"""
        try:
            # Get most central entities
            central_entities_query = """
            MATCH (e:Entity)
            WHERE e.pagerank IS NOT NULL
            RETURN e.id as id, e.name as name, e.description as description,
                   e.pagerank as pagerank, e.betweenness_centrality as betweenness,
                   e.degree_centrality as degree, labels(e) as labels,
                   properties(e) as properties
            ORDER BY e.pagerank DESC
            LIMIT 15
            """
            
            central_entities = self.neo4j.execute_query(central_entities_query)
            
            # Also get semantic context
            semantic_context = await self._get_semantic_context(question, analysis)
            
            # Combine and prioritize by centrality
            all_entities = {}
            
            # Add central entities with high priority
            for entity in central_entities:
                entity["context_priority"] = entity["pagerank"] * 10
                all_entities[entity["id"]] = entity
            
            # Add semantic entities
            for entity in semantic_context["entities"]:
                if entity["id"] not in all_entities:
                    entity["context_priority"] = entity.get("similarity", 0.5) * 5
                    all_entities[entity["id"]] = entity
                else:
                    # Boost priority for entities that are both central and semantically relevant
                    all_entities[entity["id"]]["context_priority"] += entity.get("similarity", 0.5) * 5
            
            # Sort by priority
            sorted_entities = sorted(all_entities.values(), 
                                   key=lambda x: x.get("context_priority", 0), 
                                   reverse=True)
            
            return {
                "type": "centrality_based",
                "entities": sorted_entities[:15],
                "text_chunks": semantic_context["text_chunks"],
                "centrality_insights": await self._get_centrality_insights(central_entities[:5]),
                "reasoning_paths": []
            }
            
        except Exception as e:
            logger.error(f"Centrality context retrieval failed: {e}")
            return await self._get_semantic_context(question, analysis)
    
    async def _get_comparative_context(self, question: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get context for comparative queries"""
        try:
            # Extract entities to compare
            entity_names = await self._extract_entity_names_from_question(question)
            
            if len(entity_names) >= 2:
                # Get detailed info for comparison entities
                comparison_query = """
                MATCH (e:Entity)
                WHERE e.name IN $entityNames
                OPTIONAL MATCH (e)-[r]-(other:Entity)
                WITH e, type(r) as relType, count(other) as relCount
                WITH e, collect({type: relType, count: relCount}) as relationships
                RETURN e.id as id, e.name as name, e.description as description,
                       labels(e) as labels, properties(e) as properties,
                       relationships,
                       coalesce(e.pagerank, 0) as importance,
                       coalesce(e.community_id, 'unknown') as community
                """
                
                entities = self.neo4j.execute_query(comparison_query, {
                    "entityNames": entity_names
                })
                
                # Get shared connections
                if len(entities) >= 2:
                    shared_query = """
                    MATCH (e1:Entity {name: $name1})-[]-(shared:Entity)-[]-(e2:Entity {name: $name2})
                    WHERE e1 <> e2
                    RETURN shared.id as id, shared.name as name, 
                           labels(shared) as labels
                    LIMIT 10
                    """
                    
                    shared_entities = self.neo4j.execute_query(shared_query, {
                        "name1": entity_names[0],
                        "name2": entity_names[1]
                    })
                else:
                    shared_entities = []
                
                return {
                    "type": "comparative",
                    "entities": entities,
                    "shared_entities": shared_entities,
                    "text_chunks": [],
                    "comparison_metrics": await self._calculate_comparison_metrics(entities),
                    "reasoning_paths": []
                }
            else:
                return await self._get_semantic_context(question, analysis)
                
        except Exception as e:
            logger.error(f"Comparative context retrieval failed: {e}")
            return await self._get_semantic_context(question, analysis)
    
    async def _extract_entity_names_from_question(self, question: str) -> List[str]:
        """Extract potential entity names from question"""
        try:
            # Simple approach: look for capitalized words and quoted strings
            import re
            
            # Find quoted strings
            quoted = re.findall(r'"([^"]*)"', question)
            quoted.extend(re.findall(r"'([^']*)'", question))
            
            # Find capitalized words (potential entity names)
            words = question.split()
            capitalized = [word.strip('.,!?') for word in words 
                          if word[0].isupper() and len(word) > 2 
                          and word.lower() not in ['what', 'when', 'where', 'who', 'why', 'how']]
            
            # Combine and deduplicate
            potential_entities = list(set(quoted + capitalized))
            
            # Verify entities exist in the graph
            if potential_entities:
                verify_query = """
                MATCH (e:Entity)
                WHERE e.name IN $potentialNames
                RETURN DISTINCT e.name as name
                """
                
                verified = self.neo4j.execute_query(verify_query, {
                    "potentialNames": potential_entities
                })
                
                return [e["name"] for e in verified]
            
            return []
            
        except Exception as e:
            logger.error(f"Entity name extraction failed: {e}")
            return []
    
    async def _get_centrality_insights(self, central_entities: List[Dict]) -> List[str]:
        """Generate insights about central entities"""
        insights = []
        
        for entity in central_entities[:3]:
            if entity["pagerank"] > 0.01:
                insights.append(f"{entity['name']} is a highly influential entity "
                              f"(PageRank: {entity['pagerank']:.4f})")
            
            if entity.get("betweenness", 0) > 0.1:
                insights.append(f"{entity['name']} serves as an important bridge "
                              f"between different parts of the network")
        
        return insights
    
    async def _calculate_comparison_metrics(self, entities: List[Dict]) -> Dict[str, Any]:
        """Calculate metrics for entity comparison"""
        if len(entities) < 2:
            return {}
        
        metrics = {}
        
        for i, entity in enumerate(entities):
            metrics[f"entity_{i}"] = {
                "name": entity["name"],
                "importance": entity.get("importance", 0),
                "relationship_count": sum(r.get("count", 0) for r in entity.get("relationships", [])),
                "community": entity.get("community", "unknown"),
                "labels": entity.get("labels", [])
            }
        
        return metrics
    
    async def _generate_contextual_answer(
        self, 
        question: str, 
        context: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate answer using LLM with retrieved context"""
        try:
            llm = self.llm_factory.get_llm(self.settings.default_llm_model)
            
            # Prepare context for LLM
            context_text = self._format_context_for_llm(context)
            
            # Generate system prompt based on reasoning type
            system_prompt = self._get_system_prompt(context["type"], analysis)
            
            # Create main prompt
            main_prompt = f"""
            {system_prompt}
            
            Question: {question}
            
            Available Context:
            {context_text}
            
            Instructions:
            1. Answer the question directly and comprehensively
            2. Use evidence from the provided context
            3. If using reasoning paths, explain the logical connections
            4. Acknowledge any limitations or uncertainties
            5. Provide specific examples where possible
            
            Answer:
            """
            
            response = await asyncio.to_thread(llm.predict, main_prompt)
            
            # Calculate confidence based on context quality
            confidence = self._calculate_answer_confidence(context, analysis)
            
            # Extract sources from context
            sources = self._extract_sources_from_context(context)
            
            return {
                "response": response.strip(),
                "confidence": confidence,
                "sources": sources,
                "context_type": context["type"]
            }
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your question. Please try rephrasing your question.",
                "confidence": 0.0,
                "sources": [],
                "context_type": "error"
            }
    
    def _format_context_for_llm(self, context: Dict[str, Any]) -> str:
        """Format context information for LLM consumption"""
        formatted_parts = []
        
        # Add entities
        if context.get("entities"):
            entities_text = "Relevant Entities:\n"
            for entity in context["entities"][:10]:  # Limit to top 10
                entities_text += f"- {entity['name']}"
                if entity.get("description"):
                    entities_text += f": {entity['description']}"
                entities_text += "\n"
            formatted_parts.append(entities_text)
        
        # Add text chunks
        if context.get("text_chunks"):
            chunks_text = "Relevant Text Passages:\n"
            for i, chunk in enumerate(context["text_chunks"][:5]):  # Limit to top 5
                chunks_text += f"{i+1}. {chunk['text'][:300]}...\n"
            formatted_parts.append(chunks_text)
        
        # Add reasoning paths
        if context.get("reasoning_chains"):
            paths_text = "Reasoning Paths:\n"
            for i, chain in enumerate(context["reasoning_chains"][:3]):  # Top 3 paths
                paths_text += f"{i+1}. {chain}\n"
            formatted_parts.append(paths_text)
        
        # Add communities
        if context.get("communities"):
            communities_text = "Relevant Communities:\n"
            for community in context["communities"]:
                communities_text += f"- Community {community['communityId']}: {community.get('description', 'No description')}\n"
            formatted_parts.append(communities_text)
        
        # Add paths
        if context.get("paths"):
            paths_text = "Connection Paths:\n"
            for i, path in enumerate(context["paths"][:3]):
                path_str = " -> ".join([node["name"] for node in path["nodes"]])
                paths_text += f"{i+1}. {path_str}\n"
            formatted_parts.append(paths_text)
        
        return "\n\n".join(formatted_parts)
    
    def _get_system_prompt(self, context_type: str, analysis: Dict[str, Any]) -> str:
        """Get system prompt based on context type and question analysis"""
        base_prompt = "You are an AI assistant that answers questions using knowledge graph data."
        
        if context_type == "multi_hop":
            return base_prompt + " Focus on explaining the multi-step reasoning paths that connect different concepts."
        elif context_type == "community_based":
            return base_prompt + " Emphasize relationships within communities and how entities are grouped together."
        elif context_type == "centrality_based":
            return base_prompt + " Highlight the importance and influence of key entities in the network."
        elif context_type == "comparative":
            return base_prompt + " Provide detailed comparisons and contrasts between the entities."
        elif context_type == "path_based":
            return base_prompt + " Explain the connections and pathways between entities."
        else:
            return base_prompt + " Provide comprehensive answers using the available context."
    
    def _calculate_answer_confidence(self, context: Dict[str, Any], analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for the answer"""
        confidence = 0.5  # Base confidence
        
        # Boost for relevant entities
        if context.get("entities"):
            confidence += min(0.3, len(context["entities"]) * 0.03)
        
        # Boost for text chunks
        if context.get("text_chunks"):
            confidence += min(0.2, len(context["text_chunks"]) * 0.04)
        
        # Boost for reasoning paths
        if context.get("reasoning_paths"):
            confidence += min(0.2, len(context["reasoning_paths"]) * 0.05)
            
        # Boost for high-quality context
        if context.get("confidence"):  # From multi-hop reasoning
            confidence += context["confidence"] * 0.3
        
        # Adjust based on question complexity
        complexity = analysis.get("complexity_score", 5)
        if complexity > 7:
            confidence *= 0.9  # Reduce confidence for very complex questions
        
        return min(1.0, confidence)
    
    def _extract_sources_from_context(self, context: Dict[str, Any]) -> List[str]:
        """Extract source references from context"""
        sources = []
        
        # From text chunks
        for chunk in context.get("text_chunks", []):
            if chunk.get("source"):
                sources.append(chunk["source"])
        
        # From entities (if they have source information)
        for entity in context.get("entities", []):
            if entity.get("source_document"):
                sources.append(entity["source_document"])
        
        return list(set(sources))  # Remove duplicates
    
    async def _get_analytics_insights_for_question(self, question: str, context: Dict[str, Any]) -> List[str]:
        """Get analytics insights relevant to the question"""
        insights = []
        
        try:
            # Get general graph insights
            if "important" in question.lower() or "central" in question.lower():
                central_entities = await self.analytics.get_influential_nodes("pagerank", 5)
                if central_entities:
                    insights.append(f"Most influential entities: {', '.join([e['name'] for e in central_entities[:3]])}")
            
            # Community insights
            if any(word in question.lower() for word in ["group", "cluster", "community"]):
                communities = await self.analytics.detect_communities(min_community_size=5)
                if communities:
                    insights.append(f"Found {len(communities)} major communities in the knowledge graph")
            
            # Bridge nodes
            if "connect" in question.lower() or "bridge" in question.lower():
                bridge_nodes = await self.analytics.find_bridge_nodes(5)
                if bridge_nodes:
                    insights.append(f"Key connecting entities: {', '.join([n['name'] for n in bridge_nodes[:3]])}")
            
        except Exception as e:
            logger.warning(f"Failed to get analytics insights: {e}")
        
        return insights
    
    async def _store_conversation_turn(
        self, 
        conversation_id: str, 
        question: str, 
        answer: Dict[str, Any],
        context: Dict[str, Any]
    ) -> None:
        """Store conversation turn for future reference"""
        try:
            query = """
            MERGE (conv:Conversation {id: $conversationId})
            ON CREATE SET conv.created_at = datetime()
            
            CREATE (turn:ConversationTurn {
                id: randomUUID(),
                question: $question,
                answer: $answer,
                context_type: $contextType,
                confidence: $confidence,
                timestamp: datetime()
            })
            
            CREATE (conv)-[:HAS_TURN]->(turn)
            """
            
            self.neo4j.execute_write_query(query, {
                "conversationId": conversation_id,
                "question": question,
                "answer": answer["response"],
                "contextType": context["type"],
                "confidence": answer["confidence"]
            })
            
        except Exception as e:
            logger.warning(f"Failed to store conversation turn: {e}")
    
    async def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history"""
        try:
            query = """
            MATCH (conv:Conversation {id: $conversationId})-[:HAS_TURN]->(turn:ConversationTurn)
            RETURN turn.question as question,
                   turn.answer as answer,
                   turn.context_type as context_type,
                   turn.confidence as confidence,
                   turn.timestamp as timestamp
            ORDER BY turn.timestamp
            """
            
            result = self.neo4j.execute_query(query, {"conversationId": conversation_id})
            
            return [
                {
                    "question": record["question"],
                    "answer": record["answer"],
                    "context_type": record["context_type"],
                    "confidence": record["confidence"],
                    "timestamp": record["timestamp"].isoformat() if record["timestamp"] else None
                }
                for record in result
            ]
            
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []