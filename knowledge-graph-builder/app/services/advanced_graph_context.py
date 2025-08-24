from typing import Dict, List, Any, Optional, Set, Tuple
from uuid import UUID
from app.core.neo4j_client import neo4j_client
from app.services.llm_service import llm_service
from app.core.logging import get_logger
import json
import asyncio
from dataclasses import dataclass
from enum import Enum

logger = get_logger(__name__)

class ContextType(Enum):
    DIRECT = "direct"                    # Direct entity matches
    NEIGHBORHOOD = "neighborhood"        # 1-hop neighbors  
    COMMUNITY = "community"             # Same community/cluster
    PATHWAY = "pathway"                 # Multi-hop paths
    HIERARCHICAL = "hierarchical"       # Parent-child relationships
    TEMPORAL = "temporal"               # Time-based connections
    INFLUENCE = "influence"             # High-centrality entities

@dataclass
class GraphContext:
    """Rich context extracted from graph structure"""
    primary_entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    neighborhoods: List[Dict[str, Any]]
    pathways: List[Dict[str, Any]]
    communities: List[Dict[str, Any]]
    influential_nodes: List[Dict[str, Any]]
    temporal_context: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    reasoning_chain: List[str]

class AdvancedGraphContextService:
    """Advanced context generation using graph algorithms and topology"""
    
    def __init__(self):
        self.centrality_cache = {}
        self.community_cache = {}
        
    async def generate_rich_context(
        self,
        query: str,
        graph_id: UUID,
        max_context_size: int = 5000,
        reasoning_depth: int = 3
    ) -> GraphContext:
        """
        Generate rich context using multiple graph-based techniques
        """
        
        logger.info(f"Generating advanced context for query: '{query[:50]}...'")
        
        # Step 1: Entity Recognition & Disambiguation
        primary_entities = await self._extract_and_disambiguate_entities(query, graph_id)
        
        # Step 2: Multi-Strategy Context Expansion
        context_strategies = await asyncio.gather(
            self._get_direct_context(primary_entities, graph_id),
            self._get_neighborhood_context(primary_entities, graph_id),
            self._get_pathway_context(primary_entities, graph_id, reasoning_depth),
            self._get_community_context(primary_entities, graph_id),
            self._get_influential_context(query, graph_id),
            self._get_temporal_context(primary_entities, graph_id)
        )
        
        direct_ctx, neighborhood_ctx, pathway_ctx, community_ctx, influence_ctx, temporal_ctx = context_strategies
        
        # Step 3: Context Ranking & Filtering
        ranked_context = await self._rank_and_filter_context(
            query, direct_ctx, neighborhood_ctx, pathway_ctx, 
            community_ctx, influence_ctx, temporal_ctx, max_context_size
        )
        
        # Step 4: Generate Reasoning Chain
        reasoning_chain = await self._generate_reasoning_chain(query, ranked_context)
        
        return GraphContext(
            primary_entities=primary_entities,
            relationships=ranked_context.get("relationships", []),
            neighborhoods=ranked_context.get("neighborhoods", []),
            pathways=ranked_context.get("pathways", []),
            communities=ranked_context.get("communities", []),
            influential_nodes=ranked_context.get("influential", []),
            temporal_context=ranked_context.get("temporal", []),
            confidence_scores=ranked_context.get("confidence", {}),
            reasoning_chain=reasoning_chain
        )

    async def _extract_and_disambiguate_entities(
        self, 
        query: str, 
        graph_id: UUID
    ) -> List[Dict[str, Any]]:
        """Enhanced entity recognition with disambiguation"""
        
        # Step 1: Extract potential entities from query using NLP
        entity_extraction_query = f"""
        Extract all potential entities (people, organizations, locations, concepts) from this text:
        "{query}"
        
        Return as JSON array: ["entity1", "entity2", ...]
        """
        
        try:
            response = await llm_service.llm.ainvoke([
                {"role": "system", "content": "Extract entities from text. Return only a JSON array."},
                {"role": "user", "content": entity_extraction_query}
            ])
            
            entities = json.loads(response.content.strip())
            if not isinstance(entities, list):
                entities = []
        except:
            entities = []
        
        # Step 2: Disambiguate against graph entities
        disambiguated = []
        for entity_text in entities:
            matches = await self._find_entity_matches(entity_text, graph_id)
            if matches:
                disambiguated.extend(matches[:2])  # Top 2 matches
        
        return disambiguated

    async def _find_entity_matches(
        self, 
        entity_text: str, 
        graph_id: UUID
    ) -> List[Dict[str, Any]]:
        """Find matching entities in graph with fuzzy matching"""
        
        query = """
        // Exact match
        MATCH (n)
        WHERE n.graph_id = $graph_id 
        AND (toLower(n.name) CONTAINS toLower($entity_text) 
             OR toLower($entity_text) CONTAINS toLower(n.name))
        
        // Calculate relevance score
        WITH n, 
             CASE 
                WHEN toLower(n.name) = toLower($entity_text) THEN 1.0
                WHEN toLower(n.name) CONTAINS toLower($entity_text) THEN 0.8
                WHEN toLower($entity_text) CONTAINS toLower(n.name) THEN 0.6
                ELSE 0.4
             END as relevance_score
        
        // Get entity with relationships count for importance
        OPTIONAL MATCH (n)-[r]-()
        WHERE r.graph_id = $graph_id
        
        RETURN n.id as id, n.name as name, labels(n) as labels, 
               n{.*} as properties, count(r) as relationship_count,
               relevance_score
        ORDER BY relevance_score DESC, relationship_count DESC
        LIMIT 5
        """
        
        results = await neo4j_client.execute_query(query, {
            "entity_text": entity_text,
            "graph_id": str(graph_id)
        })
        
        return [dict(result) for result in results]

    async def _get_neighborhood_context(
        self, 
        entities: List[Dict[str, Any]], 
        graph_id: UUID
    ) -> Dict[str, Any]:
        """Get rich neighborhood context (1-2 hop neighbors)"""
        
        if not entities:
            return {"neighborhoods": [], "relationships": []}
        
        entity_ids = [e["id"] for e in entities]
        
        query = """
        MATCH (start)
        WHERE start.id IN $entity_ids AND start.graph_id = $graph_id
        
        // 1-hop neighbors with relationship details
        MATCH (start)-[r1]-(neighbor1)
        WHERE r1.graph_id = $graph_id AND neighbor1.graph_id = $graph_id
        
        // Optional 2-hop neighbors for richer context
        OPTIONAL MATCH (neighbor1)-[r2]-(neighbor2)
        WHERE r2.graph_id = $graph_id AND neighbor2.graph_id = $graph_id
        AND neighbor2.id <> start.id
        
        // Collect neighborhood info
        WITH start, r1, neighbor1, 
             collect(DISTINCT {
                 node: neighbor2{.id, .name, labels: labels(neighbor2)},
                 relationship: type(r2),
                 properties: r2{.*}
             }) as second_hop
        
        RETURN start.id as center_id,
               start.name as center_name,
               {
                   relationship: type(r1),
                   properties: r1{.*},
                   neighbor: neighbor1{.id, .name, labels: labels(neighbor1)},
                   second_hop: second_hop[..3]  // Limit to 3 second-hop neighbors
               } as neighborhood_info
        """
        
        results = await neo4j_client.execute_query(query, {
            "entity_ids": entity_ids,
            "graph_id": str(graph_id)
        })
        
        neighborhoods = {}
        relationships = []
        
        for result in results:
            center_id = result["center_id"]
            if center_id not in neighborhoods:
                neighborhoods[center_id] = {
                    "center": {"id": center_id, "name": result["center_name"]},
                    "neighbors": []
                }
            
            neighborhoods[center_id]["neighbors"].append(result["neighborhood_info"])
            relationships.append({
                "source": center_id,
                "target": result["neighborhood_info"]["neighbor"]["id"],
                "type": result["neighborhood_info"]["relationship"],
                "properties": result["neighborhood_info"]["properties"]
            })
        
        return {
            "neighborhoods": list(neighborhoods.values()),
            "relationships": relationships
        }

    async def _get_pathway_context(
        self, 
        entities: List[Dict[str, Any]], 
        graph_id: UUID,
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """Find meaningful paths between entities using graph algorithms"""
        
        if len(entities) < 2:
            return {"pathways": []}
        
        pathways = []
        
        # Find paths between each pair of entities
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                entity1 = entities[i]
                entity2 = entities[j]
                
                paths = await self._find_shortest_paths(
                    entity1["id"], entity2["id"], graph_id, max_depth
                )
                pathways.extend(paths)
        
        return {"pathways": pathways}

    async def _find_shortest_paths(
        self,
        start_id: str,
        end_id: str,
        graph_id: UUID,
        max_depth: int
    ) -> List[Dict[str, Any]]:
        """Find shortest paths between two entities"""
        
        query = """
        MATCH (start {id: $start_id, graph_id: $graph_id})
        MATCH (end {id: $end_id, graph_id: $graph_id})
        
        MATCH path = shortestPath((start)-[*1..$max_depth]-(end))
        WHERE all(r in relationships(path) WHERE r.graph_id = $graph_id)
        
        WITH path, length(path) as path_length
        ORDER BY path_length
        LIMIT 3
        
        RETURN [node in nodes(path) | {
            id: node.id, 
            name: node.name, 
            labels: labels(node)
        }] as path_nodes,
        [rel in relationships(path) | {
            type: type(rel),
            properties: rel{.*}
        }] as path_relationships,
        path_length
        """
        
        results = await neo4j_client.execute_query(query, {
            "start_id": start_id,
            "end_id": end_id,
            "graph_id": str(graph_id),
            "max_depth": max_depth
        })
        
        return [
            {
                "start": start_id,
                "end": end_id,
                "nodes": result["path_nodes"],
                "relationships": result["path_relationships"],
                "length": result["path_length"]
            }
            for result in results
        ]

    async def _get_community_context(
        self, 
        entities: List[Dict[str, Any]], 
        graph_id: UUID
    ) -> Dict[str, Any]:
        """Find entities in the same communities using community detection"""
        
        if not entities:
            return {"communities": []}
        
        # Use Louvain algorithm for community detection
        community_query = """
        MATCH (n)-[r]-(m)
        WHERE n.graph_id = $graph_id AND r.graph_id = $graph_id AND m.graph_id = $graph_id
        
        WITH gds.graph.project(
            'community-graph-' + $graph_id,
            {
                Node: {
                    label: '*',
                    properties: {nodeId: 'id'}
                }
            },
            {
                Relationship: {
                    type: '*',
                    orientation: 'UNDIRECTED'
                }
            }
        ) AS g
        
        CALL gds.louvain.stream('community-graph-' + $graph_id)
        YIELD nodeId, communityId
        
        MATCH (node)
        WHERE id(node) = nodeId
        
        RETURN node.id as entity_id, 
               node.name as entity_name,
               communityId,
               labels(node) as labels
        """
        
        try:
            results = await neo4j_client.execute_query(community_query, {
                "graph_id": str(graph_id)
            })
            
            # Group by community
            communities = {}
            for result in results:
                community_id = result["communityId"]
                if community_id not in communities:
                    communities[community_id] = []
                
                communities[community_id].append({
                    "id": result["entity_id"],
                    "name": result["entity_name"],
                    "labels": result["labels"]
                })
            
            return {"communities": list(communities.values())}
            
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            return {"communities": []}

    async def _get_influential_context(
        self, 
        query: str, 
        graph_id: UUID
    ) -> Dict[str, Any]:
        """Find influential nodes using centrality measures"""
        
        # Calculate PageRank for entity importance
        pagerank_query = """
        MATCH (n)-[r]-(m)
        WHERE n.graph_id = $graph_id AND r.graph_id = $graph_id AND m.graph_id = $graph_id
        
        WITH gds.graph.project(
            'pagerank-graph-' + $graph_id,
            '*',
            '*'
        ) AS g
        
        CALL gds.pageRank.stream('pagerank-graph-' + $graph_id)
        YIELD nodeId, score
        
        MATCH (node)
        WHERE id(node) = nodeId
        
        RETURN node.id as entity_id,
               node.name as entity_name,
               labels(node) as labels,
               score
        ORDER BY score DESC
        LIMIT 10
        """
        
        try:
            results = await neo4j_client.execute_query(pagerank_query, {
                "graph_id": str(graph_id)
            })
            
            influential_nodes = [
                {
                    "id": result["entity_id"],
                    "name": result["entity_name"],
                    "labels": result["labels"],
                    "influence_score": result["score"]
                }
                for result in results
            ]
            
            return {"influential": influential_nodes}
            
        except Exception as e:
            logger.warning(f"PageRank calculation failed: {e}")
            return {"influential": []}

    async def _get_temporal_context(
        self, 
        entities: List[Dict[str, Any]], 
        graph_id: UUID
    ) -> Dict[str, Any]:
        """Get temporal context if time-based properties exist"""
        
        if not entities:
            return {"temporal": []}
        
        entity_ids = [e["id"] for e in entities]
        
        temporal_query = """
        MATCH (n)-[r]-(m)
        WHERE n.id IN $entity_ids 
        AND n.graph_id = $graph_id 
        AND r.graph_id = $graph_id
        AND (r.start_date IS NOT NULL OR r.end_date IS NOT NULL 
             OR r.created_at IS NOT NULL OR r.timestamp IS NOT NULL)
        
        RETURN n.id as source_id,
               n.name as source_name,
               type(r) as relationship_type,
               m.id as target_id,
               m.name as target_name,
               coalesce(r.start_date, r.created_at, r.timestamp) as temporal_marker,
               r{.*} as relationship_properties
        ORDER BY temporal_marker DESC
        LIMIT 20
        """
        
        try:
            results = await neo4j_client.execute_query(temporal_query, {
                "entity_ids": entity_ids,
                "graph_id": str(graph_id)
            })
            
            return {"temporal": [dict(result) for result in results]}
            
        except Exception as e:
            logger.warning(f"Temporal context extraction failed: {e}")
            return {"temporal": []}

    async def _rank_and_filter_context(
        self,
        query: str,
        *context_sources,
        max_size: int = 5000
    ) -> Dict[str, Any]:
        """Rank and filter context by relevance to maintain focus"""
        
        # Combine all context sources
        combined_context = {}
        for ctx in context_sources:
            for key, value in ctx.items():
                if key not in combined_context:
                    combined_context[key] = []
                combined_context[key].extend(value if isinstance(value, list) else [value])
        
        # Simple relevance scoring (can be enhanced with embeddings)
        query_terms = set(query.lower().split())
        
        def calculate_relevance(item):
            """Calculate relevance score for context item"""
            text_content = str(item).lower()
            matches = sum(1 for term in query_terms if term in text_content)
            return matches / len(query_terms) if query_terms else 0
        
        # Rank and filter each context type
        filtered_context = {}
        for key, items in combined_context.items():
            if items:
                scored_items = [(item, calculate_relevance(item)) for item in items]
                scored_items.sort(key=lambda x: x[1], reverse=True)
                
                # Take top items based on type
                if key == "relationships":
                    filtered_context[key] = [item for item, _ in scored_items[:15]]
                elif key == "neighborhoods":
                    filtered_context[key] = [item for item, _ in scored_items[:5]]
                else:
                    filtered_context[key] = [item for item, _ in scored_items[:10]]
        
        # Calculate confidence scores
        confidence_scores = {}
        for key, items in filtered_context.items():
            if items:
                confidence_scores[key] = min(1.0, len(items) / 5)  # Normalized confidence
        
        filtered_context["confidence"] = confidence_scores
        return filtered_context

    async def _generate_reasoning_chain(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate step-by-step reasoning chain for transparency"""
        
        reasoning_steps = []
        
        # Entity identification
        if context.get("neighborhoods"):
            reasoning_steps.append(f"Identified {len(context['neighborhoods'])} relevant entities in the graph")
        
        # Relationship analysis
        if context.get("relationships"):
            reasoning_steps.append(f"Found {len(context['relationships'])} direct relationships")
        
        # Path analysis
        if context.get("pathways"):
            reasoning_steps.append(f"Discovered {len(context['pathways'])} connection paths between entities")
        
        # Community context
        if context.get("communities"):
            reasoning_steps.append(f"Identified entities within {len(context['communities'])} communities")
        
        # Influence context
        if context.get("influential"):
            reasoning_steps.append(f"Considered {len(context['influential'])} high-influence entities")
        
        # Temporal context
        if context.get("temporal"):
            reasoning_steps.append(f"Found {len(context['temporal'])} time-based relationships")
        
        reasoning_steps.append("Synthesizing information to provide accurate, grounded response")
        
        return reasoning_steps

    async def generate_grounded_response(
        self,
        query: str,
        context: GraphContext
    ) -> Dict[str, Any]:
        """Generate response strictly grounded in graph context"""
        
        # Prepare rich context for LLM
        context_text = self._format_context_for_llm(context)
        
        system_prompt = f"""
        You are a knowledge graph assistant. Answer the question using ONLY the provided graph context.
        
        CRITICAL RULES:
        1. Use ONLY information from the graph context below
        2. If the answer isn't in the context, say "I don't have that information in the knowledge graph"
        3. Cite specific entities and relationships when possible
        4. Show your reasoning process
        5. Be precise and factual
        
        GRAPH CONTEXT:
        {context_text}
        
        REASONING CHAIN:
        {' → '.join(context.reasoning_chain)}
        """
        
        try:
            response = await llm_service.llm.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {query}"}
            ])
            
            # Validate response doesn't hallucinate
            is_grounded = await self._validate_response_grounding(
                response.content, context_text
            )
            
            return {
                "answer": response.content,
                "grounded": is_grounded,
                "context_used": context_text[:500] + "..." if len(context_text) > 500 else context_text,
                "reasoning_chain": context.reasoning_chain,
                "confidence_scores": context.confidence_scores,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                "answer": "I encountered an error while processing your question.",
                "grounded": False,
                "error": str(e),
                "success": False
            }

    def _format_context_for_llm(self, context: GraphContext) -> str:
        """Format graph context for LLM consumption"""
        
        formatted_parts = []
        
        # Primary entities
        if context.primary_entities:
            entities_text = ", ".join([f"{e['name']} ({', '.join(e.get('labels', []))})" for e in context.primary_entities])
            formatted_parts.append(f"PRIMARY ENTITIES: {entities_text}")
        
        # Direct relationships
        if context.relationships:
            rels_text = "\n".join([
                f"- {r.get('source', 'Unknown')} {r.get('type', 'RELATED_TO')} {r.get('target', 'Unknown')}"
                for r in context.relationships[:10]
            ])
            formatted_parts.append(f"RELATIONSHIPS:\n{rels_text}")
        
        # Neighborhood context
        if context.neighborhoods:
            neighborhood_text = []
            for neighborhood in context.neighborhoods[:3]:
                center = neighborhood.get('center', {})
                neighbors = neighborhood.get('neighbors', [])[:5]
                neighbor_summary = ", ".join([n.get('neighbor', {}).get('name', 'Unknown') for n in neighbors])
                neighborhood_text.append(f"- {center.get('name', 'Unknown')} is connected to: {neighbor_summary}")
            formatted_parts.append(f"NEIGHBORHOODS:\n" + "\n".join(neighborhood_text))
        
        # Pathways
        if context.pathways:
            pathway_text = []
            for pathway in context.pathways[:3]:
                nodes = [n.get('name', 'Unknown') for n in pathway.get('nodes', [])]
                pathway_text.append(f"- Path: {' → '.join(nodes)}")
            formatted_parts.append(f"CONNECTION PATHS:\n" + "\n".join(pathway_text))
        
        # Influential entities
        if context.influential_nodes:
            influential_text = ", ".join([
                f"{node.get('name', 'Unknown')} (influence: {node.get('influence_score', 0):.2f})"
                for node in context.influential_nodes[:5]
            ])
            formatted_parts.append(f"KEY ENTITIES: {influential_text}")
        
        return "\n\n".join(formatted_parts)

    async def _validate_response_grounding(self, response: str, context: str) -> bool:
        """Validate that response is grounded in provided context"""
        
        validation_prompt = f"""
        Check if this RESPONSE only uses information from the CONTEXT. Be strict.
        
        CONTEXT: {context[:1000]}
        RESPONSE: {response}
        
        Return only "GROUNDED" if response uses only context information.
        Return only "NOT_GROUNDED" if response adds external information.
        """
        
        try:
            validation = await llm_service.llm.ainvoke([
                {"role": "system", "content": "You are a fact-checker. Be very strict about grounding."},
                {"role": "user", "content": validation_prompt}
            ])
            
            return "GROUNDED" in validation.content.upper()
            
        except Exception as e:
            logger.warning(f"Response validation failed: {e}")
            return False

# Create singleton instance
advanced_graph_context = AdvancedGraphContextService()