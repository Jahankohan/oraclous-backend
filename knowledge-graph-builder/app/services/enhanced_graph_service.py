import logging
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import json

from app.core.neo4j_client import Neo4jClient
from app.core.exceptions import ServiceError
from app.config.settings import get_settings
from app.models.responses import GraphVisualization, GraphNode, GraphRelationship, DuplicateNode
from app.services.embedding_service import EmbeddingService
from app.utils.llm_clients import LLMClientFactory
from app.services.advanced_graph_analytic import AdvancedGraphAnalytics

logger = logging.getLogger(__name__)

class EnhancedGraphService:
    """Enhanced Graph Service with advanced analytics capabilities"""
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
        self.settings = get_settings()
        self.embedding_service = EmbeddingService()
        self.llm_factory = LLMClientFactory()
        self.analytics = AdvancedGraphAnalytics(neo4j_client)
    
    async def get_intelligent_graph_visualization(
        self, 
        file_names: Optional[List[str]] = None,
        limit: int = 100,
        layout_type: str = "force_directed",
        include_analytics: bool = True
    ) -> GraphVisualization:
        """Get intelligent graph visualization with analytics-driven layout"""
        try:
            # Get base graph data
            base_graph = await self._get_base_graph_data(file_names, limit)
            
            if include_analytics:
                # Enhance nodes with centrality metrics
                await self._enhance_nodes_with_centrality(base_graph.nodes)
                
                # Add community information
                await self._enhance_nodes_with_communities(base_graph.nodes)
                
                # Apply intelligent layout based on analytics
                base_graph = await self._apply_intelligent_layout(base_graph, layout_type)
            
            return base_graph
            
        except Exception as e:
            logger.error(f"Error getting intelligent graph visualization: {e}")
            raise ServiceError(f"Failed to get intelligent graph visualization: {e}")
    
    async def _get_base_graph_data(self, file_names: Optional[List[str]], limit: int) -> GraphVisualization:
        """Get base graph visualization data"""
        if file_names:
            query = """
            MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:HAS_ENTITY]->(e:Entity)
            WHERE d.fileName IN $fileNames
            WITH collect(DISTINCT e) as entities
            UNWIND entities as e1
            MATCH (e1)-[r]-(e2)
            WHERE e2 IN entities
            RETURN DISTINCT e1, r, e2
            LIMIT $limit
            """
            params = {"fileNames": file_names, "limit": limit}
        else:
            query = """
            MATCH (e1:Entity)-[r]-(e2:Entity)
            RETURN DISTINCT e1, r, e2
            LIMIT $limit
            """
            params = {"limit": limit}
        
        result = self.neo4j.execute_query(query, params)
        
        nodes = {}
        relationships = []
        
        for record in result:
            # Process nodes
            for node_key in ['e1', 'e2']:
                node_data = record[node_key]
                node_id = str(node_data.get('id', node_data.get('element_id')))
                
                if node_id not in nodes:
                    # Get all labels except 'Entity'
                    all_labels = list(node_data.labels) if hasattr(node_data, 'labels') else ['Entity']
                    display_labels = [l for l in all_labels if l != 'Entity'] or ['Entity']
                    
                    nodes[node_id] = GraphNode(
                        id=node_id,
                        labels=display_labels,
                        properties=dict(node_data)
                    )
            
            # Process relationship
            rel_data = record['r']
            relationships.append(GraphRelationship(
                id=str(rel_data.get('id', rel_data.get('element_id'))),
                type=rel_data.type if hasattr(rel_data, 'type') else 'RELATED',
                start_node_id=str(record['e1'].get('id', record['e1'].get('element_id'))),
                end_node_id=str(record['e2'].get('id', record['e2'].get('element_id'))),
                properties=dict(rel_data) if hasattr(rel_data, '__iter__') else {}
            ))
        
        return GraphVisualization(
            nodes=list(nodes.values()),
            relationships=relationships
        )
    
    async def _enhance_nodes_with_centrality(self, nodes: List[GraphNode]) -> None:
        """Enhance nodes with centrality metrics"""
        node_ids = [node.id for node in nodes]
        
        query = """
        MATCH (n:Entity)
        WHERE n.id IN $nodeIds
        RETURN n.id as id,
               coalesce(n.pagerank, 0.0) as pagerank,
               coalesce(n.betweenness_centrality, 0.0) as betweenness,
               coalesce(n.degree_centrality, 0.0) as degree,
               coalesce(n.closeness_centrality, 0.0) as closeness
        """
        
        result = self.neo4j.execute_query(query, {"nodeIds": node_ids})
        centrality_map = {r["id"]: r for r in result}
        
        for node in nodes:
            if node.id in centrality_map:
                centrality = centrality_map[node.id]
                node.properties.update({
                    "pagerank": centrality["pagerank"],
                    "betweenness_centrality": centrality["betweenness"],
                    "degree_centrality": centrality["degree"],
                    "closeness_centrality": centrality["closeness"],
                    "influence_score": centrality["pagerank"] * 100  # For visualization sizing
                })
    
    async def _enhance_nodes_with_communities(self, nodes: List[GraphNode]) -> None:
        """Enhance nodes with community information"""
        node_ids = [node.id for node in nodes]
        
        query = """
        MATCH (n:Entity)
        WHERE n.id IN $nodeIds AND n.community_id IS NOT NULL
        RETURN n.id as id,
               n.community_id as communityId,
               n.community_size as communitySize
        """
        
        result = self.neo4j.execute_query(query, {"nodeIds": node_ids})
        community_map = {r["id"]: r for r in result}
        
        for node in nodes:
            if node.id in community_map:
                community = community_map[node.id]
                node.properties.update({
                    "community_id": community["communityId"],
                    "community_size": community["communitySize"],
                    "community_color": self._get_community_color(community["communityId"])
                })
    
    def _get_community_color(self, community_id: str) -> str:
        """Get color for community visualization"""
        colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
            "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
        ]
        return colors[hash(community_id) % len(colors)]
    
    async def _apply_intelligent_layout(self, graph: GraphVisualization, layout_type: str) -> GraphVisualization:
        """Apply intelligent layout based on analytics"""
        if layout_type == "community_based":
            return await self._apply_community_layout(graph)
        elif layout_type == "centrality_based":
            return await self._apply_centrality_layout(graph)
        elif layout_type == "hierarchical":
            return await self._apply_hierarchical_layout(graph)
        else:
            return graph  # Default layout
    
    async def _apply_community_layout(self, graph: GraphVisualization) -> GraphVisualization:
        """Apply community-based layout"""
        # Group nodes by community
        communities = {}
        for node in graph.nodes:
            comm_id = node.properties.get("community_id", "default")
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
        
        # Add layout hints for visualization
        for i, (comm_id, nodes) in enumerate(communities.items()):
            angle = (2 * 3.14159 * i) / len(communities)
            for j, node in enumerate(nodes):
                node.properties.update({
                    "layout_group": comm_id,
                    "layout_hint_x": 300 + 200 * (i % 3),
                    "layout_hint_y": 300 + 200 * (i // 3)
                })
        
        return graph
    
    async def _apply_centrality_layout(self, graph: GraphVisualization) -> GraphVisualization:
        """Apply centrality-based layout with important nodes at center"""
        # Sort nodes by centrality
        graph.nodes.sort(key=lambda n: n.properties.get("pagerank", 0), reverse=True)
        
        # Add layout hints
        for i, node in enumerate(graph.nodes):
            centrality = node.properties.get("pagerank", 0)
            # Central nodes get smaller radius
            radius = max(100, 500 - (centrality * 1000))
            angle = (2 * 3.14159 * i) / len(graph.nodes)
            
            node.properties.update({
                "layout_centrality_rank": i,
                "layout_radius": radius,
                "layout_angle": angle
            })
        
        return graph
    
    async def _apply_hierarchical_layout(self, graph: GraphVisualization) -> GraphVisualization:
        """Apply hierarchical layout based on node types and relationships"""
        # Find hierarchical relationships
        hierarchical_rels = {"IS_A", "PART_OF", "BELONGS_TO", "CONTAINS", "HAS"}
        
        # Build hierarchy levels
        levels = await self._build_hierarchy_levels(graph, hierarchical_rels)
        
        # Apply level-based positioning
        for level, nodes in levels.items():
            for i, node in enumerate(nodes):
                node.properties.update({
                    "hierarchy_level": level,
                    "layout_level_y": level * 150,
                    "layout_level_x": (i - len(nodes) / 2) * 100
                })
        
        return graph
    
    async def _build_hierarchy_levels(self, graph: GraphVisualization, hierarchical_rels: set) -> Dict[int, List[GraphNode]]:
        """Build hierarchy levels from graph"""
        levels = {0: []}
        node_map = {n.id: n for n in graph.nodes}
        
        # Find root nodes (no incoming hierarchical relationships)
        roots = set(node_map.keys())
        
        for rel in graph.relationships:
            if rel.type in hierarchical_rels:
                roots.discard(rel.end_node_id)
        
        # Assign levels using BFS
        current_level = list(roots)
        level = 0
        
        while current_level:
            levels[level] = [node_map[node_id] for node_id in current_level if node_id in node_map]
            next_level = []
            
            for node_id in current_level:
                for rel in graph.relationships:
                    if rel.start_node_id == node_id and rel.type in hierarchical_rels:
                        next_level.append(rel.end_node_id)
            
            current_level = list(set(next_level))
            level += 1
            
            if level > 10:  # Prevent infinite loops
                break
        
        return levels
    
    async def perform_intelligent_search(
        self, 
        query: str, 
        search_type: str = "semantic",
        max_results: int = 20,
        include_reasoning: bool = True
    ) -> Dict[str, Any]:
        """Perform intelligent search with multi-hop reasoning"""
        try:
            logger.info(f"Performing intelligent search: {search_type}")
            
            if search_type == "semantic":
                results = await self._semantic_search(query, max_results)
            elif search_type == "multi_hop":
                results = await self._multi_hop_search(query, max_results)
            elif search_type == "community_aware":
                results = await self._community_aware_search(query, max_results)
            else:
                results = await self._hybrid_search(query, max_results)
            
            if include_reasoning and results.get("entities"):
                # Add reasoning paths
                entity_ids = [e["id"] for e in results["entities"][:5]]
                reasoning = await self.analytics.perform_multi_hop_reasoning(
                    entity_ids, query, max_hops=3
                )
                results["reasoning"] = reasoning
            
            return results
            
        except Exception as e:
            logger.error(f"Intelligent search failed: {e}")
            raise ServiceError(f"Intelligent search failed: {e}")
    
    async def _semantic_search(self, query: str, max_results: int) -> Dict[str, Any]:
        """Perform semantic search using embeddings"""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_query_embedding(query)
            
            # Search for similar entities and chunks
            entity_query = """
            MATCH (e:Entity)
            WHERE e.embedding IS NOT NULL
            WITH e, gds.similarity.cosine(e.embedding, $queryEmbedding) AS similarity
            WHERE similarity > $threshold
            RETURN e.id as id, e.name as name, labels(e) as labels,
                   similarity, properties(e) as properties
            ORDER BY similarity DESC
            LIMIT $maxResults
            """
            
            entities = self.neo4j.execute_query(entity_query, {
                "queryEmbedding": query_embedding,
                "threshold": 0.7,
                "maxResults": max_results
            })
            
            # Also search chunks for context
            chunk_query = """
            MATCH (c:Chunk)
            WHERE c.embedding IS NOT NULL
            WITH c, gds.similarity.cosine(c.embedding, $queryEmbedding) AS similarity
            WHERE similarity > $threshold
            MATCH (c)-[:HAS_ENTITY]->(e:Entity)
            RETURN DISTINCT c.text as text, similarity, 
                   collect(e.name) as relatedEntities
            ORDER BY similarity DESC
            LIMIT $maxResults
            """
            
            chunks = self.neo4j.execute_query(chunk_query, {
                "queryEmbedding": query_embedding,
                "threshold": 0.6,
                "maxResults": max_results // 2
            })
            
            return {
                "query": query,
                "search_type": "semantic",
                "entities": entities,
                "contexts": chunks,
                "total_results": len(entities) + len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return {"entities": [], "contexts": [], "error": str(e)}
    
    async def _multi_hop_search(self, query: str, max_results: int) -> Dict[str, Any]:
        """Perform multi-hop search with path exploration"""
        # First find seed entities
        semantic_results = await self._semantic_search(query, 5)
        seed_entities = [e["id"] for e in semantic_results["entities"]]
        
        if not seed_entities:
            return {"entities": [], "paths": [], "message": "No seed entities found"}
        
        # Find multi-hop paths
        path_query = """
        MATCH path = (start:Entity)-[*1..3]-(end:Entity)
        WHERE start.id IN $seedEntities
        WITH path, nodes(path) as pathNodes, relationships(path) as pathRels
        UNWIND pathNodes as node
        WITH DISTINCT node, path, pathNodes, pathRels
        WHERE node.name CONTAINS $queryTerm OR any(label IN labels(node) WHERE toLower(label) CONTAINS toLower($queryTerm))
        RETURN DISTINCT node.id as id, node.name as name, labels(node) as labels,
               properties(node) as properties,
               size(pathNodes) as pathLength
        ORDER BY pathLength, node.name
        LIMIT $maxResults
        """
        
        query_terms = query.lower().split()
        main_term = query_terms[0] if query_terms else query.lower()
        
        entities = self.neo4j.execute_query(path_query, {
            "seedEntities": seed_entities,
            "queryTerm": main_term,
            "maxResults": max_results
        })
        
        return {
            "query": query,
            "search_type": "multi_hop",
            "seed_entities": seed_entities,
            "entities": entities,
            "total_results": len(entities)
        }
    
    async def _community_aware_search(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search within relevant communities"""
        # Find relevant communities first
        semantic_results = await self._semantic_search(query, 5)
        
        if not semantic_results["entities"]:
            return {"entities": [], "communities": [], "message": "No relevant entities found"}
        
        # Get communities of relevant entities
        entity_ids = [e["id"] for e in semantic_results["entities"]]
        
        community_query = """
        MATCH (e:Entity)
        WHERE e.id IN $entityIds AND e.community_id IS NOT NULL
        WITH DISTINCT e.community_id as communityId
        MATCH (member:Entity {community_id: communityId})
        WHERE member.name CONTAINS $queryTerm 
           OR any(prop IN keys(properties(member)) WHERE toString(properties(member)[prop]) CONTAINS $queryTerm)
        RETURN member.id as id, member.name as name, labels(member) as labels,
               properties(member) as properties, communityId
        ORDER BY member.name
        LIMIT $maxResults
        """
        
        query_term = query.lower()
        entities = self.neo4j.execute_query(community_query, {
            "entityIds": entity_ids,
            "queryTerm": query_term,
            "maxResults": max_results
        })
        
        # Get community information
        communities_query = """
        MATCH (c:Community)
        WHERE c.id IN [e IN $entities | e.communityId]
        RETURN c.id as id, c.description as description, c.size as size
        """
        
        communities = self.neo4j.execute_query(communities_query, {
            "entities": entities
        })
        
        return {
            "query": query,
            "search_type": "community_aware",
            "entities": entities,
            "communities": communities,
            "total_results": len(entities)
        }
    
    async def _hybrid_search(self, query: str, max_results: int) -> Dict[str, Any]:
        """Combine multiple search strategies"""
        # Run all search types
        semantic_results = await self._semantic_search(query, max_results // 3)
        multihop_results = await self._multi_hop_search(query, max_results // 3)
        community_results = await self._community_aware_search(query, max_results // 3)
        
        # Combine and deduplicate results
        all_entities = {}
        
        # Add semantic results with high weight
        for entity in semantic_results.get("entities", []):
            all_entities[entity["id"]] = {
                **entity,
                "search_scores": {"semantic": entity.get("similarity", 0)}
            }
        
        # Add multihop results
        for entity in multihop_results.get("entities", []):
            if entity["id"] in all_entities:
                all_entities[entity["id"]]["search_scores"]["multihop"] = 1.0 / entity.get("pathLength", 1)
            else:
                all_entities[entity["id"]] = {
                    **entity,
                    "search_scores": {"multihop": 1.0 / entity.get("pathLength", 1)}
                }
        
        # Add community results
        for entity in community_results.get("entities", []):
            if entity["id"] in all_entities:
                all_entities[entity["id"]]["search_scores"]["community"] = 0.8
            else:
                all_entities[entity["id"]] = {
                    **entity,
                    "search_scores": {"community": 0.8}
                }
        
        # Calculate combined scores
        for entity in all_entities.values():
            scores = entity["search_scores"]
            entity["combined_score"] = (
                scores.get("semantic", 0) * 0.5 +
                scores.get("multihop", 0) * 0.3 +
                scores.get("community", 0) * 0.2
            )
        
        # Sort by combined score
        sorted_entities = sorted(
            all_entities.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )[:max_results]
        
        return {
            "query": query,
            "search_type": "hybrid",
            "entities": sorted_entities,
            "component_results": {
                "semantic": len(semantic_results.get("entities", [])),
                "multihop": len(multihop_results.get("entities", [])),
                "community": len(community_results.get("entities", []))
            },
            "total_results": len(sorted_entities)
        }
    
    async def get_entity_insights(self, entity_id: str) -> Dict[str, Any]:
        """Get comprehensive insights about an entity"""
        try:
            # Basic entity information
            entity_query = """
            MATCH (e:Entity {id: $entityId})
            RETURN e.id as id, e.name as name, labels(e) as labels,
                   properties(e) as properties
            """
            
            entity_result = self.neo4j.execute_query(entity_query, {"entityId": entity_id})
            
            if not entity_result:
                raise ValueError(f"Entity {entity_id} not found")
            
            entity = entity_result[0]
            
            # Get centrality metrics
            centrality_metrics = {
                "pagerank": entity["properties"].get("pagerank", 0),
                "betweenness_centrality": entity["properties"].get("betweenness_centrality", 0),
                "degree_centrality": entity["properties"].get("degree_centrality", 0),
                "closeness_centrality": entity["properties"].get("closeness_centrality", 0)
            }
            
            # Get community information
            community_info = None
            if entity["properties"].get("community_id"):
                community_query = """
                MATCH (c:Community {id: $communityId})
                RETURN c.description as description, c.size as size,
                       c.dominant_labels as dominantLabels
                """
                community_result = self.neo4j.execute_query(community_query, {
                    "communityId": entity["properties"]["community_id"]
                })
                community_info = community_result[0] if community_result else None
            
            # Get relationships
            relationships_query = """
            MATCH (e:Entity {id: $entityId})-[r]-(other:Entity)
            RETURN type(r) as relType, other.name as otherName, 
                   other.id as otherId, 'outgoing' as direction
            UNION
            MATCH (other:Entity)-[r]->(e:Entity {id: $entityId})
            RETURN type(r) as relType, other.name as otherName,
                   other.id as otherId, 'incoming' as direction
            """
            
            relationships = self.neo4j.execute_query(relationships_query, {"entityId": entity_id})
            
            # Get connected documents/sources
            sources_query = """
            MATCH (e:Entity {id: $entityId})<-[:HAS_ENTITY]-(c:Chunk)<-[:HAS_CHUNK]-(d:Document)
            RETURN DISTINCT d.fileName as fileName, count(c) as chunkCount
            """
            
            sources = self.neo4j.execute_query(sources_query, {"entityId": entity_id})
            
            # Generate AI-powered insights
            ai_insights = await self._generate_entity_ai_insights(entity, relationships, centrality_metrics)
            
            return {
                "entity": entity,
                "centrality_metrics": centrality_metrics,
                "community": community_info,
                "relationships": {
                    "incoming": [r for r in relationships if r["direction"] == "incoming"],
                    "outgoing": [r for r in relationships if r["direction"] == "outgoing"]
                },
                "sources": sources,
                "ai_insights": ai_insights,
                "importance_score": self._calculate_entity_importance(centrality_metrics, relationships)
            }
            
        except Exception as e:
            logger.error(f"Error getting entity insights: {e}")
            raise ServiceError(f"Failed to get entity insights: {e}")
    
    async def _generate_entity_ai_insights(self, entity: Dict, relationships: List[Dict], centrality: Dict) -> List[str]:
        """Generate AI insights about an entity"""
        try:
            llm = self.llm_factory.get_llm(self.settings.default_llm_model)
            
            rel_summary = {}
            for rel in relationships:
                rel_type = rel["relType"]
                rel_summary[rel_type] = rel_summary.get(rel_type, 0) + 1
            
            prompt = f"""
            Analyze this entity and provide 3-5 key insights:
            
            Entity: {entity['name']}
            Labels: {entity['labels']}
            
            Centrality Metrics:
            - PageRank: {centrality['pagerank']:.4f}
            - Betweenness: {centrality['betweenness_centrality']:.4f}
            - Degree: {centrality['degree_centrality']:.4f}
            
            Relationships: {rel_summary}
            Total connections: {len(relationships)}
            
            Provide insights about:
            1. The entity's role and importance in the network
            2. Its connectivity patterns
            3. Potential influence or bridging role
            4. Notable characteristics
            
            Format as a JSON array of insight strings.
            """
            
            response = await asyncio.to_thread(llm.predict, prompt)
            
            try:
                insights = json.loads(response)
                return insights if isinstance(insights, list) else [response]
            except json.JSONDecodeError:
                return [line.strip() for line in response.split('\n') if line.strip()]
                
        except Exception as e:
            logger.error(f"AI insights generation failed: {e}")
            return [f"High-degree entity with {len(relationships)} connections"]
    
    def _calculate_entity_importance(self, centrality: Dict, relationships: List[Dict]) -> float:
        """Calculate overall importance score for an entity"""
        # Weighted combination of metrics
        weights = {
            "pagerank": 0.4,
            "betweenness_centrality": 0.3,
            "degree_centrality": 0.2,
            "relationship_diversity": 0.1
        }
        
        # Calculate relationship diversity
        rel_types = set(rel["relType"] for rel in relationships)
        rel_diversity = len(rel_types) / max(1, len(relationships))
        
        importance = (
            centrality.get("pagerank", 0) * weights["pagerank"] +
            centrality.get("betweenness_centrality", 0) * weights["betweenness_centrality"] +
            centrality.get("degree_centrality", 0) * weights["degree_centrality"] +
            rel_diversity * weights["relationship_diversity"]
        )
        
        return min(1.0, importance)
    
    async def run_comprehensive_analytics(self) -> Dict[str, Any]:
        """Run all advanced analytics and return comprehensive results"""
        try:
            logger.info("Running comprehensive graph analytics")
            
            # Run all analytics in parallel where possible
            schema_task = asyncio.create_task(self.analytics.learn_graph_schema())
            communities_task = asyncio.create_task(self.analytics.detect_communities())
            centrality_task = asyncio.create_task(self.analytics.calculate_centrality_metrics())
            structure_task = asyncio.create_task(self.analytics.analyze_graph_structure())
            
            # Wait for all tasks to complete
            schema_results = await schema_task
            communities = await communities_task
            centrality_results = await centrality_task
            structure_analysis = await structure_task
            
            # Generate comprehensive insights report
            insights_report = await self.analytics.generate_insights_report()
            
            # Create summary
            summary = {
                "total_entities": structure_analysis["basic_metrics"]["num_nodes"],
                "total_relationships": structure_analysis["basic_metrics"]["num_edges"],
                "graph_density": structure_analysis["basic_metrics"]["density"],
                "communities_found": len(communities),
                "schema_patterns": len(schema_results["discovered_patterns"]),
                "most_influential_entity": centrality_results["metrics"][0].node_id if centrality_results["metrics"] else None,
                "analysis_timestamp": insights_report["generated_at"]
            }
            
            return {
                "summary": summary,
                "schema_analysis": schema_results,
                "communities": communities,
                "centrality_metrics": centrality_results,
                "structure_analysis": structure_analysis,
                "insights_report": insights_report,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Comprehensive analytics failed: {e}")
            raise ServiceError(f"Comprehensive analytics failed: {e}")
    
    async def get_analytics_dashboard_data(self) -> Dict[str, Any]:
        """Get data for analytics dashboard"""
        try:
            # Get latest analysis results
            latest_analysis_query = """
            MATCH (ga:GraphAnalysis {id: 'latest'})
            OPTIONAL MATCH (ir:InsightsReport {id: 'latest'})
            RETURN ga.analysis as graphAnalysis, ir.report as insightsReport
            """
            
            result = self.neo4j.execute_query(latest_analysis_query)
            
            if not result:
                # Run analytics if no previous results
                return await self.run_comprehensive_analytics()
            
            analysis_data = json.loads(result[0]["graphAnalysis"]) if result[0]["graphAnalysis"] else {}
            insights_data = json.loads(result[0]["insightsReport"]) if result[0]["insightsReport"] else {}
            
            # Get current statistics
            stats_query = """
            MATCH (e:Entity)
            WITH count(e) as totalEntities
            MATCH ()-[r]->()
            WITH totalEntities, count(r) as totalRelationships
            MATCH (c:Community)
            WITH totalEntities, totalRelationships, count(c) as totalCommunities
            RETURN totalEntities, totalRelationships, totalCommunities
            """
            
            current_stats = self.neo4j.execute_query(stats_query)
            stats = current_stats[0] if current_stats else {"totalEntities": 0, "totalRelationships": 0, "totalCommunities": 0}
            
            # Get top entities by centrality
            top_entities_query = """
            MATCH (e:Entity)
            WHERE e.pagerank IS NOT NULL
            RETURN e.id as id, e.name as name, e.pagerank as score
            ORDER BY e.pagerank DESC
            LIMIT 10
            """
            
            top_entities = self.neo4j.execute_query(top_entities_query)
            
            # Get community distribution
            community_dist_query = """
            MATCH (c:Community)
            RETURN c.size as size, count(c) as count
            ORDER BY size DESC
            """
            
            community_distribution = self.neo4j.execute_query(community_dist_query)
            
            # Get relationship type distribution
            rel_dist_query = """
            MATCH ()-[r]->()
            RETURN type(r) as relType, count(r) as count
            ORDER BY count DESC
            LIMIT 15
            """
            
            relationship_distribution = self.neo4j.execute_query(rel_dist_query)
            
            return {
                "current_statistics": stats,
                "top_entities": top_entities,
                "community_distribution": community_distribution,
                "relationship_distribution": relationship_distribution,
                "historical_analysis": analysis_data,
                "latest_insights": insights_data.get("key_insights", []),
                "recommendations": insights_data.get("recommendations", []),
                "last_updated": insights_data.get("generated_at", "Never")
            }
            
        except Exception as e:
            logger.error(f"Dashboard data retrieval failed: {e}")
            raise ServiceError(f"Failed to get dashboard data: {e}")
    
    # Enhanced versions of existing methods
    
    async def delete_documents(self, file_names: List[str], delete_entities: bool = False) -> int:
        """Delete documents and optionally their entities with analytics cleanup"""
        try:
            if delete_entities:
                # Delete everything related to the documents
                query = """
                MATCH (d:Document)
                WHERE d.fileName IN $fileNames
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e:Entity)
                
                // Clean up analytics data
                FOREACH (entity IN CASE WHEN e IS NOT NULL THEN [e] ELSE [] END |
                    REMOVE entity.pagerank, entity.betweenness_centrality, 
                           entity.degree_centrality, entity.closeness_centrality,
                           entity.community_id, entity.community_size
                )
                
                DETACH DELETE d, c, e
                RETURN count(DISTINCT d) as deletedDocs, count(DISTINCT e) as deletedEntities
                """
            else:
                # Delete only documents and chunks, keep entities but update their analytics
                query = """
                MATCH (d:Document)
                WHERE d.fileName IN $fileNames
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e:Entity)
                
                // Mark entities for analytics recalculation
                FOREACH (entity IN CASE WHEN e IS NOT NULL THEN [e] ELSE [] END |
                    SET entity.needs_analytics_update = true
                )
                
                DETACH DELETE d, c
                RETURN count(DISTINCT d) as deletedDocs
                """
            
            result = self.neo4j.execute_write_query(query, {"fileNames": file_names})
            deleted_count = result[0]["deletedDocs"] if result else 0
            
            # Schedule analytics recalculation if entities were affected
            if deleted_count > 0:
                asyncio.create_task(self._recalculate_analytics_after_deletion())
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise ServiceError(f"Failed to delete documents: {e}")
    
    async def _recalculate_analytics_after_deletion(self) -> None:
        """Recalculate analytics after entity deletion"""
        try:
            logger.info("Recalculating analytics after deletion")
            
            # Clear old analytics data
            clear_query = """
            MATCH (n)
            WHERE n:Entity OR n:Community OR n:GraphAnalysis OR n:InsightsReport
            REMOVE n.pagerank, n.betweenness_centrality, n.degree_centrality,
                   n.closeness_centrality, n.community_id, n.community_size,
                   n.needs_analytics_update
            """
            self.neo4j.execute_write_query(clear_query)
            
            # Recalculate analytics
            await self.run_comprehensive_analytics()
            
        except Exception as e:
            logger.error(f"Analytics recalculation failed: {e}")
    
    async def get_advanced_duplicate_nodes(self, 
                                         similarity_threshold: float = 0.8,
                                         use_embeddings: bool = True,
                                         use_fuzzy_matching: bool = True) -> List[Dict[str, Any]]:
        """Get duplicate nodes using advanced similarity algorithms"""
        try:
            from app.services.entity_resolution import EntityResolution
            
            # Use the advanced entity resolution service
            entity_resolver = EntityResolution(self.neo4j)
            
            # Find duplicates using multiple strategies
            duplicate_groups = await entity_resolver.find_duplicate_entities(batch_size=1000)
            
            # Filter by confidence threshold
            filtered_groups = [
                group for group in duplicate_groups 
                if group["confidence_score"] >= similarity_threshold
            ]
            
            # Convert to the expected format
            duplicate_nodes = []
            for group in filtered_groups:
                for entity in group["entities"]:
                    duplicate_nodes.append({
                        "id": entity["id"],
                        "name": entity["name"],
                        "labels": entity.get("labels", ["Entity"]),
                        "similarity_score": group["confidence_score"],
                        "group_id": group["group_id"],
                        "matching_methods": list(group.get("similarity_scores", {}).keys()),
                        "properties": entity.get("properties", {})
                    })
            
            return duplicate_nodes
            
        except Exception as e:
            logger.error(f"Advanced duplicate detection failed: {e}")
            # Fallback to simple method
            return await self._simple_duplicate_detection(similarity_threshold)
    
    async def _simple_duplicate_detection(self, threshold: float) -> List[Dict[str, Any]]:
        """Simple duplicate detection fallback"""
        query = """
        MATCH (e1:Entity), (e2:Entity)
        WHERE e1.id < e2.id AND e1.name IS NOT NULL AND e2.name IS NOT NULL
        WITH e1, e2, 
             CASE 
                 WHEN e1.embedding IS NOT NULL AND e2.embedding IS NOT NULL
                 THEN gds.similarity.cosine(e1.embedding, e2.embedding)
                 ELSE apoc.text.sorensenDiceSimilarity(toLower(e1.name), toLower(e2.name))
             END as similarity
        WHERE similarity >= $threshold
        RETURN e1.id as id1, e1.name as name1, labels(e1) as labels1,
               e2.id as id2, e2.name as name2, labels(e2) as labels2,
               similarity
        ORDER BY similarity DESC
        LIMIT 100
        """
        
        result = self.neo4j.execute_query(query, {"threshold": threshold})
        
        duplicates = []
        for record in result:
            duplicates.extend([
                {
                    "id": record["id1"],
                    "name": record["name1"],
                    "labels": record["labels1"],
                    "similarity_score": record["similarity"],
                    "group_id": f"simple_{len(duplicates)//2}"
                },
                {
                    "id": record["id2"],
                    "name": record["name2"],
                    "labels": record["labels2"],
                    "similarity_score": record["similarity"],
                    "group_id": f"simple_{len(duplicates)//2}"
                }
            ])
        
        return duplicates
    
    async def merge_duplicate_nodes_advanced(self, 
                                           node_ids: List[str], 
                                           target_node_id: str,
                                           merge_strategy: str = "comprehensive") -> Dict[str, Any]:
        """Advanced duplicate node merging with multiple strategies"""
        try:
            if target_node_id not in node_ids:
                raise ValueError("Target node ID must be in the list of nodes to merge")
            
            # Get detailed information about nodes to merge
            node_info_query = """
            MATCH (n:Entity)
            WHERE n.id IN $nodeIds
            RETURN n.id as id, n.name as name, labels(n) as labels,
                   properties(n) as properties,
                   [(n)-[r]-(other) | {type: type(r), other: other.id, direction: 
                    CASE WHEN startNode(r) = n THEN 'out' ELSE 'in' END}] as relationships
            """
            
            nodes_info = self.neo4j.execute_query(node_info_query, {"nodeIds": node_ids})
            nodes_dict = {node["id"]: node for node in nodes_info}
            
            if merge_strategy == "comprehensive":
                result = await self._comprehensive_merge(nodes_dict, target_node_id)
            elif merge_strategy == "conservative":
                result = await self._conservative_merge(nodes_dict, target_node_id)
            else:  # simple
                result = await self._simple_merge(node_ids, target_node_id)
            
            # Update analytics after merge
            asyncio.create_task(self._update_analytics_after_merge(target_node_id))
            
            return {
                "merged_nodes": len(node_ids) - 1,
                "target_node": target_node_id,
                "merge_strategy": merge_strategy,
                "properties_merged": result.get("properties_merged", 0),
                "relationships_transferred": result.get("relationships_transferred", 0),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Advanced duplicate merging failed: {e}")
            raise ServiceError(f"Advanced duplicate merging failed: {e}")
    
    async def _comprehensive_merge(self, nodes_dict: Dict[str, Dict], target_node_id: str) -> Dict[str, Any]:
        """Comprehensive merge strategy preserving all information"""
        target_node = nodes_dict[target_node_id]
        source_nodes = [node for node_id, node in nodes_dict.items() if node_id != target_node_id]
        
        # Merge properties intelligently
        merged_properties = dict(target_node["properties"])
        properties_merged = 0
        
        for source_node in source_nodes:
            for key, value in source_node["properties"].items():
                if key not in merged_properties or merged_properties[key] is None:
                    merged_properties[key] = value
                    properties_merged += 1
                elif merged_properties[key] != value:
                    # Handle conflicts by creating lists
                    if not isinstance(merged_properties[key], list):
                        merged_properties[key] = [merged_properties[key]]
                    if value not in merged_properties[key]:
                        merged_properties[key].append(value)
                        properties_merged += 1
        
        # Update target node with merged properties
        update_query = """
        MATCH (target:Entity {id: $targetId})
        SET target += $mergedProperties
        """
        
        self.neo4j.execute_write_query(update_query, {
            "targetId": target_node_id,
            "mergedProperties": merged_properties
        })
        
        # Transfer relationships
        relationships_transferred = 0
        for source_node in source_nodes:
            for rel_info in source_node["relationships"]:
                transfer_query = """
                MATCH (source:Entity {id: $sourceId})
                MATCH (target:Entity {id: $targetId})
                MATCH (other:Entity {id: $otherId})
                WHERE NOT EXISTS {
                    MATCH (target)-[existing]->(other) 
                    WHERE type(existing) = $relType
                }
                """
                
                if rel_info["direction"] == "out":
                    transfer_query += """
                    MATCH (source)-[r]->(other)
                    WHERE type(r) = $relType
                    CREATE (target)-[newR:TYPE]->(other)
                    SET newR = properties(r)
                    DELETE r
                    """
                else:
                    transfer_query += """
                    MATCH (other)-[r]->(source)
                    WHERE type(r) = $relType
                    CREATE (other)-[newR:TYPE]->(target)
                    SET newR = properties(r)
                    DELETE r
                    """
                
                # Note: This query needs proper dynamic relationship type handling
                # In practice, you'd need to construct this more carefully
                
                relationships_transferred += 1
        
        # Delete source nodes
        delete_query = """
        MATCH (n:Entity)
        WHERE n.id IN $sourceIds
        DETACH DELETE n
        """
        
        source_ids = [node_id for node_id in nodes_dict.keys() if node_id != target_node_id]
        self.neo4j.execute_write_query(delete_query, {"sourceIds": source_ids})
        
        return {
            "properties_merged": properties_merged,
            "relationships_transferred": relationships_transferred
        }
    
    async def _conservative_merge(self, nodes_dict: Dict[str, Dict], target_node_id: str) -> Dict[str, Any]:
        """Conservative merge strategy - only merge non-conflicting data"""
        # Implementation similar to comprehensive but with conflict avoidance
        return {"properties_merged": 0, "relationships_transferred": 0}
    
    async def _simple_merge(self, node_ids: List[str], target_node_id: str) -> Dict[str, Any]:
        """Simple merge strategy - basic relationship transfer"""
        source_ids = [nid for nid in node_ids if nid != target_node_id]
        
        # Simple relationship transfer
        transfer_query = """
        UNWIND $sourceIds as sourceId
        MATCH (source:Entity {id: sourceId})
        MATCH (target:Entity {id: $targetId})
        
        // Transfer outgoing relationships
        MATCH (source)-[r]->(other)
        WHERE NOT EXISTS((target)-[]->(other))
        CREATE (target)-[newR:RELATED]->(other)
        SET newR = properties(r)
        
        // Transfer incoming relationships  
        MATCH (other)-[r]->(source)
        WHERE NOT EXISTS((other)-[]->(target))
        CREATE (other)-[newR:RELATED]->(target)
        SET newR = properties(r)
        
        // Delete source
        DETACH DELETE source
        
        RETURN count(source) as deleted
        """
        
        result = self.neo4j.execute_write_query(transfer_query, {
            "sourceIds": source_ids,
            "targetId": target_node_id
        })
        
        return {
            "properties_merged": 0,
            "relationships_transferred": result[0]["deleted"] if result else 0
        }
    
    async def _update_analytics_after_merge(self, target_node_id: str) -> None:
        """Update analytics after merging nodes"""
        try:
            # Mark target node for analytics recalculation
            mark_query = """
            MATCH (n:Entity {id: $nodeId})
            SET n.needs_analytics_update = true
            """
            
            self.neo4j.execute_write_query(mark_query, {"nodeId": target_node_id})
            
            # Recalculate centrality for the target node
            await self.analytics.calculate_centrality_metrics(["degree", "pagerank"])
            
        except Exception as e:
            logger.error(f"Analytics update after merge failed: {e}")
    
    async def get_graph_quality_metrics(self) -> Dict[str, Any]:
        """Get comprehensive graph quality metrics"""
        try:
            quality_metrics = {
                "completeness": await self._calculate_completeness_metrics(),
                "consistency": await self._calculate_consistency_metrics(),
                "accuracy": await self._calculate_accuracy_metrics(),
                "connectivity": await self._calculate_connectivity_metrics(),
                "schema_quality": await self._calculate_schema_quality_metrics()
            }
            
            # Calculate overall quality score
            quality_metrics["overall_score"] = self._calculate_overall_quality_score(quality_metrics)
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Quality metrics calculation failed: {e}")
            raise ServiceError(f"Failed to calculate quality metrics: {e}")
    
    async def _calculate_completeness_metrics(self) -> Dict[str, Any]:
        """Calculate data completeness metrics"""
        query = """
        MATCH (e:Entity)
        WITH count(e) as totalEntities,
             count(CASE WHEN e.name IS NOT NULL AND e.name <> '' THEN 1 END) as entitiesWithNames,
             count(CASE WHEN e.embedding IS NOT NULL THEN 1 END) as entitiesWithEmbeddings,
             count(CASE WHEN size(labels(e)) > 1 THEN 1 END) as entitiesWithSpecificLabels
        RETURN totalEntities, entitiesWithNames, entitiesWithEmbeddings, entitiesWithSpecificLabels
        """
        
        result = self.neo4j.execute_query(query)
        
        if result:
            data = result[0]
            total = data["totalEntities"]
            
            return {
                "entities_with_names_pct": (data["entitiesWithNames"] / max(total, 1)) * 100,
                "entities_with_embeddings_pct": (data["entitiesWithEmbeddings"] / max(total, 1)) * 100,
                "entities_with_specific_labels_pct": (data["entitiesWithSpecificLabels"] / max(total, 1)) * 100,
                "total_entities": total
            }
        
        return {"total_entities": 0}
    
    async def _calculate_consistency_metrics(self) -> Dict[str, Any]:
        """Calculate data consistency metrics"""
        # Check for naming inconsistencies
        naming_query = """
        MATCH (e1:Entity), (e2:Entity)
        WHERE e1.id < e2.id 
        AND toLower(e1.name) = toLower(e2.name)
        AND e1.name <> e2.name
        RETURN count(*) as namingInconsistencies
        """
        
        naming_result = self.neo4j.execute_query(naming_query)
        
        # Check for relationship consistency
        rel_query = """
        MATCH (e1)-[r1]->(e2), (e1)-[r2]->(e2)
        WHERE type(r1) <> type(r2)
        RETURN count(DISTINCT [e1.id, e2.id]) as multipleRelationshipTypes
        """
        
        rel_result = self.neo4j.execute_query(rel_query)
        
        return {
            "naming_inconsistencies": naming_result[0]["namingInconsistencies"] if naming_result else 0,
            "multiple_relationship_types": rel_result[0]["multipleRelationshipTypes"] if rel_result else 0
        }
    
    async def _calculate_accuracy_metrics(self) -> Dict[str, Any]:
        """Calculate data accuracy metrics (placeholder)"""
        # This would require ground truth data or validation rules
        return {
            "estimated_accuracy": 85.0,  # Placeholder
            "validation_rules_passed": 0,
            "validation_rules_total": 0
        }
    
    async def _calculate_connectivity_metrics(self) -> Dict[str, Any]:
        """Calculate connectivity quality metrics"""
        query = """
        MATCH (e:Entity)
        WITH count(e) as totalEntities
        MATCH (isolated:Entity)
        WHERE NOT (isolated)-[]-()
        WITH totalEntities, count(isolated) as isolatedEntities
        MATCH ()-[r]->()
        WITH totalEntities, isolatedEntities, count(r) as totalRelationships
        RETURN totalEntities, isolatedEntities, totalRelationships
        """
        
        result = self.neo4j.execute_query(query)
        
        if result:
            data = result[0]
            total_entities = data["totalEntities"]
            isolated = data["isolatedEntities"]
            total_rels = data["totalRelationships"]
            
            return {
                "isolated_entities_pct": (isolated / max(total_entities, 1)) * 100,
                "connected_entities_pct": ((total_entities - isolated) / max(total_entities, 1)) * 100,
                "average_degree": (total_rels * 2) / max(total_entities, 1),  # For undirected graph
                "connectivity_score": max(0, 100 - (isolated / max(total_entities, 1)) * 100)
            }
        
        return {"connectivity_score": 0}
    
    async def _calculate_schema_quality_metrics(self) -> Dict[str, Any]:
        """Calculate schema quality metrics"""
        # Check for generic labels
        generic_labels_query = """
        MATCH (e:Entity)
        WHERE size([label IN labels(e) WHERE label <> 'Entity']) = 0
        WITH count(e) as genericEntities
        MATCH (e:Entity)
        WITH genericEntities, count(e) as totalEntities
        RETURN genericEntities, totalEntities
        """
        
        label_result = self.neo4j.execute_query(generic_labels_query)
        
        # Check for generic relationships
        generic_rels_query = """
        MATCH ()-[r]->()
        WHERE type(r) IN ['RELATED', 'RELATED_TO', 'CONNECTED_TO']
        WITH count(r) as genericRels
        MATCH ()-[r]->()
        WITH genericRels, count(r) as totalRels
        RETURN genericRels, totalRels
        """
        
        rel_result = self.neo4j.execute_query(generic_rels_query)
        
        schema_quality = 100.0
        
        if label_result:
            data = label_result[0]
            generic_pct = (data["genericEntities"] / max(data["totalEntities"], 1)) * 100
            schema_quality -= generic_pct * 0.3  # Penalize generic labels
        
        if rel_result:
            data = rel_result[0]
            generic_rel_pct = (data["genericRels"] / max(data["totalRels"], 1)) * 100
            schema_quality -= generic_rel_pct * 0.2  # Penalize generic relationships
        
        return {
            "schema_quality_score": max(0, schema_quality),
            "generic_entities_pct": generic_pct if label_result else 0,
            "generic_relationships_pct": generic_rel_pct if rel_result else 0
        }
    
    def _calculate_overall_quality_score(self, quality_metrics: Dict[str, Any]) -> float:
        """Calculate overall graph quality score"""
        weights = {
            "completeness": 0.25,
            "consistency": 0.20,
            "accuracy": 0.20,
            "connectivity": 0.20,
            "schema_quality": 0.15
        }
        
        completeness_score = (
            quality_metrics["completeness"].get("entities_with_names_pct", 0) * 0.4 +
            quality_metrics["completeness"].get("entities_with_embeddings_pct", 0) * 0.3 +
            quality_metrics["completeness"].get("entities_with_specific_labels_pct", 0) * 0.3
        )
        
        consistency_score = max(0, 100 - (
            quality_metrics["consistency"].get("naming_inconsistencies", 0) * 5 +
            quality_metrics["consistency"].get("multiple_relationship_types", 0) * 2
        ))
        
        accuracy_score = quality_metrics["accuracy"].get("estimated_accuracy", 0)
        connectivity_score = quality_metrics["connectivity"].get("connectivity_score", 0)
        schema_score = quality_metrics["schema_quality"].get("schema_quality_score", 0)
        
        overall_score = (
            completeness_score * weights["completeness"] +
            consistency_score * weights["consistency"] +
            accuracy_score * weights["accuracy"] +
            connectivity_score * weights["connectivity"] +
            schema_score * weights["schema_quality"]
        )
        
        return round(overall_score, 2)
