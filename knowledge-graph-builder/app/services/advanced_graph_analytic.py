import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
import networkx as nx
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass
import json

from app.core.neo4j_client import Neo4jClient
from app.core.exceptions import ServiceError
from app.config.settings import get_settings
from app.utils.llm_clients import LLMClientFactory
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

@dataclass
class SchemaPattern:
    """Represents a discovered schema pattern"""
    source_label: str
    relationship_type: str
    target_label: str
    frequency: int
    confidence: float
    examples: List[Dict[str, Any]]

@dataclass
class Community:
    """Represents a detected community"""
    id: str
    members: List[str]
    size: int
    density: float
    central_nodes: List[str]
    dominant_labels: List[str]
    description: Optional[str] = None

@dataclass
class CentralityMetrics:
    """Centrality metrics for a node"""
    node_id: str
    degree_centrality: float
    betweenness_centrality: float
    closeness_centrality: float
    pagerank: float
    eigenvector_centrality: float

class AdvancedGraphAnalytics:
    """Advanced graph processing and analytics"""
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
        self.settings = get_settings()
        self.llm_factory = LLMClientFactory()
        self.embedding_service = EmbeddingService()
        
    async def learn_graph_schema(self, min_frequency: int = 5) -> Dict[str, Any]:
        """Automatically discover and learn graph schema patterns"""
        try:
            logger.info("Starting graph schema learning...")
            
            # 1. Analyze existing patterns
            patterns = await self._discover_schema_patterns(min_frequency)
            
            # 2. Generate schema suggestions
            suggestions = await self._generate_schema_suggestions(patterns)
            
            # 3. Validate patterns against domain knowledge
            validated_patterns = await self._validate_schema_patterns(patterns)
            
            # 4. Create schema evolution recommendations
            evolution_plan = await self._create_schema_evolution_plan(validated_patterns)
            
            schema_analysis = {
                "discovered_patterns": patterns,
                "schema_suggestions": suggestions,
                "validated_patterns": validated_patterns,
                "evolution_plan": evolution_plan,
                "statistics": await self._get_schema_statistics()
            }
            
            # Store schema learning results
            await self._store_schema_analysis(schema_analysis)
            
            return schema_analysis
            
        except Exception as e:
            logger.error(f"Schema learning failed: {e}")
            raise ServiceError(f"Schema learning failed: {e}")
    
    async def _discover_schema_patterns(self, min_frequency: int) -> List[SchemaPattern]:
        """Discover common schema patterns in the graph"""
        query = """
        MATCH (source)-[r]->(target)
        WHERE source:Entity AND target:Entity
        WITH 
            CASE WHEN size(labels(source)) > 1 
                 THEN [label IN labels(source) WHERE label <> 'Entity'][0] 
                 ELSE 'Entity' END as sourceLabel,
            type(r) as relType,
            CASE WHEN size(labels(target)) > 1 
                 THEN [label IN labels(target) WHERE label <> 'Entity'][0] 
                 ELSE 'Entity' END as targetLabel,
            source, r, target
        WITH sourceLabel, relType, targetLabel, 
             count(*) as frequency,
             collect({
                 source: properties(source), 
                 relation: properties(r), 
                 target: properties(target)
             })[0..3] as examples
        WHERE frequency >= $minFreq
        RETURN sourceLabel, relType, targetLabel, frequency, examples
        ORDER BY frequency DESC
        """
        
        result = self.neo4j.execute_query(query, {"minFreq": min_frequency})
        
        patterns = []
        for record in result:
            confidence = min(1.0, record["frequency"] / 100.0)  # Normalize confidence
            
            patterns.append(SchemaPattern(
                source_label=record["sourceLabel"],
                relationship_type=record["relType"],
                target_label=record["targetLabel"],
                frequency=record["frequency"],
                confidence=confidence,
                examples=record["examples"]
            ))
        
        logger.info(f"Discovered {len(patterns)} schema patterns")
        return patterns
    
    async def _generate_schema_suggestions(self, patterns: List[SchemaPattern]) -> Dict[str, List[str]]:
        """Generate schema improvement suggestions using LLM"""
        try:
            llm = self.llm_factory.get_llm(self.settings.default_llm_model)
            
            # Prepare pattern summary for LLM
            pattern_summary = []
            for pattern in patterns[:20]:  # Limit to top 20 patterns
                pattern_summary.append({
                    "pattern": f"{pattern.source_label} -[{pattern.relationship_type}]-> {pattern.target_label}",
                    "frequency": pattern.frequency,
                    "confidence": pattern.confidence
                })
            
            prompt = f"""
            Analyze these graph schema patterns and suggest improvements:
            
            Current Patterns:
            {json.dumps(pattern_summary, indent=2)}
            
            Based on these patterns, suggest:
            1. Missing entity types that would be valuable
            2. More specific relationship types to replace generic ones
            3. Hierarchical relationships to improve structure
            4. Properties that should be standardized
            5. Potential schema refinements
            
            Respond in JSON format:
            {{
                "missing_entities": ["EntityType1", "EntityType2"],
                "specific_relationships": [
                    {{"current": "RELATED_TO", "suggested": ["WORKS_WITH", "COLLABORATES_ON"]}},
                ],
                "hierarchical_relationships": ["IS_A", "PART_OF", "BELONGS_TO"],
                "standard_properties": {{
                    "Person": ["name", "role", "department"],
                    "Organization": ["name", "type", "industry"]
                }},
                "refinements": ["suggestion1", "suggestion2"]
            }}
            """
            
            response = await asyncio.to_thread(llm.predict, prompt)
            
            try:
                suggestions = json.loads(response)
            except json.JSONDecodeError:
                # Fallback suggestions
                suggestions = {
                    "missing_entities": [],
                    "specific_relationships": [],
                    "hierarchical_relationships": ["IS_A", "PART_OF"],
                    "standard_properties": {},
                    "refinements": []
                }
            
            return suggestions
            
        except Exception as e:
            logger.warning(f"LLM schema suggestion failed: {e}")
            return {"error": str(e)}
    
    async def _validate_schema_patterns(self, patterns: List[SchemaPattern]) -> List[SchemaPattern]:
        """Validate discovered patterns against data quality metrics"""
        validated = []
        
        for pattern in patterns:
            # Calculate pattern quality metrics
            quality_score = await self._calculate_pattern_quality(pattern)
            
            if quality_score > 0.3:  # Quality threshold
                pattern.confidence *= quality_score  # Adjust confidence
                validated.append(pattern)
        
        return sorted(validated, key=lambda p: p.confidence, reverse=True)
    
    async def _calculate_pattern_quality(self, pattern: SchemaPattern) -> float:
        """Calculate quality score for a schema pattern"""
        quality_factors = {
            "frequency_score": min(1.0, pattern.frequency / 50.0),
            "label_specificity": 1.0 if pattern.source_label != "Entity" else 0.3,
            "relationship_specificity": 1.0 if pattern.relationship_type != "RELATED_TO" else 0.5
        }
        
        # Weighted average
        weights = {"frequency_score": 0.4, "label_specificity": 0.3, "relationship_specificity": 0.3}
        
        quality_score = sum(score * weights[factor] for factor, score in quality_factors.items())
        return quality_score
    
    async def _create_schema_evolution_plan(self, patterns: List[SchemaPattern]) -> Dict[str, Any]:
        """Create a plan for evolving the current schema"""
        evolution_plan = {
            "immediate_improvements": [],
            "medium_term_goals": [],
            "long_term_vision": [],
            "implementation_steps": []
        }
        
        # Analyze current schema issues
        generic_patterns = [p for p in patterns if p.source_label == "Entity" or p.relationship_type == "RELATED_TO"]
        
        if len(generic_patterns) > len(patterns) * 0.5:
            evolution_plan["immediate_improvements"].append({
                "issue": "High percentage of generic labels and relationships",
                "action": "Implement entity type classification",
                "priority": "high"
            })
        
        # Add more evolution strategies
        evolution_plan["implementation_steps"] = [
            "1. Classify generic Entity nodes into specific types",
            "2. Replace RELATED_TO relationships with specific types",
            "3. Add hierarchical relationships (IS_A, PART_OF)",
            "4. Standardize entity properties",
            "5. Implement schema validation rules"
        ]
        
        return evolution_plan
    
    async def _get_schema_statistics(self) -> Dict[str, Any]:
        """Get comprehensive schema statistics"""
        stats_queries = {
            "entity_types": """
                MATCH (n:Entity)
                UNWIND labels(n) as label
                WITH label, count(n) as count
                WHERE label <> 'Entity'
                RETURN label, count
                ORDER BY count DESC
            """,
            "relationship_types": """
                MATCH ()-[r]->()
                RETURN type(r) as relType, count(r) as count
                ORDER BY count DESC
            """,
            "graph_density": """
                MATCH (n:Entity)
                WITH count(n) as nodeCount
                MATCH ()-[r]->()
                WITH nodeCount, count(r) as edgeCount
                RETURN toFloat(edgeCount) / (nodeCount * (nodeCount - 1)) as density
            """
        }
        
        statistics = {}
        for stat_name, query in stats_queries.items():
            try:
                result = self.neo4j.execute_query(query)
                statistics[stat_name] = result
            except Exception as e:
                logger.warning(f"Failed to compute {stat_name}: {e}")
                statistics[stat_name] = []
        
        return statistics
    
    async def _store_schema_analysis(self, analysis: Dict[str, Any]) -> None:
        """Store schema analysis results"""
        query = """
        MERGE (sa:SchemaAnalysis {id: 'latest'})
        SET sa.timestamp = datetime(),
            sa.patterns_count = $patternsCount,
            sa.analysis = $analysis
        """
        
        self.neo4j.execute_write_query(query, {
            "patternsCount": len(analysis["discovered_patterns"]),
            "analysis": json.dumps(analysis, default=str)
        })

    async def perform_multi_hop_reasoning(
        self, 
        start_entities: List[str], 
        question: str, 
        max_hops: int = 3,
        reasoning_type: str = "path_based"
    ) -> Dict[str, Any]:
        """Perform multi-hop reasoning across the knowledge graph"""
        try:
            logger.info(f"Starting multi-hop reasoning: {reasoning_type}")
            
            # 1. Find relevant paths from start entities
            paths = await self._find_reasoning_paths(start_entities, max_hops)
            
            # 2. Score paths based on relevance to question
            scored_paths = await self._score_reasoning_paths(paths, question)
            
            # 3. Extract reasoning chains
            reasoning_chains = await self._extract_reasoning_chains(scored_paths)
            
            # 4. Generate final answer using LLM
            answer = await self._generate_reasoning_answer(reasoning_chains, question)
            
            return {
                "answer": answer,
                "reasoning_paths": scored_paths[:5],  # Top 5 paths
                "reasoning_chains": reasoning_chains,
                "confidence": self._calculate_reasoning_confidence(scored_paths),
                "source_entities": start_entities,
                "hops_used": max_hops
            }
            
        except Exception as e:
            logger.error(f"Multi-hop reasoning failed: {e}")
            raise ServiceError(f"Multi-hop reasoning failed: {e}")
    
    async def _find_reasoning_paths(self, start_entities: List[str], max_hops: int) -> List[Dict[str, Any]]:
        """Find all reasoning paths from start entities"""
        query = f"""
        MATCH path = (start:Entity)-[*1..{max_hops}]-(end:Entity)
        WHERE start.id IN $startEntities
        WITH path, relationships(path) as rels, nodes(path) as pathNodes
        WHERE length(path) <= {max_hops}
        RETURN 
            [n IN pathNodes | {{id: n.id, name: n.name, labels: labels(n)}}] as nodes,
            [r IN rels | {{type: type(r), properties: properties(r)}}] as relationships,
            length(path) as pathLength
        ORDER BY pathLength
        LIMIT 1000
        """
        
        result = self.neo4j.execute_query(query, {"startEntities": start_entities})
        
        paths = []
        for record in result:
            paths.append({
                "nodes": record["nodes"],
                "relationships": record["relationships"],
                "length": record["pathLength"],
                "path_string": self._create_path_string(record["nodes"], record["relationships"])
            })
        
        return paths
    
    async def _score_reasoning_paths(self, paths: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
        """Score paths based on relevance to the question"""
        try:
            # Generate embeddings for question
            question_embedding = await self.embedding_service.generate_query_embedding(question)
            
            scored_paths = []
            for path in paths:
                # Calculate path relevance score
                path_text = path["path_string"]
                
                try:
                    path_embedding = await self.embedding_service.generate_query_embedding(path_text)
                    similarity = self.embedding_service.calculate_similarity(question_embedding, path_embedding)
                except:
                    similarity = 0.0
                
                # Additional scoring factors
                length_penalty = 0.9 ** (path["length"] - 1)  # Prefer shorter paths
                entity_diversity = len(set(n["name"] for n in path["nodes"]))
                diversity_bonus = entity_diversity * 0.1
                
                final_score = similarity * length_penalty + diversity_bonus
                
                scored_paths.append({
                    **path,
                    "relevance_score": similarity,
                    "length_penalty": length_penalty,
                    "diversity_bonus": diversity_bonus,
                    "final_score": final_score
                })
            
            return sorted(scored_paths, key=lambda p: p["final_score"], reverse=True)
            
        except Exception as e:
            logger.warning(f"Path scoring failed: {e}")
            return paths
    
    async def _extract_reasoning_chains(self, scored_paths: List[Dict[str, Any]]) -> List[str]:
        """Extract human-readable reasoning chains from paths"""
        chains = []
        
        for path in scored_paths[:10]:  # Top 10 paths
            chain_parts = []
            nodes = path["nodes"]
            relationships = path["relationships"]
            
            for i in range(len(nodes) - 1):
                source = nodes[i]["name"]
                target = nodes[i + 1]["name"]
                rel_type = relationships[i]["type"].replace("_", " ").lower()
                
                chain_parts.append(f"{source} {rel_type} {target}")
            
            chains.append(" → ".join(chain_parts))
        
        return chains
    
    def _create_path_string(self, nodes: List[Dict], relationships: List[Dict]) -> str:
        """Create a string representation of a path"""
        if not nodes:
            return ""
        
        path_parts = [nodes[0]["name"]]
        
        for i, rel in enumerate(relationships):
            if i + 1 < len(nodes):
                rel_type = rel["type"].replace("_", " ")
                target = nodes[i + 1]["name"]
                path_parts.append(f"-[{rel_type}]-> {target}")
        
        return " ".join(path_parts)
    
    async def _generate_reasoning_answer(self, reasoning_chains: List[str], question: str) -> str:
        """Generate final answer using reasoning chains"""
        try:
            llm = self.llm_factory.get_llm(self.settings.default_llm_model)
            
            chains_text = "\n".join(f"- {chain}" for chain in reasoning_chains[:5])
            
            prompt = f"""
            Question: {question}
            
            Based on these reasoning paths from the knowledge graph:
            {chains_text}
            
            Provide a comprehensive answer that:
            1. Directly addresses the question
            2. Uses evidence from the reasoning paths
            3. Explains the logical connections
            4. Acknowledges any limitations or uncertainties
            
            Answer:
            """
            
            answer = await asyncio.to_thread(llm.predict, prompt)
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "Unable to generate answer due to processing error."
    
    def _calculate_reasoning_confidence(self, scored_paths: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for reasoning result"""
        if not scored_paths:
            return 0.0
        
        # Base confidence on top path scores
        top_scores = [p["final_score"] for p in scored_paths[:5]]
        avg_score = sum(top_scores) / len(top_scores)
        
        # Adjust for path consistency
        score_variance = np.var(top_scores) if len(top_scores) > 1 else 0
        consistency_factor = max(0.5, 1.0 - score_variance)
        
        return min(1.0, avg_score * consistency_factor)

    async def detect_communities(
        self, 
        algorithm: str = "louvain",
        min_community_size: int = 3,
        resolution: float = 1.0
    ) -> List[Community]:
        """Detect communities in the knowledge graph"""
        try:
            logger.info(f"Starting community detection using {algorithm}")
            
            # 1. Build NetworkX graph from Neo4j data
            nx_graph = await self._build_networkx_graph()
            
            # 2. Apply community detection algorithm
            if algorithm.lower() == "louvain":
                communities_dict = await self._louvain_community_detection(nx_graph, resolution)
            elif algorithm.lower() == "leiden":
                communities_dict = await self._leiden_community_detection(nx_graph, resolution)
            elif algorithm.lower() == "infomap":
                communities_dict = await self._infomap_community_detection(nx_graph)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # 3. Filter communities by minimum size
            filtered_communities = {
                comm_id: members for comm_id, members in communities_dict.items()
                if len(members) >= min_community_size
            }
            
            # 4. Analyze community properties
            communities = []
            for comm_id, members in filtered_communities.items():
                community = await self._analyze_community(comm_id, members, nx_graph)
                communities.append(community)
            
            # 5. Generate community descriptions
            for community in communities:
                community.description = await self._generate_community_description(community)
            
            # 6. Store community information in Neo4j
            await self._store_communities(communities)
            
            logger.info(f"Detected {len(communities)} communities")
            return sorted(communities, key=lambda c: c.size, reverse=True)
            
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            raise ServiceError(f"Community detection failed: {e}")
    
    async def _build_networkx_graph(self) -> nx.Graph:
        """Build NetworkX graph from Neo4j data"""
        query = """
        MATCH (n1:Entity)-[r]->(n2:Entity)
        RETURN n1.id as source, n2.id as target, 
               n1.name as source_name, n2.name as target_name,
               type(r) as relationship_type,
               coalesce(r.weight, 1.0) as weight
        """
        
        result = self.neo4j.execute_query(query)
        
        # Create undirected graph for community detection
        G = nx.Graph()
        
        for record in result:
            G.add_node(record["source"], name=record["source_name"])
            G.add_node(record["target"], name=record["target_name"])
            G.add_edge(
                record["source"], 
                record["target"], 
                weight=record["weight"],
                relationship_type=record["relationship_type"]
            )
        
        logger.info(f"Built NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    async def _louvain_community_detection(self, graph: nx.Graph, resolution: float) -> Dict[str, List[str]]:
        """Perform Louvain community detection"""
        try:
            import community as community_louvain
            
            # Run Louvain algorithm
            partition = await asyncio.to_thread(
                community_louvain.best_partition, 
                graph, 
                resolution=resolution
            )
            
            # Group nodes by community
            communities = defaultdict(list)
            for node, comm_id in partition.items():
                communities[str(comm_id)].append(node)
            
            return dict(communities)
            
        except ImportError:
            logger.warning("python-louvain not available, using NetworkX alternative")
            return await self._networkx_community_detection(graph)
        except Exception as e:
            logger.error(f"Louvain detection failed: {e}")
            return await self._networkx_community_detection(graph)
    
    async def _leiden_community_detection(self, graph: nx.Graph, resolution: float) -> Dict[str, List[str]]:
        """Perform Leiden community detection"""
        try:
            import leidenalg as la
            import igraph as ig
            
            # Convert to igraph
            ig_graph = ig.Graph.from_networkx(graph)
            
            # Run Leiden algorithm
            partition = await asyncio.to_thread(
                la.find_partition,
                ig_graph,
                la.RBConfigurationVertexPartition,
                resolution_parameter=resolution
            )
            
            # Convert back to communities dict
            communities = defaultdict(list)
            for i, comm_id in enumerate(partition.membership):
                node_id = list(graph.nodes())[i]
                communities[str(comm_id)].append(node_id)
            
            return dict(communities)
            
        except ImportError:
            logger.warning("leidenalg not available, falling back to Louvain")
            return await self._louvain_community_detection(graph, resolution)
        except Exception as e:
            logger.error(f"Leiden detection failed: {e}")
            return await self._networkx_community_detection(graph)
    
    async def _infomap_community_detection(self, graph: nx.Graph) -> Dict[str, List[str]]:
        """Perform Infomap community detection"""
        try:
            import infomap
            
            # Create Infomap instance
            im = infomap.Infomap()
            
            # Add nodes and edges
            for node in graph.nodes():
                im.add_node(hash(node) % (2**31))  # Convert to int
            
            for edge in graph.edges(data=True):
                weight = edge[2].get('weight', 1.0)
                im.add_link(hash(edge[0]) % (2**31), hash(edge[1]) % (2**31), weight)
            
            # Run algorithm
            await asyncio.to_thread(im.run)
            
            # Extract communities
            communities = defaultdict(list)
            node_list = list(graph.nodes())
            
            for node in im.tree:
                if node.is_leaf:
                    original_node = node_list[node.node_id % len(node_list)]
                    communities[str(node.module_id)].append(original_node)
            
            return dict(communities)
            
        except ImportError:
            logger.warning("infomap not available, falling back to NetworkX")
            return await self._networkx_community_detection(graph)
        except Exception as e:
            logger.error(f"Infomap detection failed: {e}")
            return await self._networkx_community_detection(graph)
    
    async def _networkx_community_detection(self, graph: nx.Graph) -> Dict[str, List[str]]:
        """Fallback community detection using NetworkX"""
        try:
            # Use Girvan-Newman algorithm
            communities_gen = nx.community.girvan_newman(graph)
            communities_list = next(communities_gen)
            
            communities = {}
            for i, community_set in enumerate(communities_list):
                communities[str(i)] = list(community_set)
            
            return communities
            
        except Exception as e:
            logger.error(f"NetworkX community detection failed: {e}")
            # Return single community as fallback
            return {"0": list(graph.nodes())}
    
    async def _analyze_community(self, comm_id: str, members: List[str], graph: nx.Graph) -> Community:
        """Analyze properties of a detected community"""
        subgraph = graph.subgraph(members)
        
        # Calculate community metrics
        size = len(members)
        
        # Calculate density
        possible_edges = size * (size - 1) / 2
        actual_edges = subgraph.number_of_edges()
        density = actual_edges / possible_edges if possible_edges > 0 else 0.0
        
        # Find central nodes (highest degree in subgraph)
        degrees = dict(subgraph.degree())
        central_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:3]
        
        # Get dominant labels from Neo4j
        dominant_labels = await self._get_community_labels(members)
        
        return Community(
            id=comm_id,
            members=members,
            size=size,
            density=density,
            central_nodes=central_nodes,
            dominant_labels=dominant_labels
        )
    
    async def _get_community_labels(self, member_ids: List[str]) -> List[str]:
        """Get dominant entity labels in a community"""
        query = """
        MATCH (n:Entity)
        WHERE n.id IN $memberIds
        UNWIND labels(n) as label
        WITH label, count(*) as count
        WHERE label <> 'Entity'
        RETURN label, count
        ORDER BY count DESC
        LIMIT 5
        """
        
        result = self.neo4j.execute_query(query, {"memberIds": member_ids})
        return [record["label"] for record in result]
    
    async def _generate_community_description(self, community: Community) -> str:
        """Generate natural language description of a community"""
        try:
            # Get sample entities from community
            query = """
            MATCH (n:Entity)
            WHERE n.id IN $memberIds
            RETURN n.name as name, labels(n) as labels
            LIMIT 10
            """
            
            members_info = self.neo4j.execute_query(query, {"memberIds": community.members[:10]})
            
            member_names = [info["name"] for info in members_info]
            
            llm = self.llm_factory.get_llm(self.settings.default_llm_model)
            
            prompt = f"""
            Describe this community of entities in 1-2 sentences:
            
            Community size: {community.size} entities
            Density: {community.density:.2f}
            Dominant types: {', '.join(community.dominant_labels)}
            Key entities: {', '.join(member_names[:5])}
            Central nodes: {', '.join(community.central_nodes[:3])}
            
            Description:
            """
            
            description = await asyncio.to_thread(llm.predict, prompt)
            return description.strip()
            
        except Exception as e:
            logger.error(f"Community description generation failed: {e}")
            return f"Community of {community.size} entities with {community.density:.2f} density"
    
    async def _store_communities(self, communities: List[Community]) -> None:
        """Store community information in Neo4j"""
        # First, clear existing community assignments
        clear_query = """
        MATCH (n:Entity)
        REMOVE n.community_id, n.community_size
        """
        self.neo4j.execute_write_query(clear_query)
        
        # Store community information
        for community in communities:
            # Update community members
            update_query = """
            MATCH (n:Entity)
            WHERE n.id IN $memberIds
            SET n.community_id = $communityId,
                n.community_size = $communitySize
            """
            
            self.neo4j.execute_write_query(update_query, {
                "memberIds": community.members,
                "communityId": community.id,
                "communitySize": community.size
            })
            
            # Create community summary node
            summary_query = """
            MERGE (c:Community {id: $communityId})
            SET c.size = $size,
                c.density = $density,
                c.description = $description,
                c.dominant_labels = $dominantLabels,
                c.central_nodes = $centralNodes,
                c.detected_at = datetime()
            """
            
            self.neo4j.execute_write_query(summary_query, {
                "communityId": community.id,
                "size": community.size,
                "density": community.density,
                "description": community.description,
                "dominantLabels": community.dominant_labels,
                "centralNodes": community.central_nodes
            })

    async def calculate_centrality_metrics(self, algorithms: List[str] = None) -> Dict[str, List[CentralityMetrics]]:
        """Calculate various centrality metrics for all nodes"""
        try:
            if algorithms is None:
                algorithms = ["degree", "betweenness", "closeness", "pagerank", "eigenvector"]
            
            logger.info(f"Calculating centrality metrics: {algorithms}")
            
            # Build NetworkX graph
            nx_graph = await self._build_networkx_graph()
            
            if nx_graph.number_of_nodes() == 0:
                return {"metrics": [], "summary": {}}
            
            # Calculate each centrality metric
            centrality_results = {}
            
            if "degree" in algorithms:
                centrality_results["degree"] = await self._calculate_degree_centrality(nx_graph)
            
            if "betweenness" in algorithms:
                centrality_results["betweenness"] = await self._calculate_betweenness_centrality(nx_graph)
            
            if "closeness" in algorithms:
                centrality_results["closeness"] = await self._calculate_closeness_centrality(nx_graph)
            
            if "pagerank" in algorithms:
                centrality_results["pagerank"] = await self._calculate_pagerank(nx_graph)
            
            if "eigenvector" in algorithms:
                centrality_results["eigenvector"] = await self._calculate_eigenvector_centrality(nx_graph)
            
            # Combine metrics into CentralityMetrics objects
            all_nodes = set(nx_graph.nodes())
            centrality_metrics = []
            
            for node_id in all_nodes:
                metrics = CentralityMetrics(
                    node_id=node_id,
                    degree_centrality=centrality_results.get("degree", {}).get(node_id, 0.0),
                    betweenness_centrality=centrality_results.get("betweenness", {}).get(node_id, 0.0),
                    closeness_centrality=centrality_results.get("closeness", {}).get(node_id, 0.0),
                    pagerank=centrality_results.get("pagerank", {}).get(node_id, 0.0),
                    eigenvector_centrality=centrality_results.get("eigenvector", {}).get(node_id, 0.0)
                )
                centrality_metrics.append(metrics)
            
            # Store centrality metrics in Neo4j
            await self._store_centrality_metrics(centrality_metrics)
            
            # Generate summary statistics
            summary = await self._generate_centrality_summary(centrality_metrics)
            
            # Sort by PageRank (most commonly used for ranking)
            centrality_metrics.sort(key=lambda x: x.pagerank, reverse=True)
            
            return {
                "metrics": centrality_metrics,
                "summary": summary,
                "algorithms_used": algorithms,
                "total_nodes": len(centrality_metrics)
            }
            
        except Exception as e:
            logger.error(f"Centrality calculation failed: {e}")
            raise ServiceError(f"Centrality calculation failed: {e}")
    
    async def _calculate_degree_centrality(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate degree centrality"""
        try:
            return await asyncio.to_thread(nx.degree_centrality, graph)
        except Exception as e:
            logger.error(f"Degree centrality calculation failed: {e}")
            return {}
    
    async def _calculate_betweenness_centrality(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate betweenness centrality"""
        try:
            # Use approximation for large graphs
            if graph.number_of_nodes() > 1000:
                k = min(100, graph.number_of_nodes() // 10)  # Sample size
                return await asyncio.to_thread(
                    nx.betweenness_centrality, 
                    graph, 
                    k=k,
                    seed=42
                )
            else:
                return await asyncio.to_thread(nx.betweenness_centrality, graph)
        except Exception as e:
            logger.error(f"Betweenness centrality calculation failed: {e}")
            return {}
    
    async def _calculate_closeness_centrality(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate closeness centrality"""
        try:
            return await asyncio.to_thread(nx.closeness_centrality, graph)
        except Exception as e:
            logger.error(f"Closeness centrality calculation failed: {e}")
            return {}
    
    async def _calculate_pagerank(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate PageRank"""
        try:
            return await asyncio.to_thread(
                nx.pagerank,
                graph,
                alpha=0.85,
                max_iter=100,
                tol=1e-6
            )
        except Exception as e:
            logger.error(f"PageRank calculation failed: {e}")
            return {}
    
    async def _calculate_eigenvector_centrality(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate eigenvector centrality"""
        try:
            return await asyncio.to_thread(
                nx.eigenvector_centrality,
                graph,
                max_iter=1000,
                tol=1e-6
            )
        except Exception as e:
            logger.warning(f"Eigenvector centrality calculation failed: {e}")
            # Return empty dict if calculation fails (e.g., for disconnected graphs)
            return {}
    
    async def _store_centrality_metrics(self, metrics: List[CentralityMetrics]) -> None:
        """Store centrality metrics in Neo4j"""
        batch_size = 100
        
        for i in range(0, len(metrics), batch_size):
            batch = metrics[i:i + batch_size]
            
            # Prepare batch update
            batch_data = []
            for metric in batch:
                batch_data.append({
                    "nodeId": metric.node_id,
                    "degree": metric.degree_centrality,
                    "betweenness": metric.betweenness_centrality,
                    "closeness": metric.closeness_centrality,
                    "pagerank": metric.pagerank,
                    "eigenvector": metric.eigenvector_centrality
                })
            
            # Batch update query
            query = """
            UNWIND $batch as row
            MATCH (n:Entity {id: row.nodeId})
            SET n.degree_centrality = row.degree,
                n.betweenness_centrality = row.betweenness,
                n.closeness_centrality = row.closeness,
                n.pagerank = row.pagerank,
                n.eigenvector_centrality = row.eigenvector,
                n.centrality_updated = datetime()
            """
            
            self.neo4j.execute_write_query(query, {"batch": batch_data})
        
        logger.info(f"Stored centrality metrics for {len(metrics)} nodes")
    
    async def _generate_centrality_summary(self, metrics: List[CentralityMetrics]) -> Dict[str, Any]:
        """Generate summary statistics for centrality metrics"""
        if not metrics:
            return {}
        
        # Extract values for each metric
        degree_values = [m.degree_centrality for m in metrics]
        betweenness_values = [m.betweenness_centrality for m in metrics]
        closeness_values = [m.closeness_centrality for m in metrics]
        pagerank_values = [m.pagerank for m in metrics]
        eigenvector_values = [m.eigenvector_centrality for m in metrics]
        
        def calculate_stats(values):
            if not values or all(v == 0 for v in values):
                return {"mean": 0, "std": 0, "min": 0, "max": 0}
            
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }
        
        summary = {
            "degree_centrality": calculate_stats(degree_values),
            "betweenness_centrality": calculate_stats(betweenness_values),
            "closeness_centrality": calculate_stats(closeness_values),
            "pagerank": calculate_stats(pagerank_values),
            "eigenvector_centrality": calculate_stats(eigenvector_values)
        }
        
        # Identify top nodes for each metric
        top_k = min(10, len(metrics))
        
        summary["top_nodes"] = {
            "by_degree": sorted(metrics, key=lambda x: x.degree_centrality, reverse=True)[:top_k],
            "by_betweenness": sorted(metrics, key=lambda x: x.betweenness_centrality, reverse=True)[:top_k],
            "by_closeness": sorted(metrics, key=lambda x: x.closeness_centrality, reverse=True)[:top_k],
            "by_pagerank": sorted(metrics, key=lambda x: x.pagerank, reverse=True)[:top_k],
            "by_eigenvector": sorted(metrics, key=lambda x: x.eigenvector_centrality, reverse=True)[:top_k]
        }
        
        return summary
    
    async def get_influential_nodes(self, 
                                  metric: str = "pagerank", 
                                  top_k: int = 20,
                                  min_threshold: float = None) -> List[Dict[str, Any]]:
        """Get most influential nodes based on centrality metrics"""
        try:
            # Validate metric
            valid_metrics = ["degree_centrality", "betweenness_centrality", 
                           "closeness_centrality", "pagerank", "eigenvector_centrality"]
            
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric. Must be one of: {valid_metrics}")
            
            # Query for top nodes
            query = f"""
            MATCH (n:Entity)
            WHERE n.{metric} IS NOT NULL
            {"AND n." + metric + " >= $threshold" if min_threshold else ""}
            RETURN n.id as id,
                   n.name as name,
                   labels(n) as labels,
                   n.{metric} as score,
                   n.degree_centrality as degree,
                   n.betweenness_centrality as betweenness,
                   n.pagerank as pagerank,
                   properties(n) as properties
            ORDER BY n.{metric} DESC
            LIMIT $topK
            """
            
            params = {"topK": top_k}
            if min_threshold:
                params["threshold"] = min_threshold
            
            result = self.neo4j.execute_query(query, params)
            
            influential_nodes = []
            for record in result:
                influential_nodes.append({
                    "id": record["id"],
                    "name": record["name"],
                    "labels": record["labels"],
                    "score": record["score"],
                    "all_metrics": {
                        "degree": record.get("degree", 0.0),
                        "betweenness": record.get("betweenness", 0.0),
                        "pagerank": record.get("pagerank", 0.0)
                    },
                    "properties": record["properties"]
                })
            
            return influential_nodes
            
        except Exception as e:
            logger.error(f"Failed to get influential nodes: {e}")
            raise ServiceError(f"Failed to get influential nodes: {e}")
    
    async def find_bridge_nodes(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find bridge nodes that connect different parts of the graph"""
        try:
            # Bridge nodes typically have high betweenness centrality
            bridge_nodes = await self.get_influential_nodes(
                metric="betweenness_centrality",
                top_k=top_k,
                min_threshold=0.01  # Only nodes with meaningful betweenness
            )
            
            # Enhance with connectivity information
            for node in bridge_nodes:
                # Get neighbor communities
                neighbor_query = """
                MATCH (n:Entity {id: $nodeId})-[]-(neighbor:Entity)
                WHERE neighbor.community_id IS NOT NULL
                RETURN DISTINCT neighbor.community_id as communityId, count(neighbor) as connections
                ORDER BY connections DESC
                """
                
                communities = self.neo4j.execute_query(neighbor_query, {"nodeId": node["id"]})
                node["connected_communities"] = communities
                node["bridge_score"] = len(communities)  # Number of different communities connected
            
            return sorted(bridge_nodes, key=lambda x: x["bridge_score"], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to find bridge nodes: {e}")
            return []
    
    async def analyze_graph_structure(self) -> Dict[str, Any]:
        """Perform comprehensive graph structure analysis"""
        try:
            logger.info("Starting comprehensive graph structure analysis")
            
            # Build NetworkX graph for analysis
            nx_graph = await self._build_networkx_graph()
            
            analysis = {
                "basic_metrics": await self._calculate_basic_graph_metrics(nx_graph),
                "connectivity": await self._analyze_connectivity(nx_graph),
                "clustering": await self._analyze_clustering(nx_graph),
                "path_analysis": await self._analyze_paths(nx_graph),
                "component_analysis": await self._analyze_components(nx_graph),
                "degree_distribution": await self._analyze_degree_distribution(nx_graph),
                "network_efficiency": await self._calculate_network_efficiency(nx_graph)
            }
            
            # Store analysis results
            await self._store_graph_analysis(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Graph structure analysis failed: {e}")
            raise ServiceError(f"Graph structure analysis failed: {e}")
    
    async def _calculate_basic_graph_metrics(self, graph: nx.Graph) -> Dict[str, Any]:
        """Calculate basic graph metrics"""
        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_connected": nx.is_connected(graph),
            "num_connected_components": nx.number_connected_components(graph),
            "average_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0
        }
    
    async def _analyze_connectivity(self, graph: nx.Graph) -> Dict[str, Any]:
        """Analyze graph connectivity"""
        connectivity = {
            "is_connected": nx.is_connected(graph),
            "number_connected_components": nx.number_connected_components(graph)
        }
        
        if nx.is_connected(graph):
            connectivity.update({
                "diameter": await asyncio.to_thread(nx.diameter, graph),
                "radius": await asyncio.to_thread(nx.radius, graph),
                "average_shortest_path_length": await asyncio.to_thread(nx.average_shortest_path_length, graph)
            })
        else:
            # Analyze largest component
            largest_cc = max(nx.connected_components(graph), key=len)
            largest_subgraph = graph.subgraph(largest_cc)
            
            connectivity.update({
                "largest_component_size": len(largest_cc),
                "largest_component_diameter": await asyncio.to_thread(nx.diameter, largest_subgraph),
                "largest_component_avg_path_length": await asyncio.to_thread(nx.average_shortest_path_length, largest_subgraph)
            })
        
        return connectivity
    
    async def _analyze_clustering(self, graph: nx.Graph) -> Dict[str, Any]:
        """Analyze clustering properties"""
        try:
            clustering_coeffs = await asyncio.to_thread(nx.clustering, graph)
            
            return {
                "average_clustering": nx.average_clustering(graph),
                "global_clustering": nx.transitivity(graph),
                "clustering_distribution": {
                    "mean": float(np.mean(list(clustering_coeffs.values()))),
                    "std": float(np.std(list(clustering_coeffs.values()))),
                    "max": float(np.max(list(clustering_coeffs.values()))),
                    "min": float(np.min(list(clustering_coeffs.values())))
                }
            }
        except Exception as e:
            logger.warning(f"Clustering analysis failed: {e}")
            return {"average_clustering": 0.0, "global_clustering": 0.0}
    
    async def _analyze_paths(self, graph: nx.Graph) -> Dict[str, Any]:
        """Analyze path characteristics"""
        path_analysis = {}
        
        try:
            if nx.is_connected(graph) and graph.number_of_nodes() <= 500:
                # Full analysis for smaller connected graphs
                all_paths = dict(nx.all_pairs_shortest_path_length(graph))
                path_lengths = [length for source in all_paths.values() for length in source.values()]
                
                path_analysis = {
                    "average_path_length": float(np.mean(path_lengths)),
                    "max_path_length": int(np.max(path_lengths)),
                    "path_length_distribution": dict(Counter(path_lengths))
                }
            else:
                # Sample-based analysis for larger graphs
                sample_size = min(100, graph.number_of_nodes())
                sample_nodes = np.random.choice(list(graph.nodes()), sample_size, replace=False)
                
                path_lengths = []
                for source in sample_nodes:
                    try:
                        lengths = nx.single_source_shortest_path_length(graph, source, cutoff=6)
                        path_lengths.extend(lengths.values())
                    except:
                        continue
                
                if path_lengths:
                    path_analysis = {
                        "estimated_average_path_length": float(np.mean(path_lengths)),
                        "estimated_max_path_length": int(np.max(path_lengths)),
                        "sample_size": sample_size
                    }
        
        except Exception as e:
            logger.warning(f"Path analysis failed: {e}")
            path_analysis = {"error": str(e)}
        
        return path_analysis
    
    async def _analyze_components(self, graph: nx.Graph) -> Dict[str, Any]:
        """Analyze connected components"""
        components = list(nx.connected_components(graph))
        component_sizes = [len(comp) for comp in components]
        
        return {
            "number_of_components": len(components),
            "largest_component_size": max(component_sizes) if component_sizes else 0,
            "smallest_component_size": min(component_sizes) if component_sizes else 0,
            "average_component_size": float(np.mean(component_sizes)) if component_sizes else 0,
            "component_size_distribution": dict(Counter(component_sizes))
        }
    
    async def _analyze_degree_distribution(self, graph: nx.Graph) -> Dict[str, Any]:
        """Analyze degree distribution"""
        degrees = [d for n, d in graph.degree()]
        degree_count = Counter(degrees)
        
        return {
            "degree_statistics": {
                "mean": float(np.mean(degrees)),
                "std": float(np.std(degrees)),
                "min": int(np.min(degrees)),
                "max": int(np.max(degrees)),
                "median": float(np.median(degrees))
            },
            "degree_distribution": dict(degree_count),
            "power_law_analysis": await self._analyze_power_law(degrees)
        }
    
    async def _analyze_power_law(self, degrees: List[int]) -> Dict[str, Any]:
        """Analyze if degree distribution follows power law"""
        try:
            import powerlaw
            
            # Fit power law
            fit = powerlaw.Fit(degrees, discrete=True)
            
            return {
                "alpha": float(fit.alpha),
                "xmin": int(fit.xmin),
                "sigma": float(fit.sigma),
                "power_law_likelihood": True if fit.alpha > 1 else False
            }
        
        except ImportError:
            logger.warning("powerlaw package not available")
            return {"analysis_available": False}
        except Exception as e:
            logger.warning(f"Power law analysis failed: {e}")
            return {"error": str(e)}
    
    async def _calculate_network_efficiency(self, graph: nx.Graph) -> Dict[str, Any]:
        """Calculate network efficiency metrics"""
        try:
            if graph.number_of_nodes() <= 200:
                # Global efficiency for smaller graphs
                global_eff = await asyncio.to_thread(nx.global_efficiency, graph)
                local_eff = await asyncio.to_thread(nx.local_efficiency, graph)
                
                return {
                    "global_efficiency": float(global_eff),
                    "local_efficiency": float(local_eff),
                    "analysis_type": "exact"
                }
            else:
                # Estimated efficiency for larger graphs
                sample_nodes = np.random.choice(list(graph.nodes()), min(50, graph.number_of_nodes()), replace=False)
                
                efficiencies = []
                for node in sample_nodes:
                    neighbors = list(graph.neighbors(node))
                    if len(neighbors) > 1:
                        subgraph = graph.subgraph(neighbors)
                        if subgraph.number_of_edges() > 0:
                            eff = nx.global_efficiency(subgraph)
                            efficiencies.append(eff)
                
                return {
                    "estimated_local_efficiency": float(np.mean(efficiencies)) if efficiencies else 0.0,
                    "sample_size": len(sample_nodes),
                    "analysis_type": "estimated"
                }
        
        except Exception as e:
            logger.warning(f"Network efficiency calculation failed: {e}")
            return {"error": str(e)}
    
    async def _store_graph_analysis(self, analysis: Dict[str, Any]) -> None:
        """Store graph analysis results in Neo4j"""
        query = """
        MERGE (ga:GraphAnalysis {id: 'latest'})
        SET ga.timestamp = datetime(),
            ga.analysis = $analysis,
            ga.num_nodes = $numNodes,
            ga.num_edges = $numEdges,
            ga.density = $density,
            ga.is_connected = $isConnected
        """
        
        basic_metrics = analysis.get("basic_metrics", {})
        
        self.neo4j.execute_write_query(query, {
            "analysis": json.dumps(analysis, default=str),
            "numNodes": basic_metrics.get("num_nodes", 0),
            "numEdges": basic_metrics.get("num_edges", 0),
            "density": basic_metrics.get("density", 0.0),
            "isConnected": basic_metrics.get("is_connected", False)
        })
    
    async def generate_insights_report(self) -> Dict[str, Any]:
        """Generate comprehensive insights report"""
        try:
            logger.info("Generating comprehensive insights report")
            
            # Gather all analysis results
            schema_analysis = await self.learn_graph_schema()
            communities = await self.detect_communities()
            centrality_results = await self.calculate_centrality_metrics()
            structure_analysis = await self.analyze_graph_structure()
            
            # Generate insights using LLM
            insights = await self._generate_ai_insights({
                "schema": schema_analysis,
                "communities": communities[:10],  # Top 10 communities
                "centrality": centrality_results["summary"],
                "structure": structure_analysis
            })
            
            # Compile final report
            report = {
                "generated_at": datetime.now().isoformat(),
                "summary": {
                    "total_entities": structure_analysis["basic_metrics"]["num_nodes"],
                    "total_relationships": structure_analysis["basic_metrics"]["num_edges"],
                    "graph_density": structure_analysis["basic_metrics"]["density"],
                    "communities_detected": len(communities),
                    "most_influential_entity": centrality_results["metrics"][0].node_id if centrality_results["metrics"] else None
                },
                "key_insights": insights,
                "recommendations": await self._generate_recommendations(schema_analysis, communities, centrality_results, structure_analysis),
                "detailed_analysis": {
                    "schema_patterns": schema_analysis["discovered_patterns"][:10],
                    "top_communities": communities[:5],
                    "influential_entities": centrality_results["metrics"][:10],
                    "structural_properties": structure_analysis
                }
            }
            
            # Store report
            await self._store_insights_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Insights report generation failed: {e}")
            raise ServiceError(f"Insights report generation failed: {e}")
    
    async def _generate_ai_insights(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate AI-powered insights from analysis data"""
        try:
            llm = self.llm_factory.get_llm(self.settings.default_llm_model)
            
            prompt = f"""
            Analyze this knowledge graph data and provide 5-7 key insights:
            
            Graph Structure:
            - {analysis_data['structure']['basic_metrics']['num_nodes']} entities
            - {analysis_data['structure']['basic_metrics']['num_edges']} relationships  
            - Density: {analysis_data['structure']['basic_metrics']['density']:.3f}
            - Connected: {analysis_data['structure']['basic_metrics']['is_connected']}
            
            Communities: {len(analysis_data['communities'])} detected
            Schema Patterns: {len(analysis_data['schema']['discovered_patterns'])} patterns found
            
            Key Metrics:
            {json.dumps(analysis_data['centrality'], indent=2)}
            
            Provide insights about:
            1. Overall graph structure and connectivity
            2. Community organization and clustering
            3. Important entities and their roles
            4. Schema quality and organization
            5. Data quality observations
            6. Potential improvements
            7. Interesting patterns discovered
            
            Format as a JSON array of insight strings.
            """
            
            response = await asyncio.to_thread(llm.predict, prompt)
            
            try:
                insights = json.loads(response)
                return insights if isinstance(insights, list) else [response]
            except json.JSONDecodeError:
                # Fallback to splitting by lines
                return [line.strip() for line in response.split('\n') if line.strip()]
                
        except Exception as e:
            logger.error(f"AI insights generation failed: {e}")
            return ["Unable to generate AI insights due to processing error."]
    
    async def _generate_recommendations(self, schema_analysis, communities, centrality_results, structure_analysis) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Schema recommendations
        if len(schema_analysis["discovered_patterns"]) < 10:
            recommendations.append("Consider enriching the knowledge graph with more diverse entity types and relationships")
        
        # Community recommendations  
        if len(communities) > structure_analysis["basic_metrics"]["num_nodes"] * 0.1:
            recommendations.append("High number of small communities detected - consider merging related communities")
        
        # Connectivity recommendations
        if not structure_analysis["basic_metrics"]["is_connected"]:
            recommendations.append("Graph has disconnected components - add bridging relationships to improve connectivity")
        
        # Centrality recommendations
        if centrality_results["metrics"]:
            top_centrality = centrality_results["metrics"][0].pagerank
            if top_centrality < 0.01:
                recommendations.append("Low centrality scores indicate sparse connectivity - add more relationships between entities")
        
        # Density recommendations
        density = structure_analysis["basic_metrics"]["density"]
        if density < 0.01:
            recommendations.append("Very low graph density - consider adding more relationships to improve knowledge connectivity")
        elif density > 0.1:
            recommendations.append("High graph density - review for potential over-connection or redundant relationships")
        
        return recommendations
    
    async def _store_insights_report(self, report: Dict[str, Any]) -> None:
        """Store insights report in Neo4j"""
        query = """
        MERGE (ir:InsightsReport {id: 'latest'})
        SET ir.generated_at = datetime(),
            ir.report = $report,
            ir.summary = $summary
        """
        
        self.neo4j.execute_write_query(query, {
            "report": json.dumps(report, default=str),
            "summary": json.dumps(report["summary"], default=str)
        })
