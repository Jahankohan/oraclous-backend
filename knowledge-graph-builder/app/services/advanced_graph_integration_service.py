import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
from collections import defaultdict

from app.core.neo4j_client import Neo4jClient
from neo4j.time import DateTime as Neo4jDateTime
from app.core.exceptions import ServiceError
from app.config.settings import get_settings
from app.services.advanced_graph_analytic import AdvancedGraphAnalytics
from app.services.enhanced_graph_service import EnhancedGraphService
from app.services.enhanced_chat_service import EnhancedChatService
from app.services.entity_resolution import EntityResolution, SchemaLearning

logger = logging.getLogger(__name__)

class AdvancedGraphIntegrationService:
    # --- Document Ingestion & Processing (migrated from DocumentService) ---
    async def scan_sources(self, source_type, **kwargs):
        """Scan and create document source nodes"""
        try:
            if source_type == 'local':
                return await self._scan_local_files(**kwargs)
            elif source_type == 's3':
                return await self._scan_s3_bucket(**kwargs)
            elif source_type == 'gcs':
                return await self._scan_gcs_bucket(**kwargs)
            elif source_type == 'youtube':
                return await self._scan_youtube_video(**kwargs)
            elif source_type == 'wiki':
                return await self._scan_wikipedia_page(**kwargs)
            elif source_type == 'web':
                return await self._scan_web_page(**kwargs)
            else:
                raise Exception(f"Unsupported source type: {source_type}")
        except Exception as e:
            logger.error(f"Error scanning {source_type} sources: {e}")
            raise Exception(f"Failed to scan sources: {e}")

    async def _scan_local_files(self, file_paths):
        # ...existing code from DocumentService._scan_local_files...
        documents = []
        for file_path in file_paths:
            import uuid
            from pathlib import Path
            doc_id = str(uuid.uuid4())
            file_name = Path(file_path).name
            query = """
            CREATE (d:Document {
                id: $id,
                fileName: $fileName,
                filePath: $filePath,
                sourceType: 'local',
                status: 'New',
                createdAt: datetime(),
                totalChunks: 0,
                processedAt: null
            })
            RETURN d
            """
            self.neo4j.execute_write_query(query, {
                "id": doc_id,
                "fileName": file_name,
                "filePath": file_path,
                "sourceType": 'local',
                "status": 'New'
            })
            documents.append({
                "id": doc_id,
                "file_name": file_name,
                "source_type": 'local',
                "status": 'New',
                "created_at": datetime.now(),
                "total_chunks": 0,
                "processed_at": None
            })
        return documents

    async def _scan_s3_bucket(self, bucket_name, prefix=""):
        # ...existing code from DocumentService._scan_s3_bucket...
        logger.info(f"Scanning S3 bucket: {bucket_name}")
        return []

    async def _scan_gcs_bucket(self, project_id, bucket_name, prefix=""):
        # ...existing code from DocumentService._scan_gcs_bucket...
        logger.info(f"Scanning GCS bucket: {bucket_name}")
        return []

    async def _scan_youtube_video(self, video_url):
        # ...existing code from DocumentService._scan_youtube_video...
        import uuid
        doc_id = str(uuid.uuid4())
        video_id = video_url.split('v=')[-1].split('&')[0]
        query = """
        CREATE (d:Document {
            id: $id,
            fileName: $fileName,
            url: $url,
            sourceType: 'youtube',
            status: 'New',
            createdAt: datetime(),
            totalChunks: 0,
            processedAt: null
        })
        RETURN d
        """
        self.neo4j.execute_write_query(query, {
            "id": doc_id,
            "fileName": f"youtube_video_{video_id}",
            "url": video_url,
            "sourceType": 'youtube',
            "status": 'New'
        })
        return [{
            "id": doc_id,
            "file_name": f"youtube_video_{video_id}",
            "source_type": 'youtube',
            "status": 'New',
            "created_at": datetime.now(),
            "total_chunks": 0,
            "processed_at": None
        }]

    async def _scan_wikipedia_page(self, page_title):
        # ...existing code from DocumentService._scan_wikipedia_page...
        import uuid
        doc_id = str(uuid.uuid4())
        query = """
        CREATE (d:Document {
            id: $id,
            fileName: $fileName,
            pageTitle: $pageTitle,
            sourceType: 'wiki',
            status: 'New',
            createdAt: datetime(),
            totalChunks: 0,
            processedAt: null
        })
        RETURN d
        """
        self.neo4j.execute_write_query(query, {
            "id": doc_id,
            "fileName": f"wikipedia_{page_title.replace(' ', '_')}",
            "pageTitle": page_title,
            "sourceType": 'wiki',
            "status": 'New'
        })
        return [{
            "id": doc_id,
            "file_name": f"wikipedia_{page_title.replace(' ', '_')}",
            "source_type": 'wiki',
            "status": 'New',
            "created_at": datetime.now(),
            "total_chunks": 0,
            "processed_at": None
        }]

    async def _scan_web_page(self, url):
        # ...existing code from DocumentService._scan_web_page...
        import uuid
        doc_id = str(uuid.uuid4())
        query = """
        CREATE (d:Document {
            id: $id,
            fileName: $fileName,
            url: $url,
            sourceType: 'web',
            status: 'New',
            createdAt: datetime(),
            totalChunks: 0,
            processedAt: null
        })
        RETURN d
        """
        self.neo4j.execute_write_query(query, {
            "id": doc_id,
            "fileName": f"web_page_{hash(url)}",
            "url": url,
            "sourceType": 'web',
            "status": 'New'
        })
        return [{
            "id": doc_id,
            "file_name": f"web_page_{hash(url)}",
            "source_type": 'web',
            "status": 'New',
            "created_at": datetime.now(),
            "total_chunks": 0,
            "processed_at": None
        }]

    async def get_documents_list(self):
        # ...existing code from DocumentService.get_documents_list...
        query = """
        MATCH (d:Document)
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        RETURN d.id as id,
               d.fileName as fileName,
               d.sourceType as sourceType,
               d.status as status,
               d.createdAt as createdAt,
               d.processedAt as processedAt,
               count(c) as chunkCount
        ORDER BY d.createdAt DESC
        """
        result = self.neo4j.execute_query(query)
        documents = []
        for record in result:
            created_at = record["createdAt"]
            processed_at = record.get("processedAt")
            # Convert Neo4jDateTime to Python datetime
            if isinstance(created_at, Neo4jDateTime):
                created_at = created_at.to_native()
            if isinstance(processed_at, Neo4jDateTime):
                processed_at = processed_at.to_native()
            documents.append({
                "id": record["id"],
                "file_name": record["fileName"],
                "source_type": record["sourceType"],
                "status": record["status"],
                "created_at": created_at,
                "processed_at": processed_at,
                "chunk_count": record.get("chunkCount", 0)
            })
        return documents
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
        
        # Track initialization state
        self.initialized_features = set()
        self.initialization_results = {}
    
    async def initialize_advanced_features(self, 
                                         features: Optional[List[str]] = None,
                                         force_refresh: bool = False) -> Dict[str, Any]:
        """Initialize all advanced graph features"""
        try:
            logger.info("Initializing advanced graph features...")
            
            # Check if graph has minimum data
            stats = await self._get_basic_stats()
            
            if stats["entity_count"] < 10:
                return {
                    "status": "insufficient_data",
                    "message": "Need at least 10 entities to enable advanced features",
                    "stats": stats,
                    "recommendations": [
                        "Add more documents or data sources",
                        "Verify data extraction is working correctly",
                        "Check entity extraction settings"
                    ]
                }
            
            # Default features to initialize
            if features is None:
                features = [
                    "schema_learning",
                    "entity_resolution", 
                    "community_detection",
                    "centrality_metrics",
                    "graph_analysis",
                    "embedding_enhancement"
                ]
            
            # Initialize features progressively
            self.initialization_results = {}
            
            # 1. Schema Learning (Foundation)
            if "schema_learning" in features:
                await self._initialize_schema_learning(force_refresh)
            
            # 2. Entity Resolution (Data Quality)
            if "entity_resolution" in features:
                await self._initialize_entity_resolution(force_refresh)
            
            # 3. Community Detection (Graph Structure)
            if "community_detection" in features:
                await self._initialize_community_detection(force_refresh)
            
            # 4. Centrality Metrics (Influence Analysis)
            if "centrality_metrics" in features:
                await self._initialize_centrality_metrics(force_refresh)
            
            # 5. Graph Analysis (Comprehensive Overview)
            if "graph_analysis" in features:
                await self._initialize_graph_analysis(force_refresh)
            
            # 6. Embedding Enhancement (Semantic Layer)
            if "embedding_enhancement" in features:
                await self._initialize_embedding_enhancement(force_refresh)
            
            # Generate initialization report
            report = await self._generate_initialization_report(stats)
            
            return report
            
        except Exception as e:
            logger.error(f"Advanced features initialization failed: {e}")
            raise ServiceError(f"Advanced features initialization failed: {e}")
    
    async def _initialize_schema_learning(self, force_refresh: bool = False):
        """Initialize schema learning capabilities"""
        feature_name = "schema_learning"
        logger.info(f"Initializing {feature_name}...")
        
        try:
            if not force_refresh and feature_name in self.initialized_features:
                logger.info(f"{feature_name} already initialized")
                return
            
            # Run schema learning
            schema_results = await self.analytics.learn_graph_schema(min_frequency=2)
            
            # Store enhanced schema metadata
            await self._store_schema_metadata(schema_results)
            
            self.initialization_results[feature_name] = {
                "status": "completed",
                "patterns_discovered": len(schema_results["discovered_patterns"]),
                "suggestions": len(schema_results.get("schema_suggestions", {})),
                "evolution_plan_items": len(schema_results.get("evolution_plan", {}).get("immediate_improvements", [])),
                "completion_time": datetime.now().isoformat()
            }
            
            self.initialized_features.add(feature_name)
            logger.info(f"✓ {feature_name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Schema learning initialization failed: {e}")
            self.initialization_results[feature_name] = {
                "status": "failed", 
                "error": str(e),
                "completion_time": datetime.now().isoformat()
            }
    
    async def _initialize_entity_resolution(self, force_refresh: bool = False):
        """Initialize entity resolution and deduplication"""
        feature_name = "entity_resolution"
        logger.info(f"Initializing {feature_name}...")
        
        try:
            if not force_refresh and feature_name in self.initialized_features:
                logger.info(f"{feature_name} already initialized")
                return
            
            # Find duplicate entities
            duplicates = await self.entity_resolver.find_duplicate_entities(batch_size=500)
            
            # Auto-resolve high-confidence duplicates
            if duplicates:
                resolution_results = await self.entity_resolver.resolve_entity_duplicates(
                    duplicates, 
                    auto_merge=True
                )
                
                self.initialization_results[feature_name] = {
                    "status": "completed",
                    "duplicates_found": len(duplicates),
                    "auto_merged": resolution_results["merged_groups"],
                    "entities_merged": resolution_results["merged_entities"],
                    "pending_review": resolution_results["skipped_groups"],
                    "completion_time": datetime.now().isoformat()
                }
            else:
                self.initialization_results[feature_name] = {
                    "status": "completed",
                    "duplicates_found": 0,
                    "message": "No duplicates found - graph is clean",
                    "completion_time": datetime.now().isoformat()
                }
            
            self.initialized_features.add(feature_name)
            logger.info(f"✓ {feature_name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Entity resolution initialization failed: {e}")
            self.initialization_results[feature_name] = {
                "status": "failed", 
                "error": str(e),
                "completion_time": datetime.now().isoformat()
            }
    
    async def _initialize_community_detection(self, force_refresh: bool = False):
        """Initialize community detection"""
        feature_name = "community_detection"
        logger.info(f"Initializing {feature_name}...")
        
        try:
            if not force_refresh and feature_name in self.initialized_features:
                logger.info(f"{feature_name} already initialized")
                return
            
            # Detect communities using multiple algorithms for robustness
            communities = await self.analytics.detect_communities(
                algorithm="louvain",
                min_community_size=3,
                resolution=1.0
            )
            
            # Enhance communities with additional analysis
            enhanced_communities = await self._enhance_communities(communities)
            
            self.initialization_results[feature_name] = {
                "status": "completed",
                "communities_detected": len(communities),
                "largest_community_size": max(c.size for c in communities) if communities else 0,
                "average_community_size": sum(c.size for c in communities) / len(communities) if communities else 0,
                "community_coverage": await self._calculate_community_coverage(communities),
                "completion_time": datetime.now().isoformat()
            }
            
            self.initialized_features.add(feature_name)
            logger.info(f"✓ {feature_name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Community detection initialization failed: {e}")
            self.initialization_results[feature_name] = {
                "status": "failed", 
                "error": str(e),
                "completion_time": datetime.now().isoformat()
            }
    
    async def _initialize_centrality_metrics(self, force_refresh: bool = False):
        """Initialize centrality metrics calculation"""
        feature_name = "centrality_metrics"
        logger.info(f"Initializing {feature_name}...")
        
        try:
            if not force_refresh and feature_name in self.initialized_features:
                logger.info(f"{feature_name} already initialized")
                return
            
            # Calculate centrality metrics
            centrality_results = await self.analytics.calculate_centrality_metrics(
                algorithms=["degree", "betweenness", "closeness", "pagerank", "eigenvector"]
            )
            
            # Identify influential nodes and bridge nodes
            influential_nodes = await self.analytics.get_influential_nodes(
                metric="pagerank", 
                top_k=20
            )
            
            bridge_nodes = await self.analytics.find_bridge_nodes(top_k=10)
            
            self.initialization_results[feature_name] = {
                "status": "completed",
                "metrics_calculated": len(centrality_results["metrics"]),
                "algorithms_used": centrality_results["algorithms_used"],
                "top_influential_nodes": len(influential_nodes),
                "bridge_nodes_found": len(bridge_nodes),
                "avg_pagerank": centrality_results["summary"].get("pagerank", {}).get("mean", 0),
                "completion_time": datetime.now().isoformat()
            }
            
            self.initialized_features.add(feature_name)
            logger.info(f"✓ {feature_name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Centrality metrics initialization failed: {e}")
            self.initialization_results[feature_name] = {
                "status": "failed", 
                "error": str(e),
                "completion_time": datetime.now().isoformat()
            }
    
    async def _initialize_graph_analysis(self, force_refresh: bool = False):
        """Initialize comprehensive graph structure analysis"""
        feature_name = "graph_analysis"
        logger.info(f"Initializing {feature_name}...")
        
        try:
            if not force_refresh and feature_name in self.initialized_features:
                logger.info(f"{feature_name} already initialized")
                return
            
            # Perform comprehensive graph analysis
            analysis_results = await self.analytics.analyze_graph_structure()
            
            # Generate insights report
            insights_report = await self.analytics.generate_insights_report()
            
            self.initialization_results[feature_name] = {
                "status": "completed",
                "graph_metrics": {
                    "nodes": analysis_results["basic_metrics"]["num_nodes"],
                    "edges": analysis_results["basic_metrics"]["num_edges"],
                    "density": analysis_results["basic_metrics"]["density"],
                    "is_connected": analysis_results["basic_metrics"]["is_connected"],
                    "components": analysis_results["basic_metrics"]["num_connected_components"]
                },
                "insights_generated": len(insights_report.get("key_insights", [])),
                "recommendations": len(insights_report.get("recommendations", [])),
                "completion_time": datetime.now().isoformat()
            }
            
            self.initialized_features.add(feature_name)
            logger.info(f"✓ {feature_name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Graph analysis initialization failed: {e}")
            self.initialization_results[feature_name] = {
                "status": "failed", 
                "error": str(e),
                "completion_time": datetime.now().isoformat()
            }
    
    async def _initialize_embedding_enhancement(self, force_refresh: bool = False):
        """Initialize embedding-based enhancements"""
        feature_name = "embedding_enhancement"
        logger.info(f"Initializing {feature_name}...")
        
        try:
            if not force_refresh and feature_name in self.initialized_features:
                logger.info(f"{feature_name} already initialized")
                return
            
            # Check embedding coverage
            embedding_stats = await self._analyze_embedding_coverage()
            
            # Create similarity links based on embeddings
            similarity_links = await self._create_embedding_similarity_links()
            
            # Update vector indices
            vector_indices = await self._ensure_vector_indices()
            
            self.initialization_results[feature_name] = {
                "status": "completed",
                "embedding_coverage": embedding_stats["coverage_percentage"],
                "entities_with_embeddings": embedding_stats["entities_with_embeddings"],
                "similarity_links_created": similarity_links["links_created"],
                "vector_indices": list(vector_indices.keys()),
                "completion_time": datetime.now().isoformat()
            }
            
            self.initialized_features.add(feature_name)
            logger.info(f"✓ {feature_name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Embedding enhancement initialization failed: {e}")
            self.initialization_results[feature_name] = {
                "status": "failed", 
                "error": str(e),
                "completion_time": datetime.now().isoformat()
            }
    
    async def _get_basic_stats(self) -> Dict[str, int]:
        """Get basic graph statistics"""
        query = """
        MATCH (n:Entity)
        WITH count(n) as entityCount
        MATCH ()-[r]->()
        RETURN entityCount, count(r) as relationshipCount
        """
        
        result = self.neo4j.execute_query(query)
        if result:
            return {
                "entity_count": result[0]["entityCount"],
                "relationship_count": result[0]["relationshipCount"]
            }
        return {"entity_count": 0, "relationship_count": 0}
    
    async def _store_schema_metadata(self, schema_results: Dict[str, Any]) -> None:
        """Store enhanced schema metadata"""
        query = """
        MERGE (sm:SchemaMetadata {id: 'current'})
        SET sm.last_updated = datetime(),
            sm.patterns_count = $patternsCount,
            sm.suggestions_count = $suggestionsCount,
            sm.validation_status = 'completed',
            sm.metadata = $metadata
        """
        
        self.neo4j.execute_write_query(query, {
            "patternsCount": len(schema_results.get("discovered_patterns", [])),
            "suggestionsCount": len(schema_results.get("schema_suggestions", {})),
            "metadata": json.dumps(schema_results, default=str)
        })
    
    async def _enhance_communities(self, communities: List) -> List:
        """Enhance communities with additional analysis"""
        # Add inter-community relationship analysis
        for community in communities:
            # Find relationships to other communities
            inter_community_query = """
            MATCH (n:Entity {community_id: $communityId})-[r]-(other:Entity)
            WHERE other.community_id <> $communityId AND other.community_id IS NOT NULL
            RETURN other.community_id as otherCommunity, count(r) as connections
            ORDER BY connections DESC
            LIMIT 5
            """
            
            connections = self.neo4j.execute_query(inter_community_query, {
                "communityId": community.id
            })
            
            community.inter_community_connections = connections
        
        return communities
    
    async def _calculate_community_coverage(self, communities: List) -> float:
        """Calculate what percentage of nodes are in communities"""
        total_nodes_query = "MATCH (n:Entity) RETURN count(n) as total"
        total_result = self.neo4j.execute_query(total_nodes_query)
        total_nodes = total_result[0]["total"] if total_result else 0
        
        if total_nodes == 0:
            return 0.0
        
        nodes_in_communities = sum(c.size for c in communities)
        return (nodes_in_communities / total_nodes) * 100
    
    async def _analyze_embedding_coverage(self) -> Dict[str, Any]:
        """Analyze embedding coverage across entities"""
        query = """
        MATCH (n:Entity)
        WITH count(n) as totalEntities,
             count(n.embedding) as entitiesWithEmbeddings
        RETURN totalEntities, entitiesWithEmbeddings,
               toFloat(entitiesWithEmbeddings) / totalEntities * 100 as coveragePercentage
        """
        
        result = self.neo4j.execute_query(query)
        if result:
            return {
                "total_entities": result[0]["totalEntities"],
                "entities_with_embeddings": result[0]["entitiesWithEmbeddings"],
                "coverage_percentage": result[0]["coveragePercentage"]
            }
        return {"total_entities": 0, "entities_with_embeddings": 0, "coverage_percentage": 0.0}
    
    async def _create_embedding_similarity_links(self, threshold: float = 0.85) -> Dict[str, int]:
        """Create similarity links between entities based on embeddings"""
        # This is a simplified version - in production, you'd use vector similarity search
        similarity_query = """
        MATCH (n1:Entity), (n2:Entity)
        WHERE n1.embedding IS NOT NULL AND n2.embedding IS NOT NULL
        AND id(n1) < id(n2)
        AND NOT (n1)-[:SIMILAR_TO]-(n2)
        WITH n1, n2, 
             gds.similarity.cosine(n1.embedding, n2.embedding) as similarity
        WHERE similarity > $threshold
        CREATE (n1)-[:SIMILAR_TO {similarity: similarity}]->(n2)
        RETURN count(*) as linksCreated
        """
        
        try:
            result = self.neo4j.execute_write_query(similarity_query, {"threshold": threshold})
            return {"links_created": result[0]["linksCreated"] if result else 0}
        except Exception as e:
            logger.warning(f"Failed to create similarity links: {e}")
            return {"links_created": 0}
    
    async def _ensure_vector_indices(self) -> Dict[str, str]:
        """Ensure vector indices exist for efficient similarity search"""
        indices = {}
        
        # Entity embedding index
        try:
            create_index_query = """
            CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
            FOR (n:Entity) ON (n.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }}
            """
            
            self.neo4j.execute_write_query(create_index_query)
            indices["entity_embeddings"] = "created/exists"
            
        except Exception as e:
            logger.warning(f"Failed to create entity embeddings index: {e}")
            indices["entity_embeddings"] = f"failed: {e}"
        
        return indices
    
    async def _generate_initialization_report(self, stats: Dict[str, int]) -> Dict[str, Any]:
        """Generate comprehensive initialization report"""
        successful_features = [f for f, r in self.initialization_results.items() 
                             if r.get("status") == "completed"]
        failed_features = [f for f, r in self.initialization_results.items() 
                          if r.get("status") == "failed"]
        
        # Calculate overall health score
        total_features = len(self.initialization_results)
        success_rate = len(successful_features) / total_features if total_features > 0 else 0
        
        report = {
            "status": "completed" if not failed_features else "partial",
            "initialization_time": datetime.now().isoformat(),
            "graph_stats": stats,
            "features_initialized": successful_features,
            "failed_features": failed_features,
            "success_rate": success_rate,
            "health_score": self._calculate_health_score(stats, success_rate),
            "detailed_results": self.initialization_results,
            "next_steps": self._generate_next_steps(failed_features),
            "capabilities_enabled": self._list_enabled_capabilities()
        }
        
        # Store initialization report
        await self._store_initialization_report(report)
        
        return report
    
    def _calculate_health_score(self, stats: Dict[str, int], success_rate: float) -> float:
        """Calculate overall graph health score"""
        # Base score from feature initialization success
        base_score = success_rate * 70
        
        # Bonus points for graph size and connectivity
        entity_score = min(20, stats["entity_count"] / 100 * 20)  # Max 20 points for 100+ entities
        relationship_score = min(10, stats["relationship_count"] / 200 * 10)  # Max 10 points for 200+ relationships
        
        return min(100, base_score + entity_score + relationship_score)
    
    def _generate_next_steps(self, failed_features: List[str]) -> List[str]:
        """Generate actionable next steps"""
        next_steps = []
        
        if failed_features:
            next_steps.append(f"Investigate and resolve issues with failed features: {', '.join(failed_features)}")
        
        if "entity_resolution" not in self.initialized_features:
            next_steps.append("Run entity resolution to improve data quality")
        
        if "community_detection" not in self.initialized_features:
            next_steps.append("Enable community detection to understand graph structure")
        
        if not next_steps:
            next_steps = [
                "All features initialized successfully",
                "Consider running periodic optimization",
                "Monitor graph evolution and update features as needed"
            ]
        
        return next_steps
    
    def _list_enabled_capabilities(self) -> Dict[str, List[str]]:
        """List capabilities enabled by initialized features"""
        capabilities = {
            "analytics": [],
            "search": [],
            "chat": [],
            "optimization": []
        }
        
        if "schema_learning" in self.initialized_features:
            capabilities["analytics"].extend([
                "Schema pattern analysis",
                "Entity type suggestions",
                "Relationship type recommendations"
            ])
        
        if "community_detection" in self.initialized_features:
            capabilities["analytics"].extend([
                "Community-based analysis",
                "Cluster identification",
                "Group-based insights"
            ])
        
        if "centrality_metrics" in self.initialized_features:
            capabilities["analytics"].extend([
                "Influence analysis",
                "Bridge node identification",
                "Network importance scoring"
            ])
            capabilities["search"].extend([
                "Importance-based ranking",
                "Hub node discovery"
            ])
        
        if "embedding_enhancement" in self.initialized_features:
            capabilities["search"].extend([
                "Semantic similarity search",
                "Vector-based recommendations",
                "Content-based discovery"
            ])
            capabilities["chat"].extend([
                "Semantic query understanding",
                "Context-aware responses"
            ])
        
        if "entity_resolution" in self.initialized_features:
            capabilities["optimization"].extend([
                "Duplicate entity removal",
                "Data quality improvement",
                "Graph cleanup"
            ])
        
        return capabilities
    
    async def _store_initialization_report(self, report: Dict[str, Any]) -> None:
        """Store initialization report in Neo4j"""
        query = """
        MERGE (ir:InitializationReport {id: 'latest'})
        SET ir.timestamp = datetime(),
            ir.status = $status,
            ir.success_rate = $successRate,
            ir.health_score = $healthScore,
            ir.features_count = $featuresCount,
            ir.report = $report
        """
        
        self.neo4j.execute_write_query(query, {
            "status": report["status"],
            "successRate": report["success_rate"],
            "healthScore": report["health_score"],
            "featuresCount": len(report["detailed_results"]),
            "report": json.dumps(report, default=str)
        })
    
    # Public API methods for accessing initialized features
    
    async def get_graph_insights(self) -> Dict[str, Any]:
        """Get comprehensive graph insights"""
        if "graph_analysis" not in self.initialized_features:
            raise ServiceError("Graph analysis not initialized. Run initialize_advanced_features first.")
        
        return await self.analytics.generate_insights_report()
    
    async def perform_semantic_search(self, 
                                    query: str, 
                                    limit: int = 10,
                                    similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings"""
        if "embedding_enhancement" not in self.initialized_features:
            raise ServiceError("Embedding enhancement not initialized. Run initialize_advanced_features first.")
        
        # This would integrate with your embedding service
        # Placeholder for semantic search implementation
        return []
    
    async def get_community_analysis(self, community_id: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed community analysis"""
        if "community_detection" not in self.initialized_features:
            raise ServiceError("Community detection not initialized. Run initialize_advanced_features first.")
        
        if community_id:
            # Get specific community analysis
            query = """
            MATCH (c:Community {id: $communityId})
            RETURN c
            """
            result = self.neo4j.execute_query(query, {"communityId": community_id})
            return result[0] if result else {}
        else:
            # Get all communities summary
            query = """
            MATCH (c:Community)
            RETURN c
            ORDER BY c.size DESC
            """
            return self.neo4j.execute_query(query)
    
    async def get_influential_entities(self, 
                                     metric: str = "pagerank",
                                     top_k: int = 20) -> List[Dict[str, Any]]:
        """Get most influential entities"""
        if "centrality_metrics" not in self.initialized_features:
            raise ServiceError("Centrality metrics not initialized. Run initialize_advanced_features first.")
        
        return await self.analytics.get_influential_nodes(metric=metric, top_k=top_k)
    
    async def get_initialization_status(self) -> Dict[str, Any]:
        """Get current initialization status"""
        return {
            "initialized_features": list(self.initialized_features),
            "feature_count": len(self.initialized_features),
            "last_results": self.initialization_results,
            "capabilities": self._list_enabled_capabilities()
        }
