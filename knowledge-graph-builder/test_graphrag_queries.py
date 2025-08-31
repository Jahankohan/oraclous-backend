#!/usr/bin/env python3
"""
Comprehensive GraphRAG Query Testing Suite

This test suite demonstrates and validates all the different query capabilities
of the Advanced Neo4j GraphRAG Knowledge Graph system, including:

1. Text chunk queries (semantic similarity search)
2. Entity queries (entity-focused retrieval)
3. Relationship queries (graph traversal)
4. Hybrid queries (vector + fulltext)
5. Natural language to Cypher conversion
6. Complex graph analytics queries

Features tested:
- Vector similarity search on chunks and entities
- Graph-aware context retrieval
- Hybrid search combining vector and keyword search
- Text-to-Cypher natural language queries
- Direct Cypher queries for analytics
- Performance benchmarking
"""

import asyncio
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import our pipeline
from benchmark import AdvancedGraphRAGPipeline, AdvancedPipelineConfig

# Neo4j driver for direct queries
from neo4j import GraphDatabase

# GraphRAG retrievers
from neo4j_graphrag.retrievers import (
    VectorRetriever,
    VectorCypherRetriever,
    HybridRetriever,
    Text2CypherRetriever
)

# Embeddings for query testing
from neo4j_graphrag.embeddings import OpenAIEmbeddings


class GraphRAGQueryTester:
    """Comprehensive tester for all GraphRAG query capabilities"""
    
    def __init__(self, config: AdvancedPipelineConfig):
        self.config = config
        self.driver = None
        self.pipeline = None
        self.retrievers = {}
        self.test_results = []
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for test results"""
        logger = logging.getLogger("GraphRAGTester")
        logger.setLevel(logging.INFO)
        
        # Create handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    async def initialize(self):
        """Initialize the test environment"""
        self.logger.info("🚀 Initializing GraphRAG Query Test Suite...")
        
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_password)
        )
        
        # Initialize pipeline
        self.pipeline = AdvancedGraphRAGPipeline(self.config)
        await self.pipeline.initialize()
        
        # Create retrieval system
        retrieval_system = await self.pipeline.create_retrieval_system()
        self.retrievers = {
            'vector': retrieval_system['vector_retriever'],
            'vector_cypher': retrieval_system['vector_cypher_retriever'],
            'entity': retrieval_system['entity_retriever'],
            'hybrid': retrieval_system['hybrid_retriever'],
            'text2cypher': retrieval_system['text2cypher_retriever']
        }
        
        self.logger.info(f"✅ Initialized {len(self.retrievers)} retrievers")
        
    async def test_chunk_queries(self) -> Dict[str, Any]:
        """Test 1: Text chunk semantic similarity queries"""
        self.logger.info("\n📄 Testing Text Chunk Queries...")
        
        test_queries = [
            "What is TechNova Corporation?",
            "Tell me about MedAI Platform",
            "Who are the founders of TechNova?", 
            "What is the company's flagship product?",
            "How much funding did TechNova raise?"
        ]
        
        chunk_results = []
        
        for query in test_queries:
            start_time = time.time()
            
            try:
                # Test vector retriever for chunks
                result = self.retrievers['vector'].search(
                    query_text=query,
                    top_k=5
                )
                
                duration = time.time() - start_time
                
                chunk_results.append({
                    "query": query,
                    "method": "vector_similarity",
                    "results_count": len(result.items),
                    "duration_seconds": duration,
                    "sample_results": [
                        {
                            "content": item.content[:200] + "..." if len(item.content) > 200 else item.content,
                            "metadata": item.metadata
                        } for item in result.items[:2]  # Show first 2 results
                    ],
                    "success": True
                })
                
                self.logger.info(f"   ✅ '{query}' → {len(result.items)} chunks ({duration:.3f}s)")
                
            except Exception as e:
                chunk_results.append({
                    "query": query,
                    "method": "vector_similarity",
                    "error": str(e),
                    "success": False
                })
                self.logger.error(f"   ❌ '{query}' → Error: {e}")
        
        return {
            "test_type": "chunk_queries",
            "total_queries": len(test_queries),
            "successful_queries": len([r for r in chunk_results if r["success"]]),
            "results": chunk_results
        }
    
    async def test_entity_queries(self) -> Dict[str, Any]:
        """Test 2: Entity-focused queries"""
        self.logger.info("\n👤 Testing Entity Queries...")
        
        test_queries = [
            "Find people who work at TechNova",
            "Show me healthcare companies",
            "What organizations are mentioned in the document?",
            "Tell me about Dr. Sarah Chen",
            "Find entities related to artificial intelligence"
        ]
        
        entity_results = []
        
        for query in test_queries:
            start_time = time.time()
            
            try:
                # Test entity-focused retriever
                result = self.retrievers['entity'].search(
                    query_text=query,
                    top_k=10
                )
                
                duration = time.time() - start_time
                
                entity_results.append({
                    "query": query,
                    "method": "entity_vector_search",
                    "results_count": len(result.items),
                    "duration_seconds": duration,
                    "sample_entities": [
                        item.content for item in result.items[:5]  # Show first 5 entities
                    ],
                    "success": True
                })
                
                self.logger.info(f"   ✅ '{query}' → {len(result.items)} entities ({duration:.3f}s)")
                
            except Exception as e:
                entity_results.append({
                    "query": query,
                    "method": "entity_vector_search",
                    "error": str(e),
                    "success": False
                })
                self.logger.error(f"   ❌ '{query}' → Error: {e}")
        
        return {
            "test_type": "entity_queries",
            "total_queries": len(test_queries),
            "successful_queries": len([r for r in entity_results if r["success"]]),
            "results": entity_results
        }
    
    async def test_relationship_queries(self) -> Dict[str, Any]:
        """Test 3: Relationship and graph traversal queries"""
        self.logger.info("\n🔗 Testing Relationship Queries...")
        
        test_queries = [
            "Show connections between TechNova employees",
            "Find relationships between Dr. Sarah Chen and other people",
            "What are the connections between TechNova and healthcare institutions?",
            "Show entity relationships with high confidence in medical field",
            "Find connections between MedAI Platform and regulatory bodies"
        ]
        
        relationship_results = []
        
        for query in test_queries:
            start_time = time.time()
            
            try:
                # Test vector + cypher retriever for graph-aware context
                result = self.retrievers['vector_cypher'].search(
                    query_text=query,
                    top_k=5
                )
                
                duration = time.time() - start_time
                
                relationship_results.append({
                    "query": query,
                    "method": "graph_traversal",
                    "results_count": len(result.items),
                    "duration_seconds": duration,
                    "sample_results": [
                        {
                            "content": item.content,
                            "metadata": item.metadata
                        } for item in result.items[:2]  # Show first 2 results
                    ],
                    "success": True
                })
                
                self.logger.info(f"   ✅ '{query}' → {len(result.items)} graph contexts ({duration:.3f}s)")
                
            except Exception as e:
                relationship_results.append({
                    "query": query,
                    "method": "graph_traversal",
                    "error": str(e),
                    "success": False
                })
                self.logger.error(f"   ❌ '{query}' → Error: {e}")
        
        return {
            "test_type": "relationship_queries",
            "total_queries": len(test_queries),
            "successful_queries": len([r for r in relationship_results if r["success"]]),
            "results": relationship_results
        }
    
    async def test_hybrid_queries(self) -> Dict[str, Any]:
        """Test 4: Hybrid vector + fulltext queries"""
        self.logger.info("\n🔀 Testing Hybrid Queries...")
        
        test_queries = [
            "TechNova medical AI platform",  # Should match both vector similarity and keyword
            "Dr. Sarah Chen CEO founder",
            "healthcare artificial intelligence Austin",
            "MedAI breast cancer detection",
            "FDA approval medical device"
        ]
        
        hybrid_results = []
        
        for query in test_queries:
            start_time = time.time()
            
            try:
                # Test hybrid retriever
                result = self.retrievers['hybrid'].search(
                    query_text=query,
                    top_k=8
                )
                
                duration = time.time() - start_time
                
                hybrid_results.append({
                    "query": query,
                    "method": "hybrid_search",
                    "results_count": len(result.items),
                    "duration_seconds": duration,
                    "sample_results": [
                        item.content[:150] + "..." if len(item.content) > 150 else item.content
                        for item in result.items[:3]  # Show first 3 results
                    ],
                    "success": True
                })
                
                self.logger.info(f"   ✅ '{query}' → {len(result.items)} hybrid results ({duration:.3f}s)")
                
            except Exception as e:
                hybrid_results.append({
                    "query": query,
                    "method": "hybrid_search",
                    "error": str(e),
                    "success": False
                })
                self.logger.error(f"   ❌ '{query}' → Error: {e}")
        
        return {
            "test_type": "hybrid_queries",
            "total_queries": len(test_queries),
            "successful_queries": len([r for r in hybrid_results if r["success"]]),
            "results": hybrid_results
        }
    
    async def test_text2cypher_queries(self) -> Dict[str, Any]:
        """Test 5: Natural language to Cypher conversion"""
        self.logger.info("\n🗣️ Testing Text-to-Cypher Queries...")
        
        # Create a simplified Text2CypherRetriever based on working debug script
        from neo4j_graphrag.llm.openai_llm import OpenAILLM
        import os
        
        # Ensure OpenAI API key is set
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = self.config.openai_api_key
        
        # Create simplified LLM configuration
        llm = OpenAILLM(
            model_name="gpt-4o-mini",
            model_params={
                "temperature": 0.1,
                "max_tokens": 1000,
                "response_format": {"type": "text"}  # Ensure text output, not JSON
            }
        )
        
        # Simple schema with ONLY real relationships from your database
        simple_schema = """
        Node Types:
        - __Entity__: Properties: name, description, type
        - Chunk: Properties: text, chunk_index
        - Document: Properties: path
        
        Relationships (ONLY use these exact types that exist in database):
        - FROM_CHUNK: Entity -> Chunk
        - FROM_DOCUMENT: Chunk -> Document
        - CEO_OF: Entity -> Entity
        - CTO_OF: Entity -> Entity
        - FOUNDED_BY: Entity -> Entity
        - WORKS_AT: Entity -> Entity
        - WORKED_AT: Entity -> Entity
        - LOCATED_IN: Entity -> Entity
        - DEVELOPS: Entity -> Entity
        - LEADS: Entity -> Entity
        - MANAGES: Entity -> Entity
        - PARTNERS_WITH: Entity -> Entity
        - COLLABORATES_WITH: Entity -> Entity
        - INVESTED_IN: Entity -> Entity
        - APPROVED_BY: Entity -> Entity
        - EDUCATED_AT: Entity -> Entity
        - HOLDS_DEGREE_FROM: Entity -> Entity
        - PUBLISHED_IN: Entity -> Entity
        """
        
        # Simple examples using real relationships
        # Examples using real relationships and entity names from the database
        simple_examples = [
            "Find all entities -> MATCH (e:__Entity__) RETURN e.name LIMIT 10",
            "Find TechNova CTO -> MATCH (p:__Entity__)-[:CTO_OF]->(c:__Entity__ {name: 'TechNova'}) RETURN p.name",
            "Find companies in Austin -> MATCH (c:__Entity__)-[:LOCATED_IN]->(l:__Entity__ {name: 'Austin'}) RETURN c.name",
            "Who worked at Google -> MATCH (p:__Entity__)-[:WORKED_AT]->(c:__Entity__ {name: 'Google'}) RETURN p.name"
        ]
        
        # Simple prompt that enforces real relationship usage
        simple_prompt = """
        Convert this question to a Cypher query using ONLY the provided schema.
        
        Schema: {schema}
        
        Examples: {examples}
        
        Question: {query_text}
        
        CRITICAL: Only use the exact relationship types listed in the schema. Do NOT create relationships like 'RELATED_TO' or any other relationships not in the schema.
        
        Return only the Cypher query, nothing else:
        """
        
        # Create simplified Text2CypherRetriever
        from neo4j_graphrag.retrievers import Text2CypherRetriever
        simplified_retriever = Text2CypherRetriever(
            driver=self.driver,
            llm=llm,
            neo4j_schema=simple_schema,
            custom_prompt=simple_prompt,
            examples=simple_examples
        )
        
        # Test with queries that will return actual results based on real data
        test_queries = [
            "Show me all entities",
            "Find entities named TechNova",
            "Who is the CTO of TechNova?", 
            "What companies are located in Austin?",
            "Show me people who worked at Google"
        ]
        
        text2cypher_results = []
        
        for query in test_queries:
            start_time = time.time()
            result = None
            try:
                # Test simplified text2cypher retriever
                result = simplified_retriever.search(
                    query_text=query
                )
                
                duration = time.time() - start_time
                
                # Try to capture the generated Cypher query if available
                generated_cypher = None
                
                # Method 1: Check result metadata (most common location)
                if hasattr(result, 'metadata') and result.metadata and 'cypher' in result.metadata:
                    generated_cypher = result.metadata['cypher']
                # Method 2: Check enhanced result attribute
                elif hasattr(result, 'generated_cypher'):
                    generated_cypher = result.generated_cypher
                # Method 3: Check other result attributes
                elif hasattr(result, 'cypher_query'):
                    generated_cypher = result.cypher_query
                elif hasattr(result, 'query'):
                    generated_cypher = result.query
                # Method 4: Check wrapper's stored query
                elif hasattr(self.retrievers['text2cypher'], 'last_generated_cypher'):
                    generated_cypher = self.retrievers['text2cypher'].last_generated_cypher
                else:
                    generated_cypher = 'Query not captured'
                
                text2cypher_results.append({
                    "query": query,
                    "method": "text_to_cypher",
                    "generated_cypher": generated_cypher,
                    "results_count": len(result.items),
                    "duration_seconds": duration,
                    "sample_results": [
                        item.content for item in result.items[:5]  # Show first 5 results
                    ],
                    "success": True
                })
                
                self.logger.info(f"   ✅ '{query}' → {len(result.items)} cypher results ({duration:.3f}s)")
                self.logger.info(f"   🔍 Generated Cypher: {generated_cypher}")
                
            except Exception as e:
                text2cypher_results.append({
                    "query": query,
                    "method": "text_to_cypher",
                    "generated_cypher": str(result),
                    "error": str(e),
                    "success": False
                })
                self.logger.error(f"   ❌ '{query}' → Error: {e}")
        
        
        return {
            "test_type": "text2cypher_queries",
            "total_queries": len(test_queries),
            "successful_queries": len([r for r in text2cypher_results if r["success"]]),
            "results": text2cypher_results
        }
    
    async def test_direct_cypher_queries(self) -> Dict[str, Any]:
        """Test 6: Direct Cypher analytics queries"""
        self.logger.info("\n⚡ Testing Direct Cypher Analytics...")
        
        cypher_queries = [
            {
                "name": "graph_statistics",
                "description": "Get overall graph statistics",
                "query": """
                MATCH (d:Document) 
                OPTIONAL MATCH (d)-[:FROM_DOCUMENT]->(c:Chunk)
                OPTIONAL MATCH (c)<-[:FROM_CHUNK]-(e:__Entity__)
                OPTIONAL MATCH (e)-[r]->(re:__Entity__)
                RETURN 
                    count(DISTINCT d) as documents,
                    count(DISTINCT c) as chunks, 
                    count(DISTINCT e) as entities,
                    count(DISTINCT r) as relationships
                """
            },
            {
                "name": "top_entities",
                "description": "Find most frequently mentioned entities",
                "query": """
                MATCH (e:__Entity__)-[:FROM_CHUNK]->(c:Chunk)
                RETURN e.name as entity_name, 
                       count(c) as mentions,
                       collect(DISTINCT labels(e)[0]) as entity_types
                ORDER BY mentions DESC
                LIMIT 10
                """
            },
            {
                "name": "entity_relationships",
                "description": "Find entities with most relationships",
                "query": """
                MATCH (e:__Entity__)-[r]->(re:__Entity__)
                RETURN e.name as entity,
                       count(r) as relationship_count,
                       collect(DISTINCT type(r)) as relationship_types,
                       collect(DISTINCT re.name)[0..5] as sample_related_entities
                ORDER BY relationship_count DESC
                LIMIT 10
                """
            },
            {
                "name": "document_entity_coverage",
                "description": "Analyze entity coverage per document",
                "query": """
                MATCH (d:Document)-[:FROM_DOCUMENT]->(c:Chunk)<-[:FROM_CHUNK]-(e:__Entity__)
                RETURN d.path as document,
                       count(DISTINCT c) as chunks,
                       count(DISTINCT e) as unique_entities,
                       count(e) as total_entity_mentions,
                       round(count(e) * 1.0 / count(DISTINCT c), 2) as entities_per_chunk
                ORDER BY unique_entities DESC
                """
            },
            {
                "name": "entity_resolution_analysis",
                "description": "Analyze entity resolution effectiveness",
                "query": """
                MATCH (e1:__Entity__)-[:SAME_AS]-(e2:__Entity__)
                RETURN e1.name as entity_name,
                       count(*) as resolution_links,
                       collect(DISTINCT elementId(e2))[0..3] as sample_resolved_ids
                ORDER BY resolution_links DESC
                LIMIT 10
                """
            }
        ]
        
        cypher_results = []
        
        for query_info in cypher_queries:
            start_time = time.time()
            
            try:
                with self.driver.session(database=self.config.neo4j_database) as session:
                    result = session.run(query_info["query"])
                    records = [record.data() for record in result]
                
                duration = time.time() - start_time
                
                cypher_results.append({
                    "query_name": query_info["name"],
                    "description": query_info["description"],
                    "results_count": len(records),
                    "duration_seconds": duration,
                    "results": records[:5] if len(records) > 5 else records,  # Limit output
                    "success": True
                })
                
                self.logger.info(f"   ✅ {query_info['name']} → {len(records)} results ({duration:.3f}s)")
                
            except Exception as e:
                cypher_results.append({
                    "query_name": query_info["name"],
                    "description": query_info["description"],
                    "error": str(e),
                    "success": False
                })
                self.logger.error(f"   ❌ {query_info['name']} → Error: {e}")
        
        return {
            "test_type": "direct_cypher_queries",
            "total_queries": len(cypher_queries),
            "successful_queries": len([r for r in cypher_results if r["success"]]),
            "results": cypher_results
        }
    
    async def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test 7: Performance benchmarks across different query types"""
        self.logger.info("\n📊 Testing Performance Benchmarks...")
        
        benchmark_query = "TechNova MedAI healthcare artificial intelligence"
        iterations = 3
        
        benchmark_results = {}
        
        # Test each retriever type
        retrievers_to_test = [
            ("vector", "Vector Similarity"),
            ("entity", "Entity Search"),
            ("hybrid", "Hybrid Search"),
            ("vector_cypher", "Graph Traversal")
        ]
        
        for retriever_key, retriever_name in retrievers_to_test:
            times = []
            
            for i in range(iterations):
                start_time = time.time()
                
                try:
                    result = self.retrievers[retriever_key].search(
                        query_text=benchmark_query,
                        top_k=10
                    )
                    duration = time.time() - start_time
                    times.append(duration)
                    
                except Exception as e:
                    self.logger.error(f"Benchmark failed for {retriever_name}: {e}")
                    continue
            
            if times:
                benchmark_results[retriever_key] = {
                    "retriever_name": retriever_name,
                    "iterations": len(times),
                    "avg_duration": sum(times) / len(times),
                    "min_duration": min(times),
                    "max_duration": max(times),
                    "total_duration": sum(times)
                }
                
                avg_time = sum(times) / len(times)
                self.logger.info(f"   ✅ {retriever_name}: {avg_time:.3f}s avg")
        
        return {
            "test_type": "performance_benchmarks",
            "benchmark_query": benchmark_query,
            "iterations_per_retriever": iterations,
            "results": benchmark_results
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        self.logger.info("🧪 Starting Comprehensive GraphRAG Query Test Suite")
        self.logger.info("=" * 60)
        
        overall_start = time.time()
        
        try:
            # Run all test categories
            test_results = {
                "test_suite": "GraphRAG Comprehensive Query Tests",
                "timestamp": datetime.now().isoformat(),
                "results": {}
            }
            
            # # Test 1: Chunk queries
            # test_results["results"]["chunk_queries"] = await self.test_chunk_queries()
            
            # # Test 2: Entity queries
            # test_results["results"]["entity_queries"] = await self.test_entity_queries()
            
            # # Test 3: Relationship queries
            # test_results["results"]["relationship_queries"] = await self.test_relationship_queries()
            
            # # Test 4: Hybrid queries
            # test_results["results"]["hybrid_queries"] = await self.test_hybrid_queries()
            
            # Test 5: Text2Cypher queries
            test_results["results"]["text2cypher_queries"] = await self.test_text2cypher_queries()
            
            # # Test 6: Direct Cypher queries
            # test_results["results"]["direct_cypher_queries"] = await self.test_direct_cypher_queries()
            
            # # Test 7: Performance benchmarks
            # test_results["results"]["performance_benchmarks"] = await self.test_performance_benchmarks()
            
            total_duration = time.time() - overall_start
            
            # Calculate summary statistics
            total_tests = sum(r.get("total_queries", 0) for r in test_results["results"].values())
            total_successful = sum(r.get("successful_queries", 0) for r in test_results["results"].values())
            
            test_results["summary"] = {
                "total_duration_seconds": total_duration,
                "total_test_categories": len(test_results["results"]),
                "total_queries_tested": total_tests,
                "total_successful_queries": total_successful,
                "success_rate_percentage": (total_successful / total_tests * 100) if total_tests > 0 else 0,
                "queries_per_second": total_tests / total_duration if total_duration > 0 else 0
            }
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"Test suite failed: {e}")
            raise
    
    async def save_test_results(self, results: Dict[str, Any], filename: Optional[str] = None):
        """Save test results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"graphrag_test_results_{timestamp}.json"
        
        filepath = Path(filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"💾 Test results saved to: {filepath}")
        
        return str(filepath)
    
    def print_test_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of test results"""
        print("\n" + "=" * 80)
        print("🎯 GRAPHRAG QUERY TEST SUMMARY")
        print("=" * 80)
        
        summary = results["summary"]
        
        print(f"📊 Total Test Duration: {summary['total_duration_seconds']:.2f} seconds")
        print(f"🔢 Total Test Categories: {summary['total_test_categories']}")
        print(f"📝 Total Queries Tested: {summary['total_queries_tested']}")
        print(f"✅ Successful Queries: {summary['total_successful_queries']}")
        print(f"📈 Success Rate: {summary['success_rate_percentage']:.1f}%")
        print(f"⚡ Queries per Second: {summary['queries_per_second']:.2f}")
        
        print("\n📋 DETAILED RESULTS BY CATEGORY:")
        print("-" * 50)
        
        for category, result in results["results"].items():
            if "total_queries" in result:
                success_rate = (result["successful_queries"] / result["total_queries"] * 100) if result["total_queries"] > 0 else 0
                print(f"  {category.replace('_', ' ').title()}: {result['successful_queries']}/{result['total_queries']} ({success_rate:.1f}%)")
        
        print("\n🎉 GraphRAG Query Testing Complete!")
    
    async def cleanup(self):
        """Clean up resources"""
        if self.pipeline:
            await self.pipeline.cleanup()
        if self.driver:
            self.driver.close()


async def main():
    """Main function to run the comprehensive test suite"""
    
    # Configuration for testing
    config = AdvancedPipelineConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="",
        openai_api_key="",
        
        # Test configuration
        enable_detailed_logging=True,
        benchmark_mode=True
    )
    
    # Initialize and run tests
    tester = GraphRAGQueryTester(config)
    
    try:
        await tester.initialize()
        
        # Run comprehensive test suite
        results = await tester.run_all_tests()
        
        # Print summary
        tester.print_test_summary(results)
        
        # Save results
        results_file = await tester.save_test_results(results)
        print(f"\n💾 Full results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        raise
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    # Run the comprehensive test suite
    asyncio.run(main())
