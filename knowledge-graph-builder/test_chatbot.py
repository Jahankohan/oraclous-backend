#!/usr/bin/env python3
"""
Quick test script for the GraphRAG Chatbot
"""

import asyncio
import os
from graphrag_chatbot import GraphRAGChatbot, AdvancedPipelineConfig

async def test_chatbot():
    """Test the chatbot with some sample queries"""
    
    # Configuration
    config = AdvancedPipelineConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        openai_api_key="",
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-3-large"
    )
    
    print("🤖 Testing GraphRAG Chatbot")
    print("=" * 50)
    
    try:
        async with GraphRAGChatbot(config) as chatbot:
            # Test queries
            test_queries = [
                "What entities are in the knowledge graph?",
                "Who works for TechNova?", 
                "Tell me about Michael Rodriguez",
                "What companies are mentioned?",
                "Show me relationships in the graph",
                "What is the MedAI Platform?",
                "Tell me about Dr. Sarah Chen",
                "Where is TechNova located?",
                "What does TechNova do?",
                "Who are the investors in TechNova?"
            ]
            
            session_id = "test_session"
            
            for i, query in enumerate(test_queries, 1):
                print(f"\n🗣️  Test {i}: {query}")
                print("-" * 40)
                
                try:
                    response = await chatbot.chat(query, session_id)
                    print(f"🤖 Response: {response.content}")
                    
                    # Show retrieval info
                    if response.retrieval_info:
                        strategy = response.retrieval_info.get("strategy", "unknown")
                        results_count = response.retrieval_info.get("results_count", 0)
                        retrieval_time = response.retrieval_info.get("retrieval_time", 0)
                        print(f"   📊 Strategy: {strategy}, Sources: {results_count}, Time: {retrieval_time:.2f}s")
                
                except Exception as e:
                    print(f"❌ Error: {e}")
            
            # Show final stats
            print("\n📈 Performance Summary:")
            print("-" * 25)
            stats = chatbot.get_performance_stats()
            for key, value in stats.items():
                print(f"{key}: {value}")
    
    except Exception as e:
        print(f"❌ Chatbot initialization failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Neo4j is running on bolt://localhost:7687")
        print("2. Check your Neo4j credentials (neo4j/password)")
        print("3. Ensure you have data in your knowledge graph")
        print("4. Verify OpenAI API key is valid")

if __name__ == "__main__":
    asyncio.run(test_chatbot())
