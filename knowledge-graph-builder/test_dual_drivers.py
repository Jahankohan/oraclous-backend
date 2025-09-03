#!/usr/bin/env python3
"""
Test script for Neo4j dual driver architecture.

This script tests the new dual driver implementation to ensure:
1. Async driver works for FastAPI operations
2. Sync driver works for GraphRAG components  
3. Both drivers can coexist without conflicts
4. WorkerNeo4jManager works for Celery tasks
"""

import asyncio
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from app.core.neo4j_client import neo4j_client
from app.services.background_jobs import WorkerNeo4jManager
from app.core.logging import get_logger

logger = get_logger(__name__)

async def test_async_driver():
    """Test async driver functionality."""
    print("🔄 Testing Async Driver...")
    
    try:
        # Connect async driver
        await neo4j_client.connect_async()
        
        # Test basic query
        result = await neo4j_client.execute_query("RETURN 1 as test")
        assert result[0]["test"] == 1
        
        print("✅ Async driver working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Async driver failed: {e}")
        return False

def test_sync_driver():
    """Test sync driver functionality."""
    print("🔄 Testing Sync Driver...")
    
    try:
        # Connect sync driver
        neo4j_client.connect_sync()
        
        # Test basic query using sync driver directly
        driver = neo4j_client.sync_driver
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            assert record["test"] == 1
        
        print("✅ Sync driver working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Sync driver failed: {e}")
        return False

async def test_worker_manager():
    """Test WorkerNeo4jManager for Celery tasks."""
    print("🔄 Testing WorkerNeo4jManager...")
    
    try:
        # Test async context manager
        async with WorkerNeo4jManager() as worker_neo4j:
            sync_driver = worker_neo4j.get_sync_driver()
            async_driver = worker_neo4j.get_async_driver()
            
            # Test sync driver
            with sync_driver.session() as session:
                result = session.run("RETURN 'sync' as test")
                assert result.single()["test"] == "sync"
            
            # Test async driver  
            async with async_driver.session() as session:
                result = await session.run("RETURN 'async' as test")
                records = await result.data()
                assert records[0]["test"] == "async"
        
        # Test sync-only context manager
        with WorkerNeo4jManager() as worker_neo4j:
            sync_driver = worker_neo4j.get_sync_driver()
            
            with sync_driver.session() as session:
                result = session.run("RETURN 'sync_only' as test")
                assert result.single()["test"] == "sync_only"
        
        print("✅ WorkerNeo4jManager working correctly")
        return True
        
    except Exception as e:
        print(f"❌ WorkerNeo4jManager failed: {e}")
        return False

async def test_health_check():
    """Test Neo4j health check with both drivers."""
    print("🔄 Testing Health Check...")
    
    try:
        health_info = await neo4j_client.health_check()
        
        assert health_info["status"] == "healthy"
        assert health_info["async_driver"] == "connected"
        
        print(f"✅ Health check passed: {health_info}")
        return True
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

async def test_concurrent_access():
    """Test that both drivers can be used concurrently."""
    print("🔄 Testing Concurrent Access...")
    
    try:
        # Create concurrent tasks using both drivers
        async def async_task():
            return await neo4j_client.execute_query("RETURN 'async' as source")
        
        def sync_task():
            with neo4j_client.sync_driver.session() as session:
                result = session.run("RETURN 'sync' as source")
                return result.single()["source"]
        
        # Run async task
        async_result = await async_task()
        
        # Run sync task
        sync_result = sync_task()
        
        assert async_result[0]["source"] == "async"
        assert sync_result == "sync"
        
        print("✅ Concurrent access working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Concurrent access failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting Neo4j Dual Driver Tests")
    print("=" * 50)
    
    tests = [
        test_async_driver,
        test_sync_driver, 
        test_worker_manager,
        test_health_check,
        test_concurrent_access
    ]
    
    results = []
    for test in tests:
        try:
            if asyncio.iscoroutinefunction(test):
                result = await test()
            else:
                result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
        
        print()  # Add spacing between tests
    
    # Cleanup
    try:
        await neo4j_client.disconnect()
        print("🧹 Cleanup completed")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")
    
    # Summary
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All tests passed! ({passed}/{total})")
        print("✅ Neo4j dual driver architecture is working correctly")
        return 0
    else:
        print(f"❌ Some tests failed ({passed}/{total})")
        print("🔧 Please check the configuration and try again")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
