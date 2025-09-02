"""Example showing how to use the clean pipeline - exact same functionality as original."""

import asyncio
from pathlib import Path
from pipeline import AdvancedGraphRAGPipeline
from config import AdvancedPipelineConfig


async def advanced_example():
    """Complete example maintaining all original functionality."""
    
    # Configure pipeline with all original settings
    config = AdvancedPipelineConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j", 
        neo4j_password="password",
        # Use environment variable or fallback to service account key
        # Don't pass openai_api_key - let it be loaded from environment via field factory
        
        # All original configuration options
        chunk_size=1500,
        chunk_overlap=300,
        max_concurrency=10,
        batch_size=2000,
        enable_entity_resolution=True,
        similarity_threshold=0.85,
        fuzzy_threshold=0.8,
        enable_schema_learning=True,
        enforce_schema=True,
        enable_performance_monitoring=True,
        benchmark_mode=True,
        enable_detailed_logging=True
    )
    
    # Read the comprehensive TechNova Corporation document
    document_path = Path("./document.txt")
    if document_path.exists():
        text = document_path.read_text().strip()
        print(f"📄 Loaded document: {len(text)} characters")
    else:
        # Fallback to embedded text if file not found
        text = """
        TechNova Corporation is a leading artificial intelligence company founded in Austin, Texas in 2020.
        The company was founded by Dr. Sarah Chen, who serves as the CEO, and Mark Rodriguez, who is the CTO.
        TechNova specializes in developing advanced machine learning algorithms for natural language processing.
        The company has partnerships with several major universities including MIT and Stanford.
        Dr. Chen previously worked at Google Brain and published numerous research papers on transformer architectures.
        Mark Rodriguez has a background in distributed systems and neural network optimization.
        The company's flagship product is an AI assistant that helps businesses automate customer service.
        TechNova has raised $50 million in Series A funding and employs over 200 engineers.
        """
        print("📄 Using fallback text (document.txt not found)")
    
    print("🚀 Starting Advanced GraphRAG Pipeline")
    print("=" * 60)
    
    # Use pipeline with all original functionality
    async with AdvancedGraphRAGPipeline(config) as pipeline:
        # Example 1: Process text content directly
        print("\n📄 Processing text content...")
        result = await pipeline.process_text(text, metadata={"source": "example_text"})
        print(f"✅ Processing completed: {result['success']}")
        if result['success']:
            print(f"   Duration: {result['processing_duration']:.2f}s")
            print(f"   Entities resolved: {result['entity_resolution']['total_entities_resolved']}")
        
        # Example 2: Process directory (same as original)  
        # print("\n📁 Processing document directory...")
        # directory_result = await pipeline.process_directory("./documents")
        # print(f"✅ Directory processed: {directory_result['files_processed']}/{directory_result['files_found']} files")
        
        # Example 3: Create retrieval system (same as original)
        print("\n🔍 Creating advanced retrieval system...")
        retrieval_system = await pipeline.create_retrieval_system()
        print(f"✅ Created {retrieval_system['retrievers_created']} retrieval strategies")
        
        # Example 4: Run performance benchmarks (same as original)
        print("\n📊 Running performance benchmarks...")
        benchmarks = await pipeline.run_benchmarks()
        print("✅ Benchmarks completed:")
        for key, value in benchmarks.items():
            if not key.endswith("_error"):
                print(f"   {key}: {value}")
        
        # Final metrics summary (same as original)
        print("\n📈 Final Performance Summary")
        print("=" * 40)
        final_metrics = pipeline.metrics.get_summary()
        for key, value in final_metrics.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
        print("\n🎉 Advanced GraphRAG Pipeline completed successfully!")


if __name__ == "__main__":
    asyncio.run(advanced_example())
