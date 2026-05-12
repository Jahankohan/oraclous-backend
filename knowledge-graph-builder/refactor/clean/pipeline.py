"""Clean GraphRAG pipeline following neo4j-graphrag patterns - maintaining ALL original functionality."""

import asyncio
import logging
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from components import EntityEmbedder, RelationshipEmbedder
from config import AdvancedPipelineConfig, PerformanceMetrics
from document_processor import AdvancedDocumentProcessor
from entity_resolver import MultiAlgorithmEntityResolver
from indexes import IndexManager
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
)
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.schema import GraphSchema
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.components.types import (
    DocumentInfo,
    LexicalGraphConfig,
)
from neo4j_graphrag.experimental.pipeline import Pipeline
from neo4j_graphrag.llm import OpenAILLM
from retrieval import RetrievalSystemFactory
from schema_manager import AdvancedSchemaManager


class AdvancedGraphRAGPipeline:
    """Comprehensive Neo4j GraphRAG pipeline with enterprise features - exact same functionality as original"""

    def __init__(self, config: AdvancedPipelineConfig):
        self.config = config
        self.metrics = PerformanceMetrics()
        self.logger = self._setup_logging()

        # Initialize components
        self.driver = None
        self.llm = None
        self.embedder = None
        self.document_processor = AdvancedDocumentProcessor(config)
        self.schema_manager = None
        self.entity_resolver = None

        # Pipeline components
        self.text_splitter = None
        self.extractor = None
        self.kg_writer = None
        self.pipeline = None

    def _setup_logging(self) -> logging.Logger:
        """Configure comprehensive logging - exact same as original"""
        logger = logging.getLogger(__name__)

        # Ensure log directory exists
        log_dir = Path("logs")  # you can make this configurable
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = (
            log_dir
            / f"graphrag_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        if self.config.enable_detailed_logging:
            level = logging.DEBUG
        else:
            level = logging.INFO

        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )

        return logger

    async def initialize(self):
        """Initialize all pipeline components - exact same as original"""
        self.logger.info("Initializing Advanced GraphRAG Pipeline...")

        try:
            # Initialize Neo4j driver
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
            )

            # Test connection
            with self.driver.session(database=self.config.neo4j_database) as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value != 1:
                    raise RuntimeError("Neo4j connection test failed")

            self.logger.info("Neo4j connection established successfully")

            # Initialize LLM
            self.llm = OpenAILLM(
                model_name=self.config.llm_model,
                api_key=self.config.openai_api_key,
                model_params={
                    "temperature": self.config.llm_temperature,
                    "max_tokens": self.config.llm_max_tokens,
                    "response_format": {"type": "json_object"},
                },
            )

            # Initialize embedder
            self.embedder = OpenAIEmbeddings(
                model=self.config.embedding_model, api_key=self.config.openai_api_key
            )

            # Initialize schema manager
            self.schema_manager = AdvancedSchemaManager(self.config, self.llm)

            # Initialize multi-algorithm entity resolver using library components
            self.entity_resolver = MultiAlgorithmEntityResolver(
                self.driver, self.config
            )

            # Create vector indexes
            await self._create_vector_indexes()

            self.logger.info("Pipeline initialization completed successfully")

        except Exception as e:
            self.logger.error(f"Pipeline initialization failed: {e}")
            raise

    async def _create_vector_indexes(self):
        """Create optimized vector indexes - exact same as original"""
        index_manager = IndexManager(self.driver, self.config.neo4j_database)
        index_manager.create_all_indexes(self.config.embedding_dimensions)
        self.metrics.indexes_created += 3  # Base indexes

    def _create_advanced_pipeline(self, schema) -> Pipeline:
        """Create advanced multi-component pipeline with semantic embeddings - exact same as original"""
        pipeline = Pipeline()

        # Advanced text splitter
        self.text_splitter = FixedSizeSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            approximate=self.config.approximate_chunking,
        )

        # Chunk embedder
        chunk_embedder = TextChunkEmbedder(embedder=self.embedder)

        # Advanced entity/relation extractor with lexical graph integration
        # This will handle both lexical graph (Document/Chunk nodes) and entity extraction
        self.extractor = LLMEntityRelationExtractor(
            llm=self.llm,
            create_lexical_graph=True,  # Enable lexical graph to create FROM_CHUNK relationships
            max_concurrency=self.config.max_concurrency,
            on_error=self.config.on_error,
        )

        # Semantic embedders following library patterns
        entity_embedder = EntityEmbedder(embedder=self.embedder)
        relationship_embedder = RelationshipEmbedder(embedder=self.embedder)

        # Single KG writer for complete graph
        self.entity_kg_writer = Neo4jWriter(
            driver=self.driver,
            batch_size=self.config.batch_size,
            neo4j_database=self.config.neo4j_database,
        )

        # Add components to pipeline
        pipeline.add_component(self.text_splitter, "splitter")
        pipeline.add_component(chunk_embedder, "chunk_embedder")
        pipeline.add_component(self.extractor, "extractor")
        pipeline.add_component(entity_embedder, "entity_embedder")
        pipeline.add_component(relationship_embedder, "relationship_embedder")
        pipeline.add_component(self.entity_kg_writer, "entity_writer")

        # Connect components following clean data flow
        pipeline.connect(
            "splitter", "chunk_embedder", input_config={"text_chunks": "splitter"}
        )
        pipeline.connect(
            "chunk_embedder", "extractor", input_config={"chunks": "chunk_embedder"}
        )
        pipeline.connect(
            "extractor", "entity_embedder", input_config={"graph": "extractor"}
        )
        pipeline.connect(
            "entity_embedder",
            "relationship_embedder",
            input_config={"graph": "entity_embedder"},
        )
        pipeline.connect(
            "relationship_embedder",
            "entity_writer",
            input_config={"graph": "relationship_embedder"},
        )

        return pipeline

    async def process_document(
        self, file_path: str | Path, schema: GraphSchema = None
    ) -> dict[str, Any]:
        """Process a single document through the complete pipeline - exact same as original"""
        start_time = time.time()

        try:
            # Process document
            self.logger.info(f"Processing document: {file_path}")
            text, metadata = await self.document_processor.process_file(file_path)
            self.metrics.documents_processed += 1

            # Get or learn schema
            if schema is None:
                schema = await self.schema_manager.get_or_learn_schema(
                    text[:2000]
                )  # Sample for schema learning

            # Create pipeline
            if self.pipeline is None:
                self.pipeline = self._create_advanced_pipeline(schema)

            # Create document info for entity extraction
            document_info = DocumentInfo(path=str(file_path))

            # Run pipeline
            self.logger.info("Running advanced GraphRAG pipeline...")
            pipeline_start = time.time()

            # Run pipeline with semantic embedding configuration
            result = await self.pipeline.run(
                {
                    "splitter": {"text": text},
                    "extractor": {
                        "document_info": document_info,  # Pass document info to extractor
                        "lexical_graph_config": LexicalGraphConfig(
                            id_increment_size=50, batch_size=self.config.batch_size
                        ),
                    },
                }
            )

            pipeline_duration = time.time() - pipeline_start
            self.metrics.add_processing_time(pipeline_duration)

            # Update metrics from pipeline result
            if hasattr(result, "chunks_created"):
                self.metrics.chunks_created += result.chunks_created
            if hasattr(result, "entities_extracted"):
                self.metrics.entities_extracted += result.entities_extracted
            if hasattr(result, "relationships_extracted"):
                self.metrics.relationships_extracted += result.relationships_extracted

            # Run entity resolution
            self.logger.info("Running multi-algorithm entity resolution...")
            resolution_result = await self.entity_resolver.resolve_entities()
            self.metrics.entities_resolved += resolution_result.get(
                "total_entities_resolved", 0
            )

            total_duration = time.time() - start_time

            return {
                "success": True,
                "document_path": str(file_path),
                "metadata": metadata,
                "processing_duration": total_duration,
                "pipeline_duration": pipeline_duration,
                "entity_resolution": resolution_result,
                "metrics_snapshot": self.metrics.get_summary(),
            }

        except Exception as e:
            self.logger.error(f"Document processing failed for {file_path}: {e}")
            return {
                "success": False,
                "document_path": str(file_path),
                "error": str(e),
                "processing_duration": time.time() - start_time,
            }

    async def process_text(
        self, text: str, metadata: dict[str, Any] = None
    ) -> dict[str, Any]:
        """Process raw text content through the complete pipeline"""
        start_time = time.time()

        try:
            # Use text directly
            self.logger.info("Processing text content...")
            self.metrics.documents_processed += 1

            # Use provided metadata or create default
            if metadata is None:
                metadata = {
                    "source": "text_input",
                    "format": "text",
                    "processed_at": datetime.now().isoformat(),
                }

            # Get or learn schema
            schema = await self.schema_manager.get_or_learn_schema(
                text[:2000]
            )  # Sample for schema learning

            # Create pipeline
            if self.pipeline is None:
                self.pipeline = self._create_advanced_pipeline(schema)

            # Create document info for entity extraction
            document_info = DocumentInfo(path=metadata.get("source", "text_input"))

            # Run pipeline
            self.logger.info("Running advanced GraphRAG pipeline...")
            pipeline_start = time.time()

            # Run pipeline with semantic embedding configuration
            result = await self.pipeline.run(
                {
                    "splitter": {"text": text},
                    "extractor": {
                        "document_info": document_info,  # Pass document info to extractor
                        "lexical_graph_config": LexicalGraphConfig(
                            id_increment_size=50, batch_size=self.config.batch_size
                        ),
                    },
                }
            )

            pipeline_duration = time.time() - pipeline_start
            self.metrics.add_processing_time(pipeline_duration)

            # Update metrics from pipeline result
            if hasattr(result, "chunks_created"):
                self.metrics.chunks_created += result.chunks_created
            if hasattr(result, "entities_extracted"):
                self.metrics.entities_extracted += result.entities_extracted
            if hasattr(result, "relationships_extracted"):
                self.metrics.relationships_extracted += result.relationships_extracted

            # Run entity resolution
            self.logger.info("Running multi-algorithm entity resolution...")
            resolution_result = await self.entity_resolver.resolve_entities()
            self.metrics.entities_resolved += resolution_result.get(
                "total_entities_resolved", 0
            )

            total_duration = time.time() - start_time

            return {
                "success": True,
                "source": metadata.get("source", "text_input"),
                "metadata": metadata,
                "processing_duration": total_duration,
                "pipeline_duration": pipeline_duration,
                "entity_resolution": resolution_result,
                "metrics_snapshot": self.metrics.get_summary(),
            }

        except Exception as e:
            self.logger.error(f"Text processing failed: {e}")
            return {
                "success": False,
                "source": metadata.get("source", "text_input")
                if metadata
                else "text_input",
                "error": str(e),
                "processing_duration": time.time() - start_time,
            }

    async def process_directory(self, directory_path: str | Path) -> dict[str, Any]:
        """Process all documents in a directory - exact same as original"""
        directory_path = Path(directory_path)

        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Invalid directory path: {directory_path}")

        # Find all supported files
        supported_extensions = {".pdf", ".docx", ".txt", ".md"}
        files = [
            f
            for f in directory_path.rglob("*")
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

        if not files:
            raise ValueError(f"No supported files found in {directory_path}")

        self.logger.info(f"Found {len(files)} files to process in {directory_path}")

        # Process files
        results = []
        schema = None  # Will be learned from first document

        for i, file_path in enumerate(files, 1):
            self.logger.info(f"Processing file {i}/{len(files)}: {file_path.name}")

            result = await self.process_document(file_path, schema)
            results.append(result)

            # Use schema from first successful processing for subsequent files
            if result["success"] and schema is None:
                schema = await self.schema_manager.get_or_learn_schema()

        return {
            "directory_path": str(directory_path),
            "files_found": len(files),
            "files_processed": len([r for r in results if r["success"]]),
            "files_failed": len([r for r in results if not r["success"]]),
            "results": results,
            "final_metrics": self.metrics.get_summary(),
        }

    async def create_retrieval_system(self) -> dict[str, Any]:
        """Create advanced retrieval system - exact same as original"""
        try:
            self.logger.info("Creating advanced retrieval system...")
            factory = RetrievalSystemFactory(
                driver=self.driver,
                embedder=self.embedder,
                llm=self.llm,
                database=self.config.neo4j_database,
            )

            retrieval_system = factory.create_all_retrievers()
            self.logger.info(
                f"✅ Created {retrieval_system['retrievers_created']} retrieval strategies"
            )

            return retrieval_system

        except Exception as e:
            self.logger.error(f"Retrieval system creation failed: {e}")
            raise

    async def run_benchmarks(self) -> dict[str, Any]:
        """Run comprehensive performance benchmarks - exact same as original"""
        if not self.config.benchmark_mode:
            return {"benchmarking_disabled": True}

        self.logger.info("Running performance benchmarks...")

        benchmark_results = {}

        # Query performance benchmarks
        try:
            with self.driver.session(database=self.config.neo4j_database) as session:
                # Basic query performance test
                start_time = time.time()
                result = session.run("MATCH (n) RETURN count(n) as total_nodes")
                total_nodes = result.single()["total_nodes"]
                query_duration = time.time() - start_time

                benchmark_results["query_benchmarks"] = {
                    "total_nodes": total_nodes,
                    "basic_count_duration": query_duration,
                }

        except Exception as e:
            self.logger.error(f"Query benchmarks failed: {e}")
            benchmark_results["query_benchmarks_error"] = str(e)

        # Vector search benchmarks
        try:
            if self.embedder:
                start_time = time.time()
                # Test with actual meaningful content instead of garbage
                test_embedding = self.embedder.embed_query(
                    "artificial intelligence research"
                )
                embedding_duration = time.time() - start_time

                benchmark_results["vector_benchmarks"] = {
                    "embedding_generation_duration": embedding_duration,
                    "embedding_dimensions": len(test_embedding)
                    if test_embedding
                    else 0,
                }

        except Exception as e:
            self.logger.error(f"Vector benchmarks failed: {e}")
            benchmark_results["vector_benchmarks_error"] = str(e)

        # Processing performance metrics
        if self.metrics.processing_times:
            benchmark_results.update(
                {
                    "avg_processing_time": statistics.mean(
                        self.metrics.processing_times
                    ),
                    "median_processing_time": statistics.median(
                        self.metrics.processing_times
                    ),
                    "min_processing_time": min(self.metrics.processing_times),
                    "max_processing_time": max(self.metrics.processing_times),
                    "processing_time_stddev": statistics.stdev(
                        self.metrics.processing_times
                    )
                    if len(self.metrics.processing_times) > 1
                    else 0,
                }
            )

        return benchmark_results

    async def cleanup(self):
        """Clean up resources - exact same as original"""
        self.logger.info("Cleaning up pipeline resources...")

        if self.driver:
            self.driver.close()

        self.metrics.finalize()

        self.logger.info("Pipeline cleanup completed")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()


async def main():
    """Main execution function demonstrating advanced GraphRAG capabilities - exact same as original"""

    # Configuration - customize as needed
    config = AdvancedPipelineConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        openai_api_key=os.getenv(
            "OPENAI_API_KEY",
            "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",
        ),
        # Advanced processing configuration
        chunk_size=1500,
        chunk_overlap=300,
        max_concurrency=10,
        batch_size=2000,
        # Schema and entity resolution
        enable_schema_learning=True,
        enable_entity_resolution=True,
        similarity_threshold=0.85,
        # Performance monitoring
        enable_performance_monitoring=True,
        benchmark_mode=True,
        enable_detailed_logging=True,
    )

    print("🚀 Starting Advanced Neo4j GraphRAG Pipeline")
    print("=" * 60)

    # Initialize and run pipeline
    async with AdvancedGraphRAGPipeline(config) as pipeline:
        try:
            # Example 1: Process a single document
            print("\n📄 Processing single document...")
            document_result = await pipeline.process_document("./document.txt")
            print(f"✅ Document processed: {document_result['success']}")
            if document_result["success"]:
                print(f"   Duration: {document_result['processing_duration']:.2f}s")
                print(
                    f"   Entities resolved: {document_result['entity_resolution']['total_entities_resolved']}"
                )

            # Example 2: Create retrieval system
            print("\n🔍 Creating advanced retrieval system...")
            retrieval_system = await pipeline.create_retrieval_system()
            print(
                f"✅ Created {retrieval_system['retrievers_created']} retrieval strategies"
            )

            # Example 3: Run performance benchmarks
            print("\n📊 Running performance benchmarks...")
            benchmarks = await pipeline.run_benchmarks()
            print("✅ Benchmarks completed:")
            for key, value in benchmarks.items():
                if not key.endswith("_error"):
                    print(f"   {key}: {value}")

        except Exception as e:
            print(f"❌ Pipeline execution failed: {e}")
            raise

    print("\n🎉 Advanced GraphRAG Pipeline completed successfully!")


if __name__ == "__main__":
    # Example usage
    asyncio.run(main())
