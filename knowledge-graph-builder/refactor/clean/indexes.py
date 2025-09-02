"""Index management utilities."""

from neo4j import Driver
from neo4j_graphrag.indexes import create_vector_index, create_fulltext_index


class IndexManager:
    """Manages vector and fulltext indexes for GraphRAG."""
    
    def __init__(self, driver: Driver, database: str = "neo4j"):
        self.driver = driver
        self.database = database
    
    def create_vector_indexes(self, embedding_dimensions: int = 3072):
        """Create optimized vector indexes for GraphRAG chatbot support."""
        try:
            # Primary text embeddings index for chunks
            create_vector_index(
                driver=self.driver,
                name="text_embeddings_primary",
                label="Chunk",
                embedding_property="embedding",
                dimensions=embedding_dimensions,
                similarity_fn="cosine"
            )
            
            # Entity embeddings index for semantic entity search
            create_vector_index(
                driver=self.driver,
                name="entity_embeddings",
                label="__Entity__",
                embedding_property="embedding", 
                dimensions=embedding_dimensions,
                similarity_fn="cosine"
            )
            
            # Relationship embeddings index for semantic relationship search
            create_vector_index(
                driver=self.driver,
                name="relationship_embeddings",
                label="__Relationship__",
                embedding_property="embedding",
                dimensions=embedding_dimensions,
                similarity_fn="cosine"
            )
            
            print("✅ Vector indexes created successfully")
            
        except Exception as e:
            print(f"⚠️ Vector index creation failed (may already exist): {e}")
    
    def create_fulltext_indexes(self):
        """Create fulltext indexes for hybrid GraphRAG retrieval."""
        try:
            # Fulltext index for chunk text content
            create_fulltext_index(
                driver=self.driver,
                name="chunk_text_fulltext",
                label="Chunk", 
                node_properties=["text"],
                neo4j_database=self.database
            )
            
            # Fulltext index for entity names and descriptions
            create_fulltext_index(
                driver=self.driver,
                name="entity_text_fulltext", 
                label="__Entity__",
                node_properties=["name", "description"],
                neo4j_database=self.database
            )
            
            print("✅ Fulltext indexes created for hybrid search")
            
        except Exception as e:
            print(f"⚠️ Fulltext index creation failed (may already exist): {e}")
    
    def create_all_indexes(self, embedding_dimensions: int = 3072):
        """Create all indexes needed for GraphRAG."""
        self.create_vector_indexes(embedding_dimensions)
        self.create_fulltext_indexes()
