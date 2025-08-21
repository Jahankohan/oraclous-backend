from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs.neo4j_graph import Neo4jGraph
import asyncio
import json
from uuid import UUID
from app.core.logging import get_logger
from app.services.llm_service import llm_service

logger = get_logger(__name__)

class EntityExtractor:
    """Service for extracting entities and relationships from text"""
    
    def __init__(self, neo4j_graph: Neo4jGraph):
        self.neo4j_graph = neo4j_graph
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    async def extract_entities_from_text(
        self, 
        text: str,
        user_id: str,
        graph_id: UUID,
        schema: Optional[Dict[str, Any]] = None,
        user_instructions: Optional[str] = None,
        provider: str = "openai",
        model: str = "gpt-4o-mini"
    ) -> List[GraphDocument]:
        """Extract entities and relationships from text using LLM"""
        
        try:
            # Initialize LLM if needed
            if not llm_service.is_initialized():
                success = await llm_service.initialize_llm(
                    user_id=user_id,
                    provider=provider,
                    model=model
                )
                if not success:
                    raise ValueError(f"Failed to initialize LLM: {provider}")
            
            # Set schema if provided
            if schema:
                llm_service.set_schema(schema)
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Split text into {len(chunks)} chunks for processing")
            
            # Process chunks in batches to avoid rate limits
            batch_size = 3
            all_graph_documents = []
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                # Create documents for this batch
                documents = [
                    Document(
                        page_content=chunk,
                        metadata={
                            "graph_id": str(graph_id),
                            "chunk_index": i + j,
                            "source": "text_input"
                        }
                    )
                    for j, chunk in enumerate(batch_chunks)
                ]
                
                # Extract entities from batch
                try:
                    batch_graph_docs = await llm_service.graph_transformer.aconvert_to_graph_documents(
                        documents
                    )
                    
                    # Add graph_id to all nodes and relationships
                    for graph_doc in batch_graph_docs:
                        # Add graph_id to all nodes
                        for node in graph_doc.nodes:
                            node.properties["graph_id"] = str(graph_id)
                        
                        # Add graph_id to all relationships
                        for rel in graph_doc.relationships:
                            rel.properties["graph_id"] = str(graph_id)
                    
                    all_graph_documents.extend(batch_graph_docs)
                    logger.info(f"Processed batch {i//batch_size + 1}, extracted {len(batch_graph_docs)} graph documents")
                    
                    # Small delay to respect rate limits
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                    continue
            
            logger.info(f"Total extracted: {len(all_graph_documents)} graph documents")
            return all_graph_documents
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            raise
    
    async def learn_schema_from_text(
        self, 
        text: str, 
        user_id: str,
        provider: str = "openai"
    ) -> Dict[str, List[str]]:
        """Learn potential entity types and relationships from text sample"""
        
        try:
            # Initialize LLM for schema learning
            if not llm_service.is_initialized():
                await llm_service.initialize_llm(user_id, provider)
            
            schema_prompt = f"""
            Analyze the following text and identify the main types of entities and relationships that should be extracted to build a knowledge graph.

            Text sample:
            {text[:2000]}...

            Please respond with a JSON object containing:
            - "entities": A list of entity types (e.g., ["Person", "Organization", "Location", "Product"])
            - "relationships": A list of relationship types (e.g., ["WORKS_FOR", "LOCATED_IN", "OWNS"])

            Be specific and relevant to the content. Limit to the most important 8-10 entity types and 6-8 relationship types.
            """
            
            response = await llm_service.llm.ainvoke(schema_prompt)
            
            # Parse JSON response
            try:
                schema = json.loads(response.content)
                
                # Validate schema structure
                if "entities" in schema and "relationships" in schema:
                    return schema
                else:
                    logger.warning("Invalid schema format from LLM, using defaults")
                    return self._get_default_schema()
                    
            except json.JSONDecodeError:
                logger.warning("Failed to parse schema JSON from LLM")
                return self._get_default_schema()
                
        except Exception as e:
            logger.error(f"Schema learning failed: {e}")
            return self._get_default_schema()
    
    def _get_default_schema(self) -> Dict[str, List[str]]:
        """Return default schema when auto-learning fails"""
        return {
            "entities": [
                "Person", "Organization", "Location", "Product", 
                "Event", "Concept", "Document", "Technology"
            ],
            "relationships": [
                "WORKS_FOR", "LOCATED_IN", "PART_OF", "OWNS", 
                "PARTICIPATES_IN", "RELATED_TO", "USES", "CREATED"
            ]
        }

# Global extractor instance
entity_extractor = EntityExtractor(None)  # Will be initialized with Neo4j graph later
