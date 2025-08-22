from typing import List, Dict, Any, Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain.schema.output import LLMResult
import asyncio
import json
from uuid import UUID, uuid4
from app.core.logging import get_logger
from app.services.llm_service import llm_service
from app.services.embedding_service import embedding_service
from app.services.vector_service import vector_service

logger = get_logger(__name__)

class ProductionEntityExtractor:
    """Production-grade entity extractor following Neo4j Labs patterns"""
    
    def __init__(self):
        # Advanced chunking strategy for dual graph approach
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,  # Larger overlap for context preservation
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "],
            keep_separator=True
        )
        
        # Schema evolution tracking
        self.evolved_schemas = {}
        self.entity_similarity_threshold = 0.97
        self.text_distance_threshold = 5
    
    async def extract_with_dual_graph(
        self,
        text: str,
        user_id: str,
        graph_id: UUID,
        domain_context: Optional[str] = None
    ) -> Tuple[List[GraphDocument], List[Dict[str, Any]]]:
        """
        Extract both lexical and entity graphs following Neo4j Labs dual approach
        """
        
        # Step 1: Create lexical graph (document chunks)
        chunks = await self._create_enhanced_chunks(text, graph_id)
        
        # Step 2: Dynamic schema evolution
        evolved_schema = await self._evolve_schema_from_text(text, user_id, domain_context)
        
        # Step 3: Extract entities using structured output
        entity_graph_docs = await self._extract_entities_with_structured_output(
            chunks, user_id, graph_id, evolved_schema
        )
        
        # Step 4: Advanced entity deduplication
        deduplicated_docs = await self._advanced_entity_deduplication(entity_graph_docs)
        
        # Step 5: Create cross-graph relationships
        enhanced_docs = await self._create_chunk_entity_relationships(
            chunks, deduplicated_docs, graph_id
        )
        
        return enhanced_docs, chunks
    
    async def _create_enhanced_chunks(
        self, 
        text: str, 
        graph_id: UUID
    ) -> List[Dict[str, Any]]:
        """Create enhanced text chunks with metadata and embeddings"""
        
        # Split text into semantic chunks
        documents = self.text_splitter.create_documents([text])
        
        enhanced_chunks = []
        for i, doc in enumerate(documents):
            chunk_id = f"chunk_{graph_id}_{i}"
            
            # Generate embedding for chunk
            embedding = None
            if embedding_service.is_initialized():
                try:
                    embedding = await embedding_service.embed_text(doc.page_content)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for chunk {i}: {e}")
            
            chunk_data = {
                "id": chunk_id,
                "text": doc.page_content,
                "graph_id": str(graph_id),
                "chunk_index": i,
                "char_start": i * (2000 - 400),  # Approximate position
                "char_end": min((i + 1) * (2000 - 400), len(text)),
                "embedding": embedding,
                "word_count": len(doc.page_content.split()),
                "type": "DocumentChunk"
            }
            enhanced_chunks.append(chunk_data)
        
        return enhanced_chunks
    
    async def _evolve_schema_from_text(
        self,
        text: str,
        user_id: str,
        domain_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Dynamic schema evolution using LLM analysis"""
        
        # Check if we have existing schema for this domain
        schema_key = f"{user_id}_{domain_context or 'general'}"
        
        if schema_key in self.evolved_schemas:
            base_schema = self.evolved_schemas[schema_key]
        else:
            base_schema = await self._extract_initial_schema(text, domain_context)
            self.evolved_schemas[schema_key] = base_schema
        
        # Evolve schema based on new content
        evolved_schema = await self._refine_schema_with_context(text, base_schema)
        self.evolved_schemas[schema_key] = evolved_schema
        
        return evolved_schema
    
    async def _extract_initial_schema(
        self,
        text: str,
        domain_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract initial schema using LLM with function calling"""
        
        if not llm_service.is_initialized():
            return self._get_fallback_schema()
        
        schema_prompt = f"""
        Analyze this text and identify the optimal entity types and relationship types for knowledge graph extraction.
        
        Domain Context: {domain_context or 'General'}
        
        Text Sample: {text[:3000]}...
        
        Consider:
        1. What are the main types of entities mentioned?
        2. What relationships exist between these entities?
        3. What properties would be valuable to extract?
        
        Focus on entities and relationships that appear multiple times or are central to the meaning.
        """
        
        try:
            # Use structured output for consistent schema format
            response = await llm_service.llm.ainvoke([
                {"role": "system", "content": "You are a knowledge graph schema expert. Return valid JSON only."},
                {"role": "user", "content": schema_prompt}
            ])
            
            # Parse schema from LLM response
            schema_text = response.content
            if "```json" in schema_text:
                schema_text = schema_text.split("```json")[1].split("```")[0]
            elif "```" in schema_text:
                schema_text = schema_text.split("```")[1].split("```")[0]
            
            schema = json.loads(schema_text)
            
            # Validate and clean schema
            return self._validate_and_clean_schema(schema)
            
        except Exception as e:
            logger.warning(f"Schema extraction failed: {e}")
            return self._get_fallback_schema()
    
    async def _extract_entities_with_structured_output(
        self,
        chunks: List[Dict[str, Any]],
        user_id: str,
        graph_id: UUID,
        schema: Dict[str, Any]
    ) -> List[GraphDocument]:
        """Extract entities using LLM structured output (function calling)"""
        
        if not llm_service.is_initialized():
            raise ValueError("LLM service not initialized")
        
        # Configure graph transformer with evolved schema
        llm_service.set_dynamic_schema(schema)
        
        all_graph_docs = []
        
        # Process chunks in batches for efficiency
        batch_size = 3
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            # Create documents for batch
            documents = [
                Document(
                    page_content=chunk["text"],
                    metadata={
                        "chunk_id": chunk["id"],
                        "graph_id": str(graph_id),
                        "chunk_index": chunk["chunk_index"]
                    }
                )
                for chunk in batch_chunks
            ]
            
            try:
                # Use structured output for consistent extraction
                batch_docs = await llm_service.graph_transformer.aconvert_to_graph_documents(
                    documents
                )
                
                # Enhance extracted entities with graph metadata
                for doc in batch_docs:
                    for node in doc.nodes:
                        if not hasattr(node, 'properties') or not node.properties:
                            node.properties = {}
                        node.properties.update({
                            "graph_id": str(graph_id),
                            "extraction_source": "llm_structured",
                            "confidence": self._calculate_entity_confidence(node, schema)
                        })
                    
                    for rel in doc.relationships:
                        if not hasattr(rel, 'properties') or not rel.properties:
                            rel.properties = {}
                        rel.properties.update({
                            "graph_id": str(graph_id),
                            "extraction_source": "llm_structured"
                        })
                
                all_graph_docs.extend(batch_docs)
                
            except Exception as e:
                logger.error(f"Batch extraction failed: {e}")
                continue
        
        return all_graph_docs
    
    async def _advanced_entity_deduplication(
        self,
        graph_documents: List[GraphDocument]
    ) -> List[GraphDocument]:
        """Advanced entity deduplication using multiple criteria"""
        
        if not graph_documents:
            return []
        
        # Collect all unique entities with embeddings
        entity_map = {}
        
        for doc in graph_documents:
            for node in doc.nodes:
                entity_key = self._create_entity_key(node)
                
                if entity_key not in entity_map:
                    # Generate embedding for entity
                    entity_text = self._create_entity_text(node)
                    embedding = None
                    
                    if embedding_service.is_initialized():
                        try:
                            embedding = await embedding_service.embed_text(entity_text)
                        except Exception as e:
                            logger.warning(f"Failed to generate embedding for entity {node.id}: {e}")
                    
                    entity_map[entity_key] = {
                        "node": node,
                        "embedding": embedding,
                        "entity_text": entity_text,
                        "variants": [node.properties.get("name", node.id)]
                    }
                else:
                    # Check for duplicates using multiple criteria
                    existing = entity_map[entity_key]
                    if await self._are_entities_duplicates(node, existing, embedding_service):
                        # Merge entities
                        merged_node = self._merge_entities(existing["node"], node)
                        entity_map[entity_key]["node"] = merged_node
                        entity_map[entity_key]["variants"].append(node.properties.get("name", node.id))
        
        # Rebuild graph documents with deduplicated entities
        return self._rebuild_graph_documents_with_deduplicated_entities(
            graph_documents, entity_map
        )
    
    async def _are_entities_duplicates(
        self,
        node1: Node,
        existing_entity: Dict[str, Any],
        embedding_service
    ) -> bool:
        """Multi-criteria duplicate detection"""
        
        node2 = existing_entity["node"]
        
        # 1. Exact name match
        name1 = node1.properties.get("name", node1.id).lower().strip()
        name2 = node2.properties.get("name", node2.id).lower().strip()
        
        if name1 == name2:
            return True
        
        # 2. Substring containment
        if name1 in name2 or name2 in name1:
            if min(len(name1), len(name2)) / max(len(name1), len(name2)) > 0.7:
                return True
        
        # 3. Edit distance
        edit_distance = self._calculate_edit_distance(name1, name2)
        if edit_distance <= self.text_distance_threshold:
            return True
        
        # 4. Embedding similarity (if available)
        if embedding_service.is_initialized() and existing_entity["embedding"]:
            try:
                entity1_text = self._create_entity_text(node1)
                embedding1 = await embedding_service.embed_text(entity1_text)
                
                similarity = self._cosine_similarity(embedding1, existing_entity["embedding"])
                if similarity >= self.entity_similarity_threshold:
                    return True
            except Exception as e:
                logger.warning(f"Embedding similarity check failed: {e}")
        
        return False
    
    async def _create_chunk_entity_relationships(
        self,
        chunks: List[Dict[str, Any]],
        entity_docs: List[GraphDocument],
        graph_id: UUID
    ) -> List[GraphDocument]:
        """Create relationships between chunks and entities they contain"""
        
        enhanced_docs = entity_docs.copy()
        
        # Create chunk nodes
        chunk_nodes = []
        for chunk in chunks:
            chunk_node = Node(
                id=chunk["id"],
                type="DocumentChunk",
                properties={
                    "text": chunk["text"],
                    "graph_id": str(graph_id),
                    "chunk_index": chunk["chunk_index"],
                    "word_count": chunk["word_count"],
                    "char_start": chunk["char_start"],
                    "char_end": chunk["char_end"]
                }
            )
            chunk_nodes.append(chunk_node)
        
        # Create CONTAINS relationships between chunks and entities
        chunk_entity_relationships = []
        
        for doc in entity_docs:
            source_chunk_id = None
            if hasattr(doc, 'source') and doc.source and hasattr(doc.source, 'metadata'):
                source_chunk_id = doc.source.metadata.get("chunk_id")
            
            if source_chunk_id:
                # Find the chunk node
                chunk_node = next((cn for cn in chunk_nodes if cn.id == source_chunk_id), None)
                
                if chunk_node:
                    for entity_node in doc.nodes:
                        contains_rel = Relationship(
                            source=chunk_node,
                            target=entity_node,
                            type="CONTAINS",
                            properties={
                                "graph_id": str(graph_id),
                                "extraction_source": "chunk_analysis"
                            }
                        )
                        chunk_entity_relationships.append(contains_rel)
        
        # Create a combined graph document
        if chunk_nodes or chunk_entity_relationships:
            combined_doc = GraphDocument(
                nodes=chunk_nodes,
                relationships=chunk_entity_relationships,
                source=Document(
                    page_content="Combined chunk-entity graph",
                    metadata={"graph_id": str(graph_id), "type": "lexical_graph"}
                )
            )
            enhanced_docs.append(combined_doc)
        
        return enhanced_docs
    
    # Helper methods
    def _validate_and_clean_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean extracted schema"""
        cleaned = {
            "entities": [],
            "relationships": [],
            "properties": {}
        }
        
        if "entities" in schema and isinstance(schema["entities"], list):
            cleaned["entities"] = [e for e in schema["entities"] if isinstance(e, str) and e.strip()]
        
        if "relationships" in schema and isinstance(schema["relationships"], list):
            cleaned["relationships"] = [r for r in schema["relationships"] if isinstance(r, str) and r.strip()]
        
        # Ensure minimum schema
        if not cleaned["entities"]:
            cleaned["entities"] = ["Entity", "Person", "Organization", "Concept"]
        
        if not cleaned["relationships"]:
            cleaned["relationships"] = ["RELATED_TO", "PART_OF", "ASSOCIATED_WITH"]
        
        return cleaned
    
    def _get_fallback_schema(self) -> Dict[str, Any]:
        """Fallback schema when extraction fails"""
        return {
            "entities": ["Person", "Organization", "Location", "Concept", "Event", "Product"],
            "relationships": ["WORKS_FOR", "LOCATED_IN", "PART_OF", "RELATED_TO", "PARTICIPATES_IN"],
            "properties": {}
        }
    
    def _calculate_entity_confidence(self, node: Node, schema: Dict[str, Any]) -> float:
        """Calculate confidence score for extracted entity"""
        confidence = 0.8  # Base confidence
        
        # Higher confidence if entity type is in evolved schema
        if hasattr(node, 'type') and node.type in schema.get("entities", []):
            confidence += 0.1
        
        # Higher confidence if entity has multiple properties
        if hasattr(node, 'properties') and node.properties:
            prop_count = len([v for v in node.properties.values() if v])
            confidence += min(0.1, prop_count * 0.02)
        
        return min(1.0, confidence)
    
    def _create_entity_key(self, node: Node) -> str:
        """Create unique key for entity deduplication"""
        name = node.properties.get("name", node.id) if hasattr(node, 'properties') else node.id
        entity_type = node.type if hasattr(node, 'type') else "Entity"
        return f"{entity_type}:{name.lower().strip()}"
    
    def _create_entity_text(self, node: Node) -> str:
        """Create text representation of entity for embedding"""
        text_parts = []
        
        if hasattr(node, 'properties') and node.properties:
            if "name" in node.properties:
                text_parts.append(node.properties["name"])
            if "description" in node.properties:
                text_parts.append(node.properties["description"])
        
        if not text_parts:
            text_parts.append(node.id)
        
        return " ".join(text_parts)
    
    def _calculate_edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._calculate_edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)

# Global instance
production_entity_extractor = ProductionEntityExtractor()