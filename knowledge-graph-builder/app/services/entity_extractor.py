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

class SchemaEvolutionConfig:
    """Configuration for schema evolution behavior"""
    
    STRICT_MODE = "strict"           # Never add new types
    GUIDED_MODE = "guided"           # Add new types with LLM validation
    PERMISSIVE_MODE = "permissive"   # Add any discovered types
    
    def __init__(
        self, 
        mode: str = "guided",
        max_entities: int = 20,
        max_relationships: int = 15,
        evolution_threshold: float = 0.3  # Only evolve if 30%+ new content
    ):
        self.mode = mode
        self.max_entities = max_entities
        self.max_relationships = max_relationships
        self.evolution_threshold = evolution_threshold

class EntityExtractor:
    """Production-grade entity extractor using __Entity__ base type and proper document hierarchy"""
    
    def __init__(self):
        # Advanced chunking strategy for document hierarchy
        self.chunk_overlap = 400  # Store overlap value for later use
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=self.chunk_overlap,  # Larger overlap for context preservation
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "],
            keep_separator=True
        )
        
        # Schema evolution tracking
        self.evolved_schemas = {}
        self.entity_similarity_threshold = 0.97
        self.text_distance_threshold = 5
        
        # Modern configuration using __Entity__ base type
        self.use_entity_base_type = True
        self.entity_base_label = "__Entity__"
    
    async def extract_from_documents(
        self,
        documents: List[Dict[str, Any]],
        user_id: str,
        graph_id: UUID,
        domain_context: Optional[str] = None,
        saved_schema: Optional[Dict[str, Any]] = None,
        allow_schema_evolution: bool = True
    ) -> Tuple[List[GraphDocument], Dict[str, Any]]:
        """
        Extract complete document hierarchy: Document → Chunks → Entities (using __Entity__)
        All node types get embeddings for comprehensive search
        """
        
        all_graph_docs = []
        all_chunks = []
        
        # Process each document
        for doc_data in documents:
            # Step 1: Create Document node
            doc_node = await self._create_document_node(doc_data, graph_id)
            
            # Step 2: Create enhanced chunks with proper ordering
            chunks = await self._create_enhanced_chunks_from_document(doc_data, graph_id)
            all_chunks.extend(chunks)
            
            # Step 3: Create chunk nodes and Document → Chunk relationships
            chunk_nodes, doc_chunk_rels = self._create_chunk_nodes_and_relationships(
                doc_node, chunks, graph_id
            )
            
            # Step 4: Determine extraction schema
            evolved_schema = await self._get_or_evolve_schema(
                doc_data.get("content", ""), user_id, saved_schema, 
                domain_context, allow_schema_evolution
            )
            
            # Step 5: Extract entities from chunks using __Entity__ base type
            entity_docs = await self._extract_entities_with_base_type(
                chunks, user_id, graph_id, evolved_schema
            )
            
            # Step 6: Create Chunk → Entity relationships (entities connect to chunks, not document)
            chunk_entity_rels = self._create_chunk_entity_relationships(
                chunk_nodes, entity_docs, graph_id
            )
            
            # Step 7: Combine all nodes and relationships for this document
            all_nodes = [doc_node] + chunk_nodes
            all_relationships = doc_chunk_rels + chunk_entity_rels
            
            # Add extracted entities
            for entity_doc in entity_docs:
                all_nodes.extend(entity_doc.nodes)
                all_relationships.extend(entity_doc.relationships)
            
            # Create comprehensive graph document
            graph_doc = GraphDocument(
                nodes=all_nodes,
                relationships=all_relationships,
                source=Document(
                    page_content=f"Document: {doc_data.get('title', 'Untitled')}",
                    metadata={
                        "document_id": doc_data.get("id"),
                        "graph_id": str(graph_id),
                        "type": "document_hierarchy"
                    }
                )
            )
            
            all_graph_docs.append(graph_doc)
        
        return all_graph_docs, {
            "total_documents": len(documents),
            "total_chunks": len(all_chunks),
            "schema": evolved_schema if 'evolved_schema' in locals() else None
        }

    async def _create_document_node(self, doc_data: Dict[str, Any], graph_id: UUID) -> Node:
        """Create Document node with proper metadata"""
        
        # Generate embedding for document (title + summary)
        doc_text = f"{doc_data.get('title', '')} {doc_data.get('summary', '')}"
        doc_embedding = None
        
        if embedding_service.is_initialized() and doc_text.strip():
            try:
                doc_embedding = await embedding_service.embed_text(doc_text)
            except Exception as e:
                logger.warning(f"Failed to generate document embedding: {e}")
        
        return Node(
            id=f"doc_{doc_data.get('id', uuid4())}",
            type="Document",
            properties={
                "document_id": doc_data.get("id"),
                "title": doc_data.get("title", ""),
                "filename": doc_data.get("filename", ""),
                "content_type": doc_data.get("content_type", ""),
                "summary": doc_data.get("summary", ""),
                "graph_id": str(graph_id),
                "created_at": doc_data.get("created_at"),
                "word_count": len(doc_data.get("content", "").split()),
                "char_count": len(doc_data.get("content", "")),
                "embedding": doc_embedding,
                "has_embedding": doc_embedding is not None
            }
        )

    async def _create_enhanced_chunks_from_document(
        self, 
        doc_data: Dict[str, Any], 
        graph_id: UUID
    ) -> List[Dict[str, Any]]:
        """Create enhanced text chunks with preserved ordering and embeddings"""
        
        content = doc_data.get("content", "")
        doc_id = doc_data.get("id", str(uuid4()))
        
        # Split text into semantic chunks
        documents = self.text_splitter.create_documents([content])
        
        enhanced_chunks = []
        char_position = 0
        
        for i, doc in enumerate(documents):
            chunk_id = f"chunk_{doc_id}_{i:04d}"  # Zero-padded for proper ordering
            
            # Generate embedding for chunk
            chunk_embedding = None
            if embedding_service.is_initialized():
                try:
                    chunk_embedding = await embedding_service.embed_text(doc.page_content)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for chunk {i}: {e}")
            
            # Calculate actual character positions
            chunk_start = char_position
            chunk_end = char_position + len(doc.page_content)
            char_position = chunk_end - self.chunk_overlap  # Account for overlap
            
            chunk_data = {
                "id": chunk_id,
                "text": doc.page_content,
                "graph_id": str(graph_id),
                "document_id": doc_id,
                "chunk_index": i,
                "chunk_order": i,  # Explicit ordering
                "char_start": chunk_start,
                "char_end": chunk_end,
                "embedding": chunk_embedding,
                "word_count": len(doc.page_content.split()),
                "sentence_count": len([s for s in doc.page_content.split('.') if s.strip()]),
                "type": "DocumentChunk",
                "has_embedding": chunk_embedding is not None
            }
            enhanced_chunks.append(chunk_data)
        
        logger.info(f"Created {len(enhanced_chunks)} chunks for document {doc_id}")
        return enhanced_chunks

    def _create_chunk_nodes_and_relationships(
        self,
        doc_node: Node,
        chunks: List[Dict[str, Any]],
        graph_id: UUID
    ) -> Tuple[List[Node], List[Relationship]]:
        """
        Create chunk nodes with enhanced relationships for better document navigation:
        
        Relationship Types:
        - FIRST_CHUNK: Document → First chunk only
        - PART_OF: Document → All other chunks  
        - NEXT_CHUNK: Chunk → Next chunk (sequential order)
        - SIMILAR: Chunk ↔ Chunk (semantic similarity, bidirectional)
        
        Example Queries:
        # Get first chunk of a document
        MATCH (d:Document)-[:FIRST_CHUNK]->(first:DocumentChunk) WHERE d.id = "doc_id"
        
        # Get all chunks in order
        MATCH (d:Document)-[:FIRST_CHUNK]->(first:DocumentChunk)
        OPTIONAL MATCH (first)-[:NEXT_CHUNK*]->(rest:DocumentChunk)
        RETURN first, rest ORDER BY rest.chunk_order
        
        # Find similar chunks
        MATCH (c1:DocumentChunk)-[s:SIMILAR]->(c2:DocumentChunk) 
        WHERE s.similarity_score > 0.9
        """
        
        chunk_nodes = []
        all_relationships = []
        
        # Sort chunks by order to ensure proper sequencing
        sorted_chunks = sorted(chunks, key=lambda x: x["chunk_order"])
        
        for i, chunk in enumerate(sorted_chunks):
            # Create chunk node with embedding
            chunk_node = Node(
                id=chunk["id"],
                type="DocumentChunk",
                properties={
                    "text": chunk["text"],
                    "graph_id": str(graph_id),
                    "document_id": chunk["document_id"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_order": chunk["chunk_order"],  # Preserved ordering
                    "word_count": chunk["word_count"],
                    "sentence_count": chunk["sentence_count"],
                    "char_start": chunk["char_start"],
                    "char_end": chunk["char_end"],
                    "embedding": chunk["embedding"],
                    "has_embedding": chunk["has_embedding"],
                    "is_first_chunk": i == 0,  # Mark first chunk
                    "is_last_chunk": i == len(sorted_chunks) - 1  # Mark last chunk
                }
            )
            chunk_nodes.append(chunk_node)
            
            # Create Document → Chunk relationships with specific types
            if i == 0:
                # First chunk gets FIRST_CHUNK relationship
                doc_chunk_rel = Relationship(
                    source=doc_node,
                    target=chunk_node,
                    type="FIRST_CHUNK",
                    properties={
                        "graph_id": str(graph_id),
                        "chunk_order": chunk["chunk_order"],
                        "sequence_number": chunk["chunk_index"],
                        "relationship_type": "first_chunk"
                    }
                )
            else:
                # Other chunks get PART_OF relationship
                doc_chunk_rel = Relationship(
                    source=doc_node,
                    target=chunk_node,
                    type="PART_OF",
                    properties={
                        "graph_id": str(graph_id),
                        "chunk_order": chunk["chunk_order"],
                        "sequence_number": chunk["chunk_index"],
                        "relationship_type": "part_of"
                    }
                )
            all_relationships.append(doc_chunk_rel)
            
            # Create NEXT_CHUNK relationships between consecutive chunks
            if i > 0:
                previous_chunk_node = chunk_nodes[i - 1]
                next_chunk_rel = Relationship(
                    source=previous_chunk_node,
                    target=chunk_node,
                    type="NEXT_CHUNK",
                    properties={
                        "graph_id": str(graph_id),
                        "from_order": sorted_chunks[i-1]["chunk_order"],
                        "to_order": chunk["chunk_order"],
                        "sequence_gap": chunk["chunk_order"] - sorted_chunks[i-1]["chunk_order"]
                    }
                )
                all_relationships.append(next_chunk_rel)
        
        # Create SIMILAR relationships between semantically similar chunks
        similar_relationships = self._create_chunk_similarity_relationships(
            chunk_nodes, graph_id
        )
        all_relationships.extend(similar_relationships)
        
        logger.info(f"Created {len(chunk_nodes)} chunk nodes with {len(all_relationships)} relationships "
                   f"(including {len(similar_relationships)} similarity relationships)")
        
        return chunk_nodes, all_relationships

    def _create_chunk_similarity_relationships(
        self,
        chunk_nodes: List[Node],
        graph_id: UUID,
        similarity_threshold: float = 0.85
    ) -> List[Relationship]:
        """Create SIMILAR relationships between semantically similar chunks"""
        
        similar_relationships = []
        
        # Only create similarity relationships if we have embeddings
        chunks_with_embeddings = [
            node for node in chunk_nodes 
            if node.properties.get("embedding") is not None
        ]
        
        if len(chunks_with_embeddings) < 2:
            logger.info("Not enough chunks with embeddings for similarity analysis")
            return similar_relationships
        
        # Compare each chunk with every other chunk
        for i, chunk1 in enumerate(chunks_with_embeddings):
            for j, chunk2 in enumerate(chunks_with_embeddings[i+1:], i+1):
                try:
                    embedding1 = chunk1.properties["embedding"]
                    embedding2 = chunk2.properties["embedding"]
                    
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(embedding1, embedding2)
                    
                    if similarity >= similarity_threshold:
                        # Create bidirectional SIMILAR relationships
                        similar_rel1 = Relationship(
                            source=chunk1,
                            target=chunk2,
                            type="SIMILAR",
                            properties={
                                "graph_id": str(graph_id),
                                "similarity_score": similarity,
                                "similarity_type": "semantic",
                                "threshold": similarity_threshold
                            }
                        )
                        
                        similar_rel2 = Relationship(
                            source=chunk2,
                            target=chunk1,
                            type="SIMILAR",
                            properties={
                                "graph_id": str(graph_id),
                                "similarity_score": similarity,
                                "similarity_type": "semantic",
                                "threshold": similarity_threshold
                            }
                        )
                        
                        similar_relationships.extend([similar_rel1, similar_rel2])
                        
                        logger.debug(f"Created SIMILAR relationship between chunks "
                                   f"{chunk1.properties.get('chunk_order', 'unknown')} and "
                                   f"{chunk2.properties.get('chunk_order', 'unknown')} "
                                   f"(similarity: {similarity:.3f})")
                        
                except Exception as e:
                    logger.warning(f"Failed to calculate similarity between chunks: {e}")
                    continue
        
        return similar_relationships

    async def _get_or_evolve_schema(
        self,
        text: str,
        user_id: str,
        saved_schema: Optional[Dict[str, Any]],
        domain_context: Optional[str],
        allow_evolution: bool
    ) -> Dict[str, Any]:
        """Get or evolve schema for entity extraction"""
        
        if saved_schema and saved_schema.get("entities"):
            if allow_evolution:
                return await self._evolve_schema_with_new_content(
                    text=text,
                    user_id=user_id,
                    base_schema=saved_schema,
                    domain_context=domain_context
                )
            else:
                return saved_schema
        else:
            return await self._evolve_schema_from_text(text, user_id, domain_context)

    async def _extract_entities_with_base_type(
        self,
        chunks: List[Dict[str, Any]],
        user_id: str,
        graph_id: UUID,
        schema: Dict[str, Any]
    ) -> List[GraphDocument]:
        """Extract entities using __Entity__ base type with dynamic specific types"""
        
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
                        "chunk_index": chunk["chunk_index"],
                        "document_id": chunk["document_id"]
                    }
                )
                for chunk in batch_chunks
            ]
            
            try:
                # Use structured output for consistent extraction
                batch_docs = await llm_service.graph_transformer.aconvert_to_graph_documents(
                    documents
                )
                
                # Transform entities to use __Entity__ base type with specific labels
                for doc in batch_docs:
                    transformed_nodes = []
                    
                    for node in doc.nodes:
                        # Transform to use __Entity__ base type + specific type
                        entity_type = node.type if hasattr(node, 'type') and node.type else "Entity"
                        
                        # Create node with single type first to pass LangChain validation
                        transformed_node = Node(
                            id=node.id,
                            type=entity_type,  # Start with specific type only
                            properties={
                                **(node.properties or {}),
                                "graph_id": str(graph_id),
                                "extraction_source": "llm_structured",
                                "confidence": self._calculate_entity_confidence(node, schema),
                                "entity_type": entity_type,  # Store specific type
                                "specific_labels": [entity_type],  # For multiple labels
                                "has_embedding": False  # Will be set during embedding generation
                            }
                        )
                        
                        # After node creation, manually set the type to include both labels
                        # This bypasses LangChain's validation while giving enhanced_graph_service what it needs
                        transformed_node.type = [self.entity_base_label, entity_type]
                        transformed_nodes.append(transformed_node)
                    
                    # Update relationships to use transformed nodes
                    for rel in doc.relationships:
                        if not hasattr(rel, 'properties') or not rel.properties:
                            rel.properties = {}
                        rel.properties.update({
                            "graph_id": str(graph_id),
                            "extraction_source": "llm_structured"
                        })
                    
                    # Replace nodes in document
                    doc.nodes = transformed_nodes
                
                all_graph_docs.extend(batch_docs)
                
            except Exception as e:
                logger.error(f"Batch extraction failed: {e}")
                continue
        
        return all_graph_docs

    def _create_chunk_entity_relationships(
        self,
        chunk_nodes: List[Node],
        entity_docs: List[GraphDocument],
        graph_id: UUID
    ) -> List[Relationship]:
        """Create Chunk → Entity relationships (entities connect to chunks, not documents)"""
        
        chunk_entity_relationships = []
        
        # Create mapping of chunk IDs to chunk nodes
        chunk_id_to_node = {node.id: node for node in chunk_nodes}
        
        for entity_doc in entity_docs:
            # Get source chunk ID from document metadata
            source_chunk_id = None
            if hasattr(entity_doc, 'source') and entity_doc.source and hasattr(entity_doc.source, 'metadata'):
                source_chunk_id = entity_doc.source.metadata.get("chunk_id")
            
            if source_chunk_id and source_chunk_id in chunk_id_to_node:
                chunk_node = chunk_id_to_node[source_chunk_id]
                
                for entity_node in entity_doc.nodes:
                    # Create Chunk → Entity relationship
                    chunk_entity_rel = Relationship(
                        source=chunk_node,
                        target=entity_node,
                        type="MENTIONS",
                        properties={
                            "graph_id": str(graph_id),
                            "extraction_source": "chunk_analysis",
                            "confidence": entity_node.properties.get("confidence", 1.0)
                        }
                    )
                    chunk_entity_relationships.append(chunk_entity_rel)
        
        logger.info(f"Created {len(chunk_entity_relationships)} chunk-entity relationships")
        return chunk_entity_relationships

    async def _evolve_schema_with_new_content(
        self,
        text: str,
        user_id: str,
        base_schema: Dict[str, Any],
        domain_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evolve existing schema by discovering new entity/relationship types in new content"""
        
        try:
            # Step 1: Analyze new text for potential new entity/relationship types
            new_discoveries = await self._discover_new_schema_elements(
                text=text,
                user_id=user_id,
                existing_schema=base_schema,
                domain_context=domain_context
            )
            
            # Step 2: Merge discovered types with existing schema
            evolved_schema = self._merge_schema_discoveries(base_schema, new_discoveries)
            
            # Step 3: Validate and clean merged schema
            return self._validate_evolved_schema(evolved_schema)
            
        except Exception as e:
            logger.warning(f"Schema evolution failed, using base schema: {e}")
            return base_schema

    async def _discover_new_schema_elements(
        self,
        text: str,
        user_id: str,
        existing_schema: Dict[str, Any],
        domain_context: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Discover new entity/relationship types not in existing schema"""
        
        if not llm_service.is_initialized():
            return {"entities": [], "relationships": []}
        
        existing_entities = existing_schema.get("entities", [])
        existing_relationships = existing_schema.get("relationships", [])
        
        discovery_prompt = f"""
        Analyze this text for NEW entity types and relationship types that are NOT already covered by the existing schema.
        
        EXISTING SCHEMA:
        - Entity types: {existing_entities}
        - Relationship types: {existing_relationships}
        
        NEW TEXT:
        {text[:2000]}...
        
        Find ONLY NEW types that would be valuable additions to the schema. Do not repeat existing types.
        
        Respond with JSON:
        {{
            "new_entities": ["Type1", "Type2"],
            "new_relationships": ["NEW_RELATION_1", "NEW_RELATION_2"],
            "reasoning": "Brief explanation of why these new types are needed"
        }}
        
        If no new types are needed, return empty lists.
        """
        
        try:
            response = await llm_service.llm.ainvoke([
                {"role": "system", "content": "You are a knowledge graph schema analyst. Return valid JSON only."},
                {"role": "user", "content": discovery_prompt}
            ])
            
            # Parse response
            schema_text = response.content
            if "```json" in schema_text:
                schema_text = schema_text.split("```json")[1].split("```")[0]
            elif "```" in schema_text:
                schema_text = schema_text.split("```")[1].split("```")[0]
            
            discoveries = json.loads(schema_text)
            
            logger.info(f"Schema discovery: {len(discoveries.get('new_entities', []))} new entities, {len(discoveries.get('new_relationships', []))} new relationships")
            
            return {
                "entities": discoveries.get("new_entities", []),
                "relationships": discoveries.get("new_relationships", [])
            }
            
        except Exception as e:
            logger.warning(f"Schema discovery failed: {e}")
            return {"entities": [], "relationships": []}

    def _merge_schema_discoveries(
        self,
        base_schema: Dict[str, Any],
        discoveries: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Merge new discoveries with existing schema"""
        
        # Combine entities
        all_entities = set(base_schema.get("entities", []))
        all_entities.update(discoveries.get("entities", []))
        
        # Combine relationships
        all_relationships = set(base_schema.get("relationships", []))
        all_relationships.update(discoveries.get("relationships", []))
        
        return {
            "entities": sorted(list(all_entities)),
            "relationships": sorted(list(all_relationships))
        }
    
    def _validate_evolved_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate evolved schema and apply limits"""
        
        entities = schema.get("entities", [])
        relationships = schema.get("relationships", [])
        
        # Limit schema size to prevent bloat
        MAX_ENTITIES = 20
        MAX_RELATIONSHIPS = 15
        
        if len(entities) > MAX_ENTITIES:
            logger.warning(f"Schema has {len(entities)} entities, limiting to {MAX_ENTITIES}")
            entities = entities[:MAX_ENTITIES]
        
        if len(relationships) > MAX_RELATIONSHIPS:
            logger.warning(f"Schema has {len(relationships)} relationships, limiting to {MAX_RELATIONSHIPS}")
            relationships = relationships[:MAX_RELATIONSHIPS]
        
        return {
            "entities": entities,
            "relationships": relationships
        }

    async def _update_saved_schema_if_evolved(
        self,
        graph_id: UUID,
        original_schema: Dict[str, Any],
        evolved_schema: Dict[str, Any]
    ) -> None:
        """Update the saved schema in database if it evolved"""
        
        # Check if schema actually changed
        original_entities = set(original_schema.get("entities", []))
        evolved_entities = set(evolved_schema.get("entities", []))
        
        original_relationships = set(original_schema.get("relationships", []))
        evolved_relationships = set(evolved_schema.get("relationships", []))
        
        if original_entities != evolved_entities or original_relationships != evolved_relationships:
            try:
                # Update schema in database
                from app.core.database import async_session_maker
                from app.models.graph import KnowledgeGraph
                from sqlalchemy import update
                
                async with async_session_maker() as db:
                    await db.execute(
                        update(KnowledgeGraph)
                        .where(KnowledgeGraph.id == graph_id)
                        .values(schema_config=evolved_schema)
                    )
                    await db.commit()
                    
                    logger.info(f"Updated schema for graph {graph_id}: +{len(evolved_entities - original_entities)} entities, +{len(evolved_relationships - original_relationships)} relationships")
                    
            except Exception as e:
                logger.warning(f"Failed to update saved schema: {e}")

    async def _create_enhanced_chunks(
        self, 
        text: str, 
        graph_id: UUID
    ) -> List[Dict[str, Any]]:
        """Legacy method - use _create_enhanced_chunks_from_document for new document hierarchy"""
        
        # Create a fake document structure for backward compatibility
        doc_data = {
            "id": str(uuid4()),
            "content": text,
            "title": "Legacy Document",
            "content_type": "text/plain"
        }
        
        return await self._create_enhanced_chunks_from_document(doc_data, graph_id)

    async def learn_schema_from_text(
        self, 
        text: str, 
        user_id: str,
        provider: str = "openai",
        use_diffbot: bool = True
    ) -> Dict[str, List[str]]:
        """Learn schema from text using both Diffbot and LLM approaches"""
        
        try:
            logger.info("Starting schema learning from text...")
            
            # Step 1: Try to get schema insights from Diffbot (if available)
            diffbot_schema = {}
            if use_diffbot:
                diffbot_schema = await self._learn_schema_from_diffbot(text, user_id)
            
            # Step 2: Get schema insights from LLM
            llm_schema = await self._learn_schema_from_llm(text, user_id, provider)
            
            # Step 3: Combine and consolidate schemas
            combined_schema = self._combine_schemas(diffbot_schema, llm_schema)
            
            logger.info(f"Learned schema: {len(combined_schema.get('entities', []))} entities, {len(combined_schema.get('relationships', []))} relationships")
            return combined_schema
            
        except Exception as e:
            logger.error(f"Schema learning failed: {e}")
            return self._get_fallback_schema()
    
    async def _learn_schema_from_diffbot(self, text: str, user_id: str) -> Dict[str, List[str]]:
        """Learn schema from Diffbot graph extraction (if Diffbot service is available)"""
        
        try:
            # Check if diffbot service is available
            try:
                from app.services.diffbot_graph_service import diffbot_graph_service
                
                # Extract with Diffbot to see what types it finds
                sample_graph_docs = await diffbot_graph_service.extract_graph_documents(
                    text=text[:1000],  # Use sample of text
                    user_id=user_id,
                    graph_id=UUID("00000000-0000-0000-0000-000000000000")  # Dummy UUID for schema learning
                )
                
                entity_types = set()
                relationship_types = set()
                
                for graph_doc in sample_graph_docs:
                    # Extract entity types from nodes
                    for node in graph_doc.nodes:
                        if hasattr(node, 'type') and node.type:
                            if isinstance(node.type, list):
                                entity_types.update(node.type)
                            else:
                                entity_types.add(node.type)
                    
                    # Extract relationship types
                    for rel in graph_doc.relationships:
                        if hasattr(rel, 'type') and rel.type:
                            relationship_types.add(rel.type)
                
                logger.info(f"Diffbot schema learning: {len(entity_types)} entities, {len(relationship_types)} relationships")
                
                return {
                    "entities": sorted(list(entity_types)),
                    "relationships": sorted(list(relationship_types))
                }
                
            except ImportError:
                logger.info("Diffbot service not available, skipping Diffbot schema learning")
                return {"entities": [], "relationships": []}
            
        except Exception as e:
            logger.warning(f"Diffbot schema learning failed: {e}")
            return {"entities": [], "relationships": []}
    
    async def _learn_schema_from_llm(
        self, 
        text: str, 
        user_id: str,
        provider: str = "openai"
    ) -> Dict[str, List[str]]:
        """Learn schema from LLM analysis"""
        
        try:
            # Initialize LLM for schema learning
            if not llm_service.is_initialized():
                await llm_service.initialize_llm(user_id, provider)
            
            schema_prompt = f"""
            Analyze this text and identify the optimal entity types and relationship types for knowledge graph extraction.

            Text Sample: {text[:3000]}...
        
            Consider:
            1. What are the main types of entities mentioned?
            2. What relationships exist between these entities?
            3. What properties would be valuable to extract?
        
            Focus on entities and relationships that appear multiple times or are central to the meaning.
            """
            
            # Use LLM to generate schema
            response = await llm_service.llm.ainvoke([
                {"role": "system", "content": "You are a knowledge graph schema expert. Return valid JSON only."},
                {"role": "user", "content": schema_prompt}
            ])
            
            # Parse JSON response
            try:
                schema_text = response.content
                
                # Clean up response text
                if "```json" in schema_text:
                    schema_text = schema_text.split("```json")[1].split("```")[0]
                elif "```" in schema_text:
                    schema_text = schema_text.split("```")[1].split("```")[0]
                
                schema = json.loads(schema_text)
                
                # Validate schema structure
                if "entities" in schema and "relationships" in schema:
                    logger.info(f"LLM schema learning: {len(schema['entities'])} entities, {len(schema['relationships'])} relationships")
                    # Validate and clean schema
                    return self._validate_and_clean_schema(schema)
                else:
                    logger.warning("Invalid schema format from LLM, using defaults")
                    return self._get_fallback_schema()

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse schema JSON from LLM: {e}")
                return self._get_fallback_schema()

        except Exception as e:
            logger.error(f"LLM schema learning failed: {e}")
            return self._get_fallback_schema()

    def _combine_schemas(
        self, 
        diffbot_schema: Dict[str, List[str]], 
        llm_schema: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Combine and deduplicate schemas from different sources"""
        
        # Combine entity types
        all_entities = set()
        all_entities.update(diffbot_schema.get("entities", []))
        all_entities.update(llm_schema.get("entities", []))
        
        # Combine relationship types
        all_relationships = set()
        all_relationships.update(diffbot_schema.get("relationships", []))
        all_relationships.update(llm_schema.get("relationships", []))
        
        # Add default types if none found
        if not all_entities:
            all_entities.update(self._get_fallback_schema()["entities"])

        if not all_relationships:
            all_relationships.update(self._get_fallback_schema()["relationships"])

        # Sort and limit to reasonable numbers
        final_entities = sorted(list(all_entities))[:12]  # Max 12 entity types
        final_relationships = sorted(list(all_relationships))[:10]  # Max 10 relationship types
        
        return {
            "entities": final_entities,
            "relationships": final_relationships
        }
    
    async def _evolve_schema_from_text(
        self,
        text: str,
        user_id: str,
        domain_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Dynamic schema evolution using existing learn_schema_from_text method"""
        
        # Check if we have existing schema for this domain
        schema_key = f"{user_id}_{domain_context or 'general'}"
        
        if schema_key in self.evolved_schemas:
            base_schema = self.evolved_schemas[schema_key]
        else:
            # Use the learn_schema_from_text method
            base_schema = await self.learn_schema_from_text(
                text=text,
                user_id=user_id,
                provider="openai",
                use_diffbot=True
            )
            self.evolved_schemas[schema_key] = base_schema
        
        # Evolve schema based on new content (optional enhancement)
        evolved_schema = await self._refine_schema_with_context(text, base_schema)
        self.evolved_schemas[schema_key] = evolved_schema
        
        return evolved_schema

    async def _refine_schema_with_context(
        self,
        text: str,
        base_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Refine existing schema with new text context"""
        
        # For now, just return the base schema
        # Future enhancement: analyze text to add new entity/relationship types
        return base_schema
    
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
    
    def _merge_entities(self, node1: Node, node2: Node) -> Node:
        """Merge two entity nodes"""
        
        # Use the node with more properties as the base
        if hasattr(node1, 'properties') and hasattr(node2, 'properties'):
            if len(node1.properties or {}) >= len(node2.properties or {}):
                primary, secondary = node1, node2
            else:
                primary, secondary = node2, node1
        else:
            primary, secondary = node1, node2
        
        # Merge properties
        merged_properties = dict(primary.properties) if hasattr(primary, 'properties') and primary.properties else {}
        if hasattr(secondary, 'properties') and secondary.properties:
            for key, value in secondary.properties.items():
                if key not in merged_properties and value:
                    merged_properties[key] = value
        
        # Create merged node
        merged_node = Node(
            id=primary.id,
            type=primary.type if hasattr(primary, 'type') else "Entity",
            properties=merged_properties
        )
        
        # If we don't have proper type, set it to use both labels
        if not hasattr(primary, 'type') or not isinstance(primary.type, list):
            merged_node.type = [self.entity_base_label, "Entity"]
        
        return merged_node
    
    def _rebuild_graph_documents_with_deduplicated_entities(
        self,
        graph_documents: List[GraphDocument],
        entity_map: Dict[str, Dict[str, Any]]
    ) -> List[GraphDocument]:
        """Rebuild graph documents with deduplicated entities"""
        
        # Create mapping from old node IDs to new nodes
        node_id_mapping = {}
        unique_nodes = []
        
        for entity_data in entity_map.values():
            node = entity_data["node"]
            unique_nodes.append(node)
            # Map all variants to this single node
            for variant in entity_data["variants"]:
                node_id_mapping[variant] = node.id
        
        # Rebuild documents
        rebuilt_docs = []
        for doc in graph_documents:
            # Use deduplicated nodes
            new_nodes = unique_nodes.copy()
            
            # Update relationships to use deduplicated nodes
            updated_relationships = []
            for rel in doc.relationships:
                source_id = node_id_mapping.get(rel.source.id, rel.source.id)
                target_id = node_id_mapping.get(rel.target.id, rel.target.id)
                
                # Find the actual nodes
                source_node = next((n for n in new_nodes if n.id == source_id), rel.source)
                target_node = next((n for n in new_nodes if n.id == target_id), rel.target)
                
                updated_rel = Relationship(
                    source=source_node,
                    target=target_node,
                    type=rel.type,
                    properties=rel.properties if hasattr(rel, 'properties') else {}
                )
                updated_relationships.append(updated_rel)
            
            # Create new document
            rebuilt_doc = GraphDocument(
                nodes=new_nodes,
                relationships=updated_relationships,
                source=doc.source if hasattr(doc, 'source') else None
            )
            rebuilt_docs.append(rebuilt_doc)
        
        return rebuilt_docs

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

        graph1 = node1.properties.get("graph_id") if hasattr(node1, 'properties') and node1.properties else None
        graph2 = node2.properties.get("graph_id") if hasattr(node2, 'properties') and node2.properties else None

        if name1 == name2 and graph1 == graph2:
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
    
    # Shouldn't the key creation get refactored???
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
entity_extractor = EntityExtractor()
