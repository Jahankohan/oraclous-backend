from typing import List, Dict, Any, Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
import asyncio
import json
from uuid import UUID, uuid4
from app.core.logging import get_logger
from app.services.llm_service import llm_service
from app.services.diffbot_graph_service import diffbot_graph_service

logger = get_logger(__name__)

class EntityExtractor:
    """Enhanced entity extractor with Diffbot Graph Transformer + LLM fusion"""
    
    def __init__(self, neo4j_graph=None):
        self.neo4j_graph = neo4j_graph
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Larger chunks for Diffbot
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    async def extract_entities_hybrid(
        self, 
        text: str,
        user_id: str,
        graph_id: UUID,
        schema: Optional[Dict[str, Any]] = None,
        use_diffbot: bool = True,
        provider: str = "openai",
        model: str = "gpt-4o-mini"
    ) -> List[GraphDocument]:
        """Extract entities using both Diffbot Graph Transformer and LLM, then fuse results"""
        
        try:
            logger.info(f"Starting hybrid extraction for {len(text)} characters")
            
            # Step 1: Extract with Diffbot Graph Transformer (if available)
            diffbot_results = []
            if use_diffbot:
                logger.info("Running Diffbot Graph Transformer extraction...")
                diffbot_results = await diffbot_graph_service.extract_graph_documents(
                    text=text,
                    user_id=user_id,
                    graph_id=graph_id,
                    schema=schema
                )
                logger.info(f"Diffbot extracted {len(diffbot_results)} graph documents")
            
            # Step 2: Extract with LLM
            logger.info("Running LLM extraction...")
            llm_results = await self._extract_with_llm(
                text, user_id, graph_id, schema, provider, model
            )
            logger.info(f"LLM extracted {len(llm_results)} graph documents")
            
            # Step 3: Combine results
            logger.info("Combining Diffbot and LLM results...")
            combined_results = diffbot_results + llm_results
            
            # Step 4: Deduplicate
            if combined_results:
                final_results = self._deduplicate_entities(combined_results)
                logger.info(f"Final result: {len(final_results)} deduplicated graph documents")
                return final_results
            else:
                logger.warning("No results from either Diffbot or LLM")
                return []
            
        except Exception as e:
            logger.error(f"Hybrid extraction failed: {e}")
            # Fallback to LLM-only extraction
            logger.info("Falling back to LLM-only extraction")
            return await self._extract_with_llm(text, user_id, graph_id, schema, provider, model)
    
    async def _extract_with_llm(
        self,
        text: str,
        user_id: str,
        graph_id: UUID,
        schema: Optional[Dict[str, Any]] = None,
        provider: str = "openai",
        model: str = "gpt-4o-mini"
    ) -> List[GraphDocument]:
        """Extract entities using LLM (existing logic)"""
        
        # Initialize LLM if needed
        if not llm_service.is_initialized():
            success = await llm_service.initialize_llm(
                user_id=user_id, provider=provider, model=model
            )
            if not success:
                raise ValueError(f"Failed to initialize LLM: {provider}")
        
        # Set schema if provided
        if schema:
            llm_service.set_schema(schema)
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks for LLM processing")
        
        # Process chunks in batches
        batch_size = 3
        all_graph_documents = []
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        "graph_id": str(graph_id),
                        "chunk_index": i + j,
                        "source": "llm_extraction"
                    }
                )
                for j, chunk in enumerate(batch_chunks)
            ]
            
            try:
                batch_graph_docs = await llm_service.graph_transformer.aconvert_to_graph_documents(
                    documents
                )
                
                # Add graph_id to all nodes and relationships
                for graph_doc in batch_graph_docs:
                    for node in graph_doc.nodes:
                        if not hasattr(node, 'properties') or not node.properties:
                            node.properties = {}
                        node.properties["graph_id"] = str(graph_id)
                        node.properties["extraction_source"] = "llm"
                    
                    for rel in graph_doc.relationships:
                        if not hasattr(rel, 'properties') or not rel.properties:
                            rel.properties = {}
                        rel.properties["graph_id"] = str(graph_id)
                        rel.properties["extraction_source"] = "llm"
                
                all_graph_documents.extend(batch_graph_docs)
                await asyncio.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error processing LLM batch {i//batch_size + 1}: {e}")
                continue
        
        return all_graph_documents

    async def _fuse_extraction_results(
        self, 
        diffbot_results: Dict[str, Any], 
        llm_results: List[GraphDocument],
        graph_id: UUID
    ) -> List[GraphDocument]:
        """Fuse Diffbot and LLM extraction results"""
        
        if not diffbot_results.get("entities"):
            logger.info("No Diffbot results to fuse, returning LLM results only")
            return llm_results
        
        # Convert Diffbot entities to GraphDocument format
        diffbot_graph_docs = self._convert_diffbot_to_graph_documents(
            diffbot_results, graph_id
        )
        
        # Merge with LLM results
        all_results = llm_results + diffbot_graph_docs
        
        # Deduplicate similar entities
        deduplicated_results = self._deduplicate_entities(all_results)
        
        logger.info(f"Fused results: {len(llm_results)} LLM + {len(diffbot_graph_docs)} Diffbot = {len(deduplicated_results)} final")
        
        return deduplicated_results
    
    def _convert_diffbot_to_graph_documents(
        self, 
        diffbot_results: Dict[str, Any], 
        graph_id: UUID
    ) -> List[GraphDocument]:
        """Convert Diffbot results to GraphDocument format"""
        
        nodes = []
        relationships = []
        
        # Convert Diffbot entities to nodes
        for entity in diffbot_results.get("entities", []):
            if entity["name"]:  # Skip empty entities
                node = Node(
                    id=f"diffbot_{uuid4().hex[:8]}",
                    type=entity["type"],
                    properties={
                        "name": entity["name"],
                        "confidence": entity["confidence"],
                        "wiki_uri": entity.get("wiki_uri"),
                        "summary": entity.get("summary", ""),
                        "graph_id": str(graph_id),
                        "extraction_source": "diffbot"
                    }
                )
                nodes.append(node)
        
        # Convert Diffbot facts to relationships
        node_lookup = {node.properties["name"]: node for node in nodes}
        
        for fact in diffbot_results.get("facts", []):
            subject_name = fact.get("subject", {}).get("name")
            object_name = fact.get("object", {}).get("name")
            predicate = fact.get("predicate", {}).get("name", "RELATED_TO")
            
            if subject_name and object_name:
                # Find or create nodes
                source_node = node_lookup.get(subject_name)
                target_node = node_lookup.get(object_name)
                
                if source_node and target_node:
                    relationship = Relationship(
                        source=source_node,
                        target=target_node,
                        type=predicate.upper().replace(" ", "_"),
                        properties={
                            "confidence": fact.get("confidence", 0.0),
                            "graph_id": str(graph_id),
                            "extraction_source": "diffbot"
                        }
                    )
                    relationships.append(relationship)
        
        # Create GraphDocument
        if nodes or relationships:
            return [GraphDocument(nodes=nodes, relationships=relationships)]
        else:
            return []
    
    def _deduplicate_entities(self, graph_documents: List[GraphDocument]) -> List[GraphDocument]:
        """Deduplicate similar entities from different sources"""
        
        all_nodes = []
        all_relationships = []
        source_documents = []
        
        # Collect all nodes, relationships, and source documents
        for doc in graph_documents:
            all_nodes.extend(doc.nodes)
            all_relationships.extend(doc.relationships)
            if hasattr(doc, 'source') and doc.source:
                source_documents.append(doc.source)
        
        # DEBUG: Log node properties to understand structure
        for i, node in enumerate(all_nodes[:3]):  # Log first 3 nodes
            logger.info(f"Node {i}: ID={node.id}, Type={node.type}, Properties={node.properties}")
        
        # IMPROVED: Get node name from multiple possible sources
        unique_nodes = {}
        
        for node in all_nodes:
            # Try multiple ways to get node name
            name = None
            
            if hasattr(node, 'properties') and node.properties:
                name = node.properties.get("name") or node.properties.get("id")
            
            # Fallback to node.id if no name in properties
            if not name:
                name = getattr(node, 'id', str(node))
            
            # Clean and normalize the name
            if name:
                name = str(name).strip()
                if name.startswith("http://"):
                    # Extract entity name from Wikidata URLs
                    name = name.split("/")[-1]
                
                name_key = name.lower()
                
                if name_key and name_key not in unique_nodes:
                    # Ensure node has proper name property
                    if not hasattr(node, 'properties'):
                        node.properties = {}
                    node.properties["name"] = name
                    node.id = name.replace(" ", "_").lower()
                    unique_nodes[name_key] = node
                    logger.info(f"Added node: {name} -> {node.id}")
        
        # IMPROVED: Update relationships with proper node matching
        updated_relationships = []
        
        for rel in all_relationships:
            # Get source and target names
            source_name = None
            target_name = None
            
            # Try to get source name
            if hasattr(rel.source, 'properties') and rel.source.properties:
                source_name = rel.source.properties.get("name")
            if not source_name:
                source_name = getattr(rel.source, 'id', None)
            
            # Try to get target name  
            if hasattr(rel.target, 'properties') and rel.target.properties:
                target_name = rel.target.properties.get("name")
            if not target_name:
                target_name = getattr(rel.target, 'id', None)
            
            # Clean names
            if source_name and target_name:
                source_name = str(source_name).strip()
                target_name = str(target_name).strip()
                
                # Handle Wikidata URLs
                if source_name.startswith("http://"):
                    source_name = source_name.split("/")[-1]
                if target_name.startswith("http://"):
                    target_name = target_name.split("/")[-1]
                
                source_key = source_name.lower()
                target_key = target_name.lower()
                
                logger.info(f"Processing relationship: {source_name} -[{rel.type}]-> {target_name}")
                
                if source_key in unique_nodes and target_key in unique_nodes:
                    rel.source = unique_nodes[source_key]
                    rel.target = unique_nodes[target_key]
                    updated_relationships.append(rel)
                    logger.info(f"✅ Relationship added: {source_name} -[{rel.type}]-> {target_name}")
                else:
                    logger.warning(f"❌ Skipping relationship - missing nodes: {source_name} -> {target_name}")
                    logger.warning(f"Available nodes: {list(unique_nodes.keys())}")
        
        # Return results
        final_nodes = list(unique_nodes.values())
        
        if source_documents:
            combined_source = source_documents[0]
        else:
            combined_source = Document(
                page_content="Combined extraction results",
                metadata={"source": "combined_extraction"}
            )
        
        logger.info(f"Final result: {len(final_nodes)} nodes and {len(updated_relationships)} relationships")
        
        return [GraphDocument(
            nodes=final_nodes, 
            relationships=updated_relationships,
            source=combined_source
        )]

    async def learn_schema_from_text(
        self, 
        text: str, 
        user_id: str,
        provider: str = "openai",
        use_diffbot: bool = True
    ) -> Dict[str, List[str]]:
        """Enhanced schema learning using both Diffbot and LLM"""
        
        try:
            logger.info("Starting enhanced schema learning...")
            
            # Step 1: Try to get schema insights from Diffbot
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
            logger.error(f"Enhanced schema learning failed: {e}")
            return self._get_default_schema()

    async def _learn_schema_from_diffbot(self, text: str, user_id: str) -> Dict[str, List[str]]:
        """Learn schema from Diffbot graph extraction"""
        
        try:
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
            
        except Exception as e:
            logger.warning(f"Diffbot schema learning failed: {e}")
            return {"entities": [], "relationships": []}

    async def _learn_schema_from_llm(
        self, 
        text: str, 
        user_id: str,
        provider: str = "openai"
    ) -> Dict[str, List[str]]:
        """Learn schema from LLM analysis (original method)"""
        
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
                    logger.info(f"LLM schema learning: {len(schema['entities'])} entities, {len(schema['relationships'])} relationships")
                    return schema
                else:
                    logger.warning("Invalid schema format from LLM, using defaults")
                    return self._get_default_schema()
                    
            except json.JSONDecodeError:
                logger.warning("Failed to parse schema JSON from LLM")
                return self._get_default_schema()
                
        except Exception as e:
            logger.error(f"LLM schema learning failed: {e}")
            return self._get_default_schema()
    
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
            all_entities.update(self._get_default_schema()["entities"])
        
        if not all_relationships:
            all_relationships.update(self._get_default_schema()["relationships"])
        
        # Sort and limit to reasonable numbers
        final_entities = sorted(list(all_entities))[:12]  # Max 12 entity types
        final_relationships = sorted(list(all_relationships))[:10]  # Max 10 relationship types
        
        return {
            "entities": final_entities,
            "relationships": final_relationships
        }

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


entity_extractor = EntityExtractor()
