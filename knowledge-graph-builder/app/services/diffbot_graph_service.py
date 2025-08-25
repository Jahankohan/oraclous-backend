from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_community.graphs.graph_document import GraphDocument
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
import asyncio
from uuid import UUID
from app.core.config import settings
from app.core.logging import get_logger
from app.services.credential_service import credential_service

logger = get_logger(__name__)

class DiffbotGraphService:
    """Service for Diffbot Graph Transformation (Neo4j LLM Graph Builder approach)"""
    
    def __init__(self):
        self.diffbot_transformer = None
        self.is_available = False
    
    async def initialize_diffbot(self, user_id: str) -> bool:
        """Initialize Diffbot Graph Transformer"""
        
        try:
            # Get Diffbot API key
            api_key = await self._get_api_key(user_id)
            if not api_key:
                logger.warning("No Diffbot API key available")
                self.is_available = False
                return False
            
            # Initialize DiffbotGraphTransformer
            self.diffbot_transformer = DiffbotGraphTransformer(
                diffbot_api_key=api_key,
                extract_types=["entities", "facts"]
            )
            
            self.is_available = True
            logger.info("Diffbot Graph Transformer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Diffbot Graph Transformer: {e}")
            self.is_available = False
            return False
    
    async def extract_graph_documents(
        self, 
        text: str,
        user_id: str,
        graph_id: UUID,
        schema: Optional[Dict[str, Any]] = None
    ) -> List[GraphDocument]:
        """Extract graph documents using Diffbot (Neo4j LLM Graph Builder approach)"""
        
        try:
            # Initialize if not already done
            if not self.is_available:
                success = await self.initialize_diffbot(user_id)
                if not success:
                    logger.warning("Diffbot not available, returning empty results")
                    return []
            
            # Set schema if provided
            if schema and self.diffbot_transformer:
                entities = schema.get("entities", [])
                relationships = schema.get("relationships", [])
                
                if entities:
                    self.diffbot_transformer.allowed_nodes = entities
                if relationships:
                    self.diffbot_transformer.allowed_relationships = relationships
            
            # Create document
            documents = [Document(
                page_content=text,
                metadata={
                    "graph_id": str(graph_id),
                    "source": "diffbot_extraction"
                }
            )]
            
            # Extract graph documents
            logger.info("Extracting graph documents with Diffbot Graph Transformer...")
            graph_documents = self.diffbot_transformer.convert_to_graph_documents(
                documents
            )

            logger.debug(f"Extracted {len(graph_documents)} graph documents")
            logger.debug(f"Graph documents: {graph_documents}")
            logger.debug(f"Graph documents types: {[type(doc) for doc in graph_documents]}")
            
            # Add graph_id to all nodes and relationships
            for graph_doc in graph_documents:
                # Add graph_id to all nodes
                for node in graph_doc.nodes:
                    if not hasattr(node, 'properties') or not node.properties:
                        node.properties = {}
                    node.properties["graph_id"] = str(graph_id)
                    node.properties["extraction_source"] = "diffbot"
                
                # Add graph_id to all relationships
                for rel in graph_doc.relationships:
                    if not hasattr(rel, 'properties') or not rel.properties:
                        rel.properties = {}
                    rel.properties["graph_id"] = str(graph_id)
                    rel.properties["extraction_source"] = "diffbot"
            
            logger.info(f"Diffbot extracted {len(graph_documents)} graph documents")
            return graph_documents
            
        except Exception as e:
            logger.error(f"Diffbot graph extraction failed: {e}")
            return []
    
    async def _get_api_key(self, user_id: str) -> Optional[str]:
        """Get Diffbot API key from credentials or config"""
        
        try:
            # Try to get from user credentials first
            credentials = await credential_service.get_user_credentials(user_id, "diffbot")
            if credentials and credentials.get("access_token"):
                return credentials["access_token"]
        except Exception as e:
            logger.debug(f"Could not get user Diffbot credentials: {e}")
            return settings.DIFFBOT_API_KEY
        
        # Fallback to service-level key
        api_key = settings.DIFFBOT_API_KEY
        if api_key and api_key != "your_diffbot_api_key_here":
            return api_key
        
        return None
    
    def set_schema(self, schema_config: Dict[str, Any]):
        """Configure allowed entities and relationships for Diffbot"""
        if self.diffbot_transformer and schema_config:
            entities = schema_config.get("entities", [])
            relationships = schema_config.get("relationships", [])
            
            if entities:
                self.diffbot_transformer.allowed_nodes = entities
            if relationships:
                self.diffbot_transformer.allowed_relationships = relationships

diffbot_graph_service = DiffbotGraphService()
