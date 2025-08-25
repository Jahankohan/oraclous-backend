# Knowledge Graph Builder Integration - Implementation Plan

## 🎯 Project Overview

**Objective**: Integrate Neo4j LLM Graph Builder capabilities into the Oraclous orchestrator as a new service and tool, enabling users to transform unstructured data into knowledge graphs and query them using natural language.

**Strategy**: Refactor the Neo4j LLM Graph Builder backend into a standalone FastAPI service that integrates with the existing Oraclous ecosystem while preserving its core functionality.

## 🏗️ Architecture Integration Strategy

### Current Ecosystem
- **auth-service** (Port 8000) - User authentication & OAuth
- **credential-broker-service** (Port 8002) - Credential management 
- **oraclous-core-service** (Port 8001) - Tool orchestration & workflow management
- **knowledge-graph-builder** (Port 8003) - **[NEW SERVICE]**

### Integration Points
1. **Authentication**: Use auth-service for user validation
2. **Credentials**: Use credential-broker for LLM API keys and Neo4j credentials
3. **Tool Registration**: Register as a new tool in oraclous-core-service
4. **Data Sources**: Leverage existing ingestion tools from orchestrator
5. **Storage**: Independent Neo4j database with metadata in PostgreSQL

## 📋 Implementation Checkpoints

### ✅ **Checkpoint 1: Service Foundation** (Week 1)
**Goal**: Establish the basic FastAPI service structure with Neo4j connectivity

#### Tasks:
1. **Project Structure Setup**
   ```
   knowledge-graph-builder/
   ├── app/
   │   ├── api/
   │   │   └── v1/
   │   │       ├── endpoints/
   │   │       │   ├── graphs.py
   │   │       │   ├── entities.py
   │   │       │   ├── relationships.py
   │   │       │   ├── chat.py
   │   │       │   └── health.py
   │   │       └── router.py
   │   ├── core/
   │   │   ├── config.py
   │   │   ├── neo4j_client.py
   │   │   ├── database.py
   │   │   └── logging.py
   │   ├── models/
   │   │   ├── graph.py
   │   │   ├── entity.py
   │   │   └── chat.py
   │   ├── services/
   │   │   ├── auth_service.py
   │   │   ├── credential_service.py
   │   │   └── graph_service.py
   │   ├── schemas/
   │   │   ├── graph_schemas.py
   │   │   ├── entity_schemas.py
   │   │   └── response_schemas.py
   │   └── main.py
   ├── requirements.txt
   ├── Dockerfile
   └── docker-compose.yml
   ```

2. **Core Dependencies Installation**
   ```python
   # requirements.txt
   fastapi==0.116.1
   uvicorn[standard]==0.35.0
   neo4j==5.26.0
   httpx==0.28.1
   pydantic==2.11.7
   pydantic-settings==2.10.1
   langchain==0.3.12
   langchain-openai==0.2.11
   langchain-neo4j==0.1.2
   langchain-community==0.3.12
   python-multipart==0.0.20
   python-dotenv==1.1.1
   asyncpg==0.30.0
   sqlalchemy[asyncio]==2.0.42
   ```

3. **Neo4j Connection Setup**
   ```python
   # app/core/neo4j_client.py
   from neo4j import AsyncGraphDatabase
   from typing import Optional
   
   class Neo4jClient:
       def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
           self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
           self.database = database
       
       async def health_check(self) -> bool:
           """Check Neo4j connectivity"""
           try:
               async with self.driver.session(database=self.database) as session:
                   result = await session.run("RETURN 1 as health")
                   record = await result.single()
                   return record["health"] == 1
           except Exception:
               return False
   ```

4. **Authentication Integration**
   ```python
   # app/services/auth_service.py
   import httpx
   from app.core.config import settings
   
   async def get_current_user(token: str):
       async with httpx.AsyncClient() as client:
           response = await client.get(
               f"{settings.AUTH_SERVICE_URL}/me",
               headers={"Authorization": f"Bearer {token}"}
           )
           if response.status_code == 200:
               return response.json()
           raise HTTPException(status_code=401, detail="Invalid token")
   ```

5. **Basic Health Endpoints**
   ```python
   # app/api/v1/endpoints/health.py
   @router.get("/health")
   async def health_check():
       neo4j_healthy = await neo4j_client.health_check()
       return {
           "status": "healthy" if neo4j_healthy else "unhealthy",
           "neo4j": neo4j_healthy,
           "timestamp": datetime.utcnow().isoformat()
       }
   ```

**Deliverables**:
- ✅ FastAPI service running on port 8003
- ✅ Neo4j connectivity established
- ✅ Basic authentication integration
- ✅ Health check endpoints
- ✅ Docker configuration

---

### 🔄 **Checkpoint 2: Core Graph Building** (Week 2-3)
**Goal**: Implement entity extraction, relationship extraction, and basic graph creation

#### Tasks:
1. **LLM Service Integration**
   ```python
   # app/services/llm_service.py
   from langchain_openai import ChatOpenAI
   from langchain_community.graphs.graph_transformer import LLMGraphTransformer
   
   class LLMService:
       def __init__(self):
           self.llm = None
           self.graph_transformer = None
       
       async def initialize_llm(self, provider: str, api_key: str, model: str):
           """Initialize LLM based on provider"""
           if provider == "openai":
               self.llm = ChatOpenAI(
                   api_key=api_key,
                   model=model,
                   temperature=0
               )
           # Add support for other providers (Anthropic, Gemini, etc.)
           
           self.graph_transformer = LLMGraphTransformer(llm=self.llm)
   ```

2. **Entity Extraction Service**
   ```python
   # app/services/entity_extractor.py
   from langchain_community.graphs.neo4j_graph import Neo4jGraph
   from langchain.text_splitter import RecursiveCharacterTextSplitter
   
   class EntityExtractor:
       def __init__(self, neo4j_graph: Neo4jGraph, llm_service: LLMService):
           self.neo4j_graph = neo4j_graph
           self.llm_service = llm_service
           self.text_splitter = RecursiveCharacterTextSplitter(
               chunk_size=1000,
               chunk_overlap=200
           )
       
       async def extract_entities_from_text(
           self, 
           text: str, 
           schema: Optional[Dict] = None,
           user_instructions: Optional[str] = None
       ) -> List[GraphDocument]:
           """Extract entities and relationships from text"""
           chunks = self.text_splitter.split_text(text)
           graph_documents = []
           
           for chunk in chunks:
               docs = await self.llm_service.graph_transformer.convert_to_graph_documents(
                   [Document(page_content=chunk)],
                   schema=schema,
                   include_source=True
               )
               graph_documents.extend(docs)
           
           return graph_documents
   ```

3. **Diffbot Integration** (Advanced Entity Recognition)
   ```python
   # app/services/diffbot_service.py
   import httpx
   from typing import List, Dict, Any
   
   class DiffbotService:
       def __init__(self, api_key: str):
           self.api_key = api_key
           self.base_url = "https://nl.diffbot.com/v1/"
       
       async def extract_entities(self, text: str) -> Dict[str, Any]:
           """Extract entities using Diffbot's NLP API"""
           async with httpx.AsyncClient() as client:
               response = await client.post(
                   f"{self.base_url}?token={self.api_key}",
                   json={"content": text, "lang": "en"}
               )
               return response.json()
   ```

4. **Graph Schema Management**
   ```python
   # app/services/schema_service.py
   class SchemaService:
       def __init__(self, neo4j_graph: Neo4jGraph):
           self.neo4j_graph = neo4j_graph
       
       async def learn_schema_from_text(self, text: str, llm_service: LLMService) -> Dict:
           """Auto-learn schema from sample text"""
           # Implementation for schema extraction using LLM
           pass
       
       async def get_existing_schema(self) -> Dict:
           """Get current database schema"""
           return self.neo4j_graph.get_schema
       
       async def consolidate_schema(self, existing_entities: List[str]) -> Dict:
           """Consolidate and clean up schema"""
           pass
   ```

5. **Graph Creation APIs**
   ```python
   # app/api/v1/endpoints/graphs.py
   @router.post("/graphs/{graph_id}/ingest")
   async def ingest_data(
       graph_id: str,
       data: IngestDataRequest,
       current_user: dict = Depends(get_current_user)
   ):
       """Ingest data and create graph"""
       # Extract entities and relationships
       graph_documents = await entity_extractor.extract_entities_from_text(
           text=data.content,
           schema=data.schema,
           user_instructions=data.instructions
       )
       
       # Store in Neo4j
       await graph_service.store_graph_documents(graph_id, graph_documents)
       
       return {"status": "success", "entities_count": len(graph_documents)}
   ```

**Deliverables**:
- ✅ Entity extraction using LLMs (OpenAI, Anthropic, Gemini)
- ✅ Diffbot integration for advanced NLP
- ✅ Relationship extraction and graph document creation
- ✅ Schema learning and management
- ✅ Basic graph ingestion API endpoints

---

### 🔍 **Checkpoint 3: Embeddings & Vector Search** (Week 4)
**Goal**: Implement text embeddings, vector storage, and semantic search capabilities

#### Tasks:
1. **Embedding Service**
   ```python
   # app/services/embedding_service.py
   from langchain_openai import OpenAIEmbeddings
   from langchain_community.vectorstores.neo4j_vector import Neo4jVector
   
   class EmbeddingService:
       def __init__(self, neo4j_graph: Neo4jGraph):
           self.neo4j_graph = neo4j_graph
           self.embeddings = None
           self.vector_store = None
       
       async def initialize_embeddings(self, provider: str, api_key: str):
           """Initialize embedding model"""
           if provider == "openai":
               self.embeddings = OpenAIEmbeddings(
                   api_key=api_key,
                   model="text-embedding-ada-002"
               )
           
           self.vector_store = Neo4jVector.from_existing_graph(
               embedding=self.embeddings,
               url=self.neo4j_graph.url,
               username=self.neo4j_graph.username,
               password=self.neo4j_graph.password,
               index_name="chunk_embeddings",
               node_label="Chunk",
               text_node_properties=["text"],
               embedding_node_property="embedding"
           )
       
       async def generate_embeddings_for_chunks(self, chunks: List[str]) -> List[List[float]]:
           """Generate embeddings for text chunks"""
           return await self.embeddings.embed_documents(chunks)
   ```

2. **Vector Index Management**
   ```python
   # app/services/vector_service.py
   class VectorService:
       def __init__(self, neo4j_client: Neo4jClient):
           self.neo4j_client = neo4j_client
       
       async def create_vector_indexes(self):
           """Create necessary vector indexes"""
           queries = [
               """
               CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
               FOR (c:Chunk) ON (c.embedding)
               OPTIONS {indexConfig: {
                   `vector.dimensions`: 1536,
                   `vector.similarity_function`: 'cosine'
               }}
               """,
               """
               CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
               FOR (e:Entity) ON (e.embedding)
               OPTIONS {indexConfig: {
                   `vector.dimensions`: 1536,
                   `vector.similarity_function`: 'cosine'
               }}
               """
           ]
           
           async with self.neo4j_client.driver.session() as session:
               for query in queries:
                   await session.run(query)
   ```

3. **Search Service**
   ```python
   # app/services/search_service.py
   class SearchService:
       def __init__(self, vector_service: VectorService, neo4j_graph: Neo4jGraph):
           self.vector_service = vector_service
           self.neo4j_graph = neo4j_graph
       
       async def similarity_search(
           self, 
           query: str, 
           k: int = 5, 
           filter_dict: Optional[Dict] = None
       ) -> List[Dict]:
           """Perform similarity search"""
           query_embedding = await self.vector_service.embeddings.embed_query(query)
           
           cypher_query = """
           CALL db.index.vector.queryNodes('chunk_embeddings', $k, $query_embedding)
           YIELD node, score
           RETURN node.text as text, score, node.source as source
           ORDER BY score DESC
           """
           
           async with self.neo4j_client.driver.session() as session:
               result = await session.run(
                   cypher_query, 
                   k=k, 
                   query_embedding=query_embedding
               )
               return [record.data() async for record in result]
   ```

**Deliverables**:
- ✅ Text embedding generation (OpenAI, sentence-transformers)
- ✅ Vector storage in Neo4j vector indexes
- ✅ Similarity search functionality
- ✅ Hybrid search (keyword + semantic)

---

### 🤖 **Checkpoint 4: LLM Integration & Chat Interface** (Week 5-6)
**Goal**: Implement natural language querying, GraphRAG, and conversational interface

#### Tasks:
1. **Chat Service**
   ```python
   # app/services/chat_service.py
   from langchain_community.graphs.neo4j_graph import Neo4jGraph
   from langchain.chains import GraphCypherQAChain
   
   class ChatService:
       def __init__(self, neo4j_graph: Neo4jGraph, llm_service: LLMService):
           self.neo4j_graph = neo4j_graph
           self.llm_service = llm_service
           self.cypher_chain = None
       
       async def initialize_chat(self):
           """Initialize chat chains"""
           self.cypher_chain = GraphCypherQAChain.from_llm(
               llm=self.llm_service.llm,
               graph=self.neo4j_graph,
               verbose=True,
               return_intermediate_steps=True
           )
       
       async def chat_with_graph(
           self, 
           query: str, 
           mode: str = "graph_vector",
           graph_id: str = None
       ) -> Dict[str, Any]:
           """Chat with knowledge graph using different retrieval modes"""
           
           if mode == "vector":
               return await self._vector_search_chat(query, graph_id)
           elif mode == "graph":
               return await self._graph_cypher_chat(query, graph_id)
           elif mode == "graph_vector":
               return await self._hybrid_chat(query, graph_id)
           else:
               raise ValueError(f"Unsupported chat mode: {mode}")
   ```

2. **GraphRAG Implementation**
   ```python
   # app/services/graph_rag_service.py
   class GraphRAGService:
       def __init__(self, neo4j_graph: Neo4jGraph, search_service: SearchService):
           self.neo4j_graph = neo4j_graph
           self.search_service = search_service
       
       async def graph_augmented_retrieval(
           self, 
           query: str, 
           max_entities: int = 10
       ) -> Dict[str, Any]:
           """Perform GraphRAG retrieval"""
           
           # 1. Vector search for relevant chunks
           similar_chunks = await self.search_service.similarity_search(query, k=5)
           
           # 2. Extract entities from relevant chunks
           entity_query = """
           MATCH (c:Chunk)-[:CONTAINS]->(e:Entity)
           WHERE c.id IN $chunk_ids
           RETURN e.name as entity, e.type as type, collect(c.text) as contexts
           LIMIT $max_entities
           """
           
           # 3. Get entity neighborhoods
           neighborhood_query = """
           MATCH (e:Entity {name: $entity_name})-[r]-(related:Entity)
           RETURN type(r) as relationship, related.name as related_entity, related.type as related_type
           LIMIT 5
           """
           
           # 4. Combine context for LLM
           context = self._build_context(similar_chunks, entities, neighborhoods)
           
           return {
               "context": context,
               "chunks": similar_chunks,
               "entities": entities,
               "neighborhoods": neighborhoods
           }
   ```

3. **Text-to-Cypher Service**
   ```python
   # app/services/cypher_service.py
   class CypherService:
       def __init__(self, neo4j_graph: Neo4jGraph, llm_service: LLMService):
           self.neo4j_graph = neo4j_graph
           self.llm_service = llm_service
       
       async def natural_language_to_cypher(self, question: str) -> Dict[str, Any]:
           """Convert natural language to Cypher query"""
           
           schema = self.neo4j_graph.get_schema
           
           prompt = f"""
           Given the Neo4j graph schema:
           {schema}
           
           Convert this natural language question to a Cypher query:
           "{question}"
           
           Return only the Cypher query, no explanation.
           """
           
           cypher_query = await self.llm_service.llm.invoke(prompt)
           
           # Execute query safely
           try:
               result = await self.neo4j_graph.query(cypher_query.content)
               return {
                   "cypher": cypher_query.content,
                   "result": result,
                   "success": True
               }
           except Exception as e:
               return {
                   "cypher": cypher_query.content,
                   "error": str(e),
                   "success": False
               }
   ```

4. **Chat API Endpoints**
   ```python
   # app/api/v1/endpoints/chat.py
   @router.post("/graphs/{graph_id}/chat")
   async def chat_with_graph(
       graph_id: str,
       chat_request: ChatRequest,
       current_user: dict = Depends(get_current_user)
   ):
       """Chat with knowledge graph"""
       
       response = await chat_service.chat_with_graph(
           query=chat_request.message,
           mode=chat_request.mode,
           graph_id=graph_id
       )
       
       # Save chat history
       await chat_service.save_chat_history(
           user_id=current_user["id"],
           graph_id=graph_id,
           message=chat_request.message,
           response=response["answer"],
           metadata=response.get("metadata", {})
       )
       
       return response
   ```

**Deliverables**:
- ✅ Natural language to Cypher translation
- ✅ GraphRAG implementation
- ✅ Multi-mode chat interface (vector, graph, hybrid)
- ✅ Chat history management
- ✅ Context management and conversation flow

---

### 🔧 **Checkpoint 5: Orchestrator Integration** (Week 7)
**Goal**: Register as a tool in oraclous-core-service and implement service communication

#### Tasks:
1. **Tool Definition for Core Service**
   ```python
   # To be added to oraclous-core-service
   # app/tools/implementations/analytics/knowledge_graph_builder.py
   
   class KnowledgeGraphBuilder(InternalTool):
       """Knowledge Graph Builder tool for oraclous-core-service"""
       
       def __init__(self, definition: ToolDefinition):
           super().__init__(definition)
           self.service_url = settings.KNOWLEDGE_GRAPH_SERVICE_URL
       
       @classmethod
       def get_tool_definition(cls) -> ToolDefinition:
           return ToolDefinition(
               id=generate_tool_id("Knowledge Graph Builder", "1.0.0", "ANALYTICS"),
               name="Knowledge Graph Builder",
               description="Build and query knowledge graphs from unstructured data",
               version="1.0.0",
               category=ToolCategory.ANALYTICS,
               type=ToolType.INTERNAL,
               capabilities=[
                   ToolCapability(
                       name="create_knowledge_graph",
                       description="Create knowledge graph from text data"
                   ),
                   ToolCapability(
                       name="query_knowledge_graph", 
                       description="Query knowledge graph using natural language"
                   ),
                   ToolCapability(
                       name="extract_entities",
                       description="Extract entities and relationships from text"
                   )
               ],
               input_schema=ToolSchema(
                   type="object",
                   properties={
                       "action": {
                           "type": "string",
                           "enum": ["create_graph", "query_graph", "extract_entities"],
                           "description": "Action to perform"
                       },
                       "graph_id": {
                           "type": "string", 
                           "description": "Graph identifier"
                       },
                       "content": {
                           "type": "string",
                           "description": "Text content to process"
                       },
                       "query": {
                           "type": "string",
                           "description": "Natural language query"
                       },
                       "schema": {
                           "type": "object",
                           "description": "Graph schema configuration"
                       }
                   },
                   required=["action"]
               ),
               credential_requirements=[
                   CredentialRequirement(
                       provider="openai",
                       required=True,
                       description="OpenAI API key for LLM operations"
                   ),
                   CredentialRequirement(
                       provider="neo4j",
                       required=True, 
                       description="Neo4j database credentials"
                   )
               ]
           )
   ```

2. **Service Communication**
   ```python
   # app/services/orchestrator_service.py
   class OrchestratorService:
       def __init__(self):
           self.core_service_url = settings.CORE_SERVICE_URL
       
       async def register_with_orchestrator(self):
           """Register this service with the orchestrator"""
           registration_data = {
               "service_name": "knowledge-graph-builder",
               "service_url": settings.SERVICE_URL,
               "health_endpoint": "/api/v1/health",
               "capabilities": [
                   "graph_creation",
                   "entity_extraction", 
                   "graph_querying",
                   "semantic_search"
               ]
           }
           
           async with httpx.AsyncClient() as client:
               response = await client.post(
                   f"{self.core_service_url}/api/v1/services/register",
                   json=registration_data,
                   headers={"X-Internal-Key": settings.INTERNAL_SERVICE_KEY}
               )
               return response.json()
   ```

3. **Credits Calculation**
   ```python
   # app/services/credits_service.py
   class CreditsService:
       def __init__(self):
           self.credit_rates = {
               "entity_extraction": 0.05,  # per 1000 characters
               "graph_creation": 0.10,     # per graph operation
               "chat_query": 0.02,         # per query
               "embedding_generation": 0.01 # per 1000 tokens
           }
       
       def calculate_credits(self, operation: str, **kwargs) -> Decimal:
           """Calculate credits consumed for operation"""
           if operation == "entity_extraction":
               char_count = kwargs.get("character_count", 0)
               return Decimal(str(self.credit_rates[operation] * (char_count / 1000)))
           elif operation == "chat_query":
               return Decimal(str(self.credit_rates[operation]))
           # Add other operations
           return Decimal("0.00")
   ```

**Deliverables**:
- ✅ Tool registration in oraclous-core-service
- ✅ Service-to-service communication
- ✅ Error handling and retry mechanisms
- ✅ Credits calculation integration

---

### 🚀 **Checkpoint 6: Advanced Features & Optimization** (Week 8-9)
**Goal**: Implement advanced graph analytics, performance optimization, and production readiness

#### Tasks:
1. **Graph Analytics Service**
   ```python
   # app/services/analytics_service.py
   class GraphAnalyticsService:
       def __init__(self, neo4j_client: Neo4jClient):
           self.neo4j_client = neo4j_client
       
       async def community_detection(self, graph_id: str) -> Dict[str, Any]:
           """Detect communities in the graph"""
           query = """
           CALL gds.louvain.stream('myGraph')
           YIELD nodeId, communityId
           RETURN gds.util.asNode(nodeId).name AS name, communityId
           ORDER BY communityId
           """
           # Implementation
       
       async def centrality_analysis(self, graph_id: str) -> Dict[str, Any]:
           """Calculate node centrality metrics"""
           # PageRank, Betweenness, Degree centrality
       
       async def graph_metrics(self, graph_id: str) -> Dict[str, Any]:
           """Calculate overall graph metrics"""
           return {
               "node_count": await self._count_nodes(graph_id),
               "relationship_count": await self._count_relationships(graph_id),
               "avg_degree": await self._average_degree(graph_id),
               "clustering_coefficient": await self._clustering_coefficient(graph_id)
           }
   ```

2. **Performance Optimization**
   ```python
   # app/services/cache_service.py
   from functools import wraps
   import asyncio
   from typing import Dict, Any
   
   class CacheService:
       def __init__(self):
           self._cache: Dict[str, Any] = {}
           self._cache_ttl: Dict[str, float] = {}
       
       def cache_result(self, ttl: int = 300):
           """Cache decorator for expensive operations"""
           def decorator(func):
               @wraps(func)
               async def wrapper(*args, **kwargs):
                   cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
                   
                   if cache_key in self._cache:
                       if time.time() < self._cache_ttl[cache_key]:
                           return self._cache[cache_key]
                   
                   result = await func(*args, **kwargs)
                   self._cache[cache_key] = result
                   self._cache_ttl[cache_key] = time.time() + ttl
                   
                   return result
               return wrapper
           return decorator
   ```

3. **Batch Processing**
   ```python
   # app/services/batch_service.py
   class BatchProcessor:
       def __init__(self, neo4j_client: Neo4jClient):
           self.neo4j_client = neo4j_client
           self.batch_size = 100
       
       async def batch_entity_extraction(
           self, 
           documents: List[str], 
           schema: Optional[Dict] = None
       ) -> List[GraphDocument]:
           """Process multiple documents in batches"""
           results = []
           
           for i in range(0, len(documents), self.batch_size):
               batch = documents[i:i + self.batch_size]
               
               # Process batch in parallel
               tasks = [
                   self.extract_entities_from_document(doc, schema) 
                   for doc in batch
               ]
               
               batch_results = await asyncio.gather(*tasks)
               results.extend(batch_results)
           
           return results
   ```

4. **Monitoring and Metrics**
   ```python
   # app/services/monitoring_service.py
   from prometheus_client import Counter, Histogram, Gauge
   import time
   from functools import wraps
   
   class MonitoringService:
       def __init__(self):
           self.request_count = Counter(
               'kg_requests_total', 
               'Total requests', 
               ['method', 'endpoint', 'status']
           )
           self.request_duration = Histogram(
               'kg_request_duration_seconds', 
               'Request duration'
           )
           self.graph_size = Gauge(
               'kg_graph_size', 
               'Number of nodes in graph', 
               ['graph_id']
           )
       
       def track_request(self, func):
           @wraps(func)
           async def wrapper(*args, **kwargs):
               start_time = time.time()
               try:
                   result = await func(*args, **kwargs)
                   self.request_count.labels(
                       method='POST', 
                       endpoint=func.__name__, 
                       status='success'
                   ).inc()
                   return result
               except Exception as e:
                   self.request_count.labels(
                       method='POST', 
                       endpoint=func.__name__, 
                       status='error'
                   ).inc()
                   raise
               finally:
                   duration = time.time() - start_time
                   self.request_duration.observe(duration)
           return wrapper
   ```

5. **Production Configuration**
   ```python
   # app/core/config.py
   from pydantic_settings import BaseSettings
   from typing import Optional
   
   class Settings(BaseSettings):
       # Service Configuration
       SERVICE_NAME: str = "knowledge-graph-builder"
       SERVICE_VERSION: str = "1.0.0"
       SERVICE_URL: str = "http://localhost:8003"
       
       # Database Configuration
       NEO4J_URI: str = "bolt://localhost:7687"
       NEO4J_USERNAME: str = "neo4j"
       NEO4J_PASSWORD: str = "password"
       NEO4J_DATABASE: str = "neo4j"
       
       POSTGRES_URL: str = "postgresql+asyncpg://user:pass@localhost/kgbuilder"
       
       # External Services
       AUTH_SERVICE_URL: str = "http://localhost:8000"
       CREDENTIAL_BROKER_URL: str = "http://localhost:8002"
       CORE_SERVICE_URL: str = "http://localhost:8001"
       
       # Security
       INTERNAL_SERVICE_KEY: str = "your-internal-service-key"
       JWT_SECRET_KEY: str = "your-jwt-secret"
       
       # LLM Configuration
       OPENAI_API_KEY: Optional[str] = None
       ANTHROPIC_API_KEY: Optional[str] = None
       DIFFBOT_API_KEY: Optional[str] = None
       
       # Performance Settings
       MAX_CONCURRENT_EXTRACTIONS: int = 5
       BATCH_SIZE: int = 100
       CACHE_TTL: int = 300
       
       # Monitoring
       ENABLE_METRICS: bool = True
       LOG_LEVEL: str = "INFO"
       
       class Config:
           env_file = ".env"
   
   settings = Settings()
   ```

**Deliverables**:
- ✅ Community detection algorithms
- ✅ Graph centrality analysis
- ✅ Performance optimization with caching
- ✅ Batch processing capabilities
- ✅ Monitoring and metrics collection
- ✅ Production-ready configuration

---

## 🛠️ Technical Implementation Details

### Database Schema Design

#### PostgreSQL Metadata Tables
```sql
-- Knowledge graphs metadata
CREATE TABLE knowledge_graphs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    user_id UUID NOT NULL,
    neo4j_database VARCHAR(100),
    schema_config JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    node_count INTEGER DEFAULT 0,
    relationship_count INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'active'
);

-- Graph ingestion jobs
CREATE TABLE ingestion_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_id UUID REFERENCES knowledge_graphs(id),
    source_type VARCHAR(50), -- 'text', 'pdf', 'url', 'api'
    source_content TEXT,
    status VARCHAR(50) DEFAULT 'pending',
    progress INTEGER DEFAULT 0,
    error_message TEXT,
    extracted_entities INTEGER DEFAULT 0,
    extracted_relationships INTEGER DEFAULT 0,
    credits_consumed DECIMAL(10,4) DEFAULT 0,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Chat sessions
CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_id UUID REFERENCES knowledge_graphs(id),
    user_id UUID NOT NULL,
    session_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    last_message_at TIMESTAMP DEFAULT NOW()
);

-- Chat messages
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES chat_sessions(id),
    message_type VARCHAR(20), -- 'user', 'assistant'
    content TEXT NOT NULL,
    metadata JSONB, -- retrieval mode, sources, etc.
    credits_consumed DECIMAL(10,4) DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### Neo4j Graph Schema
```cypher
// Core node types
CREATE CONSTRAINT entity_id_unique FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT chunk_id_unique FOR (c:Chunk) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT document_id_unique FOR (d:Document) REQUIRE d.id IS UNIQUE;

// Vector indexes for embeddings
CREATE VECTOR INDEX chunk_embeddings FOR (c:Chunk) ON (c.embedding)
OPTIONS {indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
}};

CREATE VECTOR INDEX entity_embeddings FOR (e:Entity) ON (e.embedding)
OPTIONS {indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
}};

// Full-text search indexes
CREATE FULLTEXT INDEX entity_fulltext FOR (e:Entity) ON EACH [e.name, e.description];
CREATE FULLTEXT INDEX chunk_fulltext FOR (c:Chunk) ON EACH [c.text];
```

### API Specifications

#### Core Endpoints
```python
# Graph Management
POST   /api/v1/graphs/                           # Create new graph
GET    /api/v1/graphs/                           # List user graphs
GET    /api/v1/graphs/{graph_id}                 # Get graph details
PUT    /api/v1/graphs/{graph_id}                 # Update graph
DELETE /api/v1/graphs/{graph_id}                 # Delete graph

# Data Ingestion
POST   /api/v1/graphs/{graph_id}/ingest          # Ingest data
GET    /api/v1/graphs/{graph_id}/jobs            # List ingestion jobs
GET    /api/v1/graphs/{graph_id}/jobs/{job_id}   # Get job status

# Entity & Relationship Management
GET    /api/v1/graphs/{graph_id}/entities        # List entities
POST   /api/v1/graphs/{graph_id}/entities        # Create entity
PUT    /api/v1/graphs/{graph_id}/entities/{id}   # Update entity
DELETE /api/v1/graphs/{graph_id}/entities/{id}   # Delete entity

GET    /api/v1/graphs/{graph_id}/relationships   # List relationships
POST   /api/v1/graphs/{graph_id}/relationships   # Create relationship

# Search & Query
POST   /api/v1/graphs/{graph_id}/search          # Search entities/chunks
POST   /api/v1/graphs/{graph_id}/cypher          # Execute Cypher query

# Chat Interface
POST   /api/v1/graphs/{graph_id}/chat            # Chat with graph
GET    /api/v1/graphs/{graph_id}/chat/sessions   # List chat sessions
GET    /api/v1/graphs/{graph_id}/chat/sessions/{session_id}/messages # Get messages

# Analytics
GET    /api/v1/graphs/{graph_id}/analytics/metrics        # Graph metrics
POST   /api/v1/graphs/{graph_id}/analytics/communities    # Community detection
POST   /api/v1/graphs/{graph_id}/analytics/centrality     # Centrality analysis

# Schema Management
GET    /api/v1/graphs/{graph_id}/schema          # Get current schema
POST   /api/v1/graphs/{graph_id}/schema/learn    # Learn schema from text
PUT    /api/v1/graphs/{graph_id}/schema          # Update schema

# Health & Monitoring
GET    /api/v1/health                            # Service health
GET    /api/v1/metrics                           # Prometheus metrics
```

### Data Flow Architecture

```mermaid
graph TB
    A[User] --> B[Auth Service]
    B --> C[Core Service]
    C --> D[Knowledge Graph Builder]
    
    D --> E[Credential Broker]
    E --> F[LLM APIs]
    E --> G[Neo4j Database]
    
    D --> H[Entity Extractor]
    D --> I[Embedding Service]
    D --> J[Chat Service]
    
    H --> K[Diffbot API]
    H --> L[LangChain]
    
    I --> M[Vector Indexes]
    J --> N[GraphRAG]
    
    D --> O[PostgreSQL Metadata]
    
    subgraph "External Services"
        F
        K
    end
    
    subgraph "Storage Layer"
        G
        M
        O
    end
```

### Integration Patterns

#### 1. Authentication Flow
```python
async def authenticate_request(request: Request) -> dict:
    """Authenticate request through auth-service"""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.AUTH_SERVICE_URL}/api/v1/auth/verify",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        return response.json()
```

#### 2. Credential Resolution
```python
async def get_llm_credentials(user_id: str, provider: str) -> dict:
    """Get LLM credentials from credential broker"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.CREDENTIAL_BROKER_URL}/api/v1/runtime-token",
            headers={"X-Internal-Key": settings.INTERNAL_SERVICE_KEY},
            json={
                "user_id": user_id,
                "provider": provider,
                "scope": "llm"
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to get {provider} credentials"
            )
        
        return response.json()
```

#### 3. Tool Execution Integration
```python
async def execute_as_tool(
    action: str,
    parameters: dict,
    user_id: str,
    execution_context: dict
) -> ExecutionResult:
    """Execute knowledge graph operations as a tool"""
    
    try:
        if action == "create_graph":
            result = await create_knowledge_graph(
                name=parameters.get("name"),
                content=parameters.get("content"),
                schema=parameters.get("schema"),
                user_id=user_id
            )
        elif action == "query_graph":
            result = await query_knowledge_graph(
                graph_id=parameters.get("graph_id"),
                query=parameters.get("query"),
                mode=parameters.get("mode", "graph_vector"),
                user_id=user_id
            )
        elif action == "extract_entities":
            result = await extract_entities(
                content=parameters.get("content"),
                schema=parameters.get("schema"),
                user_id=user_id
            )
        else:
            raise ValueError(f"Unknown action: {action}")
        
        # Calculate credits consumed
        credits = calculate_credits_for_operation(action, parameters, result)
        
        return ExecutionResult(
            success=True,
            data=result,
            credits_consumed=credits,
            metadata={
                "action": action,
                "entities_extracted": result.get("entities_count", 0),
                "relationships_created": result.get("relationships_count", 0)
            }
        )
        
    except Exception as e:
        return ExecutionResult(
            success=False,
            error=str(e),
            credits_consumed=Decimal("0.00")
        )
```

## 🚀 Deployment Strategy

### Docker Configuration
```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8003/api/v1/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
```

### Docker Compose Integration
```yaml
# docker-compose.yml (addition to existing)
services:
  knowledge-graph-builder:
    build: ./knowledge-graph-builder
    ports:
      - "8003:8003"
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USERNAME=neo4j
      - NEO4J_PASSWORD=password
      - POSTGRES_URL=postgresql+asyncpg://postgres:password@postgres:5432/kgbuilder
      - AUTH_SERVICE_URL=http://auth-service:8000
      - CREDENTIAL_BROKER_URL=http://credential-broker:8002
      - CORE_SERVICE_URL=http://oraclous-core:8001
    depends_on:
      - neo4j
      - postgres
      - auth-service
      - credential-broker
      - oraclous-core
    volumes:
      - ./knowledge-graph-builder/logs:/app/logs
    networks:
      - oraclous-network

  neo4j:
    image: neo4j:5.26-community
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc", "graph-data-science"]
      - NEO4J_apoc_export_file_enabled=true
      - NEO4J_apoc_import_file_enabled=true
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    networks:
      - oraclous-network

volumes:
  neo4j_data:
  neo4j_logs:
```

## 📋 Development Workflow

### Session Management Strategy

Given the constraint of max session limits with coding agents, here's how to manage the implementation:

#### Session 1: Foundation Setup
- **Focus**: Checkpoint 1 (Service Foundation)
- **Deliverables**: Basic FastAPI service, Neo4j connection, authentication
- **Files to create**: Core structure, configuration, health endpoints

#### Session 2: Entity Extraction Core
- **Focus**: Checkpoint 2 Part 1 (LLM Integration & Basic Extraction)
- **Deliverables**: LLM service, basic entity extraction, schema management
- **Files to create**: LLM services, extraction logic, schema handlers

#### Session 3: Graph Creation & Storage
- **Focus**: Checkpoint 2 Part 2 (Graph Creation & Neo4j Integration)
- **Deliverables**: Graph document creation, Neo4j storage, ingestion APIs
- **Files to create**: Graph services, storage logic, API endpoints

#### Session 4: Vector & Search Implementation
- **Focus**: Checkpoint 3 (Embeddings & Vector Search)
- **Deliverables**: Embedding generation, vector indexes, search functionality
- **Files to create**: Embedding services, vector operations, search APIs

#### Session 5: Chat & Query Interface
- **Focus**: Checkpoint 4 (LLM Integration & Chat)
- **Deliverables**: Chat service, GraphRAG, natural language querying
- **Files to create**: Chat logic, GraphRAG implementation, query services

#### Session 6: Orchestrator Integration
- **Focus**: Checkpoint 5 (Tool Registration & Communication)
- **Deliverables**: Tool definition, service communication, credits calculation
- **Files to create**: Tool implementation, communication services, credits logic

#### Session 7: Advanced Features & Production
- **Focus**: Checkpoint 6 (Analytics & Optimization)
- **Deliverables**: Graph analytics, performance optimization, monitoring
- **Files to create**: Analytics services, optimization features, monitoring

### Continuation Strategy

For each session, use this implementation plan as the reference document. Each session should:

1. **Reference this plan** for the specific checkpoint being implemented
2. **Follow the defined architecture** and patterns outlined here
3. **Implement the exact APIs and schemas** specified in this document
4. **Maintain consistency** with the established patterns and conventions
5. **Update progress** by marking completed deliverables

### Testing Strategy

```python
# tests/test_integration.py
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_full_workflow():
    """Test complete knowledge graph creation workflow"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # 1. Create graph
        graph_response = await client.post(
            "/api/v1/graphs/",
            json={
                "name": "Test Graph",
                "description": "Test knowledge graph"
            },
            headers={"Authorization": "Bearer test-token"}
        )
        assert graph_response.status_code == 201
        graph_id = graph_response.json()["id"]
        
        # 2. Ingest data
        ingest_response = await client.post(
            f"/api/v1/graphs/{graph_id}/ingest",
            json={
                "content": "John works at OpenAI. OpenAI is located in San Francisco.",
                "schema": {
                    "entities": ["Person", "Organization", "Location"],
                    "relationships": ["WORKS_AT", "LOCATED_IN"]
                }
            },
            headers={"Authorization": "Bearer test-token"}
        )
        assert ingest_response.status_code == 200
        
        # 3. Query graph
        chat_response = await client.post(
            f"/api/v1/graphs/{graph_id}/chat",
            json={
                "message": "Where does John work?",
                "mode": "graph_vector"
            },
            headers={"Authorization": "Bearer test-token"}
        )
        assert chat_response.status_code == 200
        assert "OpenAI" in chat_response.json()["answer"]
```

## 🎯 Success Criteria

### Technical Success Metrics
- ✅ All 6 checkpoints completed successfully
- ✅ Integration with existing Oraclous services working
- ✅ Knowledge graph creation from unstructured text (>90% accuracy)
- ✅ Natural language querying with GraphRAG responses
- ✅ Performance: <2s for entity extraction, <1s for chat queries
- ✅ Scalability: Support for 1000+ concurrent users
- ✅ Tool registration and execution through orchestrator

### Business Success Metrics
- ✅ Seamless user experience for non-technical users
- ✅ Support for multiple LLM providers (OpenAI, Anthropic, Gemini)
- ✅ Advanced entity recognition using Diffbot
- ✅ Credits-based pricing model integration
- ✅ Real-time chat interface with explainable responses

### Quality Assurance
- ✅ Comprehensive test suite (unit, integration, e2e)
- ✅ API documentation with OpenAPI/Swagger
- ✅ Error handling and graceful degradation
- ✅ Security compliance and data protection
- ✅ Monitoring and alerting setup
- ✅ Performance optimization and caching

---

This implementation plan provides a clear roadmap for integrating the Neo4j LLM Graph Builder into your Oraclous orchestrator system. Each checkpoint is designed to be manageable within a single coding session while building toward a complete, production-ready knowledge graph service.