# Project Milestone Plan: Causal Inference Integration for Knowledge Graph System

## Project Overview
This document outlines the phased implementation plan for integrating causal inference capabilities into an existing knowledge graph architecture. The project will extend the system to extract and store causal relationships from multi-modal data sources, enhancing the knowledge graph with cause-effect relationships.

## Existing Architecture Components
1. Auth - Authentication service
2. Credential service - Manages external service credentials
3. Knowledge Graph Builder - Constructs and manages the knowledge graph
4. Core Orchestrator - Data ingestion and workflow orchestration (using LangGraph)
5. Multi-modal Processor - Handles diverse data types processing

## New Component: Causal Inference Engine
A dedicated service that analyzes processed data to identify causal relationships using statistical and machine learning methods.

---

## Phase 1: Foundation and Causal Schema Design (Weeks 1-2)

### Milestone 1.1: Define Causal Relationship Schema
**Objective:** Establish standardized schema for representing causal relationships in Neo4j
**Deliverables:**
- Extended node/relationship type definitions for causal relationships
- Property schema for causal edges (confidence scores, methods, timestamps)
- JSON specification for causal relationship data exchange

**Technical Approach:**
```python
# Extended graph schema for causal relationships
class CausalRelationshipType(str, Enum):
    CAUSES = "CAUSES"
    INFLUENCES = "INFLUENCES"
    PREVENTS = "PREVENTS"
    CORRELATES_WITH = "CORRELATES_WITH"
    MEDIATES = "MEDIATES"

class CausalRelationship(BaseModel):
    source_id: str
    target_id: str
    type: CausalRelationshipType
    confidence: float = Field(..., ge=0.0, le=1.0)
    method: str  # e.g., "propensity_score_matching", "causal_discovery"
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### Milestone 1.2: Causal Inference Service Skeleton
**Objective:** Create basic FastAPI service structure for causal inference
**Deliverables:**
- FastAPI application with health check endpoint
- Basic request/response models for causal inference
- Docker containerization setup
- API documentation (OpenAPI/Swagger)

**Technical Approach:**
```python
# app/main.py
from fastapi import FastAPI
from .models import CausalAnalysisRequest, CausalAnalysisResponse

app = FastAPI(title="Causal Inference Engine")

@app.post("/analyze", response_model=CausalAnalysisResponse)
async def analyze_causality(request: CausalAnalysisRequest):
    """Endpoint for causal analysis"""
    pass

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "causal-inference-engine"}
```

---

## Phase 2: Core Causal Inference Implementation (Weeks 3-6)

### Milestone 2.1: Structured Data Causal Analysis
**Objective:** Implement causal inference methods for structured/tabular data
**Deliverables:**
- Integration with DoWhy library for causal inference
- Propensity score matching implementation
- Instrumental variables analysis
- Response transformer for Neo4j compatibility

**Technical Approach:**
```python
# app/services/structured_analyzer.py
from dowhy import CausalModel
import pandas as pd

class StructuredDataAnalyzer:
    def __init__(self):
        self.supported_methods = {
            "propensity_score_matching",
            "instrumental_variables",
            "regression_discontinuity"
        }
    
    def analyze(self, data: pd.DataFrame, treatment: str, 
                outcome: str, method: str = "propensity_score_matching") -> Dict:
        """Perform causal analysis on structured data"""
        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            graph=self._generate_graph(treatment, outcome, data.columns)
        )
        
        identified_estimand = model.identify_effect()
        estimate = model.estimate_effect(identified_estimand, 
                                       method_name=method)
        
        return {
            "estimate": estimate.value,
            "confidence_intervals": estimate.confidence_intervals,
            "method": method
        }
```

### Milestone 2.2: Text Data Causal Extraction
**Objective:** Implement causal relationship extraction from unstructured text
**Deliverables:**
- SpaCy-based causal phrase pattern matching
- BERT-based causal relationship classification
- Event extraction and causal linking
- Temporal causality detection

**Technical Approach:**
```python
# app/services/text_analyzer.py
import spacy
from transformers import pipeline

class TextDataAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.classifier = pipeline("text-classification", 
                                 model="causal-relationship-model")
    
    def extract_causal_relationships(self, text: str) -> List[Dict]:
        """Extract causal relationships from text"""
        doc = self.nlp(text)
        relationships = []
        
        # Pattern-based extraction
        for sent in doc.sents:
            causal_patterns = self._find_causal_patterns(sent)
            relationships.extend(causal_patterns)
        
        # ML-based classification
        ml_relationships = self._classify_causal_relationships(text)
        relationships.extend(ml_relationships)
        
        return relationships
```

---

## Phase 3: System Integration (Weeks 7-10)

### Milestone 3.1: Orchestrator Integration
**Objective:** Integrate causal inference with Core Orchestrator workflows
**Deliverables:**
- LangGraph workflow modifications to include causal analysis
- Conditional causal analysis based on data type and content
- Error handling and fallback mechanisms
- Performance monitoring integration

**Technical Approach:**
```python
# Updated LangGraph workflow
def create_processing_workflow():
    workflow = StateGraph(ProcessingState)
    
    # Existing nodes
    workflow.add_node("ingest_data", ingest_data)
    workflow.add_node("preprocess", preprocess_data)
    workflow.add_node("extract_entities", extract_entities)
    
    # New causal analysis node
    workflow.add_node("causal_analysis", perform_causal_analysis)
    
    # Updated edges
    workflow.add_edge("extract_entities", "causal_analysis")
    workflow.add_edge("causal_analysis", "build_graph")
    
    return workflow.compile()

def perform_causal_analysis(state: ProcessingState) -> ProcessingState:
    """Conditionally perform causal analysis based on data"""
    if state.data_type in ["structured", "text"]:
        analyzer = get_analyzer_for_type(state.data_type)
        causal_relationships = analyzer.analyze(state.processed_data)
        state.causal_relationships = causal_relationships
    return state
```

### Milestone 3.2: Knowledge Graph Builder Enhancement
**Objective:** Enhance Knowledge Graph Builder to handle causal relationships
**Deliverables:**
- Cypher query generators for causal relationships
- Conflict resolution for existing relationships
- Batch processing for large causal datasets
- Validation rules for causal relationship insertion

**Technical Approach:**
```python
# app/services/graph_builder.py
class KnowledgeGraphBuilder:
    def add_causal_relationships(self, relationships: List[CausalRelationship]):
        """Add causal relationships to the graph"""
        query = """
        UNWIND $relationships AS rel
        MATCH (source {id: rel.source_id})
        MATCH (target {id: rel.target_id})
        MERGE (source)-[r:CAUSES]->(target)
        SET r.confidence = rel.confidence,
            r.method = rel.method,
            r.metadata = rel.metadata,
            r.created_at = datetime()
        """
        self.execute_query(query, {"relationships": [r.dict() for r in relationships]})
```

---

## Phase 4: Multi-Modal Causal Inference (Weeks 11-14)

### Milestone 4.1: Temporal Data Causal Analysis
**Objective:** Implement causal inference for time-series data
**Deliverables:**
- Granger causality implementation
- Time-lagged correlation analysis
- Intervention analysis (CausalImpact)
- Seasonal adjustment for time series

**Technical Approach:**
```python
# app/services/temporal_analyzer.py
from statsmodels.tsa.stattools import grangercausalitytests

class TemporalDataAnalyzer:
    def granger_causality(self, data: pd.DataFrame, 
                         max_lag: int = 5) -> Dict[str, Any]:
        """Perform Granger causality test"""
        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        
        causality = {}
        for lag in range(1, max_lag + 1):
            p_value = results[lag][0]['ssr_chi2test'][1]
            causality[f"lag_{lag}"] = {
                "p_value": p_value,
                "significant": p_value < 0.05
            }
        
        return causality
```

### Milestone 4.2: Image and Multi-Modal Causal Analysis
**Objective:** Extract causal relationships from visual and multi-modal data
**Deliverables:**
- Visual relationship detection
- Cross-modal causal inference
- Scene graph to causal graph conversion
- Multi-modal fusion for causal analysis

**Technical Approach:**
```python
# app/services/multimodal_analyzer.py
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

class MultiModalAnalyzer:
    def __init__(self):
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    
    def analyze_image(self, image_path: str) -> List[CausalRelationship]:
        """Extract potential causal relationships from images"""
        # Generate caption
        caption = self.generate_image_caption(image_path)
        
        # Analyze caption for causal relationships
        text_analyzer = TextDataAnalyzer()
        relationships = text_analyzer.extract_causal_relationships(caption)
        
        return relationships
```

---

## Phase 5: Productionization and Optimization (Weeks 15-18)

### Milestone 5.1: Performance Optimization
**Objective:** Optimize causal inference for production scale
**Deliverables:**
- Async processing implementation
- Batch processing capabilities
- Memory management improvements
- Caching layer for repeated analyses

**Technical Approach:**
```python
# app/services/optimized_analyzer.py
from redis import asyncio as aioredis
import json

class OptimizedCausalAnalyzer:
    def __init__(self, redis_url: str):
        self.redis = aioredis.from_url(redis_url)
    
    async def analyze_with_cache(self, data: Dict, analysis_type: str) -> Dict:
        """Perform analysis with caching"""
        cache_key = f"causal_analysis:{analysis_type}:{hash(str(data))}"
        
        # Check cache
        cached_result = await self.redis.get(cache_key)
        if cached_result:
            return json.loads(cached_result)
        
        # Perform analysis
        result = await self._perform_analysis(data, analysis_type)
        
        # Cache result
        await self.redis.setex(cache_key, 3600, json.dumps(result))  # 1 hour cache
        
        return result
```

### Milestone 5.2: Monitoring and Validation
**Objective:** Implement monitoring and validation for causal inference
**Deliverables:**
- Performance metrics collection
- Quality validation framework
- Drift detection for causal models
- A/B testing framework for causal methods

**Technical Approach:**
```python
# app/monitoring/metrics.py
from prometheus_client import Counter, Histogram

class CausalMetrics:
    def __init__(self):
        self.requests_total = Counter('causal_requests_total', 
                                    'Total causal analysis requests', 
                                    ['type', 'method'])
        self.request_duration = Histogram('causal_request_duration_seconds',
                                        'Causal analysis request duration',
                                        ['type'])
        self.relationships_found = Counter('causal_relationships_found',
                                         'Causal relationships identified',
                                         ['type', 'confidence_level'])
    
    def record_analysis(self, analysis_type: str, method: str, 
                       duration: float, relationships: List):
        self.requests_total.labels(analysis_type, method).inc()
        self.request_duration.labels(analysis_type).observe(duration)
        
        for rel in relationships:
            confidence_level = "high" if rel.confidence > 0.7 else \
                             "medium" if rel.confidence > 0.4 else "low"
            self.relationships_found.labels(analysis_type, confidence_level).inc()
```

---

## Phase 6: Advanced Features and Evaluation (Weeks 19-22)

### Milestone 6.1: Counterfactual Analysis
**Objective:** Implement counterfactual reasoning capabilities
**Deliverables:**
- Counterfactual query processing
- What-if analysis engine
- Scenario simulation framework
- Intervention planning support

### Milestone 6.2: Causal Model Evaluation
**Objective:** Develop comprehensive evaluation framework
**Deliverables:**
- Ground truth dataset for causal relationships
- Evaluation metrics suite
- Model comparison framework
- Continuous evaluation pipeline

## Success Metrics
1. **Accuracy**: >80% precision in causal relationship extraction
2. **Performance**: <5s response time for typical causal analyses
3. **Scalability**: Support for batch processing of 10k+ records
4. **Integration**: Seamless workflow with existing architecture components
5. **Usability**: Comprehensive API documentation and examples

This milestone plan provides a structured approach to implementing causal inference capabilities within your knowledge graph system, with clear deliverables and technical approaches for each phase.