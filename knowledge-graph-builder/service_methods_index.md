# Service Methods and Classes Index

This document lists all classes and methods in each service under `app/services/`, with at least two lines of description for each method.

---

## advanced_graph_analytic.py
### Classes:
- **SchemaPattern**
- **Community**
- **CentralityMetrics**
- **AdvancedGraphAnalytics**

#### AdvancedGraphAnalytics Methods:
- `__init__(self, neo4j_client: Neo4jClient)`
  - Initializes the analytics service with a Neo4j client. Sets up the connection for graph operations.
- `learn_graph_schema(self, min_frequency: int = 5)`
  - Learns the schema of the graph based on frequent patterns. Useful for schema inference and suggestions.
- `_discover_schema_patterns(self, min_frequency: int)`
  - Discovers recurring schema patterns in the graph. Helps identify common structures and relationships.
- `_generate_schema_suggestions(self, patterns)`
  - Generates schema improvement suggestions from discovered patterns. Supports schema evolution planning.
- `_validate_schema_patterns(self, patterns)`
  - Validates the quality and relevance of schema patterns. Ensures only robust patterns are considered.
- `_calculate_pattern_quality(self, pattern)`
  - Calculates a quality score for a schema pattern. Used to rank and filter patterns for schema updates.
- `_create_schema_evolution_plan(self, patterns)`
  - Creates a plan for evolving the graph schema. Guides incremental schema improvements.
- `_get_schema_statistics(self)`
  - Retrieves statistics about the current schema. Useful for monitoring schema health and coverage.
- `_store_schema_analysis(self, analysis)`
  - Stores the results of schema analysis. Persists insights for future reference and reporting.
- `perform_multi_hop_reasoning(...)`
  - Performs reasoning over multiple hops in the graph. Useful for complex query answering and inference.
- `_find_reasoning_paths(self, start_entities, max_hops)`
  - Finds reasoning paths between entities up to a given hop count. Supports advanced graph traversal.
- `_score_reasoning_paths(self, paths, question)`
  - Scores reasoning paths based on relevance to a question. Helps select the best explanation chains.
- `_extract_reasoning_chains(self, scored_paths)`
  - Extracts chains of reasoning from scored paths. Used for explainable AI and graph-based answers.
- `_create_path_string(self, nodes, relationships)`
  - Converts a path of nodes and relationships into a readable string. Useful for visualization and reporting.
- `_generate_reasoning_answer(self, reasoning_chains, question)`
  - Generates a natural language answer from reasoning chains. Integrates graph reasoning with LLMs.
- `_calculate_reasoning_confidence(self, scored_paths)`
  - Calculates confidence in reasoning results. Used for answer validation and ranking.
- `detect_communities(...)`
  - Detects communities in the graph using various algorithms. Supports social network analysis and clustering.
- `_build_networkx_graph(self)`
  - Builds a NetworkX graph from Neo4j data. Enables use of advanced graph algorithms.
- `_louvain_community_detection(self, graph, resolution)`
  - Applies Louvain algorithm for community detection. Identifies clusters in large graphs.
- `_leiden_community_detection(self, graph, resolution)`
  - Uses Leiden algorithm for improved community detection. Offers better quality and speed.
- `_infomap_community_detection(self, graph)`
  - Runs Infomap algorithm to find communities. Useful for information flow analysis.
- `_networkx_community_detection(self, graph)`
  - Uses NetworkX built-in methods for community detection. Provides baseline clustering results.
- `_analyze_community(self, comm_id, members, graph)`
  - Analyzes properties of a detected community. Extracts insights about member nodes and structure.
- `_get_community_labels(self, member_ids)`
  - Retrieves labels for community members. Useful for semantic analysis and reporting.
- `_generate_community_description(self, community)`
  - Generates a textual description of a community. Supports explainability and documentation.
- `_store_communities(self, communities)`
  - Stores detected communities in the database. Persists clustering results for future use.
- `calculate_centrality_metrics(self, algorithms)`
  - Calculates centrality metrics (degree, betweenness, etc.) for nodes. Identifies influential nodes.
- `_calculate_degree_centrality(self, graph)`
  - Computes degree centrality for all nodes. Measures direct connectivity.
- `_calculate_betweenness_centrality(self, graph)`
  - Calculates betweenness centrality. Identifies nodes that bridge communities.
- `_calculate_closeness_centrality(self, graph)`
  - Computes closeness centrality. Finds nodes with shortest paths to others.
- `_calculate_pagerank(self, graph)`
  - Runs PageRank algorithm. Ranks nodes by importance in the network.
- `_calculate_eigenvector_centrality(self, graph)`
  - Calculates eigenvector centrality. Identifies nodes connected to other influential nodes.
- `_store_centrality_metrics(self, metrics)`
  - Stores centrality metrics in the database. Enables historical analysis and monitoring.
- `_generate_centrality_summary(self, metrics)`
  - Summarizes centrality results for reporting. Provides insights into graph structure.
- `get_influential_nodes(...)`
  - Retrieves the most influential nodes based on centrality. Useful for targeting key entities.
- `find_bridge_nodes(self, top_k)`
  - Finds nodes that act as bridges between communities. Supports network resilience analysis.
- `analyze_graph_structure(self)`
  - Analyzes overall graph structure. Extracts metrics like connectivity, clustering, and components.
- `_calculate_basic_graph_metrics(self, graph)`
  - Calculates basic metrics (nodes, edges, density). Useful for graph health checks.
- `_analyze_connectivity(self, graph)`
  - Analyzes connectivity patterns. Identifies isolated nodes and subgraphs.
- `_analyze_clustering(self, graph)`
  - Measures clustering coefficients. Reveals local groupings in the graph.
- `_analyze_paths(self, graph)`
  - Analyzes path lengths and distributions. Supports shortest path analysis.
- `_analyze_components(self, graph)`
  - Identifies connected components. Useful for segmentation and modularity.
- `_analyze_degree_distribution(self, graph)`
  - Examines degree distribution. Detects power-law and other patterns.
- `_analyze_power_law(self, degrees)`
  - Tests for power-law behavior in degree distribution. Indicates scale-free properties.
- `_calculate_network_efficiency(self, graph)`
  - Calculates efficiency of information flow. Useful for optimization and robustness.
- `_store_graph_analysis(self, analysis)`
  - Stores results of graph analysis. Persists metrics for future reference.

---

## chat_service.py
### Classes:
- **ChatService**

#### ChatService Methods:
- `__init__(self, neo4j_client: Neo4jClient)`
  - Initializes the chat service with a Neo4j client and sets up embedding and LLM services.
  - Prepares the service for handling chat interactions and context retrieval.
- `chat(self, message, mode, file_names=None, session_id=None)`
  - Processes a chat message and returns a response using the selected mode.
  - Handles context retrieval, LLM response generation, and error management.
- `_get_context(self, message, mode, file_names=None)`
  - Retrieves relevant context for the chat based on the selected mode (vector, graph, fulltext, etc.).
  - Supports multiple retrieval strategies for flexible chat experiences.
- `_get_vector_context(self, message, file_names=None)`
  - Uses vector similarity search to find relevant context for the message.
  - Integrates with embedding service for semantic retrieval.
- `_get_graph_context(self, message, file_names=None)`
  - Retrieves context using graph traversal methods.
  - Useful for graph-based queries and relationship exploration.
- `_get_fulltext_context(self, message, file_names=None)`
  - Uses fulltext search to find relevant context in documents.
  - Supports keyword-based retrieval for chat.
- `_get_entity_vector_context(self, message, file_names=None)`
  - Retrieves context using entity-level embeddings.
  - Enables entity-centric chat and retrieval.
- `_get_global_vector_context(self, message, file_names=None)`
  - Uses global document embeddings for context retrieval.
  - Supports document-level semantic search.
- `_extract_entities_from_text(self, text)`
  - Extracts entities from text using NER techniques.
  - Useful for entity-aware chat and graph enrichment.
- `_generate_response(self, message, context, mode)`
  - Generates a response using LLMs and the provided context.
  - Integrates context into prompt for accurate answers.
- `_format_context(self, context)`
  - Formats context for LLM prompt construction.
  - Limits context size and organizes information for the model.
- `_store_conversation(self, session_id, message, response, context)`
  - Stores the conversation turn in Neo4j for history tracking.
  - Enables session management and retrieval of past interactions.
- `clear_chat_history(self, session_id)`
  - Clears chat history for a given session.
  - Useful for privacy and session reset.
- `get_chat_history(self, session_id, limit=50)`
  - Retrieves chat history for a session, limited to a specified number of turns.
  - Supports review and analysis of previous conversations.

---

## document_processor.py
### Classes:
- **DocumentProcessor**
- **ChunkingService**
- **BatchDocumentProcessor**

#### DocumentProcessor Methods:
- `__init__(self, neo4j_client: Neo4jClient)`
  - Initializes the processor with Neo4j client and supported document types.
  - Prepares the service for multi-format document ingestion.
- `process_document(self, source, source_type="auto", metadata=None)`
  - Processes a document and returns structured content.
  - Handles type detection, parsing, and error management.
- `_detect_source_type(self, source)`
  - Auto-detects the type of the document source (PDF, web, YouTube, etc.).
  - Enables flexible document handling and routing.
- `_generate_document_id(self, source, metadata)`
  - Generates a unique document ID using hashing and timestamp.
  - Ensures document uniqueness and traceability.
- `_is_document_processed(self, doc_id)`
  - Checks if a document has already been processed in Neo4j.
  - Prevents duplicate processing and saves resources.
- `_get_processed_document(self, doc_id)`
  - Retrieves processed document data from Neo4j.
  - Supports caching and fast access to results.
- `_process_pdf(self, source, metadata)`
  - Processes PDF files and extracts text/content.
  - Handles metadata and error management for PDFs.
- `_process_docx(self, source, metadata)`
  - Processes DOCX files and extracts text/content.
  - Supports advanced parsing for Word documents.
- `_process_txt(self, source, metadata)`
  - Processes plain text files and structures content.
  - Useful for simple document ingestion.
- `_process_html(self, source, metadata)`
  - Processes HTML files and extracts text/content.
  - Enables web page ingestion and parsing.
- `_process_youtube(self, url, metadata)`
  - Processes YouTube URLs and extracts transcripts/content.
  - Supports video-based document ingestion.
- `_process_web(self, url, metadata)`
  - Processes web URLs and extracts text/content.
  - Enables ingestion from arbitrary web sources.
- `_store_document_metadata(self, doc_data)`
  - Stores document metadata in Neo4j.
  - Ensures traceability and metadata management.

#### ChunkingService Methods:
- `__init__(self, chunk_size=1000, overlap=200)`
  - Initializes chunking service with specified chunk size and overlap.
  - Prepares for advanced text chunking operations.
- `chunk_with_overlap(self, text, chunk_size=None, overlap=None)`
  - Chunks text with overlap for semantic boundaries.
  - Improves downstream NLP and embedding quality.
- `_split_into_sentences(self, text)`
  - Splits text into sentences for finer chunking.
  - Supports sentence-aware chunking strategies.
- `_create_chunk(self, text, index)`
  - Creates a chunk dictionary with text and index.
  - Enables structured chunk management.
- `_get_overlap_text(self, text, overlap_size)`
  - Extracts overlap text for chunk boundaries.
  - Ensures context continuity between chunks.

#### BatchDocumentProcessor Methods:
- `__init__(self, neo4j_client: Neo4jClient)`
  - Initializes batch processor with Neo4j client.
  - Prepares for batch document processing and progress tracking.
- `process_batch(self, sources, batch_size=5, progress_callback=None)`
  - Processes multiple documents in batches, optionally with progress callback.
  - Enables scalable document ingestion and monitoring.
- `_process_document_chunks(self, document)`
  - Processes document chunks for batch operations.
  - Supports chunk-level processing and storage.
- `_store_chunk(self, document_id, chunk)`
  - Stores a chunk in Neo4j for a given document.
  - Ensures chunk persistence and traceability.

---

## document_service.py
### Classes:
- **DocumentService**

#### DocumentService Methods:
- `__init__(self, neo4j_client: Neo4jClient)`
  - Initializes the service with Neo4j client and text splitter.
  - Prepares for document scanning and chunking.
- `scan_sources(self, source_type, **kwargs)`
  - Scans and creates document source nodes for various types (local, S3, GCS, etc.).
  - Enables flexible document ingestion from multiple sources.
- `_scan_local_files(self, file_paths)`
  - Scans local files and creates document nodes in Neo4j.
  - Supports local file management and metadata extraction.
- `_scan_s3_bucket(self, bucket_name, prefix="")`
  - Scans S3 bucket for documents and creates nodes.
  - Enables cloud-based document ingestion.
- `_scan_gcs_bucket(self, project_id, bucket_name, prefix="")`
  - Scans GCS bucket for documents and creates nodes.
  - Supports Google Cloud document ingestion.
- `_scan_youtube_video(self, video_url)`
  - Scans YouTube video for transcript and creates document node.
  - Enables video-based document ingestion.
- `_scan_wikipedia_page(self, page_title)`
  - Scans Wikipedia page and creates document node.
  - Supports knowledge extraction from Wikipedia.
- `_scan_web_page(self, url)`
  - Scans web page and creates document node.
  - Enables ingestion from arbitrary web sources.
- `load_document_content(self, document_info)`
  - Loads actual document content using appropriate loader.
  - Supports multi-format document loading and parsing.
- `_load_local_file(self, document_info)`
  - Loads local file content and adds metadata.
  - Enables file-based document retrieval and enrichment.
- `_load_youtube_content(self, document_info)`
  - Loads YouTube video transcript for a document node.
  - Supports video transcript extraction and storage.
- `_load_wikipedia_content(self, document_info)`
  - Loads Wikipedia page content for a document node.
  - Enables knowledge extraction from Wikipedia.
- `_load_web_content(self, document_info)`
  - Loads web page content for a document node.
  - Supports web-based document retrieval.
- `split_documents(self, documents)`
  - Splits documents into chunks for processing.
  - Enables chunk-level analysis and storage.
- `create_chunk_nodes(self, document_info, chunks)`
  - Creates chunk nodes and relationships in Neo4j.
  - Supports chunk management and graph enrichment.
- `get_documents_list(self)`
  - Retrieves list of all documents from Neo4j.
  - Enables document management and listing.

---

## document_service_async.py
### Classes:
- **DocumentServiceAsync**

#### DocumentServiceAsync Methods:
- `__init__(self, neo4j_pool: Neo4jPool)`
  - Initializes the async service with Neo4j pool and text splitter.
  - Prepares for scalable async document operations.
- `scan_sources(self, source_type, **kwargs)`
  - Scans and creates document source nodes asynchronously.
  - Supports async ingestion from multiple sources.
- `_scan_local_files(self, file_paths)`
  - Scans local files and creates document nodes asynchronously.
  - Enables async local file management and metadata extraction.
- `get_documents_list(self)`
  - Retrieves list of all documents asynchronously from Neo4j.
  - Supports async document management and listing.
- `create_chunk_nodes(self, document_info, chunks)`
  - Creates chunk nodes and relationships asynchronously in Neo4j.
  - Enables async chunk management and graph enrichment.

---

## embedding_service.py
### Classes:
- **EmbeddingService**

#### EmbeddingService Methods:
- `__init__(self)`
  - Initializes the embedding service and sets up model dimensions.
  - Prepares for embedding generation and model selection.
- `_get_embedding_dimensions(self)`
  - Determines embedding dimensions based on selected model.
  - Ensures compatibility with downstream tasks.
- `dimensions(self)`
  - Returns the current embedding dimensions.
  - Useful for model introspection and validation.
- `_get_embedding_model(self)`
  - Gets or creates the embedding model instance.
  - Supports multiple providers (SentenceTransformer, OpenAI, VertexAI).
- `generate_embeddings(self, texts)`
  - Generates embeddings for a list of texts using the selected model.
  - Enables semantic search and retrieval.
- `generate_query_embedding(self, query)`
  - Generates embedding for a single query string.
  - Supports query-based semantic search.
- `similarity_search(self, query_embedding, limit=10)`
  - Performs vector similarity search in Neo4j using embeddings.
  - Enables retrieval of most similar chunks/documents.
- `calculate_similarity(self, embedding1, embedding2)`
  - Calculates cosine similarity between two embeddings.
  - Useful for ranking and clustering tasks.

---

## enhanced_chat_service.py
### Classes:
- **EnhancedChatService**

#### EnhancedChatService Methods:
- `__init__(self, neo4j_client: Neo4jClient)`
  - Initializes the enhanced chat service with analytics and embedding support.
  - Prepares for advanced chat and reasoning capabilities.
- `chat_with_graph(self, message, conversation_id=None, reasoning_type="auto", max_hops=3, include_analytics=True)`
  - Handles enhanced chat with multi-hop reasoning and analytics.
  - Integrates graph analytics and LLMs for richer responses.
- `_analyze_question(self, question)`
  - Analyzes the question to understand its structure and requirements.
  - Supports reasoning strategy selection and context extraction.
- `_detect_question_patterns(self, question)`
  - Detects specific patterns in the question (relationship, comparative, path).
  - Enables tailored reasoning and response generation.
- `_fallback_question_analysis(self, question)`
  - Provides fallback analysis for questions not matching known patterns.
  - Ensures robust handling of diverse queries.
- `_select_reasoning_strategy(self, analysis)`
  - Selects the best reasoning strategy based on question analysis.
  - Supports dynamic reasoning mode selection.
- `_get_context_for_question(self, question, analysis, reasoning_type, max_hops)`
  - Retrieves context for the question using selected reasoning type.
  - Enables multi-hop and semantic context extraction.
- `_get_semantic_context(self, question, analysis)`
  - Retrieves semantic context for the question.
  - Supports LLM-based semantic enrichment.
- `_get_multi_hop_context(self, question, analysis, max_hops)`
  - Retrieves multi-hop context for complex queries.
  - Enables deep graph traversal and reasoning.
- `_get_path_based_context(self, question, analysis)`
  - Retrieves context based on graph paths.
  - Supports journey/path queries in the graph.
- `_get_community_context(self, question, analysis)`
  - Retrieves community-based context for the question.
  - Enables social network and cluster analysis.
- `_get_centrality_context(self, question, analysis)`
  - Retrieves centrality-based context for the question.
  - Supports identification of influential entities.
- `_get_comparative_context(self, question, analysis)`
  - Retrieves comparative context for the question.
  - Enables comparison of entities and relationships.
- `_extract_entity_names_from_question(self, question)`
  - Extracts entity names from the question for targeted reasoning.
  - Supports entity-centric analysis and response.
- `_get_centrality_insights(self, central_entities)`
  - Retrieves insights for central entities in the graph.
  - Enables advanced analytics in chat responses.
- `_calculate_comparison_metrics(self, entities)`
  - Calculates metrics for comparing entities.
  - Supports comparative analysis in chat.
- `_generate_contextual_answer(self, question, context, analysis)`
  - Generates a contextual answer using LLMs and analytics.
  - Integrates context and analysis for rich responses.
- `_format_context_for_llm(self, context)`
  - Formats context for LLM prompt construction.
  - Ensures clarity and relevance in model input.
- `_get_system_prompt(self, context_type, analysis)`
  - Generates system prompt for LLM based on context type.
  - Supports dynamic prompt engineering.
- `_calculate_answer_confidence(self, context, analysis)`
  - Calculates confidence score for generated answers.
  - Enables answer validation and ranking.
- `_extract_sources_from_context(self, context)`
  - Extracts source references from context for explainability.
  - Supports traceable and transparent responses.
- `_get_analytics_insights_for_question(self, question, context)`
  - Retrieves analytics insights for the question.
  - Integrates graph analytics into chat answers.
- `_store_conversation_turn(self, conversation_id, question, answer, context)`
  - Stores a conversation turn in Neo4j for history tracking.
  - Enables session management and review.
- `get_conversation_history(self, conversation_id)`
  - Retrieves conversation history for a given ID.
  - Supports review and analysis of past interactions.

---

## enhanced_graph_service.py
### Classes:
- **EnhancedGraphService**

#### EnhancedGraphService Methods:
- `__init__(self, neo4j_client: Neo4jClient)`
  - Initializes the enhanced graph service with analytics and embedding support.
  - Prepares for advanced graph operations and analytics.
- `get_intelligent_graph_visualization(self, file_names=None, limit=100, layout_type="force_directed", include_analytics=True)`
  - Retrieves intelligent graph visualization with analytics overlays.
  - Supports multiple layout types and analytics integration.
- `_get_base_graph_data(self, file_names, limit)`
  - Retrieves base graph data for visualization.
  - Enables flexible graph rendering and analysis.
- `_enhance_nodes_with_centrality(self, nodes)`
  - Enhances nodes with centrality metrics for visualization.
  - Supports identification of key entities in the graph.
- `_enhance_nodes_with_communities(self, nodes)`
  - Enhances nodes with community information for visualization.
  - Enables cluster analysis and community overlays.
- `_get_community_color(self, community_id)`
  - Assigns colors to communities for visual distinction.
  - Supports clear and informative graph layouts.
- `_apply_intelligent_layout(self, graph, layout_type)`
  - Applies intelligent layout to the graph visualization.
  - Supports force-directed, hierarchical, and other layouts.
- `_apply_community_layout(self, graph)`
  - Applies community-based layout to the graph.
  - Highlights clusters and relationships visually.
- `_apply_centrality_layout(self, graph)`
  - Applies centrality-based layout to the graph.
  - Emphasizes influential nodes in the visualization.
- `_apply_hierarchical_layout(self, graph)`
  - Applies hierarchical layout to the graph.
  - Supports layered visualization of relationships.
- `_build_hierarchy_levels(self, graph, hierarchical_rels)`
  - Builds hierarchy levels for hierarchical layout.
  - Enables structured visualization of graph layers.
- `perform_intelligent_search(self, query, search_type="semantic", max_results=20, include_reasoning=True)`
  - Performs intelligent search in the graph using semantic and reasoning methods.
  - Supports advanced query answering and analytics.
- `_semantic_search(self, query, max_results)`
  - Performs semantic search in the graph.
  - Enables LLM-powered retrieval and ranking.
- `_multi_hop_search(self, query, max_results)`
  - Performs multi-hop search for complex queries.
  - Supports deep graph traversal and reasoning.
- `_community_aware_search(self, query, max_results)`
  - Performs community-aware search in the graph.
  - Enables cluster-based retrieval and analysis.
- `_hybrid_search(self, query, max_results)`
  - Performs hybrid search combining multiple strategies.
  - Supports robust and flexible query answering.
- `get_entity_insights(self, entity_id)`
  - Retrieves insights for a specific entity in the graph.
  - Enables entity-centric analytics and reporting.
- `_generate_entity_ai_insights(self, entity, relationships, centrality)`
  - Generates AI-powered insights for an entity.
  - Integrates relationships and centrality for rich analysis.
- `_calculate_entity_importance(self, centrality, relationships)`
  - Calculates importance score for an entity.
  - Supports ranking and prioritization in analytics.
- `run_comprehensive_analytics(self)`
  - Runs comprehensive analytics on the graph.
  - Extracts multiple metrics and insights for reporting.
- `get_analytics_dashboard_data(self)`
  - Retrieves data for analytics dashboard visualization.
  - Supports monitoring and decision-making.
- `delete_documents(self, file_names, delete_entities=False)`
  - Deletes documents and optionally their entities from the graph.
  - Supports data management and cleanup.
- `_recalculate_analytics_after_deletion(self)`
  - Recalculates analytics after document deletion.
  - Ensures metrics remain accurate and up-to-date.
- `get_advanced_duplicate_nodes(self, similarity_threshold=0.8, use_embeddings=True, use_fuzzy_matching=True)`
  - Finds advanced duplicate nodes using embeddings and fuzzy matching.
  - Supports robust deduplication and entity resolution.
- `_simple_duplicate_detection(self, threshold)`
  - Performs simple duplicate detection based on threshold.
  - Enables quick identification of duplicates.
- `merge_duplicate_nodes_advanced(self, node_ids, target_node_id, merge_strategy="comprehensive")`
  - Merges duplicate nodes using advanced strategies.
  - Supports comprehensive, conservative, and simple merging.
- `_comprehensive_merge(self, nodes_dict, target_node_id)`
  - Performs comprehensive merge of duplicate nodes.
  - Ensures data integrity and completeness.
- `_conservative_merge(self, nodes_dict, target_node_id)`
  - Performs conservative merge of duplicate nodes.
  - Minimizes risk and preserves original data.
- `_simple_merge(self, node_ids, target_node_id)`
  - Performs simple merge of duplicate nodes.
  - Enables fast and straightforward deduplication.
- `_update_analytics_after_merge(self, target_node_id)`
  - Updates analytics after node merge.
  - Ensures metrics reflect latest graph state.

---

## entity_resolution.py
### Classes:
- **EntityResolution**
- **SchemaLearning**

#### EntityResolution Methods:
- `__init__(self, neo4j_client: Neo4jClient)`
  - Initializes entity resolution with Neo4j client and matching thresholds.
  - Prepares for advanced deduplication and similarity analysis.
- `find_duplicate_entities(self, batch_size=1000)`
  - Finds duplicate entities using multiple matching strategies.
  - Supports batch processing and error management.
- `resolve_entity_duplicates(self, duplicate_groups, auto_merge=False)`
  - Resolves duplicate entities by merging them.
  - Tracks merge results and errors for reporting.
- `_find_duplicates_in_batch(self, entities)`
  - Finds duplicates within a batch of entities.
  - Uses multiple similarity measures for robust detection.
- `_calculate_entity_similarity(self, entity1, entity2)`
  - Calculates similarity between two entities using exact, normalized, fuzzy, and embedding methods.
  - Supports composite similarity scoring for deduplication.
- `_normalize_entity_name(self, name)`
  - Normalizes entity name for case, whitespace, and punctuation.
  - Improves matching accuracy and consistency.
- `_calculate_property_similarity(self, props1, props2)`
  - Calculates similarity between entity properties.
  - Supports property-aware deduplication.
- `_merge_duplicate_groups(self, duplicates)`
  - Merges duplicate groups into canonical entities.
  - Ensures data integrity and traceability.
- `_score_duplicate_groups(self, groups)`
  - Scores duplicate groups for merge prioritization.
  - Enables ranking and selection of merge candidates.
- `_select_canonical_entity(self, entities)`
  - Selects the canonical entity from a group of duplicates.
  - Supports rule-based and data-driven selection.
- `_merge_entity_group(self, entities, canonical_id)`
  - Merges a group of entities into the canonical entity.
  - Ensures consistency and completeness in merges.
- `_store_merge_suggestion(self, group)`
  - Stores merge suggestion for review and approval.
  - Enables human-in-the-loop deduplication.

#### SchemaLearning Methods:
- `__init__(self, neo4j_client: Neo4jClient)`
  - Initializes schema learning with Neo4j client.
  - Prepares for schema evolution and improvement.
- `learn_schema_from_data(self, domain_context=None)`
  - Learns schema from data and domain context.
  - Supports schema inference and evolution.
- `_analyze_current_schema(self)`
  - Analyzes the current schema for improvement areas.
  - Enables targeted schema suggestions.
- `_generate_schema_suggestions(self, current_schema, domain_context=None)`
  - Generates schema suggestions using LLMs and rules.
  - Supports automated and rule-based schema improvement.
- `_generate_rule_based_suggestions(self, current_schema)`
  - Generates rule-based schema suggestions.
  - Ensures compliance and best practices.
- `_validate_schema_suggestions(self, suggestions)`
  - Validates schema suggestions for feasibility and impact.
  - Ensures only robust suggestions are applied.
- `_calculate_suggestion_feasibility(self, suggestion)`
  - Calculates feasibility score for a schema suggestion.
  - Supports prioritization and selection.
- `_estimate_suggestion_impact(self, suggestion)`
  - Estimates impact of a schema suggestion.
  - Enables decision-making for schema changes.
- `_identify_improvement_areas(self, current_schema)`
  - Identifies areas for schema improvement.
  - Supports continuous schema evolution.

---

## extraction_service.py
### Classes:
- **ExtractionService**

#### ExtractionService Methods:
- `__init__(self, neo4j_client: Neo4jClient)`
  - Initializes extraction service with Neo4j client and embedding/LLM support.
  - Prepares for knowledge graph extraction from documents.
- `extract_graph(self, file_names, model, node_labels=None, relationship_types=None, enable_schema=True)`
  - Extracts knowledge graph from documents using LLMs and graph transformers.
  - Supports schema-aware and batch extraction.
- `_get_documents_to_process(self, file_names)`
  - Retrieves documents that need processing for extraction.
  - Enables targeted and efficient extraction workflows.
- `_get_document_chunks(self, document_id)`
  - Retrieves chunks for a document from Neo4j.
  - Supports chunk-level graph extraction and embedding.
- `_extract_batch_graph(self, graph_transformer, chunks)`
  - Extracts graph from a batch of document chunks using LLMs.
  - Enables scalable and parallel graph extraction.
- `_store_graph_documents(self, graph_documents, document_id)`
  - Stores extracted graph documents in Neo4j.
  - Ensures persistence and traceability of extracted knowledge.
- `_generate_chunk_embeddings(self, chunks)`
  - Generates embeddings for document chunks.
  - Supports semantic enrichment and retrieval.
- `_update_document_status(self, document_id, status)`
  - Updates document processing status in Neo4j.
  - Enables progress tracking and workflow management.

---

## graph_service.py
### Classes:
- **GraphService**

#### GraphService Methods:
- `__init__(self, neo4j_client: Neo4jClient)`
  - Initializes graph service with Neo4j client and embedding/LLM support.
  - Prepares for core graph operations and analytics.
- `get_graph_visualization(self, file_names=None, limit=100)`
  - Retrieves graph visualization data for specified files.
  - Supports flexible and interactive graph rendering.
- `get_node_neighbors(self, node_id, depth=1)`
  - Retrieves neighbors of a specific node up to a given depth.
  - Enables exploration of graph relationships and structure.
- `delete_documents(self, file_names, delete_entities=False)`
  - Deletes documents and optionally their entities from the graph.
  - Supports data management and cleanup.
- `get_duplicate_nodes(self)`
  - Finds potential duplicate nodes based on similarity.
  - Enables deduplication and entity resolution.
- `merge_duplicate_nodes(self, node_ids, target_node_id)`
  - Merges duplicate nodes into a target node.
  - Ensures data integrity and traceability.
- `get_unconnected_nodes(self)`
  - Retrieves list of unconnected entity nodes.
  - Supports graph cleanup and optimization.
- `delete_unconnected_nodes(self, node_ids)`
  - Deletes unconnected entity nodes from the graph.
  - Enables efficient graph management.
- `post_process_graph(self)`
  - Runs post-processing tasks on the graph.
  - Supports graph optimization and enrichment.
- `find_duplicate_nodes(self, threshold=None)`
  - Finds potential duplicate entity nodes using a threshold.
  - Enables robust deduplication strategies.
- `merge_duplicate_nodes(self, node_ids, target_node_id)`
  - Merges duplicate nodes into a target node (overloaded for different strategies).
  - Ensures comprehensive deduplication.
- `_create_entity_embeddings(self)`
  - Creates embeddings for entities without existing embeddings.
  - Supports semantic enrichment and retrieval.
- `_create_similarity_relationships(self)`
  - Creates similarity relationships between entities.
  - Enables graph-based similarity search and clustering.
- `_detect_communities(self)`
  - Detects communities in the graph.
  - Supports social network analysis and clustering.
- `_generate_entity_descriptions(self)`
  - Generates descriptions for entities in the graph.
  - Enables explainability and documentation.
- `generate_schema_suggestions(self, text, model=None)`
  - Generates schema suggestions using LLMs and text input.
  - Supports schema evolution and improvement.

---

## multi_modal_processing.py
### Classes:
- **MultiModalProcessor**

#### MultiModalProcessor Methods:
- `__init__(self, neo4j_client: Neo4jClient)`
  - Initializes multi-modal processor with Neo4j client and LLM support.
  - Prepares for advanced multi-format document processing.
- `process_multimodal_document(self, file_path, document_id)`
  - Processes a multi-modal document and returns structured content.
  - Handles type detection, parsing, and error management for various formats.
- `_process_pdf_advanced(self, file_path, document_id)`
  - Processes PDF files with advanced extraction (text, images, tables).
  - Supports multi-modal enrichment and analysis.
- `_process_pdf_page(self, page, page_num, document_id)`
  - Processes a single PDF page for text, images, and tables.
  - Enables page-level analysis and extraction.
- `_ocr_pdf_page(self, page, page_num)`
  - Performs OCR on a PDF page to extract text.
  - Supports image-based text extraction and enrichment.
- `_extract_tables_from_page(self, page, page_num)`
  - Extracts tables from a PDF page.
  - Enables structured data extraction from documents.
- `_extract_images_from_page(self, page, page_num, document_id)`
  - Extracts images from a PDF page for analysis.
  - Supports image-centric document enrichment.
- `_analyze_image(self, image, image_id)`
  - Analyzes an image for content and features.
  - Enables vision-based enrichment and tagging.
- `_classify_image_content(self, image)`
  - Classifies image content using vision models.
  - Supports automated image labeling and categorization.
- `_describe_image_with_vision_model(self, image)`
  - Generates a description for an image using LLM/vision models.
  - Enables explainable image analysis.
- `_process_excel(self, file_path, document_id)`
  - Processes Excel files for tabular data extraction.
  - Supports spreadsheet-based knowledge extraction.
- `_process_excel_sheet(self, worksheet, document_id)`
  - Processes a single Excel worksheet for data and structure.
  - Enables sheet-level analysis and enrichment.
- `_detect_header_row(self, df)`
  - Detects header row in a DataFrame for accurate parsing.
  - Supports robust tabular data extraction.
- `_analyze_tabular_data(self, df, sheet_name)`
  - Analyzes tabular data for relationships and structure.
  - Enables knowledge graph enrichment from tables.
- `_detect_column_relationships(self, df)`
  - Detects relationships between columns in tabular data.
  - Supports schema inference and graph construction.
- `_has_formulas(self, worksheet)`
  - Checks if a worksheet contains formulas.
  - Enables formula-aware data extraction.
- `_analyze_cell_formatting(self, worksheet)`
  - Analyzes cell formatting for semantic cues.
  - Supports advanced spreadsheet parsing.
- `_analyze_excel_chart(self, chart, sheet_name)`
  - Analyzes Excel chart for data and structure.
  - Enables chart-based knowledge extraction.
- `_detect_pivot_tables(self, worksheet)`
  - Detects pivot tables in a worksheet.
  - Supports advanced tabular analysis.
- `_process_powerpoint(self, file_path, document_id)`
  - Processes PowerPoint files for slide and content extraction.
  - Enables presentation-based knowledge enrichment.
- `_process_powerpoint_slide(self, slide, slide_num, document_id)`
  - Processes a single PowerPoint slide for text, images, and tables.
  - Supports slide-level analysis and extraction.
- `_extract_font_info(self, shape)`
  - Extracts font information from a shape in PowerPoint.
  - Enables style-aware content analysis.
- `_extract_powerpoint_image(self, shape, slide_num, document_id)`
  - Extracts images from a PowerPoint slide for analysis.
  - Supports image-centric presentation enrichment.
- `_analyze_powerpoint_chart(self, chart, slide_num)`
  - Analyzes PowerPoint chart for data and structure.
  - Enables chart-based knowledge extraction from presentations.
- `_extract_powerpoint_table(self, table, slide_num)`
  - Extracts tables from a PowerPoint slide.
  - Supports structured data extraction from presentations.
- `_process_csv(self, file_path, document_id)`
  - Processes CSV files for tabular data extraction.
  - Enables spreadsheet-based knowledge enrichment.
- `_assess_data_quality(self, df)`
  - Assesses data quality in a DataFrame.
  - Supports robust tabular data analysis.
- `_process_json(self, file_path, document_id)`
  - Processes JSON files for structured data extraction.
  - Enables schema inference and graph enrichment.
- `_analyze_json_structure(self, data)`
  - Analyzes JSON structure for schema inference.
  - Supports automated graph construction from JSON.
- `_get_json_sample(self, data, max_items=5)`
  - Retrieves a sample from JSON data for analysis.
  - Enables quick inspection and validation.
- `_infer_json_schema(self, data)`
  - Infers schema from JSON data.
  - Supports automated schema construction.
- `_process_html(self, file_path, document_id)`
  - Processes HTML files for text and structure extraction.
  - Enables web page enrichment and parsing.
- `_process_xml(self, file_path, document_id)`
  - Processes XML files for structured data extraction.
  - Supports schema inference and graph enrichment.
- `_analyze_xml_element(self, element, max_depth=3, current_depth=0)`
  - Analyzes XML element for structure and relationships.
  - Enables deep XML parsing and graph construction.
- `_extract_xml_namespaces(self, root)`
  - Extracts XML namespaces from the root element.
  - Supports robust XML parsing and schema inference.
- `_process_image(self, file_path, document_id)`
  - Processes image files for content and feature extraction.
  - Enables vision-based knowledge enrichment.
- `_process_word(self, file_path, document_id)`
  - Processes Word files for text and structure extraction.
  - Supports document-based knowledge enrichment.

