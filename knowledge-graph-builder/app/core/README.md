# Core Directory

## 1. Overview
This directory contains foundational utilities and clients for the knowledge graph builder, including configuration, database connections, logging, and Neo4j client logic. These modules are used by most services and endpoints to interact with the database and manage application-wide settings.

## 2. Modules & Services

### config.py
- **Functionality:** Loads and manages application configuration (env vars, settings).
- **Connections:** Used by most modules to access config values.
- **Risks:** Tight coupling if config structure changes; risk of leaking secrets if not handled securely.

### database.py
- **Functionality:** Database connection management, session creation, and teardown.
- **Connections:** Used by services and models for DB access.
- **Risks:** Tight coupling to DB type (Neo4j); changes in DB require code updates.

### logging.py
- **Functionality:** Centralized logging setup and utilities.
- **Connections:** Used by all modules for logging.
- **Risks:** Logging misconfiguration can hide errors or leak sensitive data.

### neo4j_client.py
- **Functionality:** Handles connection and queries to Neo4j graph database.
- **Connections:** Used by graph-related services and endpoints.
- **Risks:** **Tight coupling** to Neo4j; schema changes require updates. Blocking calls can cause async/sync mismatches.
- **External APIs:** [Neo4j Python Driver](https://neo4j.com/docs/api/python-driver/current/)

### __init__.py
- **Functionality:** Package initializer.

### __pycache__/
- **Functionality:** Python bytecode cache; not relevant for documentation.

## 3. Interconnections
- Used by all services and endpoints for config, DB, and logging.
- Neo4j client is central for graph operations.

## 4. Example Flow
```
api/v1/endpoints/graph.py -> graph_service.py -> core/neo4j_client.py -> Neo4j
```

## 5. Notes
- **Tight Coupling:** Strong dependency on Neo4j and config structure.
- **Async/Sync Mismatches:** DB calls should be async if used in async endpoints.
- **Risks:** DB outages, config errors, logging misconfigurations.
- **Assumptions:** Neo4j is available and config is valid.

## 6. Refactoring Ideas
- Abstract DB logic to support multiple backends (e.g., SQL, document DBs).
- Use async DB drivers for better performance.
- Centralize config validation and secret management.
