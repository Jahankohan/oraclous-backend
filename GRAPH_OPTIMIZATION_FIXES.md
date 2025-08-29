# Graph Optimization Fixes

## Issues Identified and Fixed

### 1. Database Schema Mismatch
**Issue**: "Unconsumed column names: communities_count, similarity_relationships"
**Root Cause**: The optimization code was trying to update columns that didn't exist in the database schema.
**Fix**: 
- Added `similarity_relationships` and `communities_count` columns to the `KnowledgeGraph` model
- Created migration file: `alembic/versions/add_optimization_columns.py`

### 2. Neo4j GDS Query Syntax Error  
**Issue**: "Unexpected configuration keys: nodeQuery, relationshipQuery"
**Root Cause**: Using deprecated syntax for `gds.graph.project` with `nodeQuery` and `relationshipQuery` parameters.
**Fix**: Updated all GDS graph projection queries to use the correct `gds.graph.project.cypher` syntax:

**Files Updated**:
- `app/services/analytics_service.py` - Lines 213, 61, 600
- `app/services/chat_service.py` - Lines 1429, 1559

**Before**:
```cypher
CALL gds.graph.project(
    'graph_name',
    { Entity: {...} },
    { RELATIONSHIP: {...} },
    { 
        nodeQuery: 'MATCH (n) WHERE n.graph_id = $graph_id RETURN id(n) AS id',
        relationshipQuery: 'MATCH (a)-[r]-(b) WHERE ... RETURN id(a) AS source, id(b) AS target'
    }
)
```

**After**:
```cypher
CALL gds.graph.project.cypher(
    'graph_name',
    'MATCH (n) WHERE n.graph_id = $graph_id RETURN id(n) AS id, n.name AS name',
    'MATCH (a)-[r]-(b) WHERE a.graph_id = $graph_id AND b.graph_id = $graph_id AND r.graph_id = $graph_id RETURN id(a) AS source, id(b) AS target',
    { parameters: {graph_id: $graph_id} }
)
```

### 3. Embedding Service Initialization
**Issue**: "Embedding service not initialized, skipping similarity relationships"
**Root Cause**: The embedding service wasn't being initialized before attempting to create similarity relationships.
**Fix**: Added automatic initialization in `_create_similarity_relationships` function.

**File**: `app/services/background_jobs.py` - Line 522
**Change**: Added initialization attempt with fallback if it fails.

## Next Steps

1. **Run the database migration**:
   ```bash
   cd knowledge-graph-builder
   alembic upgrade head
   ```

2. **Verify Neo4j GDS is installed**:
   Make sure your Neo4j instance has the Graph Data Science library installed. You can check this by running:
   ```cypher
   CALL gds.version()
   ```

3. **Test the optimization**:
   Try running the graph optimization endpoint again to verify the fixes work.

## Files Modified

1. `app/models/graph.py` - Added database columns
2. `app/services/analytics_service.py` - Fixed GDS syntax 
3. `app/services/chat_service.py` - Fixed GDS syntax
4. `app/services/background_jobs.py` - Fixed embedding service initialization
5. `alembic/versions/add_optimization_columns.py` - Database migration

## Notes

- The fixes maintain backward compatibility
- All graph operations now use proper graph_id filtering for multi-tenancy
- Embedding service will automatically try to initialize with OpenAI if available
- Community detection falls back to simple neighbor-based clustering if GDS fails
