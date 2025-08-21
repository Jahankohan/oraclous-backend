echo "🔄 Migrating database to remove neo4j_database column..."

# Run the migration
alembic upgrade head

echo "✅ Database migration completed!"
echo ""
echo "Changes made:"
echo "- Removed neo4j_database column from knowledge_graphs table"
echo "- All graphs now use single Neo4j database with graph_id isolation"