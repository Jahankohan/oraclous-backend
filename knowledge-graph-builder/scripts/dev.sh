#!/bin/bash

echo "🔧 Starting Knowledge Graph Builder in development mode..."

# Go to project root
cd "$(dirname "$0")/../.."

# Check if .env exists in knowledge-graph-builder
if [ ! -f knowledge-graph-builder/.env ]; then
    echo "❌ .env file not found in knowledge-graph-builder/. Please create it."
    exit 1
fi

# Start all required services using the main docker-compose
echo "📦 Starting required services (neo4j, postgres, redis)..."
docker-compose up -d neo4j postgres redis

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Start the knowledge graph builder service
echo "🚀 Starting knowledge-graph-builder service..."
docker-compose up -d knowledge-graph-builder

# Start the Celery worker
echo "👷 Starting Celery worker..."
docker-compose up -d knowledge-graph-worker

echo "✅ All services started!"
echo ""
echo "Services running:"
echo "- Knowledge Graph Builder: http://localhost:8003"
echo "- Neo4j Browser: http://localhost:7474"
echo "- API Docs: http://localhost:8003/docs"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f knowledge-graph-builder"
echo "  docker-compose logs -f knowledge-graph-worker"
