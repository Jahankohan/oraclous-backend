#!/bin/bash

echo "🛑 Stopping Knowledge Graph Builder services..."

# Go to project root
cd "$(dirname "$0")/../.."

# Stop the services
docker-compose stop knowledge-graph-builder knowledge-graph-worker

echo "✅ Services stopped."
