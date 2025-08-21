#!/bin/bash

echo "🐳 Building Knowledge Graph Builder Docker image..."

# Build the image
docker build -t knowledge-graph-builder:latest .

echo "✅ Docker image built successfully!"
echo "🚀 To run: docker-compose up knowledge-graph-builder"
