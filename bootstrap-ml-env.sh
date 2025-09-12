#!/bin/bash
# Bootstrap ML Environment for External Hosting
# This script creates a complete ML environment on the host that Docker containers can mount

set -e

echo "🚀 Bootstrapping ML Environment for Archon"
echo "This will create ML dependencies on host to keep Docker images minimal"

# Create directories
ML_ENV_DIR="/media/jonathanco/Backup/archon/ml-env"
MODELS_DIR="/media/jonathanco/Backup/archon/models"
TEMP_CONTAINER="archon-ml-bootstrap"

echo "📁 Creating directories..."
mkdir -p "$ML_ENV_DIR"
mkdir -p "$MODELS_DIR"

echo "🐳 Building temporary bootstrap container..."
docker build -f python/Dockerfile.server.bootstrap -t archon-ml-bootstrap ./python

echo "🏃 Running bootstrap container to extract ML environment..."
docker run --name "$TEMP_CONTAINER" \
  -v "$ML_ENV_DIR:/output" \
  -v "$MODELS_DIR:/models" \
  archon-ml-bootstrap

echo "🧹 Cleaning up bootstrap container..."
docker rm "$TEMP_CONTAINER"
docker rmi archon-ml-bootstrap

echo "✅ ML Environment Bootstrap Complete!"
echo "📊 Directory sizes:"
du -sh "$ML_ENV_DIR"
du -sh "$MODELS_DIR"

echo ""
echo "🎯 Next steps:"
echo "1. Build slim production images: docker compose build"
echo "2. Start services: docker compose up -d"
echo "3. ML deps will be available via volume mounts"