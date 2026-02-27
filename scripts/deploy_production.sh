#!/usr/bin/env bash
# deploy_production.sh - Production deployment script
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
REGISTRY="${REGISTRY:-ghcr.io/donganhhuynh405-a11y/life-is-a-joke}"
VERSION="${VERSION:-$(git rev-parse --short HEAD 2>/dev/null || echo 'latest')}"
COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env"

echo "============================================"
echo "  Production Deployment"
echo "  Version: $VERSION"
echo "============================================"

cd "$ROOT_DIR"

# Verify environment file exists
if [ ! -f "$ENV_FILE" ]; then
  echo "✗ $ENV_FILE not found. Copy .env.template.secure to .env and configure it."
  exit 1
fi

# Check required env vars
REQUIRED_VARS=("DB_PASSWORD" "REDIS_PASSWORD" "GRAFANA_PASSWORD")
MISSING_VARS=()
for var in "${REQUIRED_VARS[@]}"; do
  if ! grep -q "^${var}=" "$ENV_FILE" 2>/dev/null; then
    MISSING_VARS+=("$var")
  fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
  echo "✗ Missing required environment variables in $ENV_FILE:"
  for var in "${MISSING_VARS[@]}"; do
    echo "  - $var"
  done
  exit 1
fi

echo "✓ Environment configuration validated"

# Pull latest images
echo ""
echo "Pulling latest base images..."
docker-compose -f "$COMPOSE_FILE" pull --quiet
echo "✓ Images pulled"

# Build application image
echo ""
echo "Building application image: $REGISTRY:$VERSION"
docker build -t "$REGISTRY:$VERSION" -t "$REGISTRY:latest" .
echo "✓ Image built"

# Run tests before deployment
echo ""
echo "Running pre-deployment tests..."
if command -v pytest &>/dev/null; then
  pytest tests/ -q --tb=short -x || {
    echo "✗ Tests failed - aborting deployment"
    exit 1
  }
  echo "✓ Tests passed"
else
  echo "⚠ pytest not available - skipping tests"
fi

# Backup database before deployment
echo ""
echo "Creating pre-deployment backup..."
bash "$SCRIPT_DIR/backup_restore.sh" backup || echo "⚠ Backup failed - continuing anyway"

# Stop existing services gracefully
echo ""
echo "Stopping existing services..."
docker-compose -f "$COMPOSE_FILE" down --timeout 30
echo "✓ Services stopped"

# Start services
echo ""
echo "Starting production services..."
export VERSION
docker-compose -f "$COMPOSE_FILE" up -d
echo "✓ Services started"

# Wait for health checks
echo ""
echo "Waiting for services to be healthy..."
TIMEOUT=120
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
  if docker-compose -f "$COMPOSE_FILE" ps | grep -q "unhealthy\|restarting"; then
    echo "  Waiting... ($ELAPSED/$TIMEOUT seconds)"
    sleep 10
    ELAPSED=$((ELAPSED + 10))
  else
    echo "✓ All services healthy"
    break
  fi
done

if [ $ELAPSED -ge $TIMEOUT ]; then
  echo "✗ Services failed health check after ${TIMEOUT}s"
  docker-compose -f "$COMPOSE_FILE" logs --tail=50
  exit 1
fi

# Show status
echo ""
echo "============================================"
echo "  Deployment Complete!"
echo "  Version: $VERSION"
echo "============================================"
echo ""
docker-compose -f "$COMPOSE_FILE" ps
echo ""
echo "Useful commands:"
echo "  make docker-logs    - View logs"
echo "  make docker-shell   - Open shell in container"
echo "  bash scripts/backup_restore.sh backup  - Manual backup"
