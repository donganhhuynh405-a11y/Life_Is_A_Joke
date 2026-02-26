#!/bin/bash
#
# QUICK START SCRIPT - Run the bot locally with Docker Compose
#
# Usage: bash run_local_demo.sh
#

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     Crypto AI Trading Bot - Local Demo                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

echo "✓ Docker and Docker Compose found"
echo ""

# Build image
echo "Building Docker image..."
docker build -t crypto-bot:local . --quiet
echo "✓ Image built successfully"
echo ""

# Start services
echo "Starting services with Docker Compose..."
docker-compose up -d --remove-orphans
echo ""

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 5

# Check services
echo ""
echo "Checking service status..."
docker-compose ps

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                   Services Running                            ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║                                                                ║"
echo "║  ✓ Redis        http://localhost:6379                        ║"
echo "║  ✓ Bot API      http://localhost:8000                        ║"
echo "║  ✓ Prometheus   http://localhost:8001/metrics                ║"
echo "║  ✓ Worker       (background Celery)                          ║"
echo "║                                                                ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║                   Next Steps                                   ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║                                                                ║"
echo "║  1. Run integration test:                                     ║"
echo "║     python tests/integration_test.py                          ║"
echo "║                                                                ║"
echo "║  2. Generate weekly report:                                   ║"
echo "║     python scripts/generate_weekly_report.py                  ║"
echo "║                                                                ║"
echo "║  3. Run failover demo:                                        ║"
echo "║     python scripts/failover_demo.py                           ║"
echo "║                                                                ║"
echo "║  4. Run 1-year backtest:                                      ║"
echo "║     python scripts/backtest_sim.py                            ║"
echo "║                                                                ║"
echo "║  5. View logs:                                                ║"
echo "║     docker logs -f bot-api                                    ║"
echo "║                                                                ║"
echo "║  6. Stop all services:                                        ║"
echo "║     docker-compose down                                       ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

echo "Tailing bot logs (press Ctrl+C to exit)..."
echo ""
docker logs -f bot-api 2>&1 | head -50 &
LOG_PID=$!

sleep 3
kill $LOG_PID 2>/dev/null || true

echo ""
echo "✓ Demo is running! Services are ready for testing."
echo ""
