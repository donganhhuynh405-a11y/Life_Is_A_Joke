#!/usr/bin/env bash
# run_tests.sh - Test runner script
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Default options
RUN_COVERAGE=false
RUN_LINT=false
VERBOSE=false
TEST_PATH="tests/"
FAIL_FAST=false
MARKERS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --coverage|-c)
      RUN_COVERAGE=true
      shift ;;
    --lint|-l)
      RUN_LINT=true
      shift ;;
    --verbose|-v)
      VERBOSE=true
      shift ;;
    --fast|-f)
      FAIL_FAST=true
      shift ;;
    --unit|-u)
      TEST_PATH="tests/"
      MARKERS="-m 'not integration'"
      shift ;;
    --integration|-i)
      TEST_PATH="tests/test_integration.py"
      shift ;;
    --path|-p)
      TEST_PATH="$2"
      shift 2 ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --coverage, -c      Run with coverage reporting"
      echo "  --lint, -l          Run linting before tests"
      echo "  --verbose, -v       Verbose output"
      echo "  --fast, -f          Stop on first failure"
      echo "  --unit, -u          Run unit tests only"
      echo "  --integration, -i   Run integration tests only"
      echo "  --path PATH, -p     Run tests in specific path"
      echo "  --help, -h          Show this help"
      exit 0 ;;
    *)
      echo "Unknown option: $1"
      exit 1 ;;
  esac
done

cd "$ROOT_DIR"

echo "============================================"
echo "  Running Tests"
echo "============================================"

# Activate venv if it exists
if [ -f "venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

# Run linting if requested
if [ "$RUN_LINT" = true ]; then
  echo ""
  echo "Running flake8 linter..."
  flake8 src/ tests/ backtester/ && echo "✓ Linting passed" || echo "⚠ Linting issues found"
fi

# Build pytest command
PYTEST_ARGS="$TEST_PATH --tb=short"

if [ "$VERBOSE" = true ]; then
  PYTEST_ARGS="$PYTEST_ARGS -v"
fi

if [ "$FAIL_FAST" = true ]; then
  PYTEST_ARGS="$PYTEST_ARGS -x"
fi

if [ -n "$MARKERS" ]; then
  PYTEST_ARGS="$PYTEST_ARGS $MARKERS"
fi

if [ "$RUN_COVERAGE" = true ]; then
  PYTEST_ARGS="$PYTEST_ARGS --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml"
fi

echo ""
echo "Running: pytest $PYTEST_ARGS"
echo ""

# Run tests
# shellcheck disable=SC2086
pytest $PYTEST_ARGS

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
  echo "✓ All tests passed!"
  if [ "$RUN_COVERAGE" = true ]; then
    echo "  Coverage report: htmlcov/index.html"
  fi
else
  echo "✗ Tests failed (exit code: $EXIT_CODE)"
fi

exit $EXIT_CODE
