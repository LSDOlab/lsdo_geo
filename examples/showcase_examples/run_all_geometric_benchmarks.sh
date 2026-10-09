#!/usr/bin/env bash
# ==============================================================================
# Master Benchmark Runner Script
# Runs all 4 geometric parameterization benchmarks (100 samples each) & plots
# ==============================================================================

# Default python executable: python3.14 (or PYTHON_BIN if exported in environment)
PYTHON_BIN="${PYTHON_BIN:-python3.14}"

# Verify python exists
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "Warning: '$PYTHON_BIN' not found on PATH. Falling back to 'python3'."
    PYTHON_BIN="python3"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Using Python: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"
echo "Working directory: $REPO_DIR"

cd "$REPO_DIR" || exit 1

"$PYTHON_BIN" "$SCRIPT_DIR/run_all_geometric_benchmarks.py" "$@"

