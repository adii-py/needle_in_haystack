#!/bin/bash
# run.sh — Quick run script with pre-configured defaults for MRCR evaluation
#
# Usage:
#   ./run.sh <API_KEY> <RUN_ID> [other_options]
#
# Example:
#   ./run.sh sk-FDSlUBMxV9PbH5-Y_Rxxxx 37ab8300-288c-4a2a-950b-df53124f0b23
#   ./run.sh sk-xxxx run-id-123 --n-needles 4 --output-dir custom/path

set -e
cd "$(dirname "$0")"

# ═══════════════════════════════════════════════════════════════════════════════
# PARSE REQUIRED ARGUMENTS (API_KEY and RUN_ID)
# ═══════════════════════════════════════════════════════════════════════════════

if [ $# -lt 2 ]; then
    echo "Usage: $0 <API_KEY> <RUN_ID> [other_options...]"
    echo ""
    echo "Required arguments:"
    echo "  API_KEY    - The API key for LLM service"
    echo "  RUN_ID     - Unique identifier for this run"
    echo ""
    echo "Optional arguments:"
    echo "  --n-needles N              Needle count (2, 4, or 8) [default: 2]"
    echo "  --context-sizes N N N      Context window sizes [default: 64000 128000 192000]"
    echo "  --samples-per-bin N        Samples per bin [default: 100]"
    echo "  --output-dir PATH          Output directory"
    echo "  --no-truncation            Skip over-limit samples"
    echo "  --no-auto-bin              Disable auto-bin mode"
    echo "  --seed N                   Random seed"
    echo "  --log-level LEVEL          DEBUG/INFO/WARNING/ERROR"
    echo ""
    echo "Example:"
    echo "  ./run.sh sk-xxxx run-001"
    echo "  ./run.sh sk-xxxx run-002 --n-needles 4 --output-dir outputs/custom"
    exit 1
fi

API_KEY="$1"
RUN_ID="$2"
shift 2
# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Context window sizes to test (tokens)
DEFAULT_CONTEXT_SIZES=(64000 128000 192000)

# Needle count filter (2, 4, or 8)
DEFAULT_N_NEEDLES=2

# Skip over-limit samples instead of truncating (official behavior)
DEFAULT_NO_TRUNCATION=true

# Use auto-bin mode (each context size runs non-overlapping samples)
DEFAULT_AUTO_BIN=true

# Samples per bin
DEFAULT_SAMPLES_PER_BIN=100

# Output directory - will append RUN_ID
DEFAULT_OUTPUT_DIR="outputs/mrcr/${RUN_ID}_needle${DEFAULT_N_NEEDLES}"

# Log level (DEBUG, INFO, WARNING, ERROR)
DEFAULT_LOG_LEVEL="INFO"

# Config file path
DEFAULT_CONFIG="evals/mrcr/config.yaml"

# Random seed
DEFAULT_SEED=42

# API URL (can be overridden via env or arg)
API_URL="${LITE_LLM_URL:-https://grid.ai.juspay.net}"
MODEL_NAME="${LITE_LLM_MODEL:-glm-latest}"

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD COMMAND ARGUMENTS
# ═══════════════════════════════════════════════════════════════════════════════

# Start with the base command
CMD_ARGS=""

# Context sizes
if [ ${#DEFAULT_CONTEXT_SIZES[@]} -gt 0 ]; then
    CMD_ARGS="$CMD_ARGS --context-sizes ${DEFAULT_CONTEXT_SIZES[@]}"
fi

# N-needles
CMD_ARGS="$CMD_ARGS --n-needles $DEFAULT_N_NEEDLES"

# No truncation
if [ "$DEFAULT_NO_TRUNCATION" = true ]; then
    CMD_ARGS="$CMD_ARGS --no-truncation"
fi

# Auto bin
if [ "$DEFAULT_AUTO_BIN" = true ]; then
    CMD_ARGS="$CMD_ARGS --auto-bin"
fi

# Samples per bin
CMD_ARGS="$CMD_ARGS --samples-per-bin $DEFAULT_SAMPLES_PER_BIN"

# Output directory
CMD_ARGS="$CMD_ARGS --output-dir $DEFAULT_OUTPUT_DIR"

# Config file
CMD_ARGS="$CMD_ARGS --config $DEFAULT_CONFIG"

# Seed
CMD_ARGS="$CMD_ARGS --seed $DEFAULT_SEED"

# Log level
CMD_ARGS="$CMD_ARGS --log-level $DEFAULT_LOG_LEVEL"

# ═══════════════════════════════════════════════════════════════════════════════
# OVERRIDE HANDLING (Process remaining arguments)
# ═══════════════════════════════════════════════════════════════════════════════

EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-needles|--output-dir|--seed|--log-level|--config|--samples-per-bin|--context-sizes)
            # These override the defaults
            EXTRA_ARGS="$EXTRA_ARGS $1 $2"
            # Update default output dir if n-needles changes
            if [ "$1" = "--n-needles" ]; then
                DEFAULT_N_NEEDLES="$2"
                DEFAULT_OUTPUT_DIR="outputs/mrcr/${RUN_ID}_needle${DEFAULT_N_NEEDLES}"
            fi
            shift 2
            ;;
        --no-truncation|--auto-bin|--truncation|--no-auto-bin)
            # Toggle flags
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
        *)
            # Pass through unknown args
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# ═══════════════════════════════════════════════════════════════════════════════
# RUN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

echo "=========================================="
echo "MRCR Multi-Context Evaluation"
echo "=========================================="
echo ""
echo "Run ID:     $RUN_ID"
echo "API Key:    ${API_KEY:0:10}..."
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running setup first..."
    ./setup.sh
fi

# Activate venv
source venv/bin/activate

# Export required environment variables
export LITE_LLM_API_KEY="$API_KEY"
export LITE_LLM_URL="$API_URL"
export LITE_LLM_MODEL="$MODEL_NAME"

echo "Configuration:"
echo "  Model:      $MODEL_NAME"
echo "  API URL:    $API_URL"
echo "  Config:     $DEFAULT_CONFIG"
echo "  Output:     $DEFAULT_OUTPUT_DIR"
echo ""
echo "Running: python -m evals.mrcr.run_multi_context $CMD_ARGS $EXTRA_ARGS"
echo ""

# Run the evaluation directly
python -m evals.mrcr.run_multi_context $CMD_ARGS $EXTRA_ARGS

echo ""
echo "Evaluation complete!"
echo "Run ID: $RUN_ID"
echo "Output: $DEFAULT_OUTPUT_DIR"
