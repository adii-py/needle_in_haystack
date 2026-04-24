#!/bin/bash
# run.sh — Benchmark execution script for MRCR evaluation
#
# Usage:
#   ./run.sh <API_KEY> <RUN_ID> [other_options]
#
# Creates output directory in CURRENT WORKING DIRECTORY:
#   ./output/
#   ./output/${RUN_ID}_results.json
#   ./output/${RUN_ID}.log
#
# Example:
#   ./run.sh sk-xxxx run-001
#   ./run.sh sk-xxxx run-002 --n-needles 4

set -euo pipefail

# Script directory for finding venv/config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
# DEFAULT CONFIGURATION (Parse overrides first, then build CMD_ARGS)
# ═══════════════════════════════════════════════════════════════════════════════

# Context window sizes to test (tokens)
CONTEXT_SIZES=(64000 128000 192000)

# Needle count filter (2, 4, or 8)
N_NEEDLES=2

# Skip over-limit samples instead of truncating (official behavior)
NO_TRUNCATION=true

# Use auto-bin mode (each context size runs non-overlapping samples)
AUTO_BIN=true

# Samples per bin
SAMPLES_PER_BIN=100

# Output directory (will be computed after parsing args)
OUTPUT_DIR=""

# Log level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL="INFO"

# Config file path (relative to script)
CONFIG="${SCRIPT_DIR}/evals/mrcr/config.yaml"

# Random seed
SEED=42

# API URL (can be overridden via env or arg)
API_URL="${LITE_LLM_URL:-https://grid.ai.juspay.net}"
MODEL_NAME="${LITE_LLM_MODEL:-glm-latest}"

# ═══════════════════════════════════════════════════════════════════════════════
# PARSE OVERRIDES FROM ARGUMENTS
# ═══════════════════════════════════════════════════════════════════════════════

EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --context-sizes)
            # Override context sizes (handles both space-separated and comma-separated)
            CONTEXT_SIZES=()
            shift
            # Check if value is comma-separated
            if [[ "$1" =~ , ]]; then
                # Comma-separated: split into array
                IFS=',' read -ra CONTEXT_SIZES <<< "$1"
                shift
            else
                # Space-separated: collect all values until next flag
                while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                    CONTEXT_SIZES+=("$1")
                    shift
                done
            fi
            ;;
        --n-needles)
            N_NEEDLES="$2"
            shift 2
            ;;
        --samples-per-bin)
            SAMPLES_PER_BIN="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --api-url)
            API_URL="$2"
            shift 2
            ;;
        --no-truncation)
            NO_TRUNCATION=true
            shift
            # Also handle --no-truncation true/false
            if [[ "$1" == "true" ]] || [[ "$1" == "false" ]]; then
                if [[ "$1" == "false" ]]; then
                    NO_TRUNCATION=false
                fi
                shift
            fi
            ;;
        --truncation)
            NO_TRUNCATION=false
            shift
            # Also handle --truncation true/false
            if [[ "$1" == "true" ]] || [[ "$1" == "false" ]]; then
                if [[ "$1" == "true" ]]; then
                    NO_TRUNCATION=false
                fi
                shift
            fi
            ;;
        --auto-bin)
            AUTO_BIN=true
            shift
            # Also handle --auto-bin true/false
            if [[ "$1" == "true" ]] || [[ "$1" == "false" ]]; then
                if [[ "$1" == "false" ]]; then
                    AUTO_BIN=false
                fi
                shift
            fi
            ;;
        --no-auto-bin)
            AUTO_BIN=false
            shift
            ;;
        --base-output-dir)
            # Skip - we handle output-dir directly
            shift
            if [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; then
                shift
            fi
            ;;
        *)
            # Keep truly unknown args
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Compute output directory if not explicitly set
# Default: ./output/ in current working directory
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="./output"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD COMMAND ARGUMENTS
# ═══════════════════════════════════════════════════════════════════════════════

# Debug: show what we parsed
# echo "DEBUG: CONTEXT_SIZES=${CONTEXT_SIZES[*]}"
# echo "DEBUG: N_NEEDLES=$N_NEEDLES"
# echo "DEBUG: EXTRA_ARGS=$EXTRA_ARGS"

# Start with the base command
CMD_ARGS=""

# Context sizes
if [ ${#CONTEXT_SIZES[@]} -gt 0 ]; then
    CMD_ARGS="$CMD_ARGS --context-sizes ${CONTEXT_SIZES[@]}"
fi

# N-needles
CMD_ARGS="$CMD_ARGS --n-needles $N_NEEDLES"

# No truncation
if [ "$NO_TRUNCATION" = true ]; then
    CMD_ARGS="$CMD_ARGS --no-truncation"
fi

# Auto bin
if [ "$AUTO_BIN" = true ]; then
    CMD_ARGS="$CMD_ARGS --auto-bin"
fi

# Samples per bin
CMD_ARGS="$CMD_ARGS --samples-per-bin $SAMPLES_PER_BIN"

# Output directory
CMD_ARGS="$CMD_ARGS --output-dir $OUTPUT_DIR"

# Config file
CMD_ARGS="$CMD_ARGS --config $CONFIG"

# Seed
CMD_ARGS="$CMD_ARGS --seed $SEED"

# Log level
CMD_ARGS="$CMD_ARGS --log-level $LOG_LEVEL"

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

# Check if venv exists (relative to script location)
VENV_DIR="${SCRIPT_DIR}/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Running setup first..."
    "${SCRIPT_DIR}/setup.sh"
fi

# Activate venv
source "${VENV_DIR}/bin/activate"

# Export required environment variables
export LITE_LLM_API_KEY="$API_KEY"
export LITE_LLM_URL="$API_URL"
export LITE_LLM_MODEL="$MODEL_NAME"

echo "Configuration:"
echo "  Model:      $MODEL_NAME"
echo "  API URL:    $API_URL"
echo "  Config:     $CONFIG"
echo "  Output:     $OUTPUT_DIR"
echo ""
echo "Running: python -m evals.mrcr.run_multi_context $CMD_ARGS $EXTRA_ARGS"
echo ""

# Run the evaluation directly
python -m evals.mrcr.run_multi_context $CMD_ARGS $EXTRA_ARGS

echo ""
echo "Evaluation complete!"
echo "Run ID: $RUN_ID"
echo "Output: $OUTPUT_DIR"
