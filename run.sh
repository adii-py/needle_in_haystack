#!/bin/bash
# run.sh — Benchmark execution script for MRCR evaluation
#
# Usage (per dashboard spec):
#   ./run.sh <api_key_starting_with_sk-> <uuid_run_id> --concurrency N --domain D --max-steps N --model M --trials N
#
# Creates output directory at ./output/:
#   ./output/${RUNNER_ID}_result.json
#   ./output/${RUNNER_ID}.log
#
# Example:
#   ./run.sh sk-xxxx 550e8400-e29b-41d4-a716-446655440000 --n-needles 4 --context-sizes 64000 128000

set -euo pipefail

# Script directory for finding venv/config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0: HANDLE HELP FLAG (before required args check)
# ═══════════════════════════════════════════════════════════════════════════════

if [[ $# -gt 0 ]] && ([[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]); then
    echo "Usage: $0 <api_key> <run_id> [options]"
    echo ""
    echo "Required positional arguments:"
    echo "  api_key    - API key (starts with sk-)"
    echo "  run_id     - UUID format run identifier"
    echo ""
    echo "Dashboard-compatible flags:"
    echo "  --concurrency N     - Not used (accepted for compatibility)"
    echo "  --domain D          - Domain (small|medium|large) maps to n-needles"
    echo "  --max-steps N       - Maps to samples-per-bin"
    echo "  --model M           - Model name to use"
    echo "  --trials N          - Maps to random seed"
    echo ""
    echo "Additional options:"
    echo "  --context-sizes N N N   - Context window sizes [default: 64000 128000 192000]"
    echo "  --n-needles N           - Needle count (2, 4, or 8) [default: 2]"
    echo "  --samples-per-bin N     - Samples per bin [default: 100]"
    echo "  --seed N                - Random seed [default: 42]"
    echo "  --log-level LEVEL       - DEBUG/INFO/WARNING/ERROR [default: INFO]"
    echo "  --api-url URL           - API endpoint [default: https://grid.ai.juspay.net/v1]"
    echo "  --[no-]truncation       - Enable/disable truncation"
    echo "  --[no-]auto-bin         - Enable/disable auto-bin mode"
    echo "  --output-dir PATH       - Output directory [default: ./output]"
    echo "  --config PATH           - Config file path"
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: PARSE POSITIONAL ARGS (API_KEY and RUNNER_ID)
# Dashboard passes: ./run.sh <api_key> <uuid> --flag1 --flag2 ...
# ═══════════════════════════════════════════════════════════════════════════════

API_KEY=""
RUNNER_ID=""

# Parse first positional arg: API key (starts with sk-)
if [[ $# -gt 0 ]] && [[ "$1" == sk-* ]]; then
    API_KEY="$1"
    shift
fi

# Parse second positional arg: UUID run ID
if [[ $# -gt 0 ]] && [[ "$1" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]]; then
    RUNNER_ID="$1"
    shift
fi

# Validate required args
if [ -z "$API_KEY" ]; then
    echo "[ERROR] API key (starting with sk-) is required as first positional argument"
    echo ""
    echo "Usage: $0 <api_key> <run_id> [options]"
    echo ""
    echo "Examples:"
    echo "  $0 sk-xxxx 550e8400-e29b-41d4-a716-446655440000"
    echo "  $0 sk-xxxx 550e8400-e29b-41d4-a716-446655440000 --n-needles 4 --samples-per-bin 50"
    exit 1
fi

if [ -z "$RUNNER_ID" ]; then
    echo "[ERROR] Run ID (UUID format) is required as second positional argument"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: DEFAULT CONFIGURATION
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

# Output directory
OUTPUT_DIR="${SCRIPT_DIR}/output"

# Log level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL="INFO"

# Config file path (relative to script)
CONFIG="${SCRIPT_DIR}/evals/mrcr/config.yaml"

# Random seed
SEED=42

# Model name (will be overridden if --model flag is passed)
MODEL_NAME=""

# API URL (set after parsing flags, so it can be overridden)
API_URL=""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: PARSE FLAG ARGUMENTS (--flag value)
# ═══════════════════════════════════════════════════════════════════════════════

while [[ $# -gt 0 ]]; do
    case $1 in
        --concurrency)
            # Map --concurrency to max_workers if supported
            # Not used by MRCR currently, but accept and ignore
            shift 2
            ;;
        --domain)
            # Map --domain to n_needles: small=2, medium=4, large=8
            DOMAIN="$2"
            case "$DOMAIN" in
                small) N_NEEDLES=2 ;;
                medium) N_NEEDLES=4 ;;
                large) N_NEEDLES=8 ;;
                *) echo "[WARNING] Unknown domain '$DOMAIN', using default n_needles=$N_NEEDLES" ;;
            esac
            shift 2
            ;;
        --max-steps)
            # Map --max-steps to samples-per-bin
            SAMPLES_PER_BIN="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --trials)
            # Map --trials to seed for reproducibility
            SEED="$2"
            shift 2
            ;;
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
        --api-url)
            API_URL="$2"
            shift 2
            ;;
        --no-truncation)
            NO_TRUNCATION=true
            shift
            ;;
        --truncation)
            NO_TRUNCATION=false
            shift
            ;;
        --auto-bin)
            AUTO_BIN=true
            shift
            ;;
        --no-auto-bin)
            AUTO_BIN=false
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "[WARNING] Unknown argument: $1 (skipping)"
            shift
            ;;
    esac
done

# Set default API URL after parsing (so --api-url can override)
if [ -z "$API_URL" ]; then
    API_URL="https://grid.ai.juspay.net/v1"
fi

# Export LiteLLM environment variables (required by runner.py)
export LITE_LLM_API_KEY="$API_KEY"
export LITE_LLM_URL="$API_URL"

# Prefix model name with openai/ if not already (OpenCompass convention)
if [ -n "$MODEL_NAME" ]; then
    if [[ ! "$MODEL_NAME" =~ ^openai/ ]]; then
        MODEL_NAME="openai/$MODEL_NAME"
    fi
    export LITE_LLM_MODEL="$MODEL_NAME"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: VALIDATE AND PREPARE
# ═══════════════════════════════════════════════════════════════════════════════

echo "=========================================="
echo "MRCR Multi-Context Evaluation"
echo "=========================================="
echo ""
echo "Run ID:     $RUNNER_ID"
echo "API Key:    ${API_KEY:0:10}..."
echo "API URL:    $API_URL"
echo "Model:      ${MODEL_NAME:-(from config/env)}"
echo ""

# Check if venv exists (relative to script location)
VENV_DIR="${SCRIPT_DIR}/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[ERROR] Virtual environment not found at $VENV_DIR"
    echo "Please run ./setup.sh first"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Activate venv
source "${VENV_DIR}/bin/activate"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: BUILD COMMAND ARRAY (NEVER use eval)
# ═══════════════════════════════════════════════════════════════════════════════

# Initialize command array
CMD_ARGS=("${VENV_DIR}/bin/python" "-m" "evals.mrcr.run_multi_context")

# Add context sizes
if [ ${#CONTEXT_SIZES[@]} -gt 0 ]; then
    CMD_ARGS+=("--context-sizes" "${CONTEXT_SIZES[@]}")
fi

# Add n-needles
CMD_ARGS+=("--n-needles" "$N_NEEDLES")

# Add truncation flag
if [ "$NO_TRUNCATION" = true ]; then
    CMD_ARGS+=("--no-truncation")
fi

# Add auto-bin flag
if [ "$AUTO_BIN" = true ]; then
    CMD_ARGS+=("--auto-bin")
fi

# Add samples-per-bin
CMD_ARGS+=("--samples-per-bin" "$SAMPLES_PER_BIN")

# Add output directory
CMD_ARGS+=("--output-dir" "$OUTPUT_DIR")

# Add config file
CMD_ARGS+=("--config" "$CONFIG")

# Add seed
CMD_ARGS+=("--seed" "$SEED")

# Add log level
CMD_ARGS+=("--log-level" "$LOG_LEVEL")

echo "Configuration:"
echo "  Context sizes:  ${CONTEXT_SIZES[*]}"
echo "  N-needles:      $N_NEEDLES"
echo "  Samples/bin:    $SAMPLES_PER_BIN"
echo "  Seed:           $SEED"
echo "  Output dir:     $OUTPUT_DIR"
echo "  Config:         $CONFIG"
echo ""
echo "Running: ${CMD_ARGS[*]}"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: EXECUTE BENCHMARK WITH LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

LOG_FILE="${OUTPUT_DIR}/${RUNNER_ID}.log"
BENCHMARK_SUCCESS=false

echo "Starting benchmark execution..."
echo "Log file: $LOG_FILE"
echo ""

# Run the benchmark, capturing output to both stdout and log file
if "${CMD_ARGS[@]}" 2>&1 | tee "$LOG_FILE"; then
    BENCHMARK_SUCCESS=true
    echo ""
    echo "Benchmark execution completed successfully"
else
    BENCHMARK_SUCCESS=false
    echo ""
    echo "[WARNING] Benchmark execution reported errors"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: EXTRACT AND FORMAT RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "Processing results..."

# Find the most recent results file generated by run_multi_context.py
# It creates: {output_dir}/{run_id}_results.json (where run_id is timestamp-based)
RESULTS_FILE=""

# Look for the most recently created *_results.json file in output dir
if [ -d "$OUTPUT_DIR" ]; then
    RESULTS_FILE=$(find "$OUTPUT_DIR" -name "*_results.json" -type f -print0 2>/dev/null | \
        xargs -0 ls -t 2>/dev/null | head -n1)
fi

# Generate final results file at required location
FINAL_RESULTS="${OUTPUT_DIR}/${RUNNER_ID}_result.json"

if [ -n "$RESULTS_FILE" ] && [ -f "$RESULTS_FILE" ]; then
    echo "Found raw results: $RESULTS_FILE"
    
    # Copy results to final location with correct naming
    cp "$RESULTS_FILE" "$FINAL_RESULTS"
    echo "Copied to: $FINAL_RESULTS"
    
    # Extract metrics for display
    if command -v python3 &> /dev/null; then
        python3 -c "
import json
import sys

try:
    with open('$FINAL_RESULTS', 'r') as f:
        data = json.load(f)
    
    metrics = data.get('metrics', {})
    main = metrics.get('main', {})
    secondary = metrics.get('secondary', {})
    
    print('')
    print('Results Summary:')
    print('=' * 50)
    print(f\"Primary Metric: {main.get('name', 'N/A')} = {main.get('value', 'N/A')}\")
    print('')
    print('Secondary Metrics:')
    for k, v in secondary.items():
        if v is not None:
            print(f\"  {k}: {v}\")
    print('=' * 50)
except Exception as e:
    print(f'Error parsing results: {e}', file=sys.stderr)
    sys.exit(1)
"
    fi
else
    echo "[WARNING] No results file found. Creating fallback results."
    
    # Create fallback results with failure metrics
    python3 -c "
import json
import sys

data = {
    'metrics': {
        'main': {
            'name': 'average_accuracy',
            'value': 0
        },
        'secondary': {
            'accuracy_64k': 0,
            'accuracy_128k': 0,
            'accuracy_192k': 0,
            'exact_accuracy_64k': 0,
            'exact_accuracy_128k': 0,
            'exact_accuracy_192k': 0
        },
        'additional': {
            'run_id': '$RUNNER_ID',
            'status': 'failed',
            'error': 'Benchmark did not produce results file',
            'n_needles': $N_NEEDLES,
            'context_sizes': ${CONTEXT_SIZES[@]},
            'auto_bin': $AUTO_BIN,
            'samples_per_bin': $SAMPLES_PER_BIN,
            'seed': $SEED
        }
    }
}

with open('$FINAL_RESULTS', 'w') as f:
    json.dump(data, f, indent=2)

print(f'Created fallback results: $FINAL_RESULTS')
"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8: FINAL STATUS
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "=========================================="
if [ "$BENCHMARK_SUCCESS" = true ]; then
    echo "Evaluation Complete"
else
    echo "Evaluation Completed (with warnings/errors)"
fi
echo "=========================================="
echo "Run ID:     $RUNNER_ID"
echo "Output:     $FINAL_RESULTS"
echo "Log:        $LOG_FILE"
echo ""
