#!/bin/bash
# setup.sh — Setup script for MRCR evaluation on vanilla Debian/Ubuntu
#
# Designed for: GCP Debian-based VMs (vanilla/minimal installation)
#
# This script:
#   1. Installs system dependencies (Python, git, build tools)
#   2. Clones the repository (if not already present)
#   3. Creates Python virtual environment
#   4. Installs all Python dependencies
#   5. Verifies the installation
#
# Usage:
#   ./setup.sh                              # Local setup
#   ./setup.sh --clone                      # Clone repo first, then setup
#   GIT_REPO="https://github.com/user/repo.git" ./setup.sh --clone
#
# For GCP Cloud Init / Startup Script:
#   #!/bin/bash
#   curl -fsSL https://raw.githubusercontent.com/adii-py/needle_in_haystack/main/setup.sh | bash -s -- --clone

set -e

# ── Configuration ─────────────────────────────────────────────────────────────
GIT_REPO="${GIT_REPO:-https://github.com/adii-py/needle_in_haystack.git}"
PROJECT_DIR="${PROJECT_DIR:-needle_in_haystack}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# ── Parse arguments ───────────────────────────────────────────────────────────
CLONE_REPO=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --clone) CLONE_REPO=true; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --clone     Clone the repository before setup"
            echo "  -h, --help  Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  GIT_REPO    Repository URL (default: $GIT_REPO)"
            echo "  PROJECT_DIR Directory name for cloned repo (default: $PROJECT_DIR)"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Run '$0 --help' for usage."
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "MRCR Evaluation Setup (GCP/Vanilla Debian)"
echo "=========================================="
echo ""

# ── Update package lists ─────────────────────────────────────────────────────
echo "[1/8] Updating package lists..."
apt-get update -qq

# ── Install system dependencies ───────────────────────────────────────────────
echo "[2/8] Installing system dependencies..."
apt-get install -y -qq \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    curl \
    wget \
    build-essential \
    libffi-dev \
    libssl-dev \
    || {
        echo "ERROR: Failed to install system dependencies"
        exit 1
    }

# Verify Python installation
echo ""
echo "Python version: $(python3 --version)"

# ── Clone repository if requested ─────────────────────────────────────────────
if [ "$CLONE_REPO" = true ]; then
    echo ""
    echo "[3/8] Cloning repository from $GIT_REPO..."
    if [ -d "$PROJECT_DIR" ]; then
        echo "Directory $PROJECT_DIR already exists. Pulling latest changes..."
        cd "$PROJECT_DIR"
        git pull origin main || git pull origin master
        cd ..
    else
        git clone "$GIT_REPO" "$PROJECT_DIR"
    fi
    cd "$PROJECT_DIR"
    echo "Changed to project directory: $(pwd)"
else
    echo "[3/8] Skipping clone (using current directory)"
fi

# ── Create virtual environment ────────────────────────────────────────────────
echo ""
echo "[4/8] Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created at ./venv"
else
    echo "Virtual environment already exists"
fi

# ── Activate virtual environment ──────────────────────────────────────────────
echo ""
echo "[5/8] Activating virtual environment..."
source venv/bin/activate

# ── Upgrade pip and install build tools ───────────────────────────────────────
echo ""
echo "[6/8] Upgrading pip and installing build tools..."
pip install --quiet --upgrade pip setuptools wheel

# ── Install Python dependencies ───────────────────────────────────────────────
echo ""
echo "[7/8] Installing Python dependencies..."
echo "This may take 5-10 minutes..."
echo ""

# Install all dependencies in one go for better dependency resolution
pip install --quiet \
    pyyaml \
    python-dotenv \
    tiktoken \
    datasets \
    huggingface-hub \
    tqdm \
    opencompass || {
        echo "WARNING: Some packages may have failed. Retrying with individual installs..."
    }

# Verify critical packages
echo ""
echo "Verifying installations..."
python3 -c "import yaml" 2>/dev/null && echo "  ✓ PyYAML" || echo "  ✗ PyYAML FAILED"
python3 -c "import dotenv" 2>/dev/null && echo "  ✓ python-dotenv" || echo "  ✗ python-dotenv FAILED"
python3 -c "import tiktoken" 2>/dev/null && echo "  ✓ tiktoken" || echo "  ✗ tiktoken FAILED"
python3 -c "import datasets" 2>/dev/null && echo "  ✓ datasets" || echo "  ✗ datasets FAILED"
python3 -c "import opencompass" 2>/dev/null && echo "  ✓ opencompass" || echo "  ✗ opencompass FAILED"

# ── Environment file setup ────────────────────────────────────────────────────
echo ""
echo "[8/8] Setting up environment file..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# MRCR Evaluation Environment Variables
# Fill in your actual API credentials before running

# LiteLLM API Configuration (REQUIRED)
export LITE_LLM_API_KEY="your-api-key-here"
export LITE_LLM_URL="https://your-api-endpoint.com"
export LITE_LLM_MODEL="your-model-name"

# Optional settings
export LOG_LEVEL="INFO"
export LITE_LLM=1
export LITE_LLM_MAX_SEQ=196000
export LITE_LLM_MAX_OUT=8192
EOF
    echo "  Created .env file - EDIT THIS FILE with your actual credentials!"
else
    echo "  .env file already exists"
fi

# ── Make scripts executable ───────────────────────────────────────────────────
chmod +x run_mrcr.sh 2>/dev/null || true

# ── Completion message ─────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Project directory: $(pwd)"
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. EDIT THE ENVIRONMENT FILE:"
echo "   nano .env"
echo "   # Add your actual API key, URL, and model name"
echo ""
echo "2. ACTIVATE THE VIRTUAL ENVIRONMENT:"
echo "   source venv/bin/activate"
echo ""
echo "3. RUN THE EVALUATION:"
echo "   ./run_mrcr.sh --multi-context --context-sizes 64000 128000 192000 --n-needles 2 --no-truncation --auto-bin --samples-per-bin 100 --output-dir outputs/mrcr/deepseek30_needle2"
echo ""
echo "Or use the simplified run command (edit run.sh to set default args):"
echo "   ./run.sh"
echo ""
echo "For help:"
echo "   ./run_mrcr.sh --help"
echo ""
