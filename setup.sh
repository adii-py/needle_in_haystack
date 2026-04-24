#!/bin/bash
# setup.sh — Setup script for MRCR evaluation on Debian 12 (GCP VM)
#
# Designed for: GCP Debian-based VMs (vanilla/minimal installation)
#
# This script:
#   1. Installs system dependencies (Python, git, build tools)
#   2. Creates Python virtual environment
#   3. Installs all Python dependencies
#   4. Verifies the installation
#
# Usage:
#   ./setup.sh                              # Local setup
#
# For GCP Cloud Init / Startup Script:
#   #!/bin/bash
#   curl -fsSL https://raw.githubusercontent.com/adii-py/needle_in_haystack/main/setup.sh | bash

set -euo pipefail

# ── Error handling ─────────────────────────────────────────────────────────────
trap 'echo "[ERROR] setup.sh failed at line $LINENO with exit code $?" >&2; exit 1' ERR

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION="3.11"

# Detect if running as root and conditionally use sudo
if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

echo "=========================================="
echo "MRCR Evaluation Setup (GCP/Vanilla Debian)"
echo "=========================================="
echo ""
echo "Project directory: $PROJECT_DIR"
echo "Running as root: $([ "$(id -u)" -eq 0 ] && echo "yes" || echo "no")"
echo ""

# ── Update package lists ─────────────────────────────────────────────────────
echo "[1/6] Updating package lists..."
$SUDO apt-get update -qq || {
    echo "[ERROR] Failed to update package lists"
    exit 1
}

# ── Install system dependencies ───────────────────────────────────────────────
echo "[2/6] Installing system dependencies..."
$SUDO apt-get install -y -qq \
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
        echo "[ERROR] Failed to install system dependencies"
        exit 1
    }

# Verify Python installation
echo ""
echo "Python version: $(python3 --version)"

# Check Python version compatibility
PYTHON_MAJOR=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1)
PYTHON_MINOR=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f2)
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
    echo "[WARNING] Python 3.11+ recommended. Found: $(python3 --version)"
fi

# ── Create virtual environment ────────────────────────────────────────────────
echo ""
echo "[3/6] Creating Python virtual environment..."
cd "$PROJECT_DIR"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created at ./venv"
else
    echo "Virtual environment already exists (skipping creation)"
fi

# ── Activate virtual environment ──────────────────────────────────────────────
echo ""
echo "[4/6] Activating virtual environment..."
source venv/bin/activate

# ── Upgrade pip and install build tools ───────────────────────────────────────
echo ""
echo "[5/6] Upgrading pip and installing build tools..."
venv/bin/pip install --quiet --upgrade pip setuptools wheel || {
    echo "[ERROR] Failed to upgrade pip/setuptools/wheel"
    exit 1
}

# ── Install Python dependencies ───────────────────────────────────────────────
echo ""
echo "[6/6] Installing Python dependencies..."
echo "This may take 5-10 minutes..."
echo ""

# Install all dependencies in one go for better dependency resolution
venv/bin/pip install --quiet \
    pyyaml \
    python-dotenv \
    tiktoken \
    datasets \
    huggingface-hub \
    tqdm \
    opencompass || {
        echo "[WARNING] Some packages may have failed. Retrying with individual installs..."
        # Retry individually if bulk install fails
        venv/bin/pip install --quiet pyyaml || echo "  ✗ pyyaml failed"
        venv/bin/pip install --quiet python-dotenv || echo "  ✗ python-dotenv failed"
        venv/bin/pip install --quiet tiktoken || echo "  ✗ tiktoken failed"
        venv/bin/pip install --quiet datasets || echo "  ✗ datasets failed"
        venv/bin/pip install --quiet huggingface-hub || echo "  ✗ huggingface-hub failed"
        venv/bin/pip install --quiet tqdm || echo "  ✗ tqdm failed"
        venv/bin/pip install --quiet opencompass || echo "  ✗ opencompass failed"
    }

# Verify critical packages using venv python
echo ""
echo "Verifying installations..."
VERIFY_FAILED=0

venv/bin/python -c "import yaml" 2>/dev/null && echo "  ✓ PyYAML" || { 
    echo "  ✗ PyYAML FAILED - attempting fix..."; 
    venv/bin/pip install --force-reinstall pyyaml || VERIFY_FAILED=1;
}
venv/bin/python -c "import dotenv" 2>/dev/null && echo "  ✓ python-dotenv" || { 
    echo "  ✗ python-dotenv FAILED - attempting fix..."; 
    venv/bin/pip install --force-reinstall python-dotenv || VERIFY_FAILED=1;
}
venv/bin/python -c "import tiktoken" 2>/dev/null && echo "  ✓ tiktoken" || { 
    echo "  ✗ tiktoken FAILED - attempting fix..."; 
    venv/bin/pip install --force-reinstall tiktoken || VERIFY_FAILED=1;
}
venv/bin/python -c "import datasets" 2>/dev/null && echo "  ✓ datasets" || { 
    echo "  ✗ datasets FAILED - attempting fix..."; 
    venv/bin/pip install --force-reinstall datasets || VERIFY_FAILED=1;
}
venv/bin/python -c "import opencompass" 2>/dev/null && echo "  ✓ opencompass" || { 
    echo "  ✗ opencompass FAILED - attempting fix..."; 
    venv/bin/pip install --force-reinstall opencompass || VERIFY_FAILED=1;
}

# Final verification - fail if any critical package is missing
echo ""
echo "Final verification..."
venv/bin/python -c "import yaml" 2>/dev/null || { echo "[ERROR] PyYAML not installed"; VERIFY_FAILED=1; }
venv/bin/python -c "import dotenv" 2>/dev/null || { echo "[ERROR] python-dotenv not installed"; VERIFY_FAILED=1; }
venv/bin/python -c "import tiktoken" 2>/dev/null || { echo "[ERROR] tiktoken not installed"; VERIFY_FAILED=1; }
venv/bin/python -c "import datasets" 2>/dev/null || { echo "[ERROR] datasets not installed"; VERIFY_FAILED=1; }

if [ $VERIFY_FAILED -eq 1 ]; then
    echo ""
    echo "[ERROR] Critical Python packages failed to install. Please check the logs above."
    exit 1
fi

echo "All critical packages verified successfully!"

# ── Environment file setup ────────────────────────────────────────────────────
echo ""
echo "Setting up environment file..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# MRCR Evaluation Environment Variables
# These are now set via run.sh command-line arguments

# Optional settings
export LOG_LEVEL="INFO"
EOF
    echo "  Created .env file"
else
    echo "  .env file already exists (skipping)"
fi

# ── Make scripts executable ───────────────────────────────────────────────────
chmod +x run.sh 2>/dev/null || true

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
echo "1. RUN THE EVALUATION:"
echo "   ./run.sh <api_key> <run_id> [options]"
echo ""
echo "   Example:"
echo "   ./run.sh sk-xxx 550e8400-e29b-41d4-a716-446655440000 --n-needles 2 --context-sizes 64000 128000 192000 --samples-per-bin 100 --auto-bin"
echo ""
echo "For help:"
echo "   ./run.sh --help"
echo ""
