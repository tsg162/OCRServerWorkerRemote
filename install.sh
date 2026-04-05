#!/usr/bin/env bash
#
# install.sh — Install and configure OCR GPU Worker on a remote instance.
#
# Usage:
#   bash install.sh [--port PORT]
#
# Default port: 5001

set -euo pipefail

PORT=5001

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash install.sh [--port PORT]"
            echo "Default port: 5001"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== OCR GPU Worker Installation ==="
echo "  Port: $PORT"
echo ""

# --- 1. Configure cache directories ---
echo "[1/5] Configuring cache directories..."
if [ -d /workspace ]; then
    CACHE_DIR="/workspace/.cache"
    echo "  Using /workspace (VAST.ai detected)"
else
    CACHE_DIR="$SCRIPT_DIR/.cache"
    echo "  Using local cache: $CACHE_DIR"
fi
export HF_HOME="$CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$PIP_CACHE_DIR"

# --- 2. Detect Python environment ---
echo "[2/5] Detecting Python environment..."
PIP="pip"
PYTHON="python3"

if [ -x /venv/main/bin/pip ]; then
    PIP="/venv/main/bin/pip"
    PYTHON="/venv/main/bin/python"
    echo "  Using VAST.ai venv: /venv/main/"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/pip" ]; then
    PIP="$VIRTUAL_ENV/bin/pip"
    PYTHON="$VIRTUAL_ENV/bin/python"
    echo "  Using active venv: $VIRTUAL_ENV"
else
    echo "  Using system Python"
    if ! command -v pip &>/dev/null; then
        apt-get update -qq && apt-get install -y -qq python3-pip
    fi
fi

# --- 3. Install dependencies ---
echo "[3/5] Installing dependencies..."
if ! $PYTHON -c "import ocrdoctotext" 2>/dev/null; then
    if [ -d "$SCRIPT_DIR/ocrdoctotext_pkg" ]; then
        $PIP install -q "$SCRIPT_DIR/ocrdoctotext_pkg/"
    elif [ -d "/workspace/OCRDocToText" ]; then
        $PIP install -q /workspace/OCRDocToText/
    else
        echo "  ERROR: ocrdoctotext package not found."
        echo "  Copy OCRDocToText to $SCRIPT_DIR/ocrdoctotext_pkg/ or /workspace/OCRDocToText/"
        exit 1
    fi
else
    echo "  ocrdoctotext already installed"
fi
$PIP install -q -r requirements.txt
echo "  Done"

# --- 4. Generate configuration ---
echo "[4/5] Generating configuration..."
SECRET=$($PYTHON -c "import secrets; print(secrets.token_urlsafe(32))")
cat > .env <<EOF
PORT=$PORT
WORKER_SECRET=$SECRET
HF_HOME=$HF_HOME
MAX_QUEUE_SIZE=500
JOB_TTL_SECONDS=3600
EOF
echo "  .env written"

# --- 5. Download model weights ---
echo "[5/5] Downloading model weights (may take a few minutes on first run)..."
$PYTHON -c "
import os
os.environ['HF_HOME'] = '$HF_HOME'
os.environ['TRANSFORMERS_CACHE'] = '$TRANSFORMERS_CACHE'
from ocrdoctotext import OCREngine
engine = OCREngine('lightonai/LightOnOCR-2-1B')
engine.load()
print('  Model loaded successfully')
"

echo ""
echo "========================================"
echo "  Installation complete!"
echo "========================================"
echo ""
echo "  WORKER_SECRET: $SECRET"
echo "  PORT:          $PORT"
echo ""
echo "Start the worker:"
echo "  cd $SCRIPT_DIR"
echo "  $PYTHON -m gpu_worker.main"
echo ""
echo "Or in background:"
echo "  nohup $PYTHON -m gpu_worker.main > worker.log 2>&1 &"
echo ""
echo "Then on your control node, add this worker:"
echo "  ./worker-ctl add <name> <tunnel-url> --key $SECRET"
