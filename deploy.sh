#!/usr/bin/env bash
#
# deploy.sh — Bootstrap the OCR GPU Worker on a fresh VAST.ai PyTorch instance.
#
# Usage:
#   1. SSH into the VAST.ai instance
#   2. Clone the repo into /workspace/
#   3. Run: bash deploy.sh
#
# Prerequisites:
#   - VAST.ai instance using the "PyTorch (Vast)" template
#   - Port 8000 exposed in the instance config
#
# Disk space notes:
#   The root disk on VAST.ai is often small. This script stores all heavy
#   data (model weights, pip cache, venv) under /workspace/ which is the
#   larger persistent volume.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== OCR GPU Worker Deployment ==="
echo ""

# --- 0. Disk space check ---
echo "[0/6] Checking disk space..."
if [ -d /workspace ]; then
    workspace_avail=$(df -BG /workspace 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G')
    root_avail=$(df -BG / 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G')
    echo "  /workspace: ${workspace_avail}G available"
    echo "  /root:      ${root_avail}G available"
    if [ "$workspace_avail" -lt 8 ] 2>/dev/null; then
        echo "  WARNING: Less than 8GB free on /workspace. Model weights alone are ~4GB."
        read -rp "  Continue anyway? [y/N] " confirm
        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            exit 1
        fi
    fi
else
    echo "  /workspace not found — running outside VAST.ai? Continuing anyway."
fi

# --- 1. Point heavy caches at /workspace ---
echo "[1/6] Configuring cache directories on /workspace..."
export HF_HOME="/workspace/.cache/huggingface"
export TRANSFORMERS_CACHE="/workspace/.cache/huggingface/hub"
export PIP_CACHE_DIR="/workspace/.cache/pip"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$PIP_CACHE_DIR"
echo "  HF_HOME=$HF_HOME"
echo "  PIP_CACHE_DIR=$PIP_CACHE_DIR"

# Write these to a sourceable file so gpu_worker.main picks them up
cat > "$SCRIPT_DIR/.cache_env" <<ENVEOF
export HF_HOME="$HF_HOME"
export TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export PIP_CACHE_DIR="$PIP_CACHE_DIR"
ENVEOF

# --- 2. Install system deps if needed ---
echo "[2/6] Checking system dependencies..."
if ! command -v pip &>/dev/null; then
    apt-get update && apt-get install -y python3-pip
fi

# --- 3. Install ocrdoctotext ---
echo "[3/6] Installing ocrdoctotext..."
if python3 -c "import ocrdoctotext" 2>/dev/null; then
    echo "  ocrdoctotext already installed"
else
    if [ -d "$SCRIPT_DIR/ocrdoctotext_pkg" ]; then
        pip install "$SCRIPT_DIR/ocrdoctotext_pkg/"
    elif [ -d "/workspace/OCRDocToText" ]; then
        pip install /workspace/OCRDocToText/
    else
        echo "  ERROR: ocrdoctotext not found."
        echo "  Copy the OCRDocToText project to $SCRIPT_DIR/ocrdoctotext_pkg/ or /workspace/OCRDocToText/"
        exit 1
    fi
fi

# --- 4. Install Python deps ---
echo "[4/6] Installing Python dependencies..."
pip install -r requirements.txt

# --- 5. Set up .env if missing ---
echo "[5/6] Checking .env configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  Created .env from .env.example — edit it now:"
    echo "    WORKER_SECRET  (required — shared secret for auth)"
    echo "    CALLBACK_URL   (control node webhook URL, optional)"
    echo "    CALLBACK_SECRET (key for webhook auth, optional)"
    echo ""
    read -rp "  Enter WORKER_SECRET (or press Enter to generate one): " secret
    if [ -z "$secret" ]; then
        secret=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
        echo "  Generated secret: $secret"
    fi
    sed -i "s|^WORKER_SECRET=.*|WORKER_SECRET=$secret|" .env

    read -rp "  Enter CALLBACK_URL (or press Enter to skip): " callback_url
    if [ -n "$callback_url" ]; then
        sed -i "s|^CALLBACK_URL=.*|CALLBACK_URL=$callback_url|" .env
    fi

    read -rp "  Enter CALLBACK_SECRET (or press Enter to skip): " callback_secret
    if [ -n "$callback_secret" ]; then
        sed -i "s|^CALLBACK_SECRET=.*|CALLBACK_SECRET=$callback_secret|" .env
    fi
fi

# --- 6. Pre-download model weights to /workspace ---
echo "[6/6] Ensuring model weights are cached..."
echo "  Downloading to $HF_HOME (this may take a few minutes on first run)..."
python3 -c "
import os
os.environ['HF_HOME'] = '$HF_HOME'
os.environ['TRANSFORMERS_CACHE'] = '$TRANSFORMERS_CACHE'
from ocrdoctotext import OCREngine
engine = OCREngine('lightonai/LightOnOCR-2-1B')
engine.load()
print('  Model loaded successfully')
"

echo ""
echo "=== Deployment complete ==="
echo ""
echo "Disk usage:"
du -sh "$HF_HOME" 2>/dev/null | awk '{print "  Model cache: " $1}'
echo ""
echo "Start the worker with:"
echo "  cd $SCRIPT_DIR && source .cache_env && python3 -m gpu_worker.main"
echo ""
echo "Or run in the background:"
echo "  cd $SCRIPT_DIR && source .cache_env && nohup python3 -m gpu_worker.main > worker.log 2>&1 &"
echo ""
echo "Test health:"
echo "  curl http://localhost:8000/health"
