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
#   - At least one HTTP port exposed (8080 recommended — it's open by default)
#
# Disk space notes:
#   The root disk on VAST.ai is often small. This script stores all heavy
#   data (model weights, pip cache) under /workspace/ which is the larger
#   persistent volume.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== OCR GPU Worker Deployment ==="
echo ""

# --- 0. Disk space check ---
echo "[0/7] Checking disk space..."
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

# --- 1. Detect available port ---
echo "[1/7] Detecting available port..."
WORKER_PORT=""

# Check which ports VAST.ai has exposed, prefer 8080 (default in most templates)
for try_port in 8080 8000 6006 1111; do
    var_name="VAST_TCP_PORT_${try_port}"
    ext_port="${!var_name:-}"
    if [ -n "$ext_port" ]; then
        WORKER_PORT="$try_port"
        EXTERNAL_PORT="$ext_port"
        echo "  Found exposed port: internal $WORKER_PORT -> external $EXTERNAL_PORT"
        break
    fi
done

if [ -z "$WORKER_PORT" ]; then
    echo "  No VAST.ai port mapping detected — defaulting to 8080"
    WORKER_PORT="8080"
    EXTERNAL_PORT="unknown"
fi

# Get public IP
PUBLIC_IP=$(curl -s --connect-timeout 5 https://api.ipify.org 2>/dev/null || echo "unknown")
echo "  Public IP: $PUBLIC_IP"
if [ "$EXTERNAL_PORT" != "unknown" ]; then
    echo "  Worker will be reachable at: http://$PUBLIC_IP:$EXTERNAL_PORT"
fi

# --- 2. Point heavy caches at /workspace ---
echo "[2/7] Configuring cache directories on /workspace..."
export HF_HOME="/workspace/.cache/huggingface"
export TRANSFORMERS_CACHE="/workspace/.cache/huggingface/hub"
export PIP_CACHE_DIR="/workspace/.cache/pip"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$PIP_CACHE_DIR"
echo "  HF_HOME=$HF_HOME"
echo "  PIP_CACHE_DIR=$PIP_CACHE_DIR"

# --- 3. Install system deps if needed ---
echo "[3/7] Checking system dependencies..."
if ! command -v pip &>/dev/null; then
    apt-get update && apt-get install -y python3-pip
fi

# --- 4. Install ocrdoctotext ---
echo "[4/7] Installing ocrdoctotext..."
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

# --- 5. Install Python deps ---
echo "[5/7] Installing Python dependencies..."
pip install -r requirements.txt

# --- 6. Set up .env ---
echo "[6/7] Configuring .env..."
if [ ! -f .env ]; then
    cp .env.example .env
    # Auto-generate secret
    secret=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    sed -i "s|^WORKER_SECRET=.*|WORKER_SECRET=$secret|" .env
    echo "  Generated WORKER_SECRET: $secret"
    echo "  (save this — you'll need it for WORKER_API_KEY on the control node)"
else
    echo "  .env already exists, keeping current config"
    secret=$(grep '^WORKER_SECRET=' .env | cut -d= -f2)
fi

# Set the detected port
sed -i "s|^PORT=.*|PORT=$WORKER_PORT|" .env 2>/dev/null || echo "PORT=$WORKER_PORT" >> .env
echo "  PORT=$WORKER_PORT"

# --- 7. Pre-download model weights to /workspace ---
echo "[7/7] Ensuring model weights are cached..."
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
echo "========================================"
echo "  Deployment complete!"
echo "========================================"
echo ""
echo "Disk usage:"
du -sh "$HF_HOME" 2>/dev/null | awk '{print "  Model cache: " $1}'
echo ""
echo "Start the worker:"
echo "  cd $SCRIPT_DIR && python3 -m gpu_worker.main"
echo ""
echo "Or in the background:"
echo "  cd $SCRIPT_DIR && nohup python3 -m gpu_worker.main > worker.log 2>&1 &"
echo ""
if [ "$EXTERNAL_PORT" != "unknown" ]; then
    echo "Worker URL (for control node .env):"
    echo "  WORKER_URL=http://$PUBLIC_IP:$EXTERNAL_PORT"
    echo "  WORKER_API_KEY=$secret"
    echo ""
    echo "Health check from your home machine:"
    echo "  curl http://$PUBLIC_IP:$EXTERNAL_PORT/health"
else
    echo "Test health locally:"
    echo "  curl http://localhost:$WORKER_PORT/health"
fi
