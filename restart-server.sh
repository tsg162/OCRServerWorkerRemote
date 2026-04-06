#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PORT=$(grep '^PORT=' .env 2>/dev/null | cut -d= -f2 || echo 5001)
TUNNEL_TOKEN=$(grep '^OCRHARBOR_TUNNEL_TOKEN=' .env 2>/dev/null | cut -d= -f2 || true)
LOG="$(pwd)/worker.log"
TUNNEL_LOG="$(pwd)/tunnel.log"

green()  { echo -e "\033[32m$*\033[0m"; }
yellow() { echo -e "\033[33m$*\033[0m"; }
red()    { echo -e "\033[31m$*\033[0m"; }

# ---------- 1. Kill old worker ----------
echo "=== Restarting OCR Worker ==="
echo ""
echo "[1/5] Stopping old worker..."

# Kill by port
if fuser "$PORT/tcp" 2>/dev/null; then
    fuser -k "$PORT/tcp" 2>/dev/null || true
    echo "  Sent kill signal to process on port $PORT"
else
    echo "  No process found on port $PORT"
fi

# Also kill by name in case port wasn't bound yet
pkill -f "python.*ocrharbor_worker" 2>/dev/null && echo "  Killed ocrharbor_worker process" || true

# Wait for port to actually free up
for i in $(seq 1 10); do
    if ! fuser "$PORT/tcp" 2>/dev/null; then
        break
    fi
    if [ "$i" -eq 10 ]; then
        red "  ERROR: Port $PORT still in use after 10s — force killing"
        fuser -9 "$PORT/tcp" 2>/dev/null || true
        sleep 1
    fi
    sleep 1
done
green "  Port $PORT is free"

# ---------- 2. Kill and restart tunnel ----------
echo ""
echo "[2/5] Restarting Cloudflare tunnel..."

if [ -z "$TUNNEL_TOKEN" ]; then
    yellow "  No OCRHARBOR_TUNNEL_TOKEN in .env — skipping tunnel"
else
    # Kill old tunnel
    pkill -f "cloudflared.*tunnel.*run" 2>/dev/null && echo "  Killed old tunnel process" || echo "  No old tunnel process found"
    sleep 1

    # Verify old tunnel is dead
    if pgrep -f "cloudflared.*tunnel.*run" >/dev/null 2>&1; then
        red "  Old tunnel still alive — force killing"
        pkill -9 -f "cloudflared.*tunnel.*run" 2>/dev/null || true
        sleep 1
    fi

    # Start new tunnel
    nohup cloudflared tunnel run --token "$TUNNEL_TOKEN" > "$TUNNEL_LOG" 2>&1 &
    TUNNEL_PID=$!
    echo "$TUNNEL_PID" > tunnel.pid
    echo "  Tunnel PID: $TUNNEL_PID"

    # Wait for tunnel to register with Cloudflare
    echo "  Waiting for tunnel to connect..."
    TUNNEL_READY=false
    for i in $(seq 1 30); do
        if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
            red "  ERROR: Tunnel process died. Last log lines:"
            tail -10 "$TUNNEL_LOG"
            exit 1
        fi
        if grep -q "Registered tunnel connection\|Connection.*registered\|INF " "$TUNNEL_LOG" 2>/dev/null; then
            TUNNEL_READY=true
            break
        fi
        sleep 1
    done

    if [ "$TUNNEL_READY" = true ]; then
        green "  Tunnel connected (took ~${i}s)"
    else
        yellow "  Tunnel may still be connecting (30s timeout) — continuing anyway"
        tail -3 "$TUNNEL_LOG"
    fi
fi

# ---------- 3. Start worker ----------
echo ""
echo "[3/5] Starting worker on port $PORT..."

nohup python3 -m ocrharbor_worker.main > "$LOG" 2>&1 &
WORKER_PID=$!
echo "  Worker PID: $WORKER_PID"

# ---------- 4. Wait for worker to load model and be ready ----------
echo ""
echo "[4/5] Waiting for worker to be ready..."

for i in $(seq 1 120); do
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then
        red "  ERROR: Worker process died after ${i}s. Last log lines:"
        tail -20 "$LOG"
        exit 1
    fi
    if grep -q "ready to accept jobs\|Uvicorn running on\|Application startup complete" "$LOG" 2>/dev/null; then
        green "  Worker is up (took ~${i}s)"
        break
    fi
    if [ "$i" -eq 120 ]; then
        yellow "  Timed out after 120s. Worker may still be loading the model."
        echo "  Check: tail -f $LOG"
        exit 1
    fi
    sleep 1
done

# ---------- 5. Verify health endpoint ----------
echo ""
echo "[5/5] Verifying health endpoint..."

HEALTH_OK=false
for i in $(seq 1 5); do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/health" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        HEALTH_OK=true
        break
    fi
    sleep 1
done

if [ "$HEALTH_OK" = true ]; then
    green "  Local health check passed (HTTP 200)"
else
    red "  WARNING: Local health check failed (HTTP $HTTP_CODE)"
    echo "  Worker may not be responding correctly. Check: tail -f $LOG"
fi

# ---------- Summary ----------
echo ""
echo "========================================"
green "  Restart complete!"
echo "========================================"
echo ""
echo "  Worker PID:  $WORKER_PID"
echo "  Worker log:  tail -f $LOG"
if [ -n "$TUNNEL_TOKEN" ]; then
    echo "  Tunnel PID:  $TUNNEL_PID"
    echo "  Tunnel log:  tail -f $TUNNEL_LOG"
fi
echo "  Stop:        kill $WORKER_PID"
echo ""
