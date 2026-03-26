# OCR GPU Worker

Remote GPU OCR worker for the OCRServer system. Runs on a VAST.ai GPU instance, accepts image jobs via API, runs LightOnOCR-2-1B inference, and reports results back to the control node via polling.

## Quick Start on VAST.ai

### 1. Create Instance

- **Template: PyTorch (Vast)** (comes with port 8080 exposed by default)
- GPU: 4+ GB VRAM (1B model in bfloat16 is ~2GB)
- No extra port config needed — the deploy script auto-detects an available exposed port

### 2. Clone and Deploy

```bash
ssh root@<vast-ip>
cd /workspace
git clone https://github.com/tsg162/OCRServerWorkerRemote.git
cd OCRServerWorkerRemote
bash deploy.sh
```

The deploy script will:
- Check available disk space
- **Auto-detect an exposed port** (prefers 8080, which PyTorch template exposes by default)
- Redirect all heavy data (model weights, pip cache) to `/workspace/` (not the small root disk)
- Install `ocrdoctotext` (bundled) and Python dependencies
- **Auto-generate `WORKER_SECRET`** (no prompts — just prints the values you need)
- Pre-download model weights to `/workspace/.cache/huggingface/` (~4GB, first run only)
- **Print the exact `WORKER_URL` and `WORKER_API_KEY`** to paste into your control node `.env`

### 3. Start

```bash
python3 -m gpu_worker.main
```

On startup, the worker logs its public IP and URL:
```
Public IP: 209.20.157.9
Worker URL: http://209.20.157.9:10300
Health check: curl http://209.20.157.9:10300/health
Model loaded — ready to accept jobs
```

### 4. Configure Control Node

Copy the values printed by `deploy.sh` into your control node's `control/.env`:
```
WORKER_URL=http://209.20.157.9:10300
WORKER_API_KEY=<the-secret-from-deploy>
```

## API

All endpoints require `Authorization: Bearer <WORKER_SECRET>`.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/jobs` | Submit OCR job (multipart: file + form fields). Returns 202. |
| GET | `/jobs/{id}` | Get job status and result |
| DELETE | `/jobs/{id}` | Cancel a queued job |
| GET | `/jobs` | List all active jobs |
| GET | `/health` | GPU status, model loaded, queue depth |

### Submit a job

```bash
curl -X POST http://localhost:8000/jobs \
  -H "Authorization: Bearer $WORKER_SECRET" \
  -F "file=@image.png" \
  -F "job_id=test123" \
  -F "callback_url="
```

### Check result

```bash
curl http://localhost:8000/jobs/test123 \
  -H "Authorization: Bearer $WORKER_SECRET"
```

## Configuration

Edit `.env` (created by deploy.sh):

| Variable | Required | Description |
|----------|----------|-------------|
| `WORKER_SECRET` | Yes | Shared secret for API auth |
| `CALLBACK_URL` | No | Control node webhook URL (polling works without this) |
| `CALLBACK_SECRET` | No | Key sent with webhook callbacks |
| `OCR_MODEL` | No | HuggingFace model ID (default: `lightonai/LightOnOCR-2-1B`) |
| `HF_HOME` | No | HuggingFace cache dir (default: `/workspace/.cache/huggingface`) |
| `MAX_QUEUE_SIZE` | No | Max queued jobs (default: 100) |

## What's Included

```
ocrdoctotext_pkg/    # Bundled OCR engine library (pip-installable)
gpu_worker/          # FastAPI worker application
  main.py            # App entry point, endpoints, lifespan
  config.py          # pydantic-settings
  auth.py            # Bearer token verification
  models.py          # Pydantic response schemas
  job_manager.py     # In-memory job queue + GPU runner
  ocr_bridge.py      # OCREngine singleton
  webhook.py         # Callback sender (optional)
deploy.sh            # One-command setup script
requirements.txt     # Python dependencies
```
