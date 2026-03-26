# OCR GPU Worker

Remote GPU OCR worker for the OCRServer system. Runs on a VAST.ai GPU instance, accepts image jobs via API, runs LightOnOCR-2-1B inference, and reports results back to the control node via polling.

## Quick Start on VAST.ai

### 1. Create Instance

- **Template: PyTorch (Vast)**
- GPU: 4+ GB VRAM (1B model in bfloat16 is ~2GB)
- **Expose port 8000**

### 2. Clone and Deploy

```bash
ssh root@<vast-ip>
cd /workspace
git clone https://github.com/<your-user>/ocr-gpu-worker.git
cd ocr-gpu-worker
bash deploy.sh
```

The deploy script will:
- Check available disk space
- Redirect all heavy data (model weights, pip cache) to `/workspace/` (not the small root disk)
- Install `ocrdoctotext` (bundled in this repo)
- Install Python dependencies
- Prompt for `WORKER_SECRET` (generates one if you press Enter)
- Pre-download and cache model weights to `/workspace/.cache/huggingface/` (~4GB, first run only)

### 3. Start

```bash
# Foreground:
python3 -m gpu_worker.main

# Background:
nohup python3 -m gpu_worker.main > worker.log 2>&1 &
```

Note: The worker automatically sets `HF_HOME=/workspace/.cache/huggingface` so model
weights are always loaded from the workspace volume, not the root disk.

### 4. Verify

```bash
curl http://localhost:8000/health
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
