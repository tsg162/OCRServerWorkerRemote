#!/usr/bin/env python3
"""
GPU Benchmark for OCRHarborWorker
==================================
Two modes:

  STANDALONE  — Submits its own synthetic images. Use when the worker is idle
                (no OCRHarbor feeding it). Good for quick isolated testing.

  LIVE        — Piggybacks on real OCRHarbor work. Changes batch_size on the
                worker, observes throughput over time windows, then moves to
                the next setting. Real pages get processed the whole time.
                Use --live to enable.

Saves JSON reports that can be compared across GPUs with --compare.

Usage:
    # Standalone (worker idle, no OCRHarbor)
    python benchmark.py --gpu-cost 0.22 --gpu-name "RTX 3090"

    # Live sweep while OCRHarbor feeds real work
    python benchmark.py --live --gpu-cost 0.22 --gpu-name "RTX 3090"

    # Live with custom window (8 min per batch_size, more accurate)
    python benchmark.py --live --window 480 --gpu-cost 0.22

    # Quick standalone sanity check
    python benchmark.py --pages 20 --batch-sizes 1,4,8

    # Compare reports from different GPUs
    python benchmark.py --compare benchmark_rtx_3090_*.json benchmark_rtx_4090_*.json
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("httpx required: pip install httpx")

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("Pillow required: pip install Pillow")


# ---------------------------------------------------------------------------
# Test image generation  (standalone mode only)
# ---------------------------------------------------------------------------

SAMPLE_TEXT = """
The quick brown fox jumps over the lazy dog. Pack my box with five dozen
liquor jugs. How vexingly quick daft zebras jump! The five boxing wizards
jump quickly. Sphinx of black quartz, judge my vow.

Section 2: Technical Content
-----------------------------
f(x) = ax² + bx + c where a ≠ 0
∫₀^∞ e^(-x²) dx = √π/2
∑(n=1 to ∞) 1/n² = π²/6

Table 1: Sample Data
| ID  | Value  | Status    |
|-----|--------|-----------|
| 001 | 42.5   | Active    |
| 002 | 18.3   | Pending   |
| 003 | 99.1   | Complete  |

This paragraph contains typical book/document text that an OCR system would
need to process. It includes mixed formatting, numbers (12345), dates
(2025-01-15), and special characters (@#$%&*).
""".strip()


def generate_test_image(page_num: int = 1) -> bytes:
    """Create a realistic document-page image (A4-ish, ~200 DPI)."""
    width, height = 1700, 2200
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    font = None
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, 24)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    draw.text((80, 60), f"Benchmark Test Page {page_num}", fill="black", font=font)
    draw.line([(80, 100), (width - 80, 100)], fill="gray", width=2)

    y = 130
    for line in SAMPLE_TEXT.split("\n"):
        draw.text((80, y), line, fill="black", font=font)
        y += 32

    draw.line([(80, height - 100), (width - 80, height - 100)], fill="gray", width=2)
    draw.text((80, height - 80), f"Page {page_num}", fill="gray", font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def load_test_images(test_dir: str | None, count: int) -> list[tuple[str, bytes]]:
    """Load test images from directory, or generate synthetic ones."""
    if test_dir:
        p = Path(test_dir)
        files = sorted(
            f for f in p.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp")
        )
        if not files:
            sys.exit(f"No image files found in {test_dir}")
        images = []
        for i in range(count):
            f = files[i % len(files)]
            images.append((f.name, f.read_bytes()))
        return images

    print(f"Generating {count} synthetic test pages...")
    return [(f"bench_page_{i+1}.png", generate_test_image(i + 1)) for i in range(count)]


# ---------------------------------------------------------------------------
# Worker client helpers
# ---------------------------------------------------------------------------

async def check_worker(client: httpx.AsyncClient, url: str, headers: dict) -> dict:
    """Hit /health and return the response, or exit."""
    try:
        resp = await client.get(f"{url}/health", timeout=10.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        sys.exit(f"Cannot reach worker at {url}/health: {e}")


async def set_batch_config(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    batch_size: int,
    batch_wait: float = 0.3,
) -> None:
    resp = await client.put(
        f"{url}/config",
        json={"batch_size": batch_size, "batch_wait_seconds": batch_wait},
        headers=headers,
    )
    resp.raise_for_status()


async def submit_job(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    filename: str,
    image_data: bytes,
) -> str:
    """Submit a job and return the job_id."""
    files = {"file": (filename, image_data, "image/png")}
    resp = await client.post(f"{url}/jobs", files=files, headers=headers, timeout=30.0)
    if resp.status_code == 429:
        await asyncio.sleep(1.0)
        resp = await client.post(f"{url}/jobs", files=files, headers=headers, timeout=30.0)
    resp.raise_for_status()
    return resp.json()["job_id"]


async def poll_job(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    job_id: str,
    timeout: float = 300.0,
) -> dict:
    """Poll until job completes or times out."""
    deadline = time.time() + timeout
    interval = 0.3
    while time.time() < deadline:
        try:
            resp = await client.get(f"{url}/jobs/{job_id}", headers=headers, timeout=10.0)
            if resp.status_code == 200:
                data = resp.json()
                if data["status"] in ("completed", "failed"):
                    return data
        except httpx.RequestError:
            pass
        await asyncio.sleep(interval)
        interval = min(interval * 1.2, 2.0)
    return {"status": "timeout", "error": "Poll timeout"}


async def get_jobs_snapshot(
    client: httpx.AsyncClient, url: str, headers: dict
) -> list[dict]:
    """GET /jobs and return list of job dicts."""
    try:
        resp = await client.get(f"{url}/jobs", headers=headers, timeout=10.0)
        resp.raise_for_status()
        return resp.json().get("jobs", [])
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Result dataclass (shared by both modes)
# ---------------------------------------------------------------------------

@dataclass
class BatchSizeResult:
    batch_size: int
    pages: int
    total_seconds: float
    pages_per_second: float = 0.0
    pages_per_hour: float = 0.0
    avg_server_elapsed: float = 0.0
    median_server_elapsed: float = 0.0
    p95_server_elapsed: float = 0.0
    failures: int = 0
    cost_per_1k_pages: float | None = None
    # Raw per-page server elapsed times (used by aggregate_cycles to recompute
    # percentiles correctly across cycles instead of averaging averages).
    elapsed_samples: list[float] = field(default_factory=list)
    # Per-cycle pg/hr values when this result was produced by aggregating
    # multiple cycles. Empty for single-run results.
    cycle_pages_per_hour: list[float] = field(default_factory=list)

    def __post_init__(self):
        if self.total_seconds > 0 and self.pages > 0:
            successful = self.pages - self.failures
            self.pages_per_second = successful / self.total_seconds
            self.pages_per_hour = self.pages_per_second * 3600


# ===========================================================================
#  MODE 1: STANDALONE  (worker idle, benchmark injects its own images)
# ===========================================================================

@dataclass
class StandaloneJobResult:
    job_id: str
    submit_time: float
    complete_time: float | None = None
    elapsed_seconds: float | None = None
    status: str = "pending"
    error: str | None = None


async def run_standalone_test(
    url: str,
    headers: dict,
    images: list[tuple[str, bytes]],
    batch_size: int,
    concurrency: int = 16,
) -> BatchSizeResult:
    """Submit synthetic images, measure throughput. Worker must be idle."""

    async with httpx.AsyncClient() as client:
        await set_batch_config(client, url, headers, batch_size)
        await asyncio.sleep(0.5)

        # Clear leftover benchmark jobs (safe when worker is idle)
        try:
            await client.delete(f"{url}/jobs", headers=headers)
        except Exception:
            pass

        semaphore = asyncio.Semaphore(concurrency)
        results: list[StandaloneJobResult] = []
        lock = asyncio.Lock()

        async def process_one(idx: int, filename: str, data: bytes):
            async with semaphore:
                t0 = time.time()
                try:
                    job_id = await submit_job(client, url, headers, filename, data)
                except Exception as e:
                    async with lock:
                        results.append(StandaloneJobResult(
                            job_id=f"err-{idx}", submit_time=t0,
                            status="failed", error=str(e),
                        ))
                    return

                resp = await poll_job(client, url, headers, job_id)
                t1 = time.time()

                jr = StandaloneJobResult(
                    job_id=job_id, submit_time=t0, complete_time=t1,
                    status=resp.get("status", "unknown"), error=resp.get("error"),
                )
                if resp.get("result") and resp["result"].get("elapsed_seconds"):
                    jr.elapsed_seconds = resp["result"]["elapsed_seconds"]

                async with lock:
                    results.append(jr)

        wall_start = time.time()
        tasks = [
            asyncio.create_task(process_one(i, fname, data))
            for i, (fname, data) in enumerate(images)
        ]
        await asyncio.gather(*tasks)
        wall_elapsed = time.time() - wall_start

        server_times = [
            r.elapsed_seconds for r in results
            if r.elapsed_seconds is not None and r.status == "completed"
        ]
        failures = sum(1 for r in results if r.status != "completed")

        result = BatchSizeResult(
            batch_size=batch_size,
            pages=len(images),
            total_seconds=round(wall_elapsed, 2),
            failures=failures,
        )

        if server_times:
            result.avg_server_elapsed = round(statistics.mean(server_times), 3)
            result.median_server_elapsed = round(statistics.median(server_times), 3)
            result.p95_server_elapsed = round(
                sorted(server_times)[int(len(server_times) * 0.95)], 3
            )

        return result


# ===========================================================================
#  MODE 2: LIVE  (OCRHarbor feeding real work, we observe throughput)
# ===========================================================================

async def run_live_observation(
    url: str,
    headers: dict,
    batch_size: int,
    window_seconds: float,
    poll_interval: float = 5.0,
) -> BatchSizeResult:
    """
    Set batch_size on the worker, then observe completions over window_seconds.
    Does NOT submit any jobs — OCRHarbor does that.

    Measures throughput by polling /jobs periodically and counting newly
    completed jobs within the observation window.
    """
    async with httpx.AsyncClient() as client:
        # Apply the batch_size config change
        await set_batch_config(client, url, headers, batch_size)

        # Give the worker a few seconds to drain any in-progress batch under
        # the old config before we start counting
        settle_time = 5.0
        print(f"  settling ({settle_time:.0f}s)...", end="", flush=True)
        await asyncio.sleep(settle_time)

        # Snapshot: record which jobs are already completed so we only count new ones
        seen_completed: set[str] = set()
        initial_jobs = await get_jobs_snapshot(client, url, headers)
        for j in initial_jobs:
            if j.get("status") == "completed":
                seen_completed.add(j["job_id"])

        completed_elapsed: list[float] = []  # server-side elapsed for each new completion
        failed_count = 0
        window_start = time.time()
        deadline = window_start + window_seconds
        last_status = ""

        while time.time() < deadline:
            await asyncio.sleep(poll_interval)
            jobs = await get_jobs_snapshot(client, url, headers)

            new_done = 0
            for j in jobs:
                jid = j.get("job_id", "")
                status = j.get("status", "")
                if jid in seen_completed:
                    continue
                if status == "completed":
                    seen_completed.add(jid)
                    new_done += 1
                    result = j.get("result")
                    if result and result.get("elapsed_seconds"):
                        completed_elapsed.append(result["elapsed_seconds"])
                elif status == "failed":
                    seen_completed.add(jid)
                    failed_count += 1

            remaining = deadline - time.time()
            total_new = len(completed_elapsed)
            rate = total_new / (time.time() - window_start) * 3600 if total_new else 0
            last_status = (
                f"\r  batch_size={batch_size}  "
                f"pages={total_new}  "
                f"~{rate:.0f} pg/hr  "
                f"remaining={max(0, remaining):.0f}s  "
            )
            print(last_status, end="", flush=True)

        wall_elapsed = time.time() - window_start
        total_pages = len(completed_elapsed)
        print()  # newline after progress

        # `pages` is the total observed (completions + failures), to match
        # standalone mode where `pages` is total submitted. __post_init__
        # subtracts failures to compute throughput.
        result = BatchSizeResult(
            batch_size=batch_size,
            pages=total_pages + failed_count,
            total_seconds=round(wall_elapsed, 2),
            failures=failed_count,
        )

        if completed_elapsed:
            result.avg_server_elapsed = round(statistics.mean(completed_elapsed), 3)
            result.median_server_elapsed = round(statistics.median(completed_elapsed), 3)
            result.p95_server_elapsed = round(
                sorted(completed_elapsed)[int(len(completed_elapsed) * 0.95)], 3
            )

        # Keep raw samples around so aggregate_cycles() can recompute stats
        # over the union of samples instead of averaging averages.
        result.elapsed_samples = list(completed_elapsed)
        return result


def aggregate_cycles(
    per_cycle: dict[int, list[BatchSizeResult]],
) -> list[BatchSizeResult]:
    """
    Combine multiple per-cycle observations of the same batch_size into a
    single BatchSizeResult.

    Throughput is computed from summed totals (NOT averaged from per-cycle
    pg/hr) so that windows containing fewer pages don't get equal weight to
    denser windows. Percentiles are recomputed over the union of all raw
    elapsed_samples for the same reason.
    """
    aggregated: list[BatchSizeResult] = []
    for bs in sorted(per_cycle.keys()):
        runs = per_cycle[bs]
        if not runs:
            continue

        total_pages = sum(r.pages for r in runs)
        total_failures = sum(r.failures for r in runs)
        total_seconds = sum(r.total_seconds for r in runs)
        all_samples: list[float] = []
        for r in runs:
            all_samples.extend(r.elapsed_samples)

        merged = BatchSizeResult(
            batch_size=bs,
            pages=total_pages,
            total_seconds=round(total_seconds, 2),
            failures=total_failures,
        )
        if all_samples:
            merged.avg_server_elapsed = round(statistics.mean(all_samples), 3)
            merged.median_server_elapsed = round(statistics.median(all_samples), 3)
            merged.p95_server_elapsed = round(
                sorted(all_samples)[int(len(all_samples) * 0.95)], 3
            )
        # Preserve per-cycle pg/hr so print_results can show variance.
        merged.cycle_pages_per_hour = [round(r.pages_per_hour, 1) for r in runs]
        merged.elapsed_samples = all_samples
        aggregated.append(merged)
    return aggregated


# ---------------------------------------------------------------------------
# Output / reporting  (shared)
# ---------------------------------------------------------------------------

def print_results(
    results: list[BatchSizeResult],
    gpu_name: str,
    gpu_cost: float | None,
    mode: str = "standalone",
) -> dict:
    print("\n" + "=" * 80)
    print(f"  BENCHMARK RESULTS — {gpu_name}  [{mode} mode]")
    print("=" * 80)

    cols = [
        ("Batch", 6), ("Pages", 6), ("Wall(s)", 8), ("pg/sec", 7),
        ("pg/hr", 8), ("avg(s)", 7), ("med(s)", 7), ("p95(s)", 7), ("fail", 5),
    ]
    if gpu_cost:
        cols.append(("$/1Kpg", 8))

    header = " | ".join(f"{name:>{w}}" for name, w in cols)
    print(f"\n{header}")
    print("-" * len(header))

    best_value = None
    best_throughput = None

    for r in results:
        if gpu_cost and r.pages_per_hour > 0:
            r.cost_per_1k_pages = round((gpu_cost / r.pages_per_hour) * 1000, 4)
            if best_value is None or r.cost_per_1k_pages < best_value.cost_per_1k_pages:
                best_value = r
        if best_throughput is None or r.pages_per_hour > best_throughput.pages_per_hour:
            best_throughput = r

        vals = [
            f"{r.batch_size:>6}", f"{r.pages:>6}", f"{r.total_seconds:>8.1f}",
            f"{r.pages_per_second:>7.2f}", f"{r.pages_per_hour:>8.0f}",
            f"{r.avg_server_elapsed:>7.3f}", f"{r.median_server_elapsed:>7.3f}",
            f"{r.p95_server_elapsed:>7.3f}", f"{r.failures:>5}",
        ]
        if gpu_cost:
            vals.append(f"{r.cost_per_1k_pages:>8.4f}" if r.cost_per_1k_pages else f"{'N/A':>8}")
        print(" | ".join(vals))

        # When this row is the result of aggregating multiple cycles, print
        # the per-cycle pg/hr values + a spread% so the user can see whether
        # the aggregate is built on consistent or wildly varying samples.
        if len(r.cycle_pages_per_hour) > 1:
            cycles_str = ", ".join(f"{v:.0f}" for v in r.cycle_pages_per_hour)
            cmin = min(r.cycle_pages_per_hour)
            cmax = max(r.cycle_pages_per_hour)
            mean = sum(r.cycle_pages_per_hour) / len(r.cycle_pages_per_hour)
            spread_pct = ((cmax - cmin) / mean * 100) if mean > 0 else 0
            print(f"{'':>6}   cycles pg/hr: [{cycles_str}]  spread={spread_pct:.0f}%")

    print()
    if best_throughput:
        print(f"  Fastest throughput : batch_size={best_throughput.batch_size}"
              f"  →  {best_throughput.pages_per_hour:.0f} pages/hr"
              f"  ({best_throughput.pages_per_second:.2f} pg/sec)")
    if best_value and gpu_cost:
        print(f"  Best value         : batch_size={best_value.batch_size}"
              f"  →  ${best_value.cost_per_1k_pages:.4f} per 1K pages"
              f"  (at ${gpu_cost:.2f}/hr)")
    print()

    report = {
        "gpu_name": gpu_name,
        "gpu_cost_per_hr": gpu_cost,
        "mode": mode,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": [
            {
                "batch_size": r.batch_size,
                "pages": r.pages,
                "total_seconds": r.total_seconds,
                "pages_per_second": round(r.pages_per_second, 3),
                "pages_per_hour": round(r.pages_per_hour, 1),
                "avg_server_elapsed": r.avg_server_elapsed,
                "median_server_elapsed": r.median_server_elapsed,
                "p95_server_elapsed": r.p95_server_elapsed,
                "failures": r.failures,
                "cost_per_1k_pages": r.cost_per_1k_pages,
                "cycle_pages_per_hour": r.cycle_pages_per_hour,
            }
            for r in results
        ],
        "best_throughput_batch_size": best_throughput.batch_size if best_throughput else None,
        "best_value_batch_size": best_value.batch_size if best_value else None,
    }

    slug = gpu_name.lower().replace(" ", "_").replace("/", "_")
    report_path = f"benchmark_{slug}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Full report saved to: {report_path}")
    print()
    return report


def print_comparison(report_files: list[str]):
    """Load multiple JSON reports and print a comparison table."""
    reports = []
    for path in report_files:
        with open(path) as f:
            reports.append(json.load(f))

    print("\n" + "=" * 80)
    print("  GPU COMPARISON")
    print("=" * 80)

    cols = [("GPU", 20), ("Mode", 11), ("Best Batch", 11), ("pg/hr", 9), ("$/hr", 7), ("$/1Kpg", 8)]
    header = " | ".join(f"{name:>{w}}" for name, w in cols)
    print(f"\n{header}")
    print("-" * len(header))

    rows = []
    for r in reports:
        best = None
        for res in r["results"]:
            if res.get("cost_per_1k_pages") is not None:
                if best is None or res["cost_per_1k_pages"] < best["cost_per_1k_pages"]:
                    best = res
        if best is None:
            best = max(r["results"], key=lambda x: x["pages_per_hour"])

        rows.append({
            "gpu": r["gpu_name"],
            "mode": r.get("mode", "standalone"),
            "batch": best["batch_size"],
            "pghr": best["pages_per_hour"],
            "costhr": r["gpu_cost_per_hr"],
            "cost1k": best.get("cost_per_1k_pages"),
        })

    rows.sort(key=lambda x: x["cost1k"] if x["cost1k"] else 999)

    for row in rows:
        cost_str = f"${row['cost1k']:.4f}" if row["cost1k"] else "N/A"
        costhr_str = f"${row['costhr']:.2f}" if row["costhr"] else "N/A"
        print(
            f"{row['gpu']:>20} | "
            f"{row['mode']:>11} | "
            f"{row['batch']:>11} | "
            f"{row['pghr']:>9.0f} | "
            f"{costhr_str:>7} | "
            f"{cost_str:>8}"
        )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _read_secret_from_dotenv() -> str:
    """Read WORKER_SECRET from .env file (same source the worker uses)."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return ""
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "WORKER_SECRET":
            # Strip optional quotes
            value = value.strip().strip("'\"")
            return value
    return ""


async def async_main(args):
    url = args.url.rstrip("/")
    secret = args.secret or os.environ.get("WORKER_SECRET", "") or _read_secret_from_dotenv()
    headers = {}
    if secret:
        headers["Authorization"] = f"Bearer {secret}"

    # Compare mode
    if args.compare:
        print_comparison(args.compare)
        return

    # Check worker health
    async with httpx.AsyncClient() as client:
        health = await check_worker(client, url, headers)

    gpu_name = args.gpu_name or health.get("gpu_name") or "Unknown GPU"
    print(f"Worker: {url}")
    print(f"GPU: {gpu_name}")
    print(f"Model loaded: {health.get('model_loaded')}")
    if args.gpu_cost:
        print(f"GPU cost: ${args.gpu_cost:.2f}/hr")

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    print(f"Batch sizes to test: {batch_sizes}")

    # ----- LIVE MODE -----
    if args.live:
        cycles = max(1, args.cycles)
        print(f"Mode: LIVE (observing real OCRHarbor traffic)")
        print(f"Window per batch_size per cycle: {args.window}s")
        print(f"Cycles: {cycles}  (randomized order each cycle)")
        total_time = cycles * len(batch_sizes) * (args.window + 5)
        print(f"Estimated total time: ~{total_time / 60:.0f} min\n")

        # Verify there's actually work flowing
        async with httpx.AsyncClient() as client:
            h = await check_worker(client, url, headers)
            qd = h.get("queue_depth", 0)
            if qd == 0:
                print("WARNING: Worker queue is empty. Make sure OCRHarbor is")
                print("         feeding jobs, otherwise there's nothing to measure.\n")

        per_cycle: dict[int, list[BatchSizeResult]] = {bs: [] for bs in batch_sizes}
        for cycle_idx in range(cycles):
            order = list(batch_sizes)
            random.shuffle(order)
            print(f"\n=== Cycle {cycle_idx + 1}/{cycles}  order: {order} ===")
            for bs in order:
                print(f"\n--- batch_size={bs} (cycle {cycle_idx + 1}/{cycles}, "
                      f"observing for {args.window}s) ---")
                result = await run_live_observation(
                    url=url, headers=headers,
                    batch_size=bs, window_seconds=args.window,
                )
                if result.pages == 0:
                    print(f"  WARNING: 0 pages completed — queue may be starved")
                per_cycle[bs].append(result)
                print(f"  Result: {result.pages_per_hour:.0f} pg/hr")

        # Aggregate cycles into one result per batch_size
        results = aggregate_cycles(per_cycle)
        for r in results:
            if args.gpu_cost and r.pages_per_hour > 0:
                r.cost_per_1k_pages = round(
                    (args.gpu_cost / r.pages_per_hour) * 1000, 4
                )

        # Restore a sensible default when done
        async with httpx.AsyncClient() as client:
            best = max(results, key=lambda r: r.pages_per_hour)
            await set_batch_config(client, url, headers, best.batch_size)
            print(f"\nAuto-set worker to best batch_size={best.batch_size}")

        print_results(results, gpu_name, args.gpu_cost, mode="live")
        return

    # ----- STANDALONE MODE -----
    print(f"Mode: STANDALONE (synthetic images, worker should be idle)")
    images = load_test_images(args.test_dir, args.pages)
    print(f"Test images: {len(images)}")
    print(f"Client concurrency: {args.concurrency}")

    # Warmup
    print("\nWarmup run (3 pages)...")
    warmup_imgs = images[:3]
    async with httpx.AsyncClient() as client:
        await set_batch_config(client, url, headers, 1)
        await asyncio.sleep(0.3)
        for fname, data in warmup_imgs:
            try:
                jid = await submit_job(client, url, headers, fname, data)
                await poll_job(client, url, headers, jid, timeout=120.0)
            except Exception as e:
                print(f"  Warmup warning: {e}")
    print("Warmup complete.\n")

    results: list[BatchSizeResult] = []
    for bs in batch_sizes:
        print(f"Testing batch_size={bs} with {len(images)} pages ...", end="", flush=True)
        result = await run_standalone_test(
            url=url, headers=headers, images=images,
            batch_size=bs, concurrency=args.concurrency,
        )
        if args.gpu_cost and result.pages_per_hour > 0:
            result.cost_per_1k_pages = round(
                (args.gpu_cost / result.pages_per_hour) * 1000, 4
            )
        results.append(result)
        print(f"  {result.pages_per_hour:.0f} pg/hr  ({result.total_seconds:.1f}s)")

    print_results(results, gpu_name, args.gpu_cost, mode="standalone")


def main():
    parser = argparse.ArgumentParser(
        description="OCRHarborWorker GPU Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standalone (worker idle)
  python benchmark.py --gpu-cost 0.22 --gpu-name "RTX 3090"

  # Live sweep while OCRHarbor feeds real work
  python benchmark.py --live --gpu-cost 0.22 --gpu-name "RTX 3090"

  # Longer observation windows for more stable numbers
  python benchmark.py --live --window 600 --gpu-cost 0.22

  # Compare reports
  python benchmark.py --compare benchmark_rtx_3090_*.json benchmark_rtx_4090_*.json
        """,
    )
    parser.add_argument("--url", default="http://localhost:5001",
                        help="Worker base URL")
    parser.add_argument("--secret", default=None,
                        help="Worker secret (auto-read from .env if omitted)")
    parser.add_argument("--gpu-name", default="",
                        help="GPU name for report (auto-detected from /health if omitted)")
    parser.add_argument("--gpu-cost", type=float, default=None,
                        help="GPU rental cost in $/hr")

    # Mode selection
    parser.add_argument("--live", action="store_true",
                        help="Live mode: observe real OCRHarbor traffic instead of injecting synthetic images")
    parser.add_argument("--window", type=int, default=600,
                        help="Seconds to observe each batch_size per cycle in live mode (default: 600 = 10 min)")
    parser.add_argument("--cycles", type=int, default=3,
                        help="Number of interleaved cycles in live mode (default: 3). Each cycle "
                             "tests every batch_size in randomized order, then results are summed "
                             "across cycles. Controls for traffic-mix and queue-depth drift.")

    # Standalone mode options
    parser.add_argument("--pages", type=int, default=50,
                        help="Number of synthetic test pages in standalone mode (default: 50)")
    parser.add_argument("--batch-sizes", default="5,6,7,8,9,10",
                        help="Comma-separated batch sizes to sweep (default: 5,6,7,8,9,10)")
    parser.add_argument("--concurrency", type=int, default=24,
                        help="Max concurrent HTTP submissions in standalone mode")
    parser.add_argument("--test-dir", default=None,
                        help="Directory of real test images for standalone mode (optional)")

    # Comparison
    parser.add_argument("--compare", nargs="+", metavar="FILE",
                        help="Compare previous benchmark JSON reports")

    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
