#!/usr/bin/env python3
"""
Live TUI monitor for the OCR GPU Worker.
Run in a separate terminal on the VAST.ai instance:
  python3 monitor.py
"""
import sys
import time
from datetime import datetime, timezone

import httpx
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

# Read port from .env or default
PORT = 5000
try:
    with open(".env") as f:
        for line in f:
            if line.startswith("PORT="):
                PORT = int(line.strip().split("=", 1)[1])
            if line.startswith("WORKER_SECRET="):
                SECRET = line.strip().split("=", 1)[1]
except FileNotFoundError:
    SECRET = ""

BASE = f"http://localhost:{PORT}"
HEADERS = {"Authorization": f"Bearer {SECRET}"} if SECRET else {}

console = Console()

# Track completed jobs for the log
completed_log: list[dict] = []
seen_completed: set[str] = set()


def fetch_health() -> dict:
    try:
        resp = httpx.get(f"{BASE}/health", timeout=3.0)
        return resp.json()
    except Exception:
        return {}


def fetch_jobs() -> list[dict]:
    try:
        resp = httpx.get(f"{BASE}/jobs", headers=HEADERS, timeout=3.0)
        data = resp.json()
        return data.get("jobs", [])
    except Exception:
        return []


def status_style(status: str) -> str:
    return {
        "queued": "yellow",
        "processing": "bold cyan",
        "completed": "bold green",
        "failed": "bold red",
        "cancelled": "dim",
    }.get(status, "white")


def format_elapsed(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return f"{seconds:.1f}s"


def ago(iso_str: str | None) -> str:
    if not iso_str:
        return "-"
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        diff = datetime.now(timezone.utc) - dt
        secs = int(diff.total_seconds())
        if secs < 60:
            return f"{secs}s ago"
        elif secs < 3600:
            return f"{secs // 60}m {secs % 60}s ago"
        else:
            return f"{secs // 3600}h {(secs % 3600) // 60}m ago"
    except Exception:
        return "-"


def build_display(health: dict, jobs: list[dict]) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="log", size=min(len(completed_log) + 3, 12)),
    )

    # Header
    gpu = health.get("gpu_name", "?")
    uptime = health.get("uptime_seconds", 0)
    model = "loaded" if health.get("model_loaded") else "loading..."
    queue = health.get("queue_depth", 0)
    status_color = "green" if health.get("status") == "ok" else "red"

    header_text = Text()
    header_text.append("  GPU: ", style="bold")
    header_text.append(f"{gpu}", style="cyan")
    header_text.append("  |  Model: ", style="bold")
    header_text.append(f"{model}", style="green" if model == "loaded" else "yellow")
    header_text.append("  |  Queue: ", style="bold")
    header_text.append(f"{queue}", style="yellow" if queue > 0 else "green")
    header_text.append("  |  Uptime: ", style="bold")
    header_text.append(f"{int(uptime)}s", style="dim")

    layout["header"].update(Panel(header_text, title="[bold]OCR GPU Worker[/bold]", border_style=status_color))

    # Active jobs table
    table = Table(expand=True, show_lines=False, padding=(0, 1))
    table.add_column("ID", style="dim", width=14)
    table.add_column("Status", width=12)
    table.add_column("File", min_width=20, max_width=40)
    table.add_column("Elapsed", justify="right", width=10)
    table.add_column("Submitted", justify="right", width=14)

    # Track newly completed
    active_jobs = []
    for job in jobs:
        jid = job.get("job_id", "")
        status = job.get("status", "")

        if status == "completed" and jid not in seen_completed:
            seen_completed.add(jid)
            elapsed = None
            if job.get("result") and job["result"].get("elapsed_seconds"):
                elapsed = job["result"]["elapsed_seconds"]
            completed_log.append({
                "id": jid,
                "file": job.get("filename", "?"),
                "elapsed": elapsed,
                "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            })
            # Keep last 20
            if len(completed_log) > 20:
                completed_log.pop(0)

        if status in ("queued", "processing"):
            active_jobs.append(job)

    if active_jobs:
        for job in active_jobs:
            status = job.get("status", "")
            elapsed = None
            if status == "processing" and job.get("started_at"):
                try:
                    started = datetime.fromisoformat(job["started_at"])
                    if started.tzinfo is None:
                        started = started.replace(tzinfo=timezone.utc)
                    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
                except Exception:
                    pass

            table.add_row(
                job.get("job_id", "")[:12],
                f"[{status_style(status)}]{status}[/]",
                job.get("filename", "?"),
                f"{elapsed:.0f}s" if elapsed else "-",
                ago(job.get("created_at")),
            )
    else:
        table.add_row("", "[dim]idle[/dim]", "[dim]waiting for jobs...[/dim]", "", "")

    layout["body"].update(Panel(table, title="Active Jobs", border_style="blue"))

    # Completed log
    log_table = Table(expand=True, show_lines=False, show_header=True, padding=(0, 1))
    log_table.add_column("Time", style="dim", width=10)
    log_table.add_column("ID", style="dim", width=14)
    log_table.add_column("File", min_width=20, max_width=40)
    log_table.add_column("Elapsed", justify="right", width=10, style="green")

    for entry in reversed(completed_log[-8:]):
        log_table.add_row(
            entry["time"],
            entry["id"][:12],
            entry["file"],
            format_elapsed(entry.get("elapsed")),
        )

    layout["log"].update(Panel(log_table, title="Recently Completed", border_style="green"))

    return layout


def main():
    console.print("[bold]Starting OCR Worker Monitor...[/bold]")
    console.print(f"Connecting to {BASE}")
    console.print("Press Ctrl+C to exit\n")

    try:
        with Live(console=console, refresh_per_second=2, screen=True) as live:
            while True:
                health = fetch_health()
                jobs = fetch_jobs()
                display = build_display(health, jobs)
                live.update(display)
                time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
