"""Results registry: one directory per evaluation run under ``$T2C_RESULTS_DIR/runs/``.

    summary.json        small, self-describing, tracked in git -> syncs between clusters
    samples.jsonl.gz    per-sample outputs (generated code, errors, metrics); kept locally

Nothing is ever overwritten: re-evaluating a model creates a new run, and reports pick
the most recent one per (benchmark, model, protocol).
"""

from __future__ import annotations

import gzip
import json
import platform
import re
import subprocess
import time
from pathlib import Path

from .. import __version__, paths


def git_state() -> dict:
    def run(*cmd: str) -> str:
        try:
            return subprocess.run(cmd, cwd=paths.REPO_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    commit = run("git", "rev-parse", "--short", "HEAD")
    dirty = bool(run("git", "status", "--porcelain", "--untracked-files=no"))
    return {"commit": commit or None, "dirty": dirty}


def new_run_id(model_name: str, benchmark: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.@-]+", "-", model_name).strip("-")
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{benchmark}-{slug}"


def runs_dir() -> Path:
    return paths.results_dir() / "runs"


def save_run(summary: dict, rows: list[dict]) -> Path:
    d = paths.ensure(runs_dir() / summary["run_id"])
    summary = {**summary, "host": platform.node(), "git": git_state(), "t2c_version": __version__}
    (d / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    with gzip.open(d / "samples.jsonl.gz", "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return d


def list_runs(include_partial: bool = False) -> list[dict]:
    out = []
    if not runs_dir().exists():
        return out
    for p in sorted(runs_dir().glob("*/summary.json")):
        try:
            s = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if s.get("partial") and not include_partial:
            continue
        out.append(s)
    return sorted(out, key=lambda s: s["created"])


def load_rows(run_id: str) -> list[dict]:
    with gzip.open(runs_dir() / run_id / "samples.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]
