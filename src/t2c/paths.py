"""Filesystem layout. Everything is overridable through environment variables so the
same code runs unchanged on different clusters (see configs/clusters/*.env).

    T2C_DATA_DIR     raw downloads, processed datasets, ground-truth meshes   (default: <repo>/data)
    T2C_OUTPUT_DIR   checkpoints, per-sample eval dumps, logs                 (default: <repo>/outputs)
    T2C_RESULTS_DIR  small, git-tracked benchmark results registry            (default: <repo>/results)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _dir(env: str, default: Path) -> Path:
    p = Path(os.environ.get(env, default)).expanduser()
    return p


def data_dir() -> Path:
    return _dir("T2C_DATA_DIR", REPO_ROOT / "data")


def output_dir() -> Path:
    return _dir("T2C_OUTPUT_DIR", REPO_ROOT / "outputs")


def results_dir() -> Path:
    return _dir("T2C_RESULTS_DIR", REPO_ROOT / "results")


def raw_dir(name: str) -> Path:
    """Untouched upstream downloads, one sub-directory per source."""
    return data_dir() / "raw" / name


def processed_dir(name: str) -> Path:
    """Cleaned, execution-verified datasets (jsonl + ground-truth meshes)."""
    return data_dir() / "processed" / name


def bench_dir(name: str) -> Path:
    """Prepared benchmark: samples.jsonl + gt/<id>.npz."""
    return data_dir() / "bench" / name


def ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
