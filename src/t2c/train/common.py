"""Shared plumbing for the trainers: config loading, run directories, lineage metadata."""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from .. import paths, prompts
from ..infer.backends import META_FILE, find_meta
from ..results.store import git_state


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    cfg["_config_path"] = str(path)
    return cfg


def run_dir(cfg: dict) -> Path:
    return paths.ensure(paths.output_dir() / "train" / cfg["run_name"])


def resolve_init(cfg: dict) -> tuple[str, str, dict | None]:
    """Where training starts from.

    Returns (path_or_id to load, base model id for lineage, meta of the parent run or None).
    ``init_from`` may be a HF model id or a checkpoint directory of an earlier t2c run.
    """
    init = cfg.get("init_from") or cfg["base_model"]
    p = Path(init)
    if not p.is_absolute() and not p.exists() and (paths.output_dir() / init).exists():
        p = paths.output_dir() / init
    if p.is_dir():
        meta = find_meta(p)
        base = (meta or {}).get("base_model") or cfg.get("base_model") or str(p)
        return str(p.resolve()), base, meta
    return init, init, None


def write_meta(out: Path, cfg: dict, stage: str, base_model: str, parent: str | None, chat_template_kwargs: dict, extra: dict | None = None) -> dict:
    """``t2c_meta.json`` makes a checkpoint self-describing: `t2c eval --model <checkpoint>`
    reads it to render prompts exactly as in training and to record lineage with the results."""
    meta = {
        "train_run": cfg["run_name"],
        "stage": stage,
        "base_model": base_model,
        "parent": parent,
        "chat_template_kwargs": chat_template_kwargs,
        "system_prompt": prompts.NATIVE_SYSTEM,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": git_state(),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        **(extra or {}),
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=1, default=str))
    (out / "config.yaml").write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, sort_keys=False))
    return meta


def is_main_process() -> bool:
    """True on rank 0 (or when not running distributed)."""
    import os

    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0
