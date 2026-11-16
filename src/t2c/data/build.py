"""Assemble training sets from the verified corpora, driven by a small YAML recipe.

kind: sft   -> <data>/<name>/{train,val}.jsonl with {id, text, code}
kind: rl    -> <data>/<name>/train.jsonl with {id, text, gt} + gt/*.npz (the verifier needs
              a ground-truth mesh per prompt; reference scripts are executed once here)

Per-entry options: ``n`` (sample size), ``max_code_chars``, and for the procedural data
``exclude_families`` -- train without some part families so that the per-family breakdown
of the ``synth`` benchmark separates seen from unseen kinds of parts.

Leakage control: ``holdout`` names the splits/benchmarks whose *objects* (DeepCAD model
ids) must never be trained on. Training rows whose uid is held out are dropped; validation
rows are taken only from held-out objects.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

from .. import paths
from ..cad.sandbox import CadExecutor
from .prepare import read_split


def _holdout_uids(specs: list[str]) -> set[str]:
    uids: set[str] = set()
    for spec in specs or []:
        kind, _, rest = spec.partition(":")
        if kind == "bench":
            from ..bench import registry

            uids |= registry.get(rest).uids()
        else:  # "<source>:<split>"
            uids |= {r["uid"] for r in read_split(kind, rest) if r.get("uid")}
    return uids


def _select(entries: list[dict], rng: random.Random, holdout: set[str], want_heldout: bool, stats: Counter) -> list[dict]:
    out = []
    for e in entries:
        rows = sorted(read_split(e["source"], e["split"]), key=lambda r: r["id"])
        tag = f"{e['source']}:{e['split']}"
        stats[f"{tag} available"] = len(rows)
        if e["source"] != "synth":  # synthetic parts have no DeepCAD identity
            rows = [r for r in rows if (r.get("uid") in holdout) == want_heldout]
        if e.get("max_code_chars"):
            rows = [r for r in rows if len(r["code"]) <= e["max_code_chars"]]
        if e.get("exclude_families"):  # synth only: keep whole part families out of training
            rows = [r for r in rows if r.get("family") not in set(e["exclude_families"])]
        rng.shuffle(rows)
        if e.get("n"):
            rows = rows[: e["n"]]
        stats[f"{tag} used"] = len(rows)
        out.extend(rows)
    rng.shuffle(out)
    return out


def build(config_path: str, workers: int | None = None) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    rng = random.Random(cfg.get("seed", 0))
    out_dir = paths.data_dir() / cfg["name"]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    paths.ensure(out_dir)
    holdout = _holdout_uids(cfg.get("holdout", []))
    stats: Counter = Counter()
    info = {"name": cfg["name"], "kind": cfg["kind"], "config": cfg, "holdout_objects": len(holdout)}

    train = _select(cfg["train"], rng, holdout, want_heldout=False, stats=stats)
    val = _select(cfg.get("val", []), rng, holdout, want_heldout=True, stats=stats)

    if cfg["kind"] == "sft":
        for split, rows in (("train", train), ("val", val)):
            with (out_dir / f"{split}.jsonl").open("w") as fh:
                for r in rows:
                    fh.write(json.dumps({"id": r["id"], "text": r["text"], "code": r["code"]}, ensure_ascii=False) + "\n")
    elif cfg["kind"] == "rl":
        with CadExecutor(workers=workers) as ex:
            for split, rows in (("train", train), ("val", val)):
                # One mesh per distinct script: several prompts can describe the same part.
                gt_of = {r["id"]: f"gt/{hashlib.sha1(r['code'].encode()).hexdigest()[:16]}.npz" for r in rows}
                todo = {}
                for r in rows:
                    todo.setdefault(gt_of[r["id"]], r["code"])
                results = ex.run([{"code": code, "save_mesh": str(out_dir / rel)} for rel, code in todo.items()], progress=f"{cfg['name']} {split} ground truth")
                good = {rel for rel, res in zip(todo, results, strict=True) if res["ok"]}
                with (out_dir / f"{split}.jsonl").open("w") as fh:
                    for r in rows:
                        if gt_of[r["id"]] in good:
                            fh.write(json.dumps({"id": r["id"], "text": r["text"], "gt": gt_of[r["id"]]}, ensure_ascii=False) + "\n")
                        else:
                            stats[f"{split} dropped (reference failed)"] += 1
    else:
        raise ValueError(f"unknown kind '{cfg['kind']}'")

    info["counts"] = {"train": sum(1 for _ in (out_dir / "train.jsonl").open()), "val": sum(1 for _ in (out_dir / "val.jsonl").open())}
    info["detail"] = dict(stats)
    (out_dir / "info.json").write_text(json.dumps(info, indent=1))
    return info


def read_built(name_or_path: str, split: str) -> tuple[list[dict], Path]:
    """Rows of a built dataset and its directory. Accepts a name relative to the data dir or a path."""
    d = Path(name_or_path)
    if not d.is_absolute() and not d.exists():
        d = paths.data_dir() / name_or_path
    path = d / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing -- run `t2c data build <recipe>` first")
    with path.open() as fh:
        return [json.loads(line) for line in fh], d
