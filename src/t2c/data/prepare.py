"""Turn the raw training corpora into clean, execution-verified datasets.

For every (description, script) pair:
  1. normalise the script to the ``result`` contract (drop export/visualisation lines);
  2. run it in the sandbox -- it must produce a valid, watertight solid;
  3. keep it, with its DeepCAD model id (``uid``) so benchmark objects can be held out.

Output: ``$T2C_DATA_DIR/processed/<source>/<split>.jsonl`` + ``stats.json``.
Deterministic: same input revision -> same output, on any machine.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from .. import paths
from ..cad.codeutil import normalize_reference_script
from ..cad.sandbox import CadExecutor

_UID_IN_EXPORT = re.compile(r"/(\d{8})\.stl")
_CADCODER_PREFIX = re.compile(r"^\s*Please based on the following description, create a CAD-Query Code to generate a model:\s*", re.I)
MAX_OPEN_FRAC = 0.005  # reference surfaces must be closed
MAX_CODE_CHARS = 6000  # drop pathological scripts (they would be truncated in training anyway)
VERIFY_TIMEOUT = 30.0


def _iter_t2cq(raw: Path) -> Iterator[dict]:
    for split, fn in (("train", "data_train.jsonl"), ("val", "data_val.jsonl"), ("test", "data_test.jsonl")):
        with (raw / "data" / fn).open() as fh:
            for i, line in enumerate(fh):
                rec = json.loads(line)
                m = _UID_IN_EXPORT.search(rec["output"])
                yield {"split": split, "index": i, "uid": m.group(1) if m else None, "text": rec["input"].strip(), "raw_code": rec["output"], "final_hint": None}


def _iter_cadcoder(raw: Path) -> Iterator[dict]:
    files = (("train_high", "cad_data_train_high.json"), ("train_middle", "cad_data_train_middle.json"), ("val", "cad_data_validation.json"))
    for split, fn in files:
        path = raw / fn
        if not path.exists():
            continue
        for i, rec in enumerate(json.loads(path.read_text())):
            user = next(m["content"] for m in rec["messages"] if m["role"] == "user")
            code = next(m["content"] for m in rec["messages"] if m["role"] == "assistant")
            uid = re.match(r"(\d{8})", rec.get("model_path") or "")
            yield {"split": split, "index": i, "uid": uid.group(1) if uid else None, "text": _CADCODER_PREFIX.sub("", user).strip(), "raw_code": code, "final_hint": "r"}


ITERATORS = {"t2cq": _iter_t2cq, "cadcoder": _iter_cadcoder}


def prepare(source: str, workers: int | None = None, limit: int | None = None, executor: CadExecutor | None = None) -> dict:
    raw = paths.raw_dir(source)
    if not (raw / ".t2c_source.json").exists():
        raise FileNotFoundError(f"{raw} is missing -- run `t2c data download {source}` first")
    out_dir = paths.ensure(paths.processed_dir(source))

    records, drops, per_split = [], Counter(), Counter()
    seen: set[str] = set()
    for rec in ITERATORS[source](raw):
        if limit and per_split[rec["split"]] >= limit:
            continue
        code = normalize_reference_script(rec["raw_code"], rec["final_hint"])
        if code is None:
            drops["unnormalizable"] += 1
            continue
        if len(code) > MAX_CODE_CHARS:
            drops["too_long"] += 1
            continue
        if not rec["text"]:
            drops["empty_text"] += 1
            continue
        key = hashlib.sha1((rec["text"] + "\0" + code).encode()).hexdigest()
        if key in seen:
            drops["duplicate"] += 1
            continue
        seen.add(key)
        rec["code"] = code
        per_split[rec["split"]] += 1
        records.append(rec)

    own = executor is None
    ex = executor or CadExecutor(workers=workers, timeout=VERIFY_TIMEOUT)
    try:
        results = ex.run([{"code": r["code"], "self_check": 48} for r in records], progress=f"verify {source}")
    finally:
        if own:
            ex.close()

    by_split: dict[str, list[dict]] = {}
    errors = Counter()
    for rec, res in zip(records, results, strict=True):
        if not res["ok"]:
            errors[res["error_type"]] += 1
            continue
        if res.get("open_frac", 0.0) > MAX_OPEN_FRAC:
            errors["not_watertight"] += 1
            continue
        name = rec["uid"] or f"i{rec['index']:07d}"
        by_split.setdefault(rec["split"], []).append(
            {
                "id": f"{source}/{rec['split']}/{name}",
                "uid": rec["uid"],
                "text": rec["text"],
                "code": rec["code"],
                "volume": res["volume"],
                "bbox": res["bbox"],
                "n_solids": res["n_solids"],
                "n_tris": res["n_tris"],
            }
        )
    stats = {"source": source, "raw_kept_after_normalisation": len(records), "dropped_before_execution": dict(drops), "execution_failures": dict(errors), "splits": {}}
    for split, rows in by_split.items():
        with (out_dir / f"{split}.jsonl").open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats["splits"][split] = len(rows)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def read_split(source: str, split: str) -> list[dict]:
    path = paths.processed_dir(source) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing -- run `t2c data prepare {source}` first")
    with path.open() as fh:
        return [json.loads(line) for line in fh]
