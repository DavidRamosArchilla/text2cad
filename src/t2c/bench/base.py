"""Benchmark interface.

A prepared benchmark is a directory ``$T2C_DATA_DIR/bench/<name>/`` holding

    samples.jsonl   one task per line: id, group, task (text shown to the model), gt, meta
    gt/*.npz        ground-truth meshes (absent when the benchmark publishes none)
    info.json       provenance + a content hash, recorded with every result

Every benchmark is scored with the same core metrics (see ``aggregate``), so numbers are
comparable across benchmarks and across models; benchmarks may add their own on top.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import paths, prompts


@dataclass
class Sample:
    id: str
    task: str
    group: str = "all"
    gt: str | None = None  # path relative to the benchmark directory
    meta: dict = field(default_factory=dict)


class Benchmark:
    name: str = ""
    description: str = ""
    protocol_system: str = prompts.NATIVE_SYSTEM  # system prompt for prompt_style="protocol"
    default_retries: int = 0  # error-feedback retries prescribed by the benchmark's protocol
    headline: str = "iou_all"  # the single number to watch
    res: int = 256
    n_points: int = 30_000
    timeout: float = 60.0

    # ------------------------------------------------------------------ storage
    @property
    def dir(self) -> Path:
        return paths.bench_dir(self.name)

    def is_prepared(self) -> bool:
        return (self.dir / "samples.jsonl").exists()

    def prepare(self, workers: int | None = None) -> dict:  # pragma: no cover - per benchmark
        raise NotImplementedError

    def _write(self, samples: list[Sample], info: dict) -> dict:
        paths.ensure(self.dir)
        body = "".join(json.dumps(asdict(s), ensure_ascii=False, sort_keys=True) + "\n" for s in samples)
        (self.dir / "samples.jsonl").write_text(body)
        info = {"name": self.name, "n": len(samples), "sha1": hashlib.sha1(body.encode()).hexdigest()[:12], **info}
        info["groups"] = {g: sum(1 for s in samples if s.group == g) for g in sorted({s.group for s in samples})}
        info["has_gt"] = any(s.gt for s in samples)
        (self.dir / "info.json").write_text(json.dumps(info, indent=1))
        return info

    def info(self) -> dict:
        return json.loads((self.dir / "info.json").read_text())

    def load(self) -> list[Sample]:
        if not self.is_prepared():
            raise FileNotFoundError(f"benchmark '{self.name}' is not prepared -- run `t2c bench prepare {self.name}`")
        with (self.dir / "samples.jsonl").open() as fh:
            return [Sample(**json.loads(line)) for line in fh]

    # ------------------------------------------------------------------ protocol
    def system_prompt(self, style: str) -> str:
        return self.protocol_system if style == "protocol" else prompts.NATIVE_SYSTEM

    def make_job(self, sample: Sample, code: str | None, style: str) -> dict:
        job = {"code": code, "timeout": self.timeout, "res": self.res, "n_points": self.n_points}
        if sample.gt:
            job["gt"] = str(self.dir / sample.gt)
        return job

    def row_extra(self, sample: Sample, result: dict) -> dict:
        """Benchmark-specific per-sample fields to keep (beyond the core ones)."""
        return {}

    def aggregate_extra(self, rows: list[dict]) -> dict:
        return {}

    # ------------------------------------------------------------------ scoring
    def aggregate(self, rows: list[dict]) -> dict:
        out = {"overall": self._agg(rows), "groups": {}}
        for g in sorted({r["group"] for r in rows}):
            out["groups"][g] = self._agg([r for r in rows if r["group"] == g])
        return out

    def _agg(self, rows: list[dict]) -> dict:
        """Core metrics.

        ir / ir_first   invalidity rate after retries / on the first attempt
        iou, cd         mean over VALID samples only (Text2CAD-Bench convention; cd is x1000)
        iou_all         mean IoU with invalid samples counted as 0 -- cannot be improved by
                        failing on hard parts, so this is the number to optimise
        """
        broken = [r for r in rows if r.get("harness_error")]
        rows = [r for r in rows if not r.get("harness_error")]
        n = len(rows)
        out: dict = {"n": n, "harness_errors": len(broken)}
        if n == 0:
            return out
        valid = [r for r in rows if r["valid"]]
        out["ir"] = 1.0 - len(valid) / n
        out["ir_first"] = 1.0 - sum(1 for r in rows if r.get("first_valid")) / n
        ious = [r["iou"] for r in valid if _num(r.get("iou"))]
        cds = [r["cd"] for r in valid if _num(r.get("cd"))]
        if ious:
            out["iou"] = statistics.fmean(ious)
            out["iou_all"] = sum(ious) / n
            out["iou_median"] = statistics.median(ious)
        if cds:
            out["cd"] = statistics.fmean(cds) * 1e3
            out["cd_median"] = statistics.median(cds) * 1e3
        dims = [r["dim_err"] for r in valid if _num(r.get("dim_err"))]
        if dims:
            out["dim_err_median"] = statistics.median(dims)
        out.update(self.aggregate_extra(rows))
        return out


def _num(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)
