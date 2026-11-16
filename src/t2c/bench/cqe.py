"""Validation of our CadQueryEval scorer against the upstream verdicts.

The dataset ships every leaderboard attempt (the code each model wrote and the verdict of
the official open3d-based scorer). Re-scoring a sample of those attempts with our
open3d-free re-implementation tells us how far our pass rates can be compared with the
published leaderboard.
"""

from __future__ import annotations

import random

from .. import paths
from ..cad.sandbox import CadExecutor
from . import registry

CHECKS = ("watertight", "single_component", "bbox_accurate", "volume_passed", "chamfer_passed", "hausdorff_passed")


def validate_against_published(limit: int = 400, workers: int | None = None, seed: int = 0) -> dict:
    import pandas as pd

    bench = registry.get("cqe")
    by_id = {s.id: s for s in bench.load()}
    df = pd.read_parquet(paths.raw_dir("cqe") / "data" / "samples.parquet")
    df = df[df["code"].fillna("").str.strip().ne("") & df["task_id"].isin(by_id)]
    rows = df.to_dict("records")
    rng = random.Random(seed)
    passed = [r for r in rows if r["passed"]]
    failed = [r for r in rows if not r["passed"]]
    rng.shuffle(passed)
    rng.shuffle(failed)
    pick = passed[: limit // 2] + failed[: limit - limit // 2]  # balanced: both error directions are visible

    with CadExecutor(workers=workers) as ex:
        results = ex.run([bench.make_job(by_id[r["task_id"]], r["code"], "protocol") for r in pick], progress="re-score published")

    agree = fp = fn = 0
    per_check = {c: [0, 0] for c in CHECKS}
    disagreements = []
    for r, res in zip(pick, results, strict=True):
        ours = bool(res.get("ok") and (res.get("cqe") or {}).get("passed"))
        theirs = bool(r["passed"])
        if ours == theirs:
            agree += 1
        else:
            fp += ours
            fn += theirs
            if len(disagreements) < 15:
                c = res.get("cqe") or {}
                disagreements.append({"model": r["name"], "task": r["task_id"], "theirs": theirs, "ours": ours, "our_error": res.get("error_type"),
                                      "ours_checks": {k: c.get(k) for k in CHECKS}, "theirs_checks": {k: r.get(k) for k in CHECKS},
                                      "chamfer_mm": [c.get("chamfer_mm"), r.get("chamfer_distance")]})  # fmt: skip
        c = res.get("cqe") or {}
        for k in CHECKS:
            if r.get(k) is not None and c.get(k) is not None:
                per_check[k][0] += int(bool(r[k]) == bool(c[k]))
                per_check[k][1] += 1
    n = len(pick)
    return {
        "n": n,
        "published_pass": len(passed[: limit // 2]),
        "agreement": agree / n,
        "ours_pass_theirs_fail": fp,
        "ours_fail_theirs_pass": fn,
        "per_check_agreement": {k: (a / b if b else None) for k, (a, b) in per_check.items()},
        "disagreements": disagreements,
    }
