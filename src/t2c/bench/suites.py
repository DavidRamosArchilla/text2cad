"""The benchmark suite.

    t2cb       Text2CAD-Bench public prompt sample -- the target benchmark. Ground truth is
               withheld by the authors, so only validity (IR) is scoreable until GT arrives.
    cqe        CadQueryEval -- 25 mm-scale parts, live leaderboard of frontier models.
    cadprompt  CADPrompt -- 200 human-written prompts over DeepCAD objects, expert GT.
    deepcad    500 held-out Text2CAD/DeepCAD descriptions (in-distribution for the SFT data).
    synth      held-out split of the procedural generator (mm-scale, exact GT, never trained on).
"""

from __future__ import annotations

import csv
import io
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from .. import paths, prompts
from ..cad import geometry as G
from ..cad.sandbox import CadExecutor
from .base import Benchmark, Sample


def _stl_to_npz(src: Path, dst: Path) -> tuple[float, int]:
    """Convert a reference STL/OBJ to our mesh format. Returns (open_frac, n_triangles)."""
    import numpy as np
    import trimesh

    tm = trimesh.load(str(src), force="mesh", process=False)
    v, f = np.asarray(tm.vertices, dtype=np.float64), np.asarray(tm.faces, dtype=np.int64)
    G.save_mesh(dst, v, f)
    return G.voxelize(G.normalize_unit_bbox(v), f, 96)[1], len(f)


# =======================================================================================
class Text2CADBench(Benchmark):
    name = "t2cb"
    description = "Text2CAD-Bench public sample: 151 parts x {geo, seq} prompts, L1-L3 (IR only unless GT is supplied)"
    protocol_system = prompts.T2CB_SYSTEM
    default_retries = 3  # paper: up to 3 retries with error feedback
    headline = "ir"

    GT_HELP = (
        "Ground truth for Text2CAD-Bench is not public. If the authors grant access, drop the files as "
        "$T2C_DATA_DIR/raw/t2cb_gt/<id>.step (e.g. L1_1.step; .stp/.stl also accepted) and re-run "
        "`t2c bench prepare t2cb` to enable CD/IoU."
    )

    def prepare(self, workers: int | None = None) -> dict:
        raw = paths.raw_dir("t2cb") / "release.csv"
        rows = list(csv.DictReader(io.StringIO(raw.read_bytes().decode("gb18030"))))  # upstream file is GB18030
        gt_src = paths.raw_dir("t2cb_gt")
        gt_dir = self.dir / "gt"
        jobs, pending = [], []
        samples: list[Sample] = []
        for row in rows:
            pid = row["id"].strip()
            level = pid.split("_")[0]
            gt_rel = None
            for ext in (".step", ".stp", ".STEP", ".stl"):
                cand = gt_src / f"{pid}{ext}"
                if cand.exists():
                    gt_rel = f"gt/{pid}.npz"
                    if ext == ".stl":
                        _stl_to_npz(cand, self.dir / gt_rel)
                    else:
                        code = f"import cadquery as cq\nresult = cq.importers.importStep({str(cand)!r})\n"
                        jobs.append({"code": code, "save_mesh": str(self.dir / gt_rel)})
                        pending.append(pid)
                    break
            for style, col in (("geo", "geo_prompt_en"), ("seq", "pro_prompt_en")):
                samples.append(Sample(id=f"{pid}/{style}", task=row[col].strip(), group=f"{level}/{style}", gt=gt_rel, meta={"level": level, "style": style}))
        failed = []
        if jobs:
            paths.ensure(gt_dir)
            with CadExecutor(workers=workers) as ex:
                for pid, res in zip(pending, ex.run(jobs), strict=True):
                    if not res["ok"]:
                        failed.append(pid)
            for s in samples:
                if s.id.split("/")[0] in failed:
                    s.gt = None
        info = self._write(samples, {"source": "AICAD/Text2CAD-Bench release.csv (30% public sample)", "gt_failed": failed})
        if not info["has_gt"]:
            info["note"] = self.GT_HELP
        return info


# =======================================================================================
class CadQueryEval(Benchmark):
    name = "cqe"
    description = "CadQueryEval: 25 mm-scale parts, pass/fail on six geometric checks; frontier leaderboard available"
    protocol_system = prompts.CQE_SYSTEM
    default_retries = 0  # upstream: each model attempts each task once
    headline = "pass_rate"

    def prepare(self, workers: int | None = None) -> dict:
        import pandas as pd

        raw = paths.raw_dir("cqe")
        tasks = pd.read_parquet(raw / "data" / "tasks.parquet")
        samples = []
        for t in tasks.sort_values("task_id", key=lambda s: s.str.extract(r"(\d+)")[0].astype(int)).to_dict("records"):
            refs = []
            alts = t.get("alternate_reference_stls")
            for rel in [t["reference_stl"], *([] if alts is None else [str(a) for a in alts])]:
                name = Path(rel).stem
                _stl_to_npz(raw / "reference" / f"{name}.stl", self.dir / "gt" / f"{name}.npz")
                refs.append(f"gt/{name}.npz")
            bb = [] if t.get("bounding_box") is None else [float(x) for x in t["bounding_box"]]
            n_comp = 1 if pd.isna(t.get("expected_components")) else int(t["expected_components"])
            task = prompts.CQE_TASK_TEMPLATE.format(
                description=str(t["description"]).strip(),
                bounding_box_str=f"{bb[0]:.1f} x {bb[1]:.1f} x {bb[2]:.1f} mm" if len(bb) == 3 else "Not specified",
                expected_components=n_comp,
            )
            samples.append(Sample(id=t["task_id"], task=task, gt=refs[0], meta={"refs": refs, "expected_components": n_comp, "operations": 0 if pd.isna(t.get("operations")) else int(t["operations"])}))
        published = self._published(pd.read_parquet(raw / "data" / "samples.parquet"))
        paths.ensure(self.dir)
        (self.dir / "published.json").write_text(json.dumps(published, indent=1))
        return self._write(samples, {"source": "drwahl/cadqueryeval", "published_models": len(published)})

    @staticmethod
    def _published(df) -> list[dict]:
        """Leaderboard rows computed from the per-attempt table shipped with the dataset."""
        out = []
        for (name, model), g in df.groupby(["name", "model"]):
            out.append(
                {
                    "name": str(name),
                    "model_id": str(model),
                    "n": int(len(g)),
                    "pass_rate": float(g["passed"].mean()),
                    "ir": float(1.0 - g["stl_created"].fillna(False).astype(bool).mean()),
                    "created": str(g["created"].max())[:10],
                }
            )
        return sorted(out, key=lambda r: -r["pass_rate"])

    def published(self) -> list[dict]:
        p = self.dir / "published.json"
        return json.loads(p.read_text()) if p.exists() else []

    def make_job(self, sample: Sample, code: str | None, style: str) -> dict:
        job = super().make_job(sample, code, style)
        job["cqe"] = {"refs": [str(self.dir / r) for r in sample.meta["refs"]], "expected_components": sample.meta["expected_components"]}
        if style == "protocol":
            job["output_stl"] = "output.stl"  # upstream contract: the script writes the STL itself
        else:
            job["tess"] = {"abs": 0.1, "ang": 0.1}  # what cq.exporters.export() would have used
        return job

    def row_extra(self, sample: Sample, result: dict) -> dict:
        c = result.get("cqe") or {}
        keys = ("passed", "watertight", "single_component", "bbox_accurate", "volume_passed", "chamfer_passed", "hausdorff_passed", "chamfer_mm", "hausdorff95_mm")
        return {f"cqe_{k}": c.get(k) for k in keys}

    def aggregate_extra(self, rows: list[dict]) -> dict:
        n = len(rows)
        out = {"pass_rate": sum(1 for r in rows if r.get("cqe_passed")) / n}
        for k in ("watertight", "single_component", "bbox_accurate", "volume_passed", "chamfer_passed", "hausdorff_passed"):
            out[f"{k}_rate"] = sum(1 for r in rows if r.get(f"cqe_{k}")) / n
        return out


# =======================================================================================
class CADPrompt(Benchmark):
    name = "cadprompt"
    description = "CADPrompt: 200 DeepCAD objects, human-written prompts ('measured' = with dimensions, 'abstract' = without)"

    def prepare(self, workers: int | None = None) -> dict:
        from ..cad.codeutil import normalize_reference_script

        base = paths.raw_dir("cadprompt") / "CADPrompt"
        samples, regen, jobs = [], [], []
        gt_source = Counter()
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            uid = d.name
            gt_rel = f"gt/{uid}.npz"
            open_frac, _ = _stl_to_npz(d / "Ground_Truth.stl", self.dir / gt_rel)
            source = "stl"
            if open_frac > 0.005:  # published mesh has holes: rebuild it from the expert script
                code = normalize_reference_script((d / "Python_Code.py").read_text(errors="replace"))
                if code:
                    jobs.append({"code": code, "save_mesh": str(self.dir / gt_rel), "self_check": 96})
                    regen.append(uid)
                    source = "script"
            gt_source[source] += 1
            for group, fn in (("measured", "Natural_Language_Descriptions_Prompt_with_specific_measurements.txt"), ("abstract", "Natural_Language_Descriptions_Prompt.txt")):
                text = (d / fn).read_text(errors="replace").strip()
                samples.append(Sample(id=f"{uid}/{group}", task=text, group=group, gt=gt_rel, meta={"uid": uid, "gt_open_frac": open_frac}))
        unusable = []
        if jobs:
            with CadExecutor(workers=workers) as ex:
                for uid, res in zip(regen, ex.run(jobs), strict=True):
                    if not res["ok"] or res.get("open_frac", 0) > 0.005:
                        unusable.append(uid)
        samples = [s for s in samples if s.meta["uid"] not in unusable]
        return self._write(samples, {"source": "Kamel773/CAD_Code_Generation", "gt_source": dict(gt_source), "dropped_non_watertight_gt": unusable})

    def uids(self) -> set[str]:
        return {s.meta["uid"] for s in self.load()}


# =======================================================================================
class DeepCADText(Benchmark):
    name = "deepcad"
    description = "500 held-out Text2CAD/DeepCAD descriptions (test split of Text-to-CadQuery); GT = executed reference script"
    N = 500
    SEED = 20261115

    def prepare(self, workers: int | None = None) -> dict:
        from ..data.prepare import read_split

        rows = sorted(read_split("t2cq", "test"), key=lambda r: r["id"])
        rows = [r for r in rows if r["uid"]]
        random.Random(self.SEED).shuffle(rows)
        rows = rows[: self.N]
        gt_dir = self.dir / "gt"
        if gt_dir.exists():
            shutil.rmtree(gt_dir)
        jobs = [{"code": r["code"], "save_mesh": str(gt_dir / f"{r['uid']}.npz")} for r in rows]
        with CadExecutor(workers=workers) as ex:
            results = ex.run(jobs, progress="deepcad GT")
        samples = [
            Sample(id=r["uid"], task=r["text"], gt=f"gt/{r['uid']}.npz", meta={"uid": r["uid"], "reference_code": r["code"]})
            for r, res in zip(rows, results, strict=True)
            if res["ok"]
        ]
        return self._write(samples, {"source": "processed/t2cq/test.jsonl", "seed": self.SEED})

    def uids(self) -> set[str]:
        return {s.meta["uid"] for s in self.load()}


# =======================================================================================
class SynthBench(Benchmark):
    name = "synth"
    description = "Held-out procedural parts in millimetres (plates, flanges, shafts, brackets, ...): exact GT, geo + seq prompts"

    def prepare(self, workers: int | None = None) -> dict:
        from ..data.prepare import read_split

        rows = read_split("synth", "test")
        gt_dir = self.dir / "gt"
        if gt_dir.exists():
            shutil.rmtree(gt_dir)
        jobs = [{"code": r["code"], "save_mesh": str(gt_dir / f"{r['uid']}.npz")} for r in rows]
        with CadExecutor(workers=workers) as ex:
            results = ex.run(jobs, progress="synth GT")
        samples = [
            Sample(id=f"{r['uid']}/{r['style']}", task=r["text"], group=f"{r['family']}", gt=f"gt/{r['uid']}.npz", meta={"uid": r["uid"], "style": r["style"], "family": r["family"]})
            for r, res in zip(rows, results, strict=True)
            if res["ok"]
        ]
        return self._write(samples, {"source": "processed/synth/test.jsonl"})
