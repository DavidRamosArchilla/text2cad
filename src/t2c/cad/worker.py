"""Code that runs *inside* a throw-away child process: execute one CadQuery script,
turn the result into a mesh and score it.

This module is pre-imported by the zygote (see zygote.py), so every child starts
with cadquery / OCP / scipy already loaded and pays ~10 ms instead of ~2 s.

Stages reached (used for reward shaping):
    0  nothing runnable (no code / syntax error / rejected by the static check)
    1  script parsed but raised or timed out
    2  script ran but did not leave a usable solid in `result`
    3  valid solid geometry
"""

from __future__ import annotations

import os
import time
import traceback

import numpy as np

try:  # heavy import, done once in the fork server
    import cadquery as cq
except Exception as _e:  # pragma: no cover - surfaced by `t2c doctor`
    cq = None
    _CQ_IMPORT_ERROR = repr(_e)
else:
    _CQ_IMPORT_ERROR = None

from . import geometry as G
from . import metrics as M

GENERATED_NAME = "generated.py"
MAX_ERR_CHARS = 1500
DEFAULT_TESS = {"rel": 5e-4, "ang": 0.2}


class CadError(Exception):
    def __init__(self, error_type: str, message: str, stage: int):
        super().__init__(message)
        self.error_type = error_type
        self.stage = stage


def _format_exception(e: BaseException) -> str:
    """Traceback restricted to frames of the generated script -- what a model needs to fix it."""
    tb = traceback.extract_tb(e.__traceback__)
    lines = [f"  line {fr.lineno}: {fr.line}" for fr in tb if fr.filename == GENERATED_NAME and fr.line]
    head = f"{type(e).__name__}: {e}".strip()
    msg = head if not lines else head + "\n" + "\n".join(lines[-4:])
    return msg[:MAX_ERR_CHARS]


def _solids_of(obj) -> list:
    """Every solid reachable from whatever the script left in `result`."""
    if isinstance(obj, cq.Assembly):
        obj = obj.toCompound()
    if isinstance(obj, cq.Workplane):
        vals = obj.vals()
    elif isinstance(obj, cq.Shape):
        vals = [obj]
    elif isinstance(obj, (list, tuple)):
        vals = []
        for o in obj:
            vals.extend(o.vals() if isinstance(o, cq.Workplane) else [o])
    else:
        raise CadError("bad_result_type", f"`result` is a {type(obj).__name__}, expected a cadquery Workplane or Shape", 2)
    solids = []
    for s in vals:
        if isinstance(s, cq.Shape):
            solids.extend(s.Solids())
    return solids


def _tessellate(shape, tess: dict) -> tuple[np.ndarray, np.ndarray]:
    bb = shape.BoundingBox()
    extent = max(bb.xlen, bb.ylen, bb.zlen)
    if not np.isfinite(extent) or extent <= 0:
        raise CadError("degenerate", "the solid has an empty bounding box", 2)
    tol = tess["abs"] if "abs" in tess else tess.get("rel", DEFAULT_TESS["rel"]) * extent
    verts, tris = shape.tessellate(tol, tess.get("ang", DEFAULT_TESS["ang"]))
    v = np.array([(p.x, p.y, p.z) for p in verts], dtype=np.float64).reshape(-1, 3)
    f = np.array(tris, dtype=np.int64).reshape(-1, 3)
    if len(f) == 0 or not np.isfinite(v).all():
        raise CadError("degenerate", "tessellation produced no triangles", 2)
    return v, f


def execute_script(code: str) -> dict:
    """Run a script and return its global namespace."""
    try:
        compiled = compile(code, GENERATED_NAME, "exec")
    except SyntaxError as e:
        raise CadError("syntax_error", f"SyntaxError: {e.msg} (line {e.lineno})", 0) from None
    ns: dict = {"__name__": "__main__"}
    # Make tracebacks show source lines of the generated script.
    import linecache

    linecache.cache[GENERATED_NAME] = (len(code), None, code.splitlines(True), GENERATED_NAME)
    try:
        exec(compiled, ns)
    except CadError:
        raise
    except SystemExit as e:
        raise CadError("runtime_error", f"SystemExit({e.code}) raised by the script", 1) from None
    except BaseException as e:  # noqa: BLE001 - generated code can raise anything
        raise CadError("runtime_error", _format_exception(e), 1) from None
    return ns


def shape_from_namespace(ns: dict):
    if "result" not in ns:
        raise CadError("no_result", "the script must assign the final solid to a variable named `result`", 2)
    try:
        solids = _solids_of(ns["result"])
    except CadError:
        raise
    except Exception as e:  # noqa: BLE001
        raise CadError("bad_result_type", f"could not read `result`: {type(e).__name__}: {e}"[:MAX_ERR_CHARS], 2) from None
    if not solids:
        raise CadError("no_solid", "`result` contains no solid body (did the script end on a face/edge selection or a 2D sketch?)", 2)
    shape = solids[0] if len(solids) == 1 else cq.Compound.makeCompound(solids)
    return shape, len(solids)


def run_job(job: dict) -> dict:
    """Execute + mesh + score. Never raises: failures are reported in the returned dict."""
    t0 = time.perf_counter()
    out: dict = {"ok": False, "stage": 0, "error_type": None, "error": None}
    if cq is None:
        out.update(error_type="harness_error", error=f"cadquery import failed: {_CQ_IMPORT_ERROR}")
        return out
    try:
        ns = execute_script(job["code"])
        out["t_exec"] = round(time.perf_counter() - t0, 3)
        out["stage"] = 2
        stl = job.get("output_stl")
        if stl:
            # CadQueryEval protocol: the script itself must write the STL.
            if not os.path.isfile(stl):
                raise CadError("no_output_file", f"the script ran but did not create '{stl}'", 2)
            import trimesh

            tm = trimesh.load(stl, force="mesh", process=False)
            v, f = np.asarray(tm.vertices, dtype=np.float64), np.asarray(tm.faces, dtype=np.int64)
            if len(f) == 0:
                raise CadError("degenerate", f"'{stl}' contains no triangles", 2)
            out["n_solids"] = None
            vol = abs(G.mesh_volume(v, f))
        else:
            shape, n_solids = shape_from_namespace(ns)
            out["n_solids"] = n_solids
            try:
                vol = abs(float(shape.Volume()))
                out["brep_valid"] = bool(shape.isValid())
                v, f = _tessellate(shape, job.get("tess") or DEFAULT_TESS)
            except CadError:
                raise
            except Exception as e:  # noqa: BLE001 - OCC can throw on malformed solids
                raise CadError("degenerate", f"could not mesh the solid: {type(e).__name__}: {e}"[:MAX_ERR_CHARS], 2) from None
        ext = v.max(0) - v.min(0)
        if not np.isfinite(vol) or vol <= 1e-12 * max(float(ext.max()), 1e-12) ** 3:
            raise CadError("degenerate", "the solid has zero volume", 2)
        out.update(volume=vol, bbox=[float(x) for x in ext], n_tris=int(len(f)))
        out["stage"] = 3
        out["ok"] = True
        if job.get("save_mesh"):
            G.save_mesh(job["save_mesh"], v, f)
        if job.get("self_check"):
            # Reference scripts: a surface with holes cannot serve as ground truth.
            out["open_frac"] = G.voxelize(G.normalize_unit_bbox(v), f, int(job["self_check"]))[1]
        if job.get("gt"):
            gv, gf = G.load_mesh(job["gt"])
            out.update(M.protocol_metrics((v, f), (gv, gf), res=job.get("res", 256), n_points=job.get("n_points", 30_000)))
        if job.get("cqe"):
            refs = [G.load_mesh(p) for p in job["cqe"]["refs"]]
            out["cqe"] = M.cqe_best((v, f), refs, job["cqe"].get("expected_components", 1))
    except CadError as e:
        out.update(ok=False, stage=e.stage, error_type=e.error_type, error=str(e))
    except MemoryError:
        out.update(ok=False, stage=1, error_type="memory", error="MemoryError: the script exceeded the memory limit")
    except Exception as e:  # noqa: BLE001
        # A failure of *our* scoring code (corrupt ground truth, bug), not of the model:
        # keep `ok`/`stage` as they were and flag it so aggregation can exclude and report it.
        out.update(error_type="harness_error", error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}")
    out["t_total"] = round(time.perf_counter() - t0, 3)
    return out



def child_entry(job: dict, wfd: int, workdir: str, mem_bytes: int, cpu_seconds: int, zygote_info: dict) -> None:
    """Body of a forked child: apply limits, run the job, write one pickled result to ``wfd``."""
    import pickle
    import resource
    import sys

    try:
        os.setsid()  # own process group: the zygote can kill us and anything we spawn
    except OSError:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))  # OCC segfaults must not litter core files
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 29, 1 << 29))
        if cpu_seconds:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))
        if mem_bytes:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except (ValueError, OSError):
        pass
    try:
        os.chdir(workdir)
        devnull = os.open(os.devnull, os.O_RDWR)
        for fd in (0, 1, 2):
            os.dup2(devnull, fd)
        sys.stdout = sys.stderr = open(os.devnull, "w")  # noqa: SIM115
        if job.get("probe"):
            res = dict(zygote_info, cadquery=getattr(cq, "__version__", None), cq_import_error=_CQ_IMPORT_ERROR)
        else:
            res = run_job(job)
    except BaseException as e:  # noqa: BLE001
        res = {"ok": False, "stage": 1, "error_type": "harness_error", "error": f"{type(e).__name__}: {e}"}
    try:
        payload = pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL)
        view = memoryview(payload)
        while view:
            view = view[os.write(wfd, view) :]
    finally:
        os._exit(0)  # skip interpreter teardown of the forked OCC state
