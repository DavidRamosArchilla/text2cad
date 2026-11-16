"""Geometric metrics.

``protocol_metrics``  Text2CAD-Bench protocol (arXiv 2605.18430, Sec. 4): both shapes are
                      normalised to a unit bounding box; Chamfer Distance on 30k surface
                      points (sum of the two mean nearest-neighbour distances, not squared);
                      IoU on a 256^3 solid voxelisation. No rotation alignment.
``cqe_checks``        CadQueryEval pass/fail checks (absolute millimetres, after rigid
                      alignment). Re-implemented without open3d; see the docstring there.

The official Text2CAD-Bench evaluation code is not public, so ``protocol_metrics`` is a
re-implementation from the paper's description, not a copy of the authors' scorer.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial import cKDTree

from . import geometry as G

# ---------------------------------------------------------------------------------------
# Text2CAD-Bench protocol
# ---------------------------------------------------------------------------------------


def chamfer(p: np.ndarray, q: np.ndarray) -> float:
    """CD(P,Q) = mean_p d(p,Q) + mean_q d(q,P), Euclidean (not squared)."""
    d_pq, _ = cKDTree(q).query(p)
    d_qp, _ = cKDTree(p).query(q)
    return float(d_pq.mean() + d_qp.mean())


def protocol_metrics(
    pred: tuple[np.ndarray, np.ndarray],
    gt: tuple[np.ndarray, np.ndarray],
    res: int = 256,
    n_points: int = 30_000,
) -> dict:
    pv, pf = pred
    gv, gf = gt
    pn = G.normalize_unit_bbox(pv)
    gn = G.normalize_unit_bbox(gv)
    out: dict = {}
    if n_points:
        # Independent seeds: a perfect prediction scores the sampling-noise floor, not 0.
        out["cd"] = chamfer(G.sample_surface(pn, pf, n_points, seed=0), G.sample_surface(gn, gf, n_points, seed=1))
    if res:
        po, p_open = G.voxelize(pn, pf, res)
        go, g_open = G.voxelize(gn, gf, res)
        out["iou"] = G.voxel_iou(po, go)
        out["open_frac"] = p_open
        out["gt_open_frac"] = g_open
    # Scale / proportions are invisible after unit-bbox normalisation; keep them separately.
    pe = pv.max(0) - pv.min(0)
    ge = gv.max(0) - gv.min(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((pe > 0) & (ge > 0), pe / ge, np.nan)
    out["dim_err"] = float(np.nanmean(np.abs(np.log(ratio)))) if np.isfinite(ratio).any() else float("inf")
    return out


# ---------------------------------------------------------------------------------------
# Rigid alignment (replacement for open3d RANSAC + ICP)
# ---------------------------------------------------------------------------------------


def _cube_rotations() -> list[np.ndarray]:
    rots = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs, strict=True)):
                m[i, p] = s
            if np.linalg.det(m) > 0:
                rots.append(m)
    return rots  # the 24 proper rotations of the cube


_CUBE_ROTS = _cube_rotations()


def _pca_frame(x: np.ndarray) -> np.ndarray:
    _, _, vt = np.linalg.svd(x - x.mean(0), full_matrices=False)
    if np.linalg.det(vt) < 0:
        vt[2] *= -1
    return vt  # rows = principal axes


def _kabsch(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rigid transform (R, t) with R @ a_i + t ~= b_i."""
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, cb - r @ ca


def _icp(src, dst, tree, r, t, max_corr, iters) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(iters):
        cur = src @ r.T + t
        dist, idx = tree.query(cur)
        m = dist <= max_corr
        if m.sum() < 16:
            break
        dr, dt = _kabsch(cur[m], dst[idx[m]])
        r, t = dr @ r, dr @ t + dt
        if np.abs(dr - np.eye(3)).max() < 1e-7 and np.abs(dt).max() < 1e-7:
            break
    return r, t


def rigid_align(src: np.ndarray, dst: np.ndarray, final_corr: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Rigid transform (R, t) mapping point cloud ``src`` onto ``dst``.

    Global stage: centroids are matched and every one of the 24 axis-aligned rotations plus
    the 4 principal-axis alignments is tried (CAD parts differ from the reference by axis
    swaps far more often than by arbitrary rotations); each candidate gets a short coarse
    ICP on a subsample. Local stage: full ICP with a correspondence radius shrinking to
    ``final_corr``.
    """
    rng = np.random.default_rng(seed)
    s = src[rng.choice(len(src), min(len(src), 2000), replace=False)]
    d = dst[rng.choice(len(dst), min(len(dst), 4000), replace=False)]
    tree_d = cKDTree(d)
    cs, cd = src.mean(0), dst.mean(0)
    diag = float(np.linalg.norm(dst.max(0) - dst.min(0)))

    cands = list(_CUBE_ROTS)
    fs, fd = _pca_frame(s), _pca_frame(d)
    for sx, sy in itertools.product((1, -1), repeat=2):
        flip = np.diag([sx, sy, sx * sy])
        cands.append(fd.T @ flip @ fs)

    best, best_score = None, np.inf
    for r0 in cands:
        r, t = _icp(s, d, tree_d, r0, cd - r0 @ cs, max_corr=0.15 * diag, iters=12)
        dist, _ = tree_d.query(s @ r.T + t)
        score = float(np.sort(dist)[: int(0.9 * len(dist))].mean())
        if score < best_score:
            best, best_score = (r, t), score
    r, t = best
    tree_full = cKDTree(dst)
    fine = src[rng.choice(len(src), min(len(src), 10_000), replace=False)]
    for corr in (max(0.05 * diag, final_corr), max(0.01 * diag, final_corr), final_corr):
        r, t = _icp(fine, dst, tree_full, r, t, max_corr=corr, iters=40)
    return r, t


# ---------------------------------------------------------------------------------------
# CadQueryEval checks
# ---------------------------------------------------------------------------------------

CQE_BBOX_TOL_MM = 1.0
CQE_CHAMFER_MM = 1.0
CQE_HAUSDORFF_MM = 1.0
CQE_VOLUME_PCT = 2.0
CQE_POINTS = 50_000
CQE_ICP_CORR_MM = 1.5


def _clean_trimesh(v: np.ndarray, f: np.ndarray):
    import trimesh

    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    m.merge_vertices(digits_vertex=4)  # ~1e-4 mm, as upstream's merge_close_vertices(0.0001)
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    return m


def _n_components(m) -> int:
    import trimesh

    if len(m.faces) == 0:
        return 0
    return len(trimesh.graph.connected_components(m.face_adjacency, nodes=np.arange(len(m.faces)), min_len=1))


def cqe_checks(
    pred: tuple[np.ndarray, np.ndarray],
    ref: tuple[np.ndarray, np.ndarray],
    expected_components: int = 1,
) -> dict:
    """CadQueryEval's six binary checks against one reference mesh (units: mm).

    Mirrors github.com/danwahl/cadqueryeval (MIT) ``geometry.perform_geometry_checks``:
    watertight, component count, volume within 2 %, and -- after rigid alignment --
    Chamfer <= 1 mm, 95th-percentile Hausdorff <= 1 mm, sorted bounding-box extents
    within 1 mm. The upstream scorer aligns with open3d FPFH-RANSAC + ICP; here the
    alignment is ``rigid_align`` (deterministic, no open3d). Agreement with the published
    per-sample verdicts is measured by ``t2c bench validate-cqe``.
    """
    out: dict = {"errors": []}
    pm, rm = _clean_trimesh(*pred), _clean_trimesh(*ref)
    if len(pm.faces) == 0:
        out["errors"].append("mesh has no triangles")
        out["passed"] = False
        return out
    out["watertight"] = bool(pm.is_watertight)
    out["n_components"] = _n_components(pm)
    out["single_component"] = out["n_components"] == expected_components
    pvol, rvol = float(pm.volume), float(rm.volume)
    out["volume_ratio"] = pvol / rvol if rvol else None
    out["volume_passed"] = bool(
        pm.is_watertight and rm.is_watertight and rvol != 0 and abs(pvol - rvol) / abs(rvol) * 100 <= CQE_VOLUME_PCT
    )

    pp = G.sample_surface(pm.vertices, pm.faces, CQE_POINTS, seed=0)
    rp = G.sample_surface(rm.vertices, rm.faces, CQE_POINTS, seed=1)
    r, t = rigid_align(pp, rp, final_corr=CQE_ICP_CORR_MM)
    pa = pp @ r.T + t
    d_pr, _ = cKDTree(rp).query(pa)
    d_rp, _ = cKDTree(pa).query(rp)
    out["chamfer_mm"] = float((d_pr.mean() + d_rp.mean()) / 2.0)
    out["hausdorff95_mm"] = float(np.percentile(np.concatenate([d_pr, d_rp]), 95))
    va = np.asarray(pm.vertices) @ r.T + t
    pe = np.sort(va.max(0) - va.min(0))
    re_ = np.sort(np.asarray(rm.vertices).max(0) - np.asarray(rm.vertices).min(0))
    out["bbox_accurate"] = bool((np.abs(pe - re_) <= CQE_BBOX_TOL_MM).all())
    out["chamfer_passed"] = out["chamfer_mm"] <= CQE_CHAMFER_MM
    out["hausdorff_passed"] = out["hausdorff95_mm"] <= CQE_HAUSDORFF_MM
    out["passed"] = all(
        out[k]
        for k in ("watertight", "single_component", "bbox_accurate", "volume_passed", "chamfer_passed", "hausdorff_passed")
    )
    return out


def cqe_best(pred, refs: list, expected_components: int = 1) -> dict:
    """Upstream rule: passing any listed reference passes; failures are reported against the first."""
    first = None
    for i, ref in enumerate(refs):
        res = cqe_checks(pred, ref, expected_components)
        res["matched_ref"] = i
        if first is None:
            first = res
        if res["passed"]:
            return res
    return first
