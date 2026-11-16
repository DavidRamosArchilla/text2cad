"""Mesh utilities used by the verifier: normalisation, solid voxelisation, surface sampling.

Pure numpy/scipy on purpose -- no open3d / embree -- so the scorer installs anywhere and
produces bit-identical numbers on every cluster.

A mesh is a pair ``(v, f)``: ``v`` float array (V, 3), ``f`` int array (F, 3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Sub-voxel offset applied to the sampling lattice so that rays never run exactly along a
# mesh edge or through a vertex (CAD parts are full of features at "nice" coordinates).
_JITTER = np.array([3.7139e-6, 6.1803e-6])
_MAX_CANDIDATES = 4_000_000  # per chunk, bounds peak memory of the rasteriser


def save_mesh(path: str | Path, v: np.ndarray, f: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, v=np.asarray(v, dtype=np.float32), f=np.asarray(f, dtype=np.int32))
    tmp.replace(path)  # atomic: concurrent readers never see a half-written file


def load_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        return z["v"].astype(np.float64), z["f"].astype(np.int64)


def bbox(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return v.min(axis=0), v.max(axis=0)


def normalize_unit_bbox(v: np.ndarray) -> np.ndarray:
    """Translate + uniformly scale so the bounding box is centred in [0,1]^3 and its
    longest side has length 1 (the 'unit bounding box' normalisation of Text2CAD-Bench)."""
    lo, hi = bbox(v)
    scale = float((hi - lo).max())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("degenerate bounding box")
    return (v - (lo + hi) / 2.0) / scale + 0.5


def mesh_volume(v: np.ndarray, f: np.ndarray) -> float:
    """Signed volume (divergence theorem). Positive for outward-oriented closed meshes."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def mesh_area(v: np.ndarray, f: np.ndarray) -> float:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def sample_surface(v: np.ndarray, f: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Uniform (area-weighted) surface samples, deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = area.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("mesh has zero surface area")
    idx = rng.choice(len(f), size=n, p=area / total)
    r1 = np.sqrt(rng.random(n))
    r2 = rng.random(n)
    w0, w1, w2 = 1.0 - r1, r1 * (1.0 - r2), r1 * r2
    return w0[:, None] * a[idx] + w1[:, None] * b[idx] + w2[:, None] * c[idx]


def voxelize(v: np.ndarray, f: np.ndarray, res: int) -> tuple[np.ndarray, float]:
    """Solid voxelisation of a closed mesh whose vertices live in [0,1]^3.

    One ray per (x, y) voxel column is cast along +z; every crossing of the surface
    toggles inside/outside (even-odd rule), so internal cavities stay empty and the
    result does not depend on triangle orientation or on vertices being shared.

    Returns ``(occ, open_frac)``: ``occ`` is a bool array (res, res, res) indexed
    [x, y, z]; ``open_frac`` is the fraction of hit columns with an odd number of
    crossings -- 0 for a watertight mesh, >0 signals holes in the surface.
    """
    tri = v[f]  # (T, 3, 3)
    xy = tri[:, :, :2] * res - 0.5 - _JITTER  # pixel coordinates: voxel centres at integers
    z = tri[:, :, 2]
    lo = np.clip(np.ceil(xy.min(axis=1)), 0, res).astype(np.int64)
    hi = np.clip(np.floor(xy.max(axis=1)), -1, res - 1).astype(np.int64)
    w = np.maximum(hi[:, 0] - lo[:, 0] + 1, 0)
    h = np.maximum(hi[:, 1] - lo[:, 1] + 1, 0)
    n = w * h
    keep = np.nonzero(n > 0)[0]

    hits: list[np.ndarray] = []
    start = 0
    csum = np.cumsum(n[keep])
    while start < len(keep):
        base = csum[start - 1] if start else 0
        stop = int(np.searchsorted(csum, base + _MAX_CANDIDATES, side="right"))
        stop = max(stop, start + 1)
        t = keep[start:stop]
        nt = n[t]
        tid = np.repeat(np.arange(len(t)), nt)
        k = np.arange(nt.sum()) - np.repeat(np.cumsum(nt) - nt, nt)
        wt = w[t][tid]
        px = lo[t, 0][tid] + k % wt
        py = lo[t, 1][tid] + k // wt
        a, b, c = xy[t, 0][tid], xy[t, 1][tid], xy[t, 2][tid]
        d = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        ok = np.abs(d) > 1e-14  # triangles seen edge-on from +z are never crossed by a ray
        d = np.where(ok, d, 1.0)
        w0 = ((b[:, 0] - px) * (c[:, 1] - py) - (b[:, 1] - py) * (c[:, 0] - px)) / d
        w1 = ((c[:, 0] - px) * (a[:, 1] - py) - (c[:, 1] - py) * (a[:, 0] - px)) / d
        w2 = 1.0 - w0 - w1
        inside = ok & (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        tz = z[t][tid]
        zz = w0 * tz[:, 0] + w1 * tz[:, 1] + w2 * tz[:, 2]
        zi = np.clip(np.ceil(zz[inside] * res - 0.5), 0, res).astype(np.int64)
        hits.append((px[inside] * res + py[inside]) * (res + 1) + zi)
        start = stop

    if not hits:
        return np.zeros((res, res, res), dtype=bool), 0.0
    flat = np.concatenate(hits)
    toggles = (np.bincount(flat, minlength=res * res * (res + 1)) & 1).astype(np.uint8)
    toggles = toggles.reshape(res, res, res + 1)
    parity = np.bitwise_xor.accumulate(toggles, axis=2)
    occ = parity[:, :, :res].astype(bool)
    hit_cols = np.bincount(flat // (res + 1), minlength=res * res) > 0
    open_cols = parity[:, :, res].reshape(-1).astype(bool)  # still "inside" after the last crossing
    open_frac = float(open_cols.sum() / max(int(hit_cols.sum()), 1))
    if open_frac > 0:
        occ[open_cols.reshape(res, res)] = False  # drop corrupted columns instead of smearing them
    return occ, open_frac


def voxel_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)
