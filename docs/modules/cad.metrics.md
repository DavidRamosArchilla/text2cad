---
tags: [module, verifier]
aliases: [metrics.py, protocol_metrics, cqe_checks, rigid_align]
---
# cad.metrics
**File:** `src/t2c/cad/metrics.py` · **Libraries:** [[Libraries#numpy|numpy]],
[[Libraries#scipy|scipy]] (`cKDTree`), [[Libraries#trimesh|trimesh]]
**Concepts:** [[Comparing 3D shapes]] · [[Rigid alignment]] · [[Metrics]]

Compares a predicted mesh with a ground-truth mesh. Two families of metrics, because the
benchmarks define them differently.

## 1. `protocol_metrics(pred, gt, res=256, n_points=30000)` — Text2CAD-Bench style
Used for every benchmark and for the RL reward.
1. Normalise both meshes to a unit bounding box ([[cad.geometry]]).
2. **Chamfer distance** `cd`: sample `n_points` on each surface (different seeds), for each
   point find the nearest point of the other cloud (KD-tree), average both directions and
   **add** them (the paper's definition; not squared). Because the two samples are
   independent, a perfect answer scores ≈ 0.01 (10 after the ×1000 used in reports), not 0.
3. **IoU**: voxelise both at `res³`, intersection / union.
4. **dim_err**: mean over x, y, z of |log(predicted extent / true extent)|, on the *real*
   sizes. Normalisation hides a part that is 10× too big or lies on the wrong axis; this
   does not.
5. `open_frac` / `gt_open_frac`: holes in the predicted / reference surface.

No rotation alignment: a correct part built along the wrong axis gets a low IoU. That
matches the paper, and it is why prompts state axes.

## 2. `cqe_checks(pred, ref, expected_components)` — CadQueryEval style
Absolute millimetres, after aligning the part to the reference. Re-implementation of the
official open3d scorer ([[Libraries#trimesh|why no open3d]]):

| Check | Passes when |
|---|---|
| watertight | every edge shared by exactly two triangles (after merging duplicate vertices) |
| single component | number of connected pieces = expected |
| volume | within 2 % of the reference |
| Chamfer | ≤ 1 mm (average of the two directions, as upstream) |
| Hausdorff 95 % | 95th percentile of all point distances ≤ 1 mm |
| bounding box | sorted extents within 1 mm |

`passed` = all six. `cqe_best(pred, refs)` tries alternative reference meshes (some tasks
accept two readings) and passes if any one passes. Agreement with the official verdicts:
300 / 300 on published attempts ([[bench.cqe]]).

## 3. `rigid_align(src, dst, final_corr)` — put two point clouds on top of each other
Needed for the CadQueryEval checks (a correct part may be modelled on another plane).
1. Match centroids and try **28 starting rotations**: the 24 rotations that map the axes
   onto axes (CAD parts usually differ by axis swaps) + 4 principal-axis alignments.
2. For each, a short coarse **ICP** on a subsample; keep the best (trimmed mean distance).
3. Refine with ICP on 10 000 points and a shrinking correspondence radius
   (5 % → 1 % of the size → 1.5 mm). Each ICP step solves the best rotation with the
   **Kabsch** algorithm (SVD).

Deterministic (seeded), unlike open3d's RANSAC.

**Used by:** [[cad.sandbox]] (worker calls `protocol_metrics` and `cqe_best`).
**Tested in:** [[test_geometry]] (IoU known values, scale invariance, orientation
sensitivity, alignment recovery, CadQueryEval pass/fail).
