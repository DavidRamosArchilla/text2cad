---
tags: [module, verifier]
aliases: [geometry.py, voxelize]
---
# cad.geometry
**File:** `src/t2c/cad/geometry.py` · **Libraries:** [[Libraries#numpy|numpy]] only
**Concepts:** [[Comparing 3D shapes]]

Low-level mesh utilities. A mesh is two numpy arrays: `v` (vertex coordinates, N×3) and
`f` (triangles as vertex indices, M×3). Pure numpy on purpose: identical numbers on every
machine, nothing to compile.

| Function | What it does |
|---|---|
| `save_mesh` / `load_mesh` | store a mesh as compressed `.npz` (written atomically: readers never see half a file) |
| `normalize_unit_bbox(v)` | move and scale uniformly so the bounding box is centred in [0,1]³ with longest side 1 — the Text2CAD-Bench normalisation |
| `mesh_volume(v, f)` | signed volume by the divergence theorem (sum of tetrahedra) |
| `mesh_area(v, f)` | surface area |
| `sample_surface(v, f, n, seed)` | `n` random points uniformly on the surface (area-weighted), deterministic per seed |
| `voxelize(v, f, res)` | solid occupancy grid `res³` + `open_frac` (see below) |
| `voxel_iou(a, b)` | intersection / union of two grids |

## How `voxelize` works
For every (x, y) column of the grid, cast a ray along +z and find where it crosses the
surface. Each crossing toggles inside/outside (the **even–odd rule**), so internal
cavities stay empty and the triangle orientation does not matter. Rays are shifted by a
tiny offset so they never run exactly along an edge (CAD parts have many features at
"round" coordinates). Work is done in chunks to bound memory.

If a column ends "inside", the surface had a hole there: the column is emptied and counted
in **`open_frac`** (fraction of such columns). `open_frac = 0` means watertight; data
preparation drops reference parts with `open_frac > 0.005`.

Speed (one part): 64³ ≈ 0.01 s, 256³ ≈ 0.2 s.

**Used by:** [[cad.metrics]], [[cad.sandbox]] (worker), [[bench.suites]].
**Tested in:** [[test_geometry]] (box, sphere, hollow sphere, through hole, open mesh).
