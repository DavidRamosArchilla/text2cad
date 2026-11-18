---
tags: [concept]
aliases: [voxel IoU, Chamfer, Hausdorff, mesh]
---
# Comparing 3D shapes

To grade a generated part we need numbers that say how close two solids are. The exact
B-rep solids are first converted to **triangle meshes** (tessellation); then:

| Idea | How | Strength | Blind spot |
|---|---|---|---|
| **Voxel IoU** | fill a 3D grid of small cubes (voxels) inside each solid; IoU = shared voxels ÷ voxels in either (the *Jaccard index*) | intuitive volume overlap, 0–1 | thin features vanish at low resolution; needs closed surfaces |
| **Chamfer distance** | sample points on both surfaces; for each point the distance to the nearest point of the other cloud; average both ways | sensitive to surface detail, works for thin walls | an average: one missing hole barely moves it |
| **Hausdorff distance** | the *worst* nearest-point distance (we use the 95th percentile, robust to a few outliers) | catches local errors | sensitive to noise |
| **Volume, bounding box** | direct comparisons | simple, absolute | many shapes share a volume |

Two practical ingredients:
- **Inside or outside?** A ray crossing the surface an odd number of times is inside (the
  **even–odd rule**) — used by our voxeliser; it only works if the surface is
  **watertight** (no holes), which we therefore check.
- **Nearest neighbours fast**: a **k-d tree** finds the closest of 30 000 points in
  microseconds.

**Normalisation** decides what is compared: Text2CAD-Bench scales both shapes into a unit
box (compares shape, ignores size), CadQueryEval works in millimetres after aligning the
parts ([[Rigid alignment]]). Our metrics and their pitfalls: [[Metrics]]; code:
[[cad.geometry]], [[cad.metrics]].

## Learn more
- Wikipedia: Polygon mesh <https://en.wikipedia.org/wiki/Polygon_mesh> · Voxel
  <https://en.wikipedia.org/wiki/Voxel> · Jaccard index (IoU)
  <https://en.wikipedia.org/wiki/Jaccard_index> · Hausdorff distance
  <https://en.wikipedia.org/wiki/Hausdorff_distance> · Even–odd rule
  <https://en.wikipedia.org/wiki/Even%E2%80%93odd_rule> · k-d tree <https://en.wikipedia.org/wiki/K-d_tree>
- Chamfer distance as used in 3D deep learning: Fan et al., *A Point Set Generation
  Network for 3D Object Reconstruction from a Single Image* (2017) <https://arxiv.org/abs/1612.00603>
- Tools: trimesh <https://trimesh.org/> · scipy `cKDTree`
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html>
