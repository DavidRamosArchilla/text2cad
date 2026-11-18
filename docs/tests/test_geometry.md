---
tags: [test]
aliases: [tests/test_geometry.py]
---
# test_geometry
**File:** `tests/test_geometry.py` · **Tests:** [[cad.geometry]], [[cad.metrics]]

Every reported metric and the RL reward depend on these functions, so they are checked
against shapes whose correct answer is known by hand.

| Test | What it checks | Why it matters |
|---|---|---|
| `test_voxel_volume_box` | a 10×20×40 box fills (10/40)·(20/40) of the normalised grid (±2 %), surface closed | the voxeliser fills solids correctly |
| `test_voxel_volume_sphere_and_cavity` | a sphere fills π/6 of its box; a hollow sphere (inner cavity) fills π/6·(1−0.6³) | the even–odd rule leaves enclosed cavities empty |
| `test_voxel_through_hole` | voxel fill of a plate with a hole equals CadQuery's exact volume | holes are empty |
| `test_iou_identical_and_scaled` | same box at 1× and 10× → IoU 1, CD at the noise floor, `dim_err` = log 10 | normalisation removes scale for IoU; `dim_err` still sees it |
| `test_iou_known_overlap` | cube vs half-height slab → IoU 0.5 | IoU arithmetic |
| `test_iou_is_orientation_sensitive` | 10×20×40 vs 40×20×10 → IoU < 0.2 | no hidden rotation alignment in the protocol metric |
| `test_open_mesh_is_flagged` | a cube without its top face → `open_frac` > 0.9 | broken ground truth is detected |
| `test_rigid_align_recovers_axis_swap_and_offset` | a part rotated (axis permutation) and moved is aligned back to within point-sampling noise | the CadQueryEval alignment finds axis swaps |
| `test_cqe_checks_pass_and_fail` | the same part modelled on another plane **passes**; a part with a larger hole **fails** (volume) | CadQueryEval verdicts in both directions |

Concepts: [[Comparing 3D shapes]], [[Rigid alignment]].
