---
tags: [benchmark]
aliases: [IoU, IoU_all, IR, Chamfer distance, dim_err, pass rate]
---
# Metrics

All benchmarks report the same core metrics, computed by [[bench.base]] from per-sample
verifier results ([[cad.metrics]]). Background on the geometry: [[Comparing 3D shapes]].

## Validity
| Metric | Definition | Notes |
|---|---|---|
| **IR** (invalid rate) | share of samples whose final answer does not produce a valid solid | "invalid" = no code, syntax error, refused code, runtime error, timeout (60 s), crash, no solid in `result`, zero volume |
| **IR@1** | the same on the first attempt | differs from IR only when retries are allowed (Text2CAD-Bench) |

## Shape (on normalised parts — the Text2CAD-Bench protocol)
Both meshes are scaled into a unit bounding box first, so size does not matter, only shape
and orientation.

| Metric | Definition | Range / good value |
|---|---|---|
| **IoU** | voxel intersection ÷ union at 256³, averaged over **valid** samples | 0–1, higher is better |
| **IoU_all** | the same sum divided by **all** samples (invalid = 0) | 0–1, higher is better |
| **CD** | Chamfer distance on 30 000 surface points, sum of the two directional means, × 1000 | lower is better; a perfect part scores ≈ 10 (sampling noise), not 0 |
| **dim err** | median over valid samples of mean\|log(predicted extent / true extent)\| over x, y, z, in real units | 0 = right size on every axis; 0.69 = a factor 2 |

## CadQueryEval checks (absolute millimetres, after rigid alignment)
| Check | Passes when |
|---|---|
| watertight | closed surface |
| single component | expected number of connected pieces |
| bounding box | sorted extents within 1 mm |
| volume | within 2 % |
| Chamfer | ≤ 1 mm (mean of the two directions) |
| Hausdorff 95 | 95th-percentile distance ≤ 1 mm |

**pass rate** = share of tasks passing all six. Each check's own rate is also reported.

## Pitfalls
- **IoU (valid) rewards failing on hard parts**: if a model gives up on the difficult
  tasks, they leave the average. IoU_all cannot be improved that way → use it to compare.
- **IR is not quality.** A model trained on DeepCAD data produces valid but wrong geometry.
- **No rotation alignment in IoU/CD**: a correct part modelled along the wrong axis scores
  low. Intended (prompts state axes) but harsh when a prompt leaves orientation open.
- **Normalisation hides scale**: a part 10× too large has IoU 1 → look at dim err.
- **Small n**: CadQueryEval has 25 tasks (4 points each); differences under ~10 points need
  repeated runs.
- **CD scale**: our CD is the paper's definition (sum, unit box, ×1000) — CadQueryEval's
  "Chamfer" is a different quantity (mean, in mm).

## In training
The GRPO reward uses the same IoU / CD / dim err at cheaper settings (64³, 4 000 points) —
[[cad.reward]]. The in-training callback logs IR, IoU, IoU_all and pass rate on a
48-sample subset — [[train.callbacks]].
