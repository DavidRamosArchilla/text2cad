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

## One part, eight answers
The reference part is a plate 80 × 50 × 10 mm with a 20 mm hole through the middle. Each
row is a different answer, scored by the repository's verifier with benchmark settings
(computed 2026-11-20). It shows what each metric notices and what it misses.

| Answer | Valid | IoU | CD ×1000 | dim err | CadQueryEval | Checks failed |
|---|---|---|---|---|---|---|
| correct part, written with different code | yes | 1.000 | 7.4 | 0.00 | pass | — |
| hole 24 mm instead of 20 | yes | 0.963 | 10.1 | 0.00 | fail | volume, Hausdorff |
| hole moved 10 mm along X | yes | 0.901 | 14.3 | 0.00 | fail | Hausdorff |
| hole missing | yes | 0.921 | 11.5 | 0.00 | fail | volume |
| thickness 12 instead of 10 | yes | 0.842 | 20.2 | 0.06 | fail | bbox, volume, Hausdorff |
| 10× too small (cm instead of mm) | yes | 1.000 | 7.4 | 2.30 | fail | bbox, volume, Chamfer, Hausdorff |
| right part, standing up (built on the XZ plane) | yes | 0.089 | 259.6 | 1.07 | pass | — |
| script crashes (impossible fillet) | no | — | — | — | fail | — |

- Metrics judge the **part, not the code**: different code for the same part scores perfectly.
  CD 7.4 is the sampling-noise floor of a perfect answer.
- **IoU is dominated by bulk volume**: wrong or missing holes cost only 4–10 %. A misplaced
  hole scores *lower* than a missing one. CadQueryEval's millimetre checks catch all of them.
- **Normalisation hides scale**: the 10× too small part has IoU 1.0 and a perfect CD; only
  dim err (2.30 = ln 10) and CadQueryEval see it.
- **IoU / CD do not align orientation, CadQueryEval does**: the standing-up part fails IoU
  (0.09) and passes CadQueryEval. Which is right depends on whether the prompt fixed the
  orientation (Text2CAD-Bench prompts do).
- An invalid answer has no shape metrics; it counts as 0 in IoU_all.

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
