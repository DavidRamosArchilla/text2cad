---
tags: [benchmark]
aliases: [cqe, CadQuery Eval]
---
# CadQueryEval (`cqe`)

25 practical parts described in plain language, judged with strict millimetre tolerances.
The benchmark with the most current frontier results.

- **Code:** <https://github.com/danwahl/cadqueryeval> (MIT) · **leaderboard:**
  <https://danwahl.net/cadqueryeval/>
- **Data:** <https://huggingface.co/datasets/drwahl/cadqueryeval>, pinned `7504b0d7`: tasks,
  reference STLs, and **every leaderboard attempt** (model, code, verdict)
- Based on CadEval (OpenSCAD): <https://github.com/wgpatrick/cadeval>
- **Adapter:** `CadQueryEval` in [[bench.suites]]; scorer in [[cad.metrics]]; validation in [[bench.cqe]]

## The tasks
Hex nut, plate with two holes, gasket, pin, channeled block, counterbored plate, hollow
torus, chamfered block, open box, swept arc, eye bolt, T-handle tap holder, pipe,
revolved triangle, square-to-circle loft, clamp, slotted plate, counterbored and through
holes, crank plate, milled part, milled plate, turned part, arm with handle, L-block, silo.

Each prompt = description (often as modelling steps) + approximate bounding box +
expected number of components. Example (task1):

> A simple hex nut without the threads modeled… 1. The extruded hexagonal prism is 10mm
> wide (flat side to flat side). 2. The hole diameter through the center of the hexagon is
> 5mm. The thickness of the nut is 5mm.

## Scoring
A part **passes** only if all six checks pass: watertight, one component, bounding box ±1 mm,
volume ±2 %, Chamfer ≤ 1 mm, 95th-percentile Hausdorff ≤ 1 mm (after aligning the part to
the reference). Some tasks accept an alternative reference. Details: [[Metrics]].

Upstream protocol (our `protocol` style): its own system prompt, the script must end with
`cq.exporters.export(result, "output.stl")`, one attempt.

## Our scorer vs the official one
The official scorer uses open3d, which does not install on this cluster (glibc) and aligns
with random RANSAC. Ours re-implements the six checks with numpy/scipy/trimesh and a
deterministic alignment. Re-scoring 300 published attempts (150 pass, 150 fail): **300 /
300 identical verdicts**. So our pass rates compare with the leaderboard.

## What it represents
"Would a mechanical engineer accept this part?" Tolerances are tight (1 mm on 10–100 mm
parts), so near-misses fail. Closest to practical usefulness of the five benchmarks.

## How to read it
- **25 tasks**: one task = 4 points. Use the IoU_all column as well (shows progress before
  parts start passing) and repeat runs before trusting small differences.
- **Frontier**: 4 published models at 25/25 (Claude Opus 5.5, GPT-5.6 Sol Pro, GPT-6 Astra,
  GPT-6 Sol), 13 of 91 at ≥ 85 %, median 52 %. Qwen-family models from 20 % to 88 %.

## Never train on it
The dataset carries a canary string and asks to be used for evaluation only. No recipe
reads it ([[Benchmark contamination]]).
