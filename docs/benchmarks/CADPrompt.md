---
tags: [benchmark]
aliases: [cadprompt, CADCodeVerify]
---
# CADPrompt (`cadprompt`)

200 DeepCAD objects with **human-written** prompts and expert-written CadQuery.

- **Paper:** *Generating CAD Code with Vision-Language Models for 3D Designs* (ICLR 2025;
  introduces CADPrompt and the CADCodeVerify refinement method) —
  <https://arxiv.org/abs/2410.05340>
- **Data:** <https://github.com/Kamel773/CAD_Code_Generation>, pinned `33dcecd6`
- **Adapter:** `CADPrompt` in [[bench.suites]]

## What is in it
Per object (folder named by its DeepCAD id): two prompts, the expert's Python code, the
ground-truth mesh (STL/OBJ) and its measurements. We use both prompts → 400 samples, two
groups:

| Group | Example (object `00000007`) |
|---|---|
| `abstract` | "write a python code using CADQuery to create an extruded sketch of a circle." |
| `measured` | "…extruding a circular sketch. The circle should have a radius of 0.75 units and the extrusion should be 0.20923 units high." |

Ground truth = the published `Ground_Truth.stl` (all 200 are watertight; the adapter would
rebuild a broken one from the expert script).

## What it represents
Real people describing DeepCAD-style objects (normalised units). Closer to how users
write than the machine-generated Text2CAD descriptions; shapes are the same family as
the training data.

## How to read it
- **`abstract` prompts are under-specified** — "a cylinder" has no proportions, so even a
  perfect model cannot reach IoU 1. Compare models on `measured`; read `abstract` as
  "does it guess sensible defaults".
- **Leakage**: 95 of the 200 objects are in the `t2cq` training split. All CADPrompt
  objects are removed from training by id ([[Dataset recipes]]).
- **Published numbers** (GPT-4, Gemini, CodeLlama, 2024): only the compile rate is
  comparable (GPT-4 zero-shot: 92 % → IR 8 %); their geometric metrics use ICP alignment,
  1 000 points and medians.
