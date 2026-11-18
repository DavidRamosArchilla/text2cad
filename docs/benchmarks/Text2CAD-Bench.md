---
tags: [benchmark]
aliases: [t2cb, Text2CAD Bench]
---
# Text2CAD-Bench (`t2cb`)

**The target benchmark of the original plan.** Text → CadQuery for real engineering parts,
at increasing difficulty.

- **Paper:** *Text2CAD-Bench: A Benchmark for LLM-based Text-to-Parametric CAD
  Generation* (May 2026) — <https://arxiv.org/abs/2605.18430>
- **Data:** <https://huggingface.co/datasets/AICAD/Text2CAD-Bench> (CC BY 4.0), pinned `76428d5f`
- **Adapter:** `Text2CADBench` in [[bench.suites]]

## What the full benchmark contains (paper)
600 human-curated parts, each with two prompts:
- **geo** — the finished shape as a non-expert would describe it;
- **seq** — modelling steps as an expert would give them.

| Level | Parts | Content |
|---|---|---|
| L1 | 200 | primitives, simple spatial relations, chamfers, through-holes |
| L2 | 200 | booleans, fillets, patterns, standard features |
| L3 | 100 | sweep, loft, shell, free-form surfaces, complex patterns |
| L4 | 100 | real-world products (judged by a vision-language model, no IoU) |

Protocol: CD on 30 000 points and IoU on 256³ voxels after unit-bounding-box
normalisation; a sample is invalid if the code fails, exceeds 60 s or yields no solid; up
to 3 retries with the error message; a system prompt with 5 in-context examples.

## What we actually have
Only `release.csv`: **151 parts (L1 60, L2 61, L3 30) × geo/seq = 302 prompts**, no ground
truth ("not publicly released to prevent benchmark contamination"; full access by request).
The evaluation code link on the card is a placeholder.

Example (L1_1, geo, shortened): *"The Z-axis is the main axis/thickness direction… The
model is a prismatic solid based on a regular hexagon… distance between opposite sides 50
millimeters… extruded in the positive Z-axis direction, total height 20 millimeters. At the
center of the top surface there is a circular through-hole…"*

So locally we measure **IR only** (with 3 retries, groups `L1/geo` … `L3/seq`). The prompt
is the zero-shot core of the paper's prompt (its 5 examples are not public) — [[prompts]].

## How to read it
- IR tells you whether the code runs and builds a solid, **not** whether it is the right
  part. The pilot 0.6B model reaches 36 % IR (GPT-5.2 published 31 %) while its CadQueryEval
  IoU is only 0.32.
- Published numbers in the report are on the full 600 parts, with the 5-shot prompt, from
  early 2026 — indicative only.
- Per-level IR still shows *where* code breaks (L3 sweeps/lofts fail most).

## Getting the ground truth
Ask the authors. Then put the files in `data/raw/t2cb_gt/<id>.step` (e.g. `L1_1.step`;
`.stp` or `.stl` also work) and re-run `t2c bench prepare t2cb`: STEP files are imported
with cadquery and meshed, and CD/IoU are computed automatically from then on.

## Published reference (paper, L1 geo / overall)
GPT-5.2: IoU 0.59 / 0.45, IR 11 % / 31 %. Claude-4.5-Sonnet: 0.54 / 0.43. Prior CAD
specialist models (Text2CAD, CADFusion, Text2CADQuery): IoU 0.03–0.08. Full table:
`configs/references.yaml`.
