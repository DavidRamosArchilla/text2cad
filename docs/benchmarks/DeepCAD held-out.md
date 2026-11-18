---
tags: [benchmark]
aliases: [deepcad, DeepCAD benchmark]
---
# DeepCAD held-out (`deepcad`)

500 descriptions from the **test split** of the [[Text-to-CadQuery dataset]] — the same
kind of data the SFT stage trains on, but objects never seen in training.

- **Adapter:** `DeepCADText` in [[bench.suites]]: 500 rows chosen with a fixed seed
  (20261115) from the verified test split; ground truth = the reference script executed
  in the sandbox.

## What it represents
**In-distribution performance**: did fine-tuning learn its training data? A model that
does well here but not on [[CADPrompt]] / [[CadQueryEval]] has learned the style of the
corpus rather than CadQuery in general.

## How to read it
- Expect large gains from SFT (Qwen3-1.7B: 0.025 → 0.61 IoU_all) — that is the sanity
  check that training works, not the result you are after.
- "Ground truth" is Gemini 2.0 Flash's script for the description, verified to run, not
  the original DeepCAD model. A model can be "wrong" here and right about the description.
- Leakage control: the whole `t2cq` test and val splits are held out of every training
  recipe ([[Dataset recipes]]).
- Included in the in-training benchmark subset by default ([[train.callbacks]]), together
  with CadQueryEval: one in-distribution curve, one target curve.
