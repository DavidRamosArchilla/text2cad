---
tags: [benchmark]
aliases: [synth benchmark]
---
# Synth benchmark (`synth`)

The held-out test split of our generator: 400 millimetre-scale parts × {geo, seq} = 800
samples, grouped by **part family** (17 groups). Exact ground truth.

- **Adapter:** `SynthBench` in [[bench.suites]] · **data:** [[Synthetic parts]] · **generator:** [[data.synth]]

## What it represents
Millimetre-scale parts in the two prompt styles of [[Text2CAD-Bench]], covering
holes, chamfers, revolves, lofts, sweeps, shells and patterns — the target distribution,
with ground truth we control. The per-family table shows exactly which kinds of parts a
model cannot build.

## How to read it
- **It saturates** once `synth` training data is in the mix (≈ 0.99–1.0 IoU_all): train and
  test come from the same templates. Then it is a regression check (did something break?),
  not a measure of skill.
- **For a real generalisation test**, train with `sft_v1_heldout.yaml` (three families
  never seen) and compare the seen and unseen families in the by-group table.
- Before any synthetic training (base models, `sft/deepcad`) it is a clean test of
  millimetre-scale, multi-operation parts.
- The text is templated, so a good score does not prove robustness to free-form prompts.
