---
tags: [benchmark, index]
aliases: [Benchmarks]
---
# Benchmarks overview

Five benchmarks, chosen so that together they answer different questions. Every model is
scored by the same code with the same core metrics ([[Metrics]]), so rows compare directly.

| Benchmark | Question it answers | n | Ground truth | Headline |
|---|---|---|---|---|
| [[Text2CAD-Bench]] (`t2cb`) | Do we make progress on *the* target benchmark of the project? | 302 prompts (151 parts × geo/seq) | **not public** → validity only | IR ↓ |
| [[CadQueryEval]] (`cqe`) | Would an engineer accept the part? (strict mm tolerances) Where are we vs current frontier models? | 25 | reference STL | pass rate ↑ |
| [[CADPrompt]] (`cadprompt`) | Does it work with *human-written* prompts? | 400 (200 × measured/abstract) | expert STL | IoU_all ↑ |
| [[DeepCAD held-out]] (`deepcad`) | Did SFT learn its training distribution? | 500 | reference script | IoU_all ↑ |
| [[Synth benchmark]] (`synth`) | Does it handle mm-scale parts and both prompt styles? Which part families fail? | 800 (400 × geo/seq) | exact | IoU_all ↑ |

Think of them on a line from *in-distribution* to *target*:

```
deepcad, synth  →  cadprompt  →  cqe  →  t2cb
(same style as training)            (real prompts, real parts)
```

A model that improves only on the left end memorised its data; improvements that reach
the right end are the ones that matter.

## Same rules for every model
- **Prompt style**: base and API models get each benchmark's *protocol* prompt; our
  fine-tuned models get the *native* prompt they were trained with ([[prompts]]). Recorded
  with every result.
- **Retries**: as the benchmark's protocol says (Text2CAD-Bench: 3 with error feedback;
  others: 0). `IR@1` always shows the first attempt.
- **Decoding**: greedy (temperature 0) unless a model config says otherwise ([[Decoding]]).
- **Verifier settings**: 256³ voxels, 30 000 surface points, 60 s timeout.

## A snapshot (2026-11-16) to illustrate how they behave
| Model | t2cb IR ↓ | cqe pass ↑ | cqe IoU_all ↑ | cadprompt IoU_all ↑ | deepcad IoU_all ↑ | synth IoU_all ↑ |
|---|---|---|---|---|---|---|
| Qwen3-1.7B base | 91.1 % | 0 % | 0.064 | 0.093 | 0.025 | 0.058 |
| Qwen3-8B base | 61.9 % | 12 % | 0.283 | 0.129 | 0.106 | 0.207 |
| Qwen3-1.7B + SFT (`sft/v1`) | 24.2 % | 20 % | 0.366 | 0.206 | 0.610 | 1.000 |
| … + GRPO (`rl/v1`) | 23.5 % | 24 % | 0.425 | 0.229 | 0.620 | 1.000 |
| best published (CadQueryEval leaderboard) | | 100 % | | | | |

Live numbers: `results/REPORT.md` ([[Reading the report]]). What the snapshot shows:
- `synth` saturates (1.000) once its training split is in the mix — useful as a regression
  check, not as a measure of skill.
- `deepcad` jumps from 0.03 to 0.61: SFT learned its distribution.
- `cadprompt` and `cqe` move much less: transfer to real prompts is the hard part.
- `t2cb` IR drops a lot, but IR says nothing about shape (see [[Text2CAD-Bench]]).

## Adding a benchmark
Write a `Benchmark` subclass with a `prepare()` in [[bench.suites]], register it in
[[bench.base]], add its headline to [[results.report]].
