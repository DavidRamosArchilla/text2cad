---
tags: [data]
aliases: [recipes, holdout, leakage]
---
# Dataset recipes

A recipe (`configs/data/*.yaml`) says which verified data goes into a training set;
[[data.build]] executes it (`t2c data build configs/data/sft_v1.yaml`). Format and options
are in that module note; here: what each recipe contains and why.

## The recipes
| Recipe | Output | Train | Val | Purpose |
|---|---|---|---|---|
| `sft_v1.yaml` | `data/sft/v1` | `t2cq` train + `cadcoder` high + `synth` train → **145 948** | 400 `t2cq` val + 200 `synth` val | main SFT mixture |
| `sft_deepcad.yaml` | `data/sft/deepcad` | `t2cq` train + `cadcoder` high → **105 948** | 400 `t2cq` val | ablation: no millimetre-scale data (the recipe of published specialists) |
| `sft_v1_heldout.yaml` | `data/sft/v1-heldout` | like v1 but without `swept_elbow`, `loft`, `gusset_bracket` | as v1 | generalisation to unseen part families (not built yet) |
| `rl_v1.yaml` | `data/rl/v1` | 3 000 `synth` + 2 000 `t2cq` (scripts ≤ 3 000 chars) + 1 000 `cadcoder` → **6 000** prompts + meshes | — | GRPO prompt pool |
| `rl_deepcad.yaml` | `data/rl/deepcad` | 3 000 `t2cq` + 1 000 `cadcoder` → **4 000** | — | DeepCAD-only RL (ablation, smoke tests) |

The RL pool is small on purpose: verifier-reward RL saturates on a few thousand prompts,
and it should lean towards the target distribution rather than what SFT already masters.

## Leakage control
All recipes use `holdout: ["bench:cadprompt", "t2cq:val", "t2cq:test"]` — 11 131 DeepCAD
objects in total:
- **CADPrompt's 200 objects**: 95 of them are in the `t2cq` training split.
- **`t2cq` val and test**: the [[DeepCAD held-out]] benchmark is drawn from the test split.

Effect in `sft/v1`: 94 `t2cq` training rows and 547 `cadcoder` rows describe held-out
objects and are removed (CAD-Coder describes the same DeepCAD objects as `t2cq`, so the
check is by object id, across corpora). Validation rows come only from held-out objects.

Why it matters: [[Benchmark contamination]]. CadQueryEval and Text2CAD-Bench are not
DeepCAD objects, and their data is never read by any recipe.

## Reproducibility
Same recipe + same seed → byte-identical files on any cluster. `data/<name>/info.json`
records the recipe and how many rows each entry contributed.
