---
tags: [module, data]
aliases: [build.py, t2c data build]
---
# data.build
**File:** `src/t2c/data/build.py` · **Libraries:** [[Libraries#PyYAML|PyYAML]], `random`, `hashlib` + [[cad.sandbox]]
**Data:** [[Dataset recipes]] · **Concepts:** [[Benchmark contamination]]

Assembles a training set from verified data, following a small YAML **recipe** in
`configs/data/`. Two kinds:

| `kind` | Output | Row format | Used by |
|---|---|---|---|
| `sft` | `data/<name>/{train,val}.jsonl` | `{id, text, code}` | [[train.sft]] |
| `rl` | `data/<name>/train.jsonl` + `gt/*.npz` | `{id, text, gt}` | [[train.grpo]] |

RL rows need a **ground-truth mesh** (the reward compares against it), so for `rl` every
distinct reference script is executed once and its mesh saved; prompts that describe the
same script share one mesh.

## Recipe format
```yaml
name: sft/v1
kind: sft
seed: 0
holdout: ["bench:cadprompt", "t2cq:val", "t2cq:test"]
train:
  - {source: t2cq, split: train}
  - {source: cadcoder, split: train_high}
  - {source: synth, split: train, n: 40000, exclude_families: [loft]}   # options
val:
  - {source: t2cq, split: val, n: 400}
```
Per entry: `n` (random sample of that size), `max_code_chars`, `exclude_families` (synth only).

## Leakage control (`holdout`)
Several datasets describe the *same* DeepCAD objects. If an object appears in a benchmark
and in training, the benchmark measures memory, not skill. `holdout` lists where the
forbidden objects come from: `bench:<name>` (the benchmark's objects) or `<source>:<split>`
(e.g. the whole t2cq test split, which the `deepcad` benchmark is drawn from). Then:
- **train** rows whose `uid` is held out are dropped — even if they come from a different
  corpus (CAD-Coder describes some of the same objects);
- **val** rows are taken *only* from held-out objects.

Synthetic parts have no DeepCAD identity and are exempt; their splits are disjoint by
construction ([[data.synth]]).

Deterministic: same recipe + same seed → byte-identical files. `info.json` stores the
recipe, counts and how many rows each entry contributed. `read_built(name, split)` reads
a built dataset back.

**Used by:** [[cli]], [[train.sft]], [[train.grpo]]. **Tested in:** [[test_build]].
