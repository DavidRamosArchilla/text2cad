---
tags: [guide]
aliases: [REPORT.md, report.html]
---
# Reading the report

`results/REPORT.md` and `results/report.html` are rebuilt after every full `t2c eval`
(or by hand with `t2c report`) from the run summaries in `results/runs/`. Code:
[[results.report]], storage: [[results.store]]. The HTML version has the same tables with
sortable columns (click a header).

## 1. Scoreboard
One row per **model × prompt style**, one or two columns per benchmark.

| Column | Meaning |
|---|---|
| Model | `name` from the model spec: a config name (`qwen3-1.7b`) or a training run (`sft-qwen3-1.7b-v1`, `…@1000` for a checkpoint) |
| Stage | `base`, `sft`, `grpo` or `api` |
| Prompt | `protocol` (benchmark's own prompt) or `native` (our training prompt) — see [[prompts]] |
| `t2cb IR ↓` | [[Text2CAD-Bench]]: invalid rate (lower is better). Only validity: no ground truth yet |
| `cqe pass ↑`, `cqe IoU_all ↑` | [[CadQueryEval]]: share of the 25 parts passing all six checks; mean IoU with failures as 0 |
| `cadprompt IoU_all ↑` | [[CADPrompt]] |
| `deepcad IoU_all ↑` | [[DeepCAD held-out]] |
| `synth IoU_all ↑` | [[Synth benchmark]] |

Each benchmark cell uses that benchmark's own retry budget (3 for `t2cb`, 0 otherwise).

**Numbers in parentheses** are the change against *the same model before fine-tuning*:
the report finds the `base` run whose model id equals the fine-tuned model's
`base_model` (lineage written by the trainer, see [[train.common]]). `pp` = percentage
points. Example: `24.0% (+24.0pp)` = 24 % now, 0 % for the base model.

## 2. One section per benchmark
- **Leaderboard** — every model with all core metrics ([[Metrics]]). Grey rows marked
  *published* are numbers from papers or leaderboards (`configs/references.yaml`, and for
  CadQueryEval the 91 models in its dataset). The note under the table says how they differ
  from our runs — read it before comparing.
- **By group** — the headline metric split by group: `L1/geo … L3/seq` for Text2CAD-Bench,
  `measured`/`abstract` for CADPrompt, one group per part family for Synth.
- **Why outputs were invalid** — share of each error type after retries: `syntax_error`,
  `runtime_error` (the script raised: wrong method name, impossible fillet…), `timeout`,
  `crash`, `no_code`, `no_result`, `no_solid`… This tells you *what kind* of mistakes a model makes.

## 3. Training runs
For every training run that has more than one evaluated checkpoint (e.g. after
`scripts/eval_checkpoints.sh <run>`), a table of headline metric against step.

## Rules the report follows
- **Latest wins.** Runs are never overwritten; for each (benchmark, model, prompt style,
  retries) the most recent run is shown. Delete a directory in `results/runs/` to drop one.
- **Smoke tests are excluded.** Runs made with `--limit` are marked `partial` and ignored.
- **Same scorer for everyone.** Every row that is not *published* was produced by the same
  verifier code with the same settings, so those rows compare directly.

## Curves during training
The report only contains evaluations of saved checkpoints. For curves along a run, use
TensorBoard (`tensorboard --logdir outputs/train`): the loss, the reward (GRPO) and the
in-training benchmark subset `eval/<benchmark>_<metric>` from [[train.callbacks]]. Those
subset numbers are for monitoring (48 samples, one attempt); report numbers from `t2c eval`.
See [[Experiment tracking]].

## Common misreadings
- A low **IR** is not a good model: DeepCAD-trained models produce valid but wrong
  geometry. Look at `IoU_all` or the CadQueryEval pass rate. [[Text2CAD-Bench]] explains
  why its column is only IR.
- **IoU (valid)** can rise while the model gets worse: failing on hard parts removes them
  from the average. `IoU_all` cannot be gamed that way.
- **CadQueryEval has 25 tasks**: one task = 4 points. Differences of 4–8 points are noise
  unless they repeat across seeds.
- **Synth near 1.0** means the model learned the generator's training distribution, not
  that it generalises — see [[Synth benchmark]].
