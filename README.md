# t2c — fine-tuning small LLMs to write CadQuery

Turn a text description of a mechanical part into an executable [CadQuery](https://cadquery.readthedocs.io)
script, and measure it properly. The repository contains the whole loop:

```
 pinned downloads ──► execution-verified corpora ──► SFT ──► GRPO (reward = CAD verifier) 
        │                                             │            │
        └──► benchmarks (5) ◄──────── t2c eval ◄──────┴────────────┘
                   │
                   └──► results registry (one JSON per run) ──► REPORT.md / report.html
```

Everything that touches geometry goes through one verifier (`src/t2c/cad/`): run the
script in a sandbox, tessellate the solid, compare it with the ground truth. The same code
scores benchmarks and produces the RL reward.

> Read [Findings that change the original plan](#findings-that-change-the-original-plan)
> before spending compute. Short version: the target benchmark does not publish its ground
> truth, and current frontier models are much better at CadQuery than the numbers the plan
> was based on.

## Quick start

```bash
scripts/bootstrap.sh                 # pinned uv + .venv from uv.lock (identical on every cluster)
source scripts/env.sh                # paths + cluster profile; do this in every shell
t2c doctor                           # checks CAD kernel, sandbox, GPUs

t2c data download                    # 5 pinned sources (~950 MB) into data/raw/
scripts/submit.sh --cpus 96 --name prep -- t2c data prepare      # verify ~185k scripts (≈25 min on 96 cores)
t2c data synth                       # procedural mm-scale parts (train/val/test)
t2c bench prepare                    # 5 benchmarks + ground-truth meshes
t2c data build configs/data/sft_v1.yaml
t2c data build configs/data/rl_v1.yaml

# baselines: any base model, no fine-tuning
scripts/submit.sh --gpus 1 --name base -- t2c eval --model configs/models/qwen3-1.7b.yaml

# stage 1 and stage 2
scripts/submit.sh --gpus 1 --name sft  -- t2c train sft  configs/sft/qwen3-1.7b.yaml
scripts/submit.sh --gpus 1 --cpus 48 --name grpo -- t2c train grpo configs/grpo/qwen3-1.7b.yaml

# evaluate what you trained (a checkpoint directory is a model)
scripts/submit.sh --gpus 1 --name eval -- t2c eval --model outputs/train/sft-qwen3-1.7b-v1/final
t2c report                           # results/REPORT.md + results/report.html
```

`scripts/submit.sh [--gpus N] [--cpus N] [--mem GB] [--time T] [--name X] [--after JOBID] -- <command>`
wraps any command in a Slurm job using the current cluster profile; the same commands run
interactively without it. `--after` chains jobs, so a whole pipeline can be queued at once:

```bash
sft=$(scripts/submit.sh --gpus 1 --name sft -- t2c train sft configs/sft/qwen3-1.7b.yaml | awk '{print $4}')
scripts/submit.sh --gpus 1 --after $sft --name eval-sft -- t2c eval --model outputs/train/sft-qwen3-1.7b-v1/final
grpo=$(scripts/submit.sh --gpus 1 --cpus 48 --after $sft --name grpo -- t2c train grpo configs/grpo/qwen3-1.7b.yaml | awk '{print $4}')
scripts/submit.sh --gpus 1 --after $grpo --name eval-grpo -- t2c eval --model outputs/train/grpo-qwen3-1.7b-v1/final
```

### Second cluster

1. `git clone` the repo, run `scripts/bootstrap.sh`.
2. Copy `configs/clusters/TEMPLATE.env.example` to `configs/clusters/<name>.env` and set the
   partition, account, GPU flag and storage paths. It is picked up automatically when the
   hostname matches `T2C_HOST_PATTERN`, or with `export T2C_CLUSTER=<name>`.
3. Re-run the data commands above. Sources are pinned to exact revisions and every random
   choice is seeded, so both clusters end up with the same datasets and benchmarks
   (`data/bench/*/info.json` has a content hash that is stored with every result).
4. If compute nodes have no internet: download data and models on the login node
   (`t2c data download`, `hf download Qwen/Qwen3-1.7B`) and set `HF_HUB_OFFLINE=1` in the profile.

Results travel through git: `results/runs/*/summary.json` is tracked, so `git pull` on
either side merges the two clusters' results and `t2c report` shows them together.

## Where things stand (2026-11-15)

Everything below was produced by the commands in this README on one H200; the live
version is `results/REPORT.md`.

| Model | Stage | Prompt | t2cb IR ↓ | cqe pass ↑ | cqe IoU_all ↑ | cadprompt IoU_all ↑ | deepcad IoU_all ↑ | synth IoU_all ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **pilot-grpo-qwen3-0.6b** | grpo | native | 34.8% (−65.2pp) | 12.0% (+12.0pp) | 0.268 | 0.182 (+0.181) | 0.579 (+0.578) | 0.993 (+0.991) |
| **pilot-sft-qwen3-0.6b** | sft | native | 36.4% (−63.6pp) | 20.0% (+20.0pp) | 0.322 | 0.172 (+0.171) | 0.565 (+0.564) | 0.990 (+0.989) |
| **qwen3-0.6b** | base | protocol | 100.0% | 0.0% | – | 0.000 | 0.001 | 0.001 |
| **qwen3-1.7b** | base | protocol | 91.1% | 0.0% | 0.064 | 0.093 | 0.025 | 0.058 |
| **qwen3-4b** | base | protocol | 74.2% | 4.0% | 0.040 | 0.125 | 0.111 | 0.076 |
| **qwen3-8b** | base | protocol | 61.9% | 12.0% | 0.283 | 0.129 | 0.106 | 0.207 |

`pilot-*` is Qwen3-0.6B trained for 13 minutes on 40k samples of the `sft/v1` mixture, then
100 GRPO steps (10 minutes) on `rl/v1`. Parentheses are the change against the untouched
model. What the numbers say:

- **Small base models cannot write CadQuery.** Qwen3 up to 8B produces invalid scripts
  more than half of the time on every benchmark (52–100 %: hallucinated methods,
  undefined variables).
- **A 13-minute SFT of the smallest model beats Qwen3-8B on every benchmark** and takes
  CadQueryEval from 0/25 to 5/25, level with the published `qwen3.7-flash` (20 %). The best
  published models are at 88–100 % on the same 25 tasks.
- **Do not read the `t2cb` column as quality.** IR only says the script ran. The same
  pilot model that is 36 % invalid there — close to GPT-5.2's published 31 % — has
  IoU 0.32 on CadQueryEval. Models trained on DeepCAD data are *good* at producing valid,
  wrong geometry. Without ground truth, `t2cb` cannot tell you whether you are winning.
- **`synth` saturates** (0.99) once its generator's training split is in the mixture: it
  verifies the plumbing and catches regressions, but only `sft_v1_heldout.yaml`
  (three part families withheld) turns it into a generalisation test.
- **GRPO did what its reward asked and no more.** Rollout validity rose from 88 % to 97 %
  during training and the in-distribution benchmarks improved (DeepCAD invalid rate
  4.4 % → 0.4 %), but CadQueryEval went from 5/25 to 3/25. With 25 tasks that difference is
  within noise, yet it is clearly not a gain: the RL pool has to look like the target
  for RL to help there.
- **Error-feedback retries do nothing for the fine-tuned models** (IR@1 = IR on `t2cb`):
  they were never trained on a repair turn.

In progress when this was written: the full `sft-qwen3-1.7b-v1` run (2 epochs, ≈3 h),
with its evaluation, GRPO stage and final evaluation queued behind it (`squeue -u $USER`).

### What to try next, in order of expected information

1. Ask the Text2CAD-Bench authors for ground truth. Until then the headline benchmark of
   the original plan is not measurable.
2. Run one or two current frontier models through `t2c eval` on `cqe`, `cadprompt`, `synth`
   and `t2cb` (copy `configs/models/api-example.yaml`). That is the real bar, measured
   with the same scorer, instead of numbers from early 2026.
3. `sft/v1` versus `sft/deepcad` versus `sft/v1-heldout` at fixed model size: how much of
   the gain is the millimetre-scale data, and how much transfers to unseen part families.
4. Model size at fixed data: 1.7B → 4B → 8B. Larger base models already know some
   CadQuery, so check that SFT on this corpus does not *lower* their CadQueryEval score.
5. Make the RL pool look like the target: more and harder generator families, prompts
   paraphrased by an LLM, a repair turn in the conversation, and a reward with rigid
   alignment for parts whose orientation the prompt leaves open.
6. Several seeds per configuration before believing differences on `cqe` (25 tasks = 4
   points per task).

## Findings that change the original plan

These came out of checking the plan against the actual sources (November 2026).

1. **Text2CAD-Bench ground truth is not public.** The Hugging Face repo contains only
   `release.csv`: 151 of the 600 parts (L1 60, L2 61, L3 30), two prompts each, no
   geometry, and the evaluation code link is a placeholder. The card says to contact the
   authors for full access. Until then only the invalidity rate (IR) can be measured on it
   locally. `t2c bench prepare t2cb` picks up ground truth automatically if you obtain it
   (drop `<id>.step` files in `data/raw/t2cb_gt/`).
2. **Frontier models have moved.** The 0.59 IoU figure is GPT-5.2 (early 2026).
   CadQueryEval, which ships every leaderboard attempt, shows four models at 25/25
   (Claude Opus 5.5, GPT-5.6 Sol Pro, GPT-6 Astra, GPT-6 Sol) and 13 of 91 at 85 % or
   better; the median published model is at 52 %. On parts of that kind there is no
   "frontier stumbles" gap left; a small fine-tuned model can at best *match* the top, at
   a fraction of the cost. Whether L3 (sweep/loft/complex patterns, 68–93 % invalid in the
   paper) still has headroom is unknown until someone re-runs current models.
3. **The plan is not uncrowded.** CAD-Coder (NeurIPS 2025) already did SFT + GRPO with a
   Chamfer-distance reward on Qwen2.5-7B, and BenchCAD reports that fine-tuning and RL
   help in-distribution but generalise poorly to unseen part families.
4. **The available training data is the wrong distribution.** Both public corpora are
   DeepCAD sketch-and-extrude parts in normalised units with long machine-written
   descriptions. Specialists trained on exactly that data score IoU 0.03–0.08 on
   Text2CAD-Bench — far *below* general LLMs. Plain SFT on it should be expected to make a
   base model worse on millimetre-scale prompts, not better.
5. **Benchmark objects leak into the training corpora.** 95 of CADPrompt's 200 objects are
   in the Text-to-CadQuery training split. The dataset builder holds out benchmark objects
   by DeepCAD id (`holdout:` in `configs/data/*.yaml`).

What the repository does about it: it scores on four benchmarks that *do* have public
ground truth, adds a procedural generator of millimetre-scale parts with exact ground
truth (training data, RL environment and a held-out benchmark in the target
distribution), imports the CadQueryEval leaderboard so current frontier numbers sit next
to yours, and makes a DeepCAD-only ablation one config away so the distribution effect is
measurable rather than assumed.

## Benchmarks

| name | what | n | ground truth | protocol prompt | retries | headline |
|---|---|---:|---|---|---:|---|
| `t2cb` | Text2CAD-Bench public sample, L1–L3 × {geo, seq} | 302 | **not public** | paper's rules (zero-shot core) | 3 | IR |
| `cqe` | CadQueryEval: mm-scale parts, six pass/fail checks | 25 | reference STL | upstream, verbatim | 0 | pass rate |
| `cadprompt` | CADPrompt: human prompts over DeepCAD objects | 400 | expert STL | generic | 0 | IoU_all |
| `deepcad` | held-out Text2CAD descriptions (in-distribution for SFT) | 500 | executed reference script | generic | 0 | IoU_all |
| `synth` | held-out procedural parts, geo + seq prompts | ~800 | exact (generator) | generic | 0 | IoU_all |

Every benchmark reports the same core metrics, so they can be read side by side:

- **IR / IR@1** — invalid rate after the allowed retries / on the first attempt. Invalid =
  no code, syntax or runtime error, timeout (60 s), kernel crash, or no solid in `result`.
- **IoU (valid)** and **CD×10³** — Text2CAD-Bench protocol: both shapes normalised to a unit
  bounding box, voxel IoU at 256³, Chamfer distance on 30k surface points (sum of the two
  mean nearest-neighbour distances), averaged over valid samples only. No rotation
  alignment: a correct part on the wrong axis scores low. A perfect answer has CD ≈ 10
  (point-sampling floor), not 0.
- **IoU_all** — IoU with invalid outputs counted as 0. Use this one to compare models: the
  valid-only mean rewards a model for failing on hard parts.
- **dim err** — median |log(extent ratio)| per axis in real units. Normalisation hides
  scale errors; this does not.
- `cqe` adds the official **pass rate** (watertight, one component, bounding box ±1 mm,
  volume ±2 %, Chamfer ≤ 1 mm, 95th-percentile Hausdorff ≤ 1 mm after rigid alignment).

How comparable are our numbers to the published ones?

- `cqe`: directly. The scorer is a re-implementation without open3d; on 300 published
  attempts (150 pass, 150 fail) it reproduces the official verdict in 300/300 cases
  (`t2c bench validate-cqe`).
- `t2cb`: indicative only. Our metric code follows the paper's description (the official
  scorer is unreleased), the public sample is 25 % of the benchmark, and the paper's
  prompt includes five in-context examples that are not public.
- `cadprompt`: only the compile rate; the paper's geometric metrics use a different protocol.

**Prompt styles.** `protocol` is the benchmark's own prompt, used for base and API models.
`native` is the short system prompt our fine-tuned models are trained with
(`src/t2c/prompts.py`). `t2c eval` picks the right one (`--prompt-style` overrides) and
stores it with the result; the report never mixes them silently.

## Tracking results

```
results/runs/<timestamp>-<benchmark>-<model>/summary.json     # tracked in git
results/runs/<timestamp>-<benchmark>-<model>/samples.jsonl.gz # per-sample code, errors, metrics (local)
results/REPORT.md, results/report.html                         # rebuilt after every eval, or `t2c report`
configs/references.yaml                                        # published numbers shown for context
```

Each summary records the model (path, base model, stage `base|sft|grpo|api`, training run,
step), the protocol (prompt style, retries, metric settings), per-group metrics, an error
breakdown, timing, host and git commit. Nothing is overwritten: a re-run is a new entry and
the report takes the latest per (benchmark, model, protocol).

The report has a scoreboard (one row per model, headline metric per benchmark, change
versus the same model before fine-tuning), a full table per benchmark with published
rows greyed out, per-group breakdowns (e.g. L1/L2/L3 × geo/seq), an invalid-output
breakdown, and a per-checkpoint table for every training run.

```bash
t2c runs                                             # list recorded runs
scripts/eval_checkpoints.sh sft-qwen3-1.7b-v1 --bench cqe synth deepcad   # progress curve of a run
t2c eval --model Qwen/Qwen3-4B --bench cqe --limit 5 # smoke test (marked partial, kept out of the report)
t2c eval --model configs/models/my-frontier.yaml     # API model: copy configs/models/api-example.yaml
```

Training curves (loss, reward, valid rate, IoU) go to TensorBoard under
`outputs/train/<run>/`; add `wandb` or `trackio` to `report_to` in a config if you use them.
GRPO also writes `reward_log.jsonl` (one line per step) so runs can be compared without a UI.

## Training

**Stage 1 — SFT** (`t2c train sft configs/sft/*.yaml`). Full fine-tuning with TRL's
`SFTTrainer`. Prompts are rendered with the tokenizer's chat template by the same function
the evaluation backends use, loss on the completion only, samples longer than `max_length`
are dropped rather than truncated. Master weights stay in fp32 with bf16 autocast (bf16
weights cannot absorb 1e-5-sized updates).

**Stage 2 — GRPO** (`t2c train grpo configs/grpo/*.yaml`). TRL's `GRPOTrainer` with vLLM
rollouts on the training GPU. For each prompt the policy samples 8 scripts; each is executed
and compared with the ground-truth solid. Reward (`src/t2c/cad/reward.py`):

| outcome | reward |
|---|---|
| no code / syntax error | 0.0 |
| parses, but raises, times out or crashes the kernel | 0.1 |
| runs, but no solid in `result` | 0.2 |
| valid solid | 0.3 + 0.5·IoU + 0.1·(1 − CD/0.3) + 0.1·exp(−dim_err/0.25) |

The ladder is deliberate: with all-or-nothing rewards a small model sees identical rewards
across a group and learns nothing. Start GRPO from an SFT checkpoint for the same reason.

Every run directory gets a `t2c_meta.json` (base model, stage, parent checkpoint, chat
template settings, config, git commit). That file is what lets `t2c eval --model <dir>`
reproduce the training-time prompt format and attribute the result to the right lineage.

Scaling up:

- Multi-GPU SFT: `accelerate launch --num_processes N -m t2c.cli train sft <config>`.
- Larger models: `configs/sft/qwen3-8b.yaml` fits one H200 (fp32 weights + AdamW ≈ 100 GB).
  For ~27–32B dense models use several GPUs (FSDP/DeepSpeed through accelerate) or switch
  `weights_dtype` to `bfloat16` with a higher learning rate; neither is tested here yet.
- Disk: checkpoints include fp32 weights and optimizer state (≈7 GB each for 0.6B, ≈20 GB
  for 1.7B, ≈100 GB for 8B). `save_total_limit` in the configs bounds it; set
  `train.save_only_model=true` if you do not need to resume.
- GRPO throughput is bounded by CPU cores (8 prompts × 8 samples = 64 CAD executions per
  step). Ask for 48+ cores per GPU; `T2C_CAD_WORKERS` overrides the worker count.
- On the big cluster, the cheapest way to use many GPUs is breadth, not one giant run:
  seeds, data recipes, reward weights and model sizes as independent single-GPU jobs, all
  landing in the same results registry.

## Data

| source | role | verified size | licence |
|---|---|---|---|
| `t2cq` Text-to-CadQuery (`ricemonster/IEEE-T-ASE`) | train | 98,473 train / 5,473 val / 5,472 test | unspecified; derived from Text2CAD (CC BY-NC-SA) |
| `cadcoder` CAD-Coder (`gudo7208/CAD-Coder`) | train | 8,116 high-quality / 57,460 medium / 8,635 val | Apache-2.0 |
| `synth` procedural generator (this repo) | train + benchmark | 20,000 parts × 2 prompts (default) | — |
| `t2cb`, `cqe`, `cadprompt` | benchmark only | | CC BY 4.0 / MIT + do-not-train canary / unspecified |

`t2c data prepare` normalises every reference script to one contract (starts with
`import cadquery as cq`, final solid in `result`, no export or display calls), executes it,
and keeps it only if it yields a valid watertight solid (99 % of the raw pairs survive;
`data/processed/<source>/stats.json` lists why the rest were dropped).

`t2c data synth` generates millimetre-scale parts from 17 families — plates with hole
patterns, flanges, stepped shafts, polygon prisms, L-brackets, U/T/I beams, shelled boxes,
flanged tubes, slotted and pocketed blocks, counterbored plates, revolved profiles, lofts,
swept elbows, gusset brackets, polar patterns, boss plates, wedges. Each part gets a
*geo* prompt (what the finished part looks like) and a *seq* prompt (modelling steps),
the two styles of Text2CAD-Bench, and a script whose executed bounding box and volume are
checked against closed-form values. Train, validation and test parts are disjoint; the
test split is the `synth` benchmark. Its prose is templated (a few phrasings per sentence),
so treat scores on `synth` as an upper bound on how the model handles free-form prompts.

`t2c data build <recipe>` assembles SFT mixtures or RL pools from these; recipes are small
YAML files in `configs/data/`:

| recipe | contents |
|---|---|
| `sft_v1.yaml` | 98k `t2cq` + 7.6k `cadcoder` high + 40k `synth` = 145,948 pairs |
| `sft_deepcad.yaml` | the same without `synth` — the ablation that isolates the distribution effect |
| `rl_v1.yaml` | 6,000 prompts with ground-truth meshes: 3k `synth`, 2k `t2cq`, 1k `cadcoder` |
| `rl_deepcad.yaml` | 4,000 DeepCAD-only prompts |

Licences matter if you publish a model: the DeepCAD-derived corpora inherit a
non-commercial licence, and CadQueryEval must never be trained on.

## The sandbox

Model-written code is executed hundreds of thousands of times, so each script runs in its
own short-lived process forked from a warm "zygote" (≈200 scripts/s on 8 cores):
wall-clock timeout with process-group kill, memory/CPU/file-size limits, no core dumps, a
private scratch directory, an empty network namespace (where unprivileged user namespaces
exist — `t2c doctor` tells you), and a static check that refuses scripts importing
`subprocess`, `socket`, `shutil` and similar. It protects against accidents, not against an
adversary: scripts can still read and write files your user can. Do not feed it untrusted
third-party code.

## Layout

```
src/t2c/cad/        verifier: sandbox (zygote), code extraction, voxel IoU / Chamfer, alignment, reward
src/t2c/data/       pinned sources, corpus verification, dataset recipes, procedural generator
src/t2c/bench/      benchmark adapters + CadQueryEval scorer validation
src/t2c/infer/      model specs and backends (vLLM, transformers, OpenAI-compatible API)
src/t2c/evalrun/    evaluation loop with error-feedback retries
src/t2c/results/    results registry and report
src/t2c/train/      SFT and GRPO
configs/            models, data recipes, training configs, cluster profiles, published references
scripts/            bootstrap, env, Slurm launcher, checkpoint sweep
tests/              verifier, sandbox, eval loop, dataset assembly, generator (pytest)
```
