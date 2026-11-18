---
tags: [guide]
aliases: [configs]
---
# Configs

Everything you change between experiments is a YAML file in `configs/`. Any field can be
overridden on the command line without editing the file:

```bash
t2c train sft configs/sft/qwen3-1.7b.yaml train.learning_rate=1e-5 run_name=sft-lr1e-5
t2c eval --model configs/models/qwen3-8b.yaml --set sampling.max_tokens=4096
```

(dotted keys, values parsed as YAML — [[train.common]], [[infer.backends]]).

## `configs/models/` — what to evaluate
One file per model that is not a checkpoint of ours. Read by [[infer.backends]].

```yaml
name: qwen3-1.7b            # name in the results
model: Qwen/Qwen3-1.7B      # Hugging Face id or local path
backend: vllm               # vllm | hf | openai
stage: base                 # lineage tag
prompt_style: protocol      # base models get each benchmark's own prompt
chat_template_kwargs: {enable_thinking: false}
sampling: {temperature: 0.0, max_tokens: 2048}
```

- `qwen3-{0.6b,1.7b,4b,8b,14b,32b}.yaml` — base models, thinking off ([[Qwen3]]).
- `qwen3-8b-think.yaml` — the same with thinking on (sampling 0.6/0.95, 6 144 tokens). It
  is the only *-think* file; copy it for other sizes (the comment in `qwen3-1.7b.yaml`
  mentions a `qwen3-1.7b-think.yaml` that does not exist).
- `api-example.yaml` — template for an API model (base URL, the *name* of the environment
  variable holding the key, concurrency, request parameters).

Checkpoints need no file: `t2c eval --model outputs/train/<run>/final` reads
`t2c_meta.json`.

## `configs/data/` — dataset recipes
Which processed data goes into a training set. Read by [[data.build]]; contents explained
in [[Dataset recipes]].

| File | Output | Purpose |
|---|---|---|
| `sft_v1.yaml` | `data/sft/v1` | main SFT mixture |
| `sft_deepcad.yaml` | `data/sft/deepcad` | ablation without the millimetre-scale synthetic parts |
| `sft_v1_heldout.yaml` | `data/sft/v1-heldout` | three synthetic families never trained on (generalisation test) |
| `rl_v1.yaml` | `data/rl/v1` | GRPO prompt pool with ground-truth meshes |
| `rl_deepcad.yaml` | `data/rl/deepcad` | DeepCAD-only RL pool (ablation / smoke tests) |

## `configs/sft/` — stage 1
Read by [[train.sft]]. Top-level keys:

| Key | Meaning |
|---|---|
| `run_name` | output directory `outputs/train/<run_name>/` |
| `base_model` / `init_from` | where to start (HF id or a previous checkpoint) |
| `chat_template_kwargs` | stored with the checkpoint and reused at evaluation |
| `data.dataset`, `data.max_length` | which built dataset; longer samples are dropped |
| `weights_dtype` | `float32` master weights ([[Mixed precision]]) |
| `report_to` | `[tensorboard]`, add `wandb` / `trackio` ([[Experiment tracking]]) |
| `eval_during_training` | benchmark subset every N steps ([[train.callbacks]]) |
| `train:` | passed unchanged to TRL's `SFTConfig` (= transformers `TrainingArguments`): epochs, batch size, learning rate, schedule, eval/save frequency… |

Files: `qwen3-0.6b.yaml` (fast, for learning the plumbing), `qwen3-1.7b.yaml`,
`qwen3-8b.yaml` (gradient checkpointing, ≈100 GB per checkpoint with optimizer state).

## `configs/grpo/` — stage 2
Read by [[train.grpo]]. Same top-level keys as SFT plus:

| Key | Meaning |
|---|---|
| `init_from` | the SFT checkpoint to start from |
| `reward:` | reward ladder and training-time verifier settings ([[cad.reward]]) |
| `chatter_penalty` | subtracted for prose outside the code block |
| `cad_workers` | CAD processes (default: allocated cores − 1) |
| `grpo:` | passed unchanged to TRL's `GRPOConfig`: group size, rollouts per step, lr, temperature, `beta`, vLLM settings… |

Walkthrough of what these do in one step: [[Walkthrough - one GRPO step]].

## `configs/clusters/` — per-cluster settings
Shell files sourced by `scripts/env.sh`: Slurm partition/account, how to ask for GPUs,
CPU and memory per GPU, where data and outputs live. See [[Scripts and clusters]].

## `configs/references.yaml` — published numbers
Numbers from papers shown as grey rows in the report, each with its source and a caveat
explaining how it differs from our protocol. CadQueryEval's leaderboard is loaded from its
dataset automatically. See [[Reading the report]].
