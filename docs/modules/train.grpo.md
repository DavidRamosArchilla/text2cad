---
tags: [module, training]
aliases: [grpo.py, CadReward, t2c train grpo]
---
# train.grpo
**File:** `src/t2c/train/grpo.py` · **Libraries:** [[Libraries#TRL|TRL]] (`GRPOTrainer`, `GRPOConfig`), [[Libraries#vLLM|vLLM]] (through TRL), [[Libraries#transformers|transformers]], [[Libraries#datasets|datasets]], [[Libraries#torch|torch]]
**Concepts:** [[GRPO]] · [[Reinforcement learning for LLMs]] · [[vLLM]]
**Guide:** [[Walkthrough - one GRPO step]] · **Config:** `configs/grpo/*.yaml` ([[Configs]])

Stage 2: reinforcement learning where the reward is the CAD verifier.

```bash
t2c train grpo configs/grpo/qwen3-1.7b.yaml
```

Start from an SFT checkpoint (`init_from`): GRPO needs a model that already produces valid
geometry fairly often, otherwise every answer in a group gets the same (low) reward and
there is nothing to learn.

## `CadReward` — the reward function TRL calls
TRL calls it with the prompts, the sampled completions and every extra dataset column as
keyword arguments — here `gt`, the path of each prompt's ground-truth mesh.

1. `extract_code` on each completion ([[cad.codeutil]]).
2. One verifier job per completion with the training-time settings (64³ voxels, 4 000
   points, 20 s), all run in parallel by the executor ([[cad.sandbox]]).
3. `reward_from_result` ([[cad.reward]]).
4. **Chatter penalty**: −`chatter_penalty` (0.05) when more than 20 characters of text sit
   outside the code block — prose costs tokens and is the usual way a policy drifts under RL.
5. Logs `cad/valid_rate`, `cad/iou_valid`, `cad/iou_all`, `cad/verify_s` and the share of
   each error type through TRL's `log_metric` (→ TensorBoard) and to
   `outputs/train/<run>/reward_log.jsonl`, one line per step.

## `main()`
1. Load config; reuse the parent checkpoint's chat-template settings.
2. Build `RewardConfig` from the `reward:` block; **start the CAD executor** (before torch/vLLM).
3. `configure_vllm_env()`, import torch / TRL.
4. Dataset rows `{"prompt": [system, user], "gt": "<absolute path>"}` from the RL recipe
   (`data/rl/v1`, see [[Dataset recipes]]).
5. Model with fp32 weights; `t2c_meta.json` (stage `grpo`, parent = the SFT checkpoint).
6. `GRPOConfig(**cfg["grpo"])`, `GRPOTrainer(model, reward_funcs=CadReward, …)` with the
   benchmark callback ([[train.callbacks]]); train; close the executor; save `final/`.

Key `grpo:` settings (see [[Walkthrough - one GRPO step]]): `num_generations: 8` (group
size), `per_device_train_batch_size: 64` (= 8 prompts × 8 answers), `temperature: 1.0`,
`learning_rate: 2e-6`, `beta: 0.0` (no KL penalty), `use_vllm: true` with
`vllm_mode: colocate`.

**Tested in:** [[test_grpo_reward]] (the reward function, called as TRL calls it); the
trainer itself only in GPU smoke runs.
