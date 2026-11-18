---
tags: [module, training]
aliases: [common.py, t2c_meta.json]
---
# train.common
**File:** `src/t2c/train/common.py` · **Libraries:** [[Libraries#PyYAML|PyYAML]]

Plumbing shared by [[train.sft]] and [[train.grpo]].

| Function | What it does |
|---|---|
| `load_config(path, overrides)` | read the YAML config and apply `key.sub=value` overrides from the command line (values parsed as YAML: `1e-5` → float) |
| `run_dir(cfg)` | `outputs/train/<run_name>/` (created if needed) |
| `resolve_init(cfg)` | where training starts: `init_from` may be a HF id or a checkpoint directory (absolute, or relative to `outputs/`); returns the path, the original base model (for lineage) and the parent run's metadata |
| `write_meta(...)` | writes `t2c_meta.json` and a copy of the config into the run directory |
| `is_main_process()` | true on rank 0 of a multi-GPU run (only it writes metadata and runs benchmark scoring) |
| `bench_eval_enabled(cfg)` | is `eval_during_training.every_steps` > 0 ([[train.callbacks]]) |

## `t2c_meta.json` — makes a checkpoint self-describing
```json
{"train_run": "sft-qwen3-1.7b-v1", "stage": "sft", "base_model": "Qwen/Qwen3-1.7B",
 "parent": null, "chat_template_kwargs": {"enable_thinking": false},
 "system_prompt": "...", "created": "...", "git": {...}, "config": {...}, "data": {...}}
```
Because of this file, `t2c eval --model outputs/train/<run>/final` (or `checkpoint-N`)
knows how to prompt the model exactly as in training and how to label the result
(stage, lineage, step) — see [[infer.backends]]. A GRPO run started from an SFT
checkpoint records it as `parent` and keeps the original `base_model`, so the report can
compare it with the untouched base model.

**Tested in:** [[test_eval]] (`test_model_spec_from_checkpoint` reads a meta file), [[test_callback]] (`bench_eval_enabled`).
