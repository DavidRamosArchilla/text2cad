---
tags: [module, training]
aliases: [callbacks.py, BenchmarkEvalCallback, eval_during_training, bench_log.jsonl]
---
# train.callbacks
**File:** `src/t2c/train/callbacks.py` · **Libraries:** [[Libraries#transformers|transformers]] (`TrainerCallback`), [[Libraries#torch|torch]]
**Concepts:** [[Experiment tracking]] · **Config:** `eval_during_training:` block ([[Configs]])

Benchmark scores **during** training, logged next to the loss, so you can see whether
lower loss / higher reward actually means better parts — and when it stops doing so.

```yaml
eval_during_training:
  every_steps: 1000        # SFT configs; 100 in GRPO configs; 0 = off
  benchmarks: [cqe, deepcad]
  limit: 48                # samples per benchmark: a fixed, evenly spread subset
  max_new_tokens: 768
  batch_size: 32
  at_start: true           # also score the untouched model at step 0
```

## How it works
- `BenchmarkEvalCallback` is a transformers `TrainerCallback`. The Trainer calls its
  hooks: `on_train_begin` (step 0, if `at_start`), `on_step_end` (every `every_steps`),
  `on_train_end` (the final weights, unless that step was just scored).
- At each of those points it builds an `InProcessBackend` around the **live training
  model**: switch to eval mode, generate greedily with `hf_generate` ([[infer.backends]])
  under bf16 autocast, switch back to training mode (even if generation fails).
- It runs `evaluate(..., style="native", retries=0, limit=48, spread=True)` from
  [[evalrun.run]] for each listed benchmark — the same scoring code as `t2c eval`, on a
  fixed subset that covers every group.
- Logs `eval_<benchmark>_{ir, iou_all, iou, pass_rate}` and `eval_bench_seconds` through
  `trainer.log()`, so they reach TensorBoard / W&B / trackio as `eval/cqe_pass_rate`, …;
  and appends one line per benchmark to `outputs/train/<run>/bench_log.jsonl`.

## Safety rules
- **Never breaks training**: any exception is printed and disables the callback for the
  rest of the run; training continues.
- Only the main process (rank 0) evaluates in multi-GPU runs.
- Benchmarks that are not prepared on this machine are skipped with one message.
- A resumed run does not re-score "step 0" (the Trainer reports step 0 before restoring
  the step counter, so `at_start` is switched off when `resume_from_checkpoint` is set).

## Cost and meaning
≈ 2–2.5 min per evaluation with three benchmarks for Qwen3-0.6B on an H200 (≈ 4 min for an
untrained model that writes until the token limit). The numbers are a monitoring signal:
48 samples, one attempt, plain `generate`. Compare them along one run; report numbers from
`t2c eval` on checkpoints ([[Reading the report]]).

`from_config(cfg, out_dir, executor, tokenizer, chat_kwargs)` builds the callback (or
returns `None` when off); [[train.sft]] and [[train.grpo]] call it.

**Tested in:** [[test_callback]].
