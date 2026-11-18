---
tags: [module, training]
aliases: [sft.py, t2c train sft]
---
# train.sft
**File:** `src/t2c/train/sft.py` · **Libraries:** [[Libraries#TRL|TRL]] (`SFTTrainer`, `SFTConfig`), [[Libraries#transformers|transformers]], [[Libraries#datasets|datasets]], [[Libraries#torch|torch]]
**Concepts:** [[Supervised fine-tuning]] · [[Full fine-tuning vs LoRA]] · [[Mixed precision]] · [[Tokenizers and chat templates]]
**Config:** `configs/sft/*.yaml` ([[Configs]])

Stage 1: full fine-tuning on (description → script) pairs.

```bash
t2c train sft configs/sft/qwen3-1.7b.yaml [key=value …]
accelerate launch --num_processes 2 -m t2c.cli train sft configs/sft/qwen3-1.7b.yaml   # several GPUs
```

## What `main()` does
1. Load the config ([[train.common]]). If in-training benchmark scoring is on, start the
   CAD executor now, before torch loads (main process only).
2. Tokenizer of the start model; read the built dataset (`data.dataset`, e.g. `sft/v1`,
   see [[Dataset recipes]]).
3. **`_encode`** — for every row:
   - prompt = `render_prompt(tokenizer, [system NATIVE_SYSTEM, user text], chat_template_kwargs)`
     — the *same* function the evaluation backends use ([[infer.backends]]);
   - completion = `format_completion(code)` (a ```` ```python ```` block) + end-of-sequence token;
   - `input_ids` = prompt + completion tokens; `labels` = −100 on the prompt tokens (no
     loss there), the completion tokens otherwise → the model is trained only to *produce
     the answer*;
   - samples longer than `max_length` (2 048) are **dropped, not truncated** — a cut-off
     script would teach the model to stop mid-code.
4. Print token-length statistics and one example exactly as the model sees it (check this
   in the job log!).
5. Write `t2c_meta.json` ([[train.common]]).
6. Load the model with **fp32 weights** (`weights_dtype`); `bf16: true` in the `train:`
   block makes the maths run in bf16 ([[Mixed precision]]).
7. `SFTConfig(**cfg["train"])` — the `train:` block goes straight to TRL / transformers
   (epochs, batch size, learning rate, schedule, eval and save frequency…).
8. `SFTTrainer(...)` with the benchmark callback ([[train.callbacks]]) when enabled; train
   (optionally `resume_from_checkpoint`).
9. Save `final/` (weights + tokenizer + `trainer_state.json`).

## What gets written
`outputs/train/<run_name>/`: `checkpoint-N/` every `save_steps`, `final/`,
`t2c_meta.json`, `config.yaml`, TensorBoard events, `bench_log.jsonl`.

Measured on one H200: Qwen3-0.6B ≈ 51 samples/s; Qwen3-1.7B ≈ 35 samples/s — two epochs
of `sft/v1` (9 104 steps of 32) took 2 h 20 min.

**Tested in:** no GPU test of the trainer itself; prompt rendering and lineage are covered
by [[test_eval]], the callback by [[test_callback]].
