---
tags: [module, evaluation]
aliases: [run.py, t2c eval, evaluate]
---
# evalrun.run
**File:** `src/t2c/evalrun/run.py` · **Libraries:** standard library; uses [[infer.backends]], [[cad.sandbox]], [[bench.base]], [[results.store]]
**Guide:** [[Walkthrough - scoring one answer]]

The evaluation loop behind `t2c eval`. Same code for a base model, a checkpoint or an API
model; only the model spec differs.

## `evaluate(spec, backend, bench, executor, style, retries, limit=None, spread=False)`
1. **Samples** — `bench.load()`; `limit` keeps the first N (smoke tests) or, with
   `spread=True`, N samples evenly spaced over the benchmark so every group is represented
   (used by [[train.callbacks]]).
2. **Conversations** — `build_messages(task, bench.system_prompt(style))` ([[prompts]]).
3. **Rounds** (1 + `retries`):
   - generate answers for the still-pending samples (one batched call);
   - `extract_code` ([[cad.codeutil]]); `bench.make_job` → `executor.run` ([[cad.sandbox]]);
   - every sample that failed (and was not a harness error) gets a retry conversation:
     its previous answer (code only, thinking removed) + the error-feedback message.
4. **Rows** — one per sample: `id`, `group`, `valid`, `first_valid`, `n_attempts`,
   `error_type`, `error`, `first_error_type`, `iou`, `cd`, `dim_err`, `volume`, `bbox`,
   final `code`, raw `response` (≤ 20 000 characters), plus benchmark extras (`cqe_*`).
5. **Aggregate** — `bench.aggregate(rows)` ([[bench.base]]), counts of error types on the
   first and last attempt, and a `protocol` record (prompt style, retries, a hash of the
   system prompt, metric settings).

Returns `(summary_parts, rows, timing)`.

## `main(args)` — what `t2c eval` does
1. Load the model spec; choose the prompt style (`auto` = the spec's).
2. Resolve benchmarks (`all` = every prepared one).
3. **Start the CAD executor, then the backend** (workers fork from a small process).
4. For each benchmark: retries = the benchmark's protocol unless `--retries`; run
   `evaluate`; save the run ([[results.store]]) with model, protocol, metrics, errors,
   timing, tags, note; print one summary line; warn about harness errors.
5. Close the backend and executor; rebuild the report ([[results.report]]) unless it was a
   `--limit` smoke test (those are saved as `partial` and kept out of the report).

**Used by:** [[cli]], [[train.callbacks]] (`evaluate` only).
**Tested in:** [[test_eval]] (loop, retries, metrics), [[test_callback]] (spread subset).
