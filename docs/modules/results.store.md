---
tags: [module, results]
aliases: [store.py, results registry, t2c runs]
---
# results.store
**File:** `src/t2c/results/store.py` · **Libraries:** `gzip`, `json`, `subprocess` (git)
**Guide:** [[Reading the report]]

The results registry: one directory per evaluation run, never overwritten.

```
results/runs/20261115-214634-cqe-sft-qwen3-1.7b-v1/
  summary.json       # small, committed to git → results sync between clusters
  samples.jsonl.gz   # one row per sample (code, answer, errors, metrics) → local only
```

**Run id** = `<YYYYmmdd-HHMMSS>-<benchmark>-<model name>`.

**`summary.json`** holds: `run_id`, `created`, `benchmark`, `bench_sha1` (which version of
the benchmark), `model` (the full model spec: path, stage, base model, training run, step,
sampling), `metrics` (overall + per group), `errors` (error types on first and final
attempt), `protocol` (prompt style, retries, settings), `timing`, `tags`, `note`, `partial`
(true for `--limit` smoke tests), plus `host`, `git` (commit + whether the tree was dirty)
and the `t2c` version.

Functions: `save_run(summary, rows)`, `list_runs(include_partial=False)` (sorted by
time), `load_rows(run_id)`, `new_run_id(model, benchmark)`, `git_state()`.

`.gitignore` excludes `results/runs/*/samples.jsonl.gz`, so a `git pull` brings the other
cluster's summaries but not their (large) per-sample files.

Inspect one run's samples:
```bash
zcat results/runs/<run-id>/samples.jsonl.gz | head -1 | python -m json.tool
```

**Used by:** [[evalrun.run]], [[results.report]], [[cli]] (`t2c runs`), [[train.common]] (`git_state`).
**Tested in:** [[test_eval]] (`test_store_and_report`).
