---
tags: [module, results]
aliases: [report.py, t2c report]
---
# results.report
**File:** `src/t2c/results/report.py` · **Libraries:** [[Libraries#PyYAML|PyYAML]], `html`
**Guide:** [[Reading the report]] (what the tables mean)

Builds `results/REPORT.md` and `results/report.html` from the run summaries.

## How it decides what to show
- `latest_runs`: for each (benchmark, model name, prompt style, retries) keep the newest run.
- `_base_run`: for a fine-tuned run, find the `base` run of its `base_model` on the same
  benchmark (preferring the same retries, then the protocol prompt) → the "(+x)" deltas.
- `HEADLINE`: the scoreboard metric per benchmark (`t2cb` IR, `cqe` pass rate, others
  `iou_all`); `SCOREBOARD_EXTRA` adds `cqe` IoU_all because pass/fail is too coarse for
  small models.
- `_load_references`: published rows from `configs/references.yaml`; for CadQueryEval the
  91 published models from `published.json`, trimmed to the top 10 + Qwen-family models.

## Sections it writes
1. **Scoreboard** — one row per model × prompt style.
2. **Per benchmark** — leaderboard (ours + published, sorted by the headline metric),
   by-group table, invalid-reason table.
3. **Training runs** — headline metric per evaluated checkpoint of each training run.

`Table` renders the same data as Markdown and as HTML; the HTML has inline CSS (light and
dark mode) and a few lines of JavaScript for sortable columns. No external files, so
`report.html` can be copied anywhere.

**Used by:** [[evalrun.run]] (after every full eval), [[cli]] (`t2c report`).
**Tested in:** [[test_eval]] (`test_store_and_report`).
