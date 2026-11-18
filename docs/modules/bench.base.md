---
tags: [module, benchmark]
aliases: [base.py, registry.py, Benchmark class, bench.registry]
---
# bench.base
**Files:** `src/t2c/bench/base.py`, `src/t2c/bench/registry.py` · **Libraries:** standard library (`statistics`, `hashlib`, `json`)
**Benchmarks:** [[Benchmarks overview]] · [[Metrics]]

The common interface every benchmark implements, and the code that computes the core
metrics — so all five benchmarks report numbers that mean the same thing.

## `Sample`
One task: `id`, `task` (text shown to the model), `group` (for per-group tables, e.g.
`L2/geo`), `gt` (path of the ground-truth mesh, relative to the benchmark directory, or
`None`), `meta` (anything else).

## `Benchmark`
Class attributes each benchmark sets:

| Attribute | Meaning | Default |
|---|---|---|
| `name`, `description` | identity | — |
| `protocol_system` | system prompt for `--prompt-style protocol` | `NATIVE_SYSTEM` |
| `default_retries` | error-feedback retries the benchmark's protocol allows | 0 |
| `headline` | metric shown on the scoreboard | `iou_all` |
| `res`, `n_points`, `timeout` | verifier settings | 256, 30 000, 60 s |

Methods:
- **Storage** — a prepared benchmark is `data/bench/<name>/` with `samples.jsonl`,
  `gt/*.npz` and `info.json` (provenance + a SHA-1 of the samples, stored with every
  result so you can tell if two clusters scored the same benchmark). `prepare()`
  (benchmark-specific), `_write()`, `load()`, `is_prepared()`, `info()`.
- **Protocol hooks** — `system_prompt(style)`; `make_job(sample, code, style)` builds the
  verifier job; `row_extra()` / `aggregate_extra()` let a benchmark add its own fields and
  metrics (CadQueryEval adds its six checks).
- **Scoring** — `aggregate(rows)` → `{"overall": {...}, "groups": {group: {...}}}`.

## Core metrics (`_agg`)
| Key | Meaning |
|---|---|
| `n` | samples scored (harness errors excluded) |
| `harness_errors` | samples where *our* code failed — excluded and reported |
| `ir`, `ir_first` | invalid rate after retries / on the first attempt |
| `iou`, `iou_median` | IoU over **valid** samples only (paper convention) |
| `iou_all` | IoU summed over valid samples ÷ all samples (invalid = 0) |
| `cd`, `cd_median` | Chamfer distance × 1000 over valid samples |
| `dim_err_median` | median size/orientation error over valid samples |

Definitions and pitfalls: [[Metrics]].

## `registry.py`
`names()`, `get(name)`, `resolve(["all"])` (= every *prepared* benchmark). Adding a
benchmark = write a `Benchmark` subclass in [[bench.suites]] and add it here.

**Used by:** [[evalrun.run]], [[train.callbacks]], [[results.report]], [[data.build]] (`bench:` holdouts), [[cli]].
**Tested in:** [[test_eval]], [[test_callback]] (both create a tiny benchmark through this interface).
