---
tags: [module]
aliases: [paths.py]
---
# paths
**File:** `src/t2c/paths.py` · **Libraries:** `os`, `pathlib` only

One place that decides where files go, so nothing else hard-codes a path. Three roots,
each overridable by an environment variable (set in the cluster profile, see
[[Scripts and clusters]]):

| Variable | Default | Holds |
|---|---|---|
| `T2C_DATA_DIR` | `<repo>/data` | raw downloads, processed corpora, benchmarks, built datasets |
| `T2C_OUTPUT_DIR` | `<repo>/outputs` | training runs, Slurm logs |
| `T2C_RESULTS_DIR` | `<repo>/results` | evaluation results and the report |

Helpers: `raw_dir(name)` → `data/raw/<name>`, `processed_dir(name)` →
`data/processed/<name>`, `bench_dir(name)` → `data/bench/<name>`, `ensure(path)` creates a
directory. `REPO_ROOT` is computed from the file location.

The tests rely on this: they point `T2C_DATA_DIR` / `T2C_RESULTS_DIR` at a temporary
directory so they never touch your real data ([[Tests overview]]).
