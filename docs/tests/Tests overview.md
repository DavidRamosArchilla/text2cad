---
tags: [test, index]
aliases: [Tests, pytest]
---
# Tests overview

44 tests in `tests/`, run with [[Libraries#pytest|pytest]]:

```bash
source scripts/env.sh
pytest                      # all tests, ≈ 30 s on the login node
pytest tests/test_geometry.py -k iou   # one file, tests whose name contains "iou"
pytest -x --durations=10    # stop at the first failure, show the slowest tests
```

No GPU, no model download and no benchmark download are needed: everything runs on the
CPU with small shapes built on the spot.

## What is real and what is faked
- **Real:** the CAD kernel, the sandbox, every metric, the evaluation loop, the results
  store and report, dataset assembly, the generator. A test that passes means the actual
  code path works.
- **Faked:** the language model. Tests that need answers use a *scripted backend* — a tiny
  class that returns pre-written answers (one correct script, one broken one, one "I cannot
  do that"), so the expected metrics are known exactly. Training loops (TRL trainers) are
  not run; the pieces we wrote around them are (reward function, callback).
- **Isolated:** tests point `T2C_DATA_DIR` / `T2C_RESULTS_DIR` at pytest's temporary
  directory (`monkeypatch.setenv`), so they never touch your real data or results.

## The shared fixture (`tests/conftest.py`)
`executor` — one `CadExecutor` with 4 workers and a 15 s timeout, created once per test
session (starting it costs ~2.5 s) and shared by all tests that run CadQuery.

## Files
| File | Tests | Protects |
|---|---|---|
| [[test_geometry]] | 9 | the numbers: voxel IoU, Chamfer, normalisation, alignment, CadQueryEval checks |
| [[test_sandbox]] | 7 | safe execution, error classification, code extraction, script normalisation, reward ordering |
| [[test_eval]] | 3 | the evaluation loop with retries, the results store and report, checkpoint model specs |
| [[test_build]] | 2 | dataset assembly — above all that held-out objects never reach training |
| [[test_synth]] | 14 | the generator: valid geometry, scripts match their descriptions, reproducible disjoint splits |
| [[test_grpo_reward]] | 1 | the GRPO reward function as TRL calls it |
| [[test_callback]] | 8 | in-training benchmark scoring: schedule, logging, and never breaking a run |

(`test_synth` counts 14 because one test runs once per family in a list of 7.)

## What is *not* covered by tests
- The TRL trainers and vLLM themselves (tested in their own projects; our use of them was
  checked in short GPU smoke runs).
- Benchmark preparation from the real downloads ([[bench.suites]]) — checked by running it
  and by [[bench.cqe]]'s comparison with the official CadQueryEval verdicts.
- The shell scripts and the command-line parsing.

When you change code, run the tests of the matching note's "Tested in" line first, then
the whole suite.
