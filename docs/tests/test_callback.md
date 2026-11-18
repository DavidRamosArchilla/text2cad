---
tags: [test]
aliases: [tests/test_callback.py]
---
# test_callback
**File:** `tests/test_callback.py` · **Tests:** [[train.callbacks]], [[evalrun.run]] (`spread`)

No GPU and no real model: the trainer and the model are replaced by small fake objects;
the scoring (sandbox + metrics) is real. A 6-sample benchmark alternates "box" tasks
(answered correctly) and "cylinder" tasks (answered with a script that raises).

| Test | What it checks |
|---|---|
| `test_schedule_and_logging` | with `every_steps=2` over steps 0–5: evaluations at steps 0, 2, 4 and 5 (final weights), not twice at the same step; logged IR = 0.5 and IoU_all = 0.5; exactly the expected metric keys; one `bench_log.jsonl` line per evaluation |
| `test_resume_does_not_rescore_the_start` | a config with `resume_from_checkpoint` turns `at_start` off, so the resumed weights are not logged as "step 0" |
| `test_only_the_main_process_evaluates` | on a non-zero rank nothing is generated or logged |
| `test_a_failure_disables_the_callback_instead_of_the_run` | a backend raising "CUDA out of memory" does not raise into the trainer; the callback disables itself and is not called again |
| `test_unprepared_benchmark_is_skipped` | a benchmark that is not prepared is skipped with a message; the others are still scored |
| `test_spread_subset_covers_all_groups` | `limit=4, spread=True` picks evenly spaced samples covering both groups; plain `limit` takes the first four |
| `test_in_process_backend_restores_training_mode` | generation happens in eval mode with the configured token limit, batch size and bf16 autocast, and the model is put back in training mode — even when generation fails |
| `test_from_config_switch` | no block or `every_steps: 0` → no callback; values from the config reach the callback; no executor (non-main rank) → no callback |

Why it matters: this code runs inside multi-hour training jobs; the main requirement is
that it can never be the reason a run dies.
