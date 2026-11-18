---
tags: [module, benchmark]
aliases: [cqe.py, validate-cqe]
---
# bench.cqe
**File:** `src/t2c/bench/cqe.py` · **Libraries:** [[Libraries#pandas and pyarrow|pandas]] + [[cad.sandbox]]
**Benchmark:** [[CadQueryEval]]

Answers one question: *do our CadQueryEval verdicts agree with the official scorer?*
Our scorer ([[cad.metrics]] `cqe_checks`) is a re-implementation without open3d, so its
pass rates are only comparable with the public leaderboard if it gives the same verdicts.

`validate_against_published(limit=400, seed=0)`:
1. Read every leaderboard attempt shipped with the dataset (`samples.parquet`: model, task,
   the code it wrote, the official `passed` and per-check results).
2. Pick a balanced sample: half officially passed, half failed (so both kinds of
   disagreement can show up).
3. Re-score each script with our verifier in protocol mode.
4. Report overall agreement, false passes / false fails, per-check agreement and up to 15
   disagreements with details.

Result on this cluster (`t2c bench validate-cqe --limit 300`): **300 / 300 verdicts agree**;
per check ≥ 99 % (the rare differences are in the component count and Hausdorff check on
borderline parts, and never flip the final verdict in that sample).

Re-run it after changing anything in [[cad.metrics]].

**Used by:** [[cli]] (`t2c bench validate-cqe`).
