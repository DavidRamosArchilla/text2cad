---
tags: [test]
aliases: [tests/test_build.py]
---
# test_build
**File:** `tests/test_build.py` · **Tests:** [[data.build]]

| Test | What it checks |
|---|---|
| `test_holdout_objects_never_reach_training` | Small fake corpora are written: `t2cq` train/val/test and a `cadcoder` split that describes two of the held-out objects again. With `holdout: [t2cq:val, t2cq:test]`: training contains none of the held-out objects (**even when they come from the other corpus**), validation contains exactly the held-out val objects, rows have the SFT format, and building twice gives **byte-identical** files. The same recipe as `kind: rl` writes one ground-truth mesh per row. |
| `test_exclude_families` | `exclude_families: [loft]` removes those synthetic parts from training |

Why it matters: a leak of benchmark objects into training would make benchmark numbers
meaningless without any visible error ([[Benchmark contamination]]); determinism is what
makes the two clusters produce the same datasets ([[Reproducible environments]]).
