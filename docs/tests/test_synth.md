---
tags: [test]
aliases: [tests/test_synth.py]
---
# test_synth
**File:** `tests/test_synth.py` · **Tests:** [[data.synth]]

The generator produces training data *and* the ground truth of a benchmark, so its three
outputs per part — script, geo text, seq text — must agree with each other and with the
geometry the script really builds.

| Test | What it checks |
|---|---|
| `test_registry_covers_enough_families` | at least 14 part families exist (there are 17) |
| `test_every_family_builds_one_watertight_solid` | every family × 10 seeds, executed: one solid, closed surface, passes `check_part` |
| `test_scripts_follow_the_project_contract` | every script starts with `import cadquery as cq`, assigns `result`, has no export/visualisation |
| `test_generation_is_deterministic` | same seed → identical code, texts and bounding box; different seed → different part |
| `test_executed_bbox_equals_stated_dimensions` (×7 families) | the bounding box the script builds equals the intended one (and the volume equals the closed-form value) |
| `test_both_prompts_state_every_key_dimension` | for 40 seeds per family, both texts mention every key dimension (a diameter may be given as a radius) and refer to an axis |
| `test_feature_positions_match_the_text` | the hole positions written in the L-bracket text are really empty in the solid, and there is material next to them; same for flange bolt holes (checked with `isInside` inside the sandbox) |
| `test_generate_dataset_writes_disjoint_verified_splits` | a small real `generate_dataset` run: counts, geo/seq pairing, row format and uid pattern, **no script shared between train, val and test**, per-family stats |

Why it matters: if a text said "hole at X = 36" while the script put it at X = 30, the
model would be trained — and graded — on contradictions.
