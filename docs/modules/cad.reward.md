---
tags: [module, verifier, training]
aliases: [reward.py, RewardConfig, reward ladder]
---
# cad.reward
**File:** `src/t2c/cad/reward.py` · **Libraries:** standard library only
**Concepts:** [[GRPO]] · [[Reinforcement learning for LLMs]]

Turns one verifier result ([[cad.sandbox]]) into one number for reinforcement learning.

## The reward ladder
| Outcome (verifier stage) | Reward (defaults) |
|---|---|
| 0 — no code, syntax error, refused | `r_none` = 0.0 |
| 1 — parses, but raises / times out / crashes | `r_parsed` = 0.1 |
| 2 — runs, but no solid in `result` | `r_ran` = 0.2 |
| 3 — valid solid | `r_valid` 0.3 + `w_iou` 0.5 × IoU + `w_cd` 0.1 × clip(1 − CD/0.3) + `w_dim` 0.1 × exp(−dim_err/0.25) |

A perfect part scores ≈ 1.0.

## Why a ladder
GRPO learns from *differences* inside a group of answers to the same prompt. If the
reward were "1 if perfect, else 0", most groups of a small model would be all zeros and
teach nothing. Partial credit for getting closer (it runs; it builds a solid; the shape
overlaps; the size is right) keeps a signal alive. The three geometric terms cover
different failures:
- **IoU** — shape overlap (scale-free);
- **CD term** — still informative when IoU ≈ 0 (thin walls, small offsets);
- **dim term** — absolute size and orientation, invisible after normalisation.

## `RewardConfig`
Holds the weights above plus the *training-time* verifier settings, cheaper than the
benchmark ones: `res = 64` voxels, `n_points = 4000`, `timeout = 20 s`. All fields can be
changed in the `reward:` block of a GRPO config ([[Configs]]).

The extra "chatter penalty" (−0.05 for prose around the code) is applied in
[[train.grpo]], not here.

**Used by:** [[train.grpo]]. **Tested in:** [[test_sandbox]] (`test_reward_is_monotone`),
[[test_grpo_reward]].
