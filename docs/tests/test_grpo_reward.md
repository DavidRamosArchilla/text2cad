---
tags: [test]
aliases: [tests/test_grpo_reward.py]
---
# test_grpo_reward
**File:** `tests/test_grpo_reward.py` · **Tests:** [[train.grpo]] (`CadReward`), [[cad.reward]]

One test, `test_reward_orders_completions`, calls the reward function exactly the way TRL's
`GRPOTrainer` does (prompts, a list of completions in chat format, the `gt` column, a
`log_metric` callback) with six answers against a ground-truth box:

| Answer | Expected reward |
|---|---|
| the exact box | > 0.95 |
| a valid box with wrong proportions | lower |
| the exact box wrapped in chatty prose | exactly 0.05 less than the exact box |
| a script that raises (impossible fillet) | lower still |
| a syntax error | 0.0 |
| the exact box as a plain string (not chat format) | same as the exact box |

It also checks the logged statistics (`cad/valid_rate` = 4/6) and the first line of
`reward_log.jsonl`.

Why it matters: the reward is the only thing GRPO optimises. A wrong ordering here
would train the model towards the wrong behaviour, silently.
