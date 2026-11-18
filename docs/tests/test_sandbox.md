---
tags: [test]
aliases: [tests/test_sandbox.py]
---
# test_sandbox
**File:** `tests/test_sandbox.py` · **Tests:** [[cad.sandbox]], [[cad.codeutil]], [[cad.reward]]

| Test | What it checks |
|---|---|
| `test_error_taxonomy` | each failure gets the right `error_type` and stage: valid box → stage 3; no code; syntax error; impossible fillet → `runtime_error` with "line 2" in the message; no `result` variable; `result` is a 2D sketch → `no_solid`; `import subprocess` → `unsafe_code` |
| `test_timeout_and_crash_are_contained` | an infinite loop is killed (`timeout`), a forced segfault is reported as `crash`, and **the pool still works afterwards** |
| `test_scratch_dir_and_no_network` | a script can write a file in its scratch directory; if network isolation is active, opening a socket fails with "unreachable" |
| `test_order_is_preserved` | 40 jobs run in parallel come back in the order they were sent |
| `test_extract_code` | takes the code from a fenced block; prefers the *last* block; ignores code inside `<think>`; accepts a cut-off fence; returns nothing for a refusal |
| `test_normalize_reference_script` | export/visualisation lines are removed and the exported variable becomes `result`; a script without cadquery is rejected |
| `test_reward_is_monotone` | the reward ladder strictly increases from "no code" to "perfect part" and a perfect part scores 1.0 |

Why it matters: the sandbox runs every script of every evaluation and RL step; a
mis-classified error changes IR and the reward, and a crash that escaped would kill a
training run. Concept: [[Sandboxing untrusted code]].
