---
tags: [module]
aliases: [prompts.py, prompt styles]
---
# prompts
**File:** `src/t2c/prompts.py` · **Libraries:** none

All prompt text in one file, and the two functions that build conversations.

## Two prompt styles
Different models are asked in different ways, and the style is stored with every result
so the report never mixes them silently ([[Reading the report]]).

| Style | Text | Used for |
|---|---|---|
| `native` | `NATIVE_SYSTEM`: short rules — start with `import cadquery as cq`, put the solid in `result`, no export/visualisation, only a ```` ```python ```` block | our fine-tuned models (they are trained with exactly this prompt) |
| `protocol` | the benchmark's own published prompt: `T2CB_SYSTEM` for [[Text2CAD-Bench]], `CQE_SYSTEM` + `CQE_TASK_TEMPLATE` for [[CadQueryEval]]; `NATIVE_SYSTEM` for benchmarks without one | base and API models, so their numbers compare with the literature |

`T2CB_SYSTEM` is the *zero-shot core* of the Text2CAD-Bench prompt: the paper prints its
rules, but its five in-context examples are only in supplementary material that is not
public. `CQE_SYSTEM` / `CQE_TASK_TEMPLATE` are copied verbatim from CadQueryEval (MIT).

## Retry message
`RETRY_TEMPLATE` is Text2CAD-Bench's error-feedback turn (paper, appendix D.2): "The
previous CadQuery code attempt failed with an error. ## Error Message: … Please analyze the
error and generate corrected code." `retry_messages(messages, response, error)` appends the
failed answer and this turn to the conversation. Used by [[evalrun.run]].

## Functions
- `build_messages(task, system)` → `[{"role": "system", …}, {"role": "user", …}]`.
- `format_completion(code)` → the code inside a ```` ```python ```` fence: *exactly* what a
  fine-tuned model should output. [[train.sft]] uses it to build the training target, so
  training and evaluation agree on the format.

See also [[Tokenizers and chat templates]] for how these messages become model input.
