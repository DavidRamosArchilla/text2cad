---
tags: [test]
aliases: [tests/test_eval.py]
---
# test_eval
**File:** `tests/test_eval.py` · **Tests:** [[evalrun.run]], [[bench.base]], [[results.store]], [[results.report]], [[infer.backends]]

End-to-end test of the evaluation path with a **scripted model** instead of an LLM. A tiny
3-sample benchmark (box, cylinder, "sphere") is created in a temporary directory; the
scripted backend answers the box correctly, the cylinder with a broken script first (and
correctly when it sees the error message), and the third with a refusal.

| Test | What it checks |
|---|---|
| `test_eval_loop_retries_and_metrics` | box valid on attempt 1 with IoU 1; cylinder invalid first, valid after one retry; refusal → `no_code`. Metrics: IR@1 = 2/3, IR = 1/3, IoU (valid) = 1, IoU_all = 2/3, per-group IR. The retry conversation contains the failed script and the "Error Message" turn, with the native system prompt. With 0 retries IR stays 2/3. |
| `test_store_and_report` | two runs (a base model and a fine-tuned one with that base) are saved and listed in order, rows can be loaded back, and the report shows both with the **delta against the base model** (+0.333) in Markdown and HTML |
| `test_model_spec_from_checkpoint` | a `checkpoint-500` directory next to a `t2c_meta.json` becomes a spec named `sft-x@500`, stage `sft`, step 500, native prompt, the training chat-template settings and the right base model |

Why it matters: these are the paths every number in the report goes through.
