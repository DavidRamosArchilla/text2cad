---
tags: [concept]
aliases: [greedy decoding, temperature, sampling, top-p]
---
# Decoding — greedy vs sampling

At each step the model gives a probability for every possible next token; *decoding* is
how one is chosen.

| Strategy | How | Used here for |
|---|---|---|
| **greedy** (`temperature: 0`) | always the most likely token → deterministic | all benchmark evaluations and the in-training subset: reproducible numbers |
| **sampling** with **temperature** T | divide scores by T before the softmax: T < 1 sharper, T > 1 flatter | GRPO rollouts (T = 1.0): a group needs *different* answers |
| **top-p** (nucleus) | sample only among the most likely tokens covering probability p | Qwen3 thinking mode (T 0.6, top-p 0.95, as Qwen recommends) |

`max_tokens` / `max_new_tokens` caps the answer length; an answer cut off at the limit
usually has an unterminated code block — `extract_code` still tries to use it
([[cad.codeutil]]), and GRPO ignores such answers (`mask_truncated_completions`).

A related evaluation idea is **pass@k**: the chance that at least one of k samples is
correct ([[Evaluating code generation]]). Our benchmarks report a single greedy attempt
(plus retries where the protocol allows).

## Learn more
- **Hugging Face blog — How to generate text** (greedy, beam search, sampling, top-k, top-p):
  <https://huggingface.co/blog/how-to-generate>
- **transformers — Generation strategies**: <https://huggingface.co/docs/transformers/generation_strategies>
