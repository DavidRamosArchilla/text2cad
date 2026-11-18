---
tags: [concept]
aliases: [Qwen, thinking mode]
---
# Qwen3

Alibaba's open-weight model family (2025, Apache-2.0). Dense models of **0.6B, 1.7B, 4B,
8B, 14B and 32B** parameters (plus mixture-of-experts models), all with the same tokenizer
and chat format — a clean ladder for "how does the result change with model size".

**Hybrid thinking.** Qwen3 can first write reasoning inside `<think>…</think>` and then
answer. The chat-template option `enable_thinking` switches it; with `false` the template
inserts an empty `<think></think>` block and the model answers directly
([[Tokenizers and chat templates]]). Here thinking is **off** by default (shorter answers,
cheaper training); `configs/models/qwen3-8b-think.yaml` evaluates the reasoning variant
(Qwen recommends sampling at temperature 0.6, top-p 0.95 in that mode — [[Decoding]]).

## Why Qwen3 here
Text-only causal LMs across many dense sizes, mature support in transformers, TRL and
vLLM, and enough Python knowledge to start from. Measured base-model results (all five
benchmarks): even the 8B produces invalid CadQuery more than half of the time
([[Benchmarks overview]]).

Newer families exist (Qwen3.5 / Qwen3.6, 2026): multimodal models with a hybrid
attention architecture, in sizes such as 2B, 4B, 9B and 27B. They are candidates for later
experiments, after checking that TRL and vLLM support them for training.

## Learn more
- **Qwen3 blog — "Think Deeper, Act Faster"**: <https://qwenlm.github.io/blog/qwen3/>
- **Model card** (thinking switch, recommended sampling): <https://huggingface.co/Qwen/Qwen3-1.7B>
- **Qwen3 Technical Report** (2025): <https://arxiv.org/abs/2505.09388>
