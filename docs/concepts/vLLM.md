---
tags: [concept]
aliases: [PagedAttention, KV cache, colocate]
---
# vLLM

An inference engine that generates text from many prompts at once, much faster than a
plain `model.generate` loop.

Two ideas make it fast:
- **KV cache**: when generating token *n+1*, attention needs the keys/values of all
  previous tokens; caching them avoids recomputing. The cache is large and grows with
  every token.
- **PagedAttention**: vLLM stores the KV cache in fixed-size pages, like an operating
  system's virtual memory, so hundreds of sequences of different lengths share GPU memory
  without waste; with **continuous batching**, finished sequences are replaced by new ones
  immediately.

## Two uses here
1. **Evaluation** — `VLLMBackend` ([[infer.backends]]): all prompts of a benchmark in one
   call; it receives token ids we rendered ourselves (so the prompt is exactly the
   training format). 25 CadQueryEval answers take seconds.
2. **GRPO rollouts** — TRL runs vLLM inside the training process on the same GPU
   (`vllm_mode: colocate`, 25 % of GPU memory) and copies the updated weights into it after
   each step. The alternative, `server` mode, uses a separate GPU for generation — useful
   for larger models on the big cluster.

We set `VLLM_USE_FLASHINFER_SAMPLER=0`: that sampler compiles CUDA code on first use and
fails on nodes without a CUDA toolkit.

## Learn more
- **vLLM blog — "vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention"** (2023, the clear explanation):
  <https://blog.vllm.ai/2023/06/20/vllm.html>
- **vLLM documentation**: <https://docs.vllm.ai/en/latest/>
- **TRL — vLLM integration** (colocate vs server mode for GRPO): <https://huggingface.co/docs/trl/vllm_integration>
- Paper: *Efficient Memory Management for LLM Serving with PagedAttention* (2023):
  <https://arxiv.org/abs/2309.06180>
