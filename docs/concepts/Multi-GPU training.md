---
tags: [concept]
aliases: [FSDP, DeepSpeed, ZeRO, accelerate, gradient checkpointing]
---
# Multi-GPU training

Needed for two reasons: **speed** (more data per second) and **memory** (a model too large
for one GPU). Full fine-tuning costs ≈ 16 bytes per parameter before activations
([[Full fine-tuning vs LoRA]]): 8B ≈ 128 GB, 32B ≈ 512 GB.

| Technique | Idea | Solves |
|---|---|---|
| **data parallel (DDP)** | every GPU holds the whole model, processes different examples, gradients are averaged | speed |
| **ZeRO / FSDP** | shard optimizer state, gradients and weights across GPUs; gather a layer's weights only when it runs | memory — the way to fully fine-tune 27–32B models |
| **gradient checkpointing** | do not keep intermediate activations; recompute them in the backward pass | activation memory, at ~30 % extra compute (on in the 8B and GRPO configs) |
| **accelerate** | one launcher for all of the above (`accelerate launch --num_processes N …`) | configuration |

State here: everything was run on **one GPU**. `accelerate launch` works with the SFT
trainer as it is; FSDP / DeepSpeed configuration for the large cluster still has to be
written and tested. For GRPO on large models, generation can move to separate GPUs
(vLLM `server` mode, [[vLLM]]). The in-training benchmark callback runs only on rank 0
([[train.callbacks]]).

## Learn more
- **Hugging Face Accelerate**: <https://huggingface.co/docs/accelerate/index> — FSDP guide
  <https://huggingface.co/docs/accelerate/usage_guides/fsdp> — DeepSpeed guide
  <https://huggingface.co/docs/accelerate/usage_guides/deepspeed>
- **Efficient training on one GPU** (memory anatomy, gradient checkpointing):
  <https://huggingface.co/docs/transformers/perf_train_gpu_one>
- DeepSpeed: <https://www.deepspeed.ai/>
- Papers: ZeRO (2019) <https://arxiv.org/abs/1910.02054> · PyTorch FSDP (2023)
  <https://arxiv.org/abs/2304.11277> · gradient checkpointing, *Training Deep Nets with
  Sublinear Memory Cost* (2016) <https://arxiv.org/abs/1604.06174>
