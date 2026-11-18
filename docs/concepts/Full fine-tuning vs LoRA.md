---
tags: [concept]
aliases: [LoRA, PEFT, full fine-tuning]
---
# Full fine-tuning vs LoRA

**Full fine-tuning** updates every weight. **LoRA** freezes the model and trains small
low-rank matrices added to some layers (often < 1 % of the parameters).

| | Full fine-tuning (used here) | LoRA |
|---|---|---|
| memory | weights + gradients + optimizer state ≈ **16 bytes per parameter** (fp32 weights 4, gradients 4, AdamW 8) + activations | frozen weights in bf16 (2 bytes/param) + small adapters |
| quality on a new domain | best | often somewhat lower; "learns less, forgets less" |
| checkpoints | full model (+ optimizer: ≈ 7 GB for 0.6B, ≈ 100 GB for 8B) | megabytes |

Rule of thumb for full fine-tuning: 1.7B ≈ 27 GB, 8B ≈ 128 GB (≈ 119 GiB) *before*
activations — at the edge of one 141 GB H200, so plan for several GPUs from 8B up
([[Multi-GPU training]]).

## Decision in this project
You chose full fine-tuning. It fits easily up to ~4B on one H200; for the planned
~27–32B models it needs sharding over many GPUs (available on the large cluster).
LoRA would be worth reconsidering for very large models or for running many cheap
ablations in parallel.

## Learn more
- **LoRA paper** (2021): <https://arxiv.org/abs/2106.09685>
- ***LoRA Learns Less and Forgets Less*** (2024) — careful comparison with full fine-tuning on
  code and maths: <https://arxiv.org/abs/2405.09673>
- **Hugging Face PEFT library** (how LoRA is used in practice): <https://huggingface.co/docs/peft/index>
- **Hugging Face — efficient training on one GPU** (memory anatomy of training):
  <https://huggingface.co/docs/transformers/perf_train_gpu_one>
