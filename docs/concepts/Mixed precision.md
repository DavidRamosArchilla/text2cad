---
tags: [concept]
aliases: [bf16, fp32, autocast, master weights]
---
# Mixed precision

Numbers on the GPU can be stored in different formats. **fp32** (32 bits) is precise;
**bf16** (16 bits) has the same range but only 2–3 significant decimal digits — and GPUs
compute in bf16 several times faster.

*Mixed precision* = do the heavy maths (matrix multiplications) in bf16, keep what needs
precision in fp32.

## Why the weights stay in fp32 here
A typical update is tiny: learning rate 2·10⁻⁵ × gradient. Added to a weight of size ~1 in
bf16, it is **rounded away** (1 + 0.00002 = 1 in bf16), so a model with bf16 weights
barely learns. So:
- `weights_dtype: float32` in the training configs: the *master* weights are fp32;
- `bf16: true`: forward and backward passes run under bf16 **autocast**;
- the in-training benchmark generation also uses bf16 autocast ([[train.callbacks]]);
- inference (vLLM) loads weights in bf16 — fine, nothing is being updated.

## Learn more
- **PyTorch blog — What every user should know about mixed precision training**:
  <https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/>
- **PyTorch AMP documentation** (`torch.autocast`): <https://pytorch.org/docs/stable/amp.html>
- **bfloat16 format** (Wikipedia): <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format>
- Paper: *Mixed Precision Training* (2017), origin of fp32 master weights: <https://arxiv.org/abs/1710.03740>
