---
tags: [concept]
aliases: [SFT, fine-tuning]
---
# Supervised fine-tuning (SFT)

Continue training a pre-trained model on **(prompt, desired answer)** pairs with the same
next-token objective as pre-training, but computing the loss **only on the answer tokens**.
The model learns the output format (one fenced CadQuery block, final solid in `result`)
and the domain (which CadQuery calls build which shapes).

What SFT can and cannot do:
- It **imitates** the data — including its mistakes and its style. Training on
  DeepCAD-derived scripts teaches DeepCAD-style parts ([[Data overview]]).
- It cannot become better than its examples; it only sees *correct* answers, never its
  own mistakes. That is what [[Reinforcement learning for LLMs|RL]] adds.
- The **validation loss** measures fit to the data's distribution, not part quality —
  hence the benchmark scores during training ([[train.callbacks]]).

## Where it shows up here
[[train.sft]] (TRL `SFTTrainer`), data `sft/v1` ([[Dataset recipes]]), config
`configs/sft/*.yaml`: learning rate 2e-5 with cosine decay, 2 epochs, batch 32, max 2 048
tokens per example. Full fine-tuning, not LoRA ([[Full fine-tuning vs LoRA]]).

Signs to watch: training loss far below validation loss (overfitting); benchmark subset
curves flattening while loss keeps falling.

## Learn more
- **Hugging Face LLM Course — chapter 11, Supervised Fine-Tuning** (hands-on with TRL):
  <https://huggingface.co/learn/llm-course/chapter11/1>
- **TRL — SFTTrainer documentation** (the exact class we use; dataset formats,
  completion-only loss): <https://huggingface.co/docs/trl/sft_trainer>
- Karpathy's "Deep Dive into LLMs" (the SFT part): <https://www.youtube.com/watch?v=7xTGNNLPyMI>
- Paper: *Training language models to follow instructions with human feedback*
  (InstructGPT, 2022) — the SFT → RL recipe: <https://arxiv.org/abs/2203.02155>
