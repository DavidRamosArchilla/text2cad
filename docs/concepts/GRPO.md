---
tags: [concept]
aliases: [Group Relative Policy Optimization]
---
# GRPO — Group Relative Policy Optimization

An RL algorithm for LLMs introduced in DeepSeekMath (2024) and made famous by
DeepSeek-R1 (2025). For each prompt:

1. sample a **group** of G answers (here G = 8) from the current model;
2. score each with the reward function;
3. **advantage** of answer *i* = (rᵢ − mean of the group) / std of the group;
4. update the model to raise the probability of tokens in answers with positive advantage
   and lower it for negative ones, with PPO-style clipping so one step cannot change the
   model too much; optionally a KL penalty keeps it close to a reference model.

The key trick vs PPO: the **group mean is the baseline**, so no value network has to be
trained — simpler and cheaper, and a natural fit for verifiable rewards.

## Consequences you will see in practice
- **No variance, no learning.** If all 8 answers get the same reward, every advantage is
  0. That is why our reward is a *ladder* with partial credit ([[cad.reward]]), why GRPO
  starts from an SFT model, and why TRL logs `frac_reward_zero_std`.
- **Sampling temperature matters**: groups need diverse answers (temperature 1.0 here).
- **Length effects**: the original loss normalisation biases answer length; later work
  (Dr. GRPO, DAPO) fixed it. TRL's default `loss_type` is now `dapo`.
- **It optimises the reward, nothing else**: improvements on the RL prompt pool need not
  transfer to benchmarks (the pilot run improved DeepCAD, not CadQueryEval).

## Here
[[train.grpo]] (TRL `GRPOTrainer`) · [[Walkthrough - one GRPO step]] · config
`configs/grpo/*.yaml` (`num_generations: 8`, `beta: 0.0` = no KL penalty, `lr 2e-6`).

## Learn more
- **TRL — GRPO Trainer documentation** (formulas, loss variants, every option we set):
  <https://huggingface.co/docs/trl/grpo_trainer>
- **Cameron R. Wolfe — Group Relative Policy Optimization (GRPO)** (clear long-form explanation):
  <https://cameronrwolfe.substack.com/p/grpo>
- **Hugging Face LLM Course — chapter 12, "Open R1 for Students"** (GRPO hands-on):
  <https://huggingface.co/learn/llm-course/chapter12/1> · the Open-R1 project:
  <https://github.com/huggingface/open-r1>
- Papers: DeepSeekMath (introduces GRPO, section 4) <https://arxiv.org/abs/2402.03300> ·
  DeepSeek-R1 <https://arxiv.org/abs/2501.12948> · DAPO <https://arxiv.org/abs/2503.14476> ·
  *Understanding R1-Zero-Like Training* (Dr. GRPO) <https://arxiv.org/abs/2503.20783>
- Closest prior work on CAD: CAD-Coder (GRPO with a Chamfer reward) — [[CAD-Coder dataset]].
