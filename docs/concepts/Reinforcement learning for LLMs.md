---
tags: [concept]
aliases: [RL, RLHF, RLVR, policy gradient, PPO]
---
# Reinforcement learning for LLMs

In RL the model (the **policy**) produces an answer (a sequence of **actions** = tokens),
receives a **reward**, and is updated to make high-reward answers more likely. Unlike
[[Supervised fine-tuning|SFT]], it learns from **its own attempts**, good and bad, so it can
fix its own typical mistakes and go beyond the examples it was shown.

The core idea is the **policy gradient**: increase the log-probability of the tokens of an
answer in proportion to how much better than expected its reward was (its *advantage*).

## Where the reward comes from
| Flavour | Reward | Example |
|---|---|---|
| **RLHF** | a learned *reward model* trained on human preferences | ChatGPT-style assistants (InstructGPT, PPO) |
| **RLVR** (verifiable rewards) | a **program that checks** the answer | maths answers, unit tests — and here, the CAD verifier |

This project is RLVR: [[cad.reward]] runs the script and compares the solid with ground
truth. No human labels, no reward model that can be fooled in subtle ways — but the
policy will still optimise exactly what the reward measures (*reward hacking*): e.g. the
prose penalty in [[train.grpo]] exists because RL tends to make answers chatty or long.

Algorithms: **PPO** (needs a separate value network to estimate expected reward) and
**[[GRPO]]** (estimates it from a group of samples instead — what we use).

## Learn more
- **OpenAI Spinning Up — Introduction to RL** (concepts, policy gradients, gentle):
  <https://spinningup.openai.com/en/latest/spinningup/rl_intro.html>
- **Hugging Face — Illustrating RLHF** (blog): <https://huggingface.co/blog/rlhf>
- **Nathan Lambert — *Reinforcement Learning from Human Feedback*** (free online book,
  covers RLHF, RLVR, PPO, GRPO): <https://rlhfbook.com/>
- **Lilian Weng — Policy Gradient Algorithms** (the maths, thorough):
  <https://lilianweng.github.io/posts/2018-04-08-policy-gradient/>
- Papers: PPO (2017) <https://arxiv.org/abs/1707.06347> · InstructGPT (2022)
  <https://arxiv.org/abs/2203.02155> · Tülu 3 (2024, coins "RL with verifiable rewards")
  <https://arxiv.org/abs/2411.15124>
