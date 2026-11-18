---
tags: [concept]
aliases: [LLM, transformer, language model]
---
# LLMs and transformers

A large language model is a neural network trained to **predict the next token** (a piece
of a word) given the tokens before it. Text is split into tokens
([[Tokenizers and chat templates]]), each token becomes a vector, a stack of *transformer*
layers mixes information between positions with **attention**, and the last layer gives a
probability for every possible next token. Generating text = pick a token, append it,
repeat ([[Decoding]]).

Training happens in phases:
1. **Pre-training** on trillions of tokens of web text and code → general knowledge (this
   is why a base Qwen3 already knows some Python and a little CadQuery).
2. **Post-training** → behaviour: [[Supervised fine-tuning]] on example answers, then
   [[Reinforcement learning for LLMs|reinforcement learning]]. This repository does
   post-training only, for one narrow task.

Words you will meet: **parameters** (the weights; "1.7B" = 1.7 billion), **context length**
(how many tokens fit in prompt + answer; `max_length: 2048` in SFT, `max_model_len: 8192`
in vLLM), **logits** (scores before the softmax).

## Where it shows up here
[[Qwen3]] models in `configs/models/`; tokens counted in [[train.sft]]; generation in [[infer.backends]].

## Learn more
- **Andrej Karpathy — "Intro to Large Language Models"** (1 h talk, beginner): the big picture.
  <https://www.youtube.com/watch?v=zjkBMFhNj_g>
- **Andrej Karpathy — "Deep Dive into LLMs like ChatGPT"** (3.5 h, beginner–intermediate):
  pre-training, SFT *and* RL in one story — the best single resource for this project.
  <https://www.youtube.com/watch?v=7xTGNNLPyMI>
- **Jay Alammar — The Illustrated Transformer** (visual explanation of attention):
  <https://jalammar.github.io/illustrated-transformer/>
- **3Blue1Brown — Neural networks series** (includes transformers and attention, very visual):
  <https://www.3blue1brown.com/topics/neural-networks>
- **Hugging Face LLM Course, chapter 1** (hands-on, uses the same libraries as this repo):
  <https://huggingface.co/learn/llm-course/chapter1/1>
- Advanced: Karpathy, "Let's build GPT: from scratch, in code" <https://www.youtube.com/watch?v=kCc8FmEb1nY>;
  the original paper, *Attention Is All You Need* (2017) <https://arxiv.org/abs/1706.03762>.
