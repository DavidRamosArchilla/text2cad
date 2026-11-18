---
tags: [concept]
aliases: [chat template, tokenizer, enable_thinking]
---
# Tokenizers and chat templates

A **tokenizer** turns text into token ids and back. Chat models are trained on
conversations written in a very specific format with special tokens. For Qwen3:

````text
<|im_start|>system
You are an expert CadQuery programmer. …<|im_end|>
<|im_start|>user
The design consists of three main parts…<|im_end|>
<|im_start|>assistant
<think>

</think>

```python
import cadquery as cq
…
```
````

The **chat template** is a small Jinja program stored with the tokenizer that turns a list
of `{"role": …, "content": …}` messages into that text. `add_generation_prompt=True` adds
the opening of the assistant turn so the model continues from there. Templates take
options: Qwen3's `enable_thinking=False` inserts the empty `<think></think>` block you see
above, which tells the model to answer directly ([[Qwen3]]).

## Why it matters here
A fine-tuned model has learned one exact format. If evaluation renders the prompt
differently (another system prompt, thinking on instead of off, a missing generation
prompt) the model sees something it never saw in training and quality drops — silently.
So:
- **one function** builds model input everywhere: `render_prompt` in [[infer.backends]],
  used by [[train.sft]] and by every local backend;
- the template options are stored in `t2c_meta.json` with the checkpoint and reused by
  `t2c eval` ([[train.common]]);
- the SFT job log prints one training example exactly as the model sees it — look at it.

Training only computes the loss on the assistant part (labels = −100 on the prompt), see
[[Supervised fine-tuning]].

## Learn more
- **Hugging Face — Chat templates** (how they work, `add_generation_prompt`, writing one):
  <https://huggingface.co/docs/transformers/chat_templating>
- **Hugging Face LLM Course, chapter 11** (SFT; covers chat templates in practice):
  <https://huggingface.co/learn/llm-course/chapter11/1>
- Qwen3 model card (thinking-mode switch and its template): <https://huggingface.co/Qwen/Qwen3-1.7B>
