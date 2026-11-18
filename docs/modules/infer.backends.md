---
tags: [module, inference]
aliases: [backends.py, ModelSpec, VLLMBackend, HFBackend, OpenAIBackend, hf_generate, render_prompt]
---
# infer.backends
**File:** `src/t2c/infer/backends.py` · **Libraries:** [[Libraries#vLLM|vLLM]], [[Libraries#transformers|transformers]], [[Libraries#torch|torch]], [[Libraries#openai|openai]], [[Libraries#PyYAML|PyYAML]]
**Concepts:** [[vLLM]] · [[Tokenizers and chat templates]] · [[Decoding]]

Two jobs: describe *which* model to run (`ModelSpec`), and turn conversations into answers
(`Backend`).

## `ModelSpec`
| Field | Meaning |
|---|---|
| `name` | how it appears in results |
| `model` | Hugging Face id, local path, or API model name |
| `backend` | `vllm`, `hf` or `openai` |
| `prompt_style` | `protocol` or `native` ([[prompts]]) |
| `chat_template_kwargs` | e.g. `{enable_thinking: false}` for [[Qwen3]] |
| `sampling` | `temperature`, `max_tokens`, `top_p`… (default greedy, 2 048 tokens) |
| `stage`, `base_model`, `train_run`, `step` | lineage, recorded with every result |
| `engine` | extra vLLM arguments (`max_model_len`, `gpu_memory_utilization`, …) |
| `api` | `base_url`, `api_key_env`, `concurrency`, `params` for API models |

`load_model_spec(arg, overrides, name)` builds it from:
- a **YAML file** in `configs/models/` ([[Configs]]);
- a **checkpoint directory** — reads `t2c_meta.json` (here or one level up): `prompt_style
  = native`, the training chat-template settings, lineage, and the step from
  `checkpoint-N` (name becomes `<run>@N`);
- a bare **Hugging Face id**.

`--set key=value` overrides any field.

## The one function that builds model input
`render_prompt(tokenizer, messages, chat_template_kwargs)` = `apply_chat_template(…,
add_generation_prompt=True)`. Training ([[train.sft]]) and every local backend call this
same function, so a fine-tuned model sees byte-identical prompts at evaluation.

## Backends
| Class | When | How |
|---|---|---|
| `VLLMBackend` | default for local models | renders + tokenises prompts itself and passes **token ids** to `vllm.LLM.generate`; defaults `dtype bfloat16`, `max_model_len 8192`, `gpu_memory_utilization 0.85`; each prompt's `max_tokens` is capped so prompt + answer fit (long retry conversations) |
| `HFBackend` | no vLLM / debugging | `AutoModelForCausalLM` + `hf_generate` (much slower) |
| `OpenAIBackend` | API models | `openai` client, `concurrency` parallel requests in threads, retries with exponential back-off on rate limits and server errors; the key is read from the environment variable named in `api_key_env` (never stored in a file) |

`hf_generate(model, tokenizer, conversations, …)` — batched `model.generate`, prompts
sorted by length to reduce padding, left padding, `no_grad`, optional bf16 autocast.
Shared by `HFBackend` and the in-training callback ([[train.callbacks]]), which calls it on
the model *being trained*.

`configure_vllm_env()` — set before importing vLLM: disables the FlashInfer sampler (it
compiles CUDA code on first use and fails on nodes without a CUDA toolkit) and vLLM's usage
statistics. Also used by [[train.grpo]].

**Used by:** [[evalrun.run]], [[train.sft]], [[train.grpo]], [[train.callbacks]], [[train.common]].
**Tested in:** [[test_eval]] (`test_model_spec_from_checkpoint`), [[test_callback]] (in-process generation).
