"""Model specifications and inference backends.

A *model spec* says what to run and how to talk to it. It comes from

* a YAML file in ``configs/models/`` (base and API models),
* a checkpoint directory written by our trainers (reads ``t2c_meta.json``), or
* a bare Hugging Face model id.

Backends take chat conversations and return one completion string each:

    vllm     in-process vLLM, the default for local models (fast batched generation)
    hf       plain transformers ``generate`` (no vLLM needed; slow, for debugging)
    openai   any OpenAI-compatible HTTP API (frontier models, OpenRouter, a vLLM server)

Local backends render the prompt with the tokenizer's chat template themselves and hand
*token ids* to the engine, so the context a fine-tuned model sees at inference is
byte-identical to what it was trained on (train/sft.py renders it the same way).
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

META_FILE = "t2c_meta.json"


@dataclass
class ModelSpec:
    name: str
    model: str  # HF id, local path, or API model name
    backend: str = "vllm"
    prompt_style: str = "protocol"  # "protocol" for base/API models, "native" for our fine-tunes
    chat_template_kwargs: dict = field(default_factory=dict)
    sampling: dict = field(default_factory=lambda: {"temperature": 0.0, "max_tokens": 2048})
    # lineage, recorded with every result
    stage: str = "base"  # base | sft | grpo | api
    base_model: str | None = None
    train_run: str | None = None
    step: int | None = None
    # backend options
    engine: dict = field(default_factory=dict)  # vllm: dtype, max_model_len, gpu_memory_utilization, ...
    api: dict = field(default_factory=dict)  # openai: base_url, api_key_env, concurrency, params{...}

    def to_dict(self) -> dict:
        return asdict(self)


def _apply_overrides(d: dict, overrides: list[str]) -> dict:
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = d
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return d


def find_meta(path: Path) -> dict | None:
    """t2c_meta.json sits in the run directory; checkpoints live one level below it."""
    for cand in (path / META_FILE, path.parent / META_FILE):
        if cand.exists():
            return json.loads(cand.read_text())
    return None


def load_model_spec(arg: str, overrides: list[str] | None = None, name: str | None = None) -> ModelSpec:
    p = Path(arg)
    if p.suffix in (".yaml", ".yml"):
        d = yaml.safe_load(p.read_text())
        d.setdefault("name", p.stem)
    elif p.is_dir():
        meta = find_meta(p)
        if meta is None:
            raise FileNotFoundError(f"{p} has no {META_FILE}; pass a configs/models/*.yaml for models not trained with t2c")
        step = None
        m = re.search(r"checkpoint-(\d+)$", p.name)
        if m:
            step = int(m.group(1))
        elif (p / "trainer_state.json").exists():
            step = json.loads((p / "trainer_state.json").read_text()).get("global_step")
        d = {
            "name": f"{meta['train_run']}" + (f"@{step}" if m else ""),
            "model": str(p.resolve()),
            "backend": "vllm",
            "prompt_style": "native",
            "chat_template_kwargs": meta.get("chat_template_kwargs", {}),
            "stage": meta["stage"],
            "base_model": meta.get("base_model"),
            "train_run": meta["train_run"],
            "step": step,
        }
    else:
        d = {"name": arg.rstrip("/").split("/")[-1], "model": arg}
    d = _apply_overrides(d, overrides or [])
    if name:
        d["name"] = name
    spec = ModelSpec(**d)
    spec.sampling = {"temperature": 0.0, "max_tokens": 2048, **(spec.sampling or {})}
    return spec


# ---------------------------------------------------------------------------------------
class Backend:
    def generate(self, conversations: list[list[dict]]) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:
        pass


def render_prompt(tokenizer, messages: list[dict], chat_template_kwargs: dict | None = None) -> str:
    """The one place where a conversation becomes model input text (training uses it too)."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **(chat_template_kwargs or {}))


def configure_vllm_env() -> None:
    """Environment defaults that make vLLM start on a bare compute node.

    FlashInfer's sampling kernels are JIT-compiled with nvcc on first use; clusters without
    a CUDA toolkit on the node fail right there. The PyTorch sampler is used instead (the
    difference is irrelevant next to generation time). Export the variable yourself to
    override. Must run before ``import vllm``.
    """
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")  # no telemetry from cluster jobs
    os.environ.setdefault("DO_NOT_TRACK", "1")


class VLLMBackend(Backend):
    def __init__(self, spec: ModelSpec):
        configure_vllm_env()
        from transformers import AutoTokenizer
        from vllm import LLM

        self.spec = spec
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model)
        engine = {"dtype": "bfloat16", "max_model_len": 8192, "gpu_memory_utilization": 0.85, "seed": 0, **spec.engine}
        self.max_model_len = engine["max_model_len"]
        self.llm = LLM(model=spec.model, **engine)

    def generate(self, conversations: list[list[dict]]) -> list[str]:
        from vllm import SamplingParams

        s = dict(self.spec.sampling)
        max_tokens = s.pop("max_tokens", 2048)
        prompts = []
        for conv in conversations:
            ids = self.tokenizer(render_prompt(self.tokenizer, conv, self.spec.chat_template_kwargs), add_special_tokens=False)["input_ids"]
            ids = ids[-(self.max_model_len - 64) :]  # never overflow the context; keep the most recent turns
            prompts.append(ids)
        # A retry conversation can be long: cap each completion so prompt + completion fits.
        params = [SamplingParams(max_tokens=max(16, min(max_tokens, self.max_model_len - len(ids))), **s) for ids in prompts]
        outs = self.llm.generate(prompts, params, use_tqdm=len(prompts) > 64)
        return [o.outputs[0].text for o in outs]

    def close(self) -> None:
        import gc

        del self.llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


class HFBackend(Backend):
    def __init__(self, spec: ModelSpec, batch_size: int = 16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec, self.batch_size, self.torch = spec, batch_size, torch
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(spec.model, dtype=dtype, device_map="auto" if torch.cuda.is_available() else None)
        self.model.eval()

    def generate(self, conversations: list[list[dict]]) -> list[str]:
        torch = self.torch
        texts = [render_prompt(self.tokenizer, c, self.spec.chat_template_kwargs) for c in conversations]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [""] * len(texts)
        s = self.spec.sampling
        gen = {"max_new_tokens": s.get("max_tokens", 2048), "do_sample": s.get("temperature", 0.0) > 0}
        if gen["do_sample"]:
            gen.update(temperature=s["temperature"], top_p=s.get("top_p", 1.0))
        for i in range(0, len(order), self.batch_size):
            idx = order[i : i + self.batch_size]
            enc = self.tokenizer([texts[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            with torch.inference_mode():
                ids = self.model.generate(**enc, **gen, pad_token_id=self.tokenizer.pad_token_id)
            for j, row in zip(idx, ids, strict=True):
                out[j] = self.tokenizer.decode(row[enc["input_ids"].shape[1] :], skip_special_tokens=True)
        return out

    def close(self) -> None:
        del self.model
        self.torch.cuda.empty_cache()


class OpenAIBackend(Backend):
    """OpenAI-compatible chat API. ``spec.api``:

    base_url      e.g. https://openrouter.ai/api/v1 (omit for api.openai.com)
    api_key_env   name of the environment variable holding the key (default OPENAI_API_KEY)
    concurrency   parallel requests (default 8)
    params        dict passed verbatim to chat.completions.create (temperature, max_tokens or
                  max_completion_tokens, reasoning_effort, extra_body, ...). When absent,
                  ``sampling`` is used.
    """

    def __init__(self, spec: ModelSpec):
        from openai import OpenAI

        self.spec = spec
        api = spec.api or {}
        key_env = api.get("api_key_env", "OPENAI_API_KEY")
        key = os.environ.get(key_env)
        if not key and not api.get("base_url", "").startswith("http://"):
            raise RuntimeError(f"environment variable {key_env} is not set (needed for model '{spec.name}')")
        self.client = OpenAI(base_url=api.get("base_url"), api_key=key or "EMPTY", timeout=api.get("timeout", 600), max_retries=0)
        self.concurrency = int(api.get("concurrency", 8))
        self.params = api.get("params") or dict(spec.sampling)

    def _one(self, conv: list[dict]) -> str:
        delay = 2.0
        for attempt in range(6):
            try:
                resp = self.client.chat.completions.create(model=self.spec.model, messages=conv, **self.params)
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - rate limits / transient network errors
                status = getattr(e, "status_code", None)
                if status is not None and status < 500 and status not in (408, 409, 429):
                    raise
                if attempt == 5:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60)
        return ""

    def generate(self, conversations: list[list[dict]]) -> list[str]:
        with ThreadPoolExecutor(self.concurrency) as pool:
            return list(pool.map(self._one, conversations))


def make_backend(spec: ModelSpec) -> Backend:
    if spec.backend == "vllm":
        return VLLMBackend(spec)
    if spec.backend == "hf":
        return HFBackend(spec)
    if spec.backend == "openai":
        return OpenAIBackend(spec)
    raise ValueError(f"unknown backend '{spec.backend}'")
