"""Benchmark scores while training.

Every ``every_steps`` optimiser steps the model being trained answers a fixed subset of
each benchmark and the scores are logged through the trainer. They therefore show up in
the same tracker as the loss -- TensorBoard, W&B or trackio, whatever ``report_to`` says --
under ``eval/<benchmark>_<metric>``, and in ``<run>/bench_log.jsonl``.

Config block (same for SFT and GRPO; ``every_steps: 0`` or no block turns it off):

    eval_during_training:
      every_steps: 1000
      benchmarks: [cqe, deepcad]
      limit: 48              # samples per benchmark: a fixed subset, evenly spread
      max_new_tokens: 768
      batch_size: 32
      at_start: true         # also score the untouched model at step 0

These numbers are a monitoring signal, not results: a subset of each benchmark, greedy
decoding with plain ``generate``, one attempt. They are comparable with each other along a
run; for numbers to report use ``t2c eval`` on a checkpoint.

Cost, measured with Qwen3-0.6B on an H200: 2 to 2.5 minutes per evaluation with three
benchmarks (121 prompts), close to 4 minutes for an untrained model that rambles to the
token limit. Generation dominates and a larger batch did not help (64 was no faster than
32), so the knobs that matter are ``benchmarks``, ``limit``, ``max_new_tokens`` and
``every_steps``.

A failure in here must never cost a training run: any error disables the callback for the
rest of the run and training continues.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from pathlib import Path

from transformers import TrainerCallback

from ..bench import registry
from ..cad.sandbox import CadExecutor
from ..evalrun.run import evaluate
from ..infer.backends import Backend, hf_generate
from .common import bench_eval_enabled

LOGGED_METRICS = ("ir", "iou_all", "iou", "pass_rate")


class InProcessBackend(Backend):
    """Generates with the live training model, then puts it back in the mode it was in."""

    def __init__(self, model, tokenizer, chat_template_kwargs: dict | None, max_new_tokens: int = 768, batch_size: int = 32):
        self.model, self.tokenizer, self.chat_template_kwargs = model, tokenizer, chat_template_kwargs or {}
        self.max_new_tokens, self.batch_size = max_new_tokens, batch_size

    def generate(self, conversations: list[list[dict]]) -> list[str]:
        was_training = self.model.training
        self.model.eval()
        try:
            return hf_generate(
                self.model,
                self.tokenizer,
                conversations,
                self.chat_template_kwargs,
                max_new_tokens=self.max_new_tokens,
                batch_size=self.batch_size,
                autocast_bf16=True,
            )
        finally:
            self.model.train(was_training)


class BenchmarkEvalCallback(TrainerCallback):
    def __init__(
        self,
        backend_factory: Callable[[object], Backend],
        executor: CadExecutor,
        benchmarks: list[str],
        every_steps: int,
        limit: int | None = 48,
        at_start: bool = True,
        log_path: str | Path | None = None,
    ):
        self.backend_factory, self.executor = backend_factory, executor
        self.benchmarks, self.every_steps, self.limit, self.at_start = list(benchmarks), int(every_steps), limit, at_start
        self.log_path = Path(log_path) if log_path else None
        self.trainer = None  # set once the trainer exists: its .log() reaches every tracker
        self.disabled = False
        self._last_step: int | None = None
        self._warned: set[str] = set()

    # ------------------------------------------------------------------ trainer events
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.at_start and state.global_step == 0:
            self._run(state, model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.every_steps > 0 and state.global_step % self.every_steps == 0:
            self._run(state, model)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self._run(state, model)  # the final weights, unless that step was just scored

    # ------------------------------------------------------------------ work
    def _run(self, state, model) -> None:
        step = state.global_step
        if self.disabled or model is None or not state.is_world_process_zero or step == self._last_step:
            return
        self._last_step = step
        t0 = time.time()
        try:
            backend = self.backend_factory(model)
            logs: dict[str, float] = {}
            for name in self.benchmarks:
                bench = registry.get(name)
                if not bench.is_prepared():
                    if name not in self._warned:
                        self._warned.add(name)
                        print(f"\n[bench-eval] skipping '{name}': not prepared (run `t2c bench prepare {name}`)", flush=True)
                    continue
                out, _, _ = evaluate(None, backend, bench, self.executor, style="native", retries=0, limit=self.limit, spread=True)
                overall = out["metrics"]["overall"]
                for key in LOGGED_METRICS:
                    if overall.get(key) is not None:
                        logs[f"eval_{name}_{key}"] = float(overall[key])
                if self.log_path is not None:
                    with self.log_path.open("a") as fh:
                        fh.write(json.dumps({"step": step, "benchmark": name, **overall}) + "\n")
            if logs:
                logs["eval_bench_seconds"] = round(time.time() - t0, 1)
                shown = "  ".join(f"{k[5:]}={v:.3f}" for k, v in logs.items() if k != "eval_bench_seconds")
                # Leading newline: the trainer's progress bar leaves the cursor mid-line.
                print(f"\n[bench-eval] step {step}: {shown}  ({logs['eval_bench_seconds']:.0f}s)", flush=True)
                if self.trainer is not None:
                    self.trainer.log(logs)
        except Exception as e:  # noqa: BLE001 - monitoring must not kill a training run
            self.disabled = True
            print(f"\n[bench-eval] DISABLED for the rest of this run after an error at step {step}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass


def from_config(cfg: dict, out_dir: Path, executor: CadExecutor | None, tokenizer, chat_template_kwargs: dict) -> BenchmarkEvalCallback | None:
    """Build the callback from a training config, or None when it is switched off."""
    if not bench_eval_enabled(cfg) or executor is None:
        return None
    ec = cfg["eval_during_training"]
    max_new_tokens, batch_size = int(ec.get("max_new_tokens", 768)), int(ec.get("batch_size", 32))
    return BenchmarkEvalCallback(
        backend_factory=lambda model: InProcessBackend(model, tokenizer, chat_template_kwargs, max_new_tokens, batch_size),
        executor=executor,
        benchmarks=ec.get("benchmarks") or ["cqe", "deepcad"],
        every_steps=int(ec["every_steps"]),
        limit=ec.get("limit", 48),
        # The Trainer fires on_train_begin before it restores a checkpoint's step counter, so a
        # resumed run would log the resumed weights as "step 0": decide it here instead.
        at_start=bool(ec.get("at_start", True)) and not cfg.get("resume_from_checkpoint"),
        log_path=out_dir / "bench_log.jsonl",
    )
