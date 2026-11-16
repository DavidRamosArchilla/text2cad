"""Stage 2: GRPO with a verifiable reward.

For each prompt the policy samples a group of scripts; every script is executed in the
sandbox and compared with the ground-truth solid (see cad/reward.py). Advantages are the
rewards standardised within the group, so the model is pushed towards whatever made some
of its own samples build a better part than the others.

    t2c train grpo configs/grpo/<config>.yaml [key=value ...]

Start from an SFT checkpoint (``init_from``): GRPO needs a policy that already produces
valid geometry reasonably often, otherwise every sample in a group gets the same reward
and there is nothing to learn from.
"""

from __future__ import annotations

import json
import re
import statistics
import time

from .. import prompts
from ..cad.codeutil import extract_code, strip_thinking
from ..cad.reward import RewardConfig, reward_from_result
from ..cad.sandbox import CadExecutor
from ..data.build import read_built
from . import common

_FENCED = re.compile(r"```.*?```", re.DOTALL)


class CadReward:
    """Reward function for ``trl.GRPOTrainer``: one float per completion.

    Extra dataset columns arrive as keyword arguments (here ``gt``: path of the ground-truth
    mesh). A small penalty discourages prose around the code block, which costs tokens and
    is the usual way a policy drifts under RL.
    """

    __name__ = "cad_reward"

    def __init__(self, executor: CadExecutor, cfg: RewardConfig, chatter_penalty: float = 0.05, log_path=None):
        self.executor, self.cfg, self.chatter_penalty, self.log_path = executor, cfg, chatter_penalty, log_path

    def __call__(self, prompts, completions, gt, log_metric=None, trainer_state=None, **kwargs) -> list[float]:  # noqa: A002
        texts = [c[-1]["content"] if isinstance(c, list) else c for c in completions]
        codes = [extract_code(t) for t in texts]
        jobs = [
            {"code": code, "gt": path, "res": self.cfg.res, "n_points": self.cfg.n_points, "timeout": self.cfg.timeout}
            for code, path in zip(codes, gt, strict=True)
        ]
        t0 = time.time()
        results = self.executor.run(jobs)
        rewards = []
        for text, res in zip(texts, results, strict=True):
            r = reward_from_result(res, self.cfg)
            if self.chatter_penalty and len(_FENCED.sub("", strip_thinking(text)).strip()) > 20:
                r -= self.chatter_penalty
            rewards.append(r)
        valid = [res for res in results if res.get("ok")]
        stats = {
            "cad/valid_rate": len(valid) / max(len(results), 1),
            "cad/iou_valid": statistics.fmean([v["iou"] for v in valid if v.get("iou") is not None] or [0.0]),
            "cad/iou_all": sum(v.get("iou") or 0.0 for v in valid) / max(len(results), 1),
            "cad/verify_s": time.time() - t0,
        }
        for kind in ("no_code", "syntax_error", "runtime_error", "timeout", "no_result", "no_solid"):
            stats[f"cad/err_{kind}"] = sum(1 for res in results if res.get("error_type") == kind) / max(len(results), 1)
        if log_metric is not None:
            for k, v in stats.items():
                log_metric(k, float(v))
        if self.log_path is not None:
            step = getattr(trainer_state, "global_step", None)
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps({"step": step, "reward_mean": statistics.fmean(rewards), **stats}) + "\n")
        return rewards


def main(config_path: str, overrides: list[str] | None = None) -> None:
    cfg = common.load_config(config_path, overrides)
    out = common.run_dir(cfg)
    init, base_model, parent_meta = common.resolve_init(cfg)
    # A checkpoint of ours dictates how prompts are rendered; a raw base model uses the config.
    chat_kwargs = (parent_meta or {}).get("chat_template_kwargs") if parent_meta else None
    if chat_kwargs is None:
        chat_kwargs = cfg.get("chat_template_kwargs") or {}

    # The CAD workers fork from a small clean process: start them before torch/vLLM load.
    reward_cfg = RewardConfig(**(cfg.get("reward") or {}))
    executor = CadExecutor(workers=cfg.get("cad_workers"), timeout=reward_cfg.timeout).start()

    from ..infer.backends import configure_vllm_env

    configure_vllm_env()
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from . import callbacks

    rows, data_dir = read_built(cfg["data"]["dataset"], "train")
    if cfg["data"].get("max_train_samples"):
        rows = rows[: int(cfg["data"]["max_train_samples"])]
    dataset = Dataset.from_list(
        [{"prompt": prompts.build_messages(r["text"]), "gt": str(data_dir / r["gt"])} for r in rows]
    )

    tokenizer = AutoTokenizer.from_pretrained(init)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(init, dtype=getattr(torch, cfg.get("weights_dtype", "float32")))

    if common.is_main_process():
        common.write_meta(
            out, cfg, "grpo", base_model, parent=init if parent_meta else None, chat_template_kwargs=chat_kwargs,
            extra={"reward": reward_cfg.to_dict(), "data": {"train": len(rows)}},
        )  # fmt: skip
        print(f"[grpo] run={cfg['run_name']} init={init} prompts={len(rows)} cad_workers={executor.workers} sandbox={executor.info}")

    args = GRPOConfig(
        output_dir=str(out),
        run_name=cfg["run_name"],
        chat_template_kwargs=chat_kwargs,
        report_to=cfg.get("report_to", ["tensorboard"]),
        seed=int(cfg.get("seed", 0)),
        **cfg["grpo"],
    )
    reward = CadReward(executor, reward_cfg, chatter_penalty=float(cfg.get("chatter_penalty", 0.05)), log_path=out / "reward_log.jsonl")
    bench_eval = callbacks.from_config(cfg, out, executor, tokenizer, chat_kwargs)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[bench_eval] if bench_eval else None,
    )
    if bench_eval:
        bench_eval.trainer = trainer
    try:
        trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    finally:
        executor.close()
    if common.is_main_process():
        trainer.save_model(str(out / "final"))
        tokenizer.save_pretrained(str(out / "final"))
        (out / "final" / "trainer_state.json").write_text(json.dumps({"global_step": trainer.state.global_step, "log_history": trainer.state.log_history}, indent=1, default=str))
        print(f"[grpo] done: {out / 'final'}  (evaluate with: t2c eval --model {out / 'final'})")
