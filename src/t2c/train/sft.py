"""Stage 1: supervised fine-tuning (full weights) on description -> CadQuery pairs.

    t2c train sft configs/sft/<config>.yaml [key=value ...]
    accelerate launch --num_processes 2 -m t2c.cli train sft configs/sft/<config>.yaml   # multi-GPU

The prompt is rendered here with the tokenizer's chat template -- exactly the function the
evaluation backends use -- and the loss is computed on the completion tokens only.
"""

from __future__ import annotations

import json

from .. import prompts
from ..data.build import read_built
from ..infer.backends import render_prompt
from . import common


def _encode(rows: list[dict], tokenizer, chat_kwargs: dict, max_length: int) -> tuple[list[dict], int]:
    out, dropped = [], 0
    eos = tokenizer.eos_token
    for r in rows:
        prompt = render_prompt(tokenizer, prompts.build_messages(r["text"]), chat_kwargs)
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(prompts.format_completion(r["code"]) + eos, add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(c_ids) > max_length:
            dropped += 1  # never train on a truncated script
            continue
        out.append({"input_ids": p_ids + c_ids, "labels": [-100] * len(p_ids) + c_ids})
    return out, dropped


def main(config_path: str, overrides: list[str] | None = None) -> None:
    cfg = common.load_config(config_path, overrides)
    # CAD workers for in-training benchmark scoring: started before the GPU stack loads,
    # and only on the main process of a multi-GPU run.
    executor = None
    if common.bench_eval_enabled(cfg) and common.is_main_process():
        from ..cad.sandbox import CadExecutor

        executor = CadExecutor(workers=cfg.get("cad_workers")).start()

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from . import callbacks

    out = common.run_dir(cfg)
    init, base_model, parent_meta = common.resolve_init(cfg)
    chat_kwargs = cfg.get("chat_template_kwargs") or {}
    max_length = int(cfg["data"].get("max_length", 2048))

    tokenizer = AutoTokenizer.from_pretrained(init)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows, _ = read_built(cfg["data"]["dataset"], "train")
    val_rows, _ = read_built(cfg["data"]["dataset"], "val")
    if cfg["data"].get("max_train_samples"):
        train_rows = train_rows[: int(cfg["data"]["max_train_samples"])]
    train, dropped = _encode(train_rows, tokenizer, chat_kwargs, max_length)
    val, _ = _encode(val_rows[: int(cfg["data"].get("max_val_samples", 500))], tokenizer, chat_kwargs, max_length)
    lengths = sorted(len(x["input_ids"]) for x in train)

    if common.is_main_process():
        common.write_meta(
            out, cfg, "sft", base_model, parent=init if parent_meta else None, chat_template_kwargs=chat_kwargs,
            extra={"data": {"train": len(train), "val": len(val), "dropped_too_long": dropped, "max_length": max_length}},
        )  # fmt: skip
        print(f"[sft] run={cfg['run_name']} init={init}")
        print(f"[sft] train={len(train)} (dropped {dropped} longer than {max_length} tokens) val={len(val)} "
              f"tokens/sample: median={lengths[len(lengths) // 2]} p99={lengths[int(len(lengths) * 0.99)]} max={lengths[-1]}")  # fmt: skip
        print("[sft] example prompt as the model sees it:\n" + tokenizer.decode(train[0]["input_ids"])[:1500])

    # Full fine-tuning keeps fp32 master weights (bf16 weights cannot absorb 1e-5-sized updates);
    # the forward/backward pass still runs in bf16 through autocast (`bf16: true`).
    model = AutoModelForCausalLM.from_pretrained(init, dtype=getattr(torch, cfg.get("weights_dtype", "float32")))
    args = SFTConfig(
        output_dir=str(out),
        run_name=cfg["run_name"],
        max_length=max_length,
        completion_only_loss=False,  # labels are already masked to the completion above
        report_to=cfg.get("report_to", ["tensorboard"]),
        save_only_model=False,
        seed=int(cfg.get("seed", 0)),
        **cfg["train"],
    )
    bench_eval = callbacks.from_config(cfg, out, executor, tokenizer, chat_kwargs)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(train),
        eval_dataset=Dataset.from_list(val) if val else None,
        processing_class=tokenizer,
        callbacks=[bench_eval] if bench_eval else None,
    )
    if bench_eval:
        bench_eval.trainer = trainer
    try:
        trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    finally:
        if executor is not None:
            executor.close()
    if common.is_main_process():
        trainer.save_model(str(out / "final"))
        tokenizer.save_pretrained(str(out / "final"))
        (out / "final" / "trainer_state.json").write_text(json.dumps({"global_step": trainer.state.global_step, "log_history": trainer.state.log_history}, indent=1))
        print(f"[sft] done: {out / 'final'}  (evaluate with: t2c eval --model {out / 'final'})")
