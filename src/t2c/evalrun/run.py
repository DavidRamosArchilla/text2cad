"""Run benchmarks for one model and record the results.

generate -> extract code -> execute + score in the sandbox -> (error-feedback retries) ->
aggregate -> results registry. Works the same for a base model, a checkpoint of ours or an
API model; only the model spec differs.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter

from .. import prompts
from ..bench import registry
from ..bench.base import Benchmark
from ..cad.codeutil import extract_code, strip_thinking
from ..cad.sandbox import CadExecutor
from ..infer.backends import Backend, ModelSpec, load_model_spec, make_backend
from ..results import store

MAX_STORED_RESPONSE = 20_000


def _assistant_turn(response: str, code: str | None) -> str:
    """What the model 'said' when we show it its failed attempt (thinking removed)."""
    if code:
        return prompts.format_completion(code)
    return strip_thinking(response)[:4000] or "(no code was produced)"


def evaluate(
    spec: ModelSpec | None,  # unused by the loop; callers without a spec (training callback) pass None
    backend: Backend,
    bench: Benchmark,
    executor: CadExecutor,
    style: str,
    retries: int,
    limit: int | None = None,
    spread: bool = False,
) -> tuple[dict, list[dict], dict]:
    """``limit`` keeps the first N samples (smoke tests); with ``spread`` it keeps N samples
    evenly spaced over the benchmark instead, so every group stays represented (used for
    the fixed subset scored during training)."""
    samples = bench.load()
    if limit and spread and len(samples) > limit:
        samples = [samples[int(i * len(samples) / limit)] for i in range(limit)]
    elif limit:
        samples = samples[:limit]
    system = bench.system_prompt(style)
    base = [prompts.build_messages(s.task, system) for s in samples]
    attempts: list[list[dict]] = [[] for _ in samples]
    timing = {"generate_s": 0.0, "verify_s": 0.0}

    pending = list(range(len(samples)))
    convs = {i: base[i] for i in pending}
    for round_no in range(retries + 1):
        if not pending:
            break
        t0 = time.time()
        responses = backend.generate([convs[i] for i in pending])
        timing["generate_s"] += time.time() - t0
        codes = [extract_code(r) for r in responses]
        t0 = time.time()
        results = executor.run(
            [bench.make_job(samples[i], c, style) for i, c in zip(pending, codes, strict=True)],
            progress=f"{bench.name} verify (attempt {round_no + 1})",
        )
        timing["verify_s"] += time.time() - t0
        nxt = []
        for i, resp, code, res in zip(pending, responses, codes, results, strict=True):
            attempts[i].append({"response": resp, "code": code, "result": res})
            if not res.get("ok") and res.get("error_type") != "harness_error":
                nxt.append(i)
                convs[i] = prompts.retry_messages(base[i], _assistant_turn(resp, code), res.get("error") or "")
        pending = nxt

    rows = []
    for s, att in zip(samples, attempts, strict=True):
        final = att[-1]
        res = final["result"]
        rows.append(
            {
                "id": s.id,
                "group": s.group,
                "valid": bool(res.get("ok")),
                "first_valid": bool(att[0]["result"].get("ok")),
                "n_attempts": len(att),
                "harness_error": res.get("error_type") == "harness_error",
                "error_type": res.get("error_type"),
                "error": res.get("error"),
                "first_error_type": att[0]["result"].get("error_type"),
                "iou": res.get("iou"),
                "cd": res.get("cd"),
                "dim_err": res.get("dim_err"),
                "volume": res.get("volume"),
                "bbox": res.get("bbox"),
                "code": final["code"],
                "response": (final["response"] or "")[:MAX_STORED_RESPONSE],
                **bench.row_extra(s, res),
            }
        )
    metrics = bench.aggregate(rows)
    errors = {
        "first_attempt": dict(Counter(r["first_error_type"] or "ok" for r in rows)),
        "final": dict(Counter(r["error_type"] or "ok" for r in rows)),
    }
    protocol = {
        "prompt_style": style,
        "retries": retries,
        "system_prompt_sha1": hashlib.sha1(system.encode()).hexdigest()[:10],
        "res": bench.res,
        "n_points": bench.n_points,
        "timeout": bench.timeout,
    }
    return {"metrics": metrics, "errors": errors, "protocol": protocol}, rows, timing


def _fmt(x, pct=False) -> str:
    if x is None:
        return "   -  "
    return f"{x * 100:5.1f}%" if pct else f"{x:6.3f}"


def main(args) -> list[dict]:
    spec = load_model_spec(args.model, overrides=args.set, name=args.name)
    if args.backend:
        spec.backend = args.backend
    style = spec.prompt_style if args.prompt_style == "auto" else args.prompt_style
    benches = registry.resolve(args.bench)
    if not benches:
        raise SystemExit("no prepared benchmarks -- run `t2c bench prepare` first")

    # Start the CAD workers before the GPU stack: they fork from a small, clean process.
    executor = CadExecutor(workers=args.workers).start()
    print(f"[eval] model={spec.name} ({spec.model}) backend={spec.backend} style={style} benchmarks={[b.name for b in benches]}")
    backend = make_backend(spec)
    summaries = []
    try:
        for bench in benches:
            retries = bench.default_retries if args.retries is None else args.retries
            t0 = time.time()
            out, rows, timing = evaluate(spec, backend, bench, executor, style, retries, args.limit)
            summary = {
                "run_id": store.new_run_id(spec.name, bench.name),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "benchmark": bench.name,
                "bench_sha1": bench.info().get("sha1"),
                "model": spec.to_dict(),
                **out,
                "timing": {**{k: round(v, 1) for k, v in timing.items()}, "total_s": round(time.time() - t0, 1)},
                "tags": list(args.tag or []),
                "note": args.note or "",
                "partial": bool(args.limit),
            }
            path = store.save_run(summary, rows)
            summaries.append(summary)
            m = summary["metrics"]["overall"]
            extra = f"  pass={_fmt(m.get('pass_rate'), True)}" if "pass_rate" in m else ""
            print(
                f"[eval] {bench.name:<10} n={m['n']:<4} IR={_fmt(m.get('ir'), True)}  IR@1={_fmt(m.get('ir_first'), True)}  "
                f"IoU={_fmt(m.get('iou'))}  IoU_all={_fmt(m.get('iou_all'))}  CD={_fmt(m.get('cd'))}{extra}  "
                f"[gen {timing['generate_s']:.0f}s, verify {timing['verify_s']:.0f}s] -> {path}"
            )
            if m.get("harness_errors"):
                print(f"[eval] WARNING: {m['harness_errors']} samples hit a scoring-harness error and were excluded; inspect {path}/samples.jsonl.gz")
    finally:
        backend.close()
        executor.close()
    if not args.limit:
        from ..results import report

        report.build()
    return summaries
