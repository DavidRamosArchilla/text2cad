"""``t2c`` command line. Heavy imports (torch, vllm, cadquery) happen inside the handlers
so that ``t2c --help`` and the bookkeeping commands stay instant."""

from __future__ import annotations

import argparse
import json
import sys


# ----------------------------------------------------------------------------- doctor
def cmd_doctor(args) -> int:
    import os
    import platform
    import shutil

    from . import paths

    print(f"python      {platform.python_version()}  ({sys.executable})")
    print(f"host        {platform.node()}  glibc {platform.libc_ver()[1]}")
    for name, p in (("data", paths.data_dir()), ("outputs", paths.output_dir()), ("results", paths.results_dir())):
        free = shutil.disk_usage(p if p.exists() else p.parent).free / 1e9
        print(f"{name:<11} {p}  ({free:,.0f} GB free)")
    print(f"HF cache    {os.environ.get('HF_HOME', '~/.cache/huggingface')}  offline={os.environ.get('HF_HUB_OFFLINE', '0')}")

    from .cad.sandbox import CadExecutor, default_workers

    with CadExecutor(workers=2) as ex:
        info = ex.info
        ok = ex.run_one({"code": "import cadquery as cq\nresult = cq.Workplane('XY').box(1, 2, 3)\n"})
    print(f"cadquery    {info.get('cadquery')}  test solid volume={ok.get('volume')}")
    net = "isolated (no network inside the sandbox)" if info.get("net_isolated") else f"NOT isolated: {info.get('net_error')}"
    print(f"sandbox     workers={default_workers()}  network: {net}")
    try:
        import torch

        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        gpus = ", ".join(f"{torch.cuda.get_device_name(i)} {torch.cuda.get_device_properties(i).total_memory / 2**30:.0f}GB" for i in range(n))
        print(f"torch       {torch.__version__}  cuda={torch.version.cuda}  gpus={n} [{gpus}]")
    except ImportError:
        print("torch       not installed (CPU-only environment: data prep and scoring still work)")
    for mod in ("transformers", "trl", "vllm"):
        try:
            from importlib.metadata import version

            print(f"{mod:<11} {version(mod)}")
        except Exception:  # noqa: BLE001
            print(f"{mod:<11} not installed")
    return 0 if ok.get("ok") else 1


# ----------------------------------------------------------------------------- data
def cmd_data_download(args) -> int:
    from .data import sources

    names = args.names or list(sources.SOURCES)
    for name in names:
        path = sources.download(name, force=args.force)
        print(f"{name}: {path}")
    print(sources.describe())
    return 0


def cmd_data_prepare(args) -> int:
    from .data import prepare

    for source in args.sources or list(prepare.ITERATORS):
        stats = prepare.prepare(source, workers=args.workers, limit=args.limit)
        print(json.dumps(stats, indent=1))
    return 0


def cmd_data_build(args) -> int:
    from .data import build

    info = build.build(args.config, workers=args.workers)
    print(json.dumps(info, indent=1))
    return 0


def cmd_data_synth(args) -> int:
    from .data import synth

    info = synth.generate_dataset(n_train=args.train, n_val=args.val, n_test=args.test, seed=args.seed, workers=args.workers)
    print(json.dumps(info, indent=1))
    return 0


# ----------------------------------------------------------------------------- benchmarks
def cmd_bench_prepare(args) -> int:
    from .bench import registry

    for name in args.names or registry.names():
        info = registry.get(name).prepare(workers=args.workers)
        print(json.dumps({name: info}, indent=1))
    return 0


def cmd_bench_list(args) -> int:
    from .bench import registry

    for name in registry.names():
        b = registry.get(name)
        state = f"{len(b.load())} samples" if b.is_prepared() else "not prepared"
        print(f"{name:<10} {state:<14} {b.description}")
    return 0


def cmd_bench_validate_cqe(args) -> int:
    from .bench import cqe

    print(json.dumps(cqe.validate_against_published(limit=args.limit, workers=args.workers), indent=1))
    return 0


# ----------------------------------------------------------------------------- eval / report
def cmd_eval(args) -> int:
    from .evalrun import run

    run.main(args)
    return 0


def cmd_report(args) -> int:
    from .results import report

    out = report.build(open_html=False)
    print("\n".join(str(p) for p in out))
    return 0


def cmd_runs(args) -> int:
    from .results import store

    for row in store.list_runs():
        m = row["metrics"]["overall"]
        print(
            f"{row['run_id']:<58} {row['benchmark']:<10} {row['model']['name']:<28} "
            f"IR={m.get('ir', float('nan')) * 100:5.1f}%  IoU={m.get('iou', float('nan')):.3f}  n={m.get('n')}"
        )
    return 0


# ----------------------------------------------------------------------------- training
def cmd_train(args) -> int:
    if args.stage == "sft":
        from .train import sft

        sft.main(args.config, args.override)
    else:
        from .train import grpo

        grpo.main(args.config, args.override)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="t2c", description="Text -> CadQuery fine-tuning toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check the environment on this machine").set_defaults(fn=cmd_doctor)

    data = sub.add_parser("data", help="download and prepare datasets").add_subparsers(dest="sub", required=True)
    d = data.add_parser("download", help="fetch pinned raw sources")
    d.add_argument("names", nargs="*")
    d.add_argument("--force", action="store_true")
    d.set_defaults(fn=cmd_data_download)
    d = data.add_parser("prepare", help="normalise + execution-verify a training corpus")
    d.add_argument("sources", nargs="*")
    d.add_argument("--workers", type=int)
    d.add_argument("--limit", type=int, help="cap per split (smoke tests)")
    d.set_defaults(fn=cmd_data_prepare)
    d = data.add_parser("build", help="assemble an SFT or RL dataset from a mixture config")
    d.add_argument("config")
    d.add_argument("--workers", type=int)
    d.set_defaults(fn=cmd_data_build)
    d = data.add_parser("synth", help="generate the procedural millimetre-scale dataset")
    d.add_argument("--train", type=int, default=20000)
    d.add_argument("--val", type=int, default=500)
    d.add_argument("--test", type=int, default=400)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--workers", type=int)
    d.set_defaults(fn=cmd_data_synth)

    bench = sub.add_parser("bench", help="benchmarks").add_subparsers(dest="sub", required=True)
    b = bench.add_parser("prepare", help="build samples + ground-truth meshes")
    b.add_argument("names", nargs="*")
    b.add_argument("--workers", type=int)
    b.set_defaults(fn=cmd_bench_prepare)
    bench.add_parser("list").set_defaults(fn=cmd_bench_list)
    b = bench.add_parser("validate-cqe", help="re-score published CadQueryEval attempts and compare verdicts")
    b.add_argument("--limit", type=int, default=400)
    b.add_argument("--workers", type=int)
    b.set_defaults(fn=cmd_bench_validate_cqe)

    e = sub.add_parser("eval", help="run benchmarks for a model and record the results")
    e.add_argument("--model", required=True, help="configs/models/*.yaml, a checkpoint directory, or a HF model id")
    e.add_argument("--bench", nargs="+", default=["all"])
    e.add_argument("--prompt-style", choices=["auto", "native", "protocol"], default="auto")
    e.add_argument("--retries", type=int, default=None, help="error-feedback retries (default: the benchmark's protocol)")
    e.add_argument("--limit", type=int, help="first N samples of each benchmark (smoke tests; not recorded as a result)")
    e.add_argument("--name", help="display name for this model in the results")
    e.add_argument("--tag", action="append", default=[])
    e.add_argument("--note", default="")
    e.add_argument("--backend", choices=["vllm", "hf", "openai"])
    e.add_argument("--workers", type=int, help="CAD verifier processes")
    e.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a model-config field")
    e.set_defaults(fn=cmd_eval)

    sub.add_parser("report", help="rebuild results/REPORT.md and results/report.html").set_defaults(fn=cmd_report)
    sub.add_parser("runs", help="list recorded evaluation runs").set_defaults(fn=cmd_runs)

    t = sub.add_parser("train", help="fine-tune")
    t.add_argument("stage", choices=["sft", "grpo"])
    t.add_argument("config")
    t.add_argument("override", nargs="*", metavar="KEY=VALUE", help="override config fields, e.g. train.learning_rate=1e-5")
    t.set_defaults(fn=cmd_train)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
