"""In-training benchmark scoring: scheduling, logging and -- above all -- never breaking a run.

No GPU and no real model: generation is replaced by a scripted backend, the scoring is real.
"""

import json
from types import SimpleNamespace

import pytest

from t2c.bench import registry
from t2c.bench.base import Benchmark, Sample
from t2c.evalrun.run import evaluate
from t2c.infer.backends import Backend
from t2c.train import callbacks, common

BOX = "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 20, 30)\n"
CYL = "import cadquery as cq\nresult = cq.Workplane('XY').circle(5).extrude(40)\n"


class TinyBench(Benchmark):
    name = "tiny"
    res, n_points = 64, 2000


class Scripted(Backend):
    """Builds the box correctly and answers everything else with a script that raises."""

    def __init__(self):
        self.calls = 0

    def generate(self, conversations):
        self.calls += 1
        good, bad = f"```python\n{BOX}```", "```python\nimport cadquery as cq\nresult = cq.Workplane('XY').box(1, 1, 1).fillet(9)\n```"
        return [good if conv[1]["content"].startswith("box") else bad for conv in conversations]


class Trainer:
    def __init__(self):
        self.logged = []

    def log(self, logs):
        self.logged.append(dict(logs))


def state(step, main=True):
    return SimpleNamespace(global_step=step, is_world_process_zero=main)


@pytest.fixture
def tiny(tmp_path, monkeypatch, executor):
    monkeypatch.setenv("T2C_DATA_DIR", str(tmp_path / "data"))
    bench = TinyBench()
    gt = bench.dir / "gt"
    assert all(r["ok"] for r in executor.run([{"code": BOX, "save_mesh": str(gt / "box.npz")}, {"code": CYL, "save_mesh": str(gt / "cyl.npz")}]))
    samples = [Sample(f"s{i}", "box" if i % 2 == 0 else "cyl", "even" if i % 2 == 0 else "odd", "gt/box.npz" if i % 2 == 0 else "gt/cyl.npz") for i in range(6)]
    bench._write(samples, {"source": "test"})
    monkeypatch.setitem(registry._BENCHMARKS, "tiny", bench)
    return bench


def make(executor, tmp_path, backend, **kw):
    cb = callbacks.BenchmarkEvalCallback(lambda model: backend, executor, ["tiny"], every_steps=2, limit=None, log_path=tmp_path / "bench_log.jsonl", **kw)
    cb.trainer = Trainer()
    return cb


def test_schedule_and_logging(tiny, executor, tmp_path):
    backend = Scripted()
    cb = make(executor, tmp_path, backend)
    model = object()
    cb.on_train_begin(None, state(0), None, model=model)
    for step in range(1, 6):
        cb.on_step_end(None, state(step), None, model=model)
    cb.on_train_end(None, state(5), None, model=model)
    cb.on_train_end(None, state(5), None, model=model)  # same step again: no second evaluation

    lines = [json.loads(x) for x in (tmp_path / "bench_log.jsonl").read_text().splitlines()]
    assert [x["step"] for x in lines] == [0, 2, 4, 5]  # start, every 2 steps, final weights
    assert backend.calls == 4
    logs = cb.trainer.logged
    assert len(logs) == 4
    # Half the parts are boxes built exactly, the other half fail: IR 0.5, IoU_all 0.5.
    assert logs[0]["eval_tiny_ir"] == pytest.approx(0.5) and logs[0]["eval_tiny_iou_all"] == pytest.approx(0.5, abs=1e-6)
    assert set(logs[0]) == {"eval_tiny_ir", "eval_tiny_iou_all", "eval_tiny_iou", "eval_bench_seconds"}
    assert lines[0]["benchmark"] == "tiny" and lines[0]["n"] == 6


def test_resume_does_not_rescore_the_start(tiny, executor, tmp_path):
    # The Trainer calls on_train_begin with global_step == 0 even when resuming (it restores
    # the step counter afterwards), so resuming has to switch `at_start` off from the config.
    cfg = {"eval_during_training": {"every_steps": 2, "benchmarks": ["tiny"]}, "resume_from_checkpoint": "outputs/train/x/checkpoint-4000"}
    resumed = callbacks.from_config(cfg, tmp_path, executor, None, {})
    assert resumed.at_start is False
    resumed.trainer = Trainer()
    resumed.on_train_begin(None, state(0), None, model=object())
    assert resumed.trainer.logged == []
    fresh = callbacks.from_config({"eval_during_training": cfg["eval_during_training"]}, tmp_path, executor, None, {})
    assert fresh.at_start is True


def test_only_the_main_process_evaluates(tiny, executor, tmp_path):
    backend = Scripted()
    cb = make(executor, tmp_path, backend)
    cb.on_step_end(None, state(2, main=False), None, model=object())
    assert backend.calls == 0 and cb.trainer.logged == []


def test_a_failure_disables_the_callback_instead_of_the_run(tiny, executor, tmp_path, capsys):
    class Broken(Backend):
        calls = 0

        def generate(self, conversations):
            Broken.calls += 1
            raise RuntimeError("CUDA out of memory")

    cb = make(executor, tmp_path, Broken())
    cb.on_step_end(None, state(2), None, model=object())  # must not raise
    assert cb.disabled and "DISABLED" in capsys.readouterr().out
    cb.on_step_end(None, state(4), None, model=object())
    assert Broken.calls == 1 and cb.trainer.logged == []


def test_unprepared_benchmark_is_skipped(tiny, executor, tmp_path, capsys):
    class Missing(Benchmark):
        name = "missing"

    registry._BENCHMARKS["missing"] = Missing()
    try:
        cb = callbacks.BenchmarkEvalCallback(lambda model: Scripted(), executor, ["missing", "tiny"], every_steps=1, limit=None)
        cb.trainer = Trainer()
        cb.on_step_end(None, state(1), None, model=object())
    finally:
        del registry._BENCHMARKS["missing"]
    assert not cb.disabled and "eval_tiny_ir" in cb.trainer.logged[0]
    assert "skipping 'missing'" in capsys.readouterr().out


def test_spread_subset_covers_all_groups(tiny, executor):
    out, rows, _ = evaluate(None, Scripted(), tiny, executor, style="native", retries=0, limit=4, spread=True)
    assert [r["id"] for r in rows] == ["s0", "s1", "s3", "s4"]  # evenly spaced, both groups present
    assert set(out["metrics"]["groups"]) == {"even", "odd"}
    _, first, _ = evaluate(None, Scripted(), tiny, executor, style="native", retries=0, limit=4)
    assert [r["id"] for r in first] == ["s0", "s1", "s2", "s3"]  # plain --limit keeps the first N


def test_in_process_backend_restores_training_mode(monkeypatch):
    class Model:
        training = True

        def eval(self):
            self.training = False

        def train(self, mode=True):
            self.training = mode

    seen = {}

    def fake_generate(model, tokenizer, conversations, chat_kwargs, **kw):
        seen.update(training=model.training, kw=kw, chat_kwargs=chat_kwargs)
        if conversations == "boom":
            raise RuntimeError("generation failed")
        return ["ok"] * len(conversations)

    monkeypatch.setattr(callbacks, "hf_generate", fake_generate)
    model = Model()
    backend = callbacks.InProcessBackend(model, tokenizer=None, chat_template_kwargs={"enable_thinking": False}, max_new_tokens=64, batch_size=4)
    assert backend.generate([[{}], [{}]]) == ["ok", "ok"]
    assert seen["training"] is False and model.training is True  # generated in eval mode, training mode restored
    assert seen["kw"] == {"max_new_tokens": 64, "batch_size": 4, "autocast_bf16": True}
    assert seen["chat_kwargs"] == {"enable_thinking": False}
    with pytest.raises(RuntimeError):
        backend.generate("boom")
    assert model.training is True  # restored even when generation raises


def test_from_config_switch(tmp_path, executor):
    assert callbacks.from_config({}, tmp_path, executor, None, {}) is None
    assert callbacks.from_config({"eval_during_training": {"every_steps": 0}}, tmp_path, executor, None, {}) is None
    cfg = {"eval_during_training": {"every_steps": 500, "benchmarks": ["cqe"], "limit": 10, "at_start": False}}
    assert common.bench_eval_enabled(cfg)
    assert callbacks.from_config(cfg, tmp_path, None, None, {}) is None  # no CAD workers on this rank
    cb = callbacks.from_config(cfg, tmp_path, executor, None, {})
    assert (cb.every_steps, cb.benchmarks, cb.limit, cb.at_start) == (500, ["cqe"], 10, False)
    assert cb.log_path == tmp_path / "bench_log.jsonl"
