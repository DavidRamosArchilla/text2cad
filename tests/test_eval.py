"""End-to-end test of the evaluation loop with a scripted model (no GPU needed)."""

import json
from argparse import Namespace

import pytest

from t2c import prompts
from t2c.bench.base import Benchmark, Sample
from t2c.evalrun import run as evalrun
from t2c.infer.backends import Backend, ModelSpec
from t2c.results import report, store

BOX = "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 20, 30)\n"
CYL = "import cadquery as cq\nresult = cq.Workplane('XY').circle(5).extrude(40)\n"


class TinyBench(Benchmark):
    name = "tiny"
    description = "three parts"
    res, n_points = 64, 2000


class ScriptedBackend(Backend):
    """Answers the first round with one good, one broken and one empty response; fixes the broken one when asked."""

    def __init__(self):
        self.calls = []

    def generate(self, conversations):
        self.calls.append(conversations)
        out = []
        for conv in conversations:
            task = conv[1]["content"]
            retry = len(conv) > 2
            if task == "box":
                out.append(f"Here you go\n```python\n{BOX}```")
            elif task == "cyl":
                good = f"```python\n{CYL}```"
                out.append(good if retry else "```python\nimport cadquery as cq\nresult = cq.Workplane('XY').circle(5).extrude(40).fillet(99)\n```")
            else:
                out.append("I am not able to help with that.")
        return out


@pytest.fixture
def tiny(tmp_path, monkeypatch, executor):
    monkeypatch.setenv("T2C_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("T2C_RESULTS_DIR", str(tmp_path / "results"))
    bench = TinyBench()
    gt = bench.dir / "gt"
    res = executor.run([{"code": BOX, "save_mesh": str(gt / "box.npz")}, {"code": CYL, "save_mesh": str(gt / "cyl.npz")}])
    assert all(r["ok"] for r in res)
    bench._write(
        [Sample("a", "box", "easy", "gt/box.npz"), Sample("b", "cyl", "easy", "gt/cyl.npz"), Sample("c", "sphere", "hard", "gt/box.npz")],
        {"source": "test"},
    )
    return bench


def test_eval_loop_retries_and_metrics(tiny, executor):
    spec = ModelSpec(name="scripted", model="none", backend="hf")
    backend = ScriptedBackend()
    out, rows, _ = evalrun.evaluate(spec, backend, tiny, executor, style="native", retries=1)
    by_id = {r["id"]: r for r in rows}
    assert by_id["a"]["valid"] and by_id["a"]["n_attempts"] == 1 and by_id["a"]["iou"] == pytest.approx(1.0, abs=1e-6)
    assert by_id["b"]["valid"] and not by_id["b"]["first_valid"] and by_id["b"]["n_attempts"] == 2
    assert not by_id["c"]["valid"] and by_id["c"]["error_type"] == "no_code"
    m = out["metrics"]["overall"]
    assert m["ir_first"] == pytest.approx(2 / 3) and m["ir"] == pytest.approx(1 / 3)
    assert m["iou"] == pytest.approx(1.0, abs=1e-6) and m["iou_all"] == pytest.approx(2 / 3, abs=1e-6)
    assert out["metrics"]["groups"]["hard"]["ir"] == 1.0
    # The retry turn carries the failed script and the kernel's error message back to the model.
    retry_conv = next(c for c in backend.calls[1] if c[1]["content"] == "cyl")
    assert retry_conv[0]["content"] == prompts.NATIVE_SYSTEM
    assert "fillet(99)" in retry_conv[2]["content"] and "Error Message" in retry_conv[3]["content"]
    # Zero retries: the broken script stays broken.
    out0, _, _ = evalrun.evaluate(spec, ScriptedBackend(), tiny, executor, style="native", retries=0)
    assert out0["metrics"]["overall"]["ir"] == pytest.approx(2 / 3)


def test_store_and_report(tiny, executor, monkeypatch):
    from t2c.bench import registry

    monkeypatch.setitem(registry._BENCHMARKS, "tiny", tiny)
    monkeypatch.setitem(report.HEADLINE, "tiny", ("iou_all", "IoU_all", True, False))
    for name, stage, retries in (("base-model", "base", 0), ("tuned-model", "sft", 1)):
        spec = ModelSpec(name=name, model="org/base" if stage == "base" else "/ckpt", backend="hf", stage=stage, base_model="org/base")
        out, rows, timing = evalrun.evaluate(spec, ScriptedBackend(), tiny, executor, style="native", retries=retries)
        store.save_run({"run_id": store.new_run_id(name, "tiny"), "created": f"2026-01-0{retries + 1}T00:00:00", "benchmark": "tiny", "model": spec.to_dict(), **out, "timing": timing, "partial": False}, rows)
    runs = store.list_runs()
    assert [r["model"]["name"] for r in runs] == ["base-model", "tuned-model"]
    assert len(store.load_rows(runs[0]["run_id"])) == 3
    md_path, html_path = report.build()
    md = md_path.read_text()
    assert "tuned-model" in md and "base-model" in md
    assert "(+0.333)" in md  # tuned 2/3 vs base 1/3 IoU_all: the delta against the base model is shown
    assert "<table>" in html_path.read_text()


def test_model_spec_from_checkpoint(tmp_path):
    from t2c.infer.backends import META_FILE, load_model_spec

    run = tmp_path / "sft-x"
    ckpt = run / "checkpoint-500"
    ckpt.mkdir(parents=True)
    (run / META_FILE).write_text(json.dumps({"train_run": "sft-x", "stage": "sft", "base_model": "Qwen/Qwen3-1.7B", "chat_template_kwargs": {"enable_thinking": False}}))
    spec = load_model_spec(str(ckpt))
    assert (spec.name, spec.stage, spec.step, spec.prompt_style) == ("sft-x@500", "sft", 500, "native")
    assert spec.chat_template_kwargs == {"enable_thinking": False} and spec.base_model == "Qwen/Qwen3-1.7B"
    assert Namespace  # keep import used
