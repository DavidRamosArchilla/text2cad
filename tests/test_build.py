"""Dataset assembly: the leakage rules are the part that must not silently break."""

import json

import yaml

from t2c import paths
from t2c.data import build

CODE = "import cadquery as cq\nresult = cq.Workplane('XY').box({}, 2, 3)\n"


def _write_split(source, split, rows):
    d = paths.ensure(paths.processed_dir(source))
    with (d / f"{split}.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _row(source, split, uid, size=1):
    return {"id": f"{source}/{split}/{uid}", "uid": uid, "text": f"a box {size} long ({uid})", "code": CODE.format(size)}


def test_holdout_objects_never_reach_training(tmp_path, monkeypatch, executor):
    monkeypatch.setenv("T2C_DATA_DIR", str(tmp_path))
    _write_split("t2cq", "train", [_row("t2cq", "train", f"{i:08d}", i + 1) for i in range(10)])
    _write_split("t2cq", "val", [_row("t2cq", "val", "00000100"), _row("t2cq", "val", "00000101")])
    _write_split("t2cq", "test", [_row("t2cq", "test", "00000200")])
    # The same DeepCAD objects described again by another corpus: must be dropped from training.
    _write_split("cadcoder", "train_high", [_row("cadcoder", "train_high", u, 7) for u in ("00000100", "00000200", "00000300")])

    recipe = {
        "name": "sft/test", "kind": "sft", "seed": 0, "holdout": ["t2cq:val", "t2cq:test"],
        "train": [{"source": "t2cq", "split": "train", "n": 6}, {"source": "cadcoder", "split": "train_high"}],
        "val": [{"source": "t2cq", "split": "val"}],
    }  # fmt: skip
    cfg = tmp_path / "recipe.yaml"
    cfg.write_text(yaml.safe_dump(recipe))
    info = build.build(str(cfg))
    train, _ = build.read_built("sft/test", "train")
    val, _ = build.read_built("sft/test", "val")
    assert info["counts"] == {"train": 7, "val": 2}
    train_uids = {r["id"].split("/")[-1] for r in train}
    assert "00000300" in train_uids and not train_uids & {"00000100", "00000200"}
    assert {r["id"].split("/")[-1] for r in val} == {"00000100", "00000101"}
    assert set(train[0]) == {"id", "text", "code"}
    # Same recipe, same seed -> identical dataset (reproducible across clusters).
    first = (paths.data_dir() / "sft/test/train.jsonl").read_text()
    build.build(str(cfg))
    assert (paths.data_dir() / "sft/test/train.jsonl").read_text() == first

    recipe.update(name="rl/test", kind="rl", val=[])
    cfg.write_text(yaml.safe_dump(recipe))
    info = build.build(str(cfg))
    rows, d = build.read_built("rl/test", "train")
    assert info["counts"]["train"] == 7 and all((d / r["gt"]).exists() for r in rows)
    assert set(rows[0]) == {"id", "text", "gt"}


def test_exclude_families(tmp_path, monkeypatch):
    monkeypatch.setenv("T2C_DATA_DIR", str(tmp_path))
    rows = []
    for i, fam in enumerate(["flange", "loft", "wedge", "loft"]):
        rows.append({"id": f"synth/train/t{i}-geo", "uid": f"t{i}", "family": fam, "style": "geo", "text": f"part {i}", "code": CODE.format(i + 1)})
    _write_split("synth", "train", rows)
    cfg = tmp_path / "r.yaml"
    cfg.write_text(yaml.safe_dump({"name": "sft/fam", "kind": "sft", "train": [{"source": "synth", "split": "train", "exclude_families": ["loft"]}], "val": []}))
    assert build.build(str(cfg))["counts"]["train"] == 2
