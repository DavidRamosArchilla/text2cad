"""The procedural generator is training data *and* a benchmark's ground truth, so the
three things it emits per part -- script, geo text, seq text -- must agree with each other
and with the geometry the script actually builds."""

import json
import random
import re

import pytest

from t2c.data import synth

SEEDS = range(10)


def _part(family: str, seed: int) -> synth.Part:
    return synth.make_part(family, random.Random(f"test/{family}/{seed}"))


@pytest.fixture(scope="module")
def verified(executor):
    """Every family x 10 seeds, executed once in the sandbox."""
    parts = [_part(fam, s) for fam in synth.FAMILIES for s in SEEDS]
    results = executor.run([{"code": p.code, "self_check": 48} for p in parts])
    return list(zip(parts, results, strict=True))


def _mentions(text: str, value: float) -> bool:
    s = synth.num(value)
    tail = "" if "." in s else r"(?:\.0)?"
    return re.search(rf"(?<![\d.]){re.escape(s)}{tail}(?!\.?\d)", text) is not None


def test_registry_covers_enough_families():
    assert len(synth.FAMILIES) >= 14


def test_every_family_builds_one_watertight_solid(verified):
    bad = [(p.family, synth.check_part(p, r), r.get("error")) for p, r in verified if synth.check_part(p, r)]
    assert not bad, bad[:5]
    for p, r in verified:
        assert r["n_solids"] == 1 and r["open_frac"] <= 0.005, p.family


def test_scripts_follow_the_project_contract(verified):
    for p, _ in verified:
        assert p.code.startswith("import cadquery as cq\n"), p.family
        assert re.search(r"^result = ", p.code, re.M) or re.search(r"^    result = ", p.code, re.M), p.family
        assert not re.search(r"export|show_object|display\(", p.code), p.family


def test_generation_is_deterministic():
    for fam in synth.FAMILIES:
        a, b = _part(fam, 3), _part(fam, 3)
        assert (a.code, a.geo, a.seq, a.bbox) == (b.code, b.geo, b.seq, b.bbox)
    assert synth._candidate(0, "test", 5).code == synth._candidate(0, "test", 5).code
    assert synth._candidate(0, "test", 5).code != synth._candidate(1, "test", 5).code


@pytest.mark.parametrize("family", ["plate_holes", "flange", "stepped_shaft", "l_bracket", "beam_profile", "swept_elbow", "polygon_prism"])
def test_executed_bbox_equals_stated_dimensions(verified, family):
    """The extents the generator intends (and writes into the prompts) are what the script builds."""
    seen = 0
    for p, r in verified:
        if p.family != family:
            continue
        seen += 1
        tol = 0.004 * max(p.bbox)  # the mesh lies slightly inside curved surfaces
        assert r["bbox"] == pytest.approx(p.bbox, abs=tol), p.code
        if p.volume is not None:
            assert r["volume"] == pytest.approx(p.volume, rel=2e-3), p.code
    assert seen == len(SEEDS)


def test_both_prompts_state_every_key_dimension():
    for fam in synth.FAMILIES:
        for s in range(40):
            p = _part(fam, s)
            assert p.dims, fam
            for style in ("geo", "seq"):
                text = getattr(p, style)
                for name, value in p.dims.items():
                    # circles may be given by radius instead of diameter
                    ok = _mentions(text, value) or (name.startswith("d_") and _mentions(text, value / 2))
                    assert ok, f"{fam} {style}: {name}={value} missing in: {text}"
                assert re.search(r"\b[XYZ]\b|[XYZ]-axis|X[YZ] plane|YZ plane", text), f"{fam} {style}: no axis reference"


def _probe(code: str, void: list[tuple], solid: list[tuple]) -> str:
    """Script + in-sandbox assertions that given points are empty / filled."""
    lines = [code, "_s = result.val()"]
    lines += [f"assert not _s.isInside({pt!r}), 'expected a hole at {pt!r}'" for pt in void]
    lines += [f"assert _s.isInside({pt!r}), 'expected material at {pt!r}'" for pt in solid]
    return "\n".join(lines) + "\n"


def test_feature_positions_match_the_text(executor):
    """Holes are where the prompt says they are (not merely the right number of them)."""
    jobs = []
    for s in range(12):
        p = _part("l_bracket", s)
        t = p.dims["t"]
        num, where = r"(\d+(?:\.\d+)?)", r"(?:, Y = 0| and Y = \+/-(\d+(?:\.\d+)?))"
        mx = re.search(rf"horizontal leg along Z, centered at X = {num}{where}", p.seq)
        mz = re.search(rf"vertical leg along X, centered at Z = {num}{where}", p.seq)
        hx, y1, vz, y2 = float(mx.group(1)), float(mx.group(2) or 0), float(mz.group(1)), float(mz.group(2) or 0)
        jobs.append({"code": _probe(p.code, [(hx, y1, t / 2), (hx, -y1, t / 2), (t / 2, y2, vz), (t / 2, -y2, vz)], [(t / 2, 0.0, t / 2), (p.dims["A"] - 0.5, 0.0, t / 2)])})
    for s in range(12):
        p = _part("flange", s)
        r, t = p.dims["d_pcd"] / 2, p.dims["T"]
        between = (r + p.dims["d_bolt"] / 2 + 1.0, 0.0, t / 2)  # just outside the hole, still inside the disc
        jobs.append({"code": _probe(p.code, [(r, 0.0, t / 2), (0.0, 0.0, t / 2)], [between])})
    results = executor.run(jobs)
    assert all(r["ok"] for r in results), [r.get("error") for r in results if not r["ok"]][:3]


def test_generate_dataset_writes_disjoint_verified_splits(tmp_path, monkeypatch):
    monkeypatch.setenv("T2C_DATA_DIR", str(tmp_path))
    stats = synth.generate_dataset(n_train=40, n_val=8, n_test=8, seed=0, workers=4)
    assert stats["parts"] == {"test": 8, "val": 8, "train": 40}
    from t2c.data.prepare import read_split

    codes = {}
    for split, n in (("train", 40), ("val", 8), ("test", 8)):
        rows = read_split("synth", split)
        assert len(rows) == 2 * n == stats["splits"][split]
        for geo, seq in zip(rows[0::2], rows[1::2], strict=True):
            assert (geo["style"], seq["style"]) == ("geo", "seq")
            assert geo["uid"] == seq["uid"] and geo["code"] == seq["code"] and geo["text"] != seq["text"]
            assert geo["id"] == f"synth/{split}/{geo['uid']}-geo" and re.fullmatch(rf"{synth.SPLIT_LETTER[split]}\d{{7}}", geo["uid"])
            assert set(geo) == {"id", "uid", "family", "style", "text", "code", "volume", "bbox", "n_solids", "n_tris"}
            assert geo["n_solids"] == 1 and geo["volume"] > 0 and len(geo["bbox"]) == 3
        codes[split] = {r["code"] for r in rows}
    assert not (codes["train"] & codes["test"]) and not (codes["train"] & codes["val"]) and not (codes["val"] & codes["test"])
    saved = json.loads((tmp_path / "processed" / "synth" / "stats.json").read_text())
    assert saved["families"].keys() == synth.FAMILIES.keys()
