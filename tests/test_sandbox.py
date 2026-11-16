from t2c.cad import codeutil
from t2c.cad.reward import RewardConfig, reward_from_result

GOOD = "import cadquery as cq\nresult = cq.Workplane('XY').box(10,20,30)\n"


def run(executor, code, **kw):
    return executor.run_one({"code": code, **kw})


def test_error_taxonomy(executor):
    assert run(executor, GOOD)["stage"] == 3
    assert run(executor, None)["error_type"] == "no_code"
    assert run(executor, "import cadquery as cq\nresult = (\n")["error_type"] == "syntax_error"
    r = run(executor, "import cadquery as cq\nresult = cq.Workplane('XY').box(1,2,3).fillet(5)\n")
    assert (r["error_type"], r["stage"]) == ("runtime_error", 1) and "line 2" in r["error"]
    assert run(executor, "import cadquery as cq\npart = cq.Workplane('XY').box(1,2,3)\n")["error_type"] == "no_result"
    assert run(executor, "import cadquery as cq\nresult = cq.Workplane('XY').rect(1,2)\n")["error_type"] == "no_solid"
    assert run(executor, "import cadquery as cq\nimport subprocess\n")["error_type"] == "unsafe_code"


def test_timeout_and_crash_are_contained(executor):
    assert run(executor, "import cadquery as cq\nwhile True: pass\n", timeout=2)["error_type"] == "timeout"
    crash = "import cadquery as cq\nimport faulthandler\nfaulthandler._sigsegv()\n"
    assert run(executor, crash)["error_type"] == "crash"
    assert run(executor, GOOD)["ok"]  # the pool survives both


def test_scratch_dir_and_no_network(executor, tmp_path):
    r = run(executor, "import cadquery as cq\nopen('junk.txt','w').write('x')\nresult = cq.Workplane('XY').box(1,1,1)\n")
    assert r["ok"]
    net = "import cadquery as cq\nimport _socket as s\nk = s.socket(s.AF_INET, s.SOCK_STREAM)\nk.settimeout(3)\nk.connect(('1.1.1.1', 443))\n"
    if executor.info["net_isolated"]:
        assert "unreachable" in run(executor, net)["error"].lower()


def test_order_is_preserved(executor):
    jobs = [{"code": f"import cadquery as cq\nresult = cq.Workplane('XY').box({i + 1},1,1)\n"} for i in range(40)]
    out = executor.run(jobs)
    assert [round(r["bbox"][0]) for r in out] == list(range(1, 41))


def test_extract_code():
    assert codeutil.extract_code("text\n```python\nimport cadquery as cq\nresult = 1\n```\nbye") == "import cadquery as cq\nresult = 1\n"
    two = "```python\nimport cadquery as cq\nresult = 1\n```\nfixed:\n```python\nimport cadquery as cq\nresult = 2\n```"
    assert codeutil.extract_code(two).endswith("result = 2\n")
    assert codeutil.extract_code("<think>```python\nimport cadquery as cq\nx=1\n```</think>") is None
    assert codeutil.extract_code("```python\nimport cadquery as cq\nresult = cq.Workplane()") is not None  # truncated fence
    assert codeutil.extract_code("I cannot do that.") is None


def test_normalize_reference_script():
    raw = (
        "import cadquery as cq\nfrom cadquery.vis import show\npart = cq.Workplane('XY').box(1,2,3)\n"
        "assembly = part\n# --- Export to STL ---\ncq.exporters.export(assembly, './stlcq/0038/00388122.stl')\n"
    )
    out = codeutil.normalize_reference_script(raw)
    assert "export" not in out and "vis" not in out and out.rstrip().endswith("result = part")
    assert codeutil.normalize_reference_script("print('hi')\n") is None


def test_reward_is_monotone():
    cfg = RewardConfig()
    ladder = [
        {"ok": False, "stage": 0},
        {"ok": False, "stage": 1},
        {"ok": False, "stage": 2},
        {"ok": True, "stage": 3, "iou": 0.1, "cd": 0.25, "dim_err": 1.0},
        {"ok": True, "stage": 3, "iou": 0.6, "cd": 0.08, "dim_err": 0.2},
        {"ok": True, "stage": 3, "iou": 1.0, "cd": 0.0, "dim_err": 0.0},
    ]
    r = [reward_from_result(x, cfg) for x in ladder]
    assert r == sorted(r) and len(set(r)) == len(r)
    assert abs(r[-1] - 1.0) < 1e-9
