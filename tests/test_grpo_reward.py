"""The GRPO reward function, exercised the way trl.GRPOTrainer calls it (no GPU needed)."""

import json

from t2c.cad.reward import RewardConfig
from t2c.train.grpo import CadReward

BOX = "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 20, 30)\n"
SLAB = "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 20, 3)\n"


def test_reward_orders_completions(executor, tmp_path):
    gt = tmp_path / "gt.npz"
    assert executor.run_one({"code": BOX, "save_mesh": str(gt)})["ok"]
    completions = [
        [{"role": "assistant", "content": f"```python\n{BOX}```"}],  # exact
        [{"role": "assistant", "content": f"```python\n{SLAB}```"}],  # valid, wrong proportions
        [{"role": "assistant", "content": f"Sure! Here is the code you asked for:\n```python\n{BOX}```\nLet me know if you need anything else."}],
        [{"role": "assistant", "content": "```python\nimport cadquery as cq\nresult = cq.Workplane('XY').box(1, 2, 3).fillet(9)\n```"}],
        [{"role": "assistant", "content": "```python\nimport cadquery as cq\nresult = (\n```"}],
        f"```python\n{BOX}```",  # plain-string completions are accepted too
    ]
    logged = {}
    log = tmp_path / "reward_log.jsonl"
    reward = CadReward(executor, RewardConfig(), chatter_penalty=0.05, log_path=log)
    r = reward(prompts=[None] * 6, completions=completions, gt=[str(gt)] * 6, log_metric=lambda k, v: logged.__setitem__(k, v))
    exact, slab, chatty, runtime, syntax, plain = r
    assert exact > 0.95 and plain == exact
    assert chatty == exact - 0.05  # same part, penalised for prose around the code
    assert exact > slab > runtime > syntax == 0.0
    assert logged["cad/valid_rate"] == 4 / 6
    assert json.loads(log.read_text().splitlines()[0])["cad/err_syntax_error"] == 1 / 6
