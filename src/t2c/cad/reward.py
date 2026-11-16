"""Scalar reward for RL from a verifier result (``worker.run_job`` output).

Dense on purpose: small models collapse under all-or-nothing rewards, so every step
towards a correct part is worth something.

    stage 0  nothing runnable (no code block, syntax error, rejected import)  -> r_none
    stage 1  parses, but raises / times out / crashes the kernel             -> r_parsed
    stage 2  runs, but leaves no usable solid in `result`                    -> r_ran
    stage 3  valid solid                                                     -> r_valid + shape terms

For a valid solid:   r = r_valid + w_iou * IoU + w_cd * cd_score + w_dim * dim_score
with (defaults) r_valid + w_iou + w_cd + w_dim = 1, so a perfect part scores ~1.

* IoU        voxel IoU after unit-bounding-box normalisation (shape, scale-free)
* cd_score   1 - CD / cd_max, clipped to [0, 1]   (gives signal when IoU is ~0, e.g. thin shells)
* dim_score  exp(-dim_err / dim_scale) where dim_err = mean |log(extent_pred / extent_gt)| per axis
             (absolute size and orientation, which normalisation hides)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass
class RewardConfig:
    r_none: float = 0.0
    r_parsed: float = 0.1
    r_ran: float = 0.2
    r_valid: float = 0.3
    w_iou: float = 0.5
    w_cd: float = 0.1
    w_dim: float = 0.1
    cd_max: float = 0.3
    dim_scale: float = 0.25
    # Verifier settings used while training (cheaper than the 256^3 / 30k benchmark settings).
    res: int = 64
    n_points: int = 4000
    timeout: float = 20.0

    def to_dict(self) -> dict:
        return asdict(self)


def reward_from_result(res: dict, cfg: RewardConfig | None = None) -> float:
    cfg = cfg or RewardConfig()
    stage = int(res.get("stage", 0))
    if not res.get("ok"):
        return (cfg.r_none, cfg.r_parsed, cfg.r_ran)[min(stage, 2)]
    r = cfg.r_valid
    iou = res.get("iou")
    if iou is not None and math.isfinite(iou):
        r += cfg.w_iou * max(0.0, min(1.0, iou))
    cd = res.get("cd")
    if cd is not None and math.isfinite(cd):
        r += cfg.w_cd * max(0.0, min(1.0, 1.0 - cd / cfg.cd_max))
    dim = res.get("dim_err")
    if dim is not None and math.isfinite(dim):
        r += cfg.w_dim * math.exp(-dim / cfg.dim_scale)
    return float(r)
