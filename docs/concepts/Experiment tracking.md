---
tags: [concept]
aliases: [TensorBoard, W&B, trackio, callbacks]
---
# Experiment tracking

Training writes metrics at every few steps; a tracker turns them into curves you can
compare across runs.

## What is logged here
| Source | Metrics | Where |
|---|---|---|
| transformers / TRL Trainer | loss, eval loss, learning rate, gradient norm; for GRPO: reward mean/std, completion lengths, clipped ratio, `frac_reward_zero_std` | tracker named in `report_to` |
| `CadReward` ([[train.grpo]]) | `cad/valid_rate`, `cad/iou_all`, `cad/iou_valid`, error shares, verification time | tracker + `reward_log.jsonl` |
| benchmark callback ([[train.callbacks]]) | `eval/<benchmark>_{ir, iou_all, iou, pass_rate}` | tracker + `bench_log.jsonl` |
| end of training | full log history | `final/trainer_state.json` |

`report_to: [tensorboard]` is the default (event files in `outputs/train/<run>/`). Add
`wandb` (hosted, needs an account) or `trackio` (local, W&B-like) if you prefer.

Viewing TensorBoard on the cluster:
```bash
tensorboard --logdir outputs/train --port 6006     # on the login node
ssh -L 6006:localhost:6006 <cluster>               # on your laptop, then open http://localhost:6006
```

**Callbacks** are how extra logic hooks into the Trainer's loop (`on_step_end`,
`on_train_end`, …); our benchmark scoring is one.

What to look for: training loss well below eval loss (overfitting); reward rising while
`eval/*` stays flat (the model learns the reward, not the task); completion length
creeping up in GRPO.

## Learn more
- TensorBoard — get started: <https://www.tensorflow.org/tensorboard/get_started>
- transformers — callbacks: <https://huggingface.co/docs/transformers/main_classes/callback>
- Weights & Biases docs: <https://docs.wandb.ai/> · trackio: <https://github.com/gradio-app/trackio>
