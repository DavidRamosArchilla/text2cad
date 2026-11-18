---
tags: [guide]
aliases: [scripts, Slurm usage, Second cluster]
---
# Scripts and clusters

Shell scripts in `scripts/` that make the same commands work on any cluster. Background:
[[Slurm]], [[Reproducible environments]].

## `scripts/bootstrap.sh` — create the environment (once per cluster)
1. Installs a pinned version of **uv** into `~/.local/bin` (your shell profile is not touched).
2. `uv sync --frozen --extra gpu --extra dev`: creates `.venv/` with *exactly* the package
   versions in `uv.lock` and uv's own standalone Python 3.12 (`python-preference =
   "only-managed"` in `pyproject.toml`), so both clusters run identical software.
   `--cpu` skips the GPU packages.
3. Runs `t2c doctor`.

## `scripts/env.sh` — source it in every shell and job
`source scripts/env.sh`:
- picks a **cluster profile** from `configs/clusters/`: the one named by `T2C_CLUSTER`, or
  the one whose `# T2C_HOST_PATTERN=` regex matches `hostname -f`;
- exports `T2C_DATA_DIR`, `T2C_OUTPUT_DIR`, `T2C_RESULTS_DIR` (defaults under the repo; see [[paths]]);
- activates `.venv`, so `t2c` is on the `PATH`.

## `configs/clusters/<name>.env` — what differs between clusters
| Variable | Example | Meaning |
|---|---|---|
| `T2C_SLURM_PARTITION` | `gpu` | partition |
| `T2C_SLURM_ACCOUNT` | (empty) | account, if the cluster uses them |
| `T2C_SLURM_EXTRA` | `--qos=long` | any extra `sbatch` flags |
| `T2C_GPU_FLAG` | `--gres=gpu:%d` | how this cluster asks for N GPUs |
| `T2C_CPUS_PER_GPU`, `T2C_MEM_PER_GPU_GB` | 24, 120 | default resources per GPU |
| `T2C_DATA_DIR`, `T2C_OUTPUT_DIR`, `HF_HOME` | `/scratch/$USER/…` | big storage |
| `HF_HUB_OFFLINE=1` | | for compute nodes without internet |

`TEMPLATE.env.example` is the starting point for a new cluster.

## `scripts/submit.sh` — run any command as a Slurm job
```bash
scripts/submit.sh [--gpus N] [--cpus N] [--mem GB] [--time D-HH:MM:SS] [--name NAME] [--after JOBID] -- <command …>
```
It builds the `sbatch` flags from the profile (CPU and memory default to *per-GPU values ×
GPUs*), writes the log to `outputs/slurm/<name>-<jobid>.out` and submits
`scripts/slurm/job.sbatch`, which sources `env.sh`, sets `T2C_CAD_WORKERS` from the
allocated cores and runs the command. `--after` chains jobs (`--dependency=afterok`), so a
whole pipeline can be queued at once:

```bash
sft=$(scripts/submit.sh --gpus 1 --name sft -- t2c train sft configs/sft/qwen3-1.7b.yaml | awk '{print $4}')
scripts/submit.sh --gpus 1 --after $sft --name eval -- t2c eval --model outputs/train/sft-qwen3-1.7b-v1/final
```

**CPU cores matter**: every evaluation and every GRPO step runs CadQuery scripts on the
CPU, one process per core. Ask for 48+ cores per GPU for GRPO.

## `scripts/eval_checkpoints.sh` — progress curve of a training run
Runs `t2c eval` on every `checkpoint-*` of a run and on `final`, so the report gets a
"Training runs" table. Extra arguments go to `t2c eval` (e.g. `--bench cqe deepcad`).

## Setting up the second cluster
1. `git clone` + `scripts/bootstrap.sh`.
2. Copy `configs/clusters/TEMPLATE.env.example` to `configs/clusters/<name>.env`, fill in
   partition, account, GPU flag, storage, `T2C_HOST_PATTERN`.
3. `t2c data download`, `t2c data prepare`, `t2c data synth`, `t2c bench prepare`,
   `t2c data build …` — sources are pinned and everything is seeded, so the result is
   identical (compare the `sha1` in `data/bench/*/info.json`).
4. Results travel through git: `results/runs/*/summary.json` is committed; per-sample
   files stay local.
