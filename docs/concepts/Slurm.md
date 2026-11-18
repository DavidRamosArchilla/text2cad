---
tags: [concept]
aliases: [sbatch, squeue, job scheduler]
---
# Slurm

The job scheduler on both clusters: you describe the resources a job needs and Slurm
starts it on a node when they are free.

Commands you will use:
| Command | Purpose |
|---|---|
| `sbatch script.sh` | submit a batch job (our wrapper: `scripts/submit.sh`) |
| `squeue -u $USER` | your queued and running jobs |
| `scancel <id>` | cancel a job |
| `sacct -j <id>` | accounting info after a job ended (state, time, memory) |
| `srun --jobid <id> --overlap nvidia-smi` | run a command inside a running job (e.g. look at GPU use) |
| `sinfo` | partitions and node states |

Options that matter here: `--partition`, `--gres=gpu:N` (or `--gpus=N`, cluster-specific),
`--cpus-per-task` (CAD verification uses one process per core), `--mem`, `--time`,
`--dependency=afterok:<id>` (start after another job succeeded; `submit.sh --after`).

How this repository wraps it: [[Scripts and clusters]].

## Learn more
- Slurm quick start: <https://slurm.schedmd.com/quickstart.html>
- `sbatch` reference (all options, including dependencies): <https://slurm.schedmd.com/sbatch.html>
