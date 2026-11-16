#!/bin/bash
# Submit any command as a Slurm job using the current cluster profile.
#
#   scripts/submit.sh [--gpus N] [--cpus N] [--mem GB] [--time D-HH:MM:SS] [--name NAME]
#                     [--after JOBID[:JOBID...]] -- <command ...>
#
# --after makes the job wait until the listed jobs have finished successfully, so a whole
# pipeline (train -> evaluate -> next stage) can be queued at once.
#
# Examples
#   scripts/submit.sh --cpus 64 --name prep -- t2c data prepare
#   scripts/submit.sh --gpus 1 --name eval-q3 -- t2c eval --model configs/models/qwen3-1.7b.yaml
#   scripts/submit.sh --gpus 2 --time 2-00:00:00 -- t2c train sft configs/sft/qwen3-1.7b.yaml
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

gpus=0; cpus=""; mem=""; time_limit="1-00:00:00"; name="t2c"; after=""
while [ $# -gt 0 ]; do
    case "$1" in
        --gpus) gpus="$2"; shift 2 ;;
        --cpus) cpus="$2"; shift 2 ;;
        --mem) mem="$2"; shift 2 ;;
        --time) time_limit="$2"; shift 2 ;;
        --name) name="$2"; shift 2 ;;
        --after) after="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "unknown option: $1 (did you forget '--' before the command?)" >&2; exit 2 ;;
    esac
done
[ $# -gt 0 ] || { sed -n '2,14p' "${BASH_SOURCE[0]}" >&2; exit 2; }

per_gpu_cpus="${T2C_CPUS_PER_GPU:-16}"; per_gpu_mem="${T2C_MEM_PER_GPU_GB:-64}"
n=$(( gpus > 0 ? gpus : 1 ))
cpus="${cpus:-$(( per_gpu_cpus * n ))}"
mem="${mem:-$(( per_gpu_mem * n ))}"

flags=(--job-name="$name" --cpus-per-task="$cpus" --mem="${mem}G" --time="$time_limit")
[ -n "${T2C_SLURM_PARTITION:-}" ] && flags+=(--partition="$T2C_SLURM_PARTITION")
[ -n "${T2C_SLURM_ACCOUNT:-}" ] && flags+=(--account="$T2C_SLURM_ACCOUNT")
# shellcheck disable=SC2059
[ "$gpus" -gt 0 ] && flags+=("$(printf -- "${T2C_GPU_FLAG:---gres=gpu:%d}" "$gpus")")
[ -n "$after" ] && flags+=(--dependency="afterok:$after" --kill-on-invalid-dep=yes)
# shellcheck disable=SC2206
[ -n "${T2C_SLURM_EXTRA:-}" ] && flags+=(${T2C_SLURM_EXTRA})

logdir="$T2C_OUTPUT_DIR/slurm"; mkdir -p "$logdir"
flags+=(--output="$logdir/%x-%j.out" --export=ALL,T2C_REPO="$T2C_REPO")
sbatch --parsable "${flags[@]}" "$T2C_REPO/scripts/slurm/job.sbatch" "$@" | { read -r id; echo "Submitted batch job $id"; }
