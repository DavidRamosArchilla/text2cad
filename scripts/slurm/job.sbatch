#!/bin/bash
# Generic job body: sets up the environment and runs the command given as arguments.
# Submit through scripts/submit.sh, which fills in partition / GPUs / CPUs for the cluster.
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
source "${T2C_REPO:?submit with scripts/submit.sh}/scripts/env.sh"
cd "$T2C_REPO"
# One CAD verifier process per allocated core, minus a couple for the main process.
export T2C_CAD_WORKERS="${T2C_CAD_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-4} > 4 ? ${SLURM_CPUS_PER_TASK:-4} - 2 : 2 ))}"
echo "[t2c] job $SLURM_JOB_ID on $(hostname)  cluster=${T2C_CLUSTER:-default}  gpus=${CUDA_VISIBLE_DEVICES:-none}  cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "[t2c] git $(git rev-parse --short HEAD 2>/dev/null || echo n/a)  cmd: $*"
exec "$@"
