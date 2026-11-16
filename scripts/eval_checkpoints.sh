#!/bin/bash
# Evaluate every checkpoint of a training run (and its final model) to get a progress curve.
#
#   scripts/eval_checkpoints.sh <run-name-or-dir> [extra `t2c eval` arguments]
#   scripts/submit.sh --gpus 1 --name evalckpt -- scripts/eval_checkpoints.sh sft-qwen3-1.7b-v1 --bench cqe synth deepcad
#
# Every checkpoint is evaluated each time the script runs (results are appended, never
# overwritten; the report uses the latest run per checkpoint).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
run="${1:?usage: eval_checkpoints.sh <run-name-or-dir> [t2c eval args]}"; shift || true
[ -d "$run" ] || run="$T2C_OUTPUT_DIR/train/$run"
[ -d "$run" ] || { echo "no such run directory: $run" >&2; exit 1; }
for ckpt in $(ls -d "$run"/checkpoint-* 2>/dev/null | sort -t- -k2 -n) "$run/final"; do
    [ -d "$ckpt" ] || continue
    echo "=== $ckpt"
    t2c eval --model "$ckpt" "$@"
done
