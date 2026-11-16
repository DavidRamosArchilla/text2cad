# Source this file (do not execute it): `source scripts/env.sh`
# Sets up paths for the current cluster. Everything cluster-specific lives in
# configs/clusters/<name>.env; pick one with T2C_CLUSTER=<name>, otherwise the profile
# whose T2C_HOST_PATTERN matches this hostname is used, otherwise built-in defaults.

T2C_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export T2C_REPO
export PATH="$HOME/.local/bin:$PATH"   # where bootstrap.sh installs uv

_t2c_profile=""
if [ -n "${T2C_CLUSTER:-}" ]; then
    _t2c_profile="$T2C_REPO/configs/clusters/${T2C_CLUSTER}.env"
    [ -f "$_t2c_profile" ] || { echo "t2c: no such cluster profile: $_t2c_profile" >&2; return 1; }
else
    for _f in "$T2C_REPO"/configs/clusters/*.env; do
        [ -f "$_f" ] || continue
        _pat="$(sed -n 's/^# *T2C_HOST_PATTERN=//p' "$_f" | head -1)"
        if [ -n "$_pat" ] && hostname -f 2>/dev/null | grep -Eq "$_pat"; then _t2c_profile="$_f"; break; fi
    done
fi
if [ -n "$_t2c_profile" ]; then
    # shellcheck disable=SC1090
    source "$_t2c_profile"
    export T2C_CLUSTER="${T2C_CLUSTER:-$(basename "$_t2c_profile" .env)}"
fi
unset _t2c_profile _f _pat

export T2C_DATA_DIR="${T2C_DATA_DIR:-$T2C_REPO/data}"
export T2C_OUTPUT_DIR="${T2C_OUTPUT_DIR:-$T2C_REPO/outputs}"
export T2C_RESULTS_DIR="${T2C_RESULTS_DIR:-$T2C_REPO/results}"
export TOKENIZERS_PARALLELISM=false
# Activate the project environment if it exists.
if [ -x "$T2C_REPO/.venv/bin/python" ]; then
    export VIRTUAL_ENV="$T2C_REPO/.venv"
    export PATH="$VIRTUAL_ENV/bin:$PATH"
fi
