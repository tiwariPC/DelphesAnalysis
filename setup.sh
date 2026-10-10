#!/usr/bin/env bash
# Create (if missing) and activate the conda env used by the whole pipeline.
#
#   source setup.sh        # creates env if needed AND leaves it active in your shell
#   ./setup.sh             # creates env if needed, only checks it (cannot activate your shell)
#
# Env name: delphesanalysis (override: ENV_NAME=foo source setup.sh)
# Python 3.11: inside the range MadGraph's model reader tolerates (<= 3.12).

ENV_NAME="${ENV_NAME:-delphesanalysis}"
PY_VERSION="${PY_VERSION:-3.11}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

_setup_fail() { echo "[setup] ERROR: $*" >&2; return 1 2>/dev/null || exit 1; }

command -v conda >/dev/null 2>&1 || _setup_fail "conda not found in PATH (install miniforge/miniconda first)" || return 1 2>/dev/null || exit 1

# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] env '$ENV_NAME' exists"
else
    echo "[setup] creating env '$ENV_NAME' (python $PY_VERSION)"
    conda create -y -n "$ENV_NAME" -c conda-forge "python=$PY_VERSION" pip || _setup_fail "conda create failed" || return 1 2>/dev/null || exit 1
fi

conda activate "$ENV_NAME" || _setup_fail "cannot activate '$ENV_NAME'" || return 1 2>/dev/null || exit 1
echo "[setup] active env: $CONDA_DEFAULT_ENV ($(python --version 2>&1))"

# Python requirements (pip skips what is already satisfied).
if ! python -c "import coffea, hist, awkward, uproot, yaml, matplotlib, seaborn, numpy" >/dev/null 2>&1; then
    echo "[setup] installing requirements.txt"
    python -m pip install -r "$REPO_DIR/requirements.txt" || _setup_fail "pip install failed" || return 1 2>/dev/null || exit 1
fi

# ROOT: required by MadGraph's 'install Delphes'. Reuse any root-config on PATH
# (including the one from conda base); otherwise install ROOT into the env.
if ! command -v root-config >/dev/null 2>&1; then
    echo "[setup] root-config not found, installing ROOT from conda-forge (large download)"
    conda install -y -n "$ENV_NAME" -c conda-forge root || echo "[setup] WARNING: ROOT install failed; MadGraph Delphes install will fail"
fi

# Compilers MadGraph needs.
for tool in gfortran g++ make; do
    command -v "$tool" >/dev/null 2>&1 || echo "[setup] WARNING: '$tool' not found in PATH (needed by MadGraph)"
done

echo "[setup] python : $(command -v python)"
echo "[setup] root   : $(root-config --version 2>/dev/null || echo missing)"
echo "[setup] combine: $(command -v combine || echo 'not found (only needed for run_all.sh --limits)')"
if [ "${BASH_SOURCE[0]:-$0}" = "$0" ]; then
    echo "[setup] run as a script: env verified, but not active in your shell. Use: source setup.sh"
fi
