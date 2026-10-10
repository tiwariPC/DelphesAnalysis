#!/usr/bin/env bash
# Create (if missing) and activate the conda env used by the whole pipeline.
#
#   source setup.sh        # creates env if needed AND leaves it active in your shell
#   ./setup.sh             # creates env if needed, only checks it (cannot activate your shell)
#
# Env name: delphesanalysis (override: ENV_NAME=foo source setup.sh)
# Packages come from environment.yml (python 3.11, compilers, analysis libraries).

ENV_NAME="${ENV_NAME:-delphesanalysis}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

_setup_fail() { echo "[setup] ERROR: $*" >&2; return 1 2>/dev/null || exit 1; }

command -v conda >/dev/null 2>&1 || _setup_fail "conda not found in PATH (install miniforge/miniconda first)" || return 1 2>/dev/null || exit 1

# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] env '$ENV_NAME' exists"
else
    echo "[setup] creating env '$ENV_NAME' from environment.yml (python, compilers, analysis libraries)"
    (cd "$REPO_DIR" && conda env create -y -n "$ENV_NAME" -f environment.yml) || _setup_fail "conda env create failed" || return 1 2>/dev/null || exit 1
fi

conda activate "$ENV_NAME" || _setup_fail "cannot activate '$ENV_NAME'" || return 1 2>/dev/null || exit 1
echo "[setup] active env: $CONDA_DEFAULT_ENV ($(python --version 2>&1))"

# Packages of an already existing env: bring them in line with environment.yml.
if ! python -c "import coffea, hist, awkward, uproot, yaml, matplotlib, seaborn, numpy" >/dev/null 2>&1; then
    echo "[setup] updating env from environment.yml"
    conda env update -n "$ENV_NAME" -f "$REPO_DIR/environment.yml" || _setup_fail "conda env update failed" || return 1 2>/dev/null || exit 1
fi

# ROOT: required by MadGraph's 'install Delphes'. Reuse any root-config on PATH
# (including the one from conda base); otherwise install ROOT into the env.
if ! command -v root-config >/dev/null 2>&1; then
    echo "[setup] root-config not found, installing ROOT from conda-forge (large download)"
    conda install -y -n "$ENV_NAME" -c conda-forge root || echo "[setup] WARNING: ROOT install failed; MadGraph Delphes install will fail"
fi

# Compilers MadGraph needs (Fortran for matrix elements, C++ for Pythia8/Delphes).
# Missing ones are installed into the env from conda-forge.
_pkgs=""
command -v gfortran >/dev/null 2>&1 || _pkgs="$_pkgs gfortran"
command -v g++      >/dev/null 2>&1 || _pkgs="$_pkgs gxx"
command -v make     >/dev/null 2>&1 || _pkgs="$_pkgs make"
if [ -n "$_pkgs" ]; then
    echo "[setup] installing compilers:$_pkgs"
    # shellcheck disable=SC2086
    conda install -y -n "$ENV_NAME" -c conda-forge $_pkgs || echo "[setup] WARNING: compiler install failed"
    # Re-activate so conda's compiler activation scripts run, then make sure the
    # plain names MadGraph searches for exist (conda ships <triplet>-gfortran / -g++).
    conda deactivate; conda activate "$ENV_NAME"
    if ! command -v gfortran >/dev/null 2>&1 && [ -n "${FC:-}" ] && [ -x "$(command -v "$FC")" ]; then
        ln -sf "$(command -v "$FC")" "$CONDA_PREFIX/bin/gfortran"
    fi
    if ! command -v g++ >/dev/null 2>&1 && [ -n "${CXX:-}" ] && [ -x "$(command -v "$CXX")" ]; then
        ln -sf "$(command -v "$CXX")" "$CONDA_PREFIX/bin/g++"
    fi
fi
for tool in gfortran g++ make; do
    command -v "$tool" >/dev/null 2>&1 || echo "[setup] WARNING: '$tool' still not found in PATH (needed by MadGraph)"
done

echo "[setup] python : $(command -v python)"
echo "[setup] root   : $(root-config --version 2>/dev/null || echo missing)"
echo "[setup] combine: $(command -v combine || echo 'not found (only needed for run_all.sh --limits)')"
if [ "${BASH_SOURCE[0]:-$0}" = "$0" ]; then
    echo "[setup] run as a script: env verified, but not active in your shell. Use: source setup.sh"
fi
