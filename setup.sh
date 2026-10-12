#!/usr/bin/env bash
# Create (if missing) and activate the conda env used by the whole pipeline.
#
#   source setup.sh        # creates env if needed AND leaves it active in your shell
#   ./setup.sh             # creates env if needed, only checks it (cannot activate your shell)
#
# Works when sourced from bash or zsh.
# Env name: delphesanalysis (override: ENV_NAME=foo source setup.sh)
# Packages come from environment.yml (python 3.11, compilers, analysis libraries).

ENV_NAME="${ENV_NAME:-delphesanalysis}"

# Shell detection: path of this file, and whether it is sourced or executed.
# (zsh has no BASH_SOURCE; ${(%):-%x} is zsh-only syntax, hence the eval.)
if [ -n "${ZSH_VERSION:-}" ]; then
    _setup_shell=zsh
    eval '_setup_self="${(%):-%x}"'
    case "${ZSH_EVAL_CONTEXT:-}" in *:file*) _setup_sourced=1 ;; *) _setup_sourced=0 ;; esac
elif [ -n "${BASH_VERSION:-}" ]; then
    _setup_shell=bash
    _setup_self="${BASH_SOURCE[0]}"
    if [ "${BASH_SOURCE[0]}" != "$0" ]; then _setup_sourced=1; else _setup_sourced=0; fi
else
    echo "[setup] ERROR: source this file from bash or zsh" >&2
    return 1 2>/dev/null || exit 1
fi
REPO_DIR="$(cd "$(dirname "$_setup_self")" && pwd)"

_setup_fail() { echo "[setup] ERROR: $*" >&2; return 1 2>/dev/null || exit 1; }

command -v conda >/dev/null 2>&1 || _setup_fail "conda not found in PATH (install miniforge/miniconda first)" || return 1 2>/dev/null || exit 1

# shellcheck disable=SC1091
eval "$(conda "shell.$_setup_shell" hook)"

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
# (arrays, not a space-separated string: zsh does not word-split "$var" when sourced)
_pkgs=()
command -v gfortran >/dev/null 2>&1 || _pkgs+=(gfortran)
command -v g++      >/dev/null 2>&1 || _pkgs+=(gxx)
command -v make     >/dev/null 2>&1 || _pkgs+=(make)
if [ ${#_pkgs[@]} -gt 0 ]; then
    echo "[setup] installing compilers: ${_pkgs[*]}"
    conda install -y -n "$ENV_NAME" -c conda-forge "${_pkgs[@]}" || echo "[setup] WARNING: compiler install failed"
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

# OPTIONAL: build CMS Combine (HiggsAnalysis-CombinedLimit) into this env via its
# standalone CMake build. Source is cloned into <repo>/combine/, binaries and the
# python package are installed into $CONDA_PREFIX, so `combine`, `text2workspace.py`
# and `import HiggsAnalysis.CombinedLimit` work whenever the env is active.
# Only needed for run_all.sh --limits.
#   INSTALL_COMBINE=1 source setup.sh
#   COMBINE_TAG=v10.2.1 INSTALL_COMBINE=1 source setup.sh   # pin a release (default: main)
if [ "${INSTALL_COMBINE:-0}" = "1" ]; then
    COMBINE_SRC="$REPO_DIR/combine/HiggsAnalysis/CombinedLimit"
    if command -v combine >/dev/null 2>&1; then
        echo "[setup] combine already on PATH ($(command -v combine)), skipping build"
    else
        echo "[setup] building CMS Combine into $CONDA_PREFIX (INSTALL_COMBINE=1)"
        # Build toolchain + Eigen3 + Boost (required by CombinedLimit's CMakeLists.txt).
        # libboost-devel = headers + cmake config only; the full 'boost' package
        # drags in python bindings that can conflict with the env's python pin.
        _pkgs=()
        command -v cmake >/dev/null 2>&1 || _pkgs+=(cmake)
        command -v ninja >/dev/null 2>&1 || _pkgs+=(ninja)
        find "$CONDA_PREFIX/share" -iname "Eigen3Config.cmake" 2>/dev/null | grep -q . || _pkgs+=(eigen)
        find "$CONDA_PREFIX/lib" "$CONDA_PREFIX/share" -iname "BoostConfig.cmake" 2>/dev/null | grep -q . || _pkgs+=(libboost-devel)
        if [ ${#_pkgs[@]} -gt 0 ]; then
            echo "[setup] installing Combine build deps: ${_pkgs[*]}"
            conda install -y -n "$ENV_NAME" -c conda-forge "${_pkgs[@]}" || _setup_fail "Combine build deps install failed" || return 1 2>/dev/null || exit 1
        fi

        if [ ! -d "$COMBINE_SRC" ]; then
            mkdir -p "$(dirname "$COMBINE_SRC")"
            git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git "$COMBINE_SRC" \
                || _setup_fail "Combine git clone failed" || return 1 2>/dev/null || exit 1
        fi
        if [ -n "${COMBINE_TAG:-}" ]; then
            git -C "$COMBINE_SRC" fetch --tags && git -C "$COMBINE_SRC" checkout "$COMBINE_TAG" \
                || _setup_fail "cannot check out Combine '$COMBINE_TAG'" || return 1 2>/dev/null || exit 1
        fi

        PY_SITE="lib/python$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages"
        NCPU="$(nproc 2>/dev/null || echo 2)"
        JOBS=$(( NCPU > 2 ? NCPU - 2 : 1 ))
        (
            cd "$COMBINE_SRC" \
            && cmake -S . -B build -G Ninja \
                     -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
                     -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
                     -DCMAKE_INSTALL_PYTHONDIR="$PY_SITE" \
                     -DUSE_VDT=OFF \
            && cmake --build build -j"$JOBS" \
            && cmake --install build
        ) || _setup_fail "Combine build failed (see output above)" || return 1 2>/dev/null || exit 1
        hash -r
        echo "[setup] Combine installed into $CONDA_PREFIX"
    fi

    # diffNuisances.py (pulls from fitDiagnostics.root) lives in test/ and is not
    # installed by the CMake build; copy it next to the other Combine tools.
    if ! command -v diffNuisances.py >/dev/null 2>&1; then
        if [ ! -f "$COMBINE_SRC/test/diffNuisances.py" ]; then
            mkdir -p "$(dirname "$COMBINE_SRC")"
            git clone --depth 1 https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git "$COMBINE_SRC" >/dev/null 2>&1 || true
        fi
        if [ -f "$COMBINE_SRC/test/diffNuisances.py" ]; then
            install -m 755 "$COMBINE_SRC/test/diffNuisances.py" "$CONDA_PREFIX/bin/diffNuisances.py"
            echo "[setup] diffNuisances.py installed to $CONDA_PREFIX/bin/"
        else
            echo "[setup] WARNING: could not fetch diffNuisances.py"
        fi
    fi

    python -c "import HiggsAnalysis.CombinedLimit" >/dev/null 2>&1 \
        && echo "[setup] python: 'import HiggsAnalysis.CombinedLimit' OK" \
        || echo "[setup] WARNING: 'import HiggsAnalysis.CombinedLimit' failed"
fi

echo "[setup] python : $(command -v python)"
echo "[setup] root   : $(root-config --version 2>/dev/null || echo missing)"
echo "[setup] combine: $(command -v combine || echo 'not found (only needed for run_all.sh --limits)')"
if [ "$_setup_sourced" = "0" ]; then
    echo "[setup] run as a script: env verified, but not active in your shell. Use: source setup.sh"
fi
# Don't leave helper variables behind in the caller's shell.
unset _setup_shell _setup_self _setup_sourced _pkgs tool COMBINE_SRC PY_SITE NCPU JOBS
