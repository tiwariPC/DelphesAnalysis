#!/usr/bin/env bash
# Run the whole pipeline: setup -> generation -> analysis -> sensitivity -> significance [-> limits].
#
#   ./run_all.sh                          # 3 signal points + all backgrounds, defaults
#   ./run_all.sh --nevents 1000           # quick test
#   ./run_all.sh --npoints 6 --bkg-nruns 3
#   ./run_all.sh --points 500:100,1000:300   # explicit signal points (mMED:mDM or mA:ma)
#   ./run_all.sh --skip-gen               # reuse existing Delphes ROOT files
#   ./run_all.sh --mode grid --limits     # full grid + Combine limits
#
# Options:
#   --signal-model M     tchannel | 2hdma (type II)           (default tchannel)
#   --dm-type T          t-channel only: s3m | s3d            (default s3m)
#   --lam L              t-channel coupling DMS3D 3 3         (default 2.2)
#   --npoints N          number of default signal points, <= 8 (default 3)
#   --points LIST        explicit signal points A:a,A:a,...   (overrides --npoints)
#   --mode grid          full mass grid instead of --npoints
#   --nevents N          events per run                        (default 10000)
#   --sig-nruns N        runs per signal point, merged         (default 1)
#   --bkg-nruns N        runs per background, merged           (default 1)
#   --jobs N             samples generated at the same time; CPU cores are
#                        split between them                    (default 4)
#   --cores N            total CPU cores for generation        (default: all)
#   --delphes-card C     Delphes card, 'CMS' or path           (default CMS)
#   --data-dir D         large files: MadGraph work dirs and logs, Delphes ROOT files
#                        (default genevents)
#   --work-dir D         MadGraph process dirs, cards and logs (default <data-dir>/work)
#   --signal-dir D       Delphes signal ROOT dir               (default <data-dir>/DelphesSignal)
#   --background-dir D   Delphes background ROOT dir           (default <data-dir>/DelphesBackground)
#   --output-dir D       analysis output                       (default output)
#   --lumi L             luminosity in fb^-1                   (default 290.0)
#   --skip-gen           skip event generation
#   --force              regenerate samples whose Delphes file already exists
#   --slim-delphes       write only the Delphes branches the analysis reads (smaller files)
#   --bkg-cuts           generator-level background cuts (genevents/genScripts/<bkg>.cuts.madevent);
#                        backgrounds go to <data-dir>/DelphesBackground_bkgcuts
#   --limits             also run Combine limits (needs 'combine')
#   --signal-xs-unit     set all signal cross-sections to 1 pb in the analysis
#   --no-tmux            run in the current shell instead of a detached tmux session
#   --attach             attach to the tmux session right after starting it
#
# Unless already inside tmux (or --no-tmux is given), the pipeline restarts itself in a
# detached tmux session so a dropped connection does not kill it. Output goes to
# logs/run_all_<timestamp>.log; reattach with 'tmux attach -t <session>'.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

USE_TMUX=1; ATTACH=0; PASS_ARGS=()
for a in "$@"; do
    case "$a" in
        --no-tmux) USE_TMUX=0;;
        --attach) ATTACH=1;;
        -h|--help) sed -n 2,44p "$0"; exit 0;;
        *) PASS_ARGS+=("$a");;
    esac
done
set -- ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}

if [ "$USE_TMUX" -eq 1 ] && [ -z "${TMUX:-}" ]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "tmux not found; running in the current shell (a dropped connection will stop it)." >&2
    else
        stamp="$(date +%Y%m%d_%H%M%S)"
        session="bbmet_${stamp}"
        mkdir -p "$REPO_DIR/logs"
        log="$REPO_DIR/logs/run_all_${stamp}.log"
        inner="cd $(printf %q "$REPO_DIR") && bash ./run_all.sh --no-tmux $(printf '%q ' "$@")"
        tmux new-session -d -s "$session" \
            "set -o pipefail; $inner 2>&1 | tee $(printf %q "$log"); echo \"[run_all exited with status \$?]\" | tee -a $(printf %q "$log")"
        # Keep the window after the pipeline ends so its last output stays visible.
        tmux set-option -t "$session" remain-on-exit on >/dev/null
        echo "Pipeline started in tmux session: $session"
        echo "  log      : $log"
        echo "  follow   : tail -f $log"
        echo "  attach   : tmux attach -t $session   (detach: Ctrl-b d)"
        echo "  close    : tmux kill-session -t $session   (after it finishes)"
        if [ "$ATTACH" -eq 1 ] && [ -t 1 ]; then exec tmux attach -t "$session"; fi
        exit 0
    fi
fi

NEVENTS=10000; MODE=points; NPOINTS=3; POINTS=""; SIGNAL_MODEL=tchannel; DM_TYPE=s3m; LAM=2.2
SIG_NRUNS=1; BKG_NRUNS=1; DELPHES_CARD=CMS; GEN_JOBS=4; GEN_CORES=0
SIGNAL_DIR=""; BKG_DIR=""; WORK_DIR=""
DATA_DIR="$REPO_DIR/genevents"
OUTPUT_DIR=output; LUMI=290.0; SKIP_GEN=0; LIMITS=0; XS_UNIT=0; FORCE=0; SLIM=0; BKG_CUTS=0

while [ $# -gt 0 ]; do
    case "$1" in
        --nevents) NEVENTS="$2"; shift 2;;
        --mode) MODE="$2"; shift 2;;
        --npoints) NPOINTS="$2"; shift 2;;
        --points) POINTS="$2"; shift 2;;
        --signal-model) SIGNAL_MODEL="$2"; shift 2;;
        --dm-type) DM_TYPE="$2"; shift 2;;
        --lam) LAM="$2"; shift 2;;
        --sig-nruns) SIG_NRUNS="$2"; shift 2;;
        --bkg-nruns) BKG_NRUNS="$2"; shift 2;;
        --jobs) GEN_JOBS="$2"; shift 2;;
        --cores) GEN_CORES="$2"; shift 2;;
        --delphes-card) DELPHES_CARD="$2"; shift 2;;
        --data-dir) DATA_DIR="$2"; shift 2;;
        --work-dir) WORK_DIR="$2"; shift 2;;
        --signal-dir) SIGNAL_DIR="$2"; shift 2;;
        --background-dir) BKG_DIR="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        --lumi) LUMI="$2"; shift 2;;
        --skip-gen) SKIP_GEN=1; shift;;
        --force) FORCE=1; shift;;
        --slim-delphes) SLIM=1; shift;;
        --bkg-cuts) BKG_CUTS=1; shift;;
        --limits) LIMITS=1; shift;;
        --signal-xs-unit) XS_UNIT=1; shift;;
        *) echo "Unknown option: $1" >&2; exit 1;;
    esac
done
DATA_DIR="${DATA_DIR/#\~/$HOME}"
SIGNAL_DIR="${SIGNAL_DIR:-$DATA_DIR/DelphesSignal}"
# Backgrounds with and without generator cuts are kept apart (files and cross-sections).
if [ "$BKG_CUTS" -eq 1 ]; then
    BKG_DIR="${BKG_DIR:-$DATA_DIR/DelphesBackground_bkgcuts}"
    BKG_XSEC="config/background_cuts_cross_sections.yaml"
else
    BKG_DIR="${BKG_DIR:-$DATA_DIR/DelphesBackground}"
    BKG_XSEC="config/background_cross_sections.yaml"
fi
WORK_DIR="${WORK_DIR:-$DATA_DIR/work}"
mkdir -p "$DATA_DIR"
echo "Data dir: $DATA_DIR"

# Cross-section table and signal file tag for the chosen signal model. t-channel files and
# cross-sections carry the DM type and coupling (e.g. s3m_lam2p2), so samples generated
# with other settings are never picked up.
if [ "$SIGNAL_MODEL" = "tchannel" ]; then
    # Same formatting as the generator: str(float(L)) with '.' -> 'p' (2.2 -> 2p2, 2 -> 2p0)
    SIGNAL_TAG="${DM_TYPE}_lam$(python3 -c 'import sys; print(str(float(sys.argv[1])).replace(".", "p"))' "$LAM")"
    SIGNAL_XSEC="config/tchannel_${SIGNAL_TAG}_cross_sections.yaml"
else
    SIGNAL_TAG=""
    SIGNAL_XSEC="config/signal_cross_sections.yaml"
fi

# Which signal points to generate: explicit list, full grid, or the first N defaults.
if [ -n "$POINTS" ]; then
    POINT_ARGS=(--points "$POINTS")
elif [ "$MODE" = "grid" ]; then
    POINT_ARGS=(--mode grid)
else
    POINT_ARGS=(--npoints "$NPOINTS")
fi

step() { echo; echo "=================== $* ==================="; }

step "1/6 setup environment"
# shellcheck disable=SC1091
# conda's (de)activate hooks (e.g. deactivate-root.sh) read unset variables, so
# nounset must be off while they run.
set +u
source "$REPO_DIR/setup.sh"
set -u

if [ "$SKIP_GEN" -eq 0 ]; then
    step "2/6 generate signal ($SIGNAL_MODEL: ${POINT_ARGS[*]}) and backgrounds, $GEN_JOBS at a time"
    python genevents/mg5_Generate_Run_bbMET.py --proc all "${POINT_ARGS[@]}" \
        --signal-model "$SIGNAL_MODEL" --dm-type "$DM_TYPE" --lam "$LAM" \
        --delphes-card "$DELPHES_CARD" --nevents "$NEVENTS" \
        --nruns "$SIG_NRUNS" --bkg-nruns "$BKG_NRUNS" --jobs "$GEN_JOBS" --cores "$GEN_CORES" \
        --data-dir "$DATA_DIR" --work-dir "$WORK_DIR" \
        --signal-outdir "$SIGNAL_DIR" --background-outdir "$BKG_DIR" \
        $( [ "$FORCE" -eq 1 ] && echo --force ) $( [ "$SLIM" -eq 1 ] && echo --slim-delphes ) \
        $( [ "$BKG_CUTS" -eq 1 ] && echo --bkg-cuts )
else
    step "2/6 generation skipped"
fi

step "3/6 analysis (regions, plots, datacards)"
ANALYSIS_ARGS=(--signal-dir "$SIGNAL_DIR" --signal-tag "$SIGNAL_TAG" --background-dir "$BKG_DIR"
               --background-xsec "$BKG_XSEC" --lumi "$LUMI"
               --output-dir "$OUTPUT_DIR" --signal-xsec "$SIGNAL_XSEC")
if [ "$XS_UNIT" -eq 1 ]; then ANALYSIS_ARGS+=(--signal-xs-unit); fi
python scripts/run_analysis_from_dirs.py "${ANALYSIS_ARGS[@]}"

step "4/6 sensitivity"
python scripts/calculate_sensitivity.py --output-dir "$OUTPUT_DIR" --output-file "$OUTPUT_DIR/sensitivity.txt"

step "5/6 significance plot"
python scripts/plot_significance.py --output-dir "$OUTPUT_DIR" --lumi "$LUMI"

if [ "$LIMITS" -eq 1 ]; then
    step "6/6 Combine limits"
    command -v combine >/dev/null 2>&1 || { echo "ERROR: 'combine' not in PATH" >&2; exit 1; }
    python scripts/run_combine_limits.py --output-dir "$OUTPUT_DIR" --signal-xsec "$SIGNAL_XSEC"
else
    step "6/6 limits skipped (use --limits)"
fi

echo; echo "Pipeline finished. Results in $OUTPUT_DIR/"
