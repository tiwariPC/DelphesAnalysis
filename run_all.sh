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
#   --lam L              t-channel coupling DMS3D 3 3         (default 1.0)
#   --npoints N          number of default signal points, <= 8 (default 3)
#   --points LIST        explicit signal points A:a,A:a,...   (overrides --npoints)
#   --mode grid          full mass grid instead of --npoints
#   --nevents N          events per run                        (default 10000)
#   --sig-nruns N        runs per signal point, merged         (default 1)
#   --bkg-nruns N        runs per background, merged           (default 1)
#   --delphes-card C     Delphes card, 'CMS' or path           (default CMS)
#   --signal-dir D       Delphes signal ROOT dir               (default genevents/DelphesSignal)
#   --background-dir D   Delphes background ROOT dir           (default genevents/DelphesBackground)
#   --output-dir D       analysis output                       (default output)
#   --lumi L             luminosity in fb^-1                   (default 290.0)
#   --skip-gen           skip event generation
#   --limits             also run Combine limits (needs 'combine')
#   --signal-xs-unit     set all signal cross-sections to 1 pb in the analysis
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

NEVENTS=10000; MODE=points; NPOINTS=3; POINTS=""; SIGNAL_MODEL=tchannel; DM_TYPE=s3m; LAM=1.0
SIG_NRUNS=1; BKG_NRUNS=1; DELPHES_CARD=CMS
SIGNAL_DIR="$REPO_DIR/genevents/DelphesSignal"; BKG_DIR="$REPO_DIR/genevents/DelphesBackground"
OUTPUT_DIR=output; LUMI=290.0; SKIP_GEN=0; LIMITS=0; XS_UNIT=0

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
        --delphes-card) DELPHES_CARD="$2"; shift 2;;
        --signal-dir) SIGNAL_DIR="$2"; shift 2;;
        --background-dir) BKG_DIR="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        --lumi) LUMI="$2"; shift 2;;
        --skip-gen) SKIP_GEN=1; shift;;
        --limits) LIMITS=1; shift;;
        --signal-xs-unit) XS_UNIT=1; shift;;
        -h|--help) sed -n 2,28p "$0"; exit 0;;
        *) echo "Unknown option: $1" >&2; exit 1;;
    esac
done

# Cross-section table the analysis reads for the chosen signal model.
if [ "$SIGNAL_MODEL" = "tchannel" ]; then
    SIGNAL_XSEC="config/tchannel_signal_cross_sections.yaml"
else
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
    step "2/6 generate signal ($SIGNAL_MODEL: ${POINT_ARGS[*]})"
    python genevents/mg5_Generate_Run_bbMET.py --proc signal "${POINT_ARGS[@]}" \
        --signal-model "$SIGNAL_MODEL" --dm-type "$DM_TYPE" --lam "$LAM" \
        --delphes-card "$DELPHES_CARD" --nevents "$NEVENTS" --nruns "$SIG_NRUNS" \
        --signal-outdir "$SIGNAL_DIR"
    step "2/6 generate backgrounds ($BKG_NRUNS run(s) each)"
    python genevents/mg5_Generate_Run_bbMET.py --proc bkg --delphes-card "$DELPHES_CARD" \
        --nevents "$NEVENTS" --nruns "$BKG_NRUNS" --background-outdir "$BKG_DIR"
else
    step "2/6 generation skipped"
fi

step "3/6 analysis (regions, plots, datacards)"
ANALYSIS_ARGS=(--signal-dir "$SIGNAL_DIR" --background-dir "$BKG_DIR" --lumi "$LUMI"
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
    python scripts/run_combine_limits.py --output-dir "$OUTPUT_DIR" --method AsymptoticLimits
else
    step "6/6 limits skipped (use --limits)"
fi

echo; echo "Pipeline finished. Results in $OUTPUT_DIR/"
