#!/usr/bin/env bash
# Run the whole pipeline: setup -> generation -> analysis -> sensitivity -> significance [-> limits].
#
#   ./run_all.sh                          # everything, defaults
#   ./run_all.sh --nevents 1000           # quick test
#   ./run_all.sh --skip-gen               # reuse existing Delphes ROOT files
#   ./run_all.sh --mode grid --limits     # full grids + Combine limits
#
# Options:
#   --nevents N          events per generated sample          (default 10000)
#   --mode single|grid   generation points                    (default single)
#   --signal-model M     tchannel | 2hdma                     (default tchannel)
#   --hdma-type 1|2      2hdma Yukawa type                    (default 1)
#   --delphes-card C     Delphes card, 'CMS' or path          (default CMS)
#   --signal-dir D       Delphes signal ROOT dir              (default genevents/DelphesSignal)
#   --background-dir D   Delphes background ROOT dir          (default genevents/DelphesBackground)
#   --output-dir D       analysis output                      (default output)
#   --lumi L             luminosity in fb^-1                  (default 290.0)
#   --skip-gen           skip event generation
#   --limits             also run Combine limits (needs 'combine')
#   --signal-xs-unit     set all signal cross-sections to 1 pb in the analysis
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

NEVENTS=10000; MODE=single; SIGNAL_MODEL=tchannel; HDMA_TYPE=1; DELPHES_CARD=CMS
SIGNAL_DIR="$REPO_DIR/genevents/DelphesSignal"; BKG_DIR="$REPO_DIR/genevents/DelphesBackground"
OUTPUT_DIR=output; LUMI=290.0; SKIP_GEN=0; LIMITS=0; XS_UNIT=0

while [ $# -gt 0 ]; do
    case "$1" in
        --nevents) NEVENTS="$2"; shift 2;;
        --mode) MODE="$2"; shift 2;;
        --signal-model) SIGNAL_MODEL="$2"; shift 2;;
        --hdma-type) HDMA_TYPE="$2"; shift 2;;
        --delphes-card) DELPHES_CARD="$2"; shift 2;;
        --signal-dir) SIGNAL_DIR="$2"; shift 2;;
        --background-dir) BKG_DIR="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        --lumi) LUMI="$2"; shift 2;;
        --skip-gen) SKIP_GEN=1; shift;;
        --limits) LIMITS=1; shift;;
        --signal-xs-unit) XS_UNIT=1; shift;;
        -h|--help) sed -n 2,22p "$0"; exit 0;;
        *) echo "Unknown option: $1" >&2; exit 1;;
    esac
done

step() { echo; echo "=================== $* ==================="; }

step "1/6 setup environment"
# shellcheck disable=SC1091
# conda's (de)activate hooks (e.g. deactivate-root.sh) read unset variables, so
# nounset must be off while they run.
set +u
source "$REPO_DIR/setup.sh"
set -u

if [ "$SKIP_GEN" -eq 0 ]; then
    step "2/6 generate signal ($SIGNAL_MODEL, $MODE)"
    python genevents/mg5_Generate_Run_bbMET.py --proc signal --mode "$MODE" \
        --signal-model "$SIGNAL_MODEL" --hdma-type "$HDMA_TYPE" --delphes-card "$DELPHES_CARD" \
        --nevents "$NEVENTS" --signal-outdir "$SIGNAL_DIR"
    step "2/6 generate backgrounds"
    python genevents/mg5_Generate_Run_bbMET.py --proc bkg --delphes-card "$DELPHES_CARD" \
        --nevents "$NEVENTS" --background-outdir "$BKG_DIR"
else
    step "2/6 generation skipped"
fi

step "3/6 analysis (regions, plots, datacards)"
ANALYSIS_ARGS=(--signal-dir "$SIGNAL_DIR" --background-dir "$BKG_DIR" --lumi "$LUMI" --output-dir "$OUTPUT_DIR")
[ "$XS_UNIT" -eq 1 ] && ANALYSIS_ARGS+=(--signal-xs-unit)
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
