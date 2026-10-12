# bbMET Analysis Pipeline

End-to-end pipeline for a bbMET dark-matter search: event generation (MadGraph5 + Pythia8 +
Delphes) for a t-channel DM or 2HDM+a signal and the SM backgrounds, then region-based analysis,
plots, datacards, sensitivity and (optionally) Combine limits.

## Run Everything With One Command

```bash
./run_all.sh --nevents 1000          # quick test: 3 signal points + all backgrounds
./run_all.sh                         # same with 10000 events per run
```

`run_all.sh` does, in order: environment setup, signal and background generation, the analysis,
sensitivity, the significance plot and (with `--limits`) Combine limits. It stops at the first
failure. Main options:

| Option | Meaning | Default |
|---|---|---|
| `--signal-model` | `tchannel` or `2hdma` (type II) | `tchannel` |
| `--dm-type`, `--lam` | t-channel only: `s3m`/`s3d` and coupling, fixed per call | `s3m`, `1.0` |
| `--npoints N` | first N built-in signal points (up to 8) | `3` |
| `--points A:a,A:a` | explicit points (`mMED:mDM` or `mA:ma`), overrides `--npoints` | |
| `--mode grid` | full mass grid instead of `--npoints` | |
| `--nevents N` | events per run | `10000` |
| `--sig-nruns`, `--bkg-nruns` | runs per sample (different seeds), merged with `hadd` | `1` |
| `--signal-dir`, `--background-dir` | where the Delphes ROOT files are written and read | `genevents/DelphesSignal`, `genevents/DelphesBackground` |
| `--skip-gen` | reuse existing ROOT files | |
| `--limits` | also run Combine limits (needs `combine`) | |

`./run_all.sh --help` lists all options. Generation details are in `genevents/README.md`.

### Environment

`source setup.sh` creates the conda env `delphesanalysis` from `environment.yml` if it does not
exist, and activates it. `run_all.sh` calls it for you. The environment pins python 3.11,
gfortran/gxx 13 and ROOT 6.32: MadGraph 3.5.6 generates Fortran that fails at run time with the
gfortran >= 14 runtime.

## Directory Structure

```
DephesAnalysis/
├── run_all.sh              # One-command pipeline (setup, generation, analysis, sensitivity, plots)
├── setup.sh                # Create/activate the conda env
├── environment.yml         # Conda environment (python, compilers, ROOT, analysis libraries)
├── genevents/              # Self-contained event generation (see genevents/README.md)
│   ├── mg5_Generate_Run_bbMET.py   # MadGraph + Pythia8 + Delphes driver
│   ├── genScripts/                 # Editable process (.mg5) and settings (.madevent) cards
│   ├── ufo_model/                  # t-channel DM and 2HDM+a type II UFO models
│   └── condorJob/                  # Legacy Condor workflow
├── src/                    # Core analysis modules
│   ├── bbdmDelphes.py     # Physics object definitions, event selection, observables
│   ├── regions.py         # Region definitions and selection logic
│   └── plotting_utils.py  # Plotting functions with CMS styling
├── scripts/                # Executable analysis scripts
│   ├── process_regions.py  # Main analysis script (process signal + backgrounds for all regions)
│   ├── run_analysis_from_dirs.py  # Auto-discovery script for signal/background directories
│   ├── calculate_sensitivity.py
│   ├── create_combined_datacard.py
│   ├── plot_significance.py
│   └── run_combine_limits.py
├── config/                 # Configuration files
│   ├── samples_config.yaml  # Background file paths only
│   ├── background_cross_sections.yaml  # Background cross-sections (single source of truth)
│   ├── signal_cross_sections.yaml  # 2HDM+a signal cross-sections (mA, ma)
│   ├── tchannel_signal_cross_sections.yaml  # t-channel signal cross-sections (mMED as mA, mDM as ma)
│   └── cuts_config.yaml   # Physics cuts and region definitions
├── docs/                   # Documentation
│   ├── README.md          # This file
│   └── *.md               # Additional guides
├── utils/                  # Utility scripts
│   └── analyze_root_structure.py
└── tests/                  # Test scripts (if any)
```

## Quick Start

### Full Analysis Workflow (One Command)

Run the complete analysis from start to finish:

```bash
# Step 1: Process all regions (creates plots, datacards, and combined datacard)
python scripts/run_analysis_from_dirs.py \
    --signal-dir DelphesSignal \
    --background-dir DelphesBackground \
    --lumi 290.0 \
    --output-dir output && \
# Step 2: Calculate sensitivity metrics
python scripts/calculate_sensitivity.py \
    --output-dir output \
    --output-file sensitivity.txt
```

**Or as a single line:**
```bash
python scripts/run_analysis_from_dirs.py --signal-dir DelphesSignal --background-dir DelphesBackground --lumi 290.0 --output-dir output && python scripts/calculate_sensitivity.py --output-dir output --output-file sensitivity.txt
```

This will:
1. ✅ Discover all signal and background files
2. ✅ Process all regions and generate plots
3. ✅ Create individual region datacards
4. ✅ **Automatically create combined datacard** (`output/plots/combined_datacard.txt`)
5. ✅ Calculate sensitivity metrics (S/B, significance) for all regions

### Option 1: Automatic Discovery from Directories (Recommended)

If you have signal and background files in directories with standard naming:

**Signal files**: `DelphesSignal/sig_bbdm_mA{mA}_ma{ma}_delphes_events.root`
(for the t-channel model, `mA` is the mediator mass and `ma` the DM mass)
**Background files**: `DelphesBackground/{bgname}_delphes_events.root`

Event generation writes exactly these names. Pass `--signal-xsec config/tchannel_signal_cross_sections.yaml`
to the analysis for t-channel signals (`run_all.sh` does this for you).

```bash
python scripts/run_analysis_from_dirs.py \
    --signal-dir DelphesSignal \
    --background-dir DelphesBackground \
    --lumi 290.0 \
    --output-dir output
```

This automatically:
- Discovers all signal files and extracts (mA, ma) from filenames
- Discovers all background files
- Counts events in signal files (or use `--signal-ngen` to specify)
- Updates `config/samples_config.yaml` with background paths
- Runs analysis with all signals overlaid on the same plots

### Option 2: Manual File Specification

#### Single Signal Point

**Using cross-section from config (recommended):**
```bash
python scripts/process_regions.py \
    --signal-file /path/to/signal.root \
    --signal-mA 300 \
    --signal-ma 50 \
    --signal-ngen 100000 \
    --lumi 290.0 \
    --output-dir output
```

**Using cross-section directly:**
```bash
python scripts/process_regions.py \
    --signal-file /path/to/signal.root \
    --signal-xs 0.0002870400 \
    --signal-ngen 100000 \
    --lumi 290.0 \
    --output-dir output
```

**With signal scaling for plots (e.g., scale by 10x for better visibility):**
```bash
python scripts/process_regions.py \
    --signal-file /path/to/signal.root \
    --signal-mA 300 \
    --signal-ma 50 \
    --signal-ngen 100000 \
    --lumi 290.0 \
    --signal-scale 10.0 \
    --output-dir output
```

**Note**: `--signal-scale` only affects the plots for visualization. The original histograms and datacards remain unchanged.

**Note**: When using `--signal-mA` and `--signal-ma`, cross-sections are automatically looked up from `config/signal_cross_sections.yaml`.

#### Multiple Signal Points (Overlaid)

```bash
python scripts/process_regions.py \
    --signal-file /path/to/signal_mA300_ma50.root \
    --signal-mA 300 \
    --signal-ma 50 \
    --signal-ngen 100000 \
    --signal-file /path/to/signal_mA500_ma100.root \
    --signal-mA 500 \
    --signal-ma 100 \
    --signal-ngen 100000 \
    --lumi 290.0 \
    --output-dir output
```

**What the analysis does:**
- Loads each ROOT file once (optimized)
- Processes all regions (SRs and CRs)
- Generates plots and datacards for each region
- Overlays multiple signals on the same plots (if multiple provided)
- Saves output to `output/plots/{region}/`

### 2. Calculate Sensitivity

```bash
python scripts/calculate_sensitivity.py \
    --output-dir output \
    --output-file sensitivity.txt
```

**Options:**
- `--output-dir`: Directory containing output from `process_regions.py` (default: "output")
- `--output-file`: Save results to file (optional, prints to stdout if not specified)
- `--region`: Calculate for specific region only (optional)
- `--format`: Output format - `table`, `csv`, or `json` (default: "table")

**Examples:**
```bash
# Calculate for all regions, save to file
python scripts/calculate_sensitivity.py --output-dir output --output-file sensitivity.txt

# Calculate for specific region
python scripts/calculate_sensitivity.py --output-dir output --region sr1b

# Output as CSV
python scripts/calculate_sensitivity.py --output-dir output --format csv --output-file sensitivity.csv
```

### 3. Combined Datacards (Automatic)

**Separate combined datacards are automatically created for each signal grid point** when running `process_regions.py`:
```
output/plots/combined_datacard_mA{mA}_ma{ma}.txt
output/plots/combined_shapes_mA{mA}_ma{ma}.root
```

For example, if you process signals with (mA=300, ma=50) and (mA=500, ma=100), you'll get:
- `output/plots/combined_datacard_mA300_ma50.txt`
- `output/plots/combined_datacard_mA500_ma100.txt`
- `output/plots/combined_shapes_mA300_ma50.root`
- `output/plots/combined_shapes_mA500_ma100.root`

Each combined datacard combines all regions (SRs and CRs) for that specific signal point.

**Manual creation (if needed):**
```bash
python scripts/create_combined_datacard.py \
    --output-dir output \
    --output-file combined_datacard.txt
```

### 4. Run Combine Limits (Optional)

```bash
python scripts/run_combine_limits.py \
    --output-dir output \
    --method AsymptoticLimits
```

**Note**: You need to specify which signal point's combined datacard to use, e.g.:
```bash
python scripts/run_combine_limits.py \
    --output-dir output \
    --datacard output/plots/combined_datacard_mA300_ma50.txt \
    --method AsymptoticLimits
```

## Configuration

### Configuration Files

- **`config/samples_config.yaml`**: Background ROOT file paths only
- **`config/background_cross_sections.yaml`**: Background cross-sections (single source of truth)
- **`config/signal_cross_sections.yaml`**: 2HDM+a signal cross-sections for all (mA, ma) points
- **`config/tchannel_signal_cross_sections.yaml`**: t-channel signal cross-sections, same layout (mMED, mDM)

The generator updates the three cross-section files from the MadGraph runs (mean over runs).
- **`config/cuts_config.yaml`**: Physics cuts and region definitions

### Editing Configuration

**Modify physics cuts and regions:**
```bash
# Edit config/cuts_config.yaml
# - Physics object cuts (jet pT, b-tag threshold, etc.)
# - Event selection cuts
# - Region definitions (SRs and CRs)
```

**Update background file paths:**
```bash
# Edit config/samples_config.yaml
# Only file paths - cross-sections are in background_cross_sections.yaml
```

**Update cross-sections:**
```bash
# Edit config/background_cross_sections.yaml (for backgrounds)
# Edit config/signal_cross_sections.yaml (for signals)
```

See `docs/CONFIG_FILES_GUIDE.md` for detailed configuration documentation.

## Output Structure

```
output/
└── plots/
    ├── combined_datacard_mA{mA}_ma{ma}.txt    # Combined datacard per signal point
    ├── combined_shapes_mA{mA}_ma{ma}.root      # Combined shapes file per signal point
    ├── sr1b/
    │   ├── cutflow.pdf
    │   ├── met.pdf
    │   ├── mbb.pdf
    │   ├── datacard_mA{mA}_ma{ma}.txt          # Datacard per signal point
    │   └── shapes_mA{mA}_ma{ma}.root           # Shapes file per signal point
    ├── sr2b/
    │   └── ...
    └── cr1b_wlnu/
        └── ...
```

## Requirements

See `environment.yml` for dependencies (`source setup.sh` creates the conda env).

## Documentation

- `docs/RUN_ANALYSIS.md` - Quick reference for running the analysis
- `docs/AUTO_DISCOVERY_GUIDE.md` - Automatic file discovery guide
- `docs/COMPLETE_ANALYSIS_GUIDE.md` - Full workflow guide
- `docs/CONFIG_FILES_GUIDE.md` - Configuration files structure
- `docs/REGIONS_GUIDE.md` - Region definitions
- `docs/CUTS_CONFIG_GUIDE.md` - Configuration guide
