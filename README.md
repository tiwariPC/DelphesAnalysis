# bbMET Analysis Pipeline

Complete end-to-end analysis pipeline for bbMET (2HDM+a) signal search using Delphes ROOT files.

## Directory Structure

```
DephesAnalysis/
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
│   ├── signal_cross_sections.yaml  # Signal cross-sections
│   └── cuts_config.yaml   # Physics cuts and region definitions
├── docs/                   # Documentation
│   ├── README.md          # This file
│   └── *.md               # Additional guides
├── utils/                  # Utility scripts
│   └── analyze_root_structure.py
└── tests/                  # Test scripts (if any)
```

## Quick Start

### Option 1: Automatic Discovery from Directories (Recommended)

If you have signal and background files in directories with standard naming:

**Signal files**: `DelphesSignal/sig_bbdm_mA{mA}_ma{ma}_delphes_events.root`
**Background files**: `DelphesBackground/{bgname}_delphes_events.root`

```bash
python scripts/run_analysis_from_dirs.py \
    --signal-dir DelphesSignal \
    --background-dir DelphesBackground \
    --lumi 139.0 \
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
    --lumi 139.0 \
    --output-dir output
```

**Using cross-section directly:**
```bash
python scripts/process_regions.py \
    --signal-file /path/to/signal.root \
    --signal-xs 0.0002870400 \
    --signal-ngen 100000 \
    --lumi 139.0 \
    --output-dir output
```

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
    --lumi 139.0 \
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
    --datacard-dir output/plots \
    --output-file sensitivity.txt
```

### 3. Create Combined Datacard

```bash
python scripts/create_combined_datacard.py \
    --datacard-dir output/plots \
    --output-file combined_datacard.txt
```

### 4. Run Combine Limits

```bash
python scripts/run_combine_limits.py \
    --datacard combined_datacard.txt \
    --output-dir combine_output
```

## Configuration

### Configuration Files

- **`config/samples_config.yaml`**: Background ROOT file paths only
- **`config/background_cross_sections.yaml`**: Background cross-sections (single source of truth)
- **`config/signal_cross_sections.yaml`**: Signal cross-sections for all (mA, ma) points
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
    ├── sr1b/
    │   ├── cutflow.pdf
    │   ├── met.pdf
    │   ├── mbb.pdf
    │   └── datacard.txt
    ├── sr2b/
    │   └── ...
    └── cr1b_wlnu/
        └── ...
```

## Requirements

See `requirements.txt` for Python dependencies.

## Documentation

- `docs/RUN_ANALYSIS.md` - Quick reference for running the analysis
- `docs/AUTO_DISCOVERY_GUIDE.md` - Automatic file discovery guide
- `docs/COMPLETE_ANALYSIS_GUIDE.md` - Full workflow guide
- `docs/CONFIG_FILES_GUIDE.md` - Configuration files structure
- `docs/REGIONS_GUIDE.md` - Region definitions
- `docs/CUTS_CONFIG_GUIDE.md` - Configuration guide
