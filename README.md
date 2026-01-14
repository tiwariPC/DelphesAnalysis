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
**Background files**: `DelphesBackground/{bgname}_delphes_events.root`

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

See `requirements.txt` for Python dependencies.

## Documentation

- `docs/RUN_ANALYSIS.md` - Quick reference for running the analysis
- `docs/AUTO_DISCOVERY_GUIDE.md` - Automatic file discovery guide
- `docs/COMPLETE_ANALYSIS_GUIDE.md` - Full workflow guide
- `docs/CONFIG_FILES_GUIDE.md` - Configuration files structure
- `docs/REGIONS_GUIDE.md` - Region definitions
- `docs/CUTS_CONFIG_GUIDE.md` - Configuration guide
