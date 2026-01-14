# Running the Analysis

This guide provides quick reference commands for running the bbMET analysis.

## Quick Start: Automatic Discovery from Directories

If you have signal and background files in separate directories with standard naming conventions, you can use the automatic discovery script:

**Signal files format**: `DelphesSignal/sig_bbdm_mA{mA}_ma{ma}_delphes_events.root`
**Background files format**: `DelphesBackground/{bgname}_delphes_events.root`

```bash
python scripts/run_analysis_from_dirs.py \
    --signal-dir DelphesSignal \
    --background-dir DelphesBackground \
    --lumi 290.0 \
    --output-dir output
```

This script will:
- Automatically discover all signal files and extract (mA, ma) values from filenames
- Automatically discover all background files
- Count events in signal files (or use `--signal-ngen` to specify)
- Update `config/samples_config.yaml` with background file paths
- Run the analysis with all signals overlaid

**Options:**
- `--signal-ngen N`: Use N for all signals (otherwise counts from files)
- `--dry-run`: Print the command without running
- `--no-auto-ngen`: Don't automatically count events

## Quick Start: Manual File Specification

### 1. Process Signal and Backgrounds

The main analysis script processes all regions (signal regions and control regions) and generates plots and datacards.

#### Single Signal Point:

**Using signal cross-section from config (recommended):**

```bash
cd /Users/ptiwari/Development/BbdmRun3/DelphesAnalysis

python scripts/process_regions.py \
    --signal-file /path/to/signal.root \
    --signal-mA 300 \
    --signal-ma 50 \
    --signal-ngen 100000 \
    --lumi 290.0 \
    --output-dir output
```

**Using signal cross-section directly:**

```bash
python scripts/process_regions.py \
    --signal-file /path/to/signal.root \
    --signal-xs 0.0002870400 \
    --signal-ngen 100000 \
    --lumi 290.0 \
    --output-dir output
```

#### Multiple Signal Points (Overlaid):

To overlay multiple signal points on the same plots, specify `--signal-file` multiple times:

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
    --signal-file /path/to/signal_mA700_ma200.root \
    --signal-mA 700 \
    --signal-ma 200 \
    --signal-ngen 100000 \
    --lumi 290.0 \
    --output-dir output
```

**Note:** All signals will be overlaid on the same plots in the same output directory. Each signal will have a different color and line style in the legend.

### 2. Example: Processing Multiple Signal Points (Overlaid)

To overlay multiple signal points on the same plots, build the command with multiple `--signal-file` arguments:

```bash
#!/bin/bash

# Signal points to overlay
SIGNAL_POINTS=(
    "300 50"
    "300 100"
    "500 50"
    "500 100"
    "700 50"
)

# Build command with all signal files
CMD="python scripts/process_regions.py"

for point in "${SIGNAL_POINTS[@]}"; do
    read mA ma <<< "$point"
    CMD="$CMD --signal-file /path/to/signal_mA${mA}_ma${ma}.root"
    CMD="$CMD --signal-mA $mA"
    CMD="$CMD --signal-ma $ma"
    CMD="$CMD --signal-ngen 100000"
done

CMD="$CMD --lumi 139.0 --output-dir output"

echo "Running: $CMD"
eval $CMD
```

This will create plots with all signal points overlaid on the same background stack.

## Configuration Files

### Cross-Section Files (Single Source of Truth)

- **Background cross-sections**: `config/background_cross_sections.yaml`
  - Contains cross-sections for all background processes (dyjets, ttbar, diboson, etc.)
  - This is the **only** source for background cross-sections

- **Signal cross-sections**: `config/signal_cross_sections.yaml`
  - Contains cross-sections for all signal points (mA, ma combinations)

### Sample Configuration

- **Background samples**: `config/samples_config.yaml`
  - Contains **only file paths** for background ROOT files
  - Cross-sections are **not** stored here - use `background_cross_sections.yaml` instead

### Cuts Configuration

- **Region definitions**: `config/cuts_config.yaml`
  - Defines all signal regions (SR) and control regions (CR)
  - Contains cut values for each region

## Command-Line Options

### Required Arguments

- `--signal-file`: Path to signal ROOT file
- `--signal-ngen`: Number of generated events in signal file

### Optional Arguments

- `--signal-file`: Signal ROOT file (can be specified multiple times for overlaying)
- `--signal-xs`: Signal cross-section in pb (one per --signal-file, or use --signal-mA/--signal-ma)
- `--signal-mA`: Signal mA value in GeV (for cross-section lookup, one per --signal-file)
- `--signal-ma`: Signal ma value in GeV (for cross-section lookup, one per --signal-file)
- `--signal-ngen`: Signal number of generated events (one per --signal-file, required)
- `--signal-label`: Signal label for legend (one per --signal-file, default: "Signal (mA, ma)")
- `--lumi`: Luminosity in fb^-1 (default: 290.0 for Run3)
- `--output-dir`: Output directory (default: "output", same for all signals when overlaying)
- `--samples-config`: Background samples config (default: "config/samples_config.yaml")
- `--background-xsec`: Background cross-sections YAML (default: "config/background_cross_sections.yaml")
- `--signal-xsec`: Signal cross-sections YAML (default: "config/signal_cross_sections.yaml")
- `--cuts-config`: Cuts configuration file (default: "config/cuts_config.yaml")

## Output Structure

After running the analysis, you'll get:

```
output/
├── plots/
│   ├── sr1b/              # Signal region, 1 b-tag
│   │   ├── cutflow.pdf
│   │   ├── met.pdf
│   │   ├── mbb.pdf
│   │   ├── ht.pdf
│   │   ├── datacard.txt
│   │   └── shapes.root
│   ├── sr2b/              # Signal region, 2 b-tags
│   │   ├── cutflow.pdf
│   │   ├── cost_star.pdf
│   │   ├── met.pdf
│   │   ├── mbb.pdf
│   │   ├── datacard.txt
│   │   └── shapes.root
│   ├── cr1b_wlnu/         # Control region, 1b, W+jets
│   │   ├── cutflow.pdf
│   │   ├── recoil.pdf
│   │   ├── mt.pdf
│   │   ├── datacard.txt
│   │   └── shapes.root
│   └── ...
```

## Troubleshooting

### Cross-section not found

If you get an error about signal cross-section not being found:

1. Check that `config/signal_cross_sections.yaml` exists
2. Verify that the (mA, ma) combination exists in the file
3. Or use `--signal-xs` to provide the cross-section directly

### Background cross-sections

Background cross-sections are automatically loaded from:
1. `config/background_cross_sections.yaml` (takes precedence)
2. `config/samples_config.yaml` (fallback)
3. Legacy `--xsec-file` if provided (overrides both)

## Next Steps

After running the analysis:

1. **Check plots**: Review the PDF files in `output/plots/`
2. **Extract limits**: Use `scripts/run_combine_limits.py` to run Combine
3. **Calculate sensitivity**: Use `scripts/calculate_sensitivity.py` for significance calculations

See `docs/COMPLETE_ANALYSIS_GUIDE.md` for more details.
