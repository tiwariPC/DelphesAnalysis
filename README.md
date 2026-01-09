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
│   ├── calculate_sensitivity.py
│   ├── create_combined_datacard.py
│   ├── plot_significance.py
│   └── run_combine_limits.py
├── config/                 # Configuration files
│   └── cuts_config.yaml   # Physics cuts and region definitions
├── docs/                   # Documentation
│   ├── README.md          # This file
│   └── *.md               # Additional guides
├── utils/                  # Utility scripts
│   └── analyze_root_structure.py
└── tests/                  # Test scripts (if any)
```

## Quick Start

### 1. Process Signal and Backgrounds for All Regions

```bash
python scripts/process_regions.py \
    --signal-file /path/to/signal.root \
    --signal-xs 1.0 \
    --signal-ngen 100000 \
    --lumi 139.0 \
    --output-dir output
```

This will:
- Load each ROOT file once (optimized)
- Process all regions (SRs and CRs)
- Generate plots and datacards for each region
- Save output to `output/plots/{region}/`

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

Edit `config/cuts_config.yaml` to modify:
- Physics object cuts (jet pT, b-tag threshold, etc.)
- Event selection cuts
- Region definitions (SRs and CRs)

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

- `docs/COMPLETE_ANALYSIS_GUIDE.md` - Full workflow guide
- `docs/REGIONS_GUIDE.md` - Region definitions
- `docs/CUTS_CONFIG_GUIDE.md` - Configuration guide


