# Stacked Plots Guide

## Overview

The `process_backgrounds.py` script now automatically:
1. **Reads cross-sections** from `/Users/ptiwari/Development/EventGen/MG5_aMC_v3_6_6/bkg.xsec`
2. **Counts events** automatically from ROOT files
3. **Generates stacked plots** for all observables (MET, m(bb), pT(bb), HT, Δφ(bb), Δφ(MET, bb))

## Quick Start

### Process Backgrounds Only (with stacked plots)

```bash
python process_backgrounds.py --lumi 139.0
```

This will:
- Read cross-sections from `bkg.xsec`
- Count events in each ROOT file
- Process all 6 backgrounds
- Generate stacked plots for all observables
- Save to `background_output/`

### Process Backgrounds + Signal (with stacked plots)

```bash
python process_backgrounds.py \
    --signal-file /path/to/signal.root \
    --signal-xs 1.0 \
    --signal-ngen 100000 \
    --lumi 139.0
```

This will generate:
- Backgrounds-only stacked plots
- Signal + backgrounds stacked plots
- Combine datacard

## Generated Plots

For each observable, you'll get:

1. **Backgrounds-only stacked plot**: `{observable}_backgrounds_stacked.pdf`
   - Shows all backgrounds stacked together
   - Useful for understanding background composition

2. **Signal vs Backgrounds stacked plot**: `{observable}_signal_vs_all_bg_stacked.pdf`
   - Shows signal overlaid on stacked backgrounds
   - Only generated if signal file is provided

### Available Observables

- `met_backgrounds_stacked.pdf` / `met_signal_vs_all_bg_stacked.pdf`
- `mbb_backgrounds_stacked.pdf` / `mbb_signal_vs_all_bg_stacked.pdf`
- `ptbb_backgrounds_stacked.pdf` / `ptbb_signal_vs_all_bg_stacked.pdf`
- `ht_backgrounds_stacked.pdf` / `ht_signal_vs_all_bg_stacked.pdf`
- `dphi_bb_backgrounds_stacked.pdf` / `dphi_bb_signal_vs_all_bg_stacked.pdf`
- `dphi_met_bb_backgrounds_stacked.pdf` / `dphi_met_bb_signal_vs_all_bg_stacked.pdf`

## Plot Features

- **Stacked backgrounds**: Each background is a different color, stacked on top of each other
- **Signal overlay**: Red line overlaid on top of stacked backgrounds
- **Log scale**: Y-axis uses log scale by default (better for wide dynamic range)
- **Color scheme**:
  - ttbar: Red
  - wlnjets: Teal
  - znnjets: Blue
  - diboson: Light salmon
  - sTop_tchannel: Mint
  - sTop_tW: Yellow

## Automatic Configuration

The script automatically:
1. Parses `bkg.xsec` file for cross-sections
2. Counts events in each ROOT file using uproot
3. Matches background names between xsec file and ROOT files

### Cross-Section File Format

The script expects `bkg.xsec` in this format:
```
# Cross-section table
background       cross_section (pb)
diboson       104.494
sTop_tchannel       200.993
...
```

### Name Matching

The script matches background names flexibly:
- `diboson` matches `bkg_diboson`
- `ttbar` matches `bkg_ttbar`
- Case-insensitive matching

## Example Output Structure

```
background_output/
├── cutflow_diboson.pdf
├── cutflow_ttbar.pdf
├── ... (individual background cutflows)
├── met_backgrounds_stacked.pdf          # Backgrounds only
├── met_signal_vs_all_bg_stacked.pdf     # Signal + backgrounds
├── mbb_backgrounds_stacked.pdf
├── mbb_signal_vs_all_bg_stacked.pdf
├── ... (all observables)
└── datacard.txt                         # If signal provided
```

## Troubleshooting

### Cross-sections not found
- Check that `bkg.xsec` exists at the expected location
- Verify the format matches the expected structure
- The script will use default values with a warning

### Event count fails
- Check that ROOT files are accessible
- Verify the TTree name is "Delphes" (default)
- The script will use default ngen=100000 with a warning

### Missing plots
- Check that backgrounds processed successfully
- Verify observables exist in processor output
- Look for warning messages in the output

## Advanced Usage

### Custom cross-section file

```bash
python process_backgrounds.py --xsec-file /path/to/custom.xsec
```

### Skip automatic event counting

Edit `process_backgrounds.py` and set `auto_count_events=False` in `get_background_xs_and_ngen()` call, then provide ngen values in a config file.

### Custom output directory

```bash
python process_backgrounds.py --output-dir my_results
```

## Publication Quality

All plots are generated with:
- 300 DPI resolution
- Publication-quality styling
- Proper legends and labels
- Grid for readability
- Tight layout for papers
