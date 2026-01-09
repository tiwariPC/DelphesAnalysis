# Signal and Control Regions Guide

This guide explains how to use the signal and control regions functionality in the bbMET analysis pipeline.

## Overview

The analysis now supports multiple signal regions (SR) and control regions (CR) as defined in the `regions.yaml` configuration file. Each region has specific selection criteria, and the pipeline processes all samples (signal and backgrounds) for each region separately.

## Region Definitions

Regions are defined in `/Users/ptiwari/Development/BbdmRun3/DarkBottomLine/configs/regions.yaml`. The file includes:

### Signal Regions (SR)
- **1b:SR**: Signal region with 1 b-tag, ≤2 jets, no leptons
- **2b:SR**: Signal region with 2 b-tags, exactly 3 jets, no leptons

### Control Regions (CR)
- **1b:CR_Wlnu**: W+jets control region in 1b category (electron or muon)
- **2b:CR_Top**: Top control region in 2b category (electron or muon)
- **1b:CR_Zll**: Z+jets control region in 1b category (electron or muon)
- **2b:CR_Zll**: Z+jets control region in 2b category (electron or muon)

## Usage

### Processing All Regions

Use the `process_regions.py` script to process signal and backgrounds for all regions:

```bash
python process_regions.py \
    --signal-file /path/to/signal.root \
    --signal-xs 1.0 \
    --signal-ngen 100000 \
    --xsec-file /path/to/bkg.xsec \
    --lumi 139.0 \
    --output-dir output \
    --regions-file /path/to/regions.yaml
```

### Output Structure

The script generates:
- **Region-specific plots**: One plot per region showing signal vs stacked backgrounds
- **CMS-style formatting**: All plots follow CMS Analysis Guidelines
- **Cutflow information**: Event counts for each region

Output files are named as: `met_{region_name}.pdf` (e.g., `met_1b_SR.pdf`)

## Region Selection Logic

The `regions.py` module provides:
- `load_regions()`: Loads region definitions from YAML
- `select_region()`: Applies region-specific cuts to events
- `get_region_type()`: Returns "SR" or "CR"
- `get_region_category()`: Returns "1b" or "2b"

## Additional Observables

For control regions, the following observables are calculated:
- **MT**: Transverse mass (for W+jets and top CRs)
- **Recoil**: Recoil = |MET + lepton| (for W+jets and top CRs)
- **Mll**: Dilepton invariant mass (for Z+jets CRs)

## CMS Plotting Guidelines

All plots follow CMS Analysis Guidelines:
- Font sizes: 42pt for main text, 36pt for labels
- CMS label in top-left corner
- Luminosity label in top-right corner
- Region name displayed on plot
- Proper axis labels with units
- Stacked backgrounds with distinct colors
- Signal overlaid as line

## Example Workflow

1. **Define regions** in `regions.yaml`
2. **Run processing**:
   ```bash
   python process_regions.py --signal-file signal.root --signal-xs 1.0 --signal-ngen 100000
   ```
3. **Review plots** in the output directory
4. **Generate datacards** (if needed) for statistical analysis

## Region Configuration Format

Each region in `regions.yaml` has:
- **description**: Human-readable description
- **cuts**: Dictionary of cut conditions
  - `Nbjets`: Number of b-jets (e.g., "==1", "==2")
  - `Njets`: Number of jets (e.g., "<=2", "==3")
  - `Nleptons`: Total number of leptons
  - `Nelectrons`, `Nmuons`: Specific lepton counts
  - `MET`: Missing transverse energy threshold (e.g., ">50")
  - `DeltaPhi`: Δφ(MET, jet) cut
  - `Recoil`: Recoil threshold (for CRs)
  - `MT`: Transverse mass threshold (for CRs)
  - `MllMin`, `MllMax`: Dilepton mass window (for Z CRs)

## Notes

- Regions are processed independently
- Each region generates separate histograms and plots
- Control regions help validate background modeling
- Signal regions are used for final limits
