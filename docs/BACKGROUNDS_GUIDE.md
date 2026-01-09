# Background Processing Guide

## Your Background Files

The following background files are configured:

1. **diboson**: `/Users/ptiwari/Development/EventGen/MG5_aMC_v3_6_6/bkg_diboson/Events/bkg_diboson_delphes_events.root`
2. **sTop_tchannel**: `/Users/ptiwari/Development/EventGen/MG5_aMC_v3_6_6/bkg_sTop_tchannel/Events/bkg_sTop_tchannel_delphes_events.root`
3. **sTop_tW**: `/Users/ptiwari/Development/EventGen/MG5_aMC_v3_6_6/bkg_sTop_tW/Events/bkg_sTop_tW_delphes_events.root`
4. **ttbar**: `/Users/ptiwari/Development/EventGen/MG5_aMC_v3_6_6/bkg_ttbar/Events/bkg_ttbar_delphes_events.root`
5. **wlnjets**: `/Users/ptiwari/Development/EventGen/MG5_aMC_v3_6_6/bkg_wlnjets/Events/bkg_wlnjets_delphes_events.root`
6. **znnjets**: `/Users/ptiwari/Development/EventGen/MG5_aMC_v3_6_6/bkg_znnjets/Events/bkg_znnjets_delphes_events.root`

## Step 1: Update Configuration

Edit `background_config.json` with your actual values:

```json
{
  "diboson": {
    "xs": 50.0,        // Update with actual cross-section in pb
    "ngen": 100000     // Update with actual number of generated events
  },
  "sTop_tchannel": {
    "xs": 200.0,
    "ngen": 100000
  },
  "sTop_tW": {
    "xs": 100.0,
    "ngen": 100000
  },
  "ttbar": {
    "xs": 831.76,      // Standard value at 13 TeV
    "ngen": 1000000
  },
  "wlnjets": {
    "xs": 200.0,
    "ngen": 1000000
  },
  "znnjets": {
    "xs": 200.0,
    "ngen": 1000000
  }
}
```

**Where to find these values:**
- **Cross-section (xs)**: Check your MG5 output or use standard values from literature
- **Number of generated events (ngen)**: Check the MG5 log files or count events in the ROOT file

## Step 2: Process Backgrounds Only

```bash
python process_backgrounds.py
```

This will:
- Process all 6 background samples
- Generate cutflows for each
- Create plots for each background
- Save results to `background_output/`

## Step 3: Process Backgrounds + Signal

```bash
python process_backgrounds.py \
    --signal-file /path/to/your/signal.root \
    --signal-xs 1.0 \
    --signal-ngen 100000 \
    --lumi 139.0
```

This will:
- Process all backgrounds
- Process your signal
- Generate combined plots (signal vs all backgrounds)
- Generate Combine datacard with signal + all backgrounds

## Step 4: View Results

After processing, check:

1. **Cutflows**: Printed to terminal and saved as PDFs
2. **Plots**: Individual background plots in `background_output/`
3. **Combined plots**: Signal vs all backgrounds (if signal provided)
4. **Datacard**: `background_output/datacard.txt` (if signal provided)

## Example: Complete Workflow

```bash
# 1. Process all backgrounds
python process_backgrounds.py --lumi 139.0

# 2. Process signal separately (optional, for individual analysis)
python run_analysis.py signal.root --xs 1.0 --ngen 100000

# 3. Process signal + backgrounds together (for datacard)
python process_backgrounds.py \
    --signal-file signal.root \
    --signal-xs 1.0 \
    --signal-ngen 100000 \
    --lumi 139.0 \
    --output-dir final_results
```

## Output Structure

```
background_output/
├── cutflow_diboson.pdf
├── cutflow_sTop_tchannel.pdf
├── cutflow_sTop_tW.pdf
├── cutflow_ttbar.pdf
├── cutflow_wlnjets.pdf
├── cutflow_znnjets.pdf
├── met_diboson.pdf
├── met_sTop_tchannel.pdf
├── ... (more plots)
├── met_signal_vs_all_bg.pdf      # If signal provided
├── mbb_signal_vs_all_bg.pdf       # If signal provided
└── datacard.txt                    # If signal provided
```

## Troubleshooting

### File not found errors
- Check that all background files exist at the specified paths
- The script will skip missing files and continue with others

### Wrong cross-sections
- Update `background_config.json` with correct values
- Or use `--config-file` to specify a different config file

### Need to process only some backgrounds
- Edit `process_backgrounds.py` and comment out unwanted backgrounds in `BACKGROUND_FILES` dictionary

## Next Steps

After processing backgrounds:
1. Review cutflows to understand background contributions
2. Generate datacard with your signal
3. Use the datacard with Combine for limit setting
4. Create scan plots over parameter space
