# ROOT File Structure Analysis

## Step 1: Analyze Your ROOT Files

Before running the analysis, first analyze the structure of your ROOT files:

```bash
# Analyze a background file
python analyze_root_structure.py /Users/ptiwari/Development/EventGen/MG5_aMC_v3_6_6/bkg_diboson/Events/bkg_diboson_delphes_events.root

# Analyze the signal file
python analyze_root_structure.py /Users/ptiwari/Development/EventGen/MG5_aMC_v3_6_6/bbdm_2hdma_type1_case1/Events/sig_bbdm_delphes_events.root
```

This will show you:
- Available branches
- Branch structure
- How to access Jet and MissingET data
- Any potential issues

## Step 2: Run the Analysis

After understanding the structure, run the analysis:

```bash
python process_backgrounds.py \
    --signal-file /Users/ptiwari/Development/EventGen/MG5_aMC_v3_6_6/bbdm_2hdma_type1_case1/Events/sig_bbdm_delphes_events.root \
    --signal-xs 1.0 \
    --signal-ngen 100000 \
    --lumi 139.0
```

## How It Works

The code now:

1. **Loads only needed branches**: Jet and MissingET (avoids Particle.fBits error)
2. **Uses Coffea processor framework**: Proper accumulator handling
3. **Creates structured awkward arrays**: Events have `events.Jet` and `events.MissingET` fields
4. **Handles errors gracefully**: Clear error messages if structure doesn't match

## Troubleshooting

If you get errors about missing fields:

1. Run the analysis script first to see the actual structure
2. Check the output to see what fields are available
3. The code will show you exactly what's available if access fails
