# ROOT File Structure Analysis

## Step 1: Analyze Your ROOT Files

Before running the analysis, inspect the structure of the Delphes files:

```bash
# A background file
python utils/analyze_root_structure.py genevents/DelphesBackground/ttbar_delphes_events.root

# A signal file
python utils/analyze_root_structure.py genevents/DelphesSignal/sig_bbdm_mA500_ma100_delphes_events.root
```

This shows:
- Available branches
- Branch structure
- How to access Jet and MissingET data
- Any potential issues

## Step 2: Run the Analysis

```bash
python scripts/run_analysis_from_dirs.py \
    --signal-dir genevents/DelphesSignal \
    --background-dir genevents/DelphesBackground \
    --lumi 290.0 --output-dir output
```

For a single explicit file use `scripts/process_regions.py` (see `docs/RUN_ANALYSIS.md`).

## How It Works

1. **Loads only needed branches**: Jet and MissingET (avoids the Particle.fBits error)
2. **Uses the Coffea processor framework**: proper accumulator handling
3. **Creates structured awkward arrays**: events have `events.Jet` and `events.MissingET` fields
4. **Handles errors gracefully**: clear messages if the structure does not match

## Troubleshooting

If you get errors about missing fields:

1. Run `utils/analyze_root_structure.py` on the file to see the actual structure
2. Check which fields are available
3. The code shows exactly what is available if access fails
