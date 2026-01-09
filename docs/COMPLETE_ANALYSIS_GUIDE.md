# Complete Analysis Workflow: Signal + Background → Sensitivity → Limits

This guide provides a complete end-to-end workflow for running the bbMET analysis, calculating sensitivity, and extracting limits using CMS Combine.

## Overview

The analysis workflow consists of three main steps:

1. **Process Signal and Backgrounds** → Generate histograms and datacards
2. **Calculate Sensitivity** → Expected significance, S/B ratios
3. **Extract Limits** → Run Combine to get exclusion limits

## Step 1: Process Signal and Backgrounds

### Prerequisites

1. **Signal file**: Delphes ROOT file with signal events
2. **Background files**: Already configured in `process_regions.py`
3. **Cross-sections**: Background cross-sections in `bkg.xsec` file

### Run the Analysis

```bash
python process_regions.py \
    --signal-file /path/to/signal.root \
    --signal-xs 1.0 \
    --signal-ngen 100000 \
    --lumi 139.0 \
    --output-dir output
```

**Parameters:**
- `--signal-file`: Path to your signal ROOT file
- `--signal-xs`: Signal cross-section in pb
- `--signal-ngen`: Number of generated events in signal file
- `--lumi`: Luminosity in fb^-1 (default: 139.0 for Run-2)
- `--output-dir`: Output directory (default: "output")

### What This Produces

For each region (SRs and CRs), you get:
- **Plots**: `output/plots/{region}/`
  - `cutflow.pdf`: Stacked cutflow for all backgrounds
  - `{obs_name}.pdf`: Stacked plots for each observable (MET, mbb, HT, etc.)
- **Datacards**: `output/plots/{region}/datacard.txt`
  - Combine-ready datacard with signal and background yields
  - Includes systematic uncertainties (lumi, btag)

### Example Output Structure

```
output/
├── plots/
│   ├── sr1b/
│   │   ├── cutflow.pdf
│   │   ├── met.pdf
│   │   ├── mbb.pdf
│   │   ├── ht.pdf
│   │   └── datacard.txt
│   ├── sr2b/
│   │   ├── cutflow.pdf
│   │   ├── met.pdf
│   │   └── datacard.txt
│   ├── cr1b_wlnu/
│   │   ├── cutflow.pdf
│   │   ├── recoil.pdf
│   │   ├── mt.pdf
│   │   └── datacard.txt
│   └── ...
```

## Step 2: Calculate Sensitivity

### Expected Significance

The sensitivity can be calculated from the datacard yields:

```python
# From datacard.txt or from histograms:
S = signal_yield
B = total_background_yield

# Expected significance (simple approximation)
Z = S / sqrt(B)  # For large B

# Or using Asimov formula
Z = sqrt(2 * ((S+B) * log(1 + S/B) - S))
```

### Extract Yields from Datacards

You can parse the datacard to get yields:

```bash
# View datacard
cat output/plots/sr1b/datacard.txt

# Extract signal and background rates
grep "rate" output/plots/sr1b/datacard.txt
```

### Sensitivity Script

A simple Python script to calculate sensitivity:

```python
import re

def parse_datacard(datacard_file):
    """Parse datacard and extract yields."""
    with open(datacard_file, 'r') as f:
        content = f.read()

    # Extract rate line
    rate_match = re.search(r'rate\s+(.+)', content)
    if rate_match:
        rates = [float(x) for x in rate_match.group(1).split()]
        signal = rates[0]
        backgrounds = rates[1:]
        return signal, backgrounds
    return None, None

# Calculate sensitivity
signal, backgrounds = parse_datacard('output/plots/sr1b/datacard.txt')
total_bg = sum(backgrounds)
significance = signal / (total_bg ** 0.5) if total_bg > 0 else 0
print(f"Signal: {signal:.2f}, Background: {total_bg:.2f}, Z: {significance:.2f}")
```

## Step 3: Extract Limits with Combine

### Prerequisites

Install CMS Combine:
```bash
# Follow CMS Combine installation instructions
# https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/
```

### Run Combine

For each region, run Combine to get limits:

```bash
# Navigate to region directory
cd output/plots/sr1b

# Run AsymptoticLimits
combine -M AsymptoticLimits datacard.txt -n sr1b

# This produces:
# - higgsCombine_sr1b.AsymptoticLimits.mH120.root
# - Limit plot and text output
```

### Extract Limit Values

```bash
# Get the limit value
combine -M AsymptoticLimits datacard.txt -n sr1b | grep "Expected 50.0%"
```

### Combine Options

**Asymptotic Limits (fast, approximate):**
```bash
combine -M AsymptoticLimits datacard.txt -n sr1b
```

**Hybrid New (more accurate, slower):**
```bash
combine -M HybridNew datacard.txt -n sr1b --LHCmode LHC-limits
```

**Frequentist Limits:**
```bash
combine -M Frequentist datacard.txt -n sr1b
```

### Combine All Regions

Use the provided script to run Combine on all regions:

```bash
python run_combine_limits.py --output-dir output
```

## Complete Example Workflow

### 1. Process Signal and Backgrounds

```bash
# Example: ma=50 signal point
python process_regions.py \
    --signal-file /Users/ptiwari/Development/EventGen/MG5_aMC_v3_6_6/bbdm_2hdma_type1_case1/Events/sig_bbdm_delphes_events.root \
    --signal-xs 1.0 \
    --signal-ngen 100000 \
    --lumi 139.0 \
    --output-dir output_ma50
```

### 2. Check Results

```bash
# View cutflow
open output_ma50/plots/sr1b/cutflow.pdf

# Check datacard
cat output_ma50/plots/sr1b/datacard.txt
```

### 3. Calculate Sensitivity

```bash
# Use the sensitivity script
python calculate_sensitivity.py --output-dir output_ma50
```

### 4. Run Combine for Limits

```bash
# Run on all regions
python run_combine_limits.py --output-dir output_ma50

# Or manually for a specific region
cd output_ma50/plots/sr1b
combine -M AsymptoticLimits datacard.txt -n sr1b
```

## Understanding the Output

### Datacard Format

```
imax 1 number of bins
jmax 6 number of processes minus 1
kmax * number of nuisance parameters

bin            sr1b
observation    -1

bin            sr1b sr1b sr1b sr1b sr1b sr1b sr1b
process        signal diboson sTop_tchannel sTop_tW ttbar wlnjets znnjets
process        0 1 2 3 4 5 6
rate           12.3456 234.5678 45.6789 12.3456 567.8901 123.4567 89.0123

lumi lnN 1.025 1.025 1.025 1.025 1.025 1.025 1.025
btag lnN 1.05 1.05 1.05 1.05 1.05 1.05 1.05
```

### Combine Output

```
Expected 50.0%: r < 2.34
Expected 16.0%: r < 1.23
Expected 84.0%: r < 4.56
Expected 97.5%: r < 6.78
Expected  2.5%: r < 0.89
```

Where `r` is the signal strength (1.0 = nominal signal cross-section).

## Tips and Best Practices

1. **Start with SRs**: Focus on signal regions first (sr1b, sr2b)
2. **Check CRs**: Use control regions to validate background modeling
3. **Systematics**: Add more systematic uncertainties as needed in datacard generation
4. **Multiple Signal Points**: Process multiple signal points and combine results
5. **Scan Analysis**: Use `scan_automation.py` for mass scans

## Troubleshooting

### No events in region
- Check cutflow to see where events are being cut
- Verify region cuts in `cuts_config.yaml`

### Combine errors
- Ensure datacard format is correct
- Check that all processes have non-zero rates
- Verify Combine installation

### Low sensitivity
- Check S/B ratio
- Consider tighter cuts
- Verify signal normalization

## Next Steps

1. **Mass Scans**: Run analysis for multiple (mA, ma) points
2. **Systematics**: Add more systematic uncertainties
3. **Shape Analysis**: Use shape datacards instead of counting
4. **Combined Limits**: Combine multiple regions for better sensitivity
