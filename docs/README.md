# bbMET Analysis Pipeline

Complete end-to-end bbMET analysis pipeline for 2HDM+a model, starting from Delphes ROOT files and producing Combine-ready datacards.

## Philosophy

- **Phenomenology first** (Delphes simulation)
- Keep analysis **CMS-like** for easy transition
- Use **Coffea-compatible logic** from day one
- Produce publication-ready outputs:
  - Cutflow tables
  - Shape histograms
  - Scan plots (mA, ma)
  - Combine datacards

## Installation

```bash
pip install -r requirements.txt
```

**Verify installation:**
```bash
python test_setup.py
```

This will check that all dependencies are installed correctly.

## How to Run

### Method 1: Command-Line Script (Easiest)

Use the provided `run_analysis.py` script:

```bash
python run_analysis.py signal.root --xs 1.0 --ngen 100000 --name signal_mA1500_ma300
```

**Options:**
- `signal.root`: Path to your Delphes ROOT file
- `--xs`: Cross-section in pb (required)
- `--ngen`: Number of generated events (required)
- `--lumi`: Luminosity in fb^-1 (default: 139.0 for Run-2)
- `--name`: Sample name (default: "signal")
- `--output-dir`: Output directory (default: "analysis_output")
- `--no-plots`: Skip generating plots

**Example:**
```bash
# Run with custom luminosity
python run_analysis.py signal_mA1500_ma300.root \
    --xs 1.5 \
    --ngen 100000 \
    --lumi 300.0 \
    --name signal_mA1500_ma300 \
    --output-dir results

# Run without plots (faster)
python run_analysis.py signal.root --xs 1.0 --ngen 100000 --no-plots
```

### Method 2: Python API (More Control)

```python
from bbdmDelphes import BBMETProcessor, load_delphes_events, print_cutflow

# Load events
events = load_delphes_events("signal.root")

# Create processor
processor = BBMETProcessor(
    lumi_fb=139.0,
    xs_pb=1.0,
    ngen=100000,
    sample_name="signal_mA1500_ma300"
)

# Process
output = processor.process(events)
output = processor.postprocess(output)

# View results
print_cutflow(output["cutflow"], "Signal")
print(f"Signal yield: {output['met'].sum():.4f} events")
```

### Method 3: Process All Backgrounds

Process all your background samples and generate combined datacard:

```bash
# Process backgrounds only
python process_backgrounds.py

# Process backgrounds + signal and generate datacard
python process_backgrounds.py \
    --signal-file /path/to/signal.root \
    --signal-xs 1.0 \
    --signal-ngen 100000

# Use custom config file with xs and ngen values
python process_backgrounds.py --config-file background_config.json
```

**Background files are automatically loaded from:**
- `bkg_diboson_delphes_events.root`
- `bkg_sTop_tchannel_delphes_events.root`
- `bkg_sTop_tW_delphes_events.root`
- `bkg_ttbar_delphes_events.root`
- `bkg_wlnjets_delphes_events.root`
- `bkg_znnjets_delphes_events.root`

**Important:** Update `background_config.json` with your actual cross-sections and number of generated events!

### Method 4: Complete Workflow (Signal + Backgrounds)

Edit `example_workflow.py` with your file paths and run:

```bash
python example_workflow.py
```

Or use it as a module:

```python
from example_workflow import run_complete_analysis
run_complete_analysis()
```

## Structure

```
DephesAnalysis/
├── bbdmDelphes.py          # Main processor and analysis logic
├── scan_automation.py       # Parameter scan automation
├── plotting_utils.py        # Publication-quality plotting
├── example_workflow.py      # Complete workflow example
├── run_analysis.py          # Command-line script (easiest way to run)
├── process_backgrounds.py   # Process all backgrounds with stacked plots
├── cuts_config.yaml         # YAML configuration for selection cuts
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Configuration

### Selection Cuts (YAML)

The analysis uses YAML configuration files for easy cut modification. Edit `cuts_config.yaml` to change selection cuts:

```yaml
jets:
  pt_min: 30.0          # Minimum jet pT in GeV
  eta_max: 2.4          # Maximum |eta| for jets

event_selection:
  njets_min: 2          # Minimum number of jets
  nbjets_min: 2         # Minimum number of b-jets
  met_min: 200.0        # Minimum MET in GeV
  dphi_min: 0.4         # Minimum Δφ(MET, jet)
  apply_dphi: true      # Whether to apply Δφ cut
```

**Use custom cuts:**
```bash
python run_analysis.py signal.root --xs 1.0 --ngen 100000 \
    --cuts-config my_cuts.yaml
```

See `CUTS_CONFIG_GUIDE.md` for detailed documentation.

## Quick Start Guide

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run Your First Analysis

The simplest way is using the command-line script:

```bash
python run_analysis.py your_signal_file.root --xs 1.0 --ngen 100000
```

Replace:
- `your_signal_file.root` with your Delphes ROOT file path
- `1.0` with your signal cross-section in pb
- `100000` with the number of generated events

### Step 3: Check Results

Results will be saved in `analysis_output/`:
- `cutflow.pdf`: Cutflow visualization
- `met.pdf`: MET distribution
- `mbb.pdf`: m(bb) distribution
- `ht.pdf`: HT distribution

The cutflow will also be printed to the terminal.

### Common Use Cases

**Run-2 analysis (139 fb^-1):**
```bash
python run_analysis.py signal.root --xs 1.0 --ngen 100000 --lumi 139.0
```

**Run-3 analysis (300 fb^-1):**
```bash
python run_analysis.py signal.root --xs 1.0 --ngen 100000 --lumi 300.0
```

**Multiple signal points (scan):**
```bash
for ma in 100 200 300 400 500; do
    python run_analysis.py signal_mA1500_ma${ma}.root \
        --xs 1.0 --ngen 100000 \
        --name signal_mA1500_ma${ma} \
        --output-dir scan_results/ma${ma}
done
```

## Detailed Usage

### 1. Process a Single Sample (Python API)

```python
from bbdmDelphes import BBMETProcessor, load_delphes_events

# Load events from Delphes ROOT file
events = load_delphes_events("signal.root")

# Create processor
processor = BBMETProcessor(
    lumi_fb=139.0,      # Run-2 luminosity
    xs_pb=1.0,          # Cross-section in pb
    ngen=100000,        # Number of generated events
    sample_name="signal_mA1500_ma300"
)

# Process
output = processor.process(events)
output = processor.postprocess(output)

# Print cutflow
from bbdmDelphes import print_cutflow
print_cutflow(output["cutflow"], "Signal")
```

### 2. Run Complete Analysis

```python
from example_workflow import run_complete_analysis
run_complete_analysis()
```

### 3. Generate Datacard

```python
from bbdmDelphes import generate_datacard, save_datacard

datacard = generate_datacard(
    signal_name="signal",
    signal_rate=3.2,
    backgrounds={"ttbar": 45.1, "zbb": 12.7},
    uncertainties={
        "lumi": 1.025,  # 2.5% luminosity uncertainty
        "btag": 1.05,    # 5% b-tagging uncertainty
    }
)

save_datacard(datacard, "datacard.txt")
```

## Physics Object Definitions

### Jets
- Anti-kT R=0.4 (Delphes default)
- pT > 30 GeV
- |η| < 2.4

### b-jets
- Delphes `Jet.BTag > 0`

### MET
- `MissingET.MET`

## Event Selection

| Step         | Cut                              |
| ------------ | -------------------------------- |
| Preselection | ≥ 2 jets                         |
| b-tag        | ≥ 2 b-jets                       |
| MET          | MET > 200 GeV                    |
| Δφ           | Δφ(MET, jet) > 0.4               |
| Mass window  | optional (for resonance studies) |

## Observables

The pipeline automatically calculates and stores:

- **MET**: Missing transverse energy
- **m(bb)**: Invariant mass of di-b-jet system
- **pT(bb)**: Transverse momentum of di-b-jet system
- **Δφ(bb)**: Azimuthal angle between b-jets
- **Δφ(MET, bb)**: Azimuthal angle between MET and di-b-jet system
- **HT**: Scalar sum of jet pT

## Normalization

Event weights are calculated as:

```python
weight = (xs_pb * lumi_fb * 1000) / ngen
```

Where:
- `xs_pb`: Cross-section in pb
- `lumi_fb`: Integrated luminosity in fb⁻¹
- `ngen`: Number of generated events

Default luminosities:
- Run-2: 139 fb⁻¹
- Run-3: 300 fb⁻¹ (projected)

## Parameter Scans

### Scan Rules

- Fix **mA = 1500 GeV** when scanning ma
- Fix **ma = 300 GeV** when scanning mA

### Example Scan

```python
from scan_automation import scan_ma_fixed_mA, plot_scan_results_ma

# Process multiple signal points
processors_dict = {}
for ma in [100, 200, 300, 400, 500, 600]:
    # Process signal_mA1500_ma{ma}.root
    output = process_signal_sample(...)
    processors_dict[ma] = output

# Run scan
results = scan_ma_fixed_mA(processors_dict, ma_values, mA_fixed=1500.0)

# Plot results
plot_scan_results_ma(results, "scan_ma.pdf")
```

## Background Splitting

**Mandatory**: Use separate ROOT files for each background:
- tt̄
- Z+bb
- W+bb
- single-top

Each background gets:
- Its own histogram
- Its own normalization
- Its own entry in the datacard

## Datacard Format

The pipeline generates Combine-compatible datacards:

```
imax 1 number of bins
jmax 2 number of processes minus 1
kmax * number of nuisance parameters

bin            sr
observation    -1

bin            sr sr sr
process        signal ttbar zbb
process        0 1 2
rate           3.2 45.1 12.7

lumi lnN       1.025 1.025 1.025
btag lnN       1.05 1.05 1.05
```

## Plotting

Publication-quality plots are available:

```python
from plotting_utils import (
    plot_cutflow,
    plot_histogram,
    plot_signal_vs_background
)

# Cutflow
plot_cutflow(output["cutflow"], "Signal", "cutflow.pdf")

# Histogram
plot_histogram(output["met"], output_file="met.pdf")

# Signal vs Background
plot_signal_vs_background(
    signal_output["met"],
    {"ttbar": ttbar_output["met"], "zbb": zbb_output["met"]},
    output_file="met_signal_vs_bg.pdf"
)
```

## Transition to CMS

| Delphes      | CMS      |
| ------------ | -------- |
| uproot       | NanoAOD  |
| Delphes btag | DeepJet  |
| MET          | PFMET    |
| Coffea       | Coffea   |
| Datacard     | Datacard |

Only inputs change — logic remains the same.

## Paper Structure (JHEP-Ready)

1. Introduction
2. 2HDM+a model
3. Event generation & detector simulation
4. Event selection
5. Signal vs background discrimination
6. Scan results
7. Comparison with CMS limits
8. Conclusions

## Next Steps

- **HTCondor submission setup**: See `scan_automation.py` for template
- **Scan automation scripts**: Ready to use
- **Publication-quality plotting**: Available in `plotting_utils.py`
- **CMS Run-3 cut tuning**: Adjust cuts in `bbdmDelphes.py`

## Citation

If you use this pipeline, please cite:
- Coffea: [arXiv:2002.12991](https://arxiv.org/abs/2002.12991)
- Delphes: [arXiv:1307.6346](https://arxiv.org/abs/1307.6346)

## License

This analysis pipeline is provided as-is for research purposes.
