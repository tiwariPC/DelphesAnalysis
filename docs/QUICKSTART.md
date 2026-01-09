# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Running Your First Analysis

### Option 1: Command-Line (Recommended for Beginners)

```bash
python run_analysis.py signal.root --xs 1.0 --ngen 100000
```

**What you need:**
- `signal.root`: Your Delphes ROOT file
- `--xs`: Cross-section in pb (from your MC generation)
- `--ngen`: Number of generated events (from your MC generation)

**Example:**
```bash
python run_analysis.py signal_mA1500_ma300.root \
    --xs 1.5 \
    --ngen 100000 \
    --name signal_mA1500_ma300 \
    --lumi 139.0
```

### Option 2: Python Script

Create a file `my_analysis.py`:

```python
from bbdmDelphes import BBMETProcessor, load_delphes_events, print_cutflow

# Load your Delphes ROOT file
events = load_delphes_events("signal.root")

# Create processor with your parameters
processor = BBMETProcessor(
    lumi_fb=139.0,        # Run-2: 139, Run-3: 300
    xs_pb=1.0,            # Your cross-section
    ngen=100000,          # Number of generated events
    sample_name="signal"
)

# Process events
output = processor.process(events)
output = processor.postprocess(output)

# View cutflow
print_cutflow(output["cutflow"], "Signal")

# Get signal yield
signal_yield = output["met"].sum()
print(f"\nSignal yield: {signal_yield:.4f} events")
```

Run it:
```bash
python my_analysis.py
```

## What You'll Get

After running, you'll have:

1. **Cutflow printed to terminal** - Shows events at each selection step
2. **Plots in `analysis_output/`**:
   - `cutflow.pdf` - Visual cutflow
   - `met.pdf` - MET distribution
   - `mbb.pdf` - m(bb) distribution
   - `ht.pdf` - HT distribution

## Processing Multiple Samples

### Signal + Backgrounds

```python
from bbdmDelphes import BBMETProcessor, load_delphes_events
from plotting_utils import plot_signal_vs_background

# Process signal
events_sig = load_delphes_events("signal.root")
proc_sig = BBMETProcessor(lumi_fb=139.0, xs_pb=1.0, ngen=100000, sample_name="signal")
output_sig = proc_sig.process(events_sig)

# Process backgrounds
backgrounds = {
    "ttbar": {"file": "ttbar.root", "xs": 831.76, "ngen": 1000000},
    "zbb": {"file": "zbb.root", "xs": 50.0, "ngen": 500000},
}

bg_outputs = {}
for name, config in backgrounds.items():
    events = load_delphes_events(config["file"])
    proc = BBMETProcessor(lumi_fb=139.0, xs_pb=config["xs"],
                         ngen=config["ngen"], sample_name=name)
    bg_outputs[name] = proc.process(events)

# Plot signal vs background
bg_hists = {name: output["met"] for name, output in bg_outputs.items()}
plot_signal_vs_background(output_sig["met"], bg_hists,
                         output_file="met_signal_vs_bg.pdf")
```

## Generating Datacards

```python
from bbdmDelphes import generate_datacard, save_datacard

# Calculate rates from histograms
signal_rate = output_sig["met"].sum()
bg_rates = {name: output["met"].sum() for name, output in bg_outputs.items()}

# Generate datacard
datacard = generate_datacard(
    signal_name="signal",
    signal_rate=signal_rate,
    backgrounds=bg_rates,
    uncertainties={
        "lumi": 1.025,  # 2.5% luminosity uncertainty
        "btag": 1.05,    # 5% b-tagging uncertainty
    }
)

# Save
save_datacard(datacard, "datacard.txt")
```

## Troubleshooting

### "File not found" error
- Check that your ROOT file path is correct
- Make sure the file exists: `ls -lh your_file.root`

### "Tree 'Delphes' not found"
- Your ROOT file might use a different tree name
- Check with: `python -c "import uproot; print(list(uproot.open('your_file.root').keys()))"`
- Modify `load_delphes_events("file.root", tree_name="YourTreeName")`

### Import errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (requires Python 3.8+)

### Empty histograms
- Check that your events pass the selection cuts
- Look at the cutflow to see where events are being cut
- Verify your Delphes file has the expected branches (Jet, MissingET, etc.)

## Next Steps

- See `README.md` for full documentation
- Check `example_workflow.py` for complete examples
- Use `scan_automation.py` for parameter scans
