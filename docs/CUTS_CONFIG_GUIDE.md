# Cuts Configuration Guide

## Overview

The analysis pipeline now supports YAML-based cut configuration, making it easy to modify selection cuts without changing code. You can:
- **Add/remove cuts** by editing a YAML file
- **Use different cuts** for different analyses
- **Override cuts** programmatically when needed

## Quick Start

### 1. Edit the Configuration File

Edit `cuts_config.yaml`:

```yaml
# Selection Cuts Configuration for bbMET Analysis

# Physics Object Definitions
jets:
  pt_min: 30.0          # Minimum jet pT in GeV
  eta_max: 2.4          # Maximum |eta| for jets

bjets:
  btag_threshold: 0     # B-tag threshold (BTag > threshold)

# Event Selection Cuts
event_selection:
  njets_min: 2          # Minimum number of jets
  nbjets_min: 2         # Minimum number of b-jets
  met_min: 200.0        # Minimum MET in GeV
  dphi_min: 0.4         # Minimum Δφ(MET, jet)
  apply_dphi: true      # Whether to apply Δφ cut
```

### 2. Run Analysis

The scripts automatically use `cuts_config.yaml`:

```bash
# Uses cuts_config.yaml automatically
python run_analysis.py signal.root --xs 1.0 --ngen 100000

# Or specify a different config file
python run_analysis.py signal.root --xs 1.0 --ngen 100000 --cuts-config my_cuts.yaml
```

## Configuration Options

### Jet Selection

```yaml
jets:
  pt_min: 30.0      # Minimum jet pT (GeV) - default: 30.0
  eta_max: 2.4      # Maximum |eta| - default: 2.4
```

**Examples:**
- Tighter jet selection: `pt_min: 40.0`
- Wider acceptance: `eta_max: 4.5`

### B-Jet Selection

```yaml
bjets:
  btag_threshold: 0    # B-tag threshold - default: 0
```

**Examples:**
- Require higher b-tag score: `btag_threshold: 0.5`
- Any b-tagged jet: `btag_threshold: 0`

### Event Selection

```yaml
event_selection:
  njets_min: 2         # Minimum number of jets - default: 2
  nbjets_min: 2       # Minimum number of b-jets - default: 2
  met_min: 200.0      # Minimum MET (GeV) - default: 200.0
  dphi_min: 0.4       # Minimum Δφ(MET, jet) - default: 0.4
  apply_dphi: true    # Apply Δφ cut - default: true
```

**Examples:**

**Tighter selection:**
```yaml
event_selection:
  njets_min: 3
  nbjets_min: 3
  met_min: 250.0
  dphi_min: 0.5
  apply_dphi: true
```

**Looser selection:**
```yaml
event_selection:
  njets_min: 2
  nbjets_min: 1
  met_min: 150.0
  dphi_min: 0.3
  apply_dphi: true
```

**Disable Δφ cut:**
```yaml
event_selection:
  apply_dphi: false
```

## Usage Examples

### Example 1: Default Cuts

```bash
# Uses cuts_config.yaml (or defaults if file doesn't exist)
python run_analysis.py signal.root --xs 1.0 --ngen 100000
```

### Example 2: Custom Cuts File

```bash
# Create my_tight_cuts.yaml with tighter selection
python run_analysis.py signal.root --xs 1.0 --ngen 100000 \
    --cuts-config my_tight_cuts.yaml
```

### Example 3: Programmatic Override

```python
from bbdmDelphes import BBMETProcessor, load_delphes_events

events = load_delphes_events("signal.root")

# Override specific cuts programmatically
processor = BBMETProcessor(
    lumi_fb=139.0,
    xs_pb=1.0,
    ngen=100000,
    sample_name="signal",
    cuts_config="cuts_config.yaml",  # Base config
    met_min=250.0,                    # Override MET cut
    nbjets_min=3                      # Override b-jet requirement
)

output = processor.process(events)
```

### Example 4: Multiple Configurations

Create different config files for different analyses:

**`cuts_loose.yaml`:**
```yaml
jets:
  pt_min: 25.0
  eta_max: 2.4
event_selection:
  njets_min: 2
  nbjets_min: 1
  met_min: 150.0
  dphi_min: 0.3
  apply_dphi: true
```

**`cuts_tight.yaml`:**
```yaml
jets:
  pt_min: 40.0
  eta_max: 2.4
event_selection:
  njets_min: 3
  nbjets_min: 3
  met_min: 250.0
  dphi_min: 0.5
  apply_dphi: true
```

Then use them:
```bash
# Loose selection
python run_analysis.py signal.root --xs 1.0 --ngen 100000 \
    --cuts-config cuts_loose.yaml

# Tight selection
python run_analysis.py signal.root --xs 1.0 --ngen 100000 \
    --cuts-config cuts_tight.yaml
```

## Adding New Cuts

To add new cuts (e.g., mass window), you need to:

1. **Add to YAML:**
```yaml
event_selection:
  # ... existing cuts ...
  mass_window:
    mbb_min: 100.0
    mbb_max: 150.0
    apply_mass_window: false
```

2. **Update `load_cuts_from_yaml()` in `bbdmDelphes.py`:**
```python
cuts = {
    # ... existing cuts ...
    "mbb_min": config.get("event_selection", {}).get("mass_window", {}).get("mbb_min", 0.0),
    "mbb_max": config.get("event_selection", {}).get("mass_window", {}).get("mbb_max", 1000.0),
    "apply_mass_window": config.get("event_selection", {}).get("mass_window", {}).get("apply_mass_window", False),
}
```

3. **Use in `apply_event_selection()` or `process()`:**
```python
if self.cuts["apply_mass_window"]:
    mask = mask & (mbb > self.cuts["mbb_min"]) & (mbb < self.cuts["mbb_max"])
```

## Cutflow Labels

The cutflow automatically uses dynamic labels based on your cuts:

- `">=2 jets"` → `">=3 jets"` if `njets_min: 3`
- `">=2 bjets"` → `">=3 bjets"` if `nbjets_min: 3`
- `"MET>200"` → `"MET>250"` if `met_min: 250.0`

## Best Practices

1. **Version control your config files**: Keep different versions for different analyses
2. **Document changes**: Add comments in YAML explaining why cuts were changed
3. **Test systematically**: Compare results with different cut configurations
4. **Use descriptive names**: `cuts_run3_tight.yaml` vs `cuts_run2_loose.yaml`

## Troubleshooting

### Config file not found
- Script will use default cuts and print a warning
- Check the file path is correct
- Use absolute path if needed: `--cuts-config /full/path/to/cuts.yaml`

### Invalid YAML syntax
- Check indentation (YAML is sensitive to spaces)
- Use a YAML validator: `python -c "import yaml; yaml.safe_load(open('cuts_config.yaml'))"`

### Cuts not applying
- Check that the YAML structure matches the expected format
- Verify cut names match exactly (case-sensitive)
- Check processor output for cut values being used

## Example Configurations

### Run-2 Standard
```yaml
jets:
  pt_min: 30.0
  eta_max: 2.4
event_selection:
  njets_min: 2
  nbjets_min: 2
  met_min: 200.0
  dphi_min: 0.4
  apply_dphi: true
```

### Run-3 Optimized
```yaml
jets:
  pt_min: 35.0
  eta_max: 2.4
event_selection:
  njets_min: 2
  nbjets_min: 2
  met_min: 220.0
  dphi_min: 0.5
  apply_dphi: true
```

### High-MET Search
```yaml
jets:
  pt_min: 40.0
  eta_max: 2.4
event_selection:
  njets_min: 3
  nbjets_min: 2
  met_min: 300.0
  dphi_min: 0.4
  apply_dphi: true
```

