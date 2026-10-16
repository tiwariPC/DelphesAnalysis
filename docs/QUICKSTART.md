# Quick Start Guide

## 1. Environment

```bash
source setup.sh      # creates the conda env "delphesanalysis" from environment.yml if needed, activates it
```

The environment provides python 3.11, gfortran/gxx 13, ROOT 6.32 and the analysis libraries
(coffea, uproot, hist, matplotlib, ...).

## 2. Everything in one command

```bash
./run_all.sh --nevents 1000
```

This generates 3 signal points and all backgrounds (MadGraph + Pythia8 + Delphes), runs the
region analysis, then sensitivity and the significance plot. Useful options:

```bash
./run_all.sh --signal-model 2hdma --npoints 5      # 2HDM+a (type II) instead of the t-channel model
./run_all.sh --points 500:100,1000:300             # explicit signal points (mMED:mDM or mA:ma)
./run_all.sh --bkg-nruns 3 --sig-nruns 2           # several merged runs per sample
./run_all.sh --skip-gen                            # reuse existing Delphes ROOT files
./run_all.sh --limits                              # also run Combine (needs `combine`)
```

`./run_all.sh --help` lists everything.

## 3. Step by step

```bash
# Generate (details: genevents/README.md)
python genevents/mg5_Generate_Run_bbMET.py --proc signal --npoints 3
python genevents/mg5_Generate_Run_bbMET.py --proc bkg

# Analyse all regions, make plots and datacards
python scripts/run_analysis_from_dirs.py \
    --signal-dir genevents/DelphesSignal \
    --background-dir genevents/DelphesBackground \
    --lumi 326.03 --output-dir output

# Sensitivity and significance
python scripts/calculate_sensitivity.py --output-dir output --output-file output/sensitivity.txt
python scripts/plot_significance.py --output-dir output --lumi 326.03
```

For t-channel signals add `--signal-xsec config/tchannel_s3m_lam2p2_cross_sections.yaml` to the
analysis command.

## 4. Results

- `output/plots/<region>/`: cutflow, kinematic distributions, per-signal datacards and shapes
- `output/plots/combined_datacard_mA{A}_ma{a}.txt`: combined datacard per signal point
- `output/sensitivity.txt`: S/B and significance per region
- `genevents/DelphesSignal/`, `genevents/DelphesBackground/`: the Delphes ROOT files

## Troubleshooting

### No signal files found
Signal files must be called `sig_bbdm_mA{A}_ma{a}_delphes_events.root`; the generator writes this
name for both models (t-channel: A = mediator mass, a = DM mass).

### Generation stops early
Look in `genevents/work/logFiles/` for the log of the failing point. Known causes:
- `Missing comma between descriptors` in `symmetry.f`: the gfortran 14 runtime is in use; the
  environment pins 13. Recreate the env from `environment.yml`.
- `Pythia ... unmatched version numbers`: `PYTHIA8DATA` points to a different Pythia; the generator
  removes it for its own runs.
- `no Delphes output at ...`: the shower or Delphes step failed; see the log of that point.

### "Tree 'Delphes' not found"
Check the tree name: `python -c "import uproot; print(list(uproot.open('file.root').keys()))"`

### Empty histograms
Check the cutflow of the region to see where events are removed, and that the file has the expected
branches (Jet, MissingET, ...). `python utils/analyze_root_structure.py file.root` lists them.

## Next Steps

- `README.md` for the full pipeline and configuration
- `genevents/README.md` for event generation
- `docs/CONFIG_FILES_GUIDE.md`, `docs/CUTS_CONFIG_GUIDE.md`, `docs/REGIONS_GUIDE.md`
