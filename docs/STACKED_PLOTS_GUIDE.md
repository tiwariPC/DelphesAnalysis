# Stacked Plots Guide

The stacked data/MC-style plots are made by `scripts/process_regions.py` (called by
`scripts/run_analysis_from_dirs.py` and `./run_all.sh`). There is no separate background script.

## What is done automatically

1. **Cross-sections** come from `config/background_cross_sections.yaml` (backgrounds) and
   `config/signal_cross_sections.yaml` or `config/tchannel_s3m_lam2p2_cross_sections.yaml` (signals).
2. **Generated events** are counted from the ROOT files.
3. **Stacked plots** of the observables (MET, m(bb), pT(bb), HT, Δφ(bb), Δφ(MET, bb)) are made for
   every region, with the signals overlaid.

## Run

```bash
python scripts/run_analysis_from_dirs.py \
    --signal-dir genevents/DelphesSignal \
    --background-dir genevents/DelphesBackground \
    --lumi 290.0 --output-dir output
```

Plots are saved in `output/plots/<region>/` as PDF and PNG.

## Useful options (`scripts/process_regions.py`)

- `--signal-scale X`: scale the signals in the plots only; histograms and datacards are unchanged.
- `--signal-xs-unit`: use 1 pb for every signal cross-section.

The plotting code is in `src/plotting_utils.py`.
