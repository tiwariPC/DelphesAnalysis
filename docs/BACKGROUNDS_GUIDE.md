# Background Processing Guide

## Backgrounds

| Name | Process | MadGraph card |
|---|---|---|
| `dyjets` | Z(→ℓℓ) + 2 jets | `genevents/genScripts/dyjets.mg5` |
| `ttbar` | tt̄ | `ttbar.mg5` |
| `diboson` | WW, WZ, ZZ | `diboson.mg5` |
| `sTop_tchannel` | single top, t-channel | `sTop_tchannel.mg5` |
| `sTop_tW` | single top, tW | `sTop_tW.mg5` |
| `wlnjets` | W(→ℓν) + 2 jets | `wlnjets.mg5` |
| `znnjets` | Z(→νν) + 2 jets | `znnjets.mg5` |

## Step 1: Generate

```bash
python genevents/mg5_Generate_Run_bbMET.py --proc bkg                       # all backgrounds
python genevents/mg5_Generate_Run_bbMET.py --proc ttbar,znnjets --nruns 3   # a subset, 3 merged runs each
python genevents/mg5_Generate_Run_bbMET.py --proc bkg --background-outdir /data/DelphesBackground
```

Output: `genevents/DelphesBackground/{name}_delphes_events.root` (or `--background-outdir`). With
`--nruns N`, N runs with different seeds are merged with `hadd` into that one file. The
generator also writes the cross-section (mean over the runs) to
`config/background_cross_sections.yaml`. See `genevents/README.md`.

`./run_all.sh` does this for you (`--bkg-nruns`, `--background-dir`).

## Step 2: Configure

- **Cross-sections**: `config/background_cross_sections.yaml` (pb), filled by the generator; edit it
  to use literature values.
- **File paths**: `config/samples_config.yaml`, updated automatically by
  `scripts/run_analysis_from_dirs.py` from the files it finds.

## Step 3: Run the analysis

```bash
python scripts/run_analysis_from_dirs.py \
    --signal-dir genevents/DelphesSignal \
    --background-dir genevents/DelphesBackground \
    --lumi 290.0 --output-dir output
```

Backgrounds are processed together with the signals; the number of generated events is counted from
each ROOT file. Plots and datacards are in `output/plots/<region>/`.

## Troubleshooting

### A background is missing in the plots
- Check that `{name}_delphes_events.root` exists in the background directory.
- The discovery looks for exactly `dyjets`, `ttbar`, `diboson`, `sTop_tchannel`, `sTop_tW`,
  `wlnjets`, `znnjets`.

### Wrong normalisation
- Check the cross-section in `config/background_cross_sections.yaml`.
- Every run of a multi-run sample is merged; the cross-section is the mean over the runs.
