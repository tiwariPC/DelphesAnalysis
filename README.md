# bbMET Analysis Pipeline

End-to-end pipeline for a bbMET dark-matter phenomenology study: event generation (MadGraph5 +
Pythia8 + Delphes) for a t-channel DM or 2HDM+a signal and the SM backgrounds, then the
signal-region analysis, plots, datacards, sensitivity and (optionally) Combine limits.

## Run Everything With One Command

```bash
./run_all.sh --nevents 1000                                   # quick test: 3 signal points + all backgrounds
./run_all.sh --data-dir /home/ptiwari/data/DelphesAnalysisFiles   # large files on a big disk
./run_all.sh --skip-gen --data-dir /home/ptiwari/data/DelphesAnalysisFiles   # analysis only
```

`run_all.sh` does, in order: environment setup, signal and background generation, the analysis,
sensitivity, the significance plot and (with `--limits`) Combine limits. It stops at the first
failure. Main options:

| Option | Meaning | Default |
|---|---|---|
| `--signal-model` | `tchannel` or `2hdma` (type II) | `tchannel` |
| `--dm-type`, `--lam` | t-channel only: `s3m`/`s3d` and coupling, fixed per call | `s3m`, `2.2` |
| `genevents/mg5_Generate_Run_bbMET.py --lam-values all` | coupling scan (λ = 1.0 ... 5.0, `DMS3D_VALS` in the script); generator only, `run_all.sh` analyses one λ at a time | |
| `--npoints N` | first N built-in signal points (up to 8) | `3` |
| `--points A:a,A:a` | explicit points (`mMED:mDM` or `mA:ma`), overrides `--npoints` | |
| `--mode grid` | full mass grid instead of `--npoints` | |
| `--nevents N` | events per run | `10000` |
| `--sig-nruns`, `--bkg-nruns` | runs per sample (different seeds), merged with `hadd` | `1` |
| `--jobs N` | samples generated at the same time; the CPU cores are split between them | `4` |
| `--cores N` | total CPU cores for generation, split between the jobs | all |
| `--slim-delphes` | write only the Delphes branches the analysis reads (about 40× smaller files) | off |
| `--bkg-cuts` | generator-level background cuts (`genevents/genScripts/<bkg>.cuts.madevent`); backgrounds go to `<data-dir>/DelphesBackground_bkgcuts`, cross-sections to `config/background_cuts_cross_sections.yaml` | off |
| `--data-dir D` | base dir for large files (see below) | `genevents` |
| `--work-dir D` | MadGraph process dirs, cards and logs | `<data-dir>/work` |
| `--signal-dir`, `--background-dir` | where the Delphes ROOT files are written and read | `<data-dir>/DelphesSignal`, `<data-dir>/DelphesBackground` |
| `--output-dir D` | analysis output (plots, datacards, sensitivity) | `output` |
| `--lumi L` | luminosity in fb⁻¹ | `326.03` |
| `--skip-gen` | reuse existing ROOT files | |
| `--force` | regenerate samples whose Delphes file already exists (otherwise they are skipped) | |
| `--limits` | also run Combine limits (needs `combine`) | |
| `--no-tmux` | run in the current shell instead of a tmux session | |
| `--attach` | attach to the tmux session right after starting it | |

`./run_all.sh --help` lists all options. Generation details are in `genevents/README.md`.

### Running in tmux

Unless it is already inside tmux (or `--no-tmux` is given), `run_all.sh` restarts itself in a
detached tmux session named `bbmet_<timestamp>`, so a dropped SSH connection does not stop it.
It prints the session name and the log file (`logs/run_all_<timestamp>.log`); follow it with
`tail -f <log>` or `tmux attach -t <session>` (detach with `Ctrl-b d`). The session stays open
after the pipeline ends so the last output can be read; close it with `tmux kill-session -t <session>`.

### Where files go

Generation writes a lot of data (MadGraph process dirs, about 90 MB of Delphes output per
1000 signal events). Put it on a large disk with `--data-dir`:

| What | Location |
|---|---|
| MadGraph process dirs, cards, logs | `<data-dir>/work/` (or `--work-dir`) |
| Signal Delphes files | `<data-dir>/DelphesSignal/` (or `--signal-dir`) |
| Background Delphes files | `<data-dir>/DelphesBackground/` (or `--background-dir`) |
| MadGraph itself | always `genevents/MG5_aMC_v3_5_6/` (its config holds absolute paths) |
| Analysis output (plots, datacards, sensitivity, limits; a few MB) | `output/` in the repo, e.g. `output/bkgcuts` (or `--output-dir`) |
| Pipeline logs | `logs/` |

Without `--data-dir` everything goes under `genevents/` in the repo. Use the same `--data-dir`
for `--skip-gen` runs so the analysis reads the files that were generated.

### Slim Delphes output and background generator cuts

`--slim-delphes` keeps only jets, electrons, muons, photons, MET, scalar HT and the generator-level
jets and MET in the Delphes output (the CMS card otherwise also writes all generator particles,
tracks, towers and energy-flow objects): about 2 KB instead of 80 KB per event.

`--bkg-cuts` applies generator-level cuts to the backgrounds so the generated events populate the
high-MET signal regions instead of the bulk of the inclusive cross-section. The cuts are in one card
per background, `genevents/genScripts/<bkg>.cuts.madevent` (e.g. `misset 150` for Z(νν)+jets,
`ptheavy 100` for ttbar), and are chosen looser than the signal regions; compare with a run without
`--bkg-cuts` to check that they remove no signal-region background. Samples generated with and
without the cuts are kept in separate directories and cross-section files.

Example, 1M events per sample in 10 runs of 100k, 10 samples at a time on 20 cores:

```bash
./run_all.sh --nevents 100000 --sig-nruns 10 --bkg-nruns 10 --jobs 10 --cores 20 \
    --slim-delphes --bkg-cuts --data-dir /home/ptiwari/data_2/DelphesAnalysisFiles
```

### Parallel generation

With `--jobs N`, N samples (signal points and backgrounds) are generated at the same time and each
MadGraph instance uses cores/N cores. Signal points running at the same time use separate copies
of the process directory (`<dir>_j1`, `<dir>_j2`, ...), which MadGraph builds from the same card
before the runs start.

### Environment

`source setup.sh` creates the conda env `delphesanalysis` from `environment.yml` if it does not
exist, and activates it (bash or zsh). `run_all.sh` calls it for you. The environment pins
python 3.11, gfortran/gxx 13 and ROOT 6.32: MadGraph 3.5.6 generates Fortran that fails at run
time with the gfortran >= 14 runtime.

Combine (only needed for `--limits`) is built on request into the same env:

```bash
INSTALL_COMBINE=1 source setup.sh                      # latest main
COMBINE_TAG=v10.2.1 INSTALL_COMBINE=1 source setup.sh  # or pin a release
```

The source is cloned into `combine/HiggsAnalysis/CombinedLimit` (git-ignored), built with
CMake against the env's ROOT, and installed into `$CONDA_PREFIX`. After that, `combine`,
`text2workspace.py` and `diffNuisances.py` are on PATH, and `import HiggsAnalysis.CombinedLimit`
works in python whenever `delphesanalysis` is active. The build adds cmake, ninja, eigen and
libboost-devel to the env.

## Selection

All cuts are in `config/cuts_config.yaml`.

**Objects.** Delphes has no ID or isolation working points: the CMS Delphes card already applies
its efficiencies and isolation to electrons, muons and photons, and taus are Delphes jets with
`TauTag` (in place of the DeepTau working points).

| Object | Selection |
|---|---|
| Muon | pT > 10 GeV, \|η\| < 2.4 |
| Electron | pT > 10 GeV, \|η\| < 2.5 |
| Photon | pT > 15 GeV, \|η\| < 2.5 |
| Tau | TauTag jet, pT > 20 GeV, \|η\| < 2.3 |
| Jet | pT > 30 GeV, \|η\| < 2.5, removed within ΔR < 0.4 of a selected lepton or photon |
| b-jet | selected jet with `BTag >= 1` |

**Signal regions.** This is a phenomenology study, so only signal regions are used (no control
regions):

| Cut | SR1b | SR2b |
|---|---|---|
| b-jets | = 1 | = 2 |
| jets | 1–2 | 2–3 |
| electrons + muons, taus, photons | 0 | 0 |
| MET | > 250 GeV | > 250 GeV |
| leading jet pT | > 100 GeV | > 100 GeV |
| min Δφ(jet, MET) | > 0.5 | > 0.5 |

The significance is quoted per signal region and for their combination per signal point. The two
regions are disjoint (exactly 1 or exactly 2 b-tags), so the significances add in quadrature.

**Delphes and the DM particle.** The generator writes a copy of the Delphes card with the model DM
states (PDG 51, 52, 53, 56, 57, 58) made invisible; the stock card would put them in the
calorimeters and the signal would have almost no MET (see `genevents/README.md`).

## Directory Structure

```
DelphesAnalysis/
├── run_all.sh              # One-command pipeline (setup, generation, analysis, sensitivity, plots)
├── setup.sh                # Create/activate the conda env
├── environment.yml         # Conda environment (python, compilers, ROOT, analysis libraries)
├── genevents/              # Self-contained event generation (see genevents/README.md)
│   ├── mg5_Generate_Run_bbMET.py   # MadGraph + Pythia8 + Delphes driver
│   ├── genScripts/                 # Editable process (.mg5) and settings (.madevent) cards
│   ├── ufo_model/                  # t-channel DM and 2HDM+a type II UFO models
│   └── condorJob/                  # Legacy Condor workflow
├── src/                    # Core analysis modules
│   ├── bbdmDelphes.py     # Object selection, jet cleaning, observables, datacards
│   ├── regions.py         # Region definitions and selection logic
│   └── plotting_utils.py  # Plotting functions with CMS styling
├── scripts/                # Executable analysis scripts
│   ├── process_regions.py  # Main analysis (signal + backgrounds in all signal regions)
│   ├── run_analysis_from_dirs.py  # Finds the signal/background files and runs process_regions.py
│   ├── calculate_sensitivity.py   # S, B, S/B and significance per region and combined
│   ├── create_combined_datacard.py
│   ├── plot_significance.py
│   └── run_combine_limits.py
├── config/                 # Configuration files
│   ├── cuts_config.yaml   # Object selection and signal-region definitions
│   ├── background_cross_sections.yaml  # Background cross-sections (written by the generator)
│   ├── signal_cross_sections.yaml  # 2HDM+a signal cross-sections (mA, ma)
│   ├── tchannel_{dm}_lam{L}_cross_sections.yaml  # t-channel signal cross-sections (mMED as mA, mDM as ma)
│   └── samples_config.yaml  # Generated list of sample files (not tracked by git)
├── utils/
│   ├── samples_config.py  # Reads/writes config/samples_config.yaml
│   └── analyze_root_structure.py
└── docs/                   # Additional guides
```

## Running the Analysis Without run_all.sh

### From the sample directories (recommended)

```bash
python scripts/run_analysis_from_dirs.py \
    --signal-dir /path/to/DelphesSignal --signal-tag s3m_lam2p2 \
    --background-dir /path/to/DelphesBackground \
    --signal-xsec config/tchannel_s3m_lam2p2_cross_sections.yaml \
    --lumi 326.03 \
    --output-dir output
python scripts/calculate_sensitivity.py --output-dir output --output-file output/sensitivity.txt
python scripts/plot_significance.py --output-dir output --lumi 326.03
```

File names (written by the generator):
- **Signal, 2HDM+a**: `sig_bbdm_mA{mA}_ma{ma}_delphes_events.root`, cross-sections in
  `config/signal_cross_sections.yaml`.
- **Signal, t-channel**: `sig_bbdm_mA{mMED}_ma{mDM}_{dm}_lam{L}_delphes_events.root`, e.g.
  `sig_bbdm_mA500_ma50_s3m_lam2p2_delphes_events.root` (S3M, λ = 2.2), cross-sections in
  `config/tchannel_{dm}_lam{L}_cross_sections.yaml`. Pass the tag with `--signal-tag s3m_lam2p2`
  so only those files are used; `run_all.sh` sets the tag and the cross-section file from
  `--dm-type` and `--lam`.
- **Background**: `{name}_delphes_events.root`

`run_analysis_from_dirs.py` finds the signal points (with `--signal-tag`, only that DM type and
coupling; it stops if one mass point appears with two tags and no tag is given) and backgrounds,
counts the generated events,
rewrites `config/samples_config.yaml` with the files found, and runs `process_regions.py` with all
signals overlaid on the same plots.

### Specifying files by hand

```bash
python scripts/process_regions.py \
    --signal-file /path/to/sig_bbdm_mA500_ma50_s3m_lam2p2_delphes_events.root \
    --signal-mA 500 --signal-ma 50 --signal-ngen 1000 \
    --signal-file /path/to/sig_bbdm_mA1000_ma100_s3m_lam2p2_delphes_events.root \
    --signal-mA 1000 --signal-ma 100 --signal-ngen 1000 \
    --signal-xsec config/tchannel_s3m_lam2p2_cross_sections.yaml \
    --lumi 326.03 --output-dir output
```

- With `--signal-mA`/`--signal-ma` the cross-section is taken from the `--signal-xsec` file;
  `--signal-xs` gives it directly.
- `--signal-scale 10` scales the signal in the plots only; histograms and datacards are unchanged.
- `--signal-xs-unit` sets every signal cross-section to 1 pb.
- Backgrounds are read from `config/samples_config.yaml`.

### Sensitivity

```bash
python scripts/calculate_sensitivity.py --output-dir output --output-file output/sensitivity.txt
python scripts/calculate_sensitivity.py --output-dir output --region sr1b
python scripts/calculate_sensitivity.py --output-dir output --format csv --output-file sensitivity.csv
```

It reads the datacards in `output/datacards/sr*/` and lists signal, background, S/B and the Asimov
significance for each region and signal point, plus `combined/<point>` (SR1b and SR2b).

`plot_significance.py` draws the same numbers as grouped bars (SR1b, SR2b and their combination)
per signal point, in CMS style, with the 3σ and 5σ levels (`significance_asimov.pdf` and `.png`). By default it
uses the Asimov formula `Z = sqrt(2((s+b)ln(1+s/b) - s))`, the same as `sensitivity.txt`;
`--formula simple` gives `s/sqrt(s+b)` instead. A second plot, `significance_punzi.pdf` and `.png`,
shows the Punzi figure of merit `eps_S / (a/2 + sqrt(B))` (a = 5 by default, `--punzi-a`), where
`eps_S = S / (sigma L)` is the fraction of the produced signal that passes the selection; it needs the
signal cross-sections (`--signal-xsec`, passed by `run_all.sh`). For SR1b + SR2b the yields of the two
regions are added; the Asimov significances of the two regions are added in quadrature. The luminosity
is 326.03 fb⁻¹. All significances are statistical only.

### Combined datacards and expected limits

`process_regions.py` writes one datacard per signal region and signal point,
`output/datacards/sr1b/datacard_mA{A}_ma{a}.txt` and `output/datacards/sr2b/...`, and combines the
two regions into `output/datacards/combined_datacard_mA{A}_ma{a}.txt` (bins `SR1`, `SR2`). They are
counting datacards (no shapes file); the background normalisations carry 20-40% lnN uncertainties
(shared between the two regions) and the luminosity 2.5% (`lumi_13p6TeV`).

With Combine installed (`INSTALL_COMBINE=1 source setup.sh`), the expected limits are made with

```bash
python scripts/run_combine_limits.py --output-dir output/bkgcuts \
    --signal-xsec config/tchannel_s3m_lam2p2_cross_sections.yaml      # or run_all.sh --limits
```

It runs `combine -M AsymptoticLimits -t -1 --run expected` (Asimov data, there is no data) on every
datacard and writes `limits.txt` and `limits.json`: the median and the 68% / 95% bands of the 95% CL
upper limit on the signal strength `r` (`r < 1`: the signal point is expected to be excluded), and with
`--signal-xsec` also the limit on the cross-section in pb. It also draws Brazilian-flag plots of the
SR1b + SR2b combination, `limits_combined.pdf/.png` (limit on `r`) and `limits_combined_xsec.pdf/.png`
(limit in pb next to the model cross-section), with the signal points ordered by mediator mass on the
x axis, labelled `(m_med, m_DM)`.

Notes: the limits include only the uncertainties in the datacard (no statistical uncertainty of the
simulation, no shape uncertainties). The asymptotic method sometimes fails for the lower bands of a
single region (`n/a` in `limits.txt`); the range of `r` given to Combine is chosen from the signal and
background yields because the method breaks down if it is far larger than the limit.

## Configuration

- **`config/cuts_config.yaml`**: object selection and signal-region definitions (edit by hand)
- **`config/background_cross_sections.yaml`**: background cross-sections
- **`config/signal_cross_sections.yaml`**: 2HDM+a signal cross-sections for all (mA, ma) points
- **`config/tchannel_{dm}_lam{L}_cross_sections.yaml`**: t-channel signal cross-sections per DM type and
  coupling (e.g. `tchannel_s3m_lam2p2_cross_sections.yaml`), same layout (mMED, mDM)
- **`config/samples_config.yaml`**: generated list of the signal and background ROOT files

The generator writes the three cross-section files from the MadGraph runs (mean over runs).

`config/samples_config.yaml` is generated and not tracked by git: the generator adds each Delphes
file it writes, and `run_analysis_from_dirs.py` rewrites it from the `--signal-dir` and
`--background-dir` it is given, dropping entries whose files no longer exist. To use other files,
point those options (or `--data-dir`) at the directories that hold them.

See `docs/CONFIG_FILES_GUIDE.md` for detailed configuration documentation.

## Output Structure

One directory per analysis, e.g. `output/bkgcuts/` (backgrounds generated with generator cuts) and
`output/inclusive/` (without), each with:

```
output/bkgcuts/
├── sensitivity.txt                          # S, B, S/B, Z per region and combined
├── significance_asimov.pdf / .png           # grouped bars per signal point, Asimov Z
├── significance_punzi.pdf / .png            # Punzi figure of merit eps_S / (a/2 + sqrt(B)), a = 5
├── limits.txt, limits.json                  # expected limits (with --limits)
├── limits_combined.pdf / .png               # Brazilian-flag plot, limit on r
├── limits_combined_xsec.pdf / .png          # Brazilian-flag plot, limit in pb
├── limits/                                  # Combine output files
├── datacards/
│   ├── combined_datacard_mA{A}_ma{a}.txt    # SR1b + SR2b, per signal point
│   ├── sr1b/datacard_mA{A}_ma{a}.txt
│   └── sr2b/datacard_mA{A}_ma{a}.txt
└── plots/
    ├── sr1b/                                # cutflow, met, mbb, ptbb, ht, dphi_jet_met_min, cost_star_jj
    └── sr2b/                                # same plus cost_star (two leading b-jets)
```

Plots use the CMS style (mplhep) for 13.6 TeV; the legends have no frame, are smaller, and spread
over several columns (and the y axis gets headroom) so they do not cover the distributions.
`cost_star_jj` is `|cos θ*| = |tanh(Δη_jj / 2)|` of the two leading jets (any flavour), `cost_star` the
same for the two leading b-jets (SR2b only). `plots` and `datacards` are cleared at the start of each
analysis run. `output/` is git-ignored.

## Requirements

See `environment.yml` for dependencies (`source setup.sh` creates the conda env).

## Documentation

- `genevents/README.md` - Event generation
- `docs/RUN_ANALYSIS.md` - Quick reference for running the analysis
- `docs/AUTO_DISCOVERY_GUIDE.md` - Automatic file discovery guide
- `docs/COMPLETE_ANALYSIS_GUIDE.md` - Full workflow guide
- `docs/CONFIG_FILES_GUIDE.md` - Configuration files structure
- `docs/REGIONS_GUIDE.md` - Region definitions
- `docs/CUTS_CONFIG_GUIDE.md` - Configuration guide
