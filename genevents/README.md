# Event Generation

`mg5_Generate_Run_bbMET.py` generates signal and background events with MadGraph5 + Pythia8 +
Delphes. MadGraph is installed in this directory; the process dirs, logs and Delphes files go to
`--data-dir` (default: this directory). Run it through `../run_all.sh`, or directly:

```bash
source ../setup.sh                                        # conda env with compilers and ROOT
python mg5_Generate_Run_bbMET.py --proc signal --npoints 3   # 3 default signal points
python mg5_Generate_Run_bbMET.py --proc bkg --nruns 3        # all backgrounds, 3 merged runs each
python mg5_Generate_Run_bbMET.py --proc all --mode grid      # everything, full grid
python mg5_Generate_Run_bbMET.py --proc signal --dry-run     # print the cards, do not run
python mg5_Generate_Run_bbMET.py --proc all --npoints 3 --jobs 4 \
    --data-dir /home/ptiwari/data/DelphesAnalysisFiles     # 4 samples at a time, files on /data
```

| Option | Meaning | Default |
|---|---|---|
| `--data-dir D` | base dir for the work dir and the Delphes files | this directory |
| `--work-dir D` | MadGraph process dirs, cards (`textFiles/`), logs (`logFiles/`) | `<data-dir>/work` |
| `--signal-outdir`, `--background-outdir` | Delphes output | `<data-dir>/DelphesSignal`, `<data-dir>/DelphesBackground` |
| `--nruns`, `--bkg-nruns` | runs per signal point / background, merged with `hadd` | `1`, `--nruns` |
| `--jobs N` | samples at the same time; each MadGraph gets cores/N cores, and concurrent signal points use their own process dir copies (`<dir>_j1`, ...) | `1` |

`python mg5_Generate_Run_bbMET.py --help` and the docstring at the top of the script list all options.

## What it does

- Downloads MadGraph 3.5.6 into `MG5_aMC_v3_5_6/` if it is not there, and installs Pythia8, Delphes
  and ExRootAnalysis into it on first use (Delphes needs ROOT in the environment).
- Reads the process and settings cards from `genScripts/`, builds the process directories in
  `<work-dir>/` (all needed dirs are built first, one at a time), runs the requested points, and
  logs to `<work-dir>/logFiles/`.
- Launches with `shower=Pythia8`, `detector=Delphes` (CMS card by default, `--delphes-card`) and
  `analysis=ExRoot`.
- Makes the DM particles invisible in the Delphes card. Stock cards only know the SM and SUSY
  invisibles, so the model DM states (PDG 51, 52, 53, 56, 57, 58; `INVISIBLE_DM_PDGS` in the script)
  would otherwise deposit their energy in the calorimeters, end up in jets and give almost no MET.
  The script writes a copy of the chosen card, `<work-dir>/<card>_invisDM.dat`, with these PDGs given zero
  calorimeter response and added to `NeutrinoFilter`, and uses that copy.
- Copies the Delphes ROOT files to `<data-dir>/DelphesSignal/` and `<data-dir>/DelphesBackground/` (or
  `--signal-outdir` / `--background-outdir`). With `--nruns N` the N runs use different seeds and are
  merged with ROOT's `hadd` into one file.
- Adds each output file to `../config/samples_config.yaml` (generated list of sample files).
- Writes the cross-sections (mean over runs) to `../config/signal_cross_sections.yaml` (2HDM+a),
  `../config/tchannel_{dm}_lam{L}_cross_sections.yaml` (t-channel, e.g. `tchannel_s3m_lam2p2_...`) and
  `../config/background_cross_sections.yaml`.

## Output file names

| Sample | File |
|---|---|
| signal, 2HDM+a | `DelphesSignal/sig_bbdm_mA{mA}_ma{ma}_delphes_events.root` |
| signal, t-channel | `DelphesSignal/sig_bbdm_mA{mMED}_ma{mDM}_{dm}_lam{L}_delphes_events.root` |
| background | `DelphesBackground/{name}_delphes_events.root` |

For the t-channel model the mediator mass `mMED` and DM mass `mDM` take the places of `mA` and
`ma`, so the same analysis code reads both, and the DM type (`--dm-type`, default `s3m`) and
coupling (`--lam`, default 2.2; `2.2` is written `2p2`) are part of the name, e.g.
`sig_bbdm_mA500_ma50_s3m_lam2p2_delphes_events.root`. Samples with different settings can share a
directory; the analysis selects them with `--signal-tag s3m_lam2p2`.

## Cards (`genScripts/`)

Each model has a process card (`.mg5`: model import, `generate`, `output`) and a settings card
(`.madevent`: `set run_card ...` / `set param_card ...` lines, ending in `done`). The `launch`,
shower, detector and Delphes-card lines are added by the script.

| Sample | Process card | Settings card |
|---|---|---|
| t-channel DM, Majorana | `s3m-tChannelDM.mg5` | `s3m-tChannelDM.madevent` |
| t-channel DM, Dirac | `s3d-tChannelDM.mg5` | `s3d-tChannelDM.madevent` |
| 2HDM+a type II | `twoHDMa-typeII.mg5` | `twoHDMa-typeII.madevent` |
| backgrounds | `dyjets.mg5`, `ttbar.mg5`, `diboson.mg5`, `sTop_tchannel.mg5`, `sTop_tW.mg5`, `wlnjets.mg5`, `znnjets.mg5` | `background.madevent` (shared) |

Placeholders in the cards are filled in by the script: `__UFO_TCHANNEL__` / `__UFO_2HDMA_T2__`
(model path in `.mg5`), `__NEVENTS__`, `__ISEED__`, `__MMED__`, `__MDM__`, `__LAM__`, `__MCA__`,
`__MSA__` (in `.madevent`). The beam energy (6800 GeV) is written directly in each settings card.
Edit a card to change a generator setting, for example add `set run_card ptj 20` before `done`.

`genScripts/signal.mg5` and `signal.madevent` are the old 2HDM+a type I cards; the script no longer
uses them.

## Notes

- Signal points: `--points A:a,A:a`, else `--npoints N` (first N of 8 built-in points per model),
  else `--mode single|grid`.
- A process directory is rebuilt if a previous run left it incomplete; an existing output ROOT file
  is skipped unless you pass `--force`.
- MadGraph is run with the stack limit raised and without `PYTHIA8DATA`, so a Pythia8 installed in
  the conda env does not clash with MadGraph's own.
- Compilers and ROOT versions matter; they are pinned in `../environment.yml`.

## Legacy: Condor workflow

`condorJob/` holds the earlier cluster workflow (hand-run `mg5_aMC`, tarballs, Condor submit
files, `extract_*crosssections.py`). It uses the 2HDM+a type I model and CERN paths, and is kept for
reference only.
