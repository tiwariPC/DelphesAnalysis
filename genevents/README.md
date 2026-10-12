# Event Generation

`mg5_Generate_Run_bbMET.py` generates signal and background events with MadGraph5 + Pythia8 +
Delphes, entirely inside this directory. Run it through `../run_all.sh`, or directly:

```bash
source ../setup.sh                                        # conda env with compilers and ROOT
python mg5_Generate_Run_bbMET.py --proc signal --npoints 3   # 3 default signal points
python mg5_Generate_Run_bbMET.py --proc bkg --nruns 3        # all backgrounds, 3 merged runs each
python mg5_Generate_Run_bbMET.py --proc all --mode grid      # everything, full grid
python mg5_Generate_Run_bbMET.py --proc signal --dry-run     # print the cards, do not run
```

`python mg5_Generate_Run_bbMET.py --help` and the docstring at the top of the script list all options.

## What it does

- Downloads MadGraph 3.5.6 into `MG5_aMC_v3_5_6/` if it is not there, and installs Pythia8, Delphes
  and ExRootAnalysis into it on first use (Delphes needs ROOT in the environment).
- Reads the process and settings cards from `genScripts/`, builds the process directories in
  `work/`, runs the requested points, and logs to `work/logFiles/`.
- Launches with `shower=Pythia8`, `detector=Delphes` (CMS card by default, `--delphes-card`) and
  `analysis=ExRoot`.
- Copies the Delphes ROOT files to `DelphesSignal/` and `DelphesBackground/` (or
  `--signal-outdir` / `--background-outdir`). With `--nruns N` the N runs use different seeds and are
  merged with ROOT's `hadd` into one file.
- Writes the cross-sections (mean over runs) to `../config/signal_cross_sections.yaml` (2HDM+a),
  `../config/tchannel_signal_cross_sections.yaml` (t-channel) and
  `../config/background_cross_sections.yaml`.

## Output file names

| Sample | File |
|---|---|
| signal | `DelphesSignal/sig_bbdm_mA{A}_ma{a}_delphes_events.root` |
| background | `DelphesBackground/{name}_delphes_events.root` |

For 2HDM+a, `A` and `a` are `mA` and `ma`. For the t-channel model they are the mediator mass
`mMED` and the DM mass `mDM`, so the same analysis code reads both. The DM type (`--dm-type`) and
coupling (`--lam`) are not in the name: use one of each per signal directory.

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
