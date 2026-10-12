#!/usr/bin/env python3
"""Self-contained MadGraph5 + Pythia8 + Delphes generation for bbMET.

Cards live in genScripts/ (<name>.mg5 process, <name>.madevent settings); MadGraph is
installed into this directory on first use.

Output files (inputs of run_analysis_from_dirs.py), one merged file per sample:
  signal      <signal-outdir>/sig_bbdm_mA{mA}_ma{ma}_delphes_events.root            (2hdma)
              <signal-outdir>/sig_bbdm_mA{mMED}_ma{mDM}_{dm}_lam{L}_delphes_events.root (tchannel,
              e.g. sig_bbdm_mA500_ma50_s3m_lam2p2_delphes_events.root)
  background  <background-outdir>/<name>_delphes_events.root

Usage: python mg5_Generate_Run_bbMET.py [options]
  --proc P              signal | bkg | all | comma list of backgrounds (default signal)
  --signal-model M      tchannel (S3M/S3D) | 2hdma (type II)     (default tchannel)
  --dm-type T           t-channel only: s3m | s3d, one per call  (default s3m)
  --lam L               t-channel coupling DMS3D 3 3, fixed per call (default 2.2)
  --points A:a,A:a,...  explicit points (mMED:mDM or mA:ma)
  --npoints N           first N built-in default points (up to 8)
  --mode single|grid    no --points/--npoints: one point, or the full grid
  --mMED/--mDM          t-channel point for single mode          (1000, 100)
  --mA/--ma             2HDM+a point for single mode             (500, 100)
  --nevents N           events per run                           (default 10000)
  --nruns N             runs per signal point, other seeds, merged with hadd (default 1)
  --bkg-nruns N         runs per background, merged with hadd    (default: --nruns)
  --jobs N              samples generated at the same time; each MadGraph gets
                        cores/N cores, and concurrent signal points use copies of the
                        process dir (<dir>_j1, ...)              (default 1)
  --cores N             total cores split between the jobs       (default: all)
  --slim-delphes        write only the Delphes branches the analysis reads (jets, leptons,
                        photons, MET, HT, gen jets/MET): much smaller files
  --bkg-cuts            generator-level cuts for the backgrounds from genScripts/<bkg>.cuts.madevent;
                        output to <data-dir>/DelphesBackground_bkgcuts
  --seed S              base iseed; point index and run number are added (default 1000)
  --data-dir D          base dir for large files                 (default genevents/)
  --work-dir D          MadGraph process dirs, cards, logs       (default <data-dir>/work)
  --signal-outdir D     (default <data-dir>/DelphesSignal)
  --background-outdir D (default <data-dir>/DelphesBackground)
  --delphes-card C      CMS (default), another MG5 card suffix, or a .dat path
  --keep-events         keep MadGraph Events/<run> directories
  --force               regenerate even if the output ROOT file exists
  --dry-run             print the cards only, do not run MadGraph

Examples:
  # t-channel DM signal
  python mg5_Generate_Run_bbMET.py --proc signal --signal-model tchannel --npoints 3
  python mg5_Generate_Run_bbMET.py --proc signal --signal-model tchannel --dm-type s3d --points 1000:100,1500:500

  # 2HDM+a signal
  python mg5_Generate_Run_bbMET.py --proc signal --signal-model 2hdma --npoints 3
  python mg5_Generate_Run_bbMET.py --proc signal --signal-model 2hdma --mA 500 --ma 100 --nruns 3

  # backgrounds, 3 merged runs each, into a chosen directory
  python mg5_Generate_Run_bbMET.py --proc bkg --nruns 3 --background-outdir DIR
  python mg5_Generate_Run_bbMET.py --proc ttbar,znnjets --nevents 100000 --delphes-card CMS

MadGraph itself stays in genevents/ (its config holds absolute paths).

Environment: MG5_EXE, MG5_PYTHON (<= 3.12), MG5_TARBALL_URL, UFO_TCHANNEL, UFO_2HDMA.
Cross-sections (mean over runs) go to config/signal_cross_sections.yaml (2hdma),
config/tchannel_{dm}_lam{L}_cross_sections.yaml (tchannel) and config/background_cross_sections.yaml
(config/background_cuts_cross_sections.yaml with --bkg-cuts).
"""

import argparse
import glob
import itertools
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import urllib.request

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_SCRIPT_DIR)

# ------------------------------------------------------------------ MadGraph env
MG5_VERSION = "3.5.6"
MG5_TARBALL_URL = os.environ.get(
    "MG5_TARBALL_URL",
    f"https://cms-project-generators.web.cern.ch/cms-project-generators/MG5_aMC_v{MG5_VERSION}.tar.gz",
)
MG5_INSTALL_ROOT = _SCRIPT_DIR
_MG5_DIRNAME = f"MG5_aMC_v{MG5_VERSION.replace('.', '_')}"
MG5_EXE = os.environ.get("MG5_EXE", "")
MG5_PYTHON = os.environ.get("MG5_PYTHON", "")
MAX_PYTHON = (3, 12)   # newest version MadGraph's model reader tolerates

# t-channel model path with NO space before the restriction (see reference script):
# "<path>/DMSimpt_v2_0-S3M_br"; a separate "-res_..." token is silently dropped.
UFO_TCHANNEL = os.environ.get("UFO_TCHANNEL", os.path.join(_SCRIPT_DIR, "ufo_model", "DMSimpt_v2_0"))
# 2HDM+a type II UFO shipped in ufo_model/Pseudoscalar_2HDM (4FS, DM particle "Xd").
UFO_2HDMA = os.environ.get("UFO_2HDMA", os.path.join(_SCRIPT_DIR, "ufo_model", "Pseudoscalar_2HDM"))

# Process dirs, MadGraph logs and the Delphes output go under the data dir (--data-dir,
# default this directory); MadGraph itself always stays here (its config holds absolute
# paths to Pythia8 and Delphes).
DATA_DIR = _SCRIPT_DIR
WORK_DIR = os.path.join(DATA_DIR, "work")
CARD_DIR = os.path.join(WORK_DIR, "textFiles")
LOG_DIR = os.path.join(WORK_DIR, "logFiles")

# ------------------------------------------------------------------ physics config

# t-channel scan (same as tChannelDM-mg5EventGeneration)
T_MED_VALS = [10, 50, 100, 200, 500, 750, 1000, 1500, 2000, 2500]
T_DM_VALS = [1, 10, 50, 100, 200, 500, 750, 1000, 1500, 2000]

# 2HDM+a scan (same as condorJob/make_joblist.py)
H_MASS_A = [300, 500, 700, 900, 1200, 1500]
H_MASS_a = [50, 100, 200, 300, 400, 500, 600, 800, 1000, 1200]

# Default points: --npoints N takes the first N of these (mMED, mDM) / (mA, ma) pairs.
DEFAULT_POINTS = {
    "tchannel": [(500, 50), (1000, 100), (1500, 500), (750, 10), (2000, 200), (1000, 500), (2500, 1000), (500, 200)],
    "2hdma": [(300, 50), (500, 100), (700, 200), (900, 300), (1200, 500), (1500, 800), (500, 50), (700, 100)],
}

# Process cards (model import, generate, output) and launch-settings cards live in
# genScripts/ and are editable: <name>.mg5 and <name>.madevent. Placeholders
# __LIKE_THIS__ are filled in by render_template(); the process directory name is taken
# from the card's own "output" line. The launch/shower/detector/Delphes-card lines are
# added by the script because they depend on the run name and on the process dir state.
TEMPLATE_DIR = os.path.join(_SCRIPT_DIR, "genScripts")

TCHANNEL_PROCS = {
    "s3m": dict(mg5="s3m-tChannelDM.mg5", madevent="s3m-tChannelDM.madevent"),
    "s3d": dict(mg5="s3d-tChannelDM.mg5", madevent="s3d-tChannelDM.madevent"),
}

HDMA_PROC = dict(mg5="twoHDMa-typeII.mg5", madevent="twoHDMa-typeII.madevent")

# Backgrounds: each one is genScripts/<name>.mg5 (process) launched with the shared
# settings card genScripts/background.madevent.
BACKGROUNDS = ["dyjets", "ttbar", "diboson", "sTop_tchannel", "sTop_tW", "wlnjets", "znnjets"]

# ------------------------------------------------------------------ MadGraph install
def local_madgraph():
    return os.path.join(MG5_INSTALL_ROOT, _MG5_DIRNAME, "bin", "mg5_aMC")


def find_madgraph():
    """Explicit MG5_EXE wins (and must exist); else the copy inside genevents/."""
    if MG5_EXE:
        if not os.path.exists(MG5_EXE):
            sys.exit(f"Error: MG5_EXE is set to {MG5_EXE}, which does not exist.\n"
                     f"Correct it, or unset it to install MadGraph {MG5_VERSION} into {MG5_INSTALL_ROOT}.")
        return MG5_EXE
    local = local_madgraph()
    return local if os.path.exists(local) else None


def _extract_root(archive):
    roots = {name.split("/", 1)[0] for name in archive.getnames() if name.strip("./")}
    roots.discard("")
    if len(roots) != 1:
        raise RuntimeError(f"expected one top-level directory in the tarball, found {sorted(roots)}")
    return roots.pop()


def install_madgraph():
    """Download+unpack MadGraph into genevents/. Staged, so an interrupted
    download cannot leave a half-populated tree mistaken for a good install."""
    destination = os.path.join(MG5_INSTALL_ROOT, _MG5_DIRNAME)
    launcher = os.path.join(destination, "bin", "mg5_aMC")
    print(f"[env] MadGraph not found, downloading {MG5_TARBALL_URL}", flush=True)
    os.makedirs(MG5_INSTALL_ROOT, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=MG5_INSTALL_ROOT) as staging:
        tarball = os.path.join(staging, "mg5.tar.gz")
        try:
            urllib.request.urlretrieve(MG5_TARBALL_URL, tarball)
        except Exception as e:
            sys.exit(f"Error: could not download MadGraph from {MG5_TARBALL_URL}\n{e}\n"
                     f"Install it manually and set MG5_EXE to its bin/mg5_aMC.")
        print("[env] extracting", flush=True)
        try:
            with tarfile.open(tarball, "r:gz") as archive:
                root = _extract_root(archive)
                try:
                    archive.extractall(staging, filter="data")
                except TypeError:      # Python < 3.12 has no filter argument
                    archive.extractall(staging)
        except (tarfile.TarError, RuntimeError) as e:
            sys.exit(f"Error: could not extract the MadGraph tarball\n{e}")
        extracted = os.path.join(staging, root)
        if not os.path.exists(os.path.join(extracted, "bin", "mg5_aMC")):
            sys.exit(f"Error: {MG5_TARBALL_URL} did not contain bin/mg5_aMC")
        if os.path.isdir(destination):
            shutil.rmtree(destination)
        shutil.move(extracted, destination)
    os.chmod(launcher, 0o755)
    print(f"[env] installed MadGraph {MG5_VERSION} at {destination}", flush=True)
    return launcher


def interpreter_version(python_exe):
    probe = subprocess.run([python_exe, "-c", "import sys; print('%d %d' % sys.version_info[:2])"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        return None
    return tuple(int(x) for x in probe.stdout.split())


def pick_python():
    """MG5_PYTHON if given, else the first interpreter <= 3.12 found."""
    if MG5_PYTHON:
        return MG5_PYTHON
    candidates = [sys.executable] + [shutil.which(f"python3.{m}") for m in (12, 11, 10, 9)]
    for exe in candidates:
        if exe and (v := interpreter_version(exe)) and v <= MAX_PYTHON:
            return exe
    return sys.executable


def check_environment():
    global MG5_EXE, MG5_PYTHON
    MG5_EXE = find_madgraph() or install_madgraph()
    MG5_PYTHON = pick_python()
    ver = interpreter_version(MG5_PYTHON)
    if ver is None:
        sys.exit(f"Error: cannot run interpreter {MG5_PYTHON}. Set MG5_PYTHON to a Python <= 3.12.")
    if ver > MAX_PYTHON:
        sys.exit(f"Error: {MG5_PYTHON} is Python {ver[0]}.{ver[1]}; MadGraph's model reader needs <= 3.12 "
                 f"(PEP 667). Set MG5_PYTHON to such an interpreter.")
    missing = [t for t in ("gfortran", "g++", "make") if shutil.which(t) is None]
    if missing:
        sys.exit(f"Error: {', '.join(missing)} not found in PATH; MadGraph cannot build processes.\n"
                 f"Run 'source setup.sh' (installs them into the conda env), or install them system-wide.")
    print(f"[env] MadGraph : {MG5_EXE}", flush=True)
    print(f"[env] Python   : {MG5_PYTHON} ({ver[0]}.{ver[1]})", flush=True)


def restore_mg5_autoupdate_default(mg5_root):
    """Earlier versions of this script wrote 'auto_update = 0' into MadGraph's config.
    MadGraph then keeps it as a string and crashes in its quit handler
    (TypeError: bad operand type for abs(): 'str'). Put the shipped default back."""
    conf = os.path.join(mg5_root, "input", "mg5_configuration.txt")
    if not os.path.exists(conf):
        return
    with open(conf) as f:
        text = f.read()
    new = re.sub(r"^auto_update\s*=.*$", "# auto_update = 7", text, flags=re.M)
    if new != text:
        with open(conf, "w") as f:
            f.write(new)
        print("[env] restored default auto_update in MadGraph config", flush=True)


def delphes_installed(mg5_root):
    """Delphes counts as installed only when its HepMC reader was built."""
    d = os.path.join(mg5_root, "Delphes")
    return os.path.isdir(d) and any(n.startswith("DelphesHepMC") and os.access(os.path.join(d, n), os.X_OK)
                                    for n in os.listdir(d))


def pythia8_installed(mg5_root):
    """Pythia8 counts as installed only when its library and MadGraph's Pythia8 interface
    were built. An interrupted build leaves HEPTools/pythia8 with just the install log,
    and MadGraph then silently switches the shower and Delphes off."""
    heptools = os.path.join(mg5_root, "HEPTools")
    lib = os.path.join(heptools, "pythia8", "lib")
    has_lib = os.path.isdir(lib) and any(n.startswith("libpythia8") for n in os.listdir(lib))
    iface = os.path.join(heptools, "MG5aMC_PY8_interface", "MG5aMC_PY8_interface")
    return has_lib and os.access(iface, os.X_OK)


def exroot_installed(mg5_root):
    return os.path.isfile(os.path.join(mg5_root, "ExRootAnalysis", "libExRootAnalysis.so"))


def ensure_tools():
    """Install Pythia8, Delphes and ExRootAnalysis into MadGraph if missing."""
    mg5_root = os.path.dirname(os.path.dirname(MG5_EXE))
    restore_mg5_autoupdate_default(mg5_root)
    heptools = os.path.join(mg5_root, "HEPTools")
    checks = {"install pythia8": pythia8_installed, "install Delphes": delphes_installed,
              "install ExRootAnalysis": exroot_installed}
    need = [cmd for cmd, ok in checks.items() if not ok(mg5_root)]
    if not need:
        return
    if "install pythia8" in need:
        for d in ("pythia8", "MG5aMC_PY8_interface"):
            partial = os.path.join(heptools, d)
            if os.path.isdir(partial):
                print(f"[env] removing incomplete {partial}", flush=True)
                shutil.rmtree(partial)
    card = os.path.join(CARD_DIR, "install_tools.txt")
    with open(card, "w") as f:
        f.write("\n".join(need) + "\n")
    print(f"[env] installing: {', '.join(need)} (needs ROOT for Delphes; Pythia8 takes "
          f"10-30 min, do not interrupt)", flush=True)
    execute_madgraph(card, "INSTALL")
    failed = [cmd.split()[1] for cmd in need if not checks[cmd](mg5_root)]
    if failed:
        sys.exit(f"Error: install of {', '.join(failed)} failed (Delphes needs ROOT/root-config in PATH). "
                 f"See {os.path.join(LOG_DIR, 'madgraph_install_tools.log')} and "
                 f"{os.path.join(heptools, 'pythia8', 'pythia8_install.log')}")


# ------------------------------------------------------------------ MG5 execution
_MG5_ERROR_MARKERS = ("interrupted with error", "Error detected in sub-command",
                      "command not executed:", "Unable to evaluate")
_ANSI = re.compile(r"\033\[[0-9;]*m")
_INTERESTING = re.compile(r"(Running Survey|Cross-section|finished|ERROR|launch|Results|Generating|==)")
_XS_RE = re.compile(r"Cross-section\s*:\s*([\d.eE+\-]+)\s*\+-\s*[\d.eE+\-]+\s*pb")


def scan_log_for_errors(log_path):
    """MadGraph exits 0 even when a sub-command dies; look for its error markers."""
    try:
        with open(log_path, errors="ignore") as f:
            for line in f:
                clean = _ANSI.sub("", line).strip()
                if any(m in clean for m in _MG5_ERROR_MARKERS):
                    return clean
    except OSError:
        return None
    return None


def parse_cross_section(log_path):
    """Last 'Cross-section : X +- Y pb' in the log, in pb; None if absent."""
    try:
        with open(log_path, errors="ignore") as f:
            hits = _XS_RE.findall(_ANSI.sub("", f.read()))
    except OSError:
        return None
    return float(hits[-1]) if hits else None


def _tail_log(log_path, stop_event, prefix):
    import time
    while not os.path.exists(log_path):
        if stop_event.is_set():
            return
        time.sleep(0.5)
    with open(log_path) as f:
        f.seek(0, 2)
        while not stop_event.is_set():
            line = f.readline()
            if not line:
                time.sleep(0.3)
                continue
            line = line.rstrip()
            if line and _INTERESTING.search(line):
                print(f"  [{prefix}] {line}", flush=True)


def _raise_stack_limit():
    """Equivalent of 'ulimit -s unlimited' (as the old condor job did): MadEvent's
    Fortran executables (gensym, madevent) crash on the default 8 MB stack."""
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    if soft != hard:
        resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))


def execute_madgraph(card_path, prefix):
    """Run one MG5 card from WORK_DIR; return the log path."""
    log_path = os.path.join(LOG_DIR, "madgraph_" + os.path.basename(card_path).replace(".txt", ".log"))
    print(f"[START] {os.path.basename(card_path)} -> {log_path}", flush=True)
    stop = threading.Event()
    tailer = threading.Thread(target=_tail_log, args=(log_path, stop, prefix), daemon=True)
    tailer.start()
    # The conda env can ship its own Pythia8 and set PYTHIA8DATA to it; MadGraph builds
    # its own (different version) and Pythia aborts on the version mismatch.
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHIA8DATA", "PYTHIA8")}
    with open(log_path, "w") as log_file:
        result = subprocess.run([MG5_PYTHON, MG5_EXE, "-f", card_path], cwd=WORK_DIR,
                                stdout=log_file, stderr=subprocess.STDOUT,
                                preexec_fn=_raise_stack_limit, env=env)
    stop.set()
    tailer.join(timeout=2)
    if result.returncode != 0:
        raise RuntimeError(f"MG5 exited {result.returncode}, check {log_path}")
    failure = scan_log_for_errors(log_path)
    if failure:
        raise RuntimeError(f"MG5 reported an error, check {log_path}\n    {failure}")
    print(f"  [DONE] {os.path.basename(card_path)}", flush=True)
    return log_path


# ------------------------------------------------------------------ cards
DELPHES_CARD = ""   # resolved path of the Delphes card, set in main()


# Stable neutral DM states of the signal models (DMSimpt: Xs Xm Xv Xc Xd Xw; 2HDM+a: Xd = 52).
# Delphes cards only know the SM/SUSY invisibles; any other PDG falls under the calorimeter
# default, so the DM would be clustered into jets instead of giving MET.
INVISIBLE_DM_PDGS = (51, 52, 53, 56, 57, 58)


# Branches the analysis reads; --slim-delphes drops all other TreeWriter outputs (generator
# particles, tracks, towers, energy-flow objects, fat jets), which make up most of the file.
SLIM_BRANCHES = ("Jet", "Electron", "Muon", "Photon", "MissingET", "ScalarHT", "GenJet", "GenMissingET")


def resolve_delphes_card(spec, slim=False):
    """'CMS' -> MG5's delphes_card_CMS.dat; any other value is a file path.
    Returns a copy in WORK_DIR with the DM PDGs made invisible and, with slim, only the
    SLIM_BRANCHES written out."""
    if os.path.isfile(spec):
        src = os.path.abspath(spec)
    else:
        mg5_root = os.path.dirname(os.path.dirname(MG5_EXE))
        src = os.path.join(mg5_root, "Template", "Common", "Cards", f"delphes_card_{spec}.dat")
        if not os.path.isfile(src):
            sys.exit(f"Error: Delphes card '{spec}' not found (looked for a file and for {src}).")
    with open(src) as f:
        text = make_dm_invisible(f.read(), src)
    suffix = "_invisDM"
    if slim:
        text = slim_tree_writer(text, src)
        suffix += "_slim"
    base = os.path.splitext(os.path.basename(src))[0]
    dest = os.path.join(WORK_DIR, f"{base}{suffix}.dat")
    with open(dest, "w") as f:
        f.write(text)
    return dest


def slim_tree_writer(text, src):
    """Comment out the TreeWriter branches not in SLIM_BRANCHES."""
    def slim(m):
        out = []
        for line in m.group(0).split("\n"):
            b = re.match(r"\s*add Branch\s+\S+\s+(\S+)\s+\S+", line)
            out.append(f"# (slim) {line.strip()}" if b and b.group(1) not in SLIM_BRANCHES else line)
        return "\n".join(out)
    text, n = re.subn(r"module TreeWriter \S+ \{.*?\n\}", slim, text, flags=re.S)
    if n == 0:
        sys.exit(f"Error: no TreeWriter module in {src}; cannot slim the Delphes output.")
    return text


def make_dm_invisible(text, src):
    """Zero calorimeter response for the DM PDGs (next to the existing neutralino entry)
    and add them to NeutrinoFilter, which keeps invisibles out of the generator-level jets."""
    pdgs =[p for p in INVISIBLE_DM_PDGS if not re.search(rf"EnergyFraction\s*\{{{p}\}}", text)]

    def add_fractions(m):
        indent = m.group(1)
        return m.group(0) + "".join(f"\n{indent}add EnergyFraction {{{p}}} {{0.0}}" for p in pdgs)
    text, n_cal = re.subn(r"^([ \t]*)add EnergyFraction \{1000022\} \{[^}]*\}", add_fractions, text, flags=re.M)
    if n_cal == 0:
        sys.exit(f"Error: {src} has no 'add EnergyFraction {{1000022}}' line; cannot mark the DM "
                 f"PDGs {INVISIBLE_DM_PDGS} invisible. Add them to its calorimeter modules by hand.")

    def add_filter(m):
        block = m.group(0)
        if pdgs:
            last = list(re.finditer(r"^([ \t]*)add PdgCode \{-?\d+\}[^\n]*$", block, flags=re.M))[-1]
            extra = "".join(f"\n{last.group(1)}add PdgCode {{{s}{p}}}" for p in pdgs for s in ("", "-"))
            block = block[:last.end()] + extra + block[last.end():]
        return block
    text = re.sub(r"module PdgCodeFilter NeutrinoFilter \{.*?\n\}", add_filter, text, flags=re.S)
    return text


NB_CORE = 0   # cores per MadGraph instance with --jobs > 1 (0: MadGraph default, all cores)


def launch_header(run_name, process_dir, dir_exists):
    """Launch block; the card path line makes MG5 copy it into Cards/."""
    # 'launch' takes no core option; the session setting is passed on to the run.
    # --no_save keeps it out of MadGraph's input/mg5_configuration.txt.
    cores = f"set run_mode 2 --no_save\nset nb_core {NB_CORE} --no_save\n" if NB_CORE else ""
    launch = cores + (f"launch {process_dir} --name={run_name}\n" if dir_exists
                      else f"launch --name={run_name}\n")
    card = f"{DELPHES_CARD}\n" if DELPHES_CARD else ""
    # analysis=OFF: the ExRoot LHE->ROOT copy is not needed (Delphes writes its own ROOT file)
    # and, for large runs, MadGraph's create_root_file races with its background gzip of the
    # LHE file and deletes both copies ("No event file corresponding to <run> run").
    return launch + "shower=Pythia8\ndetector=Delphes\nanalysis=OFF\n" + card


def render_template(name, values):
    """Read genScripts/<name> and replace every __KEY__ with values[KEY]."""
    path = os.path.join(TEMPLATE_DIR, name)
    try:
        with open(path) as f:
            text = f.read()
    except OSError as e:
        sys.exit(f"Error: cannot read launch card template {path}: {e}")
    for key, val in values.items():
        text = text.replace(f"__{key}__", str(val))
    left = sorted(set(re.findall(r"__[A-Z0-9_]+__", text)))
    if left:
        sys.exit(f"Error: unfilled placeholders {left} in {path}")
    return text if text.endswith("\n") else text + "\n"


def load_process_card(name, values):
    """Render a .mg5 card; return (text, process directory from its 'output' line)."""
    text = render_template(name, values)
    m = re.search(r"^output\s+(\S+)", text, flags=re.M)
    if not m:
        sys.exit(f"Error: no 'output <dir>' line in {os.path.join(TEMPLATE_DIR, name)}")
    return text, m.group(1)


def tchannel_settings(dm, mMED, mDM, lam, nevents, seed):
    return render_template(TCHANNEL_PROCS[dm]["madevent"], dict(
        NEVENTS=nevents, ISEED=seed, MMED=mMED, MDM=mDM, LAM=lam))


def hdma_settings(mA, ma, nevents, seed):
    return render_template(HDMA_PROC["madevent"], dict(
        NEVENTS=nevents, ISEED=seed, MCA=mA, MSA=ma))


def background_settings(bkg, nevents, seed, cuts):
    """Shared background settings; with cuts, genScripts/<bkg>.cuts.madevent (generator-level
    cuts, 'set run_card ...' lines) is inserted before the final 'done'."""
    text = render_template("background.madevent", dict(NEVENTS=nevents, ISEED=seed))
    if cuts:
        cut_card = f"{bkg}.cuts.madevent"
        if not os.path.isfile(os.path.join(TEMPLATE_DIR, cut_card)):
            sys.exit(f"Error: --bkg-cuts needs {os.path.join(TEMPLATE_DIR, cut_card)}")
        lines = [l for l in render_template(cut_card, {}).splitlines()
                 if l.strip() and not l.strip().startswith("#") and l.strip() != "done"]
        text = re.sub(r"^done\s*$", "\n".join(lines) + "\ndone", text, count=1, flags=re.M)
    return text


def reset_cards(process_dir):
    """Restore param_card and run_card from MadGraph's defaults before a relaunch, so
    settings of an earlier launch (masses, generator cuts) cannot carry over."""
    cards = os.path.join(WORK_DIR, process_dir, "Cards")
    for name in ("param_card", "run_card"):
        default, active = os.path.join(cards, f"{name}_default.dat"), os.path.join(cards, f"{name}.dat")
        if os.path.exists(default):
            shutil.copy2(default, active)


# ------------------------------------------------------------------ run one point
_xs_lock = threading.Lock()


def merge_root(files, dest):
    """Merge Delphes ROOT files into dest with ROOT's hadd."""
    hadd = shutil.which("hadd")
    if hadd is None:
        raise RuntimeError("hadd (ROOT) not found in PATH, cannot merge several runs")
    r = subprocess.run([hadd, "-f", dest, *files], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"hadd failed: {r.stderr.strip()[-300:]}")


def clear_process_libs(process_path):
    """Delete the built libraries in <process>/lib before relaunching in an existing dir.

    A relaunch rewrites Cards/run_card.dat, but MadGraph rewrites Source/run_card.inc only
    when its content changes, so lib/libgeneric.a stays older than the card. The parallel
    'make -j' of each SubProcesses/P* dir then re-enters Source/ from several targets at
    once; they rebuild lib/libmodel.a concurrently and gensym fails to link ("undefined
    reference to setpara_"). With the libraries removed, MadGraph's serial Source compile
    (run before the subprocesses) rebuilds them newer than the cards, so the subprocess
    makes find them up to date. Only build products are removed; no MadGraph file is edited."""
    lib_dir = os.path.join(process_path, "lib")
    for path in glob.glob(os.path.join(lib_dir, "lib*.a")):
        os.remove(path)


def launch_with_retry(card_path, tag, run_dir):
    """Run a launch card; retry once if MadGraph still hit a compilation error."""
    process_path = os.path.dirname(os.path.dirname(run_dir))
    for attempt in (1, 2):
        # A leftover run dir of the same name (earlier failed attempt) makes MadGraph
        # collide with its own files, e.g. "unweighted_events.root already exists".
        shutil.rmtree(run_dir, ignore_errors=True)
        if os.path.isfile(os.path.join(process_path, "SubProcesses", "proc_characteristics")):
            clear_process_libs(process_path)
        try:
            return execute_madgraph(card_path, tag)
        except RuntimeError:
            log_path = os.path.join(LOG_DIR, "madgraph_" + os.path.basename(card_path).replace(".txt", ".log"))
            with open(log_path, errors="ignore") as f:
                compile_error = "A compilation Error occurs" in f.read()
            if attempt == 2 or not compile_error:
                raise
            print(f"  [{tag}] MadGraph compilation race, retrying {os.path.basename(card_path)}", flush=True)


def run_point(*, tag, proc_card, process_dir, make_settings, seed, run_name, dest_file, xs_key, args):
    """Run args.nruns launches of one process (different seeds), merge their Delphes ROOT
    files into one file in the output directory and record the mean cross-section."""
    outdir = args.outdir_for(tag)
    dest = os.path.join(outdir, dest_file)
    if os.path.exists(dest) and not args.force:
        print(f"  [{tag}] skip {run_name}: {dest_file} exists (use --force)", flush=True)
        return
    nruns = max(1, args.bkg_nruns if tag in BACKGROUNDS else args.nruns)
    proc_path = os.path.join(WORK_DIR, process_dir)
    # A process dir is usable only if MadGraph finished writing it; a half-built one
    # (e.g. an earlier run died at 'output') is wiped and regenerated.
    dir_exists = os.path.isfile(os.path.join(proc_path, "SubProcesses", "proc_characteristics"))
    if os.path.isdir(proc_path) and not dir_exists and not args.dry_run:
        print(f"  [{tag}] removing incomplete process dir {process_dir}", flush=True)
        shutil.rmtree(proc_path)

    tmp_files, xss = [], []
    try:
        for k in range(nruns):
            rn = run_name if nruns == 1 else f"{run_name}_r{k + 1}"
            if (k > 0 or dir_exists) and not args.dry_run:
                reset_cards(process_dir)
            text = proc_card if (k == 0 and not dir_exists) else ""
            text += launch_header(rn, process_dir, True) + make_settings(seed + 1000 * k)
            card_path = os.path.join(CARD_DIR, f"point_{tag}_{rn}.txt")
            with open(card_path, "w") as f:
                f.write(text)
            if args.dry_run:
                print(f"--- {card_path}  (run {k + 1}/{nruns})\n{text}")
                continue

            log = launch_with_retry(card_path, tag, os.path.join(WORK_DIR, process_dir, "Events", rn))
            # The tag number is not fixed (the shower can open tag_2), so take the newest.
            run_dir = os.path.join(WORK_DIR, process_dir, "Events", rn)
            found = sorted(glob.glob(os.path.join(run_dir, "tag_*_delphes_events.root")),
                           key=lambda p: int(re.search(r"tag_(\d+)_", os.path.basename(p)).group(1)))
            src = found[-1] if found else os.path.join(run_dir, "tag_*_delphes_events.root")
            if not found:
                with open(log, errors="ignore") as f:
                    off = "Not Avail." in f.read()
                hint = (" (MadGraph switched shower/detector off: Pythia8 or Delphes not usable, "
                        "see madgraph_install_tools.log)") if off else ""
                raise RuntimeError(f"no Delphes output at {src}{hint}")
            os.makedirs(outdir, exist_ok=True)
            tmp = os.path.join(outdir, f".run{k + 1}.{dest_file}")
            shutil.copy2(src, tmp)
            tmp_files.append(tmp)
            xs = parse_cross_section(log)
            if xs is not None:
                xss.append(xs)
            print(f"  [{tag}] {rn}: xsec = {xs} pb", flush=True)
            if not args.keep_events:
                shutil.rmtree(os.path.join(WORK_DIR, process_dir, "Events", rn), ignore_errors=True)
        if args.dry_run:
            return
        if len(tmp_files) == 1:
            os.replace(tmp_files[0], dest)
        else:
            merge_root(tmp_files, dest)
    finally:
        for t in tmp_files:
            if os.path.exists(t):
                os.remove(t)

    print(f"  [{tag}] {run_name}: {nruns} run(s) -> {dest}", flush=True)
    record_sample(tag, xs_key, dest)
    if xss:
        record_xsec(args, tag, xs_key, sum(xss) / len(xss))


def record_sample(tag, key, dest):
    """Add the new output file to config/samples_config.yaml (thread-safe)."""
    if _REPO_DIR not in sys.path:
        sys.path.insert(0, _REPO_DIR)
    from utils import samples_config
    with _xs_lock:
        if tag in ("2HDMA", "S3M", "S3D"):
            samples_config.add_signal(*key, dest)
        else:
            samples_config.add_background(key, dest)


# ------------------------------------------------------------------ cross-section yaml
def record_xsec(args, tag, key, xs):
    """Merge one cross-section into the matching config yaml (thread-safe)."""
    with _xs_lock:
        if tag == "2HDMA":
            path = os.path.join(_REPO_DIR, "config", "signal_cross_sections.yaml")
            table = _read_nested(path, "signals")
            mA, ma = key
            table.setdefault(mA, {})[ma] = xs
            _write_nested(path, "signals", table, "Cross-section table for signal processes (2HDM+a). Units: pb.\n"
                          "# Format: mA, then ma in GeV.")
        elif tag in ("S3M", "S3D"):
            path = os.path.join(_REPO_DIR, "config", f"tchannel_{tchannel_tag(args)}_cross_sections.yaml")
            table = _read_nested(path, "signals")
            mMED, mDM = key
            table.setdefault(mMED, {})[mDM] = xs
            _write_nested(path, "signals", table, f"t-channel DM signal cross-sections, {args.dm_type.upper()}, "
                          f"lambda = {args.lam} (same layout as signal_cross_sections.yaml). Units: pb.\n"
                          "# Keys: mMED (as mA), then mDM (as ma).")
        else:
            # With generator cuts these are the cross-sections after the cuts: separate file.
            name = "background_cuts_cross_sections.yaml" if args.bkg_cuts else "background_cross_sections.yaml"
            path = os.path.join(_REPO_DIR, "config", name)
            import yaml
            data = {}
            if os.path.exists(path):
                with open(path) as f:
                    data = yaml.safe_load(f) or {}
            data.setdefault("backgrounds", {})[key] = xs
            with open(path, "w") as f:
                f.write("# Cross-section table for background processes\n# Units: pb (picobarns)\n"
                        "# Format: process_name: cross_section_value\n\n")
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def _read_nested(path, top):
    import yaml
    if os.path.exists(path):
        with open(path) as f:
            return (yaml.safe_load(f) or {}).get(top, {}) or {}
    return {}


def _write_nested(path, top, table, header):
    import yaml
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    data[top] = {k: table[k] for k in sorted(table)}
    with open(path, "w") as f:
        f.write(f"# {header}\n\n")
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


# ------------------------------------------------------------------ drivers
def fmt(x):
    return str(x).replace(".", "p")


def tchannel_tag(args):
    """DM type and coupling, e.g. 's3m_lam2p2': part of the t-channel file names, so
    samples with different settings never overwrite or get mistaken for each other."""
    return f"{args.dm_type}_lam{fmt(args.lam)}"


def signal_file_name(args, tag, A, a):
    if tag == "2HDMA":
        return f"sig_bbdm_mA{A}_ma{a}_delphes_events.root"
    return f"sig_bbdm_mA{A}_ma{a}_{tchannel_tag(args)}_delphes_events.root"


def parse_points(spec):
    """'500:50,1000:100' -> [(500, 50), (1000, 100)]"""
    pts = []
    for item in spec.split(","):
        parts = item.strip().split(":")
        if len(parts) != 2:
            sys.exit(f"Error: bad --points entry '{item}', expected A:a (e.g. 500:100)")
        pts.append((int(parts[0]), int(parts[1])))
    return pts


def select_points(args, model, single, grid):
    """--points wins, then --npoints (first N defaults), then --mode single/grid."""
    if args.points:
        return parse_points(args.points)
    if args.npoints:
        defaults = DEFAULT_POINTS[model]
        if args.npoints > len(defaults):
            sys.exit(f"Error: --npoints {args.npoints} > {len(defaults)} built-in points; use --points")
        return defaults[:args.npoints]
    return grid if args.mode == "grid" else [single]


# A task is one sample (signal point or background): its runs are sequential in one process
# dir. Tasks sharing a process card ("group") run concurrently in separate copies of the
# process dir, built by MadGraph from the same card with only the 'output' name changed.
def signal_tasks(args):
    """(tag, group card name, card values, [task dicts]) for the selected signal points."""
    if args.signal_model == "tchannel":
        dm = args.dm_type
        tag, card, values = dm.upper(), TCHANNEL_PROCS[dm]["mg5"], dict(UFO_TCHANNEL=UFO_TCHANNEL)
        grid = [(a, b) for a in T_MED_VALS for b in T_DM_VALS if a > b]
        points = select_points(args, "tchannel", (args.mMED, args.mDM), grid)
    else:
        tag, card, values = "2HDMA", HDMA_PROC["mg5"], dict(UFO_2HDMA_T2=UFO_2HDMA)
        grid = [(A, a) for A in H_MASS_A for a in H_MASS_a if a < A]
        points = select_points(args, "2hdma", (args.mA, args.ma), grid)
    tasks = []
    for i, (A, a) in enumerate(points, 1):
        if a >= A:
            raise ValueError(f"Kinematic violation: {a} >= {A} for point {A}:{a}")
        if tag == "2HDMA":
            name = f"mA{A}_ma{a}"
            settings = lambda s, A=A, a=a: hdma_settings(A, a, args.nevents, s)
        else:
            name = f"mMed{A}_mDM{a}_lam{fmt(args.lam)}"
            settings = lambda s, A=A, a=a: tchannel_settings(args.dm_type, A, a, args.lam, args.nevents, s)
        tasks.append(dict(tag=tag, name=name, make_settings=settings, seed=args.seed + i,
                          dest_file=signal_file_name(args, tag, A, a), xs_key=(A, a)))
    return tag, card, values, tasks


def background_task(bkg, args):
    return dict(tag=bkg, name=f"{bkg}_run",
                make_settings=lambda s: background_settings(bkg, args.nevents, s, args.bkg_cuts),
                seed=args.seed, dest_file=f"{bkg}_delphes_events.root", xs_key=bkg)


def with_output_dir(proc_card, process_dir):
    return re.sub(r"^output\s+\S+", f"output {process_dir}", proc_card, count=1, flags=re.M)


def prepare_process_dir(tag, proc_card, process_dir):
    """Build a process dir ('output' only) unless a complete one exists. Called serially:
    concurrent 'output' commands write the same restricted-model files and model caches."""
    proc_path = os.path.join(WORK_DIR, process_dir)
    if os.path.isfile(os.path.join(proc_path, "SubProcesses", "proc_characteristics")):
        return
    if os.path.isdir(proc_path):
        print(f"  [{tag}] removing incomplete process dir {process_dir}", flush=True)
        shutil.rmtree(proc_path)
    card_path = os.path.join(CARD_DIR, f"output_{process_dir}.txt")
    with open(card_path, "w") as f:
        f.write(proc_card)
    print(f"  [{tag}] building process dir {process_dir}", flush=True)
    execute_madgraph(card_path, tag)


def run_task(task, proc_card, process_dir, args, failed):
    try:
        print(f"  [{task['tag']}] start {task['name']} in {process_dir}", flush=True)
        run_point(tag=task["tag"], proc_card=proc_card, process_dir=process_dir,
                  make_settings=task["make_settings"], seed=task["seed"], run_name=task["name"],
                  dest_file=task["dest_file"], xs_key=task["xs_key"], args=args)
    except Exception as e:
        print(f"  [{task['tag']}] SKIP {task['name']}: {e}", flush=True)
        failed.append(f"{task['tag']}:{task['name']}")


def main():
    p = argparse.ArgumentParser(description="Self-contained MG5+Pythia8+Delphes generation for bbMET.")
    p.add_argument("--proc", default="signal",
                   help="signal | bkg | all | comma list of backgrounds "
                        f"({', '.join(BACKGROUNDS)})")
    p.add_argument("--mode", choices=["single", "grid"], default="single")
    p.add_argument("--signal-model", choices=["tchannel", "2hdma"], default="tchannel")
    p.add_argument("--dm-type", choices=["s3m", "s3d"], default="s3m",
                   help="t-channel only: Majorana S3M or Dirac S3D (one per call)")
    p.add_argument("--mMED", type=int, default=1000, help="t-channel mediator mass (single)")
    p.add_argument("--mDM", type=int, default=100, help="t-channel DM mass (single)")
    p.add_argument("--lam", type=float, default=2.2, help="t-channel DMS3D 3 3 coupling, fixed per call")
    p.add_argument("--points", default="", help="explicit points 'A:a,A:a,...' (mMED:mDM or mA:ma); overrides --mode")
    p.add_argument("--npoints", type=int, default=0, help="use the first N built-in default points")
    p.add_argument("--nruns", type=int, default=1,
                   help="runs per signal point with different seeds, merged with hadd into one file")
    p.add_argument("--bkg-nruns", type=int, default=None,
                   help="runs per background, merged with hadd (default: --nruns)")
    p.add_argument("--jobs", type=int, default=1,
                   help="samples generated at the same time; the CPU cores are split between them")
    p.add_argument("--cores", type=int, default=0,
                   help="total CPU cores to use, split between the --jobs samples (default: all)")
    p.add_argument("--mA", type=int, default=500, help="2HDM+a heavy pseudoscalar mass (single)")
    p.add_argument("--ma", type=int, default=100, help="2HDM+a light pseudoscalar mass (single)")
    p.add_argument("--nevents", type=int, default=10000)
    p.add_argument("--seed", type=int, default=1000, help="base run_card iseed")
    p.add_argument("--data-dir", default=DATA_DIR,
                   help="base dir for large files: work dir and Delphes output (default genevents/)")
    p.add_argument("--work-dir", default=None,
                   help="MadGraph process dirs, cards and logs (default <data-dir>/work)")
    p.add_argument("--signal-outdir", default=None, help="signal Delphes files (default <data-dir>/DelphesSignal)")
    p.add_argument("--background-outdir", default=None,
                   help="background Delphes files (default <data-dir>/DelphesBackground)")
    p.add_argument("--delphes-card", default="CMS",
                   help="Delphes card: 'CMS' (MG5 delphes_card_CMS.dat, default), another MG5 suffix "
                        "(ATLAS, CMS_PhaseII, ...) or a path to a .dat file")
    p.add_argument("--keep-events", action="store_true", help="keep MG5 Events/<run> dirs (large)")
    p.add_argument("--force", action="store_true", help="regenerate even if output ROOT exists")
    p.add_argument("--slim-delphes", action="store_true",
                   help="write only the Delphes branches the analysis reads (much smaller files)")
    p.add_argument("--bkg-cuts", action="store_true",
                   help="apply the generator-level cuts in genScripts/<bkg>.cuts.madevent; output goes to "
                        "<data-dir>/DelphesBackground_bkgcuts, cross-sections to "
                        "config/background_cuts_cross_sections.yaml")
    p.add_argument("--dry-run", action="store_true", help="write and print cards, do not run MG5")
    args = p.parse_args()
    global WORK_DIR, CARD_DIR, LOG_DIR
    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    WORK_DIR = os.path.abspath(os.path.expanduser(args.work_dir or os.path.join(data_dir, "work")))
    CARD_DIR = os.path.join(WORK_DIR, "textFiles")
    LOG_DIR = os.path.join(WORK_DIR, "logFiles")
    args.signal_outdir = os.path.abspath(args.signal_outdir or os.path.join(data_dir, "DelphesSignal"))
    bkg_default = "DelphesBackground_bkgcuts" if args.bkg_cuts else "DelphesBackground"
    args.background_outdir = os.path.abspath(args.background_outdir or os.path.join(data_dir, bkg_default))
    print(f"[env] work dir: {WORK_DIR}", flush=True)
    print(f"[env] output  : {args.signal_outdir} , {args.background_outdir}", flush=True)
    args.outdir_for = lambda tag: args.background_outdir if tag in BACKGROUNDS else args.signal_outdir
    if args.bkg_nruns is None:
        args.bkg_nruns = args.nruns
    args.jobs = max(1, args.jobs)

    wanted = [x.strip() for x in args.proc.split(",")]
    do_signal = "signal" in wanted or "all" in wanted
    bkgs = list(BACKGROUNDS) if ("bkg" in wanted or "all" in wanted) else [w for w in wanted if w in BACKGROUNDS]
    unknown = [w for w in wanted if w not in BACKGROUNDS and w not in ("signal", "bkg", "all")]
    if unknown:
        sys.exit(f"Unknown --proc entries: {unknown}")

    os.makedirs(CARD_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    if not args.dry_run:
        check_environment()
        ensure_tools()
        global DELPHES_CARD
        DELPHES_CARD = resolve_delphes_card(args.delphes_card, slim=args.slim_delphes)
        print(f"[env] Delphes card: {DELPHES_CARD}", flush=True)
    else:
        print("[dry-run] skipping MadGraph install check", flush=True)

    # Groups of tasks sharing one process card: (tag, card text, base process dir, tasks).
    groups = []
    if do_signal:
        tag, card, values, tasks = signal_tasks(args)
        proc_card, base_dir = load_process_card(card, values)
        groups.append((tag, proc_card, base_dir, tasks))
    for b in bkgs:
        proc_card, base_dir = load_process_card(f"{b}.mg5", {})
        groups.append((b, proc_card, base_dir, [background_task(b, args)]))

    def pending(task):
        return args.force or not os.path.exists(os.path.join(args.outdir_for(task["tag"]), task["dest_file"]))

    for tag, _, _, tasks in groups:
        for t in tasks:
            if not pending(t):
                print(f"  [{tag}] skip {t['name']}: {t['dest_file']} exists (use --force)", flush=True)
    groups = [(tag, card, base, [t for t in tasks if pending(t)]) for tag, card, base, tasks in groups]
    groups = [g for g in groups if g[3]]

    # One process dir per concurrently running task of a group: base, base_j1, base_j2, ...
    slots = {}
    for tag, card, base, tasks in groups:
        n = min(args.jobs, len(tasks))
        slots[base] = [base] + [f"{base}_j{i}" for i in range(1, n)]

    global NB_CORE
    if args.jobs > 1 or args.cores:
        total = args.cores or (os.cpu_count() or 1)
        NB_CORE = max(1, total // args.jobs)
        print(f"[env] {args.jobs} sample(s) at a time, {NB_CORE} core(s) each ({total} in total)", flush=True)

    failed = []
    if not args.dry_run:
        try:
            for tag, card, base, _ in groups:
                for d in slots[base]:
                    prepare_process_dir(tag, with_output_dir(card, d), d)
        except RuntimeError as e:
            sys.exit(f"Error: building a process dir failed: {e}")

    import queue
    from concurrent.futures import ThreadPoolExecutor
    free = {base: queue.Queue() for base in slots}
    for base, dirs in slots.items():
        for d in dirs:
            free[base].put(d)

    def worker(task, card, base):
        d = free[base].get()
        try:
            run_task(task, with_output_dir(card, d), d, args, failed)
        finally:
            free[base].put(d)

    with ThreadPoolExecutor(max_workers=1 if args.dry_run else args.jobs) as pool:
        for tag, card, base, tasks in groups:
            for task in tasks:
                pool.submit(worker, task, card, base)

    if failed:
        print(f"\nFAILED ({len(failed)}): {', '.join(failed)}", flush=True)
        sys.exit(1)
    print("\nAll requested generation finished.", flush=True)


if __name__ == "__main__":
    main()
