#!/usr/bin/env python3
"""Self-contained MadGraph5 + Pythia8 + Delphes generation for bbMET.

Cards live in genScripts/ (<name>.mg5 process, <name>.madevent settings); MadGraph is
installed into this directory on first use.

Output files (inputs of run_analysis_from_dirs.py), one merged file per sample:
  signal      <signal-outdir>/sig_bbdm_mA{A}_ma{a}_delphes_events.root
              A, a = mA, ma (2hdma) or mMED, mDM (tchannel)
  background  <background-outdir>/<name>_delphes_events.root

Usage: python mg5_Generate_Run_bbMET.py [options]
  --proc P              signal | bkg | all | comma list of backgrounds (default signal)
  --signal-model M      tchannel (S3M/S3D) | 2hdma (type II)     (default tchannel)
  --dm-type T           t-channel only: s3m | s3d, one per call  (default s3m)
  --lam L               t-channel coupling DMS3D 3 3, fixed per call (default 1.0)
  --points A:a,A:a,...  explicit points (mMED:mDM or mA:ma)
  --npoints N           first N built-in default points (up to 8)
  --mode single|grid    no --points/--npoints: one point, or the full grid
  --mMED/--mDM          t-channel point for single mode          (1000, 100)
  --mA/--ma             2HDM+a point for single mode             (500, 100)
  --nevents N           events per run                           (default 10000)
  --nruns N             runs per sample, other seeds, merged with hadd (default 1)
  --seed S              base iseed; point index and run number are added (default 1000)
  --signal-outdir D     (default genevents/DelphesSignal)
  --background-outdir D (default genevents/DelphesBackground)
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

Environment: MG5_EXE, MG5_PYTHON (<= 3.12), MG5_TARBALL_URL, UFO_TCHANNEL, UFO_2HDMA.
Cross-sections (mean over runs) go to config/{signal,tchannel_signal,background}_cross_sections.yaml.
"""

import argparse
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

WORK_DIR = os.path.join(_SCRIPT_DIR, "work")
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


def ensure_tools():
    """Install Pythia8 and Delphes into MadGraph if missing."""
    mg5_root = os.path.dirname(os.path.dirname(MG5_EXE))
    restore_mg5_autoupdate_default(mg5_root)
    heptools = os.path.join(mg5_root, "HEPTools")
    need = []
    if not os.path.isdir(os.path.join(heptools, "pythia8")):
        need.append("install pythia8")
    if not delphes_installed(mg5_root):
        need.append("install Delphes")
    if not os.path.isdir(os.path.join(mg5_root, "ExRootAnalysis")):
        need.append("install ExRootAnalysis")
    if not need:
        return
    card = os.path.join(CARD_DIR, "install_tools.txt")
    with open(card, "w") as f:
        f.write("\n".join(need) + "\n")
    print(f"[env] installing: {', '.join(need)} (needs ROOT for Delphes)", flush=True)
    execute_madgraph(card, "INSTALL")
    if "install Delphes" in need and not delphes_installed(mg5_root):
        sys.exit("Error: Delphes install failed. Ensure ROOT (root-config) is in PATH, "
                 f"see {os.path.join(LOG_DIR, 'madgraph_install_tools.log')}")


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


def resolve_delphes_card(spec):
    """'CMS' -> MG5's delphes_card_CMS.dat; any other value is a file path."""
    if os.path.isfile(spec):
        return os.path.abspath(spec)
    mg5_root = os.path.dirname(os.path.dirname(MG5_EXE))
    named = os.path.join(mg5_root, "Template", "Common", "Cards", f"delphes_card_{spec}.dat")
    if os.path.isfile(named):
        return named
    sys.exit(f"Error: Delphes card '{spec}' not found (looked for a file and for {named}).")


def launch_header(run_name, process_dir, dir_exists):
    """Launch block; the card path line makes MG5 copy it into Cards/."""
    launch = f"launch {process_dir} --name={run_name}\n" if dir_exists else f"launch --name={run_name}\n"
    card = f"{DELPHES_CARD}\n" if DELPHES_CARD else ""
    return launch + "shower=Pythia8\ndetector=Delphes\nanalysis=ExRoot\n" + card


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


def background_settings(nevents, seed):
    return render_template("background.madevent", dict(NEVENTS=nevents, ISEED=seed))


def reset_param_card(process_dir):
    cards = os.path.join(WORK_DIR, process_dir, "Cards")
    default, active = os.path.join(cards, "param_card_default.dat"), os.path.join(cards, "param_card.dat")
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


def run_point(*, tag, proc_card, process_dir, make_settings, seed, run_name, dest_file, xs_key, args):
    """Run args.nruns launches of one process (different seeds), merge their Delphes ROOT
    files into one file in the output directory and record the mean cross-section."""
    outdir = args.outdir_for(tag)
    dest = os.path.join(outdir, dest_file)
    if os.path.exists(dest) and not args.force:
        print(f"  [{tag}] skip {run_name}: {dest_file} exists (use --force)", flush=True)
        return
    nruns = max(1, args.nruns)
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
                reset_param_card(process_dir)
            text = proc_card if (k == 0 and not dir_exists) else ""
            text += launch_header(rn, process_dir, True) + make_settings(seed + 1000 * k)
            card_path = os.path.join(CARD_DIR, f"point_{tag}_{rn}.txt")
            with open(card_path, "w") as f:
                f.write(text)
            if args.dry_run:
                print(f"--- {card_path}  (run {k + 1}/{nruns})\n{text}")
                continue

            log = execute_madgraph(card_path, tag)
            src = os.path.join(WORK_DIR, process_dir, "Events", rn, "tag_1_delphes_events.root")
            if not os.path.exists(src):
                raise RuntimeError(f"no Delphes output at {src}")
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
    if xss:
        record_xsec(args, tag, xs_key, sum(xss) / len(xss))


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
            path = os.path.join(_REPO_DIR, "config", "tchannel_signal_cross_sections.yaml")
            table = _read_nested(path, "signals")
            mMED, mDM = key
            table.setdefault(mMED, {})[mDM] = xs
            _write_nested(path, "signals", table, "t-channel DM signal cross-sections (same layout as "
                          "signal_cross_sections.yaml). Units: pb.\n# Keys: mMED (as mA), then mDM (as ma). "
                          "One DM type / lambda set at a time.")
        else:
            path = os.path.join(_REPO_DIR, "config", "background_cross_sections.yaml")
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


def run_tchannel(args, failed):
    dm = args.dm_type
    cfg = TCHANNEL_PROCS[dm]
    tag = dm.upper()
    grid = [(a, b) for a in T_MED_VALS for b in T_DM_VALS if a > b]
    points = select_points(args, "tchannel", (args.mMED, args.mDM), grid)
    for i, (mMED, mDM) in enumerate(points, 1):
        if mMED <= mDM:
            raise ValueError(f"Kinematic violation: mMED ({mMED}) <= mDM ({mDM})")
        name = f"mMed{mMED}_mDM{mDM}_lam{fmt(args.lam)}"
        print(f"  [{tag}] point {i}/{len(points)}: {name}", flush=True)
        try:
            proc_card, process_dir = load_process_card(cfg["mg5"], dict(UFO_TCHANNEL=UFO_TCHANNEL))
            run_point(tag=tag, proc_card=proc_card, process_dir=process_dir,
                      make_settings=lambda s, a=mMED, b=mDM: tchannel_settings(dm, a, b, args.lam, args.nevents, s),
                      seed=args.seed + i, run_name=name,
                      dest_file=f"sig_bbdm_mA{mMED}_ma{mDM}_delphes_events.root",
                      xs_key=(mMED, mDM), args=args)
        except Exception as e:
            print(f"  [{tag}] SKIP {name}: {e}", flush=True)
            failed.append(f"{tag}:{name}")


def run_2hdma(args, failed):
    tag = "2HDMA"
    grid = [(A, a) for A in H_MASS_A for a in H_MASS_a if a < A]
    points = select_points(args, "2hdma", (args.mA, args.ma), grid)
    for i, (mA, ma) in enumerate(points, 1):
        if ma >= mA:
            raise ValueError(f"Kinematic violation: ma ({ma}) >= mA ({mA})")
        name = f"mA{mA}_ma{ma}"
        print(f"  [{tag}] point {i}/{len(points)}: {name}", flush=True)
        try:
            proc_card, process_dir = load_process_card(HDMA_PROC["mg5"], dict(UFO_2HDMA_T2=UFO_2HDMA))
            run_point(tag=tag, proc_card=proc_card, process_dir=process_dir,
                      make_settings=lambda s, a=mA, b=ma: hdma_settings(a, b, args.nevents, s),
                      seed=args.seed + i, run_name=name,
                      dest_file=f"sig_bbdm_mA{mA}_ma{ma}_delphes_events.root",
                      xs_key=(mA, ma), args=args)
        except Exception as e:
            print(f"  [{tag}] SKIP {name}: {e}", flush=True)
            failed.append(f"{tag}:{name}")


def run_background(bkg, args, failed):
    tag = bkg
    try:
        proc_card, process_dir = load_process_card(f"{bkg}.mg5", {})
        run_point(tag=tag, proc_card=proc_card, process_dir=process_dir,
                  make_settings=lambda s: background_settings(args.nevents, s),
                  seed=args.seed, run_name=f"{bkg}_run",
                  dest_file=f"{bkg}_delphes_events.root", xs_key=bkg, args=args)
    except Exception as e:
        print(f"  [{tag}] SKIP {bkg}: {e}", flush=True)
        failed.append(f"{tag}")


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
    p.add_argument("--lam", type=float, default=1.0, help="t-channel DMS3D 3 3 coupling, fixed per call")
    p.add_argument("--points", default="", help="explicit points 'A:a,A:a,...' (mMED:mDM or mA:ma); overrides --mode")
    p.add_argument("--npoints", type=int, default=0, help="use the first N built-in default points")
    p.add_argument("--nruns", type=int, default=1,
                   help="runs per sample with different seeds, merged with hadd into one file")
    p.add_argument("--mA", type=int, default=500, help="2HDM+a heavy pseudoscalar mass (single)")
    p.add_argument("--ma", type=int, default=100, help="2HDM+a light pseudoscalar mass (single)")
    p.add_argument("--nevents", type=int, default=10000)
    p.add_argument("--seed", type=int, default=1000, help="base run_card iseed")
    p.add_argument("--signal-outdir", default=os.path.join(_SCRIPT_DIR, "DelphesSignal"))
    p.add_argument("--background-outdir", default=os.path.join(_SCRIPT_DIR, "DelphesBackground"))
    p.add_argument("--delphes-card", default="CMS",
                   help="Delphes card: 'CMS' (MG5 delphes_card_CMS.dat, default), another MG5 suffix "
                        "(ATLAS, CMS_PhaseII, ...) or a path to a .dat file")
    p.add_argument("--keep-events", action="store_true", help="keep MG5 Events/<run> dirs (large)")
    p.add_argument("--force", action="store_true", help="regenerate even if output ROOT exists")
    p.add_argument("--dry-run", action="store_true", help="write and print cards, do not run MG5")
    args = p.parse_args()
    args.outdir_for = lambda tag: args.background_outdir if tag in BACKGROUNDS else args.signal_outdir

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
        DELPHES_CARD = resolve_delphes_card(args.delphes_card)
        print(f"[env] Delphes card: {DELPHES_CARD}", flush=True)
    else:
        print("[dry-run] skipping MadGraph install check", flush=True)

    failed = []
    jobs = []   # independent process dirs may run concurrently; each dir is sequential
    if do_signal:
        if args.signal_model == "tchannel":
            jobs.append(threading.Thread(target=run_tchannel, args=(args, failed), name=args.dm_type))
        else:
            jobs.append(threading.Thread(target=run_2hdma, args=(args, failed), name="2hdma"))
    if bkgs:
        def all_bkgs():
            for b in bkgs:
                run_background(b, args, failed)
        jobs.append(threading.Thread(target=all_bkgs, name="bkg"))

    for t in jobs:
        t.start()
    for t in jobs:
        t.join()

    if failed:
        print(f"\nFAILED ({len(failed)}): {', '.join(failed)}", flush=True)
        sys.exit(1)
    print("\nAll requested generation finished.", flush=True)


if __name__ == "__main__":
    main()
