"""Read and write config/samples_config.yaml, the list of Delphes files the analysis uses.

The file is generated, not edited by hand:
  - genevents/mg5_Generate_Run_bbMET.py adds an entry each time it places a file in its
    output directory;
  - scripts/run_analysis_from_dirs.py rewrites it from the signal/background directories
    it is given, so it always matches the files actually analysed.

Layout:
  backgrounds:
    ttbar: {file: /abs/path/ttbar_delphes_events.root}
  signals:
    mA500_ma50_s3m_lam2p2: {file: /abs/path/sig_bbdm_mA500_ma50_s3m_lam2p2_delphes_events.root,
                            mA: 500, ma: 50, tag: s3m_lam2p2}
  (the key is the file name without "sig_bbdm_" and "_delphes_events.root"; t-channel files
  carry the DM type and coupling as a tag, 2HDM+a files have none)

Cross-sections are kept separately in config/*_cross_sections.yaml.
"""
import os
import re

import yaml

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(REPO_DIR, "config", "samples_config.yaml")

_HEADER = ("# Generated file: list of Delphes ROOT files used by the analysis.\n"
           "# Updated by genevents/mg5_Generate_Run_bbMET.py when it writes an output file and\n"
           "# rewritten by scripts/run_analysis_from_dirs.py from the input directories.\n"
           "# Cross-sections are in config/*_cross_sections.yaml.\n\n")


def signal_key(file_path):
    """sig_bbdm_mA500_ma50_s3m_lam2p2_delphes_events.root -> mA500_ma50_s3m_lam2p2"""
    name = os.path.basename(str(file_path))
    return re.sub(r"^sig_bbdm_|_delphes_events\.root$", "", name)


def signal_entry(file_path, mA, ma):
    key = signal_key(file_path)
    entry = {"file": os.path.abspath(str(file_path)), "mA": int(mA), "ma": int(ma)}
    tag = re.sub(rf"^mA{mA}_ma{ma}_?", "", key)
    if tag:
        entry["tag"] = tag
    return key, entry


def load(path=DEFAULT_PATH):
    if os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    data.setdefault("backgrounds", {})
    data.setdefault("signals", {})
    for section in ("backgrounds", "signals"):
        if data[section] is None:
            data[section] = {}
    return data


def save(data, path=DEFAULT_PATH):
    """Write atomically, dropping entries whose file no longer exists."""
    for section in ("backgrounds", "signals"):
        data[section] = {k: v for k, v in (data.get(section) or {}).items()
                         if isinstance(v, dict) and v.get("file") and os.path.exists(v["file"])}
    data["signals"] = dict(sorted(data["signals"].items(),
                                  key=lambda kv: (kv[1].get("mA", 0), kv[1].get("ma", 0))))
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(_HEADER)
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    os.replace(tmp, path)


def add_background(name, file_path, path=DEFAULT_PATH):
    data = load(path)
    data["backgrounds"][name] = {"file": os.path.abspath(file_path)}
    save(data, path)


def add_signal(mA, ma, file_path, path=DEFAULT_PATH):
    data = load(path)
    key, entry = signal_entry(file_path, mA, ma)
    data["signals"][key] = entry
    save(data, path)


def replace_all(backgrounds, signals, path=DEFAULT_PATH):
    """backgrounds: {name: path}; signals: iterable of (path, mA, ma). Replaces both sections."""
    data = load(path)
    data["backgrounds"] = {n: {"file": os.path.abspath(str(p))} for n, p in backgrounds.items()}
    data["signals"] = dict(signal_entry(p, mA, ma) for p, mA, ma in signals)
    save(data, path)
