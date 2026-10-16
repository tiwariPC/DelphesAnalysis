#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
"""
Plot the expected significance (Asimov by default) of all signal regions and signal points.

This script reads datacards and plots the significance for each region.
"""

import argparse
import re
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for batch mode
import matplotlib.pyplot as plt
import mplhep as hep
from pathlib import Path
from typing import Dict, List, Tuple

# Use LaTeX only when a latex binary exists (matplotlib does not fall back on its own);
# otherwise mathtext, its built-in LaTeX-like rendering, is used.
plt.rcParams['text.usetex'] = shutil.which('latex') is not None
plt.rcParams['mathtext.fontset'] = 'stix'


def parse_datacard(datacard_file: Path) -> Dict:
    """Parse a datacard and extract signal and background yields."""
    with open(datacard_file, 'r') as f:
        content = f.read()

    info = {
        "bin_name": None,
        "processes": [],
        "rates": []
    }

    # Extract bin name
    bin_match = re.search(r'^bin\s+(\S+)', content, re.MULTILINE)
    if bin_match:
        info["bin_name"] = bin_match.group(1)

    # Extract process names
    proc_match = re.search(r'^process\s+(.+)', content, re.MULTILINE)
    if proc_match:
        processes = proc_match.group(1).split()
        info["processes"] = processes

    # Extract rates
    rate_match = re.search(r'^rate\s+(.+)', content, re.MULTILINE)
    if rate_match:
        rates = [float(x) for x in rate_match.group(1).split()]
        info["rates"] = rates

    return info


def calculate_significance(signal: float, background: float, formula: str = "asimov") -> float:
    """
    Expected significance of a counting experiment.

    formula "asimov": sqrt(2 ((s+b) ln(1+s/b) - s))  (the one of calculate_sensitivity.py)
    formula "simple": s / sqrt(s+b)
    (they agree for s << b; the simple one is not a Punzi figure of merit, which would need the
    signal efficiency).
    """
    if signal <= 0 or background < 0 or signal + background <= 0:
        return 0.0
    if formula == "simple":
        return signal / np.sqrt(signal + background)
    if background == 0:
        return float("inf")
    return float(np.sqrt(max(0.0, 2.0 * ((signal + background) * np.log(1.0 + signal / background) - signal))))


def find_all_datacards(output_dir: Path) -> List[Tuple[Path, str]]:
    """Find the per-region datacards written by process_regions.py.

    Current layout: <output>/datacards/<region>/datacard_mA{A}_ma{a}.txt (one per signal
    point), labelled "<region>/mA{A}_ma{a}". Older layout <output>/plots/<region>/datacard.txt
    is still read, labelled "<region>".
    """
    datacards = []

    datacards_dir = output_dir / "datacards"
    if datacards_dir.exists():
        for region_dir in sorted(p for p in datacards_dir.iterdir() if p.is_dir()):
            if not region_dir.name.startswith("sr"):
                continue  # significance is quoted for the signal regions only
            for datacard_file in sorted(region_dir.glob("datacard_*.txt")):
                point = datacard_file.stem[len("datacard_"):]
                datacards.append((datacard_file, f"{region_dir.name}/{point}"))

    plots_dir = output_dir / "plots"
    if not datacards and plots_dir.exists():
        for region_dir in sorted(p for p in plots_dir.iterdir() if p.is_dir()):
            datacard_file = region_dir / "datacard.txt"
            if datacard_file.exists():
                datacards.append((datacard_file, region_dir.name))

    return datacards


def plot_significance(datacard_infos: List[Dict], output_file: Path, lumi: float = 290.0,
                      formula: str = "asimov"):
    """
    Grouped bar plot of the significance per signal point (ordered by mediator mass): SR1b,
    SR2b and their combination (the regions are disjoint, so the significances add in
    quadrature). Saved as PDF and PNG.
    """
    hep.style.use(hep.style.CMS)

    # {point: {region: (significance, signal, background)}}
    table = {}
    for info in datacard_infos:
        process_to_rate = dict(zip(info["processes"], info["rates"]))
        # The signal process is called "sig" (current) or "signal" (older datacards)
        signal = process_to_rate.get("sig", process_to_rate.get("signal", 0.0))
        background = sum(r for proc, r in process_to_rate.items() if proc not in ("sig", "signal"))
        label = info.get("label") or info["bin_name"]
        region, _, point = label.partition("/")
        table.setdefault(point or label, {})[region if point else label] = (
            calculate_significance(signal, background, formula), signal, background)

    for point, regions in table.items():
        if len(regions) > 1:
            z = float(np.sqrt(sum(v[0] ** 2 for v in regions.values())))
            regions["combined"] = (z, sum(v[1] for v in regions.values()), sum(v[2] for v in regions.values()))

    def mass_key(point):
        m = re.search(r"mA(\d+)_ma(\d+)", point)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    points = sorted(table, key=mass_key)
    series = [("sr1b", "SR1b", "#5790fc"), ("sr2b", "SR2b", "#f89c20"), ("combined", "SR1b + SR2b", "#964a8b")]
    series = [(k, n, c) for k, n, c in series if any(k in table[p] for p in points)] or \
             [(k, k, None) for k in sorted({r for p in points for r in table[p]})]
    ylabel = (r"Expected significance $Z$" if formula == "asimov" else r"$s/\sqrt{s+b}$")

    fig, ax = plt.subplots(figsize=(11, 8))
    width = 0.8 / len(series)
    x = np.arange(len(points))
    ymin = 1e-2
    for k, (key, name, color) in enumerate(series):
        vals = np.array([table[p].get(key, (np.nan,))[0] for p in points], dtype=float)
        pos = x + (k - (len(series) - 1) / 2) * width
        bars = ax.bar(pos, vals, width * 0.92, color=color, label=name)
        for xi, v in zip(pos, vals):
            if np.isfinite(v):
                ax.text(xi, v * 1.12, f"{v:.2g}" if v < 10 else f"{v:.1f}", ha="center", va="bottom", fontsize=14)
    ax.set_yscale("log")
    ymax = np.nanmax([v[0] for p in points for v in table[p].values() if np.isfinite(v[0])])
    ax.set_ylim(ymin, ymax * 12)
    for level in (3, 5):
        if level < ymax * 12:
            ax.axhline(level, color="gray", ls="--", lw=1.5)
            ax.text(len(points) - 0.45, level * 1.05, rf"{level}$\sigma$", color="gray", ha="right", va="bottom", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([f"({mass_key(p)[0]}, {mass_key(p)[1]})" if mass_key(p) != (0, 0) else p for p in points])
    ax.set_xlim(-0.6, len(points) - 0.4)
    ax.set_xlabel(r"Signal point $(m_{\rm med},\ m_{\rm DM})$ [GeV]", labelpad=8)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", frameon=False, fontsize=18, ncol=len(series))
    hep.cms.label(ax=ax, data=False, lumi=lumi, year="Run3", com=13.6)

    plt.tight_layout()
    output_file = Path(output_file)
    for out in (output_file, output_file.with_suffix(".png")):
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"✓ Significance plot saved to: {out}")
    plt.close(fig)

    print("\n" + "=" * 78)
    print(f"Significance Summary ({'Asimov Z' if formula == 'asimov' else 's/sqrt(s+b)'}, statistical only)")
    print("=" * 78)
    print(f"{'Region/point':<28} {'Signal':>12} {'Background':>14} {'Z':>10}")
    print("-" * 78)
    for p in points:
        for key, name, _ in series:
            if key in table[p]:
                z, sg, bg = table[p][key]
                print(f"{key + '/' + p:<28} {sg:>12.2f} {bg:>14.2f} {z:>10.3f}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="Plot significance (s/sqrt(s+b)) for all regions"
    )
    parser.add_argument(
        "--formula",
        choices=["asimov", "simple"],
        default="asimov",
        help="asimov: sqrt(2((s+b)ln(1+s/b)-s)), as in calculate_sensitivity.py (default); "
             "simple: s/sqrt(s+b)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory from process_regions.py"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="significance.pdf",
        help="Output filename for significance plot"
    )
    parser.add_argument(
        "--lumi",
        type=float,
        default=290.0,
        help="Luminosity in fb^-1"
    )
    parser.add_argument(
        "--regions",
        type=str,
        nargs="+",
        default=None,
        help="Specific regions to include (default: all)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Find all datacards
    print("="*70)
    print("Finding Datacards")
    print("="*70)
    datacards = find_all_datacards(output_dir)

    if not datacards:
        print("✗ No datacards found!")
        return

    print(f"  Found {len(datacards)} datacards")

    # Filter by regions if specified
    if args.regions:
        datacards = [(dc, rn) for dc, rn in datacards if rn.split("/")[0] in args.regions]
        if not datacards:
            print(f"✗ No datacards found for specified regions!")
            return

    # Parse all datacards
    print("\n" + "="*70)
    print("Parsing Datacards")
    print("="*70)
    datacard_infos = []
    for datacard_file, region_name in datacards:
        print(f"  Parsing {region_name}...")
        try:
            info = parse_datacard(datacard_file)
            info["label"] = region_name
            datacard_infos.append(info)
        except Exception as e:
            print(f"    ✗ Error: {e}")

    if not datacard_infos:
        print("✗ No datacards parsed successfully!")
        return

    # Plot significance
    print("\n" + "="*70)
    print("Plotting Significance")
    print("="*70)
    output_file = output_dir / args.output_file
    plot_significance(datacard_infos, output_file, lumi=args.lumi, formula=args.formula)

    print("\n" + "="*70)
    print("✓ Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
