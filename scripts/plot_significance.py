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


SERIES = [("sr1b", "SR1b", "#5790fc"), ("sr2b", "SR2b", "#f89c20"), ("combined", "SR1b + SR2b", "#964a8b")]


def _mass_key(point):
    m = re.search(r"mA(\d+)_ma(\d+)", point)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def region_yields(datacard_infos):
    """{point: {region: (signal, background)}} from the datacards, plus 'combined' (the sums over
    the regions) for points with several regions."""
    yields = {}
    for info in datacard_infos:
        process_to_rate = dict(zip(info["processes"], info["rates"]))
        # The signal process is called "sig" (current) or "signal" (older datacards)
        signal = process_to_rate.get("sig", process_to_rate.get("signal", 0.0))
        background = sum(r for proc, r in process_to_rate.items() if proc not in ("sig", "signal"))
        label = info.get("label") or info["bin_name"]
        region, _, point = label.partition("/")
        yields.setdefault(point or label, {})[region if point else label] = (signal, background)
    for regions in yields.values():
        if len(regions) > 1:
            regions["combined"] = (sum(v[0] for v in regions.values()), sum(v[1] for v in regions.values()))
    return yields


def draw_grouped_bars(values, ylabel, text, output_file, lumi, levels=()):
    """Grouped bars per signal point (ordered by mediator mass), log y axis, CMS style.
    values: {point: {region: value}}; text: lines of the formula block; levels: [(y, label)]
    reference lines. Saved as PDF and PNG."""
    hep.style.use(hep.style.CMS)
    points = sorted(values, key=_mass_key)
    series = [(k, n, c) for k, n, c in SERIES if any(k in values[p] for p in points)] or \
             [(k, k, None) for k in sorted({r for p in points for r in values[p]})]
    finite = [v for p in points for v in values[p].values() if np.isfinite(v) and v > 0]
    ymax, ymin = max(finite), min(finite) / 3.0

    fig, ax = plt.subplots(figsize=(11, 8))
    width = 0.8 / len(series)
    x = np.arange(len(points))
    ax.set_yscale("log")
    for k, (key, name, color) in enumerate(series):
        vals = np.array([values[p].get(key, np.nan) for p in points], dtype=float)
        pos = x + (k - (len(series) - 1) / 2) * width
        ax.bar(pos, vals - ymin, width * 0.92, bottom=ymin, color=color, label=name)   # top of bar = value
        for xi, v in zip(pos, vals):
            if np.isfinite(v) and v > 0:
                ax.text(xi, v * 1.12, f"{v:.1f}" if v >= 10 else f"{v:.2g}", ha="center", va="bottom", fontsize=14)
    ax.set_ylim(ymin, ymax * 40)       # headroom for the legend and the formula text
    for level, label in levels:
        if ymin < level < ymax * 40:
            ax.axhline(level, color="gray", ls="--", lw=1.5)
            ax.text(len(points) - 0.45, level * 1.05, label, color="gray", ha="right", va="bottom", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([f"({_mass_key(p)[0]}, {_mass_key(p)[1]})" if _mass_key(p) != (0, 0) else p for p in points])
    ax.set_xlim(-0.6, len(points) - 0.4)
    ax.set_xlabel(r"Signal point $(m_{\rm med},\ m_{\rm DM})$ [GeV]", labelpad=8)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", frameon=False, fontsize=18, ncol=len(series))
    for k, line in enumerate(text):
        ax.text(0.03, 0.95 - 0.085 * k, line, transform=ax.transAxes, fontsize=17, va="top", ha="left",
                math_fontfamily="cm")
    hep.cms.label(ax=ax, data=False, lumi=lumi, year="Run3", com=13.6)

    plt.tight_layout()
    output_file = Path(output_file)
    for out in (output_file, output_file.with_suffix(".png")):
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"✓ Plot saved to: {out}")
    plt.close(fig)


# Formula blocks are typeset with matplotlib's Computer Modern math (LaTeX look, no latex binary
# needed); variables are explicitly italic.
QUADRATURE_TEXT = r"$\mathit{Z}_{\mathrm{1b+2b}}=\sqrt{\mathit{Z}_{\mathrm{1b}}^{2}+\mathit{Z}_{\mathrm{2b}}^{2}}$"


def plot_significance(datacard_infos: List[Dict], output_file: Path, lumi: float = 326.03,
                      formula: str = "asimov"):
    """
    Expected significance per signal point: SR1b, SR2b and their combination (the regions
    are disjoint, so the significances add in quadrature).
    """
    yields = region_yields(datacard_infos)
    table = {}
    for point, regions in yields.items():
        z = {r: calculate_significance(s, b, formula) for r, (s, b) in regions.items() if r != "combined"}
        if len(z) > 1:
            z["combined"] = float(np.sqrt(sum(v ** 2 for v in z.values())))
        table[point] = z
    if formula == "asimov":
        text = [r"$\mathit{Z}=\sqrt{2\,[(\mathit{s}+\mathit{b})\,\ln(1+\mathit{s}/\mathit{b})-\mathit{s}]}$",
                QUADRATURE_TEXT]
        ylabel = r"Expected significance $Z$"
    else:
        text = [r"$\mathit{Z}=\mathit{s}\,/\sqrt{\mathit{s}+\mathit{b}}$", QUADRATURE_TEXT]
        ylabel = r"$s/\sqrt{s+b}$"
    draw_grouped_bars(table, ylabel, text, output_file, lumi, levels=[(3, r"3$\sigma$"), (5, r"5$\sigma$")])

    print("\n" + "=" * 78)
    print(f"Significance Summary ({'Asimov Z' if formula == 'asimov' else 's/sqrt(s+b)'})")
    print("=" * 78)
    print(f"{'Region/point':<28} {'Signal':>12} {'Background':>14} {'Z':>10}")
    print("-" * 78)
    for p in sorted(table, key=_mass_key):
        for key, name, _ in SERIES:
            if key in table[p]:
                sg, bg = yields[p][key]
                print(f"{key + '/' + p:<28} {sg:>12.2f} {bg:>14.2f} {table[p][key]:>10.3f}")
    print("=" * 78)


def load_signal_xsecs(path) -> Dict[str, float]:
    """{"mA500_ma50": sigma_pb} from a signal cross-section yaml (mA/mMED, then ma/mDM)."""
    import yaml
    with open(path) as f:
        table = (yaml.safe_load(f) or {}).get("signals", {}) or {}
    return {f"mA{a}_ma{b}": xs for a, row in table.items() for b, xs in row.items()}


def plot_punzi(datacard_infos: List[Dict], output_file: Path, xsecs: Dict[str, float],
               lumi: float = 326.03, a: float = 5.0):
    """
    Punzi figure of merit  epsilon_S / (a/2 + sqrt(B))  per signal point and region
    (G. Punzi, PHYSTAT 2003), with epsilon_S = S / (sigma * L) the fraction of the produced signal
    that passes the selection and B the background yield at the luminosity L. a is the number of
    standard deviations (3: evidence, 5: discovery). Unlike the significance it needs no
    signal cross-section hypothesis in its optimisation. For SR1b + SR2b the yields of the two
    regions are added (one counting experiment), the quadrature sum does not apply.
    """
    yields = region_yields(datacard_infos)
    table, details = {}, {}
    for point, regions in yields.items():
        sigma_pb = xsecs.get(point)
        if not sigma_pb:
            print(f"  ⚠ no cross-section for {point}: skipped in the Punzi plot")
            continue
        produced = sigma_pb * 1000.0 * lumi          # events produced: sigma [fb] * L [fb^-1]
        table[point], details[point] = {}, {}
        for region, (s, b) in regions.items():
            eff = s / produced
            table[point][region] = eff / (a / 2.0 + np.sqrt(b))
            details[point][region] = (eff, b)
    if not table:
        print("✗ no signal point with a cross-section: no Punzi plot")
        return
    text = [rf"$\epsilon_{{S}}\,/\,(\mathit{{a}}/2+\sqrt{{\mathit{{B}}}})\,,\quad \mathit{{a}}={a:g}$",
            r"$\epsilon_{S}=\mathit{S}\,/\,(\sigma\,\mathit{L})\,,\quad \mathit{B}:\ \mathrm{background\ yield}$"]
    draw_grouped_bars(table, r"Punzi figure of merit $\epsilon_S/(a/2+\sqrt{B})$", text, output_file, lumi)

    print("\n" + "=" * 78)
    print(f"Punzi figure of merit, a = {a:g}")
    print("=" * 78)
    print(f"{'Region/point':<28} {'eff_S [%]':>12} {'B':>14} {'Punzi':>12}")
    print("-" * 78)
    for p in sorted(table, key=_mass_key):
        for key, name, _ in SERIES:
            if key in table[p]:
                eff, b = details[p][key]
                print(f"{key + '/' + p:<28} {100 * eff:>12.3f} {b:>14.2f} {table[p][key]:>12.3e}")
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
        default=None,
        help="Output filename of the significance plot (default: significance_<formula>.pdf, "
             "i.e. significance_asimov.pdf; the Punzi plot is significance_punzi.pdf)"
    )
    parser.add_argument(
        "--signal-xsec",
        type=str,
        default=None,
        help="signal cross-section yaml (pb): needed for the Punzi plots (signal efficiency)"
    )
    parser.add_argument(
        "--punzi-a",
        type=float,
        default=5.0,
        help="a (number of sigma) of the Punzi figure of merit: 3 evidence, 5 discovery (default 5)"
    )
    parser.add_argument(
        "--lumi",
        type=float,
        default=326.03,
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
    output_file = output_dir / (args.output_file or f"significance_{args.formula}.pdf")
    plot_significance(datacard_infos, output_file, lumi=args.lumi, formula=args.formula)

    if args.signal_xsec:
        xsecs = load_signal_xsecs(args.signal_xsec)
        print("\n" + "=" * 70)
        print(f"Plotting Punzi figure of merit (a = {args.punzi_a:g})")
        print("=" * 70)
        plot_punzi(datacard_infos, output_dir / "significance_punzi.pdf", xsecs, lumi=args.lumi, a=args.punzi_a)
    else:
        print("\n(no --signal-xsec: Punzi plot skipped)")

    print("\n" + "="*70)
    print("✓ Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
