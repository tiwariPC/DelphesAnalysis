#!/usr/bin/env python3
"""
Expected limits with CMS Combine on the datacards written by process_regions.py.

For every datacard in <output-dir>/datacards (the SR1b+SR2b combination per signal point and
the single-region ones) it runs

    combine -M AsymptoticLimits -t -1 --run expected

so the limit is the expected one, on the Asimov data set built from the background estimate
(there is no data; the datacards have observation -1). The limit is on the signal strength r,
the factor on the signal cross-section used in the analysis: r < 1 means the signal point is
expected to be excluded at 95% CL. With --signal-xsec, the limit is also given as a
cross-section, r * sigma.

Writes <output-dir>/limits.txt and <output-dir>/limits.json, and runs Combine in
<output-dir>/limits/ (the higgsCombine*.root files go there).
"""
import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

QUANTILES = ("2.5", "16.0", "50.0", "84.0", "97.5")


def read_rates(datacard):
    """(signal, background) event rates of a counting datacard (all bins)."""
    procs, rates = None, None
    for line in datacard.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "process" and not parts[1].lstrip("-").isdigit():
            procs = parts[1:]
        elif parts and parts[0] == "rate":
            rates = [float(x) for x in parts[1:]]
    if not procs or not rates or len(procs) != len(rates):
        raise ValueError(f"cannot read the process and rate lines of {datacard}")
    sig = sum(r for p, r in zip(procs, rates) if p == "sig")
    bkg = sum(r for p, r in zip(procs, rates) if p != "sig")
    return sig, bkg


def parse_combine_output(output):
    """Expected limits on r from the Combine stdout: {"2.5": value, ..., "50.0": value, ...}."""
    limits = {}
    for percentile, value in re.findall(r"Expected\s+([\d.]+)%:\s+r\s*<\s*([\d.eE+-]+)", output):
        limits[f"{float(percentile):.1f}"] = float(value)
    return limits


def _combine_once(datacard, name, work_dir, method, r_max, verbose):
    cmd = ["combine", "-M", method, str(datacard.resolve()), "-n", name, "-t", "-1", "--run", "expected",
           "--setParameterRanges", f"r=0,{r_max:.4g}", "--rMax", f"{r_max:.4g}"]
    if verbose:
        print("    " + " ".join(cmd))
    try:
        result = subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True, check=True)
    except FileNotFoundError:
        sys.exit("✗ 'combine' not found in PATH (install: INSTALL_COMBINE=1 source setup.sh)")
    except subprocess.CalledProcessError as e:
        print(f"    ✗ Combine failed ({datacard.name}):\n{e.stdout[-600:]}\n{e.stderr[-600:]}")
        return None
    if verbose:
        print(result.stdout)
    return parse_combine_output(result.stdout)


def _fits_range(limits, r_max):
    """True if the +2 sigma limit is well inside the r range (otherwise the range is too small)."""
    return "97.5" in limits and limits["97.5"] < 0.5 * r_max


def _bands_ok(limits):
    """The expected-limit bands of Combine's asymptotic method sometimes fail numerically
    (a lower band of ~1e-4 next to a median of ~100). In a healthy result the median is a
    factor 1.2-2 above the -1 sigma and about 1.5-3 above the -2 sigma limit."""
    m = limits.get("50.0")
    lo1, lo2 = limits.get("16.0"), limits.get("2.5")
    return (m is not None and lo1 is not None and lo2 is not None
            and lo1 > m / 4.0 and lo2 > m / 8.0 and lo2 < lo1 < m)


def run_combine(datacard, name, work_dir, method="AsymptoticLimits", verbose=False):
    """Run Combine on one datacard; returns {quantile: r limit} or None.

    The r range starts from the statistical estimate 2 sqrt(B) / S. With background
    normalisation uncertainties of 20-40% the limit can be far larger, so the range is
    widened by a factor 10 while the +2 sigma limit does not fit in it. It is not widened
    further: with a range far above the limit (>~100x) the asymptotic method breaks down.
    If the lower bands are unreliable they are dropped (NaN) and a warning is printed."""
    sig, bkg = read_rates(datacard)
    if sig <= 0:
        print("    ✗ no signal events in the datacard")
        return None
    r_max = max(20.0, 40.0 * math.sqrt(max(bkg, 1.0)) / sig)
    limits = None
    for _ in range(4):
        limits = _combine_once(datacard, name, work_dir, method, r_max, verbose)
        if limits is None:
            return None
        if _fits_range(limits, r_max):
            break
        r_max *= 10
    else:
        print(f"    ⚠ the +2 sigma limit of {datacard.name} does not fit in r < {r_max:.3g}")
    if "50.0" not in limits:
        return None
    if not _bands_ok(limits):
        print(f"    ⚠ unreliable lower bands for {datacard.name} (Combine asymptotic failure): "
              f"only the median and the upper bands are kept")
        limits = {q: (v if q in ("50.0", "84.0", "97.5") else float("nan")) for q, v in limits.items()}
    return limits


def find_datacards(output_dir):
    """[(label, path)]: 'combined/<point>' first, then 'sr1b/<point>', ... (label is region/point)."""
    dc_dir = output_dir / "datacards"
    found = []
    for path in sorted(dc_dir.glob("combined_datacard_*.txt")):
        found.append((f"combined/{path.stem[len('combined_datacard_'):]}", path))
    for region_dir in sorted(p for p in dc_dir.iterdir() if p.is_dir()):
        for path in sorted(region_dir.glob("datacard_*.txt")):
            found.append((f"{region_dir.name}/{path.stem[len('datacard_'):]}", path))
    return found


def load_signal_xsecs(path):
    """{"mA500_ma50": sigma_pb} from a signal cross-section yaml (mA/mMED, then ma/mDM)."""
    import yaml
    with open(path) as f:
        table = (yaml.safe_load(f) or {}).get("signals", {}) or {}
    return {f"mA{a}_ma{b}": xs for a, row in table.items() for b, xs in row.items()}


def point_masses(label):
    """'combined/mA500_ma50' or 'mA500_ma50_...' -> (500, 50)"""
    m = re.search(r"mA(\d+)_ma(\d+)", label)
    return (int(m.group(1)), int(m.group(2))) if m else None


def plot_brazil(results, xsecs, output_dir, lumi, region="combined", signal_label="t-channel DM (S3M)"):
    """Brazilian-flag plot of the expected limits of the signal points, ordered by mediator mass:
    median (dashed), 68% (green) and 95% (yellow) bands, and r = 1. A second plot shows the
    limit as a cross-section (pb) next to the model cross-section. Saves PDF and PNG files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    import numpy as np
    hep.style.use(hep.style.CMS)

    pts = []
    for label, lim in results.items():
        reg, _, point = label.partition("/")
        m = point_masses(point)
        if reg == region and m:
            pts.append((m, point, lim))
    pts.sort()
    if not pts:
        print(f"    ⚠ no '{region}' limits to plot")
        return []
    x = np.arange(len(pts))
    ticks = [f"({m[0]}, {m[1]})" for m, _, _ in pts]
    get = lambda q: np.array([lim.get(q, float("nan")) for _, _, lim in pts], dtype=float)
    med, lo1, hi1, lo2, hi2 = get("50.0"), get("16.0"), get("84.0"), get("2.5"), get("97.5")
    bands = not (np.isnan(lo1).any() or np.isnan(lo2).any())

    def draw(ax, scale, ylabel, theory=None):
        ax.set_yscale("log")
        w = 0.4
        for xi in x:
            if bands:
                ax.fill_between([xi - w, xi + w], lo2[xi] * scale[xi], hi2[xi] * scale[xi], color="#f5cb42", lw=0,
                                label=r"95% expected" if xi == 0 else None)
                ax.fill_between([xi - w, xi + w], lo1[xi] * scale[xi], hi1[xi] * scale[xi], color="#4daf4a", lw=0,
                                label=r"68% expected" if xi == 0 else None)
            ax.plot([xi - w, xi + w], [med[xi] * scale[xi]] * 2, "k--", lw=2,
                    label="Median expected" if xi == 0 else None)
        if theory is not None:
            ax.plot(x, theory, "r-o", lw=2, ms=7, label=r"Model, $\sigma_{\rm th}$")
        ax.set_xticks(x)
        ax.set_xticklabels(ticks)
        ax.set_xlim(-0.7, len(pts) - 0.3)
        ax.set_xlabel(r"Signal point $(m_{\rm med},\ m_{\rm DM})$ [GeV]", labelpad=8)
        ax.set_ylabel(ylabel)
        top = np.nanmax(hi2 * scale if bands else hi1 * scale if not np.isnan(hi1).any() else med * scale)
        ax.set_ylim(top=top * 60)
        ax.legend(loc="upper left", frameon=False, fontsize=16, ncol=2)
        hep.cms.label(ax=ax, data=False, lumi=lumi, year="Run3", com=13.6)

    saved = []
    fig, ax = plt.subplots(figsize=(10, 8))
    draw(ax, np.ones(len(pts)), r"95% CL upper limit on $r = \sigma/\sigma_{\rm th}$")
    ax.axhline(1.0, color="r", lw=2, label=r"$r = 1$")
    ax.legend(loc="upper left", frameon=False, fontsize=16, ncol=2)
    base = output_dir / f"limits_{region}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{base}.{ext}", dpi=200, bbox_inches="tight")
        saved.append(f"{base}.{ext}")
    plt.close(fig)

    th = np.array([xsecs.get(point, np.nan) for _, point, _ in pts]) if xsecs else None
    if th is not None and not np.isnan(th).any():
        fig, ax = plt.subplots(figsize=(10, 8))
        draw(ax, th, r"95% CL upper limit on $\sigma$ [pb]", theory=th)
        base = output_dir / f"limits_{region}_xsec"
        for ext in ("pdf", "png"):
            fig.savefig(f"{base}.{ext}", dpi=200, bbox_inches="tight")
            saved.append(f"{base}.{ext}")
        plt.close(fig)
    return saved


def main():
    parser = argparse.ArgumentParser(description="Expected limits with Combine on all datacards")
    parser.add_argument("--output-dir", default="output", help="output directory of process_regions.py")
    parser.add_argument("--method", default="AsymptoticLimits", choices=["AsymptoticLimits"],
                        help="Combine method (expected limits on Asimov data)")
    parser.add_argument("--signal-xsec", default=None,
                        help="signal cross-section yaml: adds the limit on the cross-section (pb)")
    parser.add_argument("--region", default=None,
                        help="only this region (e.g. sr1b, sr2b, combined)")
    parser.add_argument("--lumi", type=float, default=290.0, help="luminosity (fb^-1) for the plot label")
    parser.add_argument("--no-plot", action="store_true", help="do not draw the Brazilian-flag plots")
    parser.add_argument("--verbose", action="store_true", help="print the Combine output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not (output_dir / "datacards").is_dir():
        sys.exit(f"✗ No datacards directory in {output_dir}")
    cards = find_datacards(output_dir)
    if args.region:
        cards = [(l, p) for l, p in cards if l.split("/")[0] == args.region]
    if not cards:
        sys.exit("✗ No datacards found!")
    xsecs = load_signal_xsecs(args.signal_xsec) if args.signal_xsec else {}

    work_dir = output_dir / "limits"
    work_dir.mkdir(exist_ok=True)
    print("=" * 70)
    print(f"Expected limits (Combine {args.method}, Asimov data): {len(cards)} datacards")
    print("=" * 70)

    results = {}
    for label, path in cards:
        name = label.replace("/", "_")
        print(f"  {label}")
        limits = run_combine(path, name, work_dir, args.method, args.verbose)
        if limits is None:
            continue
        results[label] = limits
        print(f"    expected r < {limits['50.0']:.4g}   (68%: {limits.get('16.0', float('nan')):.4g}"
              f" .. {limits.get('84.0', float('nan')):.4g})")

    if not results:
        sys.exit("✗ No limits extracted!")

    lines = ["Expected 95% CL upper limits on the signal strength r (r < 1: signal point excluded)",
             "(r multiplies the signal cross-section of the analysis; Asimov data, only the uncertainties in the datacard;", ""]
    header = f"{'Region/point':<28}{'-2sigma':>10}{'-1sigma':>10}{'median':>10}{'+1sigma':>10}{'+2sigma':>10}"
    if xsecs:
        header += f"{'sigma (pb)':>13}{'limit (pb)':>13}"
    lines += [header, "-" * len(header)]
    for label, lim in sorted(results.items()):
        row = f"{label:<28}" + "".join(
            f"{lim[q]:>10.4g}" if q in lim and not math.isnan(lim[q]) else f"{'n/a':>10}" for q in QUANTILES)
        if xsecs:
            xs = xsecs.get(label.split("/", 1)[1])
            row += f"{xs:>13.4g}{lim['50.0'] * xs:>13.4g}" if xs else f"{'-':>13}{'-':>13}"
        lines.append(row)
    text = "\n".join(lines)
    (output_dir / "limits.txt").write_text(text + "\n")
    (output_dir / "limits.json").write_text(json.dumps(results, indent=2))
    print("\n" + text)
    print(f"\n✓ Written {output_dir}/limits.txt and limits.json (Combine files in {work_dir}/)")
    if not args.no_plot:
        for f in plot_brazil(results, xsecs, output_dir, args.lumi):
            print(f"✓ Saved {f}")


if __name__ == "__main__":
    main()
