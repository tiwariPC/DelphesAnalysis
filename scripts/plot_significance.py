#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
"""
Plot significance (s/sqrt(s+b)) for all regions.

This script reads datacards and plots the significance for each region.
"""

import argparse
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for batch mode
import matplotlib.pyplot as plt
import mplhep as hep
from pathlib import Path
from typing import Dict, List, Tuple

# Enable LaTeX rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['mathtext.fontset'] = 'stix'
# If LaTeX fails, matplotlib will automatically use mathtext (built-in LaTeX-like rendering)


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


def calculate_significance(signal: float, background: float) -> float:
    """
    Calculate significance: s/sqrt(s+b)

    Parameters:
    -----------
    signal : float
        Signal yield
    background : float
        Total background yield

    Returns:
    --------
    significance : float
        Significance value
    """
    if signal <= 0:
        return 0.0

    total = signal + background
    if total <= 0:
        return 0.0

    return signal / np.sqrt(total)


def find_all_datacards(output_dir: Path) -> List[Tuple[Path, str]]:
    """Find all datacard.txt files."""
    plots_dir = output_dir / "plots"

    if not plots_dir.exists():
        return []

    datacards = []
    for region_dir in plots_dir.iterdir():
        if region_dir.is_dir():
            datacard_file = region_dir / "datacard.txt"
            if datacard_file.exists():
                datacards.append((datacard_file, region_dir.name))

    return datacards


def plot_significance(datacard_infos: List[Dict], output_file: Path, lumi: float = 290.0):
    """
    Plot significance for all regions.

    Parameters:
    -----------
    datacard_infos : list
        List of datacard info dictionaries
    output_file : Path
        Output file path
    lumi : float
        Luminosity in fb^-1
    """
    # Set CMS style
    hep.style.use(hep.style.CMS)

    # Calculate significance for each region
    regions = []
    significances = []
    signal_yields = []
    bg_yields = []

    for info in datacard_infos:
        # Find signal and background rates
        process_to_rate = dict(zip(info["processes"], info["rates"]))

        signal = process_to_rate.get("signal", 0.0)
        background = sum([rate for proc, rate in process_to_rate.items() if proc != "signal"])

        sig = calculate_significance(signal, background)

        regions.append(info["bin_name"])
        significances.append(sig)
        signal_yields.append(signal)
        bg_yields.append(background)

    # Sort by significance (descending)
    sorted_data = sorted(zip(regions, significances, signal_yields, bg_yields),
                        key=lambda x: x[1], reverse=True)
    regions, significances, signal_yields, bg_yields = zip(*sorted_data)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Bar plot
    bars = ax.bar(range(len(regions)), significances, color='steelblue', alpha=0.7)

    # Add value labels on bars
    for i, (bar, sig) in enumerate(zip(bars, significances)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{sig:.3f}',
               ha='center', va='bottom', fontsize=10)

    # Set labels
    ax.set_xlabel(r"Region", labelpad=10, fontsize=14)
    ax.set_ylabel(r"$s/\sqrt{s+b}$", labelpad=10, fontsize=14)
    ax.set_title(r"Significance by Region", fontsize=16, pad=15)

    # Set x-axis ticks
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions, rotation=45, ha='right', fontsize=10)

    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Add CMS label
    hep.cms.label(ax=ax, data=False, lumi=lumi, year="Run3")

    # Add text box with summary
    textstr = f'Total regions: {len(regions)}\n'
    textstr += f'Max significance: {max(significances):.3f}\n'
    textstr += f'Min significance: {min(significances):.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Significance plot saved to: {output_file}")

    # Print summary
    print("\n" + "="*70)
    print("Significance Summary")
    print("="*70)
    print(f"{'Region':<20} {'Signal':>12} {'Background':>12} {'s/√(s+b)':>12}")
    print("-"*70)
    for region, sig, s, b in zip(regions, significances, signal_yields, bg_yields):
        print(f"{region:<20} {s:>12.2f} {b:>12.2f} {sig:>12.3f}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Plot significance (s/sqrt(s+b)) for all regions"
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
        datacards = [(dc, rn) for dc, rn in datacards if rn in args.regions]
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
    plot_significance(datacard_infos, output_file, lumi=args.lumi)

    print("\n" + "="*70)
    print("✓ Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
