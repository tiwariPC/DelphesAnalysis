#!/usr/bin/env python3
"""
Standalone script to analyze jet properties from Delphes ROOT files.

This script loads a Delphes ROOT file and displays detailed information about jets,
including pT, eta, phi, b-tagging, and distributions.
"""

import sys
import numpy as np
import awkward as ak
from pathlib import Path

try:
    import uproot
except ImportError:
    print("Error: uproot is required. Install with: pip install uproot")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plots will not be generated.")


def load_delphes_file(root_file):
    """Load Delphes events from ROOT file."""
    try:
        file = uproot.open(root_file)
        if "Delphes" not in file:
            available = list(file.keys())
            file.close()
            raise KeyError(f"Tree 'Delphes' not found. Available trees: {available}")

        tree = file["Delphes"]

        # Load Jet branch
        try:
            jet_pt = tree["Jet/Jet.PT"].array(library="ak")
            jet_eta = tree["Jet/Jet.Eta"].array(library="ak")
            jet_phi = tree["Jet/Jet.Phi"].array(library="ak")
            jet_btag = tree["Jet/Jet.BTag"].array(library="ak")
            jet_mass = tree["Jet/Jet.Mass"].array(library="ak")
        except Exception as e:
            file.close()
            raise RuntimeError(f"Error loading Jet branches: {e}")

        file.close()

        return {
            'PT': jet_pt,
            'Eta': jet_eta,
            'Phi': jet_phi,
            'BTag': jet_btag,
            'Mass': jet_mass
        }
    except Exception as e:
        raise RuntimeError(f"Error opening file {root_file}: {e}")


def analyze_jets(jets, output_dir=None):
    """Analyze and display jet properties."""
    print("=" * 80)
    print("JET PROPERTIES ANALYSIS")
    print("=" * 80)
    print()

    # Basic statistics
    n_events = len(jets['PT'])
    print(f"Total number of events: {n_events:,}")

    # Count jets per event
    n_jets_per_event = ak.num(jets['PT'], axis=1)
    total_jets = ak.sum(n_jets_per_event)
    avg_jets_per_event = ak.mean(n_jets_per_event)

    print(f"Total number of jets: {total_jets:,}")
    print(f"Average jets per event: {avg_jets_per_event:.2f}")
    print(f"Min jets per event: {ak.min(n_jets_per_event)}")
    print(f"Max jets per event: {ak.max(n_jets_per_event)}")
    print()

    # Flatten all jets for statistics
    all_pt = ak.flatten(jets['PT'], axis=None)
    all_eta = ak.flatten(jets['Eta'], axis=None)
    all_phi = ak.flatten(jets['Phi'], axis=None)
    all_btag = ak.flatten(jets['BTag'], axis=None)
    all_mass = ak.flatten(jets['Mass'], axis=None)

    # Convert to numpy for easier statistics
    pt_np = np.asarray(ak.to_numpy(all_pt))
    eta_np = np.asarray(ak.to_numpy(all_eta))
    phi_np = np.asarray(ak.to_numpy(all_phi))
    btag_np = np.asarray(ak.to_numpy(all_btag))
    mass_np = np.asarray(ak.to_numpy(all_mass))

    print("=" * 80)
    print("JET pT DISTRIBUTION")
    print("=" * 80)
    print(f"  Mean: {np.mean(pt_np):.2f} GeV")
    print(f"  Median: {np.median(pt_np):.2f} GeV")
    print(f"  Std Dev: {np.std(pt_np):.2f} GeV")
    print(f"  Min: {np.min(pt_np):.2f} GeV")
    print(f"  Max: {np.max(pt_np):.2f} GeV")
    print(f"  25th percentile: {np.percentile(pt_np, 25):.2f} GeV")
    print(f"  75th percentile: {np.percentile(pt_np, 75):.2f} GeV")
    print()

    print("=" * 80)
    print("JET ETA DISTRIBUTION")
    print("=" * 80)
    print(f"  Mean: {np.mean(eta_np):.4f}")
    print(f"  Median: {np.median(eta_np):.4f}")
    print(f"  Std Dev: {np.std(eta_np):.4f}")
    print(f"  Min: {np.min(eta_np):.4f}")
    print(f"  Max: {np.max(eta_np):.4f}")
    print(f"  |eta| < 2.4: {np.sum(np.abs(eta_np) < 2.4):,} ({100*np.sum(np.abs(eta_np) < 2.4)/len(eta_np):.1f}%)")
    print()

    print("=" * 80)
    print("JET PHI DISTRIBUTION")
    print("=" * 80)
    print(f"  Mean: {np.mean(phi_np):.4f} rad")
    print(f"  Median: {np.median(phi_np):.4f} rad")
    print(f"  Std Dev: {np.std(phi_np):.4f} rad")
    print(f"  Min: {np.min(phi_np):.4f} rad")
    print(f"  Max: {np.max(phi_np):.4f} rad")
    print()

    print("=" * 80)
    print("B-TAGGING DISTRIBUTION")
    print("=" * 80)
    print(f"  Mean: {np.mean(btag_np):.4f}")
    print(f"  Median: {np.median(btag_np):.4f}")
    print(f"  Std Dev: {np.std(btag_np):.4f}")
    print(f"  Min: {np.min(btag_np):.4f}")
    print(f"  Max: {np.max(btag_np):.4f}")
    print(f"  BTag > 0: {np.sum(btag_np > 0):,} ({100*np.sum(btag_np > 0)/len(btag_np):.1f}%)")
    print(f"  BTag > 0.5: {np.sum(btag_np > 0.5):,} ({100*np.sum(btag_np > 0.5)/len(btag_np):.1f}%)")
    print(f"  BTag > 0.8: {np.sum(btag_np > 0.8):,} ({100*np.sum(btag_np > 0.8)/len(btag_np):.1f}%)")
    print()

    print("=" * 80)
    print("JET MASS DISTRIBUTION")
    print("=" * 80)
    print(f"  Mean: {np.mean(mass_np):.2f} GeV")
    print(f"  Median: {np.median(mass_np):.2f} GeV")
    print(f"  Std Dev: {np.std(mass_np):.2f} GeV")
    print(f"  Min: {np.min(mass_np):.2f} GeV")
    print(f"  Max: {np.max(mass_np):.2f} GeV")
    print()

    # Jet multiplicity distribution
    print("=" * 80)
    print("JET MULTIPLICITY DISTRIBUTION")
    print("=" * 80)
    unique_counts, counts = np.unique(ak.to_numpy(n_jets_per_event), return_counts=True)
    for n, count in zip(unique_counts, counts):
        percentage = 100 * count / n_events
        print(f"  {n} jets: {count:,} events ({percentage:.1f}%)")
    print()

    # Leading jet properties
    print("=" * 80)
    print("LEADING JET PROPERTIES (by pT)")
    print("=" * 80)
    leading_pt = ak.firsts(jets['PT'][ak.argsort(jets['PT'], axis=1, ascending=False, stable=True)], axis=1)
    leading_eta = ak.firsts(jets['Eta'][ak.argsort(jets['PT'], axis=1, ascending=False, stable=True)], axis=1)
    leading_btag = ak.firsts(jets['BTag'][ak.argsort(jets['PT'], axis=1, ascending=False, stable=True)], axis=1)

    leading_pt_np = np.asarray(ak.to_numpy(ak.fill_none(leading_pt, 0.0)))
    leading_eta_np = np.asarray(ak.to_numpy(ak.fill_none(leading_eta, 0.0)))
    leading_btag_np = np.asarray(ak.to_numpy(ak.fill_none(leading_btag, 0.0)))

    valid_leading = leading_pt_np > 0
    if np.sum(valid_leading) > 0:
        print(f"  Mean pT: {np.mean(leading_pt_np[valid_leading]):.2f} GeV")
        print(f"  Median pT: {np.median(leading_pt_np[valid_leading]):.2f} GeV")
        print(f"  Mean |eta|: {np.mean(np.abs(leading_eta_np[valid_leading])):.4f}")
        print(f"  Mean BTag: {np.mean(leading_btag_np[valid_leading]):.4f}")
        print(f"  Leading jet pT > 30 GeV: {np.sum(leading_pt_np > 30):,} ({100*np.sum(leading_pt_np > 30)/len(leading_pt_np):.1f}%)")
        print(f"  Leading jet pT > 100 GeV: {np.sum(leading_pt_np > 100):,} ({100*np.sum(leading_pt_np > 100)/len(leading_pt_np):.1f}%)")
    print()

    # B-jet statistics
    print("=" * 80)
    print("B-JET STATISTICS")
    print("=" * 80)
    bjets_mask = btag_np > 0
    n_bjets = np.sum(bjets_mask)
    if n_bjets > 0:
        print(f"  Total b-jets (BTag > 0): {n_bjets:,} ({100*n_bjets/len(btag_np):.1f}%)")
        print(f"  Mean b-jet pT: {np.mean(pt_np[bjets_mask]):.2f} GeV")
        print(f"  Mean b-jet |eta|: {np.mean(np.abs(eta_np[bjets_mask])):.4f}")
        print(f"  Mean b-jet BTag: {np.mean(btag_np[bjets_mask]):.4f}")

    # Count b-jets per event
    bjets_per_event = ak.sum(jets['BTag'] > 0, axis=1)
    unique_bcounts, bcounts = np.unique(ak.to_numpy(bjets_per_event), return_counts=True)
    print(f"\n  B-jet multiplicity:")
    for n, count in zip(unique_bcounts, bcounts):
        percentage = 100 * count / n_events
        print(f"    {n} b-jets: {count:,} events ({percentage:.1f}%)")
    print()

    # Create plots if matplotlib is available
    if HAS_MATPLOTLIB and output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("GENERATING PLOTS")
        print("=" * 80)

        # Plot 1: Jet pT distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(pt_np, bins=50, range=(0, 500), edgecolor='black', alpha=0.7)
        ax.set_xlabel('Jet pT [GeV]', fontsize=12)
        ax.set_ylabel('Number of Jets', fontsize=12)
        ax.set_title('Jet pT Distribution', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'jet_pt_distribution.pdf', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path / 'jet_pt_distribution.pdf'}")

        # Plot 2: Jet eta distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(eta_np, bins=50, range=(-5, 5), edgecolor='black', alpha=0.7)
        ax.set_xlabel('Jet η', fontsize=12)
        ax.set_ylabel('Number of Jets', fontsize=12)
        ax.set_title('Jet η Distribution', fontsize=14)
        ax.axvline(-2.4, color='r', linestyle='--', label='|η| = 2.4')
        ax.axvline(2.4, color='r', linestyle='--')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'jet_eta_distribution.pdf', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path / 'jet_eta_distribution.pdf'}")

        # Plot 3: B-tagging distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(btag_np, bins=50, range=(0, 1), edgecolor='black', alpha=0.7)
        ax.set_xlabel('B-Tag Score', fontsize=12)
        ax.set_ylabel('Number of Jets', fontsize=12)
        ax.set_title('B-Tagging Distribution', fontsize=14)
        ax.axvline(0.5, color='r', linestyle='--', label='BTag = 0.5')
        ax.axvline(0.8, color='orange', linestyle='--', label='BTag = 0.8')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'jet_btag_distribution.pdf', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path / 'jet_btag_distribution.pdf'}")

        # Plot 4: Jet multiplicity
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(unique_counts, counts, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Number of Jets per Event', fontsize=12)
        ax.set_ylabel('Number of Events', fontsize=12)
        ax.set_title('Jet Multiplicity Distribution', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_path / 'jet_multiplicity.pdf', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path / 'jet_multiplicity.pdf'}")

        # Plot 5: Leading jet pT
        if np.sum(valid_leading) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(leading_pt_np[valid_leading], bins=50, range=(0, 500), edgecolor='black', alpha=0.7)
            ax.set_xlabel('Leading Jet pT [GeV]', fontsize=12)
            ax.set_ylabel('Number of Events', fontsize=12)
            ax.set_title('Leading Jet pT Distribution', fontsize=14)
            ax.axvline(30, color='r', linestyle='--', label='pT = 30 GeV')
            ax.axvline(100, color='orange', linestyle='--', label='pT = 100 GeV')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / 'leading_jet_pt.pdf', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {output_path / 'leading_jet_pt.pdf'}")

        # Plot 6: pT vs eta scatter (sample)
        if len(pt_np) > 10000:
            # Sample for performance
            sample_idx = np.random.choice(len(pt_np), size=10000, replace=False)
            pt_sample = pt_np[sample_idx]
            eta_sample = eta_np[sample_idx]
        else:
            pt_sample = pt_np
            eta_sample = eta_np

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(eta_sample, pt_sample, alpha=0.3, s=1)
        ax.set_xlabel('Jet η', fontsize=12)
        ax.set_ylabel('Jet pT [GeV]', fontsize=12)
        ax.set_title('Jet pT vs η', fontsize=14)
        ax.axvline(-2.4, color='r', linestyle='--', alpha=0.5)
        ax.axvline(2.4, color='r', linestyle='--', alpha=0.5)
        ax.axhline(30, color='r', linestyle='--', alpha=0.5, label='pT = 30 GeV')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'jet_pt_vs_eta.pdf', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path / 'jet_pt_vs_eta.pdf'}")

        print()
        print(f"All plots saved to: {output_path}")

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze jet properties from Delphes ROOT file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_jets.py signal.root
  python analyze_jets.py signal.root --output-dir jet_analysis
        """
    )

    parser.add_argument(
        'root_file',
        type=str,
        help='Path to Delphes ROOT file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save plots (default: no plots)'
    )

    args = parser.parse_args()

    # Check if file exists
    root_path = Path(args.root_file)
    if not root_path.exists():
        print(f"Error: File not found: {args.root_file}")
        sys.exit(1)

    print(f"Loading jets from: {args.root_file}")
    print()

    try:
        jets = load_delphes_file(args.root_file)
        analyze_jets(jets, output_dir=args.output_dir)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

