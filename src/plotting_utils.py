"""
Publication-quality plotting utilities for bbMET analysis.
Uses mplhep for CMS-compliant plotting following:
https://cms-analysis.docs.cern.ch/guidelines/plotting/
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for batch mode
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mplhep as hep
from hist import Hist
from typing import Dict, List, Optional, Tuple
from math import log10, floor

# Enable LaTeX rendering for matplotlib
# Set LaTeX rendering - matplotlib will fall back to mathtext if LaTeX is unavailable
plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering
plt.rcParams['mathtext.default'] = 'regular'  # Use regular math text
plt.rcParams['mathtext.fontset'] = 'stix'  # Use STIX fonts for better math rendering
# If LaTeX fails, matplotlib will automatically use mathtext (built-in LaTeX-like rendering)

# Set CMS style using mplhep
# This automatically configures all CMS plotting standards
hep.style.use(hep.style.CMS)


def plot_cutflow(cutflow_dict: Optional[Dict[str, int]] = None,
                 sample_name: str = "Sample",
                 output_file: Optional[str] = None,
                 normalize: bool = False,
                 lumi: float = 290.0,
                 background_cutflows: Optional[Dict[str, Dict[str, int]]] = None):
    """
    Plot cutflow as stacked bar chart with CMS styling using mplhep.

    Parameters:
    -----------
    cutflow_dict : dict, optional
        Single cutflow dictionary (for backward compatibility)
    sample_name : str
        Sample name
    output_file : str
        Output filename (optional)
    normalize : bool
        Normalize to first cut (default: False)
    lumi : float
        Luminosity in fb^-1 (for CMS label)
    background_cutflows : dict, optional
        Dictionary of {bg_name: cutflow_dict} for stacked plot
    """
    # Use background_cutflows if provided, otherwise use single cutflow_dict
    if background_cutflows is not None and len(background_cutflows) > 0:
        # Get all cut names from first background (assuming all have same structure)
        # Try to find a background with the most cuts (in case some are empty)
        max_cuts = 0
        first_bg = None
        for bg_cutflow in background_cutflows.values():
            if isinstance(bg_cutflow, dict) and len(bg_cutflow) > max_cuts:
                max_cuts = len(bg_cutflow)
                first_bg = bg_cutflow

        # Fallback to first background if we didn't find one
        if first_bg is None:
            first_bg = list(background_cutflows.values())[0]

        cuts = list(first_bg.keys())

        # Debug: Check what we're getting
        if len(cuts) == 1:
            print(f"  WARNING plot_cutflow: Only 1 cut found! first_bg type: {type(first_bg)}, keys: {cuts}")
            print(f"    first_bg contents: {first_bg}")
            print(f"    background_cutflows keys: {list(background_cutflows.keys())}")
            for bg_name, bg_cf in background_cutflows.items():
                print(f"      {bg_name}: {type(bg_cf)}, keys: {list(bg_cf.keys())}")

        # Get background colors and labels (using provided hex codes)
        bg_colors = {
            'ttbar': '#bd1f01',      # Red
            'wlnjets': '#3f90da',     # Blue
            'znnjets': '#832db6',     # Purple
            'dyjets': '#94a4a2',      # Gray
            'diboson': '#ffa90e',     # Orange
            'stop': '#a96b59',        # Brown (combined single top)
            'sTop_tchannel': '#a96b59',  # Brown (kept for backward compatibility)
            'sTop_tW': '#e76300',     # Orange-red (kept for backward compatibility)
        }

        bg_labels = {
            'ttbar': r'$t\bar{t}$',
            'wlnjets': r'$W$+jets',
            'znnjets': r'$Z$+jets',
            'dyjets': r'$DY$+jets',
            'diboson': r'Diboson',
            'stop': r'Single top',
            'sTop_tchannel': r'Single top',  # Kept for backward compatibility
            'sTop_tW': r'Single top ($tW$)',  # Kept for backward compatibility
        }

        # Calculate total events per background (for sorting)
        bg_totals = {}
        for bg_name, bg_cutflow in background_cutflows.items():
            # Use first cut value as total (no preselection anymore)
            if cuts and cuts[0] in bg_cutflow:
                bg_totals[bg_name] = bg_cutflow[cuts[0]]
            else:
                bg_totals[bg_name] = 0

        # Sort backgrounds by total (smallest first) for better visibility
        sorted_bg_names = sorted(bg_totals.items(), key=lambda x: x[1])
        sorted_bg_names = [name for name, _ in sorted_bg_names]

        fig, ax = plt.subplots(figsize=(12, 8))

        # Prepare data for stacking
        n_cuts = len(cuts)
        x_pos = np.arange(n_cuts)
        width = 1.0  # No gaps between bars

        # Stack bars for each background
        bottom = np.zeros(n_cuts)
        for bg_name in sorted_bg_names:
            if bg_name not in background_cutflows:
                continue
            bg_cutflow = background_cutflows[bg_name]
            bg_events = [bg_cutflow.get(cut, 0) for cut in cuts]

            if normalize and len(bg_events) > 0 and bg_events[0] > 0:
                bg_events = [e / bg_events[0] for e in bg_events]

            color = bg_colors.get(bg_name, plt.cm.Set3(len(bg_colors) % 12))
            label = bg_labels.get(bg_name, bg_name)

            bars = ax.bar(x_pos, bg_events, width=width, bottom=bottom,
                         label=label, color=color, alpha=0.85)
            bottom += np.array(bg_events)

        ylabel = r"Efficiency" if normalize else r"Events"
        ax.set_xlabel(r"Selection Step", labelpad=10)
        ax.set_ylabel(ylabel, labelpad=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(cuts, rotation=45, ha='right')

        # Use log scale for y-axis
        ax.set_yscale('log')
        # Custom formatter to show only exact powers of 10 (10, 10^2, 10^3, etc.)
        def format_power_of_10(x, p):
            if x <= 0:
                return ''
            exponent = int(round(log10(x)))
            # Check if value is close to a power of 10 (within 1% tolerance)
            if abs(x - 10**exponent) / (10**exponent) < 0.01:
                return f'$10^{{{exponent}}}$'
            return ''
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_power_of_10))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        # Set major ticks only at powers of 10
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))

        # Find first and last cuts with non-zero total events to set x-axis limits
        # Calculate this BEFORE setting y-axis limits so we can use it for y-axis too
        total_events = np.zeros(n_cuts)
        for bg_cutflow in background_cutflows.values():
            for i, cut in enumerate(cuts):
                total_events[i] += bg_cutflow.get(cut, 0)

        # Set y-axis limits for cutflow: max=10^9, min=100 (as per ROOT logic)
        # But if max value is less than 100, adjust minimum
        max_value = np.max(total_events) if len(total_events) > 0 else 100
        ymin = min(100, max_value * 0.1) if max_value > 0 else 0.1
        ax.set_ylim(bottom=ymin, top=1000000000)

        # Always show all cuts on x-axis, regardless of whether they have zero events
        # This ensures all selection steps are visible in the cutflow plot
        ax.set_xlim(-0.5, n_cuts - 0.5)

        ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)

        # Add legend
        ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=False,
                 framealpha=1.0, edgecolor='black')

        # Add CMS label using mplhep
        hep.cms.label(ax=ax, data=False, lumi=lumi, year="Run3")

    else:
        # Single cutflow (backward compatibility)
        cuts = list(cutflow_dict.keys())
        events = list(cutflow_dict.values())

        if normalize and len(events) > 0:
            events = [e / events[0] for e in events]
            ylabel = r"Efficiency"
        else:
            ylabel = r"Events"

        fig, ax = plt.subplots(figsize=(10, 8))

        bars = ax.bar(range(len(cuts)), events, width=1.0, color='steelblue',
                     alpha=0.7, edgecolor='black', linewidth=0.5)

        ax.set_xlabel(r"Selection Step", labelpad=10)
        ax.set_ylabel(ylabel, labelpad=10)
        ax.set_xticks(range(len(cuts)))
        ax.set_xticklabels(cuts, rotation=45, ha='right')

        # Use log scale with powers of 10 only
        ax.set_yscale('log')
        # Custom formatter to show only exact powers of 10 (10, 10^2, 10^3, etc.)
        def format_power_of_10(x, p):
            if x <= 0:
                return ''
            exponent = int(round(log10(x)))
            # Check if value is close to a power of 10 (within 1% tolerance)
            if abs(x - 10**exponent) / (10**exponent) < 0.01:
                return f'$10^{{{exponent}}}$'
            return ''
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_power_of_10))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        # Set major ticks only at powers of 10
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))

        # Set y-axis limits for cutflow: max=10^9, min=100 (as per ROOT logic)
        ax.set_ylim(bottom=100, top=1000000000)

        # Find first and last cuts with non-zero events to set x-axis limits
        non_zero_indices = np.where(np.array(events) > 0)[0]

        if len(non_zero_indices) > 0:
            # Set x-axis limits to start from first non-zero cut and end at last non-zero cut
            first_cut_idx = non_zero_indices[0]
            last_cut_idx = non_zero_indices[-1]

            # Set x-axis limits: bars are at range(len(cuts)) with width=1.0
            # So we need to go from first_cut_idx - 0.5 to last_cut_idx + 0.5
            xmin = first_cut_idx - 0.5
            xmax = last_cut_idx + 0.5

            ax.set_xlim(xmin, xmax)

        ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)

        # Add CMS label using mplhep
        hep.cms.label(ax=ax, data=False, lumi=lumi, year="Run3")

    plt.tight_layout(pad=0.5)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved cutflow plot to {output_file}")
    # In batch mode, we don't show plots


def plot_histogram(histogram: Hist,
                  xlabel: Optional[str] = None,
                  ylabel: str = "Events",
                  title: Optional[str] = None,
                  output_file: Optional[str] = None,
                  logy: bool = False,
                  normalize: bool = False,
                  region_name: Optional[str] = None,
                  lumi: float = 290.0):
    """
    Plot histogram with CMS publication quality using mplhep.

    Parameters:
    -----------
    histogram : Hist
        Histogram object
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label (default: "Events")
    title : str
        Plot title (not used in CMS style, kept for compatibility)
    output_file : str
        Output filename (optional)
    logy : bool
        Use log scale for y-axis (default: False)
    normalize : bool
        Normalize histogram (default: False)
    region_name : str, optional
        Region name to display
    lumi : float
        Luminosity in fb^-1 (for CMS label)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    values = histogram.values()
    edges = histogram.axes[0].edges
    centers = (edges[:-1] + edges[1:]) / 2

    if normalize:
        integral = np.sum(values)
        if integral > 0:
            values = values / integral
        ylabel = r"Normalized Events"
    else:
        ylabel = r"Events"

    ax.hist(centers, bins=edges, weights=values,
           histtype='step', linewidth=2.5, color='steelblue')
    ax.fill_between(centers, 0, values, alpha=0.3, color='steelblue')

    max_value = np.max(values) if len(values) > 0 else 1.0

    if logy:
        ax.set_yscale('log')
        # Custom formatter to show only exact powers of 10 (10, 10^2, 10^3, etc.)
        def format_power_of_10(x, p):
            if x <= 0:
                return ''
            exponent = int(round(log10(x)))
            # Check if value is close to a power of 10 (within 1% tolerance)
            if abs(x - 10**exponent) / (10**exponent) < 0.01:
                return f'$10^{{{exponent}}}$'
            return ''
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_power_of_10))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        # Set major ticks only at powers of 10
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))

        # Set y-axis limits: max=max_value*1000, min=0.1 (as per ROOT logic)
        if max_value > 0:
            ax.set_ylim(bottom=0.1, top=max_value * 1000)
    else:
        # Linear scale: max=max_value*1.75, min=0 (as per ROOT logic)
        if max_value > 0:
            ax.set_ylim(bottom=0, top=max_value * 1.75)

    # Set axis labels with LaTeX formatting
    xlabel_text = xlabel or histogram.axes[0].label or "x"
    # Wrap in $ if it contains LaTeX syntax (underscores, subscripts, etc.)
    if "_" in xlabel_text or "{" in xlabel_text or "^" in xlabel_text:
        if not xlabel_text.startswith("$"):
            xlabel_text = f"${xlabel_text}$"
    ax.set_xlabel(xlabel_text, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)

    # Add CMS label using mplhep
    hep.cms.label(ax=ax, data=False, lumi=lumi, year="Run3")

    # Add region label if provided (below CMS label)
    if region_name:
        ax.text(0.02, 0.88, region_name, transform=ax.transAxes,
                fontsize=32, verticalalignment='top', style='italic')

    plt.tight_layout(pad=0.5)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved histogram plot to {output_file}")
    # In batch mode, we don't show plots


def plot_signal_vs_background(signal_hist: Optional[Hist] = None,
                             signal_hists: Optional[Dict[str, Hist]] = None,
                             background_hists: Dict[str, Hist] = None,
                             xlabel: Optional[str] = None,
                             title: Optional[str] = None,
                             output_file: Optional[str] = None,
                             logy: bool = True,
                             stack: bool = True,
                             region_name: Optional[str] = None,
                             lumi: float = 290.0):
    """
    Plot signal vs stacked backgrounds using mplhep for CMS styling.
    Supports overlaying multiple signal points.

    Parameters:
    -----------
    signal_hist : Hist or None
        Single signal histogram (for backward compatibility)
    signal_hists : dict, optional
        Dictionary of {label: histogram} for multiple signal points to overlay
        If provided, this takes precedence over signal_hist
    background_hists : dict
        Dictionary of {name: histogram} for backgrounds
    xlabel : str
        X-axis label
    title : str
        Plot title (not used in CMS style)
    output_file : str
        Output filename (optional)
    logy : bool
        Use log scale for y-axis (default: True)
    stack : bool
        Stack backgrounds (default: True)
    region_name : str, optional
        Region name to display
    lumi : float
        Luminosity in fb^-1 (for CMS label)
    """
    if not background_hists:
        print("Warning: No background histograms provided")
        return

    # CMS-style figure size
    fig, ax = plt.subplots(figsize=(10, 8))

    # Determine which signal(s) to use
    signals_to_plot = {}
    if signal_hists is not None and len(signal_hists) > 0:
        signals_to_plot = signal_hists
    elif signal_hist is not None:
        signals_to_plot = {"Signal": signal_hist}

    # Get common binning from first background (or signal if available)
    if signals_to_plot:
        first_sig = list(signals_to_plot.values())[0]
        edges = first_sig.axes[0].edges
        xlabel = xlabel or first_sig.axes[0].label or "x"
    else:
        first_bg = list(background_hists.values())[0]
        edges = first_bg.axes[0].edges
        xlabel = xlabel or first_bg.axes[0].label or "x"

    # Wrap xlabel in $ for LaTeX rendering if it contains LaTeX syntax
    if xlabel and ("_" in xlabel or "{" in xlabel or "^" in xlabel):
        if not xlabel.startswith("$"):
            xlabel = f"${xlabel}$"

    centers = (edges[:-1] + edges[1:]) / 2
    # Calculate width for each bin (handles variable binning)
    bin_widths = edges[1:] - edges[:-1]

    # Color scheme for backgrounds (using provided hex codes)
    bg_colors = {
        'ttbar': '#bd1f01',      # Red
        'wlnjets': '#3f90da',     # Blue
        'znnjets': '#832db6',     # Purple
        'dyjets': '#94a4a2',      # Gray
        'diboson': '#ffa90e',     # Orange
        'stop': '#a96b59',        # Brown (combined single top)
        'sTop_tchannel': '#a96b59',  # Brown (kept for backward compatibility)
        'sTop_tW': '#e76300',     # Orange-red (kept for backward compatibility)
    }

    # CMS label mapping (LaTeX formatted)
    bg_labels = {
        'ttbar': r'$t\bar{t}$',
        'wlnjets': r'$W$+jets',
        'znnjets': r'$Z$+jets',
        'dyjets': r'$DY$+jets',
        'diboson': r'Diboson',
        'stop': r'Single top',
        'sTop_tchannel': r'Single top',  # Kept for backward compatibility
        'sTop_tW': r'Single top ($tW$)',  # Kept for backward compatibility
    }

    # Plot backgrounds (stacked)
    if stack:
        bottom = np.zeros(len(centers))
        for i, (name, bg_hist) in enumerate(background_hists.items()):
            bg_values = bg_hist.values()
            color = bg_colors.get(name, plt.cm.Set3(i / len(background_hists)))
            label = bg_labels.get(name, name)
            # Use bar plot for stacked histograms with proper bin widths (handles variable binning)
            # ax.bar accepts width as an array for variable binning
            ax.bar(centers, bg_values, width=bin_widths,
                  bottom=bottom, label=label, color=color, alpha=0.85)
            bottom += bg_values
    else:
        for i, (name, bg_hist) in enumerate(background_hists.items()):
            bg_values = bg_hist.values()
            color = bg_colors.get(name, plt.cm.Set3(i / len(background_hists)))
            label = bg_labels.get(name, name)
            ax.hist(centers, bins=edges, weights=bg_values,
                   histtype='step', linewidth=2.5, label=label, color=color)

    # Plot signal(s) if provided - overlay multiple signals with different colors/linestyles
    signal_colors = ['#E24A33', '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4749']
    signal_linestyles = ['-', '--', '-.', ':', '-', '--', '-.']

    for i, (sig_label, sig_hist) in enumerate(signals_to_plot.items()):
        signal_values = sig_hist.values()
        color = signal_colors[i % len(signal_colors)]
        linestyle = signal_linestyles[i % len(signal_linestyles)]
        ax.hist(centers, bins=edges, weights=signal_values,
               histtype='step', linewidth=3.0, label=sig_label,
               color=color, linestyle=linestyle, zorder=10)

    # Calculate maximum value across all histograms (signal + stacked backgrounds)
    max_value = 0.0
    for sig_hist in signals_to_plot.values():
        signal_vals = sig_hist.values()
        if len(signal_vals) > 0:
            max_value = max(max_value, np.max(signal_vals))

    if stack:
        # For stacked plots, sum all backgrounds to get total
        stacked_total = np.zeros(len(centers))
        for bg_hist in background_hists.values():
            stacked_total += bg_hist.values()
        if len(stacked_total) > 0:
            max_value = max(max_value, np.max(stacked_total))
    else:
        # For non-stacked plots, take max of individual backgrounds
        for bg_hist in background_hists.values():
            bg_vals = bg_hist.values()
            if len(bg_vals) > 0:
                max_value = max(max_value, np.max(bg_vals))

    # Only use log scale if there's positive data
    if logy and max_value > 0:
        try:
            ax.set_yscale('log')
            # Custom formatter to show only exact powers of 10 (10, 10^2, 10^3, etc.)
            def format_power_of_10(x, p):
                if x <= 0:
                    return ''
                exponent = int(round(log10(x)))
                # Check if value is close to a power of 10 (within 1% tolerance)
                if abs(x - 10**exponent) / (10**exponent) < 0.01:
                    return f'$10^{{{exponent}}}$'
                return ''
            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_power_of_10))
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            # Set major ticks only at powers of 10
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))

            # Set y-axis limits: max=max_value*1000, min=0.1 (as per ROOT logic)
            if max_value > 0:
                ax.set_ylim(bottom=0.1, top=max_value * 1000)
        except (ValueError, RuntimeError):
            # Fall back to linear scale if log scale fails
            ax.set_yscale('linear')
            # For linear scale: max=max_value*1.75, min=0 (as per ROOT logic)
            if max_value > 0:
                ax.set_ylim(bottom=0, top=max_value * 1.75)
    else:
        # Linear scale: max=max_value*1.75, min=0 (as per ROOT logic)
        if max_value > 0:
            ax.set_ylim(bottom=0, top=max_value * 1.75)

    # Set axis labels with LaTeX formatting
    # xlabel should already be wrapped in $ from above
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(r"Events", labelpad=10)

    # Add CMS label using mplhep
    hep.cms.label(ax=ax, data=False, lumi=lumi, year="Run3")

    # Add region label if provided (below CMS label)
    if region_name:
        ax.text(0.02, 0.88, region_name, transform=ax.transAxes,
                fontsize=32, verticalalignment='top', style='italic')

    # Set x-axis limits to show full range from first to last bin edge
    # This ensures variable binning (like [250, 300, 400, 550, 1000]) shows the full range
    xmin = edges[0]
    xmax = edges[-1]
    ax.set_xlim(xmin, xmax)

    # Set x-axis ticks to show bin edges for variable binning (makes it clear where bins are)
    # For variable binning, show all edges; for regular binning, show fewer ticks
    if len(edges) <= 6:  # Variable binning (typically 4-5 edges)
        ax.set_xticks(edges)
        ax.set_xticklabels([f'{int(e)}' if e >= 1 else f'{e:.2f}' for e in edges])
    else:  # Regular binning - show fewer ticks
        # Show first, middle, and last edges
        tick_indices = [0, len(edges)//2, len(edges)-1]
        ax.set_xticks([edges[i] for i in tick_indices])

    # Legend (mplhep style handles most of this, but we customize)
    legend = ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=False,
                      ncol=1, framealpha=1.0, edgecolor='black')
    legend.get_frame().set_linewidth(1.5)

    plt.tight_layout(pad=0.5)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved plot to {output_file}")
    # In batch mode, we don't show plots


def plot_mass_scan(ma_values: List[float],
                  mbb_peaks: List[float],
                  output_file: Optional[str] = None,
                  lumi: float = 290.0):
    """
    Plot m(bb) peak vs ma (for resonance studies) with CMS styling.

    Parameters:
    -----------
    ma_values : List[float]
        List of ma values
    mbb_peaks : List[float]
        List of m(bb) peak values
    output_file : str
        Output filename (optional)
    lumi : float
        Luminosity in fb^-1 (for CMS label)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(ma_values, mbb_peaks, 'o-', linewidth=2, markersize=8, color='red', label=r'$m(bb)$ peak')
    ax.plot(ma_values, ma_values, '--', linewidth=1, color='gray', alpha=0.5, label=r'$m_a = m(bb)$')

    ax.set_xlabel(r"$m_a$ [GeV]", labelpad=10)
    ax.set_ylabel(r"$m(bb)$ peak [GeV]", labelpad=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # Add CMS label using mplhep
    hep.cms.label(ax=ax, data=False, lumi=lumi, year="Run3")

    plt.tight_layout(pad=0.5)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved mass scan plot to {output_file}")
    # In batch mode, we don't show plots


def plot_efficiency_curve(ma_values: List[float],
                         efficiencies: List[float],
                         output_file: Optional[str] = None,
                         lumi: float = 290.0):
    """
    Plot efficiency curve vs ma with CMS styling.

    Parameters:
    -----------
    ma_values : List[float]
        List of ma values
    efficiencies : List[float]
        List of efficiencies
    output_file : str
        Output filename (optional)
    lumi : float
        Luminosity in fb^-1 (for CMS label)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(ma_values, efficiencies, 'o-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel(r"$m_a$ [GeV]", labelpad=10)
    ax.set_ylabel(r"Efficiency", labelpad=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.1)

    # Add CMS label using mplhep
    hep.cms.label(ax=ax, data=False, lumi=lumi, year="Run3")

    plt.tight_layout(pad=0.5)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved efficiency plot to {output_file}")
    # In batch mode, we don't show plots


if __name__ == "__main__":
    """
    Example usage of plotting utilities.
    """
    print("Plotting Utilities Module")
    print("=" * 50)
    print("\nThis module provides publication-quality plotting functions:")
    print("1. Cutflow plots")
    print("2. Histogram plots")
    print("3. Signal vs background plots")
    print("4. Mass scan plots")
    print("5. Efficiency curves")
