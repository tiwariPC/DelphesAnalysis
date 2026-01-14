#!/usr/bin/env python3
"""
Process signal and background samples for all regions defined in regions.yaml.

Generates region-specific plots and datacards following CMS guidelines.
Optimized: Load each ROOT file once, process all regions.
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import uproot
import numpy as np
import awkward as ak

from src.bbdmDelphes import (
    load_delphes_events,
    build_jets,
    select_bjets,
    build_met,
    build_electrons,
    build_muons,
    calculate_mt,
    calculate_recoil,
    calculate_mll,
    calculate_observables,
    generate_datacard,
    save_datacard,
    create_shapes_file
)
from src.regions import load_regions, get_region_type, get_region_category
from src.plotting_utils import plot_cutflow, plot_histogram, plot_signal_vs_background
from collections import OrderedDict
import hist

# Binning configuration
# Variable binning based on region category:
# - 1b regions: MET/Recoil with [250, 300, 400, 550, 1000] (4 bins)
# - 2b regions: cost_star with [0.0, 0.25, 0.50, 0.75, 1.0] (4 bins)
PLOT_BINNING = {
    # For 1b regions: variable binning for MET/Recoil
    "met": {"bins": [250, 300, 400, 550, 1000], "variable": True},  # 4 bins for 1b regions
    "recoil": {"bins": [250, 300, 400, 550, 1000], "variable": True},  # 4 bins for 1b regions
    # For 2b regions: variable binning for cost_star
    "cost_star": {"bins": [0.0, 0.25, 0.50, 0.75, 1.0], "variable": True},  # 4 bins for 2b regions
    # Other observables use regular binning
    "mbb": {"range": (0, 2000), "bins": 25, "variable": False},
    "ptbb": {"range": (0, 800), "bins": 20, "variable": False},
    "dphi_jet_met_min": {"range": (0, np.pi), "bins": 20, "variable": False},
    "ht": {"range": (0, 2000), "bins": 25, "variable": False},
    "mt": {"range": (0, 300), "bins": 20, "variable": False},
    "mll": {"range": (70, 110), "bins": 20, "variable": False},
    "lep1_pt": {"range": (30, 500), "bins": 30, "variable": False},  # Leading lepton pT (matches StackPlotter: 30-500 GeV)
    "min_dphi": {"range": (0, 3.2), "bins": 32, "variable": False},  # Alias for dphi_jet_met_min (matches StackPlotter: 0-3.2)
}

def load_background_cross_sections(config_file="config/background_cross_sections.yaml"):
    """
    Load background cross-sections from YAML file.

    Parameters:
    -----------
    config_file : str
        Path to background cross-sections YAML file

    Returns:
    --------
    background_xs : dict
        Dictionary mapping bg_name -> cross_section (pb)
    """
    background_xs = {}
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        if config and 'backgrounds' in config:
            background_xs = {k: float(v) for k, v in config['backgrounds'].items()}
    except FileNotFoundError:
        print(f"  ⚠ Warning: Background cross-sections file not found: {config_file}")
    except Exception as e:
        print(f"  ⚠ Warning: Error loading background cross-sections: {e}")

    return background_xs


def load_signal_cross_sections(config_file="config/signal_cross_sections.yaml"):
    """
    Load signal cross-sections from YAML file.

    Parameters:
    -----------
    config_file : str
        Path to signal cross-sections YAML file

    Returns:
    --------
    signal_xs : dict
        Dictionary mapping (mA, ma) -> cross_section (pb)
        Format: {(mA_value, ma_value): xs_value}
    """
    signal_xs = {}
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        if config and 'signals' in config:
            for mA, ma_dict in config['signals'].items():
                mA_int = int(mA)
                for ma, xs in ma_dict.items():
                    ma_int = int(ma)
                    signal_xs[(mA_int, ma_int)] = float(xs)
    except FileNotFoundError:
        print(f"  ⚠ Warning: Signal cross-sections file not found: {config_file}")
    except Exception as e:
        print(f"  ⚠ Warning: Error loading signal cross-sections: {e}")

    return signal_xs


def load_samples_config(config_file):
    """
    Load background samples configuration from YAML file.
    Only loads file paths - cross-sections are loaded separately from background_cross_sections.yaml

    Returns:
        tuple: (background_files_dict, background_xs_dict)
            - background_files_dict: dict mapping bg_name -> file_path
            - background_xs_dict: dict mapping bg_name -> cross_section (pb) - empty dict, use load_background_cross_sections() instead
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    background_files = {}

    if 'backgrounds' in config:
        for bg_name, bg_config in config['backgrounds'].items():
            if 'file' in bg_config:
                background_files[bg_name] = bg_config['file']
            # Note: cross_section removed from samples_config.yaml - use background_cross_sections.yaml instead

    return background_files, {}  # Return empty dict for xs - must load separately


def parse_xsec_file(xsec_file):
    """
    Parse cross-section file (legacy support).
    Kept for backward compatibility if xsec-file is still provided.
    """
    xs_dict = {}
    with open(xsec_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '---' in line:
                continue
            if 'cross_section' in line.lower() or 'background' in line.lower():
                continue
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                try:
                    xs = float(parts[1])
                    xs_dict[name] = xs
                except ValueError:
                    continue
    return xs_dict


def count_events_in_root_file(root_file, tree_name="Delphes"):
    """Count number of events in a ROOT file."""
    try:
        with uproot.open(root_file) as file:
            tree = file[tree_name]
            return tree.num_entries
    except Exception as e:
        print(f"  Warning: Could not count events: {e}")
        return None


def load_sample_and_build_objects(file_path, cuts_config=None):
    """
    Load events and build all physics objects once per sample.
    Returns dict with events, jets, bjets, met, electrons, muons, additional_vars
    """
    try:
        events = load_delphes_events(file_path)
    except Exception as e:
        print(f"  ✗ Error loading events: {e}")
        return None

    from src.bbdmDelphes import load_cuts_from_yaml, get_default_cuts
    if cuts_config:
        cuts = load_cuts_from_yaml(cuts_config)
    else:
        cuts = get_default_cuts()

    jets = build_jets(events, pt_min=cuts["jet_pt_min"], eta_max=cuts["jet_eta_max"])

    # Preselection: require at least one jet with pt > 30
    njets = ak.num(jets['PT'], axis=1)
    preselection_mask = njets >= 1

    # Apply preselection to all collections
    events = events[preselection_mask]
    jets = {k: v[preselection_mask] for k, v in jets.items()}

    bjets = select_bjets(jets, btag_threshold=cuts["btag_threshold"])
    met = build_met(events)
    electrons = build_electrons(events)
    muons = build_muons(events)

    additional_vars = {}
    nleptons = ak.zeros_like(ak.num(jets['PT'], axis=1))
    if electrons is not None:
        nleptons = nleptons + ak.num(electrons['PT'], axis=1)
    if muons is not None:
        nleptons = nleptons + ak.num(muons['PT'], axis=1)

    # Calculate MT and Recoil
    if ak.sum(nleptons >= 1) > 0:
        el_pt = electrons['PT'] if electrons is not None else None
        el_phi = electrons['Phi'] if electrons is not None else None
        mu_pt = muons['PT'] if muons is not None else None
        mu_phi = muons['Phi'] if muons is not None else None

        arrays_to_concat_pt = []
        arrays_to_concat_phi = []
        if el_pt is not None:
            arrays_to_concat_pt.append(el_pt)
            arrays_to_concat_phi.append(el_phi)
        if mu_pt is not None:
            arrays_to_concat_pt.append(mu_pt)
            arrays_to_concat_phi.append(mu_phi)

        if len(arrays_to_concat_pt) > 0:
            combined_leptons_pt = ak.concatenate(arrays_to_concat_pt, axis=1)
            combined_leptons_phi = ak.concatenate(arrays_to_concat_phi, axis=1)
        else:
            n_events = len(nleptons)
            combined_leptons_pt = ak.Array([[]] * n_events)
            combined_leptons_phi = ak.Array([[]] * n_events)

        mt = calculate_mt(combined_leptons_pt, combined_leptons_phi, met['MET'], met['Phi'])
        recoil = calculate_recoil(combined_leptons_pt, combined_leptons_phi, met['MET'], met['Phi'])
        additional_vars['MT'] = mt
        additional_vars['Recoil'] = recoil

    # Calculate Mll
    if ak.sum(nleptons >= 2) > 0:
        el_pt = electrons['PT'] if electrons is not None else None
        el_eta = electrons['Eta'] if electrons is not None else None
        el_phi = electrons['Phi'] if electrons is not None else None
        el_charge = electrons['Charge'] if electrons is not None else None
        mu_pt = muons['PT'] if muons is not None else None
        mu_eta = muons['Eta'] if muons is not None else None
        mu_phi = muons['Phi'] if muons is not None else None
        mu_charge = muons['Charge'] if muons is not None else None

        arrays_to_concat_pt = []
        arrays_to_concat_eta = []
        arrays_to_concat_phi = []
        arrays_to_concat_charge = []
        if el_pt is not None and ak.sum(ak.num(el_pt, axis=1) > 0) > 0:
            arrays_to_concat_pt.append(el_pt)
            arrays_to_concat_eta.append(el_eta)
            arrays_to_concat_phi.append(el_phi)
            arrays_to_concat_charge.append(el_charge)
        if mu_pt is not None and ak.sum(ak.num(mu_pt, axis=1) > 0) > 0:
            arrays_to_concat_pt.append(mu_pt)
            arrays_to_concat_eta.append(mu_eta)
            arrays_to_concat_phi.append(mu_phi)
            arrays_to_concat_charge.append(mu_charge)

        if len(arrays_to_concat_pt) > 0:
            all_leptons_pt = ak.concatenate(arrays_to_concat_pt, axis=1)
            all_leptons_eta = ak.concatenate(arrays_to_concat_eta, axis=1)
            all_leptons_phi = ak.concatenate(arrays_to_concat_phi, axis=1)
            all_leptons_charge = ak.concatenate(arrays_to_concat_charge, axis=1)

            sorted_indices = ak.argsort(all_leptons_pt, axis=1, ascending=False)
            sorted_pt = all_leptons_pt[sorted_indices]
            sorted_eta = all_leptons_eta[sorted_indices]
            sorted_phi = all_leptons_phi[sorted_indices]
            sorted_charge = all_leptons_charge[sorted_indices]

            lepton1_pt = ak.firsts(sorted_pt, axis=1)
            lepton1_eta = ak.firsts(sorted_eta, axis=1)
            lepton1_phi = ak.firsts(sorted_phi, axis=1)
            lepton1_charge = ak.firsts(sorted_charge, axis=1)

            padded_pt = ak.pad_none(sorted_pt, 2, axis=1)
            padded_eta = ak.pad_none(sorted_eta, 2, axis=1)
            padded_phi = ak.pad_none(sorted_phi, 2, axis=1)
            padded_charge = ak.pad_none(sorted_charge, 2, axis=1)

            try:
                lepton2_pt = ak.fill_none(padded_pt[:, 1], 0.0)
                lepton2_eta = ak.fill_none(padded_eta[:, 1], 0.0)
                lepton2_phi = ak.fill_none(padded_phi[:, 1], 0.0)
                lepton2_charge = ak.fill_none(padded_charge[:, 1], 0)
            except (IndexError, TypeError):
                lepton2_pt = ak.fill_none(ak.firsts(padded_pt[:, 1:], axis=1), 0.0)
                lepton2_eta = ak.fill_none(ak.firsts(padded_eta[:, 1:], axis=1), 0.0)
                lepton2_phi = ak.fill_none(ak.firsts(padded_phi[:, 1:], axis=1), 0.0)
                lepton2_charge = ak.fill_none(ak.firsts(padded_charge[:, 1:], axis=1), 0)

            lepton1_pt = ak.fill_none(lepton1_pt, 0.0)
            lepton1_eta = ak.fill_none(lepton1_eta, 0.0)
            lepton1_phi = ak.fill_none(lepton1_phi, 0.0)
            lepton1_charge = ak.fill_none(lepton1_charge, 0)

            mll = calculate_mll(lepton1_pt, lepton1_eta, lepton1_phi, lepton1_charge,
                               lepton2_pt, lepton2_eta, lepton2_phi, lepton2_charge)
            additional_vars['Mll'] = mll

    return {
        "events": events,
        "jets": jets,
        "bjets": bjets,
        "met": met,
        "electrons": electrons,
        "muons": muons,
        "additional_vars": additional_vars
    }


def calculate_dphi_jet_met_min_all_events(jets, met):
    """
    Calculate dphi_jet_met_min for all events (not filtered).
    This is needed for the cutflow to work correctly.

    Computes: min([DeltaPhi(jet_phi, met_phi) for jet_phi in jets]) per event

    Parameters:
    -----------
    jets : dict
        Jet collection with 'PT', 'Eta', 'Phi', 'BTag' keys
    met : dict
        MET collection with 'MET' and 'Phi' keys

    Returns:
    --------
    dphi_jet_met_min : np.ndarray
        Array of minimum dphi(jet, MET) per event (NaN if no jets)
    """
    from src.bbdmDelphes import DeltaPhi

    # Get MET phi (single value per event) and jets phi
    met_phi = ak.flatten(met['Phi'], axis=1)
    jets_phi = jets['Phi']

    # Expand MET phi to match jets structure: one MET phi per jet
    n_jets_per_event = ak.num(jets_phi, axis=1)
    met_phi_np = np.asarray(ak.to_numpy(met_phi))
    met_phi_expanded = ak.unflatten(ak.Array(np.repeat(met_phi_np, ak.to_numpy(n_jets_per_event))), n_jets_per_event)

    # Calculate Δφ for all jets: DeltaPhi(jet_phi, met_phi) for each jet
    dphi_all_jets = DeltaPhi(jets_phi, met_phi_expanded)

    # Get minimum dphi per event (NaN if no jets)
    min_dphi_per_event = ak.min(dphi_all_jets, axis=1)
    dphi_jet_met = np.asarray(ak.to_numpy(ak.fill_none(min_dphi_per_event, np.nan)))

    return dphi_jet_met


def process_sample_for_region(sample_name, events, jets, bjets, met, electrons, muons,
                              additional_vars, xs_pb, ngen, region_name, regions_config,
                              lumi_fb=139.0, output_dir=None, cuts_config=None):
    """Process a sample for a specific region using pre-loaded events and objects."""
    # Build cutflow step by step - apply cuts in the order they appear in region config
    from src.regions import parse_cut_condition
    from collections import OrderedDict

    cutflow = OrderedDict()
    n_events = len(events)

    # Apply cuts sequentially in the order they appear in the region configuration
    region_cuts_raw = regions_config[region_name].get('cuts', {})
    # Ensure region_cuts is an OrderedDict to preserve order
    if not isinstance(region_cuts_raw, OrderedDict):
        region_cuts = OrderedDict(region_cuts_raw)
    else:
        region_cuts = region_cuts_raw

    # Check for duplicate cut names (YAML dicts shouldn't have duplicates, but verify)
    cut_names_list = list(region_cuts.keys())
    if len(cut_names_list) != len(set(cut_names_list)):
        seen = set()
        duplicates = []
        for name in cut_names_list:
            if name in seen:
                duplicates.append(name)
            seen.add(name)
        if duplicates:
            print(f"  ⚠ WARNING: Duplicate cut names found in {region_name}: {duplicates}")
            print(f"     Only the last occurrence of each duplicate will be used!")

    mask = ak.Array(np.ones(n_events, dtype=bool))

    # Pre-calculate common variables for ALL events (needed for cutflow)
    njets = ak.num(jets['PT'], axis=1)
    nbjets = ak.num(bjets['PT'], axis=1)
    met_values = ak.flatten(met['MET'], axis=1)

    # Pre-calculate dphi_jet_met_min for ALL events (needed for cutflow)
    dphi_jet_met_min_all = calculate_dphi_jet_met_min_all_events(jets, met)


    # Calculate lepton counts once
    nleptons = ak.zeros_like(njets)
    if electrons is not None:
        nleptons = nleptons + ak.num(electrons.get('PT', ak.Array([])), axis=1)
    if muons is not None:
        nleptons = nleptons + ak.num(muons.get('PT', ak.Array([])), axis=1)

    # Calculate weight for scaling cutflow to match weighted histograms
    # Weight = (cross_section * luminosity * 1000) / number_of_generated_events
    weight = (xs_pb * lumi_fb * 1000.0) / ngen if ngen > 0 else 1.0

    # Apply cuts cumulatively and record weighted counts in cutflow
    handled_cuts = set()
    for cut_name, cut_value in region_cuts.items():
        if cut_name in handled_cuts:
            print(f"  ⚠ WARNING: Cut '{cut_name}' appears multiple times in {region_name} - only last value used")
        handled_cuts.add(cut_name)
        if cut_name == 'Njets':
            mask = mask & parse_cut_condition(cut_value, njets)
            cutflow["Njets"] = ak.sum(mask) * weight

        elif cut_name == 'Nbjets':
            mask = mask & parse_cut_condition(cut_value, nbjets)
            cutflow["Nbjets"] = ak.sum(mask) * weight

        elif cut_name == 'Nleptons':
            mask = mask & parse_cut_condition(cut_value, nleptons)
            cutflow["Nleptons"] = ak.sum(mask) * weight

        elif cut_name == 'MET':
            mask = mask & parse_cut_condition(cut_value, met_values)
            cutflow["MET"] = ak.sum(mask) * weight

        elif cut_name == 'Recoil' and 'Recoil' in additional_vars:
            recoil = additional_vars.get('Recoil')
            if recoil is not None:
                mask = mask & parse_cut_condition(cut_value, recoil)
                cutflow["Recoil"] = ak.sum(mask) * weight

        elif cut_name == 'MT' and 'MT' in additional_vars:
            mt = additional_vars.get('MT')
            if mt is not None:
                mask = mask & parse_cut_condition(cut_value, mt)
                cutflow["MT"] = ak.sum(mask) * weight

        elif cut_name == 'MllMin' and 'Mll' in additional_vars:
            mll = additional_vars.get('Mll')
            if mll is not None:
                if isinstance(mll, ak.Array):
                    mll_np = np.asarray(ak.to_numpy(mll))
                else:
                    mll_np = np.asarray(mll)
                threshold = float(cut_value.replace('>', ''))
                mask = mask & ak.Array(mll_np >= threshold)
                cutflow["MllMin"] = ak.sum(mask) * weight

        elif cut_name == 'MllMax' and 'Mll' in additional_vars:
            mll = additional_vars.get('Mll')
            if mll is not None:
                if isinstance(mll, ak.Array):
                    mll_np = np.asarray(ak.to_numpy(mll))
                else:
                    mll_np = np.asarray(mll)
                threshold = float(cut_value.replace('<', ''))
                mask = mask & ak.Array(mll_np <= threshold)
                cutflow["MllMax"] = ak.sum(mask) * weight

        elif cut_name == 'NAdditionalJets':
            nadditional = njets - nbjets
            mask = mask & parse_cut_condition(cut_value, nadditional)
            cutflow["NAdditionalJets"] = ak.sum(mask) * weight

        elif cut_name == 'Jet1Pt':
            jet1_pt = ak.firsts(jets['PT'], axis=1)
            jet1_pt = ak.fill_none(jet1_pt, 0.0)
            mask = mask & parse_cut_condition(cut_value, jet1_pt)
            cutflow["Jet1Pt"] = ak.sum(mask) * weight

        elif cut_name == 'dphi_jet_met_min':
            valid_dphi = ~np.isnan(dphi_jet_met_min_all)
            dphi_mask = np.zeros(n_events, dtype=bool)
            if np.sum(valid_dphi) > 0:
                dphi_valid = parse_cut_condition(cut_value, dphi_jet_met_min_all[valid_dphi])
                if isinstance(dphi_valid, ak.Array):
                    dphi_mask[valid_dphi] = np.asarray(ak.to_numpy(dphi_valid))
                else:
                    dphi_mask[valid_dphi] = np.asarray(dphi_valid)
            mask = mask & ak.Array(dphi_mask)
            cutflow["dphi_jet_met_min"] = ak.sum(mask) * weight

        else:
            print(f"  ⚠ WARNING: Cut '{cut_name}' in {region_name} is not handled in cutflow logic!")

    n_selected = ak.sum(mask)

    # Calculate observables for selected events
    selected_jets = {k: v[mask] for k, v in jets.items()}
    selected_bjets = {k: v[mask] for k, v in bjets.items()}
    selected_met = {k: v[mask] for k, v in met.items()}
    obs = calculate_observables(selected_jets, selected_bjets, selected_met, ak.Array(np.ones(ak.sum(mask), dtype=bool)))

    for key in additional_vars:
        if key in ['MT', 'Recoil', 'Mll']:
            obs[key.lower()] = additional_vars[key][mask]

    # Add lepton-specific observables for W control regions (matching StackPlotter)
    if 'CR_Wlnu' in region_name or 'CR_Top' in region_name:
        # Get leading lepton pT (either electron or muon, whichever has higher pT)
        lep1_pt = np.full(n_selected, np.nan)

        if electrons is not None and 'PT' in electrons:
            el_pt = ak.firsts(electrons['PT'][mask], axis=1)
            el_pt = ak.fill_none(el_pt, np.nan)
            el_pt_np = np.asarray(ak.to_numpy(el_pt))
        else:
            el_pt_np = np.full(n_selected, np.nan)

        if muons is not None and 'PT' in muons:
            mu_pt = ak.firsts(muons['PT'][mask], axis=1)
            mu_pt = ak.fill_none(mu_pt, np.nan)
            mu_pt_np = np.asarray(ak.to_numpy(mu_pt))
        else:
            mu_pt_np = np.full(n_selected, np.nan)

        # Take the maximum (leading lepton) - handles cases where both exist
        lep1_pt = np.maximum(np.nan_to_num(el_pt_np, nan=-1), np.nan_to_num(mu_pt_np, nan=-1))
        lep1_pt[lep1_pt < 0] = np.nan  # Restore NaN where both were NaN
        obs['lep1_pt'] = lep1_pt

    weights = np.ones(n_selected) * weight

    histograms = {}
    region_type = get_region_type(region_name)
    region_category = get_region_category(region_name)
    if region_category == "2b":
        sr_observables = ["met", "mbb", "ptbb", "dphi_jet_met_min", "ht", "cost_star"]
        cr_observables = ["recoil", "mt", "mll", "mbb", "ptbb", "dphi_jet_met_min", "ht", "cost_star"]
    else:  # 1b category
        sr_observables = ["met", "mbb", "ptbb", "dphi_jet_met_min", "ht"]
        cr_observables = ["recoil", "mt", "mll", "mbb", "ptbb", "dphi_jet_met_min", "ht"]

    # Add lepton-specific observables for W and Top control regions (matching StackPlotter)
    if 'CR_Wlnu' in region_name or 'CR_Top' in region_name:
        cr_observables.extend(["lep1_pt", "min_dphi"])

    observables_to_plot = sr_observables if region_type == "SR" else cr_observables

    for obs_name in observables_to_plot:
        # Handle alias: min_dphi uses dphi_jet_met_min data
        if obs_name == "min_dphi":
            if "dphi_jet_met_min" not in obs:
                continue
            obs_data = obs["dphi_jet_met_min"]
            obs_name_for_binning = "min_dphi"
        else:
            if obs_name not in obs:
                continue
            obs_data = obs[obs_name]
            obs_name_for_binning = obs_name

        if obs_name_for_binning not in PLOT_BINNING:
            continue

        binning = PLOT_BINNING[obs_name_for_binning]
        valid = ~np.isnan(obs_data)
        if np.sum(valid) > 0:
            label_map = {
                "met": r"$E_T^{\rm miss}$ [GeV]",
                "mbb": r"$m(bb)$ [GeV]",
                "ptbb": r"$p_T(bb)$ [GeV]",
                "dphi_jet_met_min": r"$\min(\Delta\phi(\rm jet, E_T^{\rm miss}))$",
                "min_dphi": r"$\min(\Delta\phi(\rm jet, E_T^{\rm miss}))$",  # Alias
                "ht": r"$H_T$ [GeV]",
                "cost_star": r"$|\cos\theta^{*}|$",
                "recoil": r"Recoil [GeV]",
                "mt": r"$M_T$ [GeV]",
                "mll": r"$m(\ell\ell)$ [GeV]",
                "lep1_pt": r"$p_T(\ell_1)$ [GeV]",
            }
            label = label_map.get(obs_name, obs_name)

            if binning.get("variable", False):
                bin_edges = np.array(binning["bins"])
                histograms[obs_name] = hist.Hist(
                    hist.axis.Variable(bin_edges, name=obs_name, label=label)
                )
                # Get the last bin edge for overflow handling
                last_edge = bin_edges[-1]
            else:
                histograms[obs_name] = hist.Hist(
                    hist.axis.Regular(binning["bins"], binning["range"][0], binning["range"][1],
                                    name=obs_name, label=label)
                )
                # Get the last bin edge for overflow handling
                last_edge = binning["range"][1]

            # Handle overflow: clip values >= last_edge to a value in the last bin
            # This ensures overflow events are added to the last bin
            # Convert to numpy for easier manipulation
            if isinstance(obs_data, ak.Array):
                data_to_fill = np.asarray(ak.to_numpy(obs_data[valid]))
            else:
                data_to_fill = np.asarray(obs_data[valid]).copy()
            weights_to_fill = weights[valid].copy()

            # Find overflow events (values >= last_edge)
            overflow_mask = data_to_fill >= last_edge
            if np.sum(overflow_mask) > 0:
                # Clip overflow values to a value that's definitely in the last bin
                # For variable binning [..., 550, 1000], last bin is [550, 1000)
                # Use a value just below last_edge to ensure it goes to last bin
                if binning.get("variable", False):
                    # For variable binning, last bin is [bin_edges[-2], bin_edges[-1])
                    # Use a value just below the upper edge to put overflow in the last bin
                    data_to_fill[overflow_mask] = last_edge - 1e-6
                else:
                    # For regular binning, use a value just below the upper range
                    data_to_fill[overflow_mask] = last_edge - 1e-6

            # Fill histogram (overflow events will now be in the last bin)
            histograms[obs_name].fill(**{obs_name: data_to_fill}, weight=weights_to_fill)

    return {
        "cutflow": cutflow,
        "histograms": histograms,
        "observables": obs,
        "n_selected": n_selected  # Store separately, not in cutflow for plotting
    }


def main():
    parser = argparse.ArgumentParser(description="Process samples for all regions")
    parser.add_argument("--signal-file", type=str, action='append', help="Signal ROOT file (can be specified multiple times for overlaying)")
    parser.add_argument("--signal-xs", type=float, action='append', help="Signal cross-section in pb (one per --signal-file, or use --signal-mA/--signal-ma)")
    parser.add_argument("--signal-mA", type=int, action='append', help="Signal mA value (for cross-section lookup, one per --signal-file)")
    parser.add_argument("--signal-ma", type=int, action='append', help="Signal ma value (for cross-section lookup, one per --signal-file)")
    parser.add_argument("--signal-ngen", type=int, action='append', help="Signal number of generated events (one per --signal-file)")
    parser.add_argument("--signal-label", type=str, action='append', help="Signal label for legend (one per --signal-file, default: 'Signal (mA, ma)')")
    parser.add_argument("--samples-config", type=str,
                       default="config/samples_config.yaml",
                       help="Samples configuration file (YAML with background files and cross-sections)")
    parser.add_argument("--background-xsec", type=str,
                       default="config/background_cross_sections.yaml",
                       help="Background cross-sections YAML file")
    parser.add_argument("--signal-xsec", type=str,
                       default="config/signal_cross_sections.yaml",
                       help="Signal cross-sections YAML file")
    parser.add_argument("--xsec-file", type=str, default=None,
                       help="Background cross-section file (legacy, overrides YAML if provided)")
    parser.add_argument("--lumi", type=float, default=139.0, help="Luminosity in fb^-1")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--cuts-config", type=str, default="config/cuts_config.yaml", help="Cuts configuration file")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    regions_config = load_regions(args.cuts_config)
    background_files, _ = load_samples_config(args.samples_config)  # Only load file paths

    # Load background cross-sections from dedicated file (single source of truth)
    bg_xs = load_background_cross_sections(args.background_xsec)

    # Legacy support: override with xsec-file if provided
    if args.xsec_file:
        legacy_xs = parse_xsec_file(args.xsec_file)
        for bg_name, xs_value in legacy_xs.items():
            bg_xs[bg_name] = xs_value

    # Process multiple signals if provided
    if not args.signal_file:
        print("  ✗ Error: Must provide at least one --signal-file")
        return

    signal_files = args.signal_file
    signal_xs_list = args.signal_xs or [None] * len(signal_files)
    signal_mA_list = args.signal_mA or [None] * len(signal_files)
    signal_ma_list = args.signal_ma or [None] * len(signal_files)
    signal_ngen_list = args.signal_ngen or [None] * len(signal_files)
    signal_labels = args.signal_label or [None] * len(signal_files)

    # Ensure all lists have the same length
    n_signals = len(signal_files)
    if len(signal_xs_list) != n_signals:
        signal_xs_list = signal_xs_list[:n_signals] if len(signal_xs_list) > n_signals else signal_xs_list + [None] * (n_signals - len(signal_xs_list))
    if len(signal_mA_list) != n_signals:
        signal_mA_list = signal_mA_list[:n_signals] if len(signal_mA_list) > n_signals else signal_mA_list + [None] * (n_signals - len(signal_mA_list))
    if len(signal_ma_list) != n_signals:
        signal_ma_list = signal_ma_list[:n_signals] if len(signal_ma_list) > n_signals else signal_ma_list + [None] * (n_signals - len(signal_ma_list))
    if len(signal_ngen_list) != n_signals:
        print(f"  ✗ Error: Must provide --signal-ngen for each signal file ({n_signals} files)")
        return
    if len(signal_labels) != n_signals:
        signal_labels = signal_labels[:n_signals] if len(signal_labels) > n_signals else signal_labels + [None] * (n_signals - len(signal_labels))

    # Load signal cross-sections dictionary
    signal_xs_dict = load_signal_cross_sections(args.signal_xsec)

    # Process each signal
    all_signal_outputs = []  # List of {region_name: output} for each signal
    signal_info_list = []  # List of (label, xs, ngen) for each signal

    for i, (sig_file, sig_xs, sig_mA, sig_ma, sig_ngen, sig_label) in enumerate(zip(
        signal_files, signal_xs_list, signal_mA_list, signal_ma_list, signal_ngen_list, signal_labels
    )):
        # Determine cross-section
        if sig_xs is None:
            if sig_mA is not None and sig_ma is not None:
                signal_key = (sig_mA, sig_ma)
                if signal_key in signal_xs_dict:
                    sig_xs = signal_xs_dict[signal_key]
                    print(f"  ✓ Signal {i+1}: Loaded cross-section from config: mA={sig_mA}, ma={sig_ma}, xs={sig_xs:.10f} pb")
                else:
                    print(f"  ✗ Error: Signal cross-section not found for mA={sig_mA}, ma={sig_ma}")
                    continue
            else:
                print(f"  ✗ Error: Signal {i+1}: Must provide either --signal-xs or both --signal-mA and --signal-ma")
                continue

        # Generate label if not provided
        if sig_label is None:
            if sig_mA is not None and sig_ma is not None:
                sig_label = f"Signal (mA={sig_mA}, ma={sig_ma})"
            else:
                sig_label = f"Signal {i+1}"

        signal_info_list.append((sig_label, sig_xs, sig_ngen))

        # Load and process signal
        signal_data = load_sample_and_build_objects(sig_file, args.cuts_config)
        if signal_data is None:
            print(f"  ✗ Failed to load signal {i+1}: {sig_file}")
            continue

        signal_outputs = {}
        for region_name in regions_config.keys():
            output = process_sample_for_region(
                sig_label, signal_data["events"], signal_data["jets"], signal_data["bjets"],
                signal_data["met"], signal_data["electrons"], signal_data["muons"],
                signal_data["additional_vars"], sig_xs, sig_ngen,
                region_name, regions_config, args.lumi, output_dir, args.cuts_config
            )
            if output:
                signal_outputs[region_name] = output

        all_signal_outputs.append(signal_outputs)

    background_outputs = {}
    for bg_name, bg_file in background_files.items():
        if bg_name not in bg_xs:
            continue

        xs_pb = bg_xs[bg_name]
        ngen = count_events_in_root_file(bg_file)
        if ngen is None:
            continue

        bg_data = load_sample_and_build_objects(bg_file, args.cuts_config)
        if bg_data is None:
            continue

        background_outputs[bg_name] = {}
        for region_name in regions_config.keys():
            output = process_sample_for_region(
                bg_name, bg_data["events"], bg_data["jets"], bg_data["bjets"],
                bg_data["met"], bg_data["electrons"], bg_data["muons"],
                bg_data["additional_vars"], xs_pb, ngen,
                region_name, regions_config, args.lumi, output_dir, args.cuts_config
            )
            if output:
                background_outputs[bg_name][region_name] = output

    single_top_data = {}
    if "sTop_tchannel" in background_outputs:
        single_top_data["sTop_tchannel"] = background_outputs["sTop_tchannel"].copy()
    if "sTop_tW" in background_outputs:
        single_top_data["sTop_tW"] = background_outputs["sTop_tW"].copy()

    if "sTop_tchannel" in background_outputs and "sTop_tW" in background_outputs:
        import hist
        background_outputs["stop"] = {}
        for region_name in regions_config.keys():
            if region_name in background_outputs["sTop_tchannel"] and region_name in background_outputs["sTop_tW"]:
                tchannel_cutflow = background_outputs["sTop_tchannel"][region_name]["cutflow"]
                tW_cutflow = background_outputs["sTop_tW"][region_name]["cutflow"]
                combined_cutflow = OrderedDict()
                for key in tchannel_cutflow.keys():
                    combined_cutflow[key] = tchannel_cutflow[key] + tW_cutflow.get(key, 0)

                tchannel_hists = background_outputs["sTop_tchannel"][region_name]["histograms"]
                tW_hists = background_outputs["sTop_tW"][region_name]["histograms"]
                combined_hists = {}
                for obs_name in tchannel_hists.keys():
                    if obs_name in tW_hists:
                        combined_hists[obs_name] = tchannel_hists[obs_name] + tW_hists[obs_name]

                background_outputs["stop"][region_name] = {
                    "cutflow": combined_cutflow,
                    "histograms": combined_hists,
                    "observables": background_outputs["sTop_tchannel"][region_name]["observables"]
                }

        del background_outputs["sTop_tchannel"]
        del background_outputs["sTop_tW"]

    for region_name in regions_config.keys():
        region_type = get_region_type(region_name)
        region_category = get_region_category(region_name)

        if region_type == "SR":
            region_dir = plots_dir / f"sr{region_category}"
        else:
            cr_part = region_name.split(':')[1]
            cr_name = cr_part.replace('CR_', '').lower()
            region_dir = plots_dir / f"cr{region_category}_{cr_name}"

        region_dir.mkdir(exist_ok=True)

        bg_cutflows = {}
        for bg_name in background_outputs:
            if region_name in background_outputs[bg_name]:
                bg_cutflows[bg_name] = background_outputs[bg_name][region_name]["cutflow"]

        if bg_cutflows:
            plot_cutflow(background_cutflows=bg_cutflows, output_file=str(region_dir / "cutflow.pdf"), lumi=args.lumi)

        # Collect histograms for each observable - overlay all signals
        signal_hists = {}  # Dictionary of {label: {obs_name: hist}}
        for i, signal_outputs in enumerate(all_signal_outputs):
            if region_name in signal_outputs:
                sig_label = signal_info_list[i][0]
                signal_hists[sig_label] = signal_outputs[region_name]["histograms"]

        bg_hists = {}
        for bg_name in background_outputs:
            if region_name in background_outputs[bg_name]:
                bg_hists[bg_name] = background_outputs[bg_name][region_name]["histograms"]

        # Plot each observable - only relevant ones for this region type
        if region_type == "SR":
            if region_category == "1b":
                # 1b SR: MET is main variable (with variable binning)
                observables_to_plot = ["met", "mbb", "ptbb", "dphi_jet_met_min", "ht"]
            else:  # 2b SR
                # 2b SR: cost_star is main variable (with variable binning), also plot MET
                observables_to_plot = ["cost_star", "met", "mbb", "ptbb", "dphi_jet_met_min", "ht"]
        else:  # CR
            if region_category == "1b":
                # 1b CR: Recoil is main variable (with variable binning)
                observables_to_plot = ["recoil", "mt", "mll", "mbb", "ptbb", "dphi_jet_met_min", "ht"]
                # Add lepton pT and min_dPhi for W/Top control regions (matching StackPlotter)
                if 'CR_Wlnu' in region_name or 'CR_Top' in region_name:
                    observables_to_plot.extend(["lep1_pt", "min_dphi"])
            else:  # 2b CR
                # 2b CR: cost_star is main variable (with variable binning)
                observables_to_plot = ["cost_star", "recoil", "mt", "mll", "mbb", "ptbb", "dphi_jet_met_min", "ht"]
                # Add lepton pT and min_dPhi for W/Top control regions (matching StackPlotter)
                if 'CR_Wlnu' in region_name or 'CR_Top' in region_name:
                    observables_to_plot.extend(["lep1_pt", "min_dphi"])

        for obs_name in observables_to_plot:
            bg_hists_obs = {}
            for bg, h in bg_hists.items():
                if obs_name in h:
                    bg_hists_obs[bg] = h[obs_name]

            if not bg_hists_obs:
                continue

            # Collect all signal histograms for this observable
            signal_hists_obs = {}
            if region_type == "SR":
                for sig_label, sig_hists in signal_hists.items():
                    if obs_name in sig_hists:
                        sig_hist_obs = sig_hists[obs_name]
                        signal_values = sig_hist_obs.values()
                        if len(signal_values) > 0 and np.sum(signal_values) > 0:
                            signal_hists_obs[sig_label] = sig_hist_obs

            try:
                first_bg_hist = list(bg_hists_obs.values())[0]
                xlabel_from_hist = first_bg_hist.axes[0].label if hasattr(first_bg_hist.axes[0], 'label') else None
                # Use signal_hists parameter for multiple signals, signal_hist for backward compatibility
                plot_signal_vs_background(
                    signal_hist=None if signal_hists_obs else None,
                    signal_hists=signal_hists_obs if signal_hists_obs else None,
                    background_hists=bg_hists_obs,
                    xlabel=xlabel_from_hist,
                    output_file=str(region_dir / f"{obs_name}.pdf"),
                    lumi=args.lumi
                )

                # Save ROOT file for this observable
                root_file = region_dir / f"{obs_name}.root"
                try:
                    with uproot.recreate(str(root_file)) as f:
                        # Save all signal histograms
                        for sig_label, sig_hist_obs in signal_hists_obs.items():
                            values = sig_hist_obs.values()
                            edges = sig_hist_obs.axes[0].edges
                            # Use sanitized label for ROOT key (no special characters)
                            root_key = sig_label.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("=", "")
                            f[root_key] = (values, edges)

                        # Save background histograms
                        for bg_name, bg_hist in bg_hists_obs.items():
                            values = bg_hist.values()
                            edges = bg_hist.axes[0].edges
                            f[bg_name] = (values, edges)
                except Exception as e:
                    print(f"    ⚠ Warning: Could not save ROOT file for {obs_name}: {e}")

            except Exception as e:
                print(f"    ✗ Error plotting {obs_name}: {e}")
                import traceback
                traceback.print_exc()

        # Generate datacard using first signal (for Combine compatibility)
        if all_signal_outputs and len(all_signal_outputs) > 0 and bg_cutflows:
            first_signal_outputs = all_signal_outputs[0]
            if region_name in first_signal_outputs:
                obs_name = "met" if region_type == "SR" else "recoil"
                first_sig_label, first_sig_xs, first_sig_ngen = signal_info_list[0]
                signal_yield = first_signal_outputs[region_name]["n_selected"] * (first_sig_xs * args.lumi * 1000.0) / first_sig_ngen
            bg_rates = {}
            for bg_name in background_outputs:
                if region_name in background_outputs[bg_name]:
                    if bg_name == "stop":
                        tchannel_yield = 0
                        tW_yield = 0
                        if "sTop_tchannel" in bg_xs and "sTop_tchannel" in background_files and "sTop_tchannel" in single_top_data:
                            tchannel_ngen = count_events_in_root_file(background_files["sTop_tchannel"])
                            if tchannel_ngen and region_name in single_top_data["sTop_tchannel"]:
                                tchannel_yield = single_top_data["sTop_tchannel"][region_name]["n_selected"] * (bg_xs["sTop_tchannel"] * args.lumi * 1000.0) / tchannel_ngen
                        if "sTop_tW" in bg_xs and "sTop_tW" in background_files and "sTop_tW" in single_top_data:
                            tW_ngen = count_events_in_root_file(background_files["sTop_tW"])
                            if tW_ngen and region_name in single_top_data["sTop_tW"]:
                                tW_yield = single_top_data["sTop_tW"][region_name]["n_selected"] * (bg_xs["sTop_tW"] * args.lumi * 1000.0) / tW_ngen
                        bg_rates["stop"] = tchannel_yield + tW_yield
                    else:
                        bg_yield = background_outputs[bg_name][region_name]["n_selected"] * (bg_xs[bg_name] * args.lumi * 1000.0) / count_events_in_root_file(background_files[bg_name])
                        bg_rates[bg_name] = bg_yield

            dominant_bkg = None
            if region_type == "CR" and bg_rates:
                dominant_bkg = max(bg_rates.keys(), key=lambda x: bg_rates[x])

            bin_name = region_type
            datacard = generate_datacard(
                signal_name="sig",
                signal_rate=signal_yield,
                backgrounds=bg_rates,
                region_type=region_type,
                bin_name=bin_name,
                dominant_bkg=dominant_bkg,
                observation=-1,
                shapes_file="shapes.root"
            )
            save_datacard(datacard, str(region_dir / "datacard.txt"))

            if region_type == "SR":
                main_observable = "cost_star" if region_category == "2b" else "met"
            else:
                main_observable = "cost_star" if region_category == "2b" else "recoil"
            shapes_file = region_dir / "shapes.root"
            # Use first signal for shapes file (Combine compatibility)
            first_signal_hist = signal_hists[signal_info_list[0][0]] if signal_hists else None
            success = create_shapes_file(
                signal_hist=first_signal_hist,
                background_hists=bg_hists,
                bin_name=bin_name,
                output_file=str(shapes_file),
                main_observable=main_observable
            )

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
