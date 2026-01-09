"""
Complete end-to-end bbMET analysis pipeline for 2HDM+a model.
Starts from Delphes ROOT files and produces Combine-ready datacards.

Philosophy: Phenomenology first (Delphes), CMS-like analysis, Coffea-compatible.
"""

import numpy as np
import awkward as ak
import hist
from typing import Dict, Any, Optional
import json
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("Warning: PyYAML not installed. Install with: pip install pyyaml")
    print("  YAML configuration files will not work. Using default cuts.")


# ============================================================================
# 0. Configuration Loading
# ============================================================================

def load_cuts_from_yaml(yaml_file="cuts_config.yaml"):
    """
    Load selection cuts from YAML configuration file.

    Parameters:
    -----------
    yaml_file : str
        Path to YAML configuration file

    Returns:
    --------
    cuts : dict
        Dictionary of cut parameters
    """
    yaml_path = Path(yaml_file)

    if not HAS_YAML:
        print(f"Warning: PyYAML not available. Cannot load {yaml_file}. Using defaults.")
        return get_default_cuts()

    if not yaml_path.exists():
        # Return default cuts if file doesn't exist
        print(f"Warning: Cuts config file {yaml_file} not found. Using defaults.")
        return get_default_cuts()

    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        # Extract cuts from YAML structure
        cuts = {
            "jet_pt_min": config.get("jets", {}).get("pt_min", 30.0),
            "jet_eta_max": config.get("jets", {}).get("eta_max", 2.4),
            "btag_threshold": config.get("bjets", {}).get("btag_threshold", 0),
        }

        return cuts
    except Exception as e:
        print(f"Error loading cuts from {yaml_file}: {e}")
        print("Using default cuts.")
        return get_default_cuts()


def get_default_cuts():
    """
    Get default cut values.

    Returns:
    --------
    cuts : dict
        Dictionary of default cut parameters
    """
    return {
        "jet_pt_min": 30.0,
        "jet_eta_max": 2.4,
        "btag_threshold": 0,
    }


# ============================================================================
# 0. Delphes File Loading Helper
# ============================================================================

def load_delphes_events(root_file, tree_name="Delphes", library="ak"):
    """
    Load Delphes events from ROOT file using uproot.

    This function properly loads Delphes ROOT files and returns events in a format
    compatible with Coffea processors. It loads only Jet and MissingET branches
    to avoid issues with problematic branches.

    Parameters:
    -----------
    root_file : str
        Path to Delphes ROOT file
    tree_name : str
        Name of TTree in ROOT file (default: "Delphes")
    library : str
        Library to use for arrays ("ak" for awkward, "np" for numpy)

    Returns:
    --------
    events : awkward.Array
        Events as awkward array with Jet and MissingET branches
    """
    try:
        import uproot
    except ImportError:
        raise ImportError("uproot is required. Install with: pip install uproot")

    try:
        # Open file with uproot
        file = uproot.open(root_file)

        if tree_name not in file:
            available = list(file.keys())
            file.close()
            raise KeyError(f"Tree '{tree_name}' not found in {root_file}. "
                          f"Available trees: {available}")

        tree = file[tree_name]

        # Get available branches
        available_branches = [str(b) for b in tree.keys()]

        # For bbMET analysis, we only need Jet and MissingET branches
        # Check if they exist
        has_jet = "Jet" in available_branches
        has_met = "MissingET" in available_branches

        if not has_jet or not has_met:
            file.close()
            raise RuntimeError(
                f"Required branches not found in {root_file}.\n"
                f"  Jet found: {has_jet}\n"
                f"  MissingET found: {has_met}\n"
                f"Available branches: {available_branches[:20]}..."
            )

        # Load specific Jet, MissingET, and lepton sub-branches
        # This avoids the Particle.fBits error and gives us direct access to the data
        # Structure: Jet/Jet.PT, Jet/Jet.Eta, etc. are arrays of arrays (one per event)
        branches_to_load = [
            "Jet/Jet.PT",
            "Jet/Jet.Eta",
            "Jet/Jet.Phi",
            "Jet/Jet.BTag",
            "MissingET/MissingET.MET",
            "MissingET/MissingET.Phi"
        ]

        # Add lepton branches if available
        available_branches_str = [str(b) for b in available_branches]
        if any("Electron" in b for b in available_branches_str):
            branches_to_load.extend([
                "Electron/Electron.PT",
                "Electron/Electron.Eta",
                "Electron/Electron.Phi",
                "Electron/Electron.Charge"
            ])
        if any("Muon" in b for b in available_branches_str):
            branches_to_load.extend([
                "Muon/Muon.PT",
                "Muon/Muon.Eta",
                "Muon/Muon.Phi",
                "Muon/Muon.Charge"
            ])

        try:
            # Load with uproot - returns awkward array by default with library="ak"
            events_dict = tree.arrays(branches_to_load, library=library)
            file.close()

            # uproot returns a dict-like structure when loading multiple branches
            # Convert to a structured awkward array for easier access
            # This creates an array where each event has fields like 'Jet/Jet.PT', etc.
            if isinstance(events_dict, dict):
                # Zip into structured array for easier access
                events = ak.zip(events_dict)
            else:
                # Already an awkward array
                events = events_dict

            return events

        except Exception as e:
            file.close()
            # If loading fails, provide helpful error message
            raise RuntimeError(
                f"Failed to load Delphes file {root_file}.\n"
                f"Error: {e}\n\n"
                f"Try running: python analyze_root_structure.py {root_file}\n"
                f"to diagnose the file structure."
            )

    except Exception as e:
        if isinstance(e, (KeyError, RuntimeError)):
            raise
        raise RuntimeError(f"Error loading Delphes file {root_file}: {e}")


# ============================================================================
# 1. Helper Functions for Field Access
# ============================================================================

def get_field(events, field_name):
    """
    Extract a field from events array.

    Parameters:
    -----------
    events : awkward.Array
        Events array with fields like 'Jet/Jet.PT', 'MissingET/MissingET.MET'
    field_name : str
        Full field name (e.g., 'Jet/Jet.PT')

    Returns:
    --------
    field_array : awkward.Array
        Array of arrays (one per event) or array of values
    """
    if hasattr(events, field_name):
        return getattr(events, field_name)
    elif isinstance(events, ak.Array) and hasattr(events, 'fields') and field_name in events.fields:
        return events[field_name]
    else:
        available = events.fields if hasattr(events, 'fields') else 'N/A'
        raise AttributeError(f"Field {field_name} not found. Available fields: {available}")


# ============================================================================
# 0.5. Utility Functions
# ============================================================================

def Phi_mpi_pi(x):
    """
    Wrap angle to [-π, π] range.
    Vectorized implementation of the standard CMS Phi_mpi_pi function.

    Parameters:
    -----------
    x : array-like
        Angle values (can be numpy array or awkward array)

    Returns:
    --------
    x_wrapped : same type as input
        Angles wrapped to [-π, π]
    """
    kPI = np.pi
    kTWOPI = 2 * kPI

    # Vectorized version using modulo operation (more efficient than while loops)
    if isinstance(x, ak.Array):
        # For awkward arrays, use numpy operations that work element-wise
        # Convert to list, process, convert back
        x_list = ak.to_list(x)
        if isinstance(x_list, list) and len(x_list) > 0 and isinstance(x_list[0], list):
            # Jagged array (array of arrays)
            x_wrapped_list = [[val - kTWOPI * np.floor((val + kPI) / kTWOPI) for val in event] for event in x_list]
        else:
            # Regular array
            x_wrapped_list = [val - kTWOPI * np.floor((val + kPI) / kTWOPI) for val in x_list]
        return ak.Array(x_wrapped_list)
    else:
        x = np.asarray(x)
        return x - kTWOPI * np.floor((x + kPI) / kTWOPI)


def DeltaPhi(phi1, phi2):
    """
    Calculate Δφ between two phi angles, wrapped to [0, π].
    Implementation following CMS standard DeltaPhi function.

    Parameters:
    -----------
    phi1 : array-like
        First phi angle(s)
    phi2 : array-like
        Second phi angle(s)

    Returns:
    --------
    dphi : same type as input
        Absolute delta phi in [0, π] range
    """
    # Calculate difference and wrap to [-π, π]
    phi_diff = phi1 - phi2
    phi_wrapped = Phi_mpi_pi(phi_diff)

    # Return absolute value (in [0, π])
    if isinstance(phi_wrapped, ak.Array):
        # Use numpy abs on the underlying values
        phi_wrapped_list = ak.to_list(phi_wrapped)
        if isinstance(phi_wrapped_list, list) and len(phi_wrapped_list) > 0 and isinstance(phi_wrapped_list[0], list):
            # Jagged array
            dphi_list = [[abs(val) for val in event] for event in phi_wrapped_list]
        else:
            # Regular array
            dphi_list = [abs(val) for val in phi_wrapped_list]
        return ak.Array(dphi_list)
    else:
        return np.abs(phi_wrapped)


# ============================================================================
# 1. Physics Object Definitions
# ============================================================================

def build_jets(events, pt_min=30.0, eta_max=2.4):
    """
    Build jets from Delphes events.

    Parameters:
    -----------
    events : awkward.Array
        Delphes events (from load_delphes_events)
    pt_min : float
        Minimum jet pT in GeV (default: 30)
    eta_max : float
        Maximum |eta| (default: 2.4)

    Returns:
    --------
    jets : awkward.Array
        Selected jets with pT > pt_min and |eta| < eta_max
    """
    # Access Jet fields directly
    # Structure: events['Jet/Jet.PT'] is an array of arrays (one array per event)
    try:
        jet_pt = get_field(events, 'Jet/Jet.PT')
        jet_eta = get_field(events, 'Jet/Jet.Eta')
        jet_phi = get_field(events, 'Jet/Jet.Phi')
        jet_btag = get_field(events, 'Jet/Jet.BTag')
    except Exception as e:
        raise RuntimeError(f"Error accessing Jet fields: {e}")

    # Apply selection mask per jet
    # jet_pt and jet_eta are arrays of arrays (one array per event)
    # Create mask: for each event, which jets pass the cuts
    mask = (jet_pt > pt_min) & (np.abs(jet_eta) < eta_max)

    # Apply mask to select jets
    # For each event, select only jets that pass the cuts
    selected_jets = {
        'PT': jet_pt[mask],
        'Eta': jet_eta[mask],
        'Phi': jet_phi[mask],
        'BTag': jet_btag[mask]
    }

    return selected_jets


def select_bjets(jets, btag_threshold=0):
    """
    Select b-jets using Delphes BTag.

    Parameters:
    -----------
    jets : dict
        Jet collection (from build_jets) with 'PT', 'Eta', 'Phi', 'BTag' keys
    btag_threshold : float
        B-tag threshold (default: 0, meaning BTag > 0)

    Returns:
    --------
    bjets : dict
        Selected b-jets with same structure as jets
    """
    # Access BTag field
    btag = jets['BTag']

    # Apply mask per jet
    # btag is an array of arrays (one array per event)
    # Use >= instead of > to include threshold value (BTag values are typically 0 or 1)
    mask = btag >= btag_threshold

    # Apply mask to select b-jets
    selected_bjets = {
        'PT': jets['PT'][mask],
        'Eta': jets['Eta'][mask],
        'Phi': jets['Phi'][mask],
        'BTag': jets['BTag'][mask]
    }

    return selected_bjets


def build_met(events):
    """
    Build MET from Delphes events.

    Parameters:
    -----------
    events : awkward.Array
        Delphes events (from load_delphes_events)

    Returns:
    --------
    met : awkward.Array
        MET with pt and phi attributes
    """
    # Access MissingET fields directly
    # Structure: events['MissingET/MissingET.MET'] is an array of arrays (one array per event)
    # Each event has one MET value, so each array has length 1
    try:
        met_pt = get_field(events, 'MissingET/MissingET.MET')
        met_phi = get_field(events, 'MissingET/MissingET.Phi')
    except Exception as e:
        raise RuntimeError(f"Error accessing MissingET fields: {e}")

    # MissingET has one entry per event
    # met_pt and met_phi are arrays of arrays where each inner array has one element
    # Extract the single value from each event's array for easier access
    # met_pt[0] is an array with one element, we want just the value
    # But for consistency with jets (which are arrays of arrays), keep as arrays of arrays
    met = {
        'MET': met_pt,
        'Phi': met_phi
    }

    return met


def build_electrons(events, pt_min=10.0, eta_max=2.5):
    """
    Build electrons from Delphes events.

    Parameters:
    -----------
    events : awkward.Array
        Delphes events (from load_delphes_events)
    pt_min : float
        Minimum electron pT in GeV (default: 10.0)
    eta_max : float
        Maximum |eta| for electrons (default: 2.5)

    Returns:
    --------
    electrons : dict or None
        Dictionary with 'PT', 'Eta', 'Phi', 'Charge' keys, or None if not available
    """
    try:
        electron_pt = get_field(events, 'Electron/Electron.PT')
        electron_eta = get_field(events, 'Electron/Electron.Eta')
        electron_phi = get_field(events, 'Electron/Electron.Phi')
        electron_charge = get_field(events, 'Electron/Electron.Charge')
    except (AttributeError, KeyError):
        return None

    # Apply selection
    mask = (electron_pt > pt_min) & (np.abs(electron_eta) < eta_max)

    selected_electrons = {
        'PT': electron_pt[mask],
        'Eta': electron_eta[mask],
        'Phi': electron_phi[mask],
        'Charge': electron_charge[mask]
    }

    return selected_electrons


def build_muons(events, pt_min=10.0, eta_max=2.4):
    """
    Build muons from Delphes events.

    Parameters:
    -----------
    events : awkward.Array
        Delphes events (from load_delphes_events)
    pt_min : float
        Minimum muon pT in GeV (default: 10.0)
    eta_max : float
        Maximum |eta| for muons (default: 2.4)

    Returns:
    --------
    muons : dict or None
        Dictionary with 'PT', 'Eta', 'Phi', 'Charge' keys, or None if not available
    """
    try:
        muon_pt = get_field(events, 'Muon/Muon.PT')
        muon_eta = get_field(events, 'Muon/Muon.Eta')
        muon_phi = get_field(events, 'Muon/Muon.Phi')
        muon_charge = get_field(events, 'Muon/Muon.Charge')
    except (AttributeError, KeyError):
        return None

    # Apply selection
    mask = (muon_pt > pt_min) & (np.abs(muon_eta) < eta_max)

    selected_muons = {
        'PT': muon_pt[mask],
        'Eta': muon_eta[mask],
        'Phi': muon_phi[mask],
        'Charge': muon_charge[mask]
    }

    return selected_muons


def calculate_mt(lepton_pt, lepton_phi, met_pt, met_phi):
    """
    Calculate transverse mass MT = sqrt(2 * pT_l * MET * (1 - cos(Δφ))).

    Parameters:
    -----------
    lepton_pt : array
        Lepton pT (array of arrays, one per event)
    lepton_phi : array
        Lepton phi (array of arrays, one per event)
    met_pt : array
        MET (array of arrays, one per event)
    met_phi : array
        MET phi (array of arrays, one per event)

    Returns:
    --------
    mt : array
        Transverse mass (one value per event)
    """
    # Get leading lepton
    lepton_pt_flat = ak.firsts(lepton_pt, axis=1)
    lepton_phi_flat = ak.firsts(lepton_phi, axis=1)
    met_pt_flat = ak.flatten(met_pt, axis=1)
    met_phi_flat = ak.flatten(met_phi, axis=1)

    # Fill None values
    lepton_pt_flat = ak.fill_none(lepton_pt_flat, 0.0)
    lepton_phi_flat = ak.fill_none(lepton_phi_flat, 0.0)

    # Calculate Δφ using DeltaPhi function
    dphi = DeltaPhi(lepton_phi_flat, met_phi_flat)

    # Calculate MT
    mt = np.sqrt(2.0 * lepton_pt_flat * met_pt_flat * (1.0 - np.cos(dphi)))

    return mt


def calculate_recoil(lepton_pt, lepton_phi, met_pt, met_phi):
    """
    Calculate recoil = |MET + lepton|.

    Parameters:
    -----------
    lepton_pt : array
        Lepton pT (array of arrays, one per event)
    lepton_phi : array
        Lepton phi (array of arrays, one per event)
    met_pt : array
        MET (array of arrays, one per event)
    met_phi : array
        MET phi (array of arrays, one per event)

    Returns:
    --------
    recoil : array
        Recoil (one value per event)
    """
    # Get leading lepton
    lepton_pt_flat = ak.firsts(lepton_pt, axis=1)
    lepton_phi_flat = ak.firsts(lepton_phi, axis=1)
    met_pt_flat = ak.flatten(met_pt, axis=1)
    met_phi_flat = ak.flatten(met_phi, axis=1)

    # Fill None values
    lepton_pt_flat = ak.fill_none(lepton_pt_flat, 0.0)
    lepton_phi_flat = ak.fill_none(lepton_phi_flat, 0.0)

    # Calculate vector sum
    lepton_px = lepton_pt_flat * np.cos(lepton_phi_flat)
    lepton_py = lepton_pt_flat * np.sin(lepton_phi_flat)
    met_px = met_pt_flat * np.cos(met_phi_flat)
    met_py = met_pt_flat * np.sin(met_phi_flat)

    recoil = np.sqrt((lepton_px + met_px)**2 + (lepton_py + met_py)**2)

    return recoil


def calculate_mll(lepton1_pt, lepton1_eta, lepton1_phi, lepton1_charge,
                   lepton2_pt, lepton2_eta, lepton2_phi, lepton2_charge):
    """
    Calculate dilepton invariant mass Mll.

    Parameters:
    -----------
    lepton1_pt, lepton1_eta, lepton1_phi, lepton1_charge : arrays
        First lepton 4-vector components (can be 1D or 2D arrays)
    lepton2_pt, lepton2_eta, lepton2_phi, lepton2_charge : arrays
        Second lepton 4-vector components (can be 1D or 2D arrays)

    Returns:
    --------
    mll : array
        Dilepton invariant mass (one value per event)
    """
    # Check if arrays are already 1D (from process_regions.py) or 2D
    # Try to get the dimensionality
    try:
        # Check if we can call ak.num with axis=1 (would fail for 1D)
        _ = ak.num(lepton1_pt, axis=1)
        # If we get here, it's 2D, so extract first element
        l1_pt = ak.firsts(lepton1_pt, axis=1)
        l1_eta = ak.firsts(lepton1_eta, axis=1)
        l1_phi = ak.firsts(lepton1_phi, axis=1)
        l2_pt = ak.pad_none(lepton2_pt, 2, axis=1)[:, 1]
        l2_eta = ak.pad_none(lepton2_eta, 2, axis=1)[:, 1]
        l2_phi = ak.pad_none(lepton2_phi, 2, axis=1)[:, 1]
    except (ValueError, TypeError, AttributeError):
        # Already 1D arrays, use directly
        l1_pt = lepton1_pt
        l1_eta = lepton1_eta
        l1_phi = lepton1_phi
        l2_pt = lepton2_pt
        l2_eta = lepton2_eta
        l2_phi = lepton2_phi

    # Fill None values
    l1_pt = ak.fill_none(l1_pt, 0.0)
    l1_eta = ak.fill_none(l1_eta, 0.0)
    l1_phi = ak.fill_none(l1_phi, 0.0)
    l2_pt = ak.fill_none(l2_pt, 0.0)
    l2_eta = ak.fill_none(l2_eta, 0.0)
    l2_phi = ak.fill_none(l2_phi, 0.0)

    # Convert to numpy
    l1_pt = np.asarray(ak.to_numpy(l1_pt))
    l1_eta = np.asarray(ak.to_numpy(l1_eta))
    l1_phi = np.asarray(ak.to_numpy(l1_phi))
    l2_pt = np.asarray(ak.to_numpy(l2_pt))
    l2_eta = np.asarray(ak.to_numpy(l2_eta))
    l2_phi = np.asarray(ak.to_numpy(l2_phi))

    # Calculate 4-vectors
    l1_px = l1_pt * np.cos(l1_phi)
    l1_py = l1_pt * np.sin(l1_phi)
    l1_pz = l1_pt * np.sinh(l1_eta)
    l1_e = l1_pt * np.cosh(l1_eta)

    l2_px = l2_pt * np.cos(l2_phi)
    l2_py = l2_pt * np.sin(l2_phi)
    l2_pz = l2_pt * np.sinh(l2_eta)
    l2_e = l2_pt * np.cosh(l2_eta)

    # Calculate invariant mass
    mll = np.sqrt(np.maximum(0, (l1_e + l2_e)**2 - (l1_px + l2_px)**2 - (l1_py + l2_py)**2 - (l1_pz + l2_pz)**2))

    return mll


# ============================================================================
# 3. Observables Calculation
# ============================================================================

def calculate_observables(jets, bjets, met, mask):
    """
    Calculate all observables for selected events.

    Parameters:
    -----------
    jets : dict
        Jet collection with 'PT', 'Eta', 'Phi', 'BTag' keys (arrays of arrays)
    bjets : dict
        B-jet collection with same structure as jets
    met : dict
        MET collection with 'MET' and 'Phi' keys (arrays of arrays)
    mask : boolean array
        Event selection mask

    Returns:
    --------
    observables : dict
        Dictionary of calculated observables
    """
    # Select events passing cuts
    # Apply mask to each field in the dicts
    selected_bjets = {k: v[mask] for k, v in bjets.items()}
    selected_met = {k: v[mask] for k, v in met.items()}
    selected_jets = {k: v[mask] for k, v in jets.items()}

    # Count b-jets per event
    n_bjets = ak.num(selected_bjets['PT'], axis=1)

    # Require at least 2 b-jets for dijet calculation
    has_dijet = n_bjets >= 2

    # Extract fields for leading two b-jets
    # selected_bjets['PT'] is an array of arrays, get first and second elements
    bjets_with_dijet = {k: v[has_dijet] for k, v in selected_bjets.items()}

    # Get leading two b-jets
    bjet1_pt = ak.firsts(bjets_with_dijet['PT'], axis=1)
    bjet1_eta = ak.firsts(bjets_with_dijet['Eta'], axis=1)
    bjet1_phi = ak.firsts(bjets_with_dijet['Phi'], axis=1)
    bjet2_pt = ak.pad_none(bjets_with_dijet['PT'], 2, axis=1)[:, 1]
    bjet2_eta = ak.pad_none(bjets_with_dijet['Eta'], 2, axis=1)[:, 1]
    bjet2_phi = ak.pad_none(bjets_with_dijet['Phi'], 2, axis=1)[:, 1]

    # Fill None values with 0.0
    bjet1_pt = ak.fill_none(bjet1_pt, 0.0)
    bjet1_eta = ak.fill_none(bjet1_eta, 0.0)
    bjet1_phi = ak.fill_none(bjet1_phi, 0.0)
    bjet2_pt = ak.fill_none(bjet2_pt, 0.0)
    bjet2_eta = ak.fill_none(bjet2_eta, 0.0)
    bjet2_phi = ak.fill_none(bjet2_phi, 0.0)

    # Convert to numpy arrays for calculations
    bjet1_pt = np.asarray(ak.to_numpy(bjet1_pt))
    bjet1_eta = np.asarray(ak.to_numpy(bjet1_eta))
    bjet1_phi = np.asarray(ak.to_numpy(bjet1_phi))
    bjet2_pt = np.asarray(ak.to_numpy(bjet2_pt))
    bjet2_eta = np.asarray(ak.to_numpy(bjet2_eta))
    bjet2_phi = np.asarray(ak.to_numpy(bjet2_phi))

    # Calculate dijet 4-vector
    dijet_px = bjet1_pt * np.cos(bjet1_phi) + bjet2_pt * np.cos(bjet2_phi)
    dijet_py = bjet1_pt * np.sin(bjet1_phi) + bjet2_pt * np.sin(bjet2_phi)
    dijet_pz = bjet1_pt * np.sinh(bjet1_eta) + bjet2_pt * np.sinh(bjet2_eta)

    # Approximate mass calculation
    e1 = bjet1_pt * np.cosh(bjet1_eta)
    e2 = bjet2_pt * np.cosh(bjet2_eta)
    dijet_e = e1 + e2

    dijet_pt = np.sqrt(dijet_px**2 + dijet_py**2)
    dijet_mass = np.sqrt(np.maximum(0, dijet_e**2 - dijet_px**2 - dijet_py**2 - dijet_pz**2))

    # Δφ(jet, MET) - minimum delta phi among all jets and MET
    # This is the useful variable - delta phi between MET and all jets
    # Get MET phi for all selected events
    met_phi = ak.flatten(selected_met['Phi'], axis=1)
    met_phi_np = np.asarray(ak.to_numpy(met_phi))

    # Calculate dphi for all jets in all events using vectorized operations
    jets_phi = selected_jets['Phi']
    n_jets = ak.num(jets_phi, axis=1)

    # Create expanded met_phi array matching jets_phi structure
    met_phi_list = []
    for i in range(len(met_phi_np)):
        n_jet = int(ak.to_numpy(n_jets)[i])
        met_phi_list.append([met_phi_np[i]] * n_jet)
    met_phi_repeated = ak.Array(met_phi_list)

    # Calculate Δφ for all jets using DeltaPhi function
    # Handle case where some events have no jets
    dphi_all_jets = DeltaPhi(jets_phi, met_phi_repeated)

    # Get minimum dphi per event (NaN if no jets)
    # Use ak.min with axis=1, but handle empty arrays
    try:
        # Check if we have any jets at all
        if len(dphi_all_jets) == 0:
            # No events with jets
            dphi_jet_met = np.full(len(met_phi_np), np.nan)
        else:
            min_dphi_per_event = ak.min(dphi_all_jets, axis=1)
            dphi_jet_met = np.asarray(ak.to_numpy(ak.fill_none(min_dphi_per_event, np.nan)))
    except (ValueError, AttributeError) as e:
        # Fallback: calculate manually for each event
        dphi_jet_met = np.full(len(met_phi_np), np.nan)
        jets_phi_list = ak.to_list(jets_phi)
        for i in range(len(met_phi_np)):
            if i < len(jets_phi_list) and len(jets_phi_list[i]) > 0:
                event_jets_phi = ak.Array(jets_phi_list[i])
                event_met_phi = ak.Array([met_phi_np[i]] * len(jets_phi_list[i]))
                event_dphi = DeltaPhi(event_jets_phi, event_met_phi)
                if isinstance(event_dphi, ak.Array):
                    event_dphi_list = ak.to_list(event_dphi)
                    if isinstance(event_dphi_list, list) and len(event_dphi_list) > 0:
                        dphi_jet_met[i] = np.min(event_dphi_list)
                elif isinstance(event_dphi, (list, np.ndarray)):
                    dphi_jet_met[i] = np.min(event_dphi)

    # Costheta star: cost_star = abs(tanh(dEtaJet12 / 2))
    # where dEtaJet12 is delta eta between two b-jets
    # Only calculated for 2b category (requires at least 2 b-jets)
    # Get two leading b-jets by pT
    bjets_pt = selected_bjets['PT']
    bjets_eta = selected_bjets['Eta']

    # For events with at least 2 b-jets (required for 2b category)
    n_bjets = ak.num(bjets_pt, axis=1)
    has_at_least_2_bjets = n_bjets >= 2

    # Get leading two b-jets by sorting by pT (descending)
    # Sort b-jets by pT and take first two
    bjets_sorted_by_pt = bjets_eta[ak.argsort(bjets_pt, axis=1, ascending=False, stable=True)]

    # Get leading and subleading b-jet etas
    leading_bjet_eta = ak.firsts(bjets_sorted_by_pt, axis=1)
    subleading_bjet_eta = ak.pad_none(bjets_sorted_by_pt, 2, axis=1)[:, 1]

    # Convert to numpy for calculation
    leading_bjet_eta_np = np.asarray(ak.to_numpy(ak.fill_none(leading_bjet_eta, 0.0)))
    subleading_bjet_eta_np = np.asarray(ak.to_numpy(ak.fill_none(subleading_bjet_eta, 0.0)))

    # Calculate dEtaJet12 = |eta1 - eta2| for all events
    deta_jet12 = np.abs(leading_bjet_eta_np - subleading_bjet_eta_np)

    # Calculate cost_star = abs(tanh(dEtaJet12 / 2))
    # Formula: abs(np.tanh(df.dEtaJet12.div(2)))
    # Only meaningful for events with at least 2 b-jets (2b category)
    cost_star = np.abs(np.tanh(deta_jet12 / 2.0))

    # HT (scalar sum of jet pT)
    ht = ak.sum(selected_jets['PT'], axis=1)
    ht = np.asarray(ak.to_numpy(ht))

    # MET
    met_pt = ak.flatten(selected_met['MET'], axis=1)
    met_pt = np.asarray(ak.to_numpy(met_pt))

    # Create arrays with same length as selected events
    n_selected = len(met_pt)
    mbb = np.full(n_selected, np.nan)
    ptbb = np.full(n_selected, np.nan)
    dphi_jet_met_min_full = np.full(n_selected, np.nan)  # Stores min(Δφ(jet, MET)) - minimum among all jets
    cost_star_full = np.full(n_selected, np.nan)

    # dphi_jet_met_min: minimum Δφ(jet, MET) among all jets in the event
    # Calculated for all events with at least one jet
    # Check if events have at least one jet
    n_jets = ak.num(selected_jets['PT'], axis=1)
    has_jet = n_jets >= 1
    idx_has_jet = np.where(has_jet)[0]
    dphi_jet_met_min_full[idx_has_jet] = dphi_jet_met[idx_has_jet]

    # Fill values for events with dijet
    # has_dijet is relative to selected events
    idx_has_dijet = np.where(has_dijet)[0]
    mbb[idx_has_dijet] = dijet_mass
    ptbb[idx_has_dijet] = dijet_pt

    # Fill cost_star for events with at least 2 b-jets (for 2b category only)
    # cost_star uses the two leading b-jets (by pT)
    idx_has_2_bjets = np.where(has_at_least_2_bjets)[0]
    if len(idx_has_2_bjets) > 0:
        cost_star_full[idx_has_2_bjets] = cost_star[idx_has_2_bjets]

    return {
        "met": met_pt,
        "mbb": mbb,
        "ptbb": ptbb,
        "dphi_jet_met_min": dphi_jet_met_min_full,  # min(Δφ(jet, MET)) - minimum among all jets
        "ht": ht,
        "cost_star": cost_star_full,
    }


# ============================================================================
# 4. Normalization
# ============================================================================

def calculate_weight(xs_pb, lumi_fb, ngen):
    """
    Calculate event weight for normalization.

    Parameters:
    -----------
    xs_pb : float
        Cross-section in pb
    lumi_fb : float
        Integrated luminosity in fb^-1
    ngen : int
        Number of generated events

    Returns:
    --------
    weight : float
        Event weight
    """
    return (xs_pb * lumi_fb * 1000.0) / ngen


# ============================================================================
# 5. Scan Functionality
# ============================================================================

def scan_ma_fixed_mA(processors_dict, ma_values, mA_fixed=1500.0):
    """
    Scan over ma values with fixed mA.

    Parameters:
    -----------
    processors_dict : dict
        Dictionary of {ma_value: processor_output}
    ma_values : array-like
        Array of ma values to scan
    mA_fixed : float
        Fixed mA value in GeV (default: 1500)

    Returns:
    --------
    results : dict
        Dictionary with scan results
    """
    results = {
        "ma": ma_values,
        "mA": mA_fixed,
        "sigma_br": [],
        "met_efficiency": [],
        "mbb_peak": [],
        "significance": [],
    }

    for ma in ma_values:
        if ma not in processors_dict:
            continue

        output = processors_dict[ma]

        # Extract signal yield
        signal_yield = output["cutflow"]["selected"]

        # Calculate σ × BR (normalized by weight)
        # This would come from theory calculation
        # Placeholder: use yield as proxy
        results["sigma_br"].append(signal_yield)

        # MET efficiency
        n_selected = output["cutflow"]["selected"]
        n_total = output["cutflow"]["all"]
        results["met_efficiency"].append(n_selected / n_total if n_total > 0 else 0)

        # m(bb) peak (most probable value)
        mbb_hist = output["mbb"]
        if mbb_hist.sum() > 0:
            # Find bin with maximum content
            max_bin = np.argmax(mbb_hist.values())
            bin_edges = mbb_hist.axes[0].edges
            peak = (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2
            results["mbb_peak"].append(peak)
        else:
            results["mbb_peak"].append(0)

        # Significance (S / sqrt(B))
        # Would need background for this
        results["significance"].append(0)  # Placeholder

    return results


def scan_mA_fixed_ma(processors_dict, mA_values, ma_fixed=300.0):
    """
    Scan over mA values with fixed ma.

    Parameters:
    -----------
    processors_dict : dict
        Dictionary of {mA_value: processor_output}
    mA_values : array-like
        Array of mA values to scan
    ma_fixed : float
        Fixed ma value in GeV (default: 300)

    Returns:
    --------
    results : dict
        Dictionary with scan results
    """
    results = {
        "mA": mA_values,
        "ma": ma_fixed,
        "sigma_br": [],
        "met_efficiency": [],
        "mbb_peak": [],
        "significance": [],
    }

    for mA in mA_values:
        if mA not in processors_dict:
            continue

        output = processors_dict[mA]

        # Extract signal yield
        signal_yield = output["cutflow"]["selected"]

        # Calculate σ × BR
        results["sigma_br"].append(signal_yield)

        # MET efficiency
        n_selected = output["cutflow"]["selected"]
        n_total = output["cutflow"]["all"]
        results["met_efficiency"].append(n_selected / n_total if n_total > 0 else 0)

        # m(bb) peak
        mbb_hist = output["mbb"]
        if mbb_hist.sum() > 0:
            max_bin = np.argmax(mbb_hist.values())
            bin_edges = mbb_hist.axes[0].edges
            peak = (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2
            results["mbb_peak"].append(peak)
        else:
            results["mbb_peak"].append(0)

        # Significance
        results["significance"].append(0)  # Placeholder

    return results


# ============================================================================
# 7. Datacard Generation
# ============================================================================

def generate_datacard(signal_name, signal_rate,
                     backgrounds,  # dict of {name: rate}
                     region_type="SR",  # "SR" or "CR"
                     bin_name=None,  # Will be set to "SR" or "CR" if None
                     dominant_bkg=None,  # Name of dominant background (for CR normalization)
                     observation=-1,
                     shapes_file="shapes.root"):
    """
    Generate Combine datacard following CMS format.

    Parameters:
    -----------
    signal_name : str
        Signal process name
    signal_rate : float
        Signal event rate
    backgrounds : dict
        Dictionary of {background_name: rate}
    region_type : str
        "SR" or "CR" (default: "SR")
    bin_name : str
        Bin name (default: "SR" or "CR" based on region_type)
    dominant_bkg : str
        Name of dominant background (for CR normalization, default: highest rate)
    observation : int
        Observed events (-1 for Asimov)
    shapes_file : str
        Name of shapes ROOT file (default: "shapes.root")

    Returns:
    --------
    datacard : str
        Datacard content as string
    """
    # Set bin name if not provided
    if bin_name is None:
        bin_name = region_type

    # Build process list: signal first (process 0), then backgrounds (1, 2, 3, ...)
    bg_names = list(backgrounds.keys())
    processes = [signal_name] + bg_names
    rates = [signal_rate] + [backgrounds[b] for b in bg_names]
    n_processes = len(processes)
    n_backgrounds = len(bg_names)

    # Determine dominant background if not provided (for CR)
    if dominant_bkg is None and region_type == "CR" and bg_names:
        # Find background with highest rate
        dominant_bkg = max(bg_names, key=lambda x: backgrounds[x])

    # Build datacard
    lines = []
    lines.append(f"imax 1")
    lines.append(f"jmax {n_backgrounds}")
    lines.append(f"kmax *")
    lines.append("---------------------------------")

    # Shapes line
    lines.append(f"shapes * {bin_name} {shapes_file} $PROCESS_{bin_name} $PROCESS_{bin_name}_$SYSTEMATIC")
    lines.append("---------------------------------")

    # Bin and observation
    lines.append(f"bin           {bin_name}")
    lines.append(f"observation   {observation}")
    lines.append("---------------------------------")

    # Process definitions
    lines.append("bin           " + " ".join([bin_name] * n_processes))
    lines.append("process       " + " ".join(processes))
    lines.append("process       " + " ".join([str(i) for i in range(n_processes)]))
    lines.append("rate          " + " ".join([f"{r:.4f}" for r in rates]))
    lines.append("---------------------------------")
    lines.append("")

    # Add uncertainties based on region type
    if region_type == "CR":
        # Luminosity (correlated with SR)
        lines.append("# Luminosity (same nuisance name → correlated with SR)")
        lines.append(f"lumi_13TeV     lnN  " + " ".join(["1.025"] * n_processes))
        lines.append("")

        # Signal suppressed in CR
        lines.append("# Signal suppressed in CR (or effectively zero)")
        sig_unc = ["1.00"] + ["-"] * n_backgrounds
        lines.append(f"sig_{bin_name}_rate    lnN  " + " ".join(sig_unc))
        lines.append("")

        # Dominant background floats freely
        if dominant_bkg:
            lines.append("# Let dominant background float freely")
            dom_bkg_unc = ["-"] * n_processes
            dom_bkg_idx = bg_names.index(dominant_bkg) + 1  # +1 because signal is at index 0
            dom_bkg_unc[dom_bkg_idx] = "5.00"
            lines.append(f"norm_{dominant_bkg}_{bin_name}   lnN  " + " ".join(dom_bkg_unc))
            lines.append("")

        # Subdominant backgrounds constrained
        lines.append("# Subdominant backgrounds constrained")
        for bg_name in bg_names:
            if bg_name != dominant_bkg:
                bg_unc = ["-"] * n_processes
                bg_idx = bg_names.index(bg_name) + 1
                # Set constraint based on background type (can be customized)
                constraint = "1.30"  # Default 30% uncertainty
                if "ttbar" in bg_name.lower() or "tt" in bg_name.lower():
                    constraint = "1.20"
                elif "wlnjets" in bg_name.lower() or "wjets" in bg_name.lower():
                    constraint = "1.30"
                elif "znnjets" in bg_name.lower() or "znu" in bg_name.lower():
                    constraint = "1.25"
                elif "dyjets" in bg_name.lower() or "dy" in bg_name.lower():
                    constraint = "1.40"
                elif "stop" in bg_name.lower():
                    constraint = "1.35"
                bg_unc[bg_idx] = constraint
                lines.append(f"norm_{bg_name}      lnN  " + " ".join(bg_unc))
        lines.append("")

        # Shape uncertainty
        lines.append("# Shape uncertainty")
        lines.append(f"jes            shape " + " ".join(["1"] * n_processes))

    else:  # SR
        # Luminosity uncertainty (correlated everywhere)
        lines.append("# Luminosity uncertainty (correlated everywhere)")
        lines.append(f"lumi_13TeV     lnN  " + " ".join(["1.025"] * n_processes))
        lines.append("")

        # Signal theory uncertainty
        lines.append("# Signal theory uncertainty")
        sig_xsec_unc = ["1.10"] + ["-"] * n_backgrounds
        lines.append(f"sig_xsec       lnN  " + " ".join(sig_xsec_unc))
        lines.append("")

        # Background normalization uncertainties
        lines.append("# Background normalization uncertainties")
        for bg_name in bg_names:
            bg_unc = ["-"] * n_processes
            bg_idx = bg_names.index(bg_name) + 1
            # Set constraint based on background type
            constraint = "1.30"  # Default 30% uncertainty
            if "ttbar" in bg_name.lower() or "tt" in bg_name.lower():
                constraint = "1.20"
            elif "wlnjets" in bg_name.lower() or "wjets" in bg_name.lower():
                constraint = "1.30"
            elif "znnjets" in bg_name.lower() or "znu" in bg_name.lower():
                constraint = "1.25"
            elif "dyjets" in bg_name.lower() or "dy" in bg_name.lower():
                constraint = "1.40"
            elif "stop" in bg_name.lower():
                constraint = "1.35"
            bg_unc[bg_idx] = constraint
            lines.append(f"norm_{bg_name}      lnN  " + " ".join(bg_unc))
        lines.append("")

        # Shape uncertainty
        lines.append("# Shape uncertainty (example)")
        lines.append(f"jes            shape " + " ".join(["1"] * n_processes))

    return "\n".join(lines)


def save_datacard(datacard_content, filename):
    """
    Save datacard to file.

    Parameters:
    -----------
    datacard_content : str
        Datacard content
    filename : str
        Output filename
    """
    with open(filename, 'w') as f:
        f.write(datacard_content)


# ============================================================================
# 8. Utility Functions
# ============================================================================

def print_cutflow(cutflow_dict, sample_name="Sample"):
    """
    Print cutflow as LaTeX table.

    Parameters:
    -----------
    cutflow_dict : dict
        Cutflow dictionary
    sample_name : str
        Sample name for table caption
    """
    print(f"\n\\begin{{table}}[h]")
    print(f"\\centering")
    print(f"\\caption{{Cutflow for {sample_name}}}")
    print(f"\\begin{{tabular}}{{lr}}")
    print(f"\\hline")
    print(f"Selection & Events \\\\")
    print(f"\\hline")

    for cut, n_events in cutflow_dict.items():
        print(f"{cut} & {n_events:,} \\\\")

    print(f"\\hline")
    print(f"\\end{{tabular}}")
    print(f"\\end{{table}}\n")


def export_histogram_to_root(histogram, filename, hist_name="hist"):
    """
    Export histogram to ROOT file (requires uproot).

    Parameters:
    -----------
    histogram : hist.Hist
        Histogram object
    filename : str
        Output ROOT filename
    hist_name : str
        Histogram name in ROOT file
    """
    try:
        import uproot
        with uproot.recreate(filename) as f:
            # Convert hist to ROOT format
            values = histogram.values()
            edges = histogram.axes[0].edges
            f[hist_name] = (values, edges)
    except ImportError:
        print("uproot not available. Install with: pip install uproot")


def combine_shapes_files(shapes_files, bin_names, output_file):
    """
    Combine multiple shapes ROOT files into a single file.

    Parameters:
    -----------
    shapes_files : list
        List of paths to shapes.root files
    bin_names : list
        List of bin names corresponding to each shapes file
    output_file : str
        Output combined shapes ROOT filename

    Returns:
    --------
    success : bool
        True if shapes file was created successfully
    """
    try:
        import uproot
        import numpy as np
    except ImportError:
        print("✗ uproot not available. Install with: pip install uproot")
        return False

    try:
        from pathlib import Path

        with uproot.recreate(output_file) as out_file:
            for shapes_file, bin_name in zip(shapes_files, bin_names):
                shapes_path = Path(shapes_file)
                if not shapes_path.exists():
                    print(f"  ⚠ Warning: Shapes file not found: {shapes_file}")
                    continue

                with uproot.open(shapes_file) as in_file:
                    for key in in_file.keys():
                        # Read histogram
                        hist = in_file[key]
                        values = hist.values()
                        edges = hist.edges() if hasattr(hist, 'edges') else hist.axis().edges()

                        # Write with same name (histogram names should already include bin name)
                        out_file[key] = (values, edges)

        return True
    except Exception as e:
        print(f"✗ Error combining shapes files: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_shapes_file(signal_hist, background_hists, bin_name, output_file,
                       main_observable="met"):
    """
    Create Combine shapes ROOT file from histograms.

    Parameters:
    -----------
    signal_hist : dict or None
        Dictionary of {observable: hist.Hist} for signal, or None
    background_hists : dict
        Dictionary of {bg_name: {observable: hist.Hist}} for backgrounds
    bin_name : str
        Bin name (e.g., "SR" or "CR")
    output_file : str
        Output ROOT filename (e.g., "shapes.root")
    main_observable : str
        Main observable to use (default: "met" for SR, "recoil" for CR)

    Returns:
    --------
    success : bool
        True if shapes file was created successfully
    """
    try:
        import uproot
        import numpy as np
    except ImportError:
        print("✗ uproot not available. Install with: pip install uproot")
        return False

    # Determine main observable if not provided
    if main_observable is None:
        main_observable = "met" if bin_name == "SR" else "recoil"

    # Open ROOT file for writing
    try:
        with uproot.recreate(output_file) as f:
            # Write signal histogram
            if signal_hist is not None and main_observable in signal_hist:
                sig_hist = signal_hist[main_observable]
                values = sig_hist.values()
                edges = sig_hist.axes[0].edges
                hist_name = f"sig_{bin_name}"
                # Create TH1D format: (values, edges) where values are bin contents
                # uproot will create a TH1D from this
                f[hist_name] = (values, edges)

            # Write background histograms
            for bg_name, bg_hists in background_hists.items():
                if main_observable in bg_hists:
                    bg_hist = bg_hists[main_observable]
                    values = bg_hist.values()
                    edges = bg_hist.axes[0].edges
                    hist_name = f"{bg_name}_{bin_name}"
                    f[hist_name] = (values, edges)

        return True
    except Exception as e:
        print(f"✗ Error creating shapes file {output_file}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# 9. Example Usage
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the bbMET processor.

    This shows how to:
    1. Process Delphes ROOT files
    2. Generate cutflows
    3. Create histograms
    4. Generate datacards
    """

    print("bbMET Analysis Pipeline")
    print("=" * 50)
    print("\nThis module provides:")
    print("1. Physics object definitions (jets, b-jets, MET)")
    print("2. Event selection cuts")
    print("3. Coffea processor for production-level analysis")
    print("4. Observable calculations (MET, m(bb), pT(bb), Δφ, HT)")
    print("5. Normalization with cross-section and luminosity")
    print("6. Scan plot functionality (mA, ma)")
    print("7. Combine datacard generation")
    print("\nExample usage:")
    print("""
    from bbdmDelphes import BBMETProcessor, load_delphes_events

    # Process signal (Delphes ROOT file)
    events = load_delphes_events("signal.root")

    processor = BBMETProcessor(
        lumi_fb=139.0,
        xs_pb=1.0,  # Set your cross-section
        ngen=100000,  # Set your number of generated events
        sample_name="signal_mA1500_ma300"
    )
    output = processor.process(events)

    # Process backgrounds
    # ttbar, Z+bb, W+bb, single-top (separate files)

    # Generate datacard
    from bbdmDelphes import generate_datacard
    datacard = generate_datacard(
        signal_name="signal",
        signal_rate=3.2,
        backgrounds={"ttbar": 45.1, "zbb": 12.7},
        uncertainties={
            "lumi": {"signal": 1.025, "ttbar": 1.025, "zbb": 1.025},
            "btag": 1.05  # Same for all
        }
    )
    """)
