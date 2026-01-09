"""
Region definitions and selection logic for bbMET analysis.

Loads regions from YAML configuration and provides region selection functions.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import awkward as ak


def load_regions(regions_file: str = None) -> Dict[str, Any]:
    """
    Load region definitions from YAML file.

    If regions_file is None, tries to load from cuts_config.yaml in current directory.

    Parameters:
    -----------
    regions_file : str, optional
        Path to YAML file containing regions. If None, uses cuts_config.yaml

    Returns:
    --------
    regions : dict
        Dictionary of region definitions
    """
    if regions_file is None:
        # Try to load from cuts_config.yaml in current directory
        regions_file = "cuts_config.yaml"

    regions_path = Path(regions_file)

    if not regions_path.exists():
        raise FileNotFoundError(f"Regions file not found: {regions_file}")

    # Use safe_load which preserves order in Python 3.7+ (dicts are ordered by default)
    # To be extra safe, we'll ensure the cuts dict is OrderedDict when processing
    with open(regions_path, 'r') as f:
        config = yaml.safe_load(f)

    regions = config.get('regions', {})

    if not regions:
        raise ValueError(f"No 'regions' section found in {regions_file}")

    return regions


def parse_cut_condition(cut_value: str, variable_value: Any) -> ak.Array:
    """
    Parse and evaluate a cut condition.

    Parameters:
    -----------
    cut_value : str
        Cut condition string (e.g., "==1", ">50", "<=2")
    variable_value : Any
        Value to compare (awkward array or numpy array)

    Returns:
    --------
    result : ak.Array
        Boolean array result of the condition
    """
    # Convert to numpy if it's an awkward array for easier comparison
    if isinstance(variable_value, ak.Array):
        var_np = ak.to_numpy(variable_value)
    else:
        var_np = np.asarray(variable_value)

    if isinstance(cut_value, (int, float)):
        # Direct comparison
        return ak.Array(var_np == cut_value)

    if not isinstance(cut_value, str):
        return ak.Array(var_np == cut_value)

    # Parse string conditions
    cut_value = cut_value.strip()

    if cut_value.startswith("=="):
        threshold = float(cut_value[2:])
        return ak.Array(var_np == threshold)
    elif cut_value.startswith("!="):
        threshold = float(cut_value[2:])
        return ak.Array(var_np != threshold)
    elif cut_value.startswith(">="):
        threshold = float(cut_value[2:])
        return ak.Array(var_np >= threshold)
    elif cut_value.startswith("<="):
        threshold = float(cut_value[2:])
        return ak.Array(var_np <= threshold)
    elif cut_value.startswith(">"):
        threshold = float(cut_value[1:])
        return ak.Array(var_np > threshold)
    elif cut_value.startswith("<"):
        threshold = float(cut_value[1:])
        return ak.Array(var_np < threshold)
    else:
        # Try to parse as number
        try:
            threshold = float(cut_value)
            return ak.Array(var_np == threshold)
        except ValueError:
            raise ValueError(f"Could not parse cut condition: {cut_value}")


def select_region(events: ak.Array,
                  region_name: str,
                  regions_config: Dict[str, Any],
                  jets: Dict[str, ak.Array],
                  bjets: Dict[str, ak.Array],
                  met: Dict[str, ak.Array],
                  electrons: Optional[Dict[str, ak.Array]] = None,
                  muons: Optional[Dict[str, ak.Array]] = None,
                  taus: Optional[Dict[str, ak.Array]] = None,
                  additional_vars: Optional[Dict[str, Any]] = None) -> ak.Array:
    """
    Apply region selection cuts.

    Parameters:
    -----------
    events : ak.Array
        Events array
    region_name : str
        Name of region (e.g., "1b:SR", "2b:CR_Wlnu_mu")
    regions_config : dict
        Region configuration dictionary
    jets : dict
        Jet collection with 'PT', 'Eta', 'Phi', 'BTag' keys
    bjets : dict
        B-jet collection
    met : dict
        MET collection with 'MET' and 'Phi' keys
    electrons : dict, optional
        Electron collection
    muons : dict, optional
        Muon collection
    taus : dict, optional
        Tau collection (not used, kept for API compatibility)
    additional_vars : dict, optional
        Additional variables (MT, Recoil, Mll, etc.)

    Returns:
    --------
    mask : ak.Array
        Boolean mask for events passing region selection
    """
    if region_name not in regions_config:
        raise ValueError(f"Region '{region_name}' not found in configuration")

    region_cuts = regions_config[region_name].get('cuts', {})

    # Count objects per event
    njets = ak.num(jets['PT'], axis=1)
    nbjets = ak.num(bjets['PT'], axis=1)

    # Start with all events passing - use njets to get the right shape
    n_events = len(njets)
    mask = ak.Array(np.ones(n_events, dtype=bool))

    # Apply cuts
    if 'Njets' in region_cuts:
        mask = mask & parse_cut_condition(region_cuts['Njets'], njets)

    if 'NjetsMin' in region_cuts:
        mask = mask & (njets >= float(region_cuts['NjetsMin'].replace('>', '')))

    if 'Nbjets' in region_cuts:
        mask = mask & parse_cut_condition(region_cuts['Nbjets'], nbjets)

    if 'Nleptons' in region_cuts:
        nleptons = 0
        if electrons is not None:
            nleptons = nleptons + ak.num(electrons.get('PT', ak.Array([])), axis=1)
        if muons is not None:
            nleptons = nleptons + ak.num(muons.get('PT', ak.Array([])), axis=1)
        mask = mask & parse_cut_condition(region_cuts['Nleptons'], nleptons)

    if 'Nelectrons' in region_cuts:
        if electrons is None:
            nelectrons = ak.zeros_like(njets)
        else:
            nelectrons = ak.num(electrons.get('PT', ak.Array([])), axis=1)
        mask = mask & parse_cut_condition(region_cuts['Nelectrons'], nelectrons)

    if 'Nmuons' in region_cuts:
        if muons is None:
            nmuons = ak.zeros_like(njets)
        else:
            nmuons = ak.num(muons.get('PT', ak.Array([])), axis=1)
        mask = mask & parse_cut_condition(region_cuts['Nmuons'], nmuons)

    if 'Ntaus' in region_cuts:
        # Taus are not available in input files, so always set to 0
        ntaus = ak.zeros_like(njets)
        mask = mask & parse_cut_condition(region_cuts['Ntaus'], ntaus)

    if 'MET' in region_cuts:
        met_values = ak.flatten(met['MET'], axis=1)
        mask = mask & parse_cut_condition(region_cuts['MET'], met_values)

    if 'DeltaPhi' in region_cuts:
        # Calculate Δφ(MET, leading jet) using DeltaPhi function
        from src.bbdmDelphes import DeltaPhi
        leading_jet_phi = ak.firsts(jets['Phi'], axis=1)
        leading_jet_phi = ak.fill_none(leading_jet_phi, 0.0)
        met_phi = ak.flatten(met['Phi'], axis=1)
        dphi_met_jet = DeltaPhi(leading_jet_phi, met_phi)
        # Convert to numpy for comparison
        if isinstance(dphi_met_jet, ak.Array):
            dphi_met_jet = np.asarray(ak.to_numpy(dphi_met_jet))
        mask = mask & parse_cut_condition(region_cuts['DeltaPhi'], dphi_met_jet)

    if 'Recoil' in region_cuts and additional_vars is not None:
        # Try different recoil keys
        recoil = None
        for key in ['Recoil', 'Recoil_mu', 'Recoil_el']:
            if key in additional_vars:
                recoil = additional_vars[key]
                break
        if recoil is not None:
            mask = mask & parse_cut_condition(region_cuts['Recoil'], recoil)

    if 'MT' in region_cuts and additional_vars is not None:
        # Try different MT keys
        mt = None
        for key in ['MT', 'MT_mu', 'MT_el']:
            if key in additional_vars:
                mt = additional_vars[key]
                break
        if mt is not None:
            mask = mask & parse_cut_condition(region_cuts['MT'], mt)

    if 'MllMin' in region_cuts and additional_vars is not None:
        # Try different Mll keys
        mll = None
        for key in ['Mll', 'Mll_mu', 'Mll_el']:
            if key in additional_vars:
                mll = additional_vars[key]
                break
        if mll is not None:
            # Convert to numpy for comparison
            if isinstance(mll, ak.Array):
                mll_np = np.asarray(ak.to_numpy(mll))
            else:
                mll_np = np.asarray(mll)
            threshold = float(region_cuts['MllMin'].replace('>', ''))
            mask = mask & ak.Array(mll_np >= threshold)

    if 'MllMax' in region_cuts and additional_vars is not None:
        # Try different Mll keys
        mll = None
        for key in ['Mll', 'Mll_mu', 'Mll_el']:
            if key in additional_vars:
                mll = additional_vars[key]
                break
        if mll is not None:
            # Convert to numpy for comparison
            if isinstance(mll, ak.Array):
                mll_np = np.asarray(ak.to_numpy(mll))
            else:
                mll_np = np.asarray(mll)
            threshold = float(region_cuts['MllMax'].replace('<', ''))
            mask = mask & ak.Array(mll_np <= threshold)

    if 'NAdditionalJets' in region_cuts:
        # Additional jets = total jets - b-jets
        nadditional = njets - nbjets
        mask = mask & parse_cut_condition(region_cuts['NAdditionalJets'], nadditional)

    if 'Jet1Pt' in region_cuts:
        jet1_pt = ak.firsts(jets['PT'], axis=1)
        jet1_pt = ak.fill_none(jet1_pt, 0.0)
        mask = mask & parse_cut_condition(region_cuts['Jet1Pt'], jet1_pt)

    if 'dphi_jet_met_min' in region_cuts:
        # Calculate minimum dphi(jet, MET) for all jets in each event
        from src.bbdmDelphes import DeltaPhi

        # Get MET phi (single value per event)
        met_phi = ak.flatten(met['Phi'], axis=1)
        met_phi_np = np.asarray(ak.to_numpy(met_phi))

        # Calculate dphi for all jets in all events
        jets_phi = jets['Phi']
        n_jets_per_event = ak.num(jets_phi, axis=1)

        # Create expanded met_phi array matching jets_phi structure
        met_phi_list = []
        for i in range(len(met_phi_np)):
            n_jet = int(ak.to_numpy(n_jets_per_event)[i])
            met_phi_list.append([met_phi_np[i]] * n_jet)
        met_phi_repeated = ak.Array(met_phi_list)

        # Calculate Δφ for all jets using DeltaPhi function
        dphi_all_jets = DeltaPhi(jets_phi, met_phi_repeated)

        # Get minimum dphi per event (NaN if no jets)
        try:
            if len(dphi_all_jets) == 0:
                dphi_jet_met_min = np.full(len(met_phi_np), np.nan)
            else:
                min_dphi_per_event = ak.min(dphi_all_jets, axis=1)
                dphi_jet_met_min = np.asarray(ak.to_numpy(ak.fill_none(min_dphi_per_event, np.nan)))
        except (ValueError, AttributeError):
            # Fallback: calculate manually for each event
            dphi_jet_met_min = np.full(len(met_phi_np), np.nan)
            jets_phi_list = ak.to_list(jets_phi)
            for i in range(len(met_phi_np)):
                if i < len(jets_phi_list) and len(jets_phi_list[i]) > 0:
                    event_jets_phi = ak.Array(jets_phi_list[i])
                    event_met_phi = ak.Array([met_phi_np[i]] * len(jets_phi_list[i]))
                    event_dphi = DeltaPhi(event_jets_phi, event_met_phi)
                    if isinstance(event_dphi, ak.Array):
                        event_dphi_list = ak.to_list(event_dphi)
                        if isinstance(event_dphi_list, list) and len(event_dphi_list) > 0:
                            dphi_jet_met_min[i] = np.min(event_dphi_list)
                    elif isinstance(event_dphi, (list, np.ndarray)):
                        dphi_jet_met_min[i] = np.min(event_dphi)

        # Handle NaN values (events without jets) - these fail the cut
        valid_dphi = ~np.isnan(dphi_jet_met_min)
        dphi_mask = np.zeros(n_events, dtype=bool)
        if np.sum(valid_dphi) > 0:
            dphi_valid = parse_cut_condition(region_cuts['dphi_jet_met_min'], dphi_jet_met_min[valid_dphi])
            if isinstance(dphi_valid, ak.Array):
                dphi_mask[valid_dphi] = np.asarray(ak.to_numpy(dphi_valid))
            else:
                dphi_mask[valid_dphi] = np.asarray(dphi_valid)

        mask = mask & ak.Array(dphi_mask)

    return mask


def get_region_type(region_name: str) -> str:
    """
    Get region type (SR or CR) from region name.

    Parameters:
    -----------
    region_name : str
        Region name (e.g., "1b:SR", "2b:CR_Wlnu_mu")

    Returns:
    --------
    region_type : str
        "SR" or "CR"
    """
    if ":SR" in region_name:
        return "SR"
    elif ":CR" in region_name:
        return "CR"
    else:
        return "UNKNOWN"


def get_region_category(region_name: str) -> str:
    """
    Get region category (1b or 2b) from region name.

    Parameters:
    -----------
    region_name : str
        Region name (e.g., "1b:SR", "2b:CR_Wlnu_mu")

    Returns:
    --------
    category : str
        "1b" or "2b"
    """
    if region_name.startswith("1b:"):
        return "1b"
    elif region_name.startswith("2b:"):
        return "2b"
    else:
        return "unknown"
