#!/usr/bin/env python3
"""
Analyze the structure of Delphes ROOT files to understand the data format.
"""

import sys
import uproot
import awkward as ak
from pathlib import Path


def analyze_root_file(root_file, tree_name="Delphes"):
    """
    Analyze the structure of a ROOT file.

    Parameters:
    -----------
    root_file : str
        Path to ROOT file
    tree_name : str
        Name of TTree
    """
    print("=" * 70)
    print(f"Analyzing ROOT file: {root_file}")
    print("=" * 70)

    if not Path(root_file).exists():
        print(f"Error: File not found: {root_file}")
        return

    try:
        file = uproot.open(root_file)

        # List all keys
        print(f"\n1. All keys in file:")
        print(f"   {list(file.keys())}")

        if tree_name not in file:
            print(f"\nError: Tree '{tree_name}' not found!")
            file.close()
            return

        tree = file[tree_name]

        # Get tree info
        print(f"\n2. Tree '{tree_name}' information:")
        print(f"   Number of entries: {tree.num_entries}")
        try:
            print(f"   Number of baskets: {tree.num_baskets}")
        except AttributeError:
            print(f"   Number of baskets: N/A (not available in this uproot version)")

        # List all branches
        print(f"\n3. All branches in tree:")
        all_branches = list(tree.keys())
        for i, branch in enumerate(all_branches[:20]):  # Show first 20
            print(f"   {i+1}. {branch}")
        if len(all_branches) > 20:
            print(f"   ... and {len(all_branches) - 20} more branches")

        # Try to load Jet branch
        print(f"\n4. Analyzing Jet branch:")
        if "Jet" in all_branches:
            try:
                jet_branch = tree["Jet"]
                print(f"   Jet branch type: {type(jet_branch)}")
                try:
                    print(f"   Jet branch interpretation: {jet_branch.interpretation}")
                except AttributeError:
                    print(f"   Jet branch interpretation: N/A")

                # Try to load first few events (need multiple to get array, not record)
                jet_data = tree["Jet"].arrays(entry_start=0, entry_stop=min(5, tree.num_entries), library="ak")
                print(f"   Jet data type: {type(jet_data)}")

                # Check if it's a Record or Array
                if isinstance(jet_data, ak.Record):
                    print(f"   Jet data is a Record (single event)")
                    print(f"   Jet Record fields: {jet_data.fields if hasattr(jet_data, 'fields') else 'N/A'}")
                    # Check if it has PT, Eta, etc. as arrays
                    if hasattr(jet_data, 'PT'):
                        pt_array = jet_data.PT
                        print(f"   Jet.PT type: {type(pt_array)}, length: {len(pt_array) if hasattr(pt_array, '__len__') else 'N/A'}")
                        if len(pt_array) > 0:
                            print(f"   First jet PT: {pt_array[0]}")
                    if hasattr(jet_data, 'Eta'):
                        eta_array = jet_data.Eta
                        print(f"   Jet.Eta type: {type(eta_array)}")
                        if len(eta_array) > 0:
                            print(f"   First jet Eta: {eta_array[0]}")
                    if hasattr(jet_data, 'BTag'):
                        btag_array = jet_data.BTag
                        print(f"   Jet.BTag type: {type(btag_array)}")
                        if len(btag_array) > 0:
                            print(f"   First jet BTag: {btag_array[0]}")
                else:
                    # It's an array
                    try:
                        print(f"   Jet data shape: {ak.num(jet_data)}")
                        if len(jet_data) > 0:
                            first_event = jet_data[0]
                            if isinstance(first_event, ak.Record):
                                print(f"   First event is a Record with fields: {first_event.fields if hasattr(first_event, 'fields') else 'N/A'}")
                            elif hasattr(first_event, '__len__') and len(first_event) > 0:
                                first_jet = first_event[0]
                                print(f"   First jet fields: {first_jet.fields if hasattr(first_jet, 'fields') else 'N/A'}")
                                if hasattr(first_jet, 'PT'):
                                    print(f"   First jet PT: {first_jet.PT}")
                                if hasattr(first_jet, 'Eta'):
                                    print(f"   First jet Eta: {first_jet.Eta}")
                                if hasattr(first_jet, 'BTag'):
                                    print(f"   First jet BTag: {first_jet.BTag}")
                    except Exception as e:
                        print(f"   Could not analyze array structure: {e}")
            except Exception as e:
                print(f"   Error loading Jet: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"   Jet branch not found!")

        # Try to load MissingET branch
        print(f"\n5. Analyzing MissingET branch:")
        if "MissingET" in all_branches:
            try:
                met_branch = tree["MissingET"]
                print(f"   MissingET branch type: {type(met_branch)}")
                try:
                    print(f"   MissingET branch interpretation: {met_branch.interpretation}")
                except AttributeError:
                    print(f"   MissingET branch interpretation: N/A")

                # Try to load first few events (need multiple to get array, not record)
                met_data = tree["MissingET"].arrays(entry_start=0, entry_stop=min(5, tree.num_entries), library="ak")
                print(f"   MissingET data type: {type(met_data)}")

                # Check if it's a Record or Array
                if isinstance(met_data, ak.Record):
                    print(f"   MissingET data is a Record (single event)")
                    print(f"   MissingET Record fields: {met_data.fields if hasattr(met_data, 'fields') else 'N/A'}")
                    if hasattr(met_data, 'MET'):
                        met_value = met_data.MET
                        print(f"   MissingET.MET type: {type(met_value)}, value: {met_value}")
                    if hasattr(met_data, 'Phi'):
                        phi_value = met_data.Phi
                        print(f"   MissingET.Phi type: {type(phi_value)}, value: {phi_value}")
                else:
                    # It's an array
                    try:
                        print(f"   MissingET data shape: {ak.num(met_data)}")
                        if len(met_data) > 0:
                            first_met = met_data[0]
                            if isinstance(first_met, ak.Record):
                                print(f"   First MET is a Record with fields: {first_met.fields if hasattr(first_met, 'fields') else 'N/A'}")
                                if hasattr(first_met, 'MET'):
                                    print(f"   First MET value: {first_met.MET}")
                                if hasattr(first_met, 'Phi'):
                                    print(f"   First MET Phi: {first_met.Phi}")
                            else:
                                print(f"   First MET type: {type(first_met)}")
                                if hasattr(first_met, 'MET'):
                                    print(f"   First MET value: {first_met.MET}")
                                if hasattr(first_met, 'Phi'):
                                    print(f"   First MET Phi: {first_met.Phi}")
                    except Exception as e:
                        print(f"   Could not analyze array structure: {e}")
            except Exception as e:
                print(f"   Error loading MissingET: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"   MissingET branch not found!")

        # Try loading both together
        print(f"\n6. Loading Jet and MissingET together:")
        try:
            events = tree.arrays(["Jet", "MissingET"], library="ak", entry_start=0, entry_stop=min(5, tree.num_entries))
            print(f"   Events type: {type(events)}")
            print(f"   Events keys/fields: {events.fields if hasattr(events, 'fields') else list(events.keys()) if isinstance(events, dict) else 'N/A'}")

            if hasattr(events, "Jet"):
                print(f"   Can access events.Jet: Yes")
                print(f"   events.Jet type: {type(events.Jet)}")
            elif "Jet" in events:
                print(f"   Can access events['Jet']: Yes")
                print(f"   events['Jet'] type: {type(events['Jet'])}")

            if hasattr(events, "MissingET"):
                print(f"   Can access events.MissingET: Yes")
            elif "MissingET" in events:
                print(f"   Can access events['MissingET']: Yes")

        except Exception as e:
            print(f"   Error loading together: {e}")
            import traceback
            traceback.print_exc()

        file.close()

        print("\n" + "=" * 70)
        print("Analysis complete!")
        print("=" * 70)

    except Exception as e:
        print(f"Error analyzing file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_root_structure.py <root_file> [tree_name]")
        print("\nExample:")
        print("  python analyze_root_structure.py signal.root")
        print("  python analyze_root_structure.py signal.root Delphes")
        sys.exit(1)

    root_file = sys.argv[1]
    tree_name = sys.argv[2] if len(sys.argv) > 2 else "Delphes"

    analyze_root_file(root_file, tree_name)
