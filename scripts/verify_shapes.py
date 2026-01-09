#!/usr/bin/env python3
"""
Verify that shapes ROOT files are correctly formatted for Combine.
"""

import sys
from pathlib import Path
import uproot

def verify_shapes_file(shapes_file):
    """Verify a shapes ROOT file."""
    shapes_file = Path(shapes_file)
    if not shapes_file.exists():
        print(f"✗ Shapes file not found: {shapes_file}")
        return False

    try:
        with uproot.open(shapes_file) as f:
            keys = list(f.keys())
            print(f"\n✓ Shapes file: {shapes_file}")
            print(f"  Histograms found: {len(keys)}")
            for key in keys:
                hist = f[key]
                # Get histogram properties using uproot API
                values = hist.values()
                # For TH1D, get bin edges from axis
                axis = hist.axis()
                n_bins = axis.member("fNbins")
                integral = sum(values)
                print(f"    - {key}: {n_bins} bins, integral = {integral:.2f}")
            return True
    except Exception as e:
        print(f"✗ Error reading {shapes_file}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        shapes_file = sys.argv[1]
        verify_shapes_file(shapes_file)
    else:
        # Test default locations
        output_dir = Path("output/plots")
        for region_dir in output_dir.glob("*"):
            shapes_file = region_dir / "shapes.root"
            if shapes_file.exists():
                verify_shapes_file(shapes_file)
