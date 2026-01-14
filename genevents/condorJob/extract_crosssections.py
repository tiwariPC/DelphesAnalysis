#!/usr/bin/env python3
"""
Extract cross-section values from output files and create a table with averages.
"""

import os
import re
from collections import defaultdict

# Background names from make_joblist.py
backgrounds = [
    "dyjets",
    "ttbar",
    "diboson",
    "sTop_tchannel",
    "sTop_tW",
    "wlnjets",
    "znnjets"
]

# Directory containing output files
output_dir = "logs/output"

# Pattern to match cross-section line: "Cross-section :   104.3 +- 0.2021 pb"
cross_section_pattern = re.compile(r'Cross-section\s*:\s*([\d.]+)\s*\+-\s*[\d.]+\s*pb')

def extract_cross_section(filepath):
    """Extract cross-section value from a file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            # Find all matches
            matches = cross_section_pattern.findall(content)
            if matches:
                # Return the last match (usually the final result)
                return float(matches[-1])
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
    return None

def main():
    # Dictionary to store cross-sections for each background
    bkg_crosssections = defaultdict(list)

    # Get all .out files in the output directory
    if not os.path.exists(output_dir):
        print(f"Error: Directory {output_dir} does not exist!")
        return

    # Process each file
    for filename in os.listdir(output_dir):
        if not filename.endswith('.out'):
            continue

        # Check which background this file belongs to
        bkg_name = None
        for bkg in backgrounds:
            if filename.startswith(bkg + '_'):
                bkg_name = bkg
                break

        if bkg_name:
            filepath = os.path.join(output_dir, filename)
            cross_section = extract_cross_section(filepath)
            if cross_section is not None:
                bkg_crosssections[bkg_name].append(cross_section)
                print(f"Found {bkg_name}: {cross_section:.4f} pb (from {filename})")

    # Calculate averages and create table
    output_file = "crosssection_table.txt"
    with open(output_file, 'w') as f:
        f.write("# Cross-section table\n")
        f.write("# Format: bkg | cross_section (pb)\n")
        f.write("#" + "=" * 80 + "\n")
        f.write(f"{'background':<20} {'cross_section (pb)'}\n")
        f.write("-" * 80 + "\n")

        for bkg in backgrounds:
            if bkg in bkg_crosssections:
                values = bkg_crosssections[bkg]
                n_files = len(values)
                avg = sum(values) / n_files if n_files > 0 else 0.0
                f.write(f"{bkg:<20} {avg:.3f}\n")
            else:
                f.write(f"{bkg:<20} N/A\n")

    # Also print summary
    print("\n" + "=" * 70)
    print("Cross-Section Summary")
    print("=" * 70)
    print(f"{'Background':<20} {'N files':<10} {'Average [pb]':<15}")
    print("-" * 70)
    for bkg in backgrounds:
        if bkg in bkg_crosssections:
            values = bkg_crosssections[bkg]
            n_files = len(values)
            avg = sum(values) / n_files if n_files > 0 else 0.0
            print(f"{bkg:<20} {n_files:<10} {avg:<15.4f}")
        else:
            print(f"{bkg:<20} {'0':<10} {'N/A':<15}")

    print(f"\n[✔] Results written to {output_file}")

if __name__ == "__main__":
    main()
