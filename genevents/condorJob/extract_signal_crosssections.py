#!/usr/bin/env python3
"""
Extract cross-section values from signal output files and create a table with averages.
Files are named as: mA{value}ma{value}seed{seed}.{jobid}.{number}.out
"""

import os
import re
from collections import defaultdict

# Directory containing output files
output_dir = "logs/output"

# Pattern to match cross-section line: "Cross-section :   0.0002919 +- 0 pb"
# Also handles scientific notation like "3.928e-05"
cross_section_pattern = re.compile(r'Cross-section\s*:\s*([\d.eE+-]+)\s*\+-\s*[\d.eE+-]+\s*pb')

# Pattern to extract mA and ma from filename: mA300ma50seed...
filename_pattern = re.compile(r'mA(\d+)ma(\d+)seed')

def extract_cross_section(filepath):
    """Extract cross-section value from a file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            # Find all matches
            matches = cross_section_pattern.findall(content)
            if matches:
                # Return the last match (usually the final result)
                value_str = matches[-1]
                # Handle scientific notation
                if 'e' in value_str.lower() or 'E' in value_str:
                    return float(value_str)
                else:
                    return float(value_str)
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
    return None

def parse_filename(filename):
    """Extract mA and ma values from filename."""
    match = filename_pattern.match(filename)
    if match:
        mA = int(match.group(1))
        ma = int(match.group(2))
        return mA, ma
    return None, None

def main():
    # Dictionary to store cross-sections: {(mA, ma): [values]}
    signal_crosssections = defaultdict(list)

    # Get all .out files in the output directory
    if not os.path.exists(output_dir):
        print(f"Error: Directory {output_dir} does not exist!")
        return

    # Process each file
    for filename in os.listdir(output_dir):
        if not filename.endswith('.out'):
            continue

        # Check if this is a signal file (starts with mA)
        if not filename.startswith('mA'):
            continue

        # Parse mA and ma from filename
        mA, ma = parse_filename(filename)
        if mA is None or ma is None:
            continue

        filepath = os.path.join(output_dir, filename)
        cross_section = extract_cross_section(filepath)
        if cross_section is not None:
            signal_crosssections[(mA, ma)].append(cross_section)
            print(f"Found mA{mA} ma{ma}: {cross_section:.10f} pb (from {filename})")

    # Sort by mA first, then ma
    sorted_keys = sorted(signal_crosssections.keys(), key=lambda x: (x[0], x[1]))

    # Create table
    output_file = "signal_crosssection_table.txt"
    with open(output_file, 'w') as f:
        f.write("# Cross-section table for signal\n")
        f.write("# Format: mA | ma | cross_section (pb)\n")
        f.write("#" + "=" * 80 + "\n")
        f.write(f"{'mA':<10} {'ma':<10} {'cross_section (pb)':<20}\n")
        f.write("-" * 80 + "\n")

        for mA, ma in sorted_keys:
            values = signal_crosssections[(mA, ma)]
            n_files = len(values)
            avg = sum(values) / n_files if n_files > 0 else 0.0
            # Use decimal format with enough precision
            f.write(f"{mA:<10} {ma:<10} {avg:.10f}\n")

    # Also print summary
    print("\n" + "=" * 80)
    print("Signal Cross-Section Summary")
    print("=" * 80)
    print(f"{'mA':<10} {'ma':<10} {'N files':<10} {'Average [pb]':<20}")
    print("-" * 80)
    for mA, ma in sorted_keys:
        values = signal_crosssections[(mA, ma)]
        n_files = len(values)
        avg = sum(values) / n_files if n_files > 0 else 0.0
        print(f"{mA:<10} {ma:<10} {n_files:<10} {avg:.10f}")

    print(f"\n[✔] Results written to {output_file}")

if __name__ == "__main__":
    main()
