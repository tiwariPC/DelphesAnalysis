#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
"""
Create a combined datacard from all regions for limit extraction.

This script combines datacards from all regions into a single datacard
that can be used with CMS Combine for limit extraction.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.bbdmDelphes import combine_shapes_files


def parse_datacard(datacard_file: Path) -> Dict:
    """
    Parse a single datacard file.

    Returns:
    --------
    info : dict
        Dictionary with processes, rates, uncertainties, bin_name, shapes_file
    """
    with open(datacard_file, 'r') as f:
        content = f.read()

    info = {
        "bin_name": None,
        "observation": None,
        "processes": [],
        "rates": [],
        "uncertainties": {},
        "shapes_file": None,
        "shapes_line": None
    }

    # Extract bin name (first occurrence after separator)
    bin_match = re.search(r'^bin\s+(\S+)', content, re.MULTILINE)
    if bin_match:
        info["bin_name"] = bin_match.group(1)

    # Extract observation
    obs_match = re.search(r'^observation\s+(-?\d+)', content, re.MULTILINE)
    if obs_match:
        info["observation"] = int(obs_match.group(1))

    # Extract shapes line
    shapes_match = re.search(r'^shapes\s+(.+)', content, re.MULTILINE)
    if shapes_match:
        info["shapes_line"] = shapes_match.group(1)
        # Extract shapes file name (second field)
        shapes_parts = shapes_match.group(1).split()
        if len(shapes_parts) >= 2:
            info["shapes_file"] = shapes_parts[1]

    # Extract process names (first process line after separator)
    proc_lines = re.findall(r'^process\s+(.+)', content, re.MULTILINE)
    if len(proc_lines) >= 1:
        # First process line has names, second has numbers
        processes = proc_lines[0].split()
        info["processes"] = processes

    # Extract rates
    rate_match = re.search(r'^rate\s+(.+)', content, re.MULTILINE)
    if rate_match:
        rates_str = rate_match.group(1).split()
        rates = []
        for r in rates_str:
            try:
                rates.append(float(r))
            except ValueError:
                rates.append(0.0)
        info["rates"] = rates

    # Extract uncertainties (skip comment lines)
    # Format: uncertainty_name type value1 value2 ...
    # Skip lines starting with #
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # Match uncertainty lines (not bin, observation, process, rate, shapes, or separator)
        if (line and
            not line.startswith('bin') and
            not line.startswith('observation') and
            not line.startswith('process') and
            not line.startswith('rate') and
            not line.startswith('shapes') and
            not line.startswith('---')):
            parts = line.split()
            if len(parts) >= 3:
                unc_name = parts[0]
                unc_type = parts[1]
                # Handle "-" as 1.0 (no effect)
                unc_values = []
                for v in parts[2:]:
                    if v == '-':
                        unc_values.append(1.0)
                    else:
                        try:
                            unc_values.append(float(v))
                        except ValueError:
                            unc_values.append(1.0)

                info["uncertainties"][unc_name] = {
                    "type": unc_type,
                    "values": unc_values
                }

    return info


def combine_datacards(datacard_infos: List[Dict], output_dir: Path = None) -> str:
    """
    Combine multiple datacards into a single datacard.

    Parameters:
    -----------
    datacard_infos : list
        List of datacard info dictionaries
    output_dir : Path, optional
        Output directory for creating combined shapes file

    Returns:
    --------
    combined_datacard : str
        Combined datacard content
    """
    # Get all unique processes across all bins
    all_processes = set()
    for info in datacard_infos:
        all_processes.update(info["processes"])

    # Order: signal first, then backgrounds alphabetically
    signal_processes = [p for p in all_processes if p == "sig"]
    bg_processes = sorted([p for p in all_processes if p != "sig"])
    process_order = signal_processes + bg_processes

    # Get all unique uncertainties
    all_uncertainties = set()
    for info in datacard_infos:
        all_uncertainties.update(info["uncertainties"].keys())

    # Build combined datacard
    lines = []
    lines.append(f"imax {len(datacard_infos)}")
    lines.append(f"jmax {len(process_order) - 1}")
    lines.append(f"kmax *")
    lines.append("---------------------------------")

    # Shapes lines will be added after we determine final_bin_names (see below)

    # Bin names (one line listing all bins)
    # Each region becomes a separate bin
    # If multiple regions have same bin name (SR or CR), number them (SR1, SR2, CR1, CR2, etc.)
    bin_names = [info["bin_name"] for info in datacard_infos]

    # Create numbered bin names for the declaration line
    # Count occurrences of each bin type
    bin_name_counts = {}
    numbered_bin_names = []
    for bn in bin_names:
        if bn not in bin_name_counts:
            bin_name_counts[bn] = 0
        bin_name_counts[bn] += 1
        count = bin_name_counts[bn]
        # Number bins if there are multiple of the same type
        if bin_name_counts[bn] == 1 and bin_name_counts.get(bn, 0) <= 1:
            # First occurrence - check if there will be more
            remaining = bin_names.count(bn) - 1
            if remaining > 0:
                numbered_bin_names.append(f"{bn}{count}")
            else:
                numbered_bin_names.append(bn)
        else:
            numbered_bin_names.append(f"{bn}{count}")

    # Actually, simpler: number all bins of same type
    final_bin_names = []
    sr_count = 1
    cr_count = 1
    for bn in bin_names:
        if bn == "SR":
            if bin_names.count("SR") > 1:
                final_bin_names.append(f"SR{sr_count}")
                sr_count += 1
            else:
                final_bin_names.append("SR")
        elif bn == "CR":
            if bin_names.count("CR") > 1:
                final_bin_names.append(f"CR{cr_count}")
                cr_count += 1
            else:
                final_bin_names.append("CR")
        else:
            final_bin_names.append(bn)

    # Now add shapes lines (after we know final_bin_names)
    if output_dir:
        combined_shapes_file = "combined_shapes.root"
        # Build shapes line: one per numbered bin name
        # Format: shapes * bin_name shapes_file $PROCESS_base_name $PROCESS_base_name_$SYSTEMATIC
        # Extract base names for shapes (SR1 -> SR, CR1 -> CR)
        for final_bin in final_bin_names:
            base_name = final_bin.rstrip('1234567890')  # Remove trailing digits
            lines.append(f"shapes * {final_bin} {combined_shapes_file} $PROCESS_{base_name} $PROCESS_{base_name}_$SYSTEMATIC")
    else:
        # Reference original shapes files
        shapes_by_bin = {}
        for info in datacard_infos:
            bin_name = info["bin_name"]
            if bin_name not in shapes_by_bin and info.get("shapes_line"):
                shapes_by_bin[bin_name] = info["shapes_line"]
        # Map final bins to base names for shapes
        for final_bin in final_bin_names:
            base_name = final_bin.rstrip('1234567890')
            if base_name in shapes_by_bin:
                # Update shapes line to use numbered bin
                shapes_line = shapes_by_bin[base_name]
                # Replace base name with numbered bin in shapes line
                updated_shapes = shapes_line.replace(f" {base_name} ", f" {final_bin} ")
                lines.append(f"shapes {updated_shapes}")

    lines.append("---------------------------------")

    lines.append("bin           " + " ".join(final_bin_names))

    # Observations (one per bin)
    observations = [str(info["observation"]) for info in datacard_infos]
    lines.append("observation   " + " ".join(observations))
    lines.append("---------------------------------")

    # Process lines
    # Format: one "bin" line with all bins repeated for each process, then process names, then numbers, then rates
    # For each bin (in order), list all processes (in order)
    process_lines = []
    rate_lines = []

    # Process each datacard info in order (matching final_bin_names)
    for i, info in enumerate(datacard_infos):
        # Create a mapping from process to rate for this bin
        process_to_rate = dict(zip(info["processes"], info["rates"]))

        # Build process line for this bin
        bin_processes = []
        bin_rates = []

        for proc in process_order:
            if proc in process_to_rate:
                bin_processes.append(proc)
                bin_rates.append(process_to_rate[proc])
            else:
                # Process not in this bin, use 0.0
                bin_processes.append(proc)
                bin_rates.append(0.0)

        process_lines.append(" ".join(bin_processes))
        rate_lines.append(" ".join([f"{r:.4f}" for r in bin_rates]))

    # Build the combined process section
    # One "bin" line: repeat each bin name for each process
    combined_bin_line = []
    for bin_name in final_bin_names:
        combined_bin_line.extend([bin_name] * len(process_order))
    lines.append("bin           " + " ".join(combined_bin_line))

    # One "process" line: repeat process order for each bin
    combined_process_line = process_order * len(final_bin_names)
    lines.append("process       " + " ".join(combined_process_line))

    # One "process" line with numbers: 0,1,2,3,4 repeated for each bin
    process_numbers = []
    for i in range(len(final_bin_names)):
        process_numbers.extend([str(j) for j in range(len(process_order))])
    lines.append("process       " + " ".join(process_numbers))

    # One "rate" line: all rates for all bins
    combined_rate_line = " ".join(rate_lines)
    lines.append("rate          " + combined_rate_line)

    lines.append("---------------------------------")
    lines.append("")

    # Uncertainties
    for unc_name in sorted(all_uncertainties):
        unc_type = None
        unc_values_all = []

        # Collect uncertainty values for all bins and processes (in bin order)
        for info in datacard_infos:

            if info and unc_name in info["uncertainties"]:
                unc_data = info["uncertainties"][unc_name]
                if unc_type is None:
                    unc_type = unc_data["type"]

                # Map values to process order
                process_to_unc = dict(zip(info["processes"], unc_data["values"]))
                for proc in process_order:
                    if proc in process_to_unc:
                        val = process_to_unc[proc]
                        # Use "-" for 1.0 to match format
                        if val == 1.0:
                            unc_values_all.append("-")
                        else:
                            unc_values_all.append(str(val))
                    else:
                        # Process not in this bin, use "-" (no effect)
                        unc_values_all.append("-")
            else:
                # Uncertainty not in this bin, use "-" for all processes
                for proc in process_order:
                    unc_values_all.append("-")

        if unc_type:
            lines.append(f"{unc_name}     {unc_type}  " + " ".join(unc_values_all))
            lines.append("")

    return "\n".join(lines)


def find_all_datacards(output_dir: Path) -> List[Tuple[Path, str]]:
    """Find all datacard.txt files in output directory."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Create combined datacard from all regions"
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
        default="combined_datacard.txt",
        help="Output filename for combined datacard"
    )
    parser.add_argument(
        "--regions",
        type=str,
        nargs="+",
        default=None,
        help="Specific regions to include (default: all)"
    )
    parser.add_argument(
        "--sr-cr-only",
        action="store_true",
        help="Combine only SR and CR regions (exclude category-specific regions)"
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

    print(f"  Found {len(datacards)} datacards:")
    for _, region_name in datacards:
        print(f"    - {region_name}")

    # Filter by regions if specified
    if args.regions:
        datacards = [(dc, rn) for dc, rn in datacards if rn in args.regions]
        if not datacards:
            print(f"✗ No datacards found for specified regions!")
            return

    # Filter for SR+CR only if requested
    if args.sr_cr_only:
        # Group by category and combine SR and CR
        # e.g., sr1b + cr1b_wlnu, sr2b + cr2b_top
        import re
        sr_cr_datacards = []
        for dc, rn in datacards:
            # Check if it's an SR or CR region
            if rn.startswith('sr') or rn.startswith('cr'):
                sr_cr_datacards.append((dc, rn))
        datacards = sr_cr_datacards
        if not datacards:
            print(f"✗ No SR/CR datacards found!")
            return
        print(f"  Filtered to {len(datacards)} SR/CR datacards")

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
            print(f"    ✓ {len(info['processes'])} processes, {len(info['rates'])} rates")
        except Exception as e:
            print(f"    ✗ Error: {e}")

    if not datacard_infos:
        print("✗ No datacards parsed successfully!")
        return

    # Combine datacards
    print("\n" + "="*70)
    print("Combining Datacards")
    print("="*70)
    combined_datacard = combine_datacards(datacard_infos, output_dir=output_dir)

    # Save combined datacard in plots directory (outside region dirs but inside plots)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_file = plots_dir / args.output_file
    with open(output_file, 'w') as f:
        f.write(combined_datacard)

    print(f"\n✓ Combined datacard saved to: {output_file}")
    print(f"  Bins: {len(datacard_infos)}")
    print(f"  Processes: {len(set().union(*[info['processes'] for info in datacard_infos]))}")

    # Try to create combined shapes file
    shapes_files = []
    bin_names_list = []
    for datacard_file, region_name in datacards:
        shapes_file = datacard_file.parent / "shapes.root"
        if shapes_file.exists():
            shapes_files.append(str(shapes_file))
            bin_names_list.append(datacard_infos[datacards.index((datacard_file, region_name))]["bin_name"])

    if shapes_files:
        print(f"\n  Creating combined shapes file...")
        combined_shapes_file = plots_dir / "combined_shapes.root"
        success = combine_shapes_files(shapes_files, bin_names_list, str(combined_shapes_file))
        if success:
            print(f"  ✓ Combined shapes file saved to: {combined_shapes_file}")
        else:
            print(f"  ⚠ Could not create combined shapes file")
            print(f"    You may need to create it manually or ensure individual")
            print(f"    shapes.root files are accessible for each bin.")

    print("\n" + "="*70)
    print("✓ Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
