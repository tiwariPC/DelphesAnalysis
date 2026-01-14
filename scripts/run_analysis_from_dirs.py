#!/usr/bin/env python3
"""
Automatically discover and process signal and background files from directories.

Signal files: DelphesSignal/sig_bbdm_mA{mA}_ma{ma}_delphes_events.root
Background files: DelphesBackground/{bgname}_delphes_events.root
"""

import sys
import argparse
import re
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Background names expected in the directory
EXPECTED_BACKGROUNDS = [
    "dyjets",
    "ttbar",
    "diboson",
    "sTop_tchannel",
    "sTop_tW",
    "wlnjets",
    "znnjets"
]


def parse_signal_filename(filename: str) -> Optional[Tuple[int, int]]:
    """
    Parse signal filename to extract mA and ma values.

    Expected format: sig_bbdm_mA{mA}_ma{ma}_delphes_events.root
    Example: sig_bbdm_mA1200_ma1000_delphes_events.root -> (1200, 1000)

    Returns:
        Tuple of (mA, ma) or None if parsing fails
    """
    # Pattern: sig_bbdm_mA{number}_ma{number}_delphes_events.root
    pattern = r'sig_bbdm_mA(\d+)_ma(\d+)_delphes_events\.root'
    match = re.search(pattern, filename)
    if match:
        mA = int(match.group(1))
        ma = int(match.group(2))
        return (mA, ma)
    return None


def discover_signal_files(signal_dir: Path) -> List[Tuple[Path, int, int]]:
    """
    Discover all signal files in the directory and extract their (mA, ma) values.

    Returns:
        List of (file_path, mA, ma) tuples, sorted by mA then ma
    """
    signal_files = []

    if not signal_dir.exists():
        print(f"  ✗ Error: Signal directory does not exist: {signal_dir}")
        return signal_files

    for file_path in signal_dir.glob("*.root"):
        filename = file_path.name
        parsed = parse_signal_filename(filename)
        if parsed:
            mA, ma = parsed
            signal_files.append((file_path, mA, ma))
            print(f"  ✓ Found signal: {filename} -> mA={mA}, ma={ma}")
        else:
            print(f"  ⚠ Warning: Could not parse signal filename: {filename}")

    # Sort by mA, then ma
    signal_files.sort(key=lambda x: (x[1], x[2]))
    return signal_files


def discover_background_files(background_dir: Path) -> Dict[str, Path]:
    """
    Discover background files in the directory.

    Expected format: {bgname}_delphes_events.root

    Returns:
        Dictionary mapping background name to file path
    """
    background_files = {}

    if not background_dir.exists():
        print(f"  ✗ Error: Background directory does not exist: {background_dir}")
        return background_files

    for bg_name in EXPECTED_BACKGROUNDS:
        # Try different possible filename patterns
        patterns = [
            f"{bg_name}_delphes_events.root",
            f"bkg_{bg_name}_delphes_events.root",
            f"{bg_name}.root"
        ]

        found = False
        for pattern in patterns:
            file_path = background_dir / pattern
            if file_path.exists():
                background_files[bg_name] = file_path
                print(f"  ✓ Found background: {bg_name} -> {file_path.name}")
                found = True
                break

        if not found:
            print(f"  ⚠ Warning: Background file not found for: {bg_name}")

    return background_files


def count_events_in_root_file(root_file: Path) -> Optional[int]:
    """Count number of events in a ROOT file using uproot."""
    try:
        import uproot
        with uproot.open(str(root_file)) as file:
            tree = file.get("Delphes")
            if tree is None:
                # Try to find any tree
                keys = list(file.keys())
                if keys:
                    tree = file[keys[0]]
                else:
                    return None
            return tree.num_entries
    except Exception as e:
        print(f"  ⚠ Warning: Could not count events in {root_file.name}: {e}")
        return None


def build_process_regions_command(
    signal_files: List[Tuple[Path, int, int]],
    background_files: Dict[str, Path],
    signal_ngen: Optional[int] = None,
    lumi: float = 139.0,
    output_dir: str = "output",
    signal_xsec_file: str = "config/signal_cross_sections.yaml",
    background_xsec_file: str = "config/background_cross_sections.yaml",
    cuts_config: str = "config/cuts_config.yaml",
    auto_ngen: bool = True
) -> List[str]:
    """
    Build the command to run process_regions.py with discovered files.

    Returns:
        List of command arguments
    """
    cmd = ["python", "scripts/process_regions.py"]

    # Add all signal files
    signals_to_add = []
    for file_path, mA, ma in signal_files:
        ngen = None
        if signal_ngen is not None:
            ngen = signal_ngen
        elif auto_ngen:
            # Try to count events automatically
            ngen = count_events_in_root_file(file_path)
            if not ngen:
                print(f"  ⚠ Warning: Could not determine ngen for {file_path.name}, skipping")
                continue

        if ngen is None:
            print(f"  ⚠ Warning: No ngen specified for {file_path.name}, skipping")
            continue

        signals_to_add.append((file_path, mA, ma, ngen))

    if not signals_to_add:
        print("  ✗ Error: No valid signal files found (all missing ngen)")
        return []

    # Add all valid signals to command
    for file_path, mA, ma, ngen in signals_to_add:
        cmd.extend(["--signal-file", str(file_path)])
        cmd.extend(["--signal-mA", str(mA)])
        cmd.extend(["--signal-ma", str(ma)])
        cmd.extend(["--signal-ngen", str(ngen)])

    # Add other options
    cmd.extend(["--lumi", str(lumi)])
    cmd.extend(["--output-dir", output_dir])
    cmd.extend(["--signal-xsec", signal_xsec_file])
    cmd.extend(["--background-xsec", background_xsec_file])
    cmd.extend(["--cuts-config", cuts_config])

    # Note: Background files are loaded from samples_config.yaml
    # We'll update that file or create a temporary one

    return cmd


def update_samples_config(background_files: Dict[str, Path], config_file: str = "config/samples_config.yaml"):
    """
    Update samples_config.yaml with discovered background file paths.
    Note: Cross-sections are NOT stored here - they are in background_cross_sections.yaml
    """
    import yaml

    config_path = Path(config_file)

    # Load existing config or create new one
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    if 'backgrounds' not in config:
        config['backgrounds'] = {}

    # Update file paths for discovered backgrounds (only file paths, no cross-sections)
    for bg_name, file_path in background_files.items():
        if bg_name not in config['backgrounds']:
            config['backgrounds'][bg_name] = {}
        config['backgrounds'][bg_name]['file'] = str(file_path.absolute())
        # Remove cross_section if present (should be in background_cross_sections.yaml)
        if 'cross_section' in config['backgrounds'][bg_name]:
            del config['backgrounds'][bg_name]['cross_section']

    # Write updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  ✓ Updated {config_file} with background file paths")
    print(f"    Note: Cross-sections are in background_cross_sections.yaml")


def main():
    parser = argparse.ArgumentParser(
        description="Automatically discover and process signal and background files from directories"
    )
    parser.add_argument("--signal-dir", type=str, required=True,
                       help="Directory containing signal ROOT files")
    parser.add_argument("--background-dir", type=str, required=True,
                       help="Directory containing background ROOT files")
    parser.add_argument("--signal-ngen", type=int, default=None,
                       help="Number of generated events for all signals (if not provided, will try to count from files)")
    parser.add_argument("--lumi", type=float, default=139.0,
                       help="Luminosity in fb^-1 (default: 139.0)")
    parser.add_argument("--output-dir", type=str, default="output",
                       help="Output directory (default: output)")
    parser.add_argument("--samples-config", type=str, default="config/samples_config.yaml",
                       help="Samples configuration file to update (default: config/samples_config.yaml)")
    parser.add_argument("--signal-xsec", type=str, default="config/signal_cross_sections.yaml",
                       help="Signal cross-sections YAML file")
    parser.add_argument("--background-xsec", type=str, default="config/background_cross_sections.yaml",
                       help="Background cross-sections YAML file")
    parser.add_argument("--cuts-config", type=str, default="config/cuts_config.yaml",
                       help="Cuts configuration file")
    parser.add_argument("--dry-run", action="store_true",
                       help="Only print the command, don't run it")
    parser.add_argument("--no-auto-ngen", action="store_true",
                       help="Don't automatically count events from ROOT files")

    args = parser.parse_args()

    signal_dir = Path(args.signal_dir)
    background_dir = Path(args.background_dir)

    print("=" * 70)
    print("Discovering Signal and Background Files")
    print("=" * 70)

    # Discover signal files
    print(f"\nScanning signal directory: {signal_dir}")
    signal_files = discover_signal_files(signal_dir)

    if not signal_files:
        print("  ✗ Error: No signal files found!")
        return 1

    print(f"\n  Found {len(signal_files)} signal file(s)")

    # Discover background files
    print(f"\nScanning background directory: {background_dir}")
    background_files = discover_background_files(background_dir)

    if not background_files:
        print("  ✗ Error: No background files found!")
        return 1

    print(f"\n  Found {len(background_files)} background file(s)")

    # Update samples config with background file paths
    print(f"\nUpdating {args.samples_config}...")
    update_samples_config(background_files, args.samples_config)

    # Build command
    print(f"\nBuilding command...")
    cmd = build_process_regions_command(
        signal_files=signal_files,
        background_files=background_files,
        signal_ngen=args.signal_ngen,
        lumi=args.lumi,
        output_dir=args.output_dir,
        signal_xsec_file=args.signal_xsec,
        background_xsec_file=args.background_xsec,
        cuts_config=args.cuts_config,
        auto_ngen=not args.no_auto_ngen
    )

    # Check if we have any signals
    if not any("--signal-file" in str(arg) for arg in cmd):
        print("  ✗ Error: No valid signal files to process")
        return 1

    # Print command
    print("\n" + "=" * 70)
    print("Command to run:")
    print("=" * 70)
    print(" ".join(cmd))
    print("=" * 70)

    if args.dry_run:
        print("\n  (Dry run - not executing)")
        return 0

    # Run command
    print("\nRunning analysis...")
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✓ Analysis complete!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Analysis failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n✗ Analysis interrupted by user")
        return 1


if __name__ == "__main__":
    sys.exit(main())
