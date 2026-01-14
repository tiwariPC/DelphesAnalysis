#!/usr/bin/env python3

import os
import re
import subprocess
from collections import defaultdict

input_dir = "/eos/cms/store/group/phys_susy/sus-23-008/DelphesSignalFiles"
output_dir = "/eos/cms/store/group/phys_susy/sus-23-008/DelphesSignal"

# Create output dir if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Regex to extract masses
pattern = re.compile(r"sig_bbdm_mA(\d+)_ma(\d+)_.*_delphes_events\.root$")

# Keep dictionary of lists
files_by_mass = defaultdict(list)

# Scan for files
for f in os.listdir(input_dir):
    match = pattern.match(f)
    if match:
        mA = match.group(1)
        ma = match.group(2)
        files_by_mass[(mA, ma)].append(os.path.join(input_dir, f))

# Loop over all combinations found
for (mA, ma), file_list in files_by_mass.items():
    # Output file name
    outfile = os.path.join(
        output_dir,
        f"sig_bbdm_mA{mA}_ma{ma}_delphes_events.root"
    )

    print(f"\n>> Merging mA={mA}, ma={ma}")
    print(f"   Files found: {len(file_list)}")

    if len(file_list) == 0:
        print("   !! No files found, skipping")
        continue

    # Run hadd
    subprocess.run(["hadd", "-f", outfile] + file_list)
    print(f"   ✔ Output written to {outfile}")

print("\nDone!")
