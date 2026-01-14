#!/usr/bin/env python

import os
import re
import subprocess
from collections import defaultdict

input_dir = "/eos/cms/store/group/phys_susy/sus-23-008/DelphesBackgroundFiles"
output_dir = "/eos/cms/store/group/phys_susy/sus-23-008/DelphesBackground"

# Make dirs
os.makedirs(output_dir, exist_ok=True)

# Background samples you expect
bkg_processes = [
    "wlnjets",
    "znnjets",
    "ttbar",
    "dyjets",
    "diboson",
    "sTop_tchannel",
    "sTop_tW"
]

# Build dict to collect files per bkg name
files_by_bkg = defaultdict(list)

# Loop over files and classify
for f in os.listdir(input_dir):
    for tag in bkg_processes:
        if f.startswith(tag) and f.endswith("_delphes_events.root"):
            files_by_bkg[tag].append(os.path.join(input_dir, f))
            break

# Merge each sample
for bkg, files in files_by_bkg.items():
    outfile = os.path.join(output_dir, f"{bkg}_delphes_events.root")

    print(f"\n>> Merging {bkg}")
    print(f"   Files found: {len(files)}")

    if len(files) == 0:
        print("   !! No files found, skipping")
        continue

    subprocess.run(["hadd", "-f", outfile] + files)
    print(f"   ✔ Output written to {outfile}")

print("\nDone!")