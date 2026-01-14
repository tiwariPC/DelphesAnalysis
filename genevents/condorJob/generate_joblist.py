#!/usr/bin/env python3
"""
Generate signal.joblist file with random seeds for each ma value.
For each ma in [50, 100, 200, 300, 400, 500], generates 10 random seeds.
"""

import random

# Define ma values
ma_values = [50, 100, 200, 300, 400, 500]

# Number of random seeds per ma
seeds_per_ma = 10

# Output file
output_file = "signal.joblist"

# Generate random seeds and write to file
with open(output_file, 'w') as f:
    for ma in ma_values:
        for _ in range(seeds_per_ma):
            # Generate a random seed (using a reasonable range)
            random_seed = random.randint(10000, 999999)
            f.write(f"{ma} {random_seed}\n")

print(f"Generated {output_file} with {len(ma_values) * seeds_per_ma} entries")
print(f"Each ma value ({ma_values}) has {seeds_per_ma} random seeds")

