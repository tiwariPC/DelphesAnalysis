#!/usr/bin/env python3
import random
import argparse

massA = [300, 500, 700, 900, 1200, 1500]
massa = [50, 100, 200, 300, 400, 500, 600, 800, 1000, 1200]

backgrounds = [
    "dyjets",
    "ttbar",
    "diboson",
    "sTop_tchannel",
    "sTop_tW",
    "wlnjets",
    "znnjets"
]

def make_signal(nseeds):
    jobid = 0
    with open("signal.joblist", "w") as f:
        for mA in massA:
            for ma in massa:
                if ma < mA:
                    for _ in range(nseeds):
                        seed = random.randint(100000, 999999)
                        f.write(f"{mA} {ma} {seed}\n")
                        jobid += 1
    print(f"[✔] Wrote signal.joblist with {jobid} jobs ({nseeds} seeds per point)")


def make_background(nseeds):
    jobid = 0
    with open("background.joblist", "w") as f:
        for bkg in backgrounds:
            for _ in range(nseeds):
                seed = random.randint(100000, 999999)
                f.write(f"{bkg} {seed}\n")
                jobid += 1
    print(f"[✔] Wrote background.joblist with {jobid} jobs ({nseeds} seeds per process)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MadGraph joblists")
    parser.add_argument("type", choices=["signal", "background"],
                        help="Choose which joblist to make")
    parser.add_argument("-n", "--nseeds", type=int, default=20,
                        help="Number of seeds per point (default: 20)")
    args = parser.parse_args()

    if args.type == "signal":
        make_signal(args.nseeds)
    else:
        make_background(args.nseeds)