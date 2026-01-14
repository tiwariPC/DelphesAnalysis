# Event Generation Instructions

This directory contains scripts and configuration files for generating Monte Carlo events using MadGraph5.

## Directory Structure

- `genScripts/`: Contains MadGraph5 launch cards (`.mg5` files) for different processes
- `condorJob/`: Contains Condor job submission scripts and job lists

## Step 1: Generate MadGraph Processes from genlaunchCards

### For Backgrounds:

Generate each background process using `.mg5` files from `../genlaunchCards`:

```bash
cd /afs/cern.ch/work/p/ptiwari/madgraph/MG5_aMC_v3_5_12
./bin/mg5_aMC ../genlaunchCards/dyjets.mg5
./bin/mg5_aMC ../genlaunchCards/ttbar.mg5
./bin/mg5_aMC ../genlaunchCards/diboson.mg5
./bin/mg5_aMC ../genlaunchCards/sTop_tchannel.mg5
./bin/mg5_aMC ../genlaunchCards/sTop_tW.mg5
./bin/mg5_aMC ../genlaunchCards/wlnjets.mg5
./bin/mg5_aMC ../genlaunchCards/znnjets.mg5
```

Create tar.gz archives for each background:

```bash
tar -czf dyjets.tar.gz dyjets/
tar -czf ttbar.tar.gz ttbar/
tar -czf diboson.tar.gz diboson/
tar -czf sTop_tchannel.tar.gz sTop_tchannel/
tar -czf sTop_tW.tar.gz sTop_tW/
tar -czf wlnjets.tar.gz wlnjets/
tar -czf znnjets.tar.gz znnjets/
```

### For Signal:

Generate the signal process:

```bash
./bin/mg5_aMC ../genlaunchCards/signal.mg5
```

Create tar.gz archive:

```bash
tar -czf bbdm_2hdma_type1_case1.tar.gz bbdm_2hdma_type1_case1/
```

## Step 2: Prepare Launch Cards

Copy launch card templates from `../genlaunchCards` to `condorJob/`:

```bash
cd condorJob
cp ../genlaunchCards/background.madevent .
cp ../genlaunchCards/signal.madevent .
```

## Step 3: Generate Job Lists

```bash
cd condorJob
python3 make_joblist.py background -n 20
python3 make_joblist.py signal -n 10
```

## Step 4: Submit Condor Jobs

```bash
condor_submit background.submit
condor_submit signal.submit
```

## Step 5: Extract Cross-Sections (after jobs complete)

```bash
python3 extract_crosssections.py
python3 extract_signal_crosssections.py
```
