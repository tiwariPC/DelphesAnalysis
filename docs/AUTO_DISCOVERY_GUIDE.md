# Automatic File Discovery Guide

This guide explains how to use the automatic file discovery script to process signals and backgrounds from directories.

## File Naming Conventions

The script expects files to follow these naming patterns:

### Signal Files
- **Format**: `sig_bbdm_mA{mA}_ma{ma}_delphes_events.root`
- **Example**: `sig_bbdm_mA1200_ma1000_delphes_events.root`
- **Location**: All signal files should be in one directory (e.g., `DelphesSignal/`)

### Background Files
- **Format**: `{bgname}_delphes_events.root` or `bkg_{bgname}_delphes_events.root`
- **Examples**:
  - `diboson_delphes_events.root`
  - `ttbar_delphes_events.root`
  - `wlnjets_delphes_events.root`
- **Location**: All background files should be in one directory (e.g., `DelphesBackground/`)

## Usage

### Basic Usage

```bash
python scripts/run_analysis_from_dirs.py \
    --signal-dir DelphesSignal \
    --background-dir DelphesBackground \
    --output-dir output
```

This will:
1. Scan `DelphesSignal/` for all signal files matching the pattern
2. Extract (mA, ma) values from filenames
3. Scan `DelphesBackground/` for background files
4. Count events in signal files automatically (or use `--signal-ngen`)
5. Update `config/samples_config.yaml` with background file paths
6. Run the analysis with all signals overlaid on the same plots

### Specify Number of Generated Events

If you want to specify the same number of generated events for all signals:

```bash
python scripts/run_analysis_from_dirs.py \
    --signal-dir DelphesSignal \
    --background-dir DelphesBackground \
    --signal-ngen 100000 \
    --output-dir output
```

### Dry Run (Preview Command)

To see what command would be run without executing it:

```bash
python scripts/run_analysis_from_dirs.py \
    --signal-dir DelphesSignal \
    --background-dir DelphesBackground \
    --dry-run
```

### Disable Automatic Event Counting

If you don't want the script to count events from ROOT files:

```bash
python scripts/run_analysis_from_dirs.py \
    --signal-dir DelphesSignal \
    --background-dir DelphesBackground \
    --signal-ngen 100000 \
    --no-auto-ngen \
    --output-dir output
```

## Expected Background Files

The script looks for these background processes:
- `dyjets`
- `ttbar`
- `diboson`
- `sTop_tchannel`
- `sTop_tW`
- `wlnjets`
- `znnjets`

## Output

- **Plots**: All signals will be overlaid on the same plots in `output/plots/`
- **Datacards**: Generated for each region (uses first signal for Combine compatibility)
- **ROOT files**: Histograms saved for each observable

## Example Directory Structure

```
DelphesSignal/
├── sig_bbdm_mA300_ma50_delphes_events.root
├── sig_bbdm_mA300_ma100_delphes_events.root
├── sig_bbdm_mA500_ma200_delphes_events.root
└── ...

DelphesBackground/
├── diboson_delphes_events.root
├── ttbar_delphes_events.root
├── wlnjets_delphes_events.root
└── ...
```

## Troubleshooting

### No signal files found
- Check that signal files follow the naming pattern: `sig_bbdm_mA{mA}_ma{ma}_delphes_events.root`
- Verify the `--signal-dir` path is correct

### No background files found
- Check that background files follow the naming pattern: `{bgname}_delphes_events.root`
- Verify the `--background-dir` path is correct
- The script looks for exact matches of background names (dyjets, ttbar, etc.)

### Could not determine ngen
- Provide `--signal-ngen` to specify the number of generated events
- Or ensure ROOT files are readable and contain a "Delphes" tree

### Cross-section not found
- Ensure `config/signal_cross_sections.yaml` contains entries for all (mA, ma) combinations
- Or use `--signal-xs` manually with `process_regions.py` instead

## Advanced: Custom Configuration Files

You can specify custom configuration files:

```bash
python scripts/run_analysis_from_dirs.py \
    --signal-dir DelphesSignal \
    --background-dir DelphesBackground \
    --samples-config config/my_samples.yaml \
    --signal-xsec config/my_signal_xsec.yaml \
    --background-xsec config/my_background_xsec.yaml \
    --cuts-config config/my_cuts.yaml \
    --output-dir output
```
