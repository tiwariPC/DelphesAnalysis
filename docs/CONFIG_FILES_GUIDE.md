# Configuration Files Guide

This guide explains the structure and purpose of each configuration file in the analysis.

## File Organization

The configuration is split into logical files to avoid duplication:

1. **File paths** → `samples_config.yaml`
2. **Background cross-sections** → `background_cross_sections.yaml`
3. **Signal cross-sections** → `signal_cross_sections.yaml`
4. **Cuts and regions** → `cuts_config.yaml`

## Configuration Files

### 1. `config/samples_config.yaml`

**Purpose**: Contains file paths for background ROOT files only.

**Structure**:
```yaml
backgrounds:
  diboson:
    file: /path/to/diboson_delphes_events.root
  ttbar:
    file: /path/to/ttbar_delphes_events.root
  # ... etc
```

**Note**: Cross-sections are **NOT** stored here. They are in `background_cross_sections.yaml`.

### 2. `config/background_cross_sections.yaml`

**Purpose**: Single source of truth for all background cross-sections.

**Structure**:
```yaml
backgrounds:
  dyjets: 340.160
  ttbar: 559.385
  diboson: 104.530
  sTop_tchannel: 201.055
  sTop_tW: 61.521
  wlnjets: 3486.150
  znnjets: 1007.100
```

**Units**: picobarns (pb)

### 3. `config/signal_cross_sections.yaml`

**Purpose**: Contains cross-sections for all signal points.

**Structure**:
```yaml
signals:
  300:
    50: 0.0002870400
    100: 0.0001964800
    200: 0.0000933470
  500:
    50: 0.0002886500
    100: 0.0001970500
    # ... etc
```

**Format**: Hierarchical structure by mA (mass A) and ma (mass a) values.
**Units**: picobarns (pb)

### 4. `config/cuts_config.yaml`

**Purpose**: Defines selection cuts and analysis regions.

**Structure**:
```yaml
jets:
  pt_min: 30.0
  eta_max: 2.4

bjets:
  btag_threshold: 1

regions:
  "1b:SR":
    description: "Signal region, 1 b-tag category"
    cuts:
      Nbjets: "==1"
      Njets: "<=2"
      MET: ">250"
  # ... etc
```

## Why This Organization?

### Separation of Concerns

- **File paths** change when you move files or use different directories
- **Cross-sections** are physics constants that don't change with file locations
- **Cuts** are analysis-specific and independent of samples

### Single Source of Truth

- Cross-sections are stored in **one place only** (`background_cross_sections.yaml`)
- No duplication means no confusion about which values are correct
- Easy to update: change cross-sections in one file, affects entire analysis

### Flexibility

- Can easily swap file paths without touching cross-sections
- Can update cross-sections without changing file paths
- Can share cross-section files across different analyses

## Updating Configuration

### Update Background File Paths

Edit `config/samples_config.yaml`:
```yaml
backgrounds:
  ttbar:
    file: /new/path/to/ttbar_delphes_events.root
```

### Update Background Cross-Sections

Edit `config/background_cross_sections.yaml`:
```yaml
backgrounds:
  ttbar: 600.0  # Updated cross-section
```

### Update Signal Cross-Sections

Edit `config/signal_cross_sections.yaml`:
```yaml
signals:
  300:
    50: 0.0003000000  # Updated cross-section
```

### Add New Background

1. Add file path to `samples_config.yaml`
2. Add cross-section to `background_cross_sections.yaml`

## Migration from Old Format

If you have an old `samples_config.yaml` with cross-sections:

1. Extract cross-sections to `background_cross_sections.yaml`
2. Remove `cross_section` fields from `samples_config.yaml`
3. Keep only `file` paths in `samples_config.yaml`

The code will automatically use `background_cross_sections.yaml` for cross-sections.
