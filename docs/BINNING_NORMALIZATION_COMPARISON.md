# Binning and Normalization Comparison

This document compares the binning and normalization logic between the reference `StackPlotter_addMeanWeight.py` and our implementation.

## Key Findings

### 1. Normalization Formula

**StackPlotter:**
```python
total_events = h_total_mcweight.Integral()  # Sum of MC weights
normlisation = (xsec * luminosity) / total_events
h_temp.Scale(normlisation)
```
- `luminosity` is in **pb** (e.g., 35.90 * 1000 = 35900 pb for 2016)
- `total_events` is the **integral of MC weights** (`h_total_mcweight.Integral()`)
- This gives: `(xsec_pb * luminosity_pb) / sum_of_weights`

**Our Implementation:**
```python
weight = (xs_pb * lumi_fb * 1000.0) / ngen
```
- `lumi_fb * 1000` converts fb^-1 to pb (same as StackPlotter's `luminosity`)
- `ngen` is the **raw number of generated events** (from `count_events_in_root_file()`)
- This gives: `(xsec_pb * luminosity_pb) / n_events`

**Difference:**
- StackPlotter uses **sum of MC weights** (weighted integral)
- We use **raw event count**
- If events have generator weights ≠ 1, these will differ

### 2. Binning

**Both implementations use the same binning:**

**For MET/Recoil (1b regions):**
```python
bins = [250, 300, 400, 550, 1000]  # 4 bins
```

**For cost_star (2b regions):**
```python
bins = [0.0, 0.25, 0.50, 0.75, 1.0]  # 4 bins
```

✅ **Binning matches exactly**

### 3. Overflow Handling

**StackPlotter:**
```python
def set_overflow(hist):
    bin_num = hist.GetXaxis().GetNbins()
    # Add overflow bin content to last bin
    hist.SetBinContent(bin_num, hist.GetBinContent(bin_num+1) + hist.GetBinContent(bin_num))
    hist.SetBinContent(bin_num+1, 0.)
    return hist
```
- Handles overflow **after** histogram is filled
- Adds overflow bin content to the last regular bin

**Our Implementation:**
```python
# Clip overflow values BEFORE filling
overflow_mask = data_to_fill >= last_edge
if np.sum(overflow_mask) > 0:
    data_to_fill[overflow_mask] = last_edge - 1e-6
# Then fill histogram
histograms[obs_name].fill(**{obs_name: data_to_fill}, weight=weights_to_fill)
```
- Handles overflow **before** filling
- Clips values to put them in the last bin

**Difference:**
- StackPlotter: Post-processing approach (adds overflow to last bin)
- Our code: Pre-processing approach (clips values before filling)
- Both achieve the same result, but our approach is more efficient

### 4. Signal Normalization

**StackPlotter:**
```python
total = signal_files[key].Get('h_total_mcweight')
sig_hist[key].Scale(luminosity * sig_sample_xsec.getSigXsec_official(key) / total.Integral())
```
- Uses `h_total_mcweight.Integral()` for normalization
- Same pattern as backgrounds

**Our Implementation:**
```python
weight = (xs_pb * lumi_fb * 1000.0) / ngen
```
- Uses raw event count `ngen`
- Applied as weight during histogram filling

## Implementation Update

### Normalization Using Total Weight Histogram

**StackPlotter approach:**
```python
h_total_weight = f.Get('h_total_mcweight')
total_events = h_total_weight.Integral()  # Sum of MC weights
normlisation = (xsec * luminosity) / total_events
h_temp.Scale(normlisation)
```

**Our updated implementation:**
```python
# First try to get total weight from histogram (matches StackPlotter)
total_weight = get_total_weight_from_root_file(root_file, "h_total_mcweight")
if total_weight is None:
    total_weight = count_events_in_root_file(root_file)  # Fall back to event count

# Normalization (same formula)
weight = (xs_pb * lumi_fb * 1000.0) / total_weight
```

**Key points:**
- ✅ Checks for `h_total_mcweight` histogram first (matches StackPlotter)
- ✅ Falls back to raw event count if histogram not found
- ✅ Uses same normalization formula: `(xsec * luminosity) / total_weight`
- ✅ Works with both weighted and unweighted samples

## Action Items

1. ✅ **Binning**: Already matches StackPlotter
2. ✅ **Overflow handling**: Functionally equivalent (different approach, same result)
3. ✅ **Normalization**: Updated to use `h_total_mcweight` histogram if available
   - Checks for histogram first (matches StackPlotter)
   - Falls back to event count if histogram not found
   - Same normalization formula: `(xsec * luminosity) / total_weight`

## Verification

To verify normalization is correct:
1. Check if Delphes ROOT files contain generator weights
2. Compare yields between StackPlotter and our code for the same sample
3. If yields differ significantly, switch to weighted normalization
