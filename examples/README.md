# Examples

This directory contains example scripts demonstrating how to use **SeisFMTD** 
for seismic full moment tensor inversion.

## Quick Start

```bash
cd examples
python run_hmc_mt_example.py       # Moment tensor only
python run_hmc_mtd_example.py   # Moment tensor + Depth
```

## Example Files

### `run_hmc_example.py` - Moment Tensor Inversion

Uses `DHMC_MT` solver to invert for **6 parameters** (moment tensor only):
- Pre-computed SGT and waveform data as input
- Suitable when depth is well-constrained

### `run_hmc_mtd_example.py` - Moment Tensor + Depth Inversion

Uses `DHMC_MTD` solver to invert for **7 parameters** (moment tensor + depth):
- Requires SGT manager for real-time depth interpolation
- Suitable when depth is poorly constrained
- Outputs depth uncertainty from posterior samples

1. **Data structure requirements** - What format your SGT and waveform data should be in
2. **Required parameters** - All parameters needed to initialize the solver
3. **HMC configuration** - How to set up the sampler
4. **Results analysis** - How to interpret the output

**Before running**, you need to:
1. Implement `load_sgt_data()` to load your SGT data
2. Implement `load_observed_data()` to load your waveforms
3. Update source and station locations

## Data Requirements

### SGT Data (Strain Green's Tensors)

For each station, you need:
```python
pnl_sgt  # shape: (n_samples, 3, 6) - P-wave SGT
srf_sgt  # shape: (n_samples, 3, 6) - S-wave SGT
```
- First dimension: time samples
- Second dimension: 3 components (N, E, Z)
- Third dimension: 6 moment tensor elements

### Observed Waveforms

For each station, you need:
```python
pnl_data  # shape: (3, n_samples) - P-wave window
srf_data  # shape: (3, n_samples) - S-wave window
```
- First dimension: 3 components (E, N, Z)
- Second dimension: time samples

## Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `dt` | Sampling interval (s) | 0.5 |
| `n_tp_max_shift` | Max P-wave shift (samples) | 2-10 |
| `n_ts_max_shift` | Max S-wave shift (samples) | 5-20 |
| `w_pnl` | P-wave weight | 0.5-1.0 |
| `w_srf` | S-wave weight | 1.0 |
| `n_samples` | Number of HMC samples | 500-2000 |

## Output

Results are saved to your specified output directory:
- `Samples_N{n}_*.pkl` - Posterior samples
- Console output shows mean solution and statistics
