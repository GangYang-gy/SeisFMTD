"""
Test DFilters optimization using real SAC waveforms
Compare results before and after optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from obspy.core.trace import Trace

# ==================== Original Implementation (ObsPy) ====================

def DFilter_syn_original(data, freqmin, freqmax):
    """Original version - using ObsPy Trace"""
    n_dim = data.shape[0] if data.ndim > 1 else 1
    if data.ndim == 1:
        data = data.reshape(1, -1)
    new_data = np.zeros_like(data)
    for i in range(n_dim):
        tr = Trace(data=data[i].copy())
        tr.stats.delta = 0.5
        tr.detrend('demean')
        tr.detrend('linear')
        tr.taper(max_percentage=0.05, type="hann")
        tr.filter('bandpass', freqmin=freqmin, freqmax=freqmax, zerophase=True)
        new_data[i] = tr.data
    return new_data.squeeze()


# ==================== Optimized Implementation ====================
from DFilters import DFilter_syn

# ==================== Main Test ====================

def test_with_real_sac():
    """Run comparison test using real SAC waveforms"""
    
    # Read SAC file
    sac_path = "./BCOK.XHE.sac"
    st = read(sac_path)
    tr = st[0]
    
    print("=" * 60)
    print(f"SAC File: {sac_path.split('/')[-1]}")
    print(f"Sampling Rate: {tr.stats.sampling_rate} Hz")
    print(f"Delta: {tr.stats.delta} s")
    print(f"N Samples: {tr.stats.npts}")
    print("=" * 60)
    
    # Prepare data (simulate 3 components)
    data = np.array([tr.data.copy(), tr.data.copy(), tr.data.copy()])
    
    # Filter parameters
    freqmin, freqmax = 1/30, 1/5
    
    # Run both versions
    result_orig = DFilter_syn_original(data, freqmin, freqmax)
    result_new = DFilter_syn(data, freqmin, freqmax)
    
    # Calculate difference
    diff = result_orig[0] - result_new[0]
    max_abs_diff = np.max(np.abs(diff))
    max_rel_diff = max_abs_diff / (np.max(np.abs(result_orig[0])) + 1e-10)
    
    print(f"\nMax Absolute Difference: {max_abs_diff:.2e}")
    print(f"Max Relative Difference: {max_rel_diff:.2e}")
    print(f"{'✅ PASS' if max_rel_diff < 1e-6 else '⚠️  Difference detected'}")
    
    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    t = np.arange(len(result_orig[0])) * 0.5  # time in seconds
    
    # Raw data
    ax = axes[0]
    ax.plot(t, tr.data, 'k-', linewidth=0.5)
    ax.set_title(f'Raw SAC Data: {sac_path.split("/")[-1]}')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    
    # Filtered waveform comparison (global)
    ax = axes[1]
    ax.plot(t, result_orig[0], 'b-', label='Original (ObsPy)', alpha=0.7, linewidth=1)
    ax.plot(t, result_new[0], 'r--', label='Optimized (SciPy)', alpha=0.7, linewidth=1)
    ax.set_title(f'Filtered Waveform Comparison (bandpass {round(freqmin, 3)}-{round(freqmax, 3)} Hz)')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Difference
    ax = axes[2]
    ax.plot(t, diff, 'g-', linewidth=0.5)
    ax.set_title(f'Difference (max abs = {max_abs_diff:.2e})')
    ax.set_ylabel('Diff')
    ax.grid(True, alpha=0.3)
    
    # Zoom in (waveform start)
    ax = axes[3]
    n_zoom = min(100, len(result_orig[0]))
    ax.plot(t[:n_zoom], result_orig[0, :n_zoom], 'b-', label='Original', linewidth=1.5)
    ax.plot(t[:n_zoom], result_new[0, :n_zoom], 'r--', label='Optimized', linewidth=1.5)
    ax.set_title(f'Zoom: First {n_zoom} samples')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = 'dfilters_real_sac_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {output_file}")
    plt.close()


if __name__ == "__main__":
    test_with_real_sac()

