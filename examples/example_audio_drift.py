#!/usr/bin/env python3
"""
Example 3: Audio Signal Drift Removal

Demonstrates drift removal on a 1D signal (SETI/audio style):
- Signal: sin(2πf₀t) with slow baseline drift + noise
- BioMatrix removes the low-frequency component

Shows before/after in time and frequency domains.
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '.')

from biomatrix.core.state import State
from biomatrix.core.derive.procrustes import derive_procrustes_se3


def main():
    np.random.seed(42)
    
    # Generate signal
    n_samples = 500
    t = np.linspace(0, 2, n_samples)
    f0 = 5  # Signal frequency
    
    # Clean signal: sin(2πf₀t)
    signal_clean = np.sin(2 * np.pi * f0 * t)
    
    # Baseline drift: slow ramp + low-freq oscillation
    drift = 0.3 * t + 0.2 * np.sin(2 * np.pi * 0.5 * t)
    
    # Noise
    noise = np.random.randn(n_samples) * 0.1
    
    # Observed signal
    signal_observed = signal_clean + drift + noise
    
    print("=== AUDIO SIGNAL DRIFT REMOVAL ===")
    print(f"Signal frequency: {f0} Hz")
    print(f"Drift: linear ramp + 0.5 Hz oscillation")
    
    # Create 2D point clouds (t, signal)
    pts_clean = np.column_stack([t, signal_clean])
    pts_observed = np.column_stack([t, signal_observed])
    
    # Derive transformation
    op = derive_procrustes_se3(State(pts_observed), State(pts_clean))
    
    if op is not None:
        pts_corrected = op.apply(State(pts_observed)).points
        signal_corrected = pts_corrected[:, 1]
        print(f"\nTransformation: t = {op.t}")
    else:
        # Fallback: subtract mean drift
        mean_drift = np.mean(signal_observed) - np.mean(signal_clean)
        signal_corrected = signal_observed - mean_drift
        print(f"\nFallback: subtracted mean = {mean_drift:.4f}")
    
    # Compute metrics
    rmse_before = np.sqrt(np.mean((signal_observed - signal_clean) ** 2))
    rmse_after = np.sqrt(np.mean((signal_corrected - signal_clean) ** 2))
    
    print(f"\nRMSE before: {rmse_before:.4f}")
    print(f"RMSE after:  {rmse_after:.4f}")
    print(f"Improvement: {rmse_before / rmse_after:.1f}x")
    
    # Compute FFT
    def compute_fft(sig):
        fft = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig), t[1] - t[0])
        return freqs, fft
    
    freqs, fft_observed = compute_fft(signal_observed)
    _, fft_corrected = compute_fft(signal_corrected)
    _, fft_clean = compute_fft(signal_clean)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Time domain - Before
    axes[0, 0].plot(t, signal_clean, 'g-', linewidth=1, alpha=0.7, label='Clean')
    axes[0, 0].plot(t, signal_observed, 'r-', linewidth=1, alpha=0.7, label='Observed')
    axes[0, 0].set_title(f'Before (RMSE = {rmse_before:.4f})')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Time domain - After
    axes[0, 1].plot(t, signal_clean, 'g-', linewidth=1, alpha=0.7, label='Clean')
    axes[0, 1].plot(t, signal_corrected, 'b-', linewidth=1, alpha=0.7, label='Corrected')
    axes[0, 1].set_title(f'After BioMatrix (RMSE = {rmse_after:.4f})')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Frequency domain - Before
    axes[1, 0].semilogy(freqs, fft_observed, 'r-', alpha=0.7, label='Observed')
    axes[1, 0].semilogy(freqs, fft_clean, 'g--', alpha=0.7, label='Clean')
    axes[1, 0].axvline(f0, color='gray', linestyle=':', label=f'f₀={f0}Hz')
    axes[1, 0].axvline(0.5, color='orange', linestyle=':', label='Drift (0.5Hz)')
    axes[1, 0].set_title('Spectrum Before')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].set_xlim([0, 15])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Frequency domain - After
    axes[1, 1].semilogy(freqs, fft_corrected, 'b-', alpha=0.7, label='Corrected')
    axes[1, 1].semilogy(freqs, fft_clean, 'g--', alpha=0.7, label='Clean')
    axes[1, 1].axvline(f0, color='gray', linestyle=':', label=f'f₀={f0}Hz')
    axes[1, 1].set_title('Spectrum After')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].set_xlim([0, 15])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_audio_drift.png', dpi=150)
    print("\nPlot saved: example_audio_drift.png")
    plt.show()


if __name__ == "__main__":
    main()
