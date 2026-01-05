#!/usr/bin/env python3
"""
Example 1: 1D Drift + Noise on Sine Wave

Demonstrates BioMatrix on a simple 1D signal:
- y = sin(2πx) with drift d(x) = 0.2x + noise

Shows that Procrustes removes drift and recovers the clean signal.
"""

import numpy as np
import matplotlib.pyplot as plt

# Add biomatrix to path
import sys
sys.path.insert(0, '.')

from biomatrix.core.state import State
from biomatrix.core.derive.procrustes import derive_procrustes_se3


def main():
    np.random.seed(42)
    
    # Generate signal
    n_points = 100
    x = np.linspace(0, 1, n_points)
    
    # Clean signal: y = sin(2πx)
    y_clean = np.sin(2 * np.pi * x)
    
    # Add drift: d(x) = 0.2x
    drift = 0.2 * x
    
    # Add noise: σ = 0.05
    noise = np.random.randn(n_points) * 0.05
    
    # Observed signal: y + drift + noise
    y_noisy = y_clean + drift + noise
    
    # Create 2D point clouds
    pts_clean = np.column_stack([x, y_clean])
    pts_noisy = np.column_stack([x, y_noisy])
    
    # Derive transformation: noisy -> clean
    op = derive_procrustes_se3(State(pts_noisy), State(pts_clean))
    
    if op is None:
        print("Procrustes failed - using centroid correction")
        shift = np.mean(pts_clean, axis=0) - np.mean(pts_noisy, axis=0)
        pts_corrected = pts_noisy + shift
    else:
        # Apply correction
        pts_corrected = op.apply(State(pts_noisy)).points
        print(f"Transformation: A = {op.A.diagonal()}, t = {op.t}")
    
    # Extract corrected y values
    y_corrected = pts_corrected[:, 1]
    
    # Compute RMSE
    rmse_before = np.sqrt(np.mean((y_noisy - y_clean) ** 2))
    rmse_after = np.sqrt(np.mean((y_corrected - y_clean) ** 2))
    
    print(f"\n=== RESULTS ===")
    print(f"RMSE before correction: {rmse_before:.4f}")
    print(f"RMSE after correction:  {rmse_after:.4f}")
    print(f"Improvement: {rmse_before / rmse_after:.1f}x")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before
    axes[0].plot(x, y_clean, 'g-', linewidth=2, label='True: sin(2πx)')
    axes[0].plot(x, y_noisy, 'r.', alpha=0.5, label='Observed (drift + noise)')
    axes[0].set_title(f'Before Correction (RMSE = {rmse_before:.4f})')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # After
    axes[1].plot(x, y_clean, 'g-', linewidth=2, label='True: sin(2πx)')
    axes[1].plot(x, y_corrected, 'b.', alpha=0.5, label='Corrected')
    axes[1].set_title(f'After BioMatrix Correction (RMSE = {rmse_after:.4f})')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_1d_drift.png', dpi=150)
    print("\nPlot saved: example_1d_drift.png")
    plt.show()


if __name__ == "__main__":
    main()
