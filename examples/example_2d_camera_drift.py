#!/usr/bin/env python3
"""
Example 2: 2D Camera Drift Simulation

Simulates a field of points (stars, image features) that drifts between frames.
Demonstrates that BioMatrix recovers the exact translation.
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '.')

from biomatrix.core.state import State
from biomatrix.core.derive.procrustes import derive_procrustes_se3


def main():
    np.random.seed(42)
    
    # Generate random "star field" - 100 points in 2D
    n_points = 100
    frame0 = np.random.randn(n_points, 2) * 10
    
    # Known drift: [+2.5, -1.3]
    true_drift = np.array([2.5, -1.3])
    
    # Frame 1 = Frame 0 + drift + noise
    noise_std = 0.1
    noise = np.random.randn(n_points, 2) * noise_std
    frame1 = frame0 + true_drift + noise
    
    # Add outliers (5%)
    n_outliers = int(0.05 * n_points)
    outlier_idx = np.random.choice(n_points, n_outliers, replace=False)
    frame1[outlier_idx] += np.random.randn(n_outliers, 2) * 5  # Large deviation
    
    print("=== 2D CAMERA DRIFT SIMULATION ===")
    print(f"True drift: {true_drift}")
    print(f"Noise σ: {noise_std}")
    print(f"Outliers: {n_outliers}")
    
    # Derive transformation: frame0 -> frame1
    op = derive_procrustes_se3(State(frame0), State(frame1))
    
    if op is not None:
        detected_drift = op.t
        print(f"\nDetected drift: {detected_drift}")
        print(f"Error: {np.linalg.norm(detected_drift - true_drift):.4f}")
        
        # Apply correction (inverse: subtract drift)
        frame1_corrected = frame1 - detected_drift
    else:
        print("Procrustes failed - using centroid")
        detected_drift = np.mean(frame1, axis=0) - np.mean(frame0, axis=0)
        frame1_corrected = frame1 - detected_drift
    
    # Compute alignment error
    residuals_before = np.linalg.norm(frame1 - frame0, axis=1)
    residuals_after = np.linalg.norm(frame1_corrected - frame0, axis=1)
    
    print(f"\nMean residual before: {np.mean(residuals_before):.4f}")
    print(f"Mean residual after:  {np.mean(residuals_after):.4f}")
    print(f"Improvement: {np.mean(residuals_before) / np.mean(residuals_after):.1f}x")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Frame 0 (reference)
    axes[0].scatter(frame0[:, 0], frame0[:, 1], c='blue', alpha=0.6, label='Frame 0')
    axes[0].set_title('Frame 0 (Reference)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)
    
    # Before correction
    axes[1].scatter(frame0[:, 0], frame0[:, 1], c='blue', alpha=0.4, label='Frame 0')
    axes[1].scatter(frame1[:, 0], frame1[:, 1], c='red', alpha=0.4, label='Frame 1 (drifted)')
    axes[1].scatter(frame1[outlier_idx, 0], frame1[outlier_idx, 1], 
                    c='orange', s=100, marker='x', label='Outliers')
    axes[1].set_title(f'Before (Mean residual = {np.mean(residuals_before):.2f})')
    axes[1].set_xlabel('x')
    axes[1].legend()
    axes[1].axis('equal')
    axes[1].grid(True, alpha=0.3)
    
    # After correction
    axes[2].scatter(frame0[:, 0], frame0[:, 1], c='blue', alpha=0.4, label='Frame 0')
    axes[2].scatter(frame1_corrected[:, 0], frame1_corrected[:, 1], 
                    c='green', alpha=0.4, label='Frame 1 (corrected)')
    axes[2].set_title(f'After BioMatrix (Mean residual = {np.mean(residuals_after):.2f})')
    axes[2].set_xlabel('x')
    axes[2].legend()
    axes[2].axis('equal')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_2d_drift.png', dpi=150)
    print("\nPlot saved: example_2d_drift.png")
    plt.show()


if __name__ == "__main__":
    main()
