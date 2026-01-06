#!/usr/bin/env python3
"""
Demo: 3D LiDAR Point Cloud Registration

Demonstrates BioMatrix on realistic 3D point cloud alignment:
- Fractal terrain generation (sum of harmonics)
- SE(3) Procrustes registration between scans
- Vehicle motion estimation

Application: Autonomous driving, drone mapping, SLAM.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.insert(0, '.')

from biomatrix.core.state import State
from biomatrix.core.derive.procrustes import derive_procrustes_se3


def generate_fractal_terrain(n_side=30, scale=5.0):
    """
    Generate fractal-like terrain using sum of sinusoids.
    
    This mimics natural terrain with multi-scale features.
    """
    x = np.linspace(-scale, scale, n_side)
    y = np.linspace(-scale, scale, n_side)
    X, Y = np.meshgrid(x, y)
    X, Y = X.flatten(), Y.flatten()
    
    # Fractal: sum of harmonics at different frequencies
    Z = np.zeros_like(X)
    frequencies = [0.3, 0.7, 1.5, 3.0, 5.0]
    for freq in frequencies:
        amplitude = 2.0 / freq
        Z += amplitude * np.sin(freq * X + 0.3) * np.cos(freq * Y + 0.5)
    
    # Add some random variation
    Z += 0.3 * np.sin(2 * X) * np.sin(2 * Y)
    
    return np.column_stack([X, Y, Z])


def main():
    print("=" * 60)
    print("BIOMATRIX DEMO: 3D LiDAR Point Cloud Registration")
    print("=" * 60)
    print("\nSimulating vehicle motion between LiDAR scans...")
    
    np.random.seed(42)
    
    # Generate fractal terrain (scan 1)
    scan1 = generate_fractal_terrain(n_side=30)
    n_points = len(scan1)
    
    # Simulated vehicle motion
    true_drift = np.array([0.5, 0.3, 0.1])  # X, Y, Z translation
    noise_sigma = 0.1
    
    # Generate scan 2 with motion + noise
    noise = np.random.randn(n_points, 3) * noise_sigma
    scan2 = scan1 + true_drift + noise
    
    print(f"\nPoints per scan: {n_points}")
    print(f"True vehicle motion: {true_drift}")
    print(f"Measurement noise σ: {noise_sigma}")
    
    # Apply BioMatrix Procrustes
    op = derive_procrustes_se3(State(scan1), State(scan2))
    
    if op is not None:
        detected_drift = op.t
        scan2_aligned = scan2 - detected_drift
        print(f"\n=== PROCRUSTES RESULT ===")
        print(f"Detected motion: {detected_drift}")
        print(f"Detection error: {np.linalg.norm(detected_drift - true_drift):.6f}")
    else:
        print("Procrustes failed - using centroid fallback")
        detected_drift = np.mean(scan2, axis=0) - np.mean(scan1, axis=0)
        scan2_aligned = scan2 - detected_drift
    
    # Compute alignment quality
    error_before = np.mean(np.linalg.norm(scan2 - scan1, axis=1))
    error_after = np.mean(np.linalg.norm(scan2_aligned - scan1, axis=1))
    improvement = error_before / error_after
    
    print(f"\n=== ALIGNMENT QUALITY ===")
    print(f"Mean error before: {error_before:.4f}")
    print(f"Mean error after:  {error_after:.4f}")
    print(f"Improvement: {improvement:.1f}x")
    
    # Visualization
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Reference terrain
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(scan1[:, 0], scan1[:, 1], scan1[:, 2], 
                c=scan1[:, 2], cmap='terrain', s=5, alpha=0.7)
    ax1.set_title('Scan 1: Reference Terrain')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # 2. Before alignment
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(scan1[:, 0], scan1[:, 1], scan1[:, 2], 
                c='blue', s=3, alpha=0.3, label='Scan 1')
    ax2.scatter(scan2[:, 0], scan2[:, 1], scan2[:, 2], 
                c='red', s=3, alpha=0.3, label='Scan 2 (drifted)')
    ax2.set_title(f'Before (Error: {error_before:.3f})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    # 3. After alignment
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(scan1[:, 0], scan1[:, 1], scan1[:, 2], 
                c='blue', s=3, alpha=0.3, label='Scan 1')
    ax3.scatter(scan2_aligned[:, 0], scan2_aligned[:, 1], scan2_aligned[:, 2], 
                c='green', s=3, alpha=0.3, label='Scan 2 (aligned)')
    ax3.set_title(f'After Procrustes (Error: {error_after:.3f})')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('demo_lidar_3d.png', dpi=150)
    print("\nPlot saved: demo_lidar_3d.png")
    plt.show()
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print("""
BioMatrix aligns 3D point clouds in closed-form:
- No ICP iterations needed (single SVD)
- O(N·D²) complexity = real-time capable
- Works identically on 2D, 3D, or any N dimensions

For unknown correspondences, use this as inner loop of ICP
or add nearest-neighbor matching as preprocessing step.
""")


if __name__ == "__main__":
    main()
