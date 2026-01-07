#!/usr/bin/env python3
"""
Demo: 3D Walker Dynamics - Deriving Motion Laws from Point Cloud Sequences

This demo shows how BioMatrix can derive GLOBAL DYNAMIC LAWS from a sequence
of 3D point clouds representing a walking skeleton, WITHOUT any neural network.

Key concepts demonstrated:
- Velocity estimation via Procrustes between frames
- Period/cadence detection via FFT of limb oscillations
- Pure geometric inference (no training data needed)

This is what differentiates BioMatrix from YOLO-style detection:
instead of learning features, we derive physical laws algebraically.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.insert(0, '.')

from biomatrix.core.state import State
from biomatrix.core.derive.procrustes import derive_procrustes_se3


# ============================================================
# 3D WALKING SKELETON GENERATOR
# ============================================================

def generate_walking_skeleton(n_frames=100, velocity=0.1, period=20, noise_std=0.02):
    """
    Generate a synthetic 3D walking skeleton sequence.
    
    Joint layout (13 joints):
        0: pelvis (root)
        1: spine
        2: neck
        3: head
        4-5: left/right shoulder
        6-7: left/right elbow
        8-9: left/right hip
        10-11: left/right knee
        12-13: left/right ankle
    
    Args:
        n_frames: Number of frames
        velocity: Forward speed (units/frame)
        period: Walking cycle period (frames)
        noise_std: Measurement noise
    
    Returns:
        frames: List of (14, 3) arrays
        true_velocity: Ground truth velocity
        true_period: Ground truth period
    """
    np.random.seed(42)
    
    # Base skeleton (T-pose)
    base_skeleton = np.array([
        [0, 0, 1.0],      # 0: pelvis
        [0, 0, 1.3],      # 1: spine
        [0, 0, 1.5],      # 2: neck
        [0, 0, 1.7],      # 3: head
        [-0.2, 0, 1.4],   # 4: left shoulder
        [0.2, 0, 1.4],    # 5: right shoulder
        [-0.4, 0, 1.2],   # 6: left elbow
        [0.4, 0, 1.2],    # 7: right elbow
        [-0.1, 0, 0.9],   # 8: left hip
        [0.1, 0, 0.9],    # 9: right hip
        [-0.1, 0, 0.5],   # 10: left knee
        [0.1, 0, 0.5],    # 11: right knee
        [-0.1, 0, 0.0],   # 12: left ankle
        [0.1, 0, 0.0],    # 13: right ankle
    ])
    
    frames = []
    
    for t in range(n_frames):
        skeleton = base_skeleton.copy()
        
        # Forward motion (X direction)
        skeleton[:, 0] += velocity * t
        
        # Walking oscillation
        phase = 2 * np.pi * t / period
        
        # Leg swing (opposing phases)
        leg_swing = 0.15 * np.sin(phase)
        skeleton[10, 1] += leg_swing      # left knee
        skeleton[11, 1] -= leg_swing      # right knee
        skeleton[12, 1] += leg_swing * 1.5  # left ankle
        skeleton[13, 1] -= leg_swing * 1.5  # right ankle
        
        # Arm swing (opposite to legs)
        arm_swing = 0.1 * np.sin(phase)
        skeleton[6, 1] -= arm_swing   # left elbow
        skeleton[7, 1] += arm_swing   # right elbow
        
        # Hip rotation
        skeleton[0, 1] += 0.02 * np.sin(phase)
        
        # Add measurement noise
        skeleton += np.random.randn(*skeleton.shape) * noise_std
        
        frames.append(skeleton)
    
    return frames, velocity, period


# ============================================================
# DYNAMIC LAW DERIVATION
# ============================================================

def derive_walker_dynamics(frames):
    """
    Derive walking dynamics from point cloud sequence using BioMatrix.
    
    Derives:
    1. Global velocity (via Procrustes frame-to-frame)
    2. Walking period (via FFT of limb oscillations)
    
    Returns:
        estimated_velocity: Derived forward speed
        estimated_period: Derived walking cycle
        velocities: Per-frame velocity vectors
        oscillation_spectrum: FFT of limb motion
    """
    
    # 1. Derive instantaneous velocities via Procrustes
    velocities = []
    
    for i in range(1, len(frames)):
        state_prev = State(frames[i-1])
        state_curr = State(frames[i])
        
        op = derive_procrustes_se3(state_prev, state_curr)
        
        if op is not None:
            # Translation component = velocity
            velocities.append(op.t)
        else:
            # Fallback: centroid difference
            v = np.mean(frames[i], axis=0) - np.mean(frames[i-1], axis=0)
            velocities.append(v)
    
    velocities = np.array(velocities)
    
    # Global velocity = mean translation (X component is forward)
    mean_velocity = np.mean(velocities, axis=0)
    estimated_velocity = mean_velocity[0]  # Forward is X
    
    # 2. Derive period via FFT of limb oscillations
    # Track left ankle (index 12) Y-position relative to pelvis
    left_ankle_y = np.array([f[12, 1] - f[0, 1] for f in frames])
    
    # FFT
    fft = np.abs(np.fft.rfft(left_ankle_y))
    freqs = np.fft.rfftfreq(len(frames))
    
    # Find dominant frequency (skip DC)
    peak_idx = np.argmax(fft[1:]) + 1
    peak_freq = freqs[peak_idx]
    
    # Period = 1 / frequency
    estimated_period = 1.0 / peak_freq if peak_freq > 0 else len(frames)
    
    return estimated_velocity, estimated_period, velocities, (freqs, fft)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("BIOMATRIX DEMO: 3D Walker Dynamics")
    print("=" * 60)
    print("\nDeriving motion laws from 3D skeleton sequences...")
    print("No neural network. Pure geometric algebra.\n")
    
    # Generate walking sequence
    true_velocity = 0.1
    true_period = 20
    
    frames, _, _ = generate_walking_skeleton(
        n_frames=100,
        velocity=true_velocity,
        period=true_period,
        noise_std=0.02
    )
    
    print(f"Generated {len(frames)} frames")
    print(f"True velocity: {true_velocity:.3f} units/frame")
    print(f"True period: {true_period} frames/cycle")
    
    # Derive dynamics
    est_velocity, est_period, velocities, (freqs, fft) = derive_walker_dynamics(frames)
    
    print("\n" + "=" * 40)
    print("DERIVED PARAMETERS (via Procrustes + FFT)")
    print("=" * 40)
    print(f"Estimated velocity: {est_velocity:.4f} units/frame")
    print(f"Velocity error: {abs(est_velocity - true_velocity) / true_velocity * 100:.1f}%")
    print(f"\nEstimated period: {est_period:.1f} frames/cycle")
    print(f"Period error: {abs(est_period - true_period) / true_period * 100:.1f}%")
    
    # Visualize
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Skeleton trajectory
    ax1 = fig.add_subplot(131, projection='3d')
    for i in [0, 25, 50, 75, 99]:
        f = frames[i]
        ax1.scatter(f[:, 0], f[:, 1], f[:, 2], alpha=0.3 + 0.07*i/20)
        # Draw skeleton lines
        connections = [(0,1), (1,2), (2,3), (1,4), (1,5), (4,6), (5,7),
                       (0,8), (0,9), (8,10), (9,11), (10,12), (11,13)]
        for a, b in connections:
            ax1.plot([f[a,0], f[b,0]], [f[a,1], f[b,1]], [f[a,2], f[b,2]], 
                     'b-', alpha=0.3 + 0.07*i/20)
    ax1.set_xlabel('X (forward)')
    ax1.set_ylabel('Y (lateral)')
    ax1.set_zlabel('Z (vertical)')
    ax1.set_title('3D Skeleton Trajectory')
    
    # 2. Velocity over time
    ax2 = fig.add_subplot(132)
    ax2.plot(velocities[:, 0], 'b-', label='Vx (forward)', linewidth=2)
    ax2.plot(velocities[:, 1], 'g-', label='Vy (lateral)', alpha=0.7)
    ax2.plot(velocities[:, 2], 'r-', label='Vz (vertical)', alpha=0.7)
    ax2.axhline(true_velocity, color='b', linestyle='--', label=f'True Vx={true_velocity}')
    ax2.axhline(est_velocity, color='orange', linestyle=':', label=f'Est Vx={est_velocity:.3f}')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Velocity')
    ax2.set_title(f'Frame-to-Frame Velocity (Procrustes)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. FFT spectrum
    ax3 = fig.add_subplot(133)
    # Convert to cycles/frame for readability
    ax3.plot(freqs * len(frames), fft, 'b-', linewidth=2)
    ax3.axvline(len(frames) / true_period, color='g', linestyle='--', 
                label=f'True: {true_period} frames/cycle')
    ax3.axvline(len(frames) / est_period, color='orange', linestyle=':', 
                label=f'Est: {est_period:.1f} frames/cycle')
    ax3.set_xlabel('Frequency (cycles/sequence)')
    ax3.set_ylabel('Magnitude')
    ax3.set_title('Limb Oscillation Spectrum (FFT)')
    ax3.set_xlim([0, 15])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_walker_dynamics.png', dpi=150)
    print("\nPlot saved: demo_walker_dynamics.png")
    plt.show()
    
    # Final summary
    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print("""
BioMatrix derived PHYSICAL LAWS from raw point clouds:
- Forward velocity: inferred via SE(3) Procrustes
- Walking period: inferred via FFT on limb oscillations

This is NOT detection (like YOLO), but UNDERSTANDING:
- No training data needed
- Works on any N-dimensional skeleton
- Laws are explicit and interpretable
""")


if __name__ == "__main__":
    main()
