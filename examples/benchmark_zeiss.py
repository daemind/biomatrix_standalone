#!/usr/bin/env python3
"""
SCIENTIFIC BENCHMARK: Zeiss Plugin

Non-trivial tests for:
1. Drift detection accuracy
2. Drift correction error
3. Fiber tracking continuity
4. Noise robustness
5. Multi-frame accumulation
6. Fiducial-based precision (NEW)
"""

import numpy as np
import sys


from biomatrix.plugins.zeiss import LiveTracker, DriftCorrector, FiberLinker
from biomatrix.plugins.zeiss.fiducial import FiducialDriftCorrector
from biomatrix.core.state import State

# ============================================================
# DATA GENERATORS (Realistic Microscopy Scenarios)
# ============================================================

def generate_smlm_sequence(n_frames=100, n_molecules=200, drift_rate=0.01, noise_std=0.1):
    """
    Generate realistic SMLM sequence with:
    - Linear thermal drift
    - Gaussian localization noise
    - Blinking (molecules appear/disappear)
    """
    np.random.seed(42)
    
    # Fixed molecular positions
    base_positions = np.random.randn(n_molecules, 3) * 10
    
    frames = []
    true_drifts = []
    
    for t in range(n_frames):
        # Linear drift
        drift = np.array([drift_rate * t, drift_rate * t * 0.5, drift_rate * t * 0.2])
        true_drifts.append(drift)
        
        # Blinking: random subset visible
        visible = np.random.rand(n_molecules) > 0.3
        pts = base_positions[visible] + drift
        
        # Localization noise
        pts += np.random.randn(pts.shape[0], 3) * noise_std
        
        frames.append(pts)
    
    return frames, np.array(true_drifts)

def generate_fiber_sequence(n_frames=50, n_fibers=5, drift_rate=0.02, noise_std=0.05):
    """
    Generate DNA fiber tracking sequence with:
    - Multiple fibers
    - Linear drift
    - Fiber deformation (slight bending)
    """
    np.random.seed(42)
    
    frames = []
    fiber_trajectories = {i: [] for i in range(n_fibers)}
    
    for t in range(n_frames):
        drift = np.array([drift_rate * t, drift_rate * t * 0.3, 0])
        
        all_pts = []
        for fid in range(n_fibers):
            # Fiber as line segment
            fiber_start = np.array([fid * 5, 0, 0])
            n_pts = 30
            s = np.linspace(0, 10, n_pts)
            
            # Fiber points with slight bend
            bend = 0.1 * np.sin(s * 0.5 + t * 0.1)
            fiber_pts = np.column_stack([
                fiber_start[0] + s * 0.1,
                fiber_start[1] + s + bend,
                np.zeros(n_pts)
            ])
            
            # Apply drift and noise
            fiber_pts += drift
            fiber_pts += np.random.randn(n_pts, 3) * noise_std
            
            all_pts.append(fiber_pts)
            fiber_trajectories[fid].append(fiber_pts.mean(axis=0))
        
        frames.append(np.vstack(all_pts))
    
    return frames, fiber_trajectories

def generate_noisy_drift(n_frames=100, drift_rate=0.05, jitter_std=0.02):
    """
    Generate drift with non-linear jitter (realistic thermal fluctuations).
    """
    np.random.seed(42)
    
    n_pts = 100
    base = np.random.randn(n_pts, 3) * 5
    
    frames = []
    true_drifts = []
    
    cumulative_drift = np.zeros(3)
    
    for t in range(n_frames):
        # Base linear drift
        linear = np.array([drift_rate, drift_rate * 0.5, drift_rate * 0.2])
        
        # Non-linear jitter (Brownian-like)
        jitter = np.random.randn(3) * jitter_std
        
        cumulative_drift += linear + jitter
        true_drifts.append(cumulative_drift.copy())
        
        pts = base + cumulative_drift + np.random.randn(n_pts, 3) * 0.1
        frames.append(pts)
    
    return frames, np.array(true_drifts)

# ============================================================
# BENCHMARK TESTS
# ============================================================

def benchmark_drift_detection():
    """Test drift detection accuracy."""
    print("\n" + "=" * 60)
    print("TEST 1: DRIFT DETECTION ACCURACY")
    print("=" * 60)
    
    frames, true_drifts = generate_smlm_sequence(n_frames=50, drift_rate=0.05)
    
    tracker = LiveTracker()
    detected = []
    
    for pts in frames:
        op = tracker.track(pts)
        if op is not None:
            detected.append(op.t)
    
    # Compare cumulative detected vs true
    detected_cumulative = np.cumsum(detected, axis=0)
    true_final = true_drifts[-1]
    detected_final = tracker.get_total_drift()
    
    error = np.linalg.norm(detected_final - true_final)
    relative_error = error / np.linalg.norm(true_final)
    
    print(f"True final drift:     {true_final}")
    print(f"Detected final drift: {detected_final}")
    print(f"Absolute error: {error:.6f}")
    print(f"Relative error: {100 * relative_error:.2f}%")
    
    return relative_error < 0.1  # Pass if <10% error

def benchmark_drift_correction():
    """Test drift correction quality."""
    print("\n" + "=" * 60)
    print("TEST 2: DRIFT CORRECTION QUALITY")
    print("=" * 60)
    
    frames, true_drifts = generate_smlm_sequence(n_frames=100, drift_rate=0.02)
    
    corrector = DriftCorrector(window_size=5)
    
    # Get first frame as reference
    ref_centroid = np.mean(frames[0], axis=0)
    
    corrected_centroids = []
    uncorrected_centroids = []
    
    for pts in frames:
        corrected = corrector.correct(pts)
        corrected_centroids.append(np.mean(corrected, axis=0))
        uncorrected_centroids.append(np.mean(pts, axis=0))
    
    # Measure stability (std of centroid positions)
    corrected_std = np.std(corrected_centroids, axis=0)
    uncorrected_std = np.std(uncorrected_centroids, axis=0)
    
    improvement = np.mean(uncorrected_std) / np.mean(corrected_std)
    
    print(f"Uncorrected centroid std: {uncorrected_std}")
    print(f"Corrected centroid std:   {corrected_std}")
    print(f"Stability improvement: {improvement:.1f}x")
    
    return improvement > 2.0  # Pass if 2x more stable

def benchmark_fiber_tracking():
    """Test fiber tracking continuity."""
    print("\n" + "=" * 60)
    print("TEST 3: FIBER TRACKING CONTINUITY")
    print("=" * 60)
    
    frames, true_trajectories = generate_fiber_sequence(n_frames=30, n_fibers=5)
    
    linker = FiberLinker(max_displacement=2.0)
    
    tracked_ids = []
    for pts in frames:
        result = linker.process(pts)
        tracked_ids.append(set(result.keys()))
    
    # Check ID persistence
    first_ids = tracked_ids[0]
    persistent_count = sum(1 for ids in tracked_ids if first_ids & ids)
    
    continuity_rate = persistent_count / len(frames)
    
    print(f"Initial fibers: {len(first_ids)}")
    print(f"Frames with persistent fibers: {persistent_count}/{len(frames)}")
    print(f"Continuity rate: {100 * continuity_rate:.1f}%")
    
    return continuity_rate > 0.8  # Pass if >80% continuity

def benchmark_noise_robustness():
    """Test robustness to high noise."""
    print("\n" + "=" * 60)
    print("TEST 4: NOISE ROBUSTNESS")
    print("=" * 60)
    
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
    results = []
    
    for noise in noise_levels:
        frames, true_drifts = generate_smlm_sequence(n_frames=30, drift_rate=0.05, noise_std=noise)
        
        tracker = LiveTracker()
        for pts in frames:
            tracker.track(pts)
        
        detected_final = tracker.get_total_drift()
        true_final = true_drifts[-1]
        
        error = np.linalg.norm(detected_final - true_final) / np.linalg.norm(true_final)
        results.append((noise, error))
        print(f"Noise σ={noise:.2f}: Relative error = {100 * error:.1f}%")
    
    # Pass if error stays below 50% even at high noise
    max_error = max(r[1] for r in results)
    return max_error < 0.5

def benchmark_nonlinear_drift():
    """Test with non-linear (Brownian) drift."""
    print("\n" + "=" * 60)
    print("TEST 5: NON-LINEAR DRIFT (BROWNIAN)")
    print("=" * 60)
    
    frames, true_drifts = generate_noisy_drift(n_frames=100, jitter_std=0.03)
    
    corrector = DriftCorrector(window_size=10)
    
    errors = []
    for i, pts in enumerate(frames):
        corrected = corrector.correct(pts)
        
        # Residual drift after correction
        residual = np.mean(corrected, axis=0) - np.mean(frames[0], axis=0)
        errors.append(np.linalg.norm(residual))
    
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    print(f"Mean residual error: {mean_error:.6f}")
    print(f"Max residual error:  {max_error:.6f}")
    
    return mean_error < 0.5

def benchmark_fiducial_precision():
    """Test fiducial-based sub-nm precision."""
    print("\n" + "=" * 60)
    print("TEST 6: FIDUCIAL-BASED PRECISION")
    print("=" * 60)
    
    np.random.seed(42)
    
    n_molecules = 100
    n_fiducials = 5
    n_frames = 50
    drift_rate = 0.02
    
    fiducial_base = np.random.randn(n_fiducials, 3) * 2
    known_indices = np.arange(n_molecules, n_molecules + n_fiducials)
    
    corrector = FiducialDriftCorrector(known_fiducial_indices=known_indices)
    
    errors = []
    for t in range(n_frames):
        true_drift = np.array([drift_rate * t, drift_rate * t * 0.5, drift_rate * t * 0.2])
        
        molecules = np.random.randn(n_molecules, 3) * 5 + true_drift
        fiducials = fiducial_base + true_drift + np.random.randn(n_fiducials, 3) * 0.01
        pts = np.vstack([molecules, fiducials])
        
        corrected, detected = corrector.correct(pts)
        err = np.linalg.norm(detected - true_drift)
        errors.append(err)
    
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    print(f"Mean error: {mean_error:.6f} nm")
    print(f"Max error:  {max_error:.6f} nm")
    print(f"Precision: {'SUB-NM' if mean_error < 0.1 else 'NM-SCALE'}")
    
    return mean_error < 0.1  # Pass if sub-nm precision

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ZEISS PLUGIN SCIENTIFIC BENCHMARK")
    print("=" * 60)
    
    tests = [
        ("Drift Detection", benchmark_drift_detection),
        ("Drift Correction", benchmark_drift_correction),
        ("Fiber Tracking", benchmark_fiber_tracking),
        ("Noise Robustness", benchmark_noise_robustness),
        ("Non-linear Drift", benchmark_nonlinear_drift),
        ("Fiducial Precision", benchmark_fiducial_precision),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, "PASS" if passed else "FAIL"))
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((name, "ERROR"))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, status in results:
        symbol = "✅" if status == "PASS" else "❌"
        print(f"{symbol} {name}: {status}")
    
    passed = sum(1 for _, s in results if s == "PASS")
    print(f"\nTotal: {passed}/{len(tests)} tests passed")
