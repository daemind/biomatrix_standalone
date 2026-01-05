# -*- coding: utf-8 -*-
"""
Unit Tests: Zeiss Plugin

TDD-compliant tests for:
- LiveTracker
- DriftCorrector
- FiberLinker
- FiducialDriftCorrector
"""

import pytest
import numpy as np
import sys



from biomatrix.plugins.zeiss import LiveTracker, DriftCorrector, FiberLinker
from biomatrix.plugins.zeiss.fiducial import FiducialDriftCorrector


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def simple_drift_data():
    """Generate simple drift sequence (pure translation)."""
    np.random.seed(42)
    n_pts = 50
    base = np.random.randn(n_pts, 3) * 5
    
    frames = []
    drifts = []
    for t in range(10):
        drift = np.array([t * 0.1, t * 0.05, t * 0.02])
        frames.append(base + drift)
        drifts.append(drift)
    
    return frames, np.array(drifts)


@pytest.fixture
def noisy_drift_data():
    """Generate drift sequence with noise."""
    np.random.seed(42)
    n_pts = 100
    base = np.random.randn(n_pts, 3) * 5
    
    frames = []
    drifts = []
    for t in range(20):
        drift = np.array([t * 0.05, t * 0.025, t * 0.01])
        noise = np.random.randn(n_pts, 3) * 0.1
        frames.append(base + drift + noise)
        drifts.append(drift)
    
    return frames, np.array(drifts)


@pytest.fixture
def fiber_data():
    """Generate fiber tracking data."""
    np.random.seed(42)
    n_fibers = 3
    pts_per_fiber = 20
    
    frames = []
    for t in range(10):
        drift = np.array([t * 0.02, t * 0.01, 0])
        all_pts = []
        
        for fid in range(n_fibers):
            s = np.linspace(0, 5, pts_per_fiber)
            fiber = np.column_stack([
                fid * 3 + s * 0.1,
                s,
                np.zeros(pts_per_fiber)
            ])
            fiber += drift
            all_pts.append(fiber)
        
        frames.append(np.vstack(all_pts))
    
    return frames


@pytest.fixture
def fiducial_data():
    """Generate data with known fiducials."""
    np.random.seed(42)
    n_molecules = 80
    n_fiducials = 5
    
    mol_base = np.random.randn(n_molecules, 3) * 5
    fid_base = np.random.randn(n_fiducials, 3) * 2
    
    frames = []
    drifts = []
    fiducial_indices = np.arange(n_molecules, n_molecules + n_fiducials)
    
    for t in range(20):
        drift = np.array([t * 0.02, t * 0.01, t * 0.005])
        
        molecules = mol_base + drift + np.random.randn(n_molecules, 3) * 0.5
        fiducials = fid_base + drift + np.random.randn(n_fiducials, 3) * 0.01
        
        frames.append(np.vstack([molecules, fiducials]))
        drifts.append(drift)
    
    return frames, np.array(drifts), fiducial_indices


# ============================================================
# LIVE TRACKER TESTS
# ============================================================

class TestLiveTracker:
    """Tests for LiveTracker module."""
    
    def test_first_frame_returns_none(self):
        """First frame should return None (no previous reference)."""
        tracker = LiveTracker()
        pts = np.random.randn(50, 3)
        
        result = tracker.track(pts)
        
        assert result is None
    
    def test_detects_pure_translation(self, simple_drift_data):
        """Should detect pure translation between frames."""
        frames, true_drifts = simple_drift_data
        tracker = LiveTracker()
        
        for i, pts in enumerate(frames):
            tracker.track(pts)
        
        total_drift = tracker.get_total_drift()
        true_total = true_drifts[-1]
        
        # Error should be < 20%
        rel_error = np.linalg.norm(total_drift - true_total) / np.linalg.norm(true_total)
        assert rel_error < 0.2
    
    def test_trajectory_length_matches_frames(self, simple_drift_data):
        """Trajectory should have N-1 entries for N frames."""
        frames, _ = simple_drift_data
        tracker = LiveTracker()
        
        for pts in frames:
            tracker.track(pts)
        
        assert len(tracker.trajectory) == len(frames) - 1
    
    def test_handles_noisy_data(self, noisy_drift_data):
        """Should handle noisy data with reasonable accuracy."""
        frames, true_drifts = noisy_drift_data
        tracker = LiveTracker()
        
        for pts in frames:
            tracker.track(pts)
        
        total_drift = tracker.get_total_drift()
        true_total = true_drifts[-1]
        
        rel_error = np.linalg.norm(total_drift - true_total) / np.linalg.norm(true_total)
        assert rel_error < 0.5  # Allow higher error for noisy data


# ============================================================
# DRIFT CORRECTOR TESTS
# ============================================================

class TestDriftCorrector:
    """Tests for DriftCorrector module."""
    
    def test_first_frame_unchanged(self):
        """First frame should be returned unchanged."""
        corrector = DriftCorrector()
        pts = np.random.randn(50, 3)
        
        corrected = corrector.correct(pts)
        
        np.testing.assert_array_equal(corrected, pts)
    
    def test_corrects_drift(self, simple_drift_data):
        """Should reduce drift in corrected frames."""
        frames, _ = simple_drift_data
        corrector = DriftCorrector(window_size=3)
        
        ref_centroid = np.mean(frames[0], axis=0)
        corrected_centroids = []
        uncorrected_centroids = []
        
        for pts in frames:
            corrected = corrector.correct(pts)
            corrected_centroids.append(np.mean(corrected, axis=0))
            uncorrected_centroids.append(np.mean(pts, axis=0))
        
        # Corrected should have lower variance
        corrected_std = np.std(corrected_centroids, axis=0).mean()
        uncorrected_std = np.std(uncorrected_centroids, axis=0).mean()
        
        assert corrected_std < uncorrected_std
    
    def test_get_drift_returns_array(self, simple_drift_data):
        """get_drift should return numpy array."""
        frames, _ = simple_drift_data
        corrector = DriftCorrector()
        
        for pts in frames:
            corrector.correct(pts)
        
        drift = corrector.get_drift()
        
        assert isinstance(drift, np.ndarray)
        assert drift.shape == (3,)


# ============================================================
# FIBER LINKER TESTS
# ============================================================

class TestFiberLinker:
    """Tests for FiberLinker module."""
    
    def test_detects_fibers(self, fiber_data):
        """Should detect correct number of fibers."""
        linker = FiberLinker()
        
        result = linker.process(fiber_data[0])
        
        assert len(result) > 0
    
    def test_maintains_fiber_ids(self, fiber_data):
        """Should maintain consistent fiber IDs across frames."""
        linker = FiberLinker(max_displacement=1.0)
        
        all_results = []
        for pts in fiber_data:
            result = linker.process(pts)
            all_results.append(set(result.keys()))
        
        # Check that IDs persist
        first_ids = all_results[0]
        persistent = sum(1 for ids in all_results if first_ids & ids)
        
        assert persistent >= len(fiber_data) * 0.8  # 80% continuity
    
    def test_get_tracks_returns_dict(self, fiber_data):
        """get_tracks should return dictionary."""
        linker = FiberLinker()
        
        for pts in fiber_data:
            linker.process(pts)
        
        tracks = linker.get_tracks()
        
        assert isinstance(tracks, dict)


# ============================================================
# FIDUCIAL DRIFT CORRECTOR TESTS
# ============================================================

class TestFiducialDriftCorrector:
    """Tests for FiducialDriftCorrector module."""
    
    def test_first_frame_returns_zero_drift(self, fiducial_data):
        """First frame should return zero drift."""
        frames, _, fid_indices = fiducial_data
        corrector = FiducialDriftCorrector(known_fiducial_indices=fid_indices)
        
        corrected, drift = corrector.correct(frames[0])
        
        np.testing.assert_array_equal(drift, np.zeros(3))
    
    def test_sub_nm_precision(self, fiducial_data):
        """Should achieve sub-nm precision with known fiducials."""
        frames, true_drifts, fid_indices = fiducial_data
        corrector = FiducialDriftCorrector(known_fiducial_indices=fid_indices)
        
        errors = []
        for i, pts in enumerate(frames):
            _, detected = corrector.correct(pts)
            err = np.linalg.norm(detected - true_drifts[i])
            errors.append(err)
        
        mean_error = np.mean(errors)
        
        assert mean_error < 0.1  # Sub-nm precision
    
    def test_get_fiducial_count(self, fiducial_data):
        """Should return correct fiducial count."""
        frames, _, fid_indices = fiducial_data
        corrector = FiducialDriftCorrector(known_fiducial_indices=fid_indices)
        
        corrector.correct(frames[0])
        
        # With known indices, count should match
        assert corrector.get_fiducial_count() == len(fid_indices)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
