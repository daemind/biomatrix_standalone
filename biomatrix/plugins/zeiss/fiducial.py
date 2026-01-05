# -*- coding: utf-8 -*-
"""
Fiducial Detection - High-precision drift correction using reference markers

Fiducial markers (beads, quantum dots) provide stable reference points
for sub-nanometer drift correction in SMLM.

N-dimensional agnostic.
"""

import numpy as np
from typing import Optional, List, Tuple
from collections import deque
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment




from biomatrix.core.state import State
from biomatrix.core.derive.procrustes import derive_procrustes_se3, ProcrustesOperator


class FiducialDetector:
    """
    Detect and track fiducial markers across frames.
    
    Fiducials are identified by:
    1. High temporal stability (low variance across frames)
    2. High intensity (typically brighter than molecules)
    3. Consistent appearance
    """
    
    def __init__(
        self,
        min_frames: int = 10,
        stability_threshold: float = 0.5,
        intensity_percentile: float = 90
    ):
        """
        Args:
            min_frames: Minimum frames to establish fiducial identity
            stability_threshold: Max variance for fiducial classification
            intensity_percentile: Minimum intensity for fiducials
        """
        self.min_frames = min_frames
        self.stability_threshold = stability_threshold
        self.intensity_percentile = intensity_percentile
        
        self.point_history = deque(maxlen=100)
        self.fiducials = None
        self.fiducial_ids = []
        
    def update(self, pts: np.ndarray, intensities: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Update point history and detect fiducials.
        
        Args:
            pts: Point cloud (N x D)
            intensities: Optional intensity values per point
            
        Returns:
            Array of fiducial indices in current frame
        """
        self.point_history.append(pts.copy())
        
        if len(self.point_history) < self.min_frames:
            return np.array([])
            
        # Detect fiducials based on history
        return self._detect_fiducials(pts, intensities)
        
    def _detect_fiducials(
        self,
        pts: np.ndarray,
        intensities: Optional[np.ndarray]
    ) -> np.ndarray:
        """Detect stable points that are likely fiducials."""
        
        # Method 1: Track points across all frames and find stable ones
        n_pts = len(pts)
        
        if n_pts == 0:
            return np.array([])
            
        # For each point in current frame, find matches in history
        stability_scores = np.zeros(n_pts)
        
        for i, pt in enumerate(pts):
            matches = 0
            total_dist = 0
            
            for hist_pts in self.point_history:
                if len(hist_pts) == 0:
                    continue
                    
                # Find nearest point
                dists = np.linalg.norm(hist_pts - pt, axis=1)
                min_dist = np.min(dists)
                
                if min_dist < self.stability_threshold:
                    matches += 1
                    total_dist += min_dist
                    
            # Stability = presence ratio * (1 / average distance)
            if matches > 0:
                presence_ratio = matches / len(self.point_history)
                avg_dist = total_dist / matches
                stability_scores[i] = presence_ratio / (avg_dist + 0.01)
                
        # Select top stable points as fiducials
        threshold = np.percentile(stability_scores, self.intensity_percentile)
        fiducial_mask = stability_scores >= threshold
        
        # Additional intensity filter if available
        if intensities is not None:
            intensity_threshold = np.percentile(intensities, self.intensity_percentile)
            fiducial_mask &= (intensities >= intensity_threshold)
            
        self.fiducial_ids = np.where(fiducial_mask)[0]
        self.fiducials = pts[fiducial_mask]
        
        return self.fiducial_ids
        
    def get_fiducials(self, pts: np.ndarray) -> np.ndarray:
        """Get fiducial points from current frame."""
        if len(self.fiducial_ids) == 0:
            return np.empty((0, pts.shape[1]))
        return pts[self.fiducial_ids]


class FiducialDriftCorrector:
    """
    High-precision drift correction using fiducial markers.
    
    Achieves sub-nanometer precision by:
    1. Using known fiducial indices (best) or auto-detection
    2. Tracking only fiducials for drift estimation
    3. Averaging over multiple fiducials
    """
    
    def __init__(
        self,
        detector: Optional[FiducialDetector] = None,
        window_size: int = 5,
        known_fiducial_indices: Optional[np.ndarray] = None
    ):
        """
        Args:
            detector: FiducialDetector instance (or create new)
            window_size: Frames for sliding average
            known_fiducial_indices: If provided, use these indices as fiducials
                                    (best precision, ~0.01nm)
        """
        self.detector = detector or FiducialDetector()
        self.window_size = window_size
        self.known_fiducial_indices = known_fiducial_indices
        
        self.reference_centroid = None
        self.drift_history = deque(maxlen=window_size)
        self.cumulative_drift = None
        
    def correct(
        self,
        pts: np.ndarray,
        intensities: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply fiducial-based drift correction.
        
        Args:
            pts: Point cloud (N x D)
            intensities: Optional intensity values
            
        Returns:
            Tuple of (corrected_points, detected_drift)
        """
        D = pts.shape[1]
        
        # Get fiducials - known indices or auto-detected
        if self.known_fiducial_indices is not None:
            fiducials = pts[self.known_fiducial_indices]
        else:
            self.detector.update(pts, intensities)
            fiducials = self.detector.get_fiducials(pts)
        
        if len(fiducials) < 2:
            return pts, np.zeros(D)
            
        # Compute fiducial centroid
        curr_centroid = np.mean(fiducials, axis=0)
        
        # Initialize reference
        if self.reference_centroid is None:
            self.reference_centroid = curr_centroid.copy()
            self.cumulative_drift = np.zeros(D)
            return pts, np.zeros(D)
            
        # Compute drift from reference
        drift = curr_centroid - self.reference_centroid
        self.drift_history.append(drift)
        
        # Smooth drift (sliding average)
        smooth_drift = np.mean(list(self.drift_history), axis=0)
        self.cumulative_drift = smooth_drift
        
        # Apply correction
        corrected = pts - self.cumulative_drift
        
        return corrected, self.cumulative_drift
        
    def _match_fiducials(
        self,
        ref: np.ndarray,
        curr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Match fiducials between frames using Hungarian algorithm."""
        dist_matrix = cdist(ref, curr)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        
        # Filter by distance
        threshold = 1.0
        matched_ref = []
        matched_curr = []
        
        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] < threshold:
                matched_ref.append(ref[r])
                matched_curr.append(curr[c])
                
        return np.array(matched_ref), np.array(matched_curr)
        
    def get_drift(self) -> np.ndarray:
        """Get current estimated drift."""
        return self.cumulative_drift.copy()
        
    def get_fiducial_count(self) -> int:
        """Get number of detected fiducials."""
        if self.known_fiducial_indices is not None:
            return len(self.known_fiducial_indices)
        return len(self.detector.fiducial_ids)
