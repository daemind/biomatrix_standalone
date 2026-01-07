# -*- coding: utf-8 -*-
"""
Drift Corrector - Thermal drift compensation

Uses sliding window Procrustes to estimate and correct drift.
N-dimensional agnostic.
"""

import numpy as np
from typing import Optional
from collections import deque




from biomatrix.core.state import State
from biomatrix.core.derive.procrustes import derive_procrustes_se3, ProcrustesOperator


class DriftCorrector:
    """
    Thermal drift correction using sliding window average.
    
    Accumulates frame-to-frame transformations and applies
    inverse correction to stabilize point clouds.
    """
    
    def __init__(self, window_size: int = 10, reference_mode: str = "first"):
        """
        Args:
            window_size: Frames for sliding average
            reference_mode: "first" (absolute) or "sliding" (relative)
        """
        self.window_size = window_size
        self.reference_mode = reference_mode
        
        self.reference_state = None
        self.drift_history = deque(maxlen=window_size)
        self.cumulative_drift = np.zeros(3)
        
    def correct(self, pts: np.ndarray) -> np.ndarray:
        """
        Apply drift correction to point cloud.
        
        Args:
            pts: Input points (N x D)
            
        Returns:
            Drift-corrected points
        """
        D = pts.shape[1]
        
        # Initialize reference
        if self.reference_state is None:
            self.reference_state = State(pts)
            self.reference_centroid = np.mean(pts, axis=0)
            self.cumulative_drift = np.zeros(D)
            return pts
            
        # Estimate drift from centroid shift (robust to blinking)
        curr_centroid = np.mean(pts, axis=0)
        drift = curr_centroid - self.reference_centroid
        
        self.drift_history.append(drift)
        
        # Smoothed drift (sliding average)
        smooth_drift = np.mean(list(self.drift_history), axis=0)
        
        # Update cumulative for absolute mode
        if self.reference_mode == "first":
            self.cumulative_drift = smooth_drift
        else:
            self.cumulative_drift += drift
            self.reference_centroid = curr_centroid
            
        # Apply inverse correction
        corrected = pts - self.cumulative_drift
        
        return corrected
        
    def get_drift(self) -> np.ndarray:
        """Get current estimated drift."""
        return self.cumulative_drift.copy()
        
    def get_drift_history(self) -> np.ndarray:
        """Get drift time series."""
        return np.array(list(self.drift_history))
        
    def reset(self, new_reference: Optional[np.ndarray] = None):
        """Reset drift correction."""
        if new_reference is not None:
            self.reference_state = State(new_reference)
        else:
            self.reference_state = None
        self.drift_history.clear()
        self.cumulative_drift = np.zeros(3)
