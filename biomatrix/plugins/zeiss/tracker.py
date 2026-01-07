# -*- coding: utf-8 -*-
"""
Live Tracker - Real-time SE(3) tracking using Procrustes

N-dimensional agnostic: works for 2D and 3D point clouds.
Handles variable point counts via nearest-neighbor matching.
"""

import numpy as np
from typing import Optional, List
from collections import deque
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment




from biomatrix.core.state import State
from biomatrix.core.derive.procrustes import derive_procrustes_se3, ProcrustesOperator


class LiveTracker:
    """
    Real-time SE(3) tracking between consecutive frames.
    
    Uses Procrustes analysis with point matching to handle
    blinking molecules (variable point counts).
    """
    
    def __init__(self, history_size: int = 100, match_threshold: float = 2.0):
        """
        Args:
            history_size: Number of past transforms to keep
            match_threshold: Max distance for point matching
        """
        self.prev_state = None
        self.prev_pts = None
        self.trajectory = deque(maxlen=history_size)
        self.cumulative_transform = None
        self.match_threshold = match_threshold
        
    def track(self, pts: np.ndarray) -> Optional[ProcrustesOperator]:
        """
        Derive transformation from previous frame with RANSAC outlier rejection.
        
        Args:
            pts: Current frame point cloud (N x D)
            
        Returns:
            ProcrustesOperator or None if first frame
        """
        if self.prev_pts is None:
            self.prev_pts = pts.copy()
            self.prev_state = State(pts)
            return None
            
        # Match points between frames
        matched_prev, matched_curr = self._match_points(self.prev_pts, pts)
        
        if len(matched_prev) < 3:
            return self._centroid_fallback(pts)
            
        # RANSAC-based robust estimation
        op = self._ransac_procrustes(matched_prev, matched_curr)
        
        if op is None:
            return self._centroid_fallback(pts)
            
        self.trajectory.append(op)
        self._update_cumulative(op)
        
        self.prev_pts = pts.copy()
        self.prev_state = State(pts)
        return op
        
    def _ransac_procrustes(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        n_iter: int = 20
    ) -> Optional[ProcrustesOperator]:
        """
        RANSAC-based robust Procrustes estimation.
        
        Repeatedly samples subsets and finds the transformation
        with the most inliers.
        """
        n_pts = len(pts1)
        if n_pts < 3:
            return None
            
        # Adaptive inlier threshold based on point spread
        spread = np.std(pts1)
        inlier_threshold = max(0.3, 0.2 * spread)
            
        best_op = None
        best_inliers = 0
        
        # Adaptive sample size (15% of points, min 3)
        sample_size = max(3, int(0.15 * n_pts))
        
        for _ in range(n_iter):
            # Random sample
            indices = np.random.choice(n_pts, sample_size, replace=False)
            
            # Fit on sample
            op = derive_procrustes_se3(State(pts1[indices]), State(pts2[indices]))
            
            if op is None:
                continue
                
            # Count inliers
            pred = (op.A @ pts1.T).T + op.t
            residuals = np.linalg.norm(pred - pts2, axis=1)
            n_inliers = np.sum(residuals < inlier_threshold)
            
            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_op = op
                
        # Refit on all inliers if we have a good model
        if best_op is not None and best_inliers > sample_size:
            pred = (best_op.A @ pts1.T).T + best_op.t
            residuals = np.linalg.norm(pred - pts2, axis=1)
            inlier_mask = residuals < inlier_threshold
            
            if np.sum(inlier_mask) >= 3:
                final_op = derive_procrustes_se3(
                    State(pts1[inlier_mask]),
                    State(pts2[inlier_mask])
                )
                if final_op is not None:
                    return final_op
                    
        return best_op
        
    def _match_points(self, pts1: np.ndarray, pts2: np.ndarray):
        """Match points between frames using Hungarian algorithm."""
        dist_matrix = cdist(pts1, pts2)
        
        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        
        # Filter by distance threshold
        matched_prev = []
        matched_curr = []
        
        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] < self.match_threshold:
                matched_prev.append(pts1[r])
                matched_curr.append(pts2[c])
        
        return np.array(matched_prev) if matched_prev else np.empty((0, pts1.shape[1])), \
               np.array(matched_curr) if matched_curr else np.empty((0, pts2.shape[1]))
               
    def _centroid_fallback(self, pts: np.ndarray) -> Optional[ProcrustesOperator]:
        """Fallback: estimate drift from centroid shift."""
        curr_centroid = np.mean(pts, axis=0)
        prev_centroid = np.mean(self.prev_pts, axis=0)
        
        D = len(curr_centroid)
        drift = curr_centroid - prev_centroid
        
        # Create identity rotation with translation
        op = ProcrustesOperator(np.eye(D), drift)
        
        self.trajectory.append(op)
        self._update_cumulative(op)
        
        self.prev_pts = pts.copy()
        self.prev_state = State(pts)
        return op
        
    def _update_cumulative(self, op: ProcrustesOperator):
        """Update cumulative transformation."""
        if self.cumulative_transform is None:
            self.cumulative_transform = op
        else:
            # Compose: T_new = T_op @ T_prev
            A_new = op.A @ self.cumulative_transform.A
            t_new = op.A @ self.cumulative_transform.t + op.t
            self.cumulative_transform = ProcrustesOperator(A_new, t_new)
            
    def get_total_drift(self) -> np.ndarray:
        """Get total accumulated drift vector."""
        if self.cumulative_transform is None:
            return np.zeros(3)
        return self.cumulative_transform.t
        
    def get_trajectory(self) -> List[np.ndarray]:
        """Get trajectory as list of translation vectors."""
        return [op.t for op in self.trajectory]
        
    def reset(self):
        """Reset tracker state."""
        self.prev_state = None
        self.prev_pts = None
        self.trajectory.clear()
        self.cumulative_transform = None
