# -*- coding: utf-8 -*-
"""
Fiber Linker - DNA fiber tracking across frames

Links fiber segments between frames using topological signatures.
N-dimensional agnostic.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment




from biomatrix.core.state import State
from biomatrix.core.topology import partition_by_connectivity
from biomatrix.core.signatures import compute_universal_signature


class FiberLinker:
    """
    DNA fiber tracking across frames.
    
    Uses topological signatures to match fiber segments
    between consecutive frames.
    """
    
    def __init__(self, max_displacement: float = 5.0):
        """
        Args:
            max_displacement: Maximum allowed fiber movement between frames
        """
        self.max_displacement = max_displacement
        self.prev_fibers = None
        self.fiber_ids = {}
        self.next_id = 0
        self.tracks = {}  # fiber_id -> list of centroids
        
    def process(self, pts: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Process frame and link fibers.
        
        Args:
            pts: Point cloud (N x D)
            
        Returns:
            Dict mapping fiber_id to fiber points
        """
        # Partition into fiber components
        fibers = partition_by_connectivity(State(pts))
        
        if not fibers:
            return {}
            
        # First frame: assign new IDs
        if self.prev_fibers is None:
            self.prev_fibers = fibers
            result = {}
            for f in fibers:
                fid = self._new_id()
                result[fid] = f.points
                self.tracks[fid] = [f.centroid]
            return result
            
        # Match fibers between frames
        matches = self._match_fibers(self.prev_fibers, fibers)
        
        # Build result with tracked IDs
        result = {}
        matched_prev = set()
        matched_curr = set()
        
        for prev_idx, curr_idx in matches:
            fid = list(self.tracks.keys())[prev_idx] if prev_idx < len(self.tracks) else self._new_id()
            result[fid] = fibers[curr_idx].points
            self.tracks[fid].append(fibers[curr_idx].centroid)
            matched_prev.add(prev_idx)
            matched_curr.add(curr_idx)
            
        # New fibers (unmatched current)
        for i, f in enumerate(fibers):
            if i not in matched_curr:
                fid = self._new_id()
                result[fid] = f.points
                self.tracks[fid] = [f.centroid]
                
        self.prev_fibers = fibers
        return result
        
    def _match_fibers(
        self, 
        prev: List[State], 
        curr: List[State]
    ) -> List[Tuple[int, int]]:
        """Match fibers using Hungarian algorithm on centroid distances."""
        if not prev or not curr:
            return []
            
        # Compute centroid distance matrix
        prev_centroids = np.array([f.centroid for f in prev])
        curr_centroids = np.array([f.centroid for f in curr])
        
        dist_matrix = cdist(prev_centroids, curr_centroids)
        
        # Mask large displacements
        dist_matrix[dist_matrix > self.max_displacement] = 1e6
        
        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        
        # Filter matches exceeding threshold
        matches = [
            (r, c) for r, c in zip(row_ind, col_ind)
            if dist_matrix[r, c] < self.max_displacement
        ]
        
        return matches
        
    def _new_id(self) -> int:
        """Generate new fiber ID."""
        fid = self.next_id
        self.next_id += 1
        return fid
        
    def get_tracks(self) -> Dict[int, np.ndarray]:
        """Get all fiber trajectories."""
        return {fid: np.array(track) for fid, track in self.tracks.items()}
        
    def reset(self):
        """Reset tracker state."""
        self.prev_fibers = None
        self.fiber_ids.clear()
        self.next_id = 0
        self.tracks.clear()
