# -*- coding: utf-8 -*-
"""
operators/replication.py - Replication and Tiling Operators

Contains operators for pattern repetition, tiling, and Minkowski sums.
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

from .base import Operator
from ..state import State



@dataclass
class LinearSequenceOperator(Operator):
    """
    Chasles Iterative Sum: Sum_{k=0}^{n-1} τ(k*d)
    
    Arithmetic progression of translation operator.
    Places copies of the pattern at regular intervals.
    
    Output = Union of τ(k*d)(A) for k in [0, n-1]
    """
    step: np.ndarray  # Translation step d
    count: int        # Number of repetitions n
    
    def apply(self, state: State) -> State:
        # ALGEBRAIC: Broadcast translation using outer product (no loop)
        n_pts = state.n_points
        k_vals = np.arange(self.count)[:, np.newaxis]  # (count, 1)
        offsets = k_vals * self.step[np.newaxis, :]     # (count, step_dims)
        
        # Broadcast: for each k, add offset to all points
        step_dims = len(self.step)
        expanded = state.points[np.newaxis, :, :].repeat(self.count, axis=0)  # (count, n_pts, n_dims)
        expanded[:, :, :step_dims] += offsets[:, np.newaxis, :]  # Add offsets to spatial dims
        
        # Flatten to (count * n_pts, n_dims)
        new_points = expanded.reshape(-1, state.n_dims)
        
        return State(points=new_points)


@dataclass
class KroneckerOperator(Operator):
    """
    Simulates a Fractal Expansion (s_in x s_in).
    Replaces each point in s_in with a scaled copy of the template.
    """
    template: State
    scale: np.ndarray  # (n_dims,)
    pivot: Optional[np.ndarray] = None  # Center of expansion
    anchor_mode: str = 'bbox'  # 'bbox' or 'origin'

    def apply(self, state: State) -> State:
        # Self-referential: A Tensor A
        template = self.template if self.template is not None else state
        
        # Pure Tensor-based Kronecker Expansion (No loops, No If/Else)
        s_arr = np.atleast_1d(self.scale).astype(float)
        if s_arr.size == 1: 
            s_arr = np.full(state.n_dims, s_arr.item())
        
        # Resize scale to match n_dims
        scale_vec = np.ones(state.n_dims)
        copy_len = min(len(s_arr), state.n_dims)
        scale_vec[:copy_len] = s_arr[:copy_len]
        
        # ALGEBRAIC: Dictionary-based pivot resolution
        pivot_map = {
            'origin': np.zeros(state.n_dims),
            'centroid': state.centroid,
            'bbox_min': state.bbox_min,
        }
        p_vec = pivot_map.get(self.pivot, state.bbox_min) if isinstance(self.pivot, str) else (state.bbox_min if self.pivot is None else np.array(self.pivot))

        # Resize pivot to match n_dims
        pivot = np.zeros(state.n_dims)
        p_arr = np.atleast_1d(p_vec)
        copy_len = min(len(p_arr), state.n_dims)
        pivot[:copy_len] = p_arr[:copy_len]

        # Transform Anchors
        anchors_transformed = (state.points - pivot) * scale_vec + pivot

        # Prepare Template
        if self.anchor_mode == 'origin':
            template_relative = template.points
        else:
            template_relative = template.points - template.bbox_min

        # One-Shot Outer Sum
        expanded = anchors_transformed[:, np.newaxis, :] + template_relative[np.newaxis, :, :]
        
        # Reshape to final point cloud
        result_pts = expanded.reshape(-1, state.n_dims)
        
        return State(points=result_pts)


@dataclass
class RepeatOperator(Operator):
    """
    Repeat/Fractal operator: unified interface for repetition patterns.
    
    Delegates to:
        - KroneckerOperator for 'self' pattern (A Tensor A)
        - LinearSequenceOperator for 'sequence' pattern (Sum τ(k*d))
        - Grid tiling for 'grid' pattern
    """
    scale: Union[int, float, np.ndarray]
    pattern: str = 'self'  # 'self', 'sequence', 'grid'
    step: np.ndarray = None
    pivot: Union[np.ndarray, str, None] = None
    anchor_mode: str = 'origin'
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
        
        # Prepare dimension-agnostic scales
        if isinstance(self.scale, (int, float)):
            scales = np.array([float(self.scale)] * state.n_dims)
        else:
            scales = np.array(self.scale, dtype=float)
            
        # Guard: ensure scales match n_dims
        if len(scales) > state.n_dims:
            scales = scales[:state.n_dims]
        elif len(scales) < state.n_dims:
            scales = np.pad(scales, (0, state.n_dims - len(scales)), constant_values=1.0)
            
        # ALGEBRAIC: Pattern dispatch via dictionary (no if/elif chain)
        def apply_self():
            op = KroneckerOperator(template=None, scale=scales, pivot=self.pivot, anchor_mode=self.anchor_mode)
            return op.apply(state)
        
        def apply_sequence():
            if self.step is None: 
                return state.copy()
            count = int(np.max(scales))
            return LinearSequenceOperator(step=self.step, count=count).apply(state)
        
        def apply_grid():
            return self._apply_grid_pattern(state, scales)
        
        pattern_dispatch = {
            'self': apply_self,
            'sequence': apply_sequence,
            'grid': apply_grid,
        }
        
        return pattern_dispatch.get(self.pattern, lambda: state.copy())()
    
    def _apply_grid_pattern(self, state: State, scales: np.ndarray) -> State:
        """ALGEBRAIC Grid tiling using broadcasting."""
        if self.step is not None:
            step = np.array(self.step)
        else:
            step = state.spread + 1
            step = step * (scales > 0)
        
        n_tiles = np.maximum(1, scales.astype(int))
        
        # Generate grid coordinates using meshgrid (no explicit loop)
        ranges = [np.arange(n) for n in n_tiles]
        grid_coords = np.stack(np.meshgrid(*ranges, indexing='ij'), axis=-1).reshape(-1, len(n_tiles))
        
        # Compute offsets: (n_grid_pts, n_dims)
        offsets = grid_coords * step
        
        # Broadcast: state.points (n_pts, n_dims) + offsets (n_grid, n_dims)
        expanded = state.points[np.newaxis, :, :] + offsets[:, np.newaxis, :]
        
        # Flatten to (n_grid * n_pts, n_dims)
        new_points = expanded.reshape(-1, state.n_dims)
        
        return State(points=new_points)


@dataclass
class ReplicationOperator(Operator):
    """
    Replication Operator: Minkowski Sum (S_in (+) K).
    
    Algebraic Structure:
    S_out = { x + k | x ∈ S_in, k ∈ K }
    
    Generates K copies of the input state, translated by vectors in K.
    """
    kernel: np.ndarray  # (K, D) kernel vectors
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
            
        # ALGEBRAIC: Minkowski Sum via Broadcasting
        X = state.points  # (N, D)
        K = self.kernel   # (K, D)
        
        Y = X[:, np.newaxis, :] + K[np.newaxis, :, :]
        result_points = Y.reshape(-1, state.n_dims)
        
        return State(result_points)


@dataclass
class TilingOperator(Operator):
    """
    Algebraic Tiling: Repeated Translation.
    
    Output = Union_k (Input + t_k)
    
    Used for filling, pattern repetition, and generative tasks.
    """
    translations: np.ndarray  # (K, D) translation vectors
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
            
        # ALGEBRAIC: Broadcasting (no loops)
        X = state.points  # (N, D)
        T = self.translations  # (K, D)
        
        # Broadcast: (N, 1, D) + (1, K, D) -> (N, K, D) -> (N*K, D)
        expanded = X[:, np.newaxis, :] + T[np.newaxis, :, :]
        result_pts = expanded.reshape(-1, state.n_dims)
        
        # Remove duplicates 
        unique_pts = np.unique(result_pts, axis=0)
        
        return State(unique_pts)
    
    def __repr__(self):
        return f"Tiling(|T|={len(self.translations)})"


@dataclass
class ReflectTilingOperator(Operator):
    """
    Reflection + Tiling Composition.
    
    Output = Union_k (Reflect(Input) + t_k)
    """
    translations: np.ndarray
    reflection_signs: np.ndarray  # Signs for reflection per dimension
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
        
        # Reflect around centroid
        centroid = state.centroid
        reflected = centroid + (state.points - centroid) * self.reflection_signs
        
        # Apply tiling
        T = self.translations
        expanded = reflected[:, np.newaxis, :] + T[np.newaxis, :, :]
        result_pts = expanded.reshape(-1, state.n_dims)
        
        return State(np.unique(result_pts, axis=0))
    
    def __repr__(self):
        return f"ReflectTiling(signs={self.reflection_signs}, |T|={len(self.translations)})"


@dataclass
class AffineTilingOperator(Operator):
    """
    Affine Transform + Tiling Composition.
    
    Output = Union_k (Matrix @ (Input - Centroid) + Centroid + t_k)
    """
    translations: np.ndarray
    matrix: np.ndarray  # Linear transformation matrix
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
        
        # Guard: dimension mismatch
        if self.matrix.shape[0] != state.n_dims:
            return state.copy()
        
        # Apply affine centered on centroid
        centroid = state.centroid
        centered = state.points - centroid
        transformed = centered @ self.matrix.T + centroid
        
        # Apply tiling
        T = self.translations
        expanded = transformed[:, np.newaxis, :] + T[np.newaxis, :, :]
        result_pts = expanded.reshape(-1, state.n_dims)
        
        return State(np.unique(result_pts, axis=0))
    
    def __repr__(self):
        return f"AffineTiling(M={self.matrix.shape}, |T|={len(self.translations)})"


@dataclass
class MinkowskiSumOperator(Operator):
    """
    Minkowski Sum Operator (Convolutive Echo).
    
    Algebraic Definition:
    T(S) = S ⊕ K = {s + k | s ∈ S, k ∈ K}
    
    This generalizes "Projective Echo", "Stamp", and "Dilation".
    """
    kernel: State
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
        
        # ALGEBRAIC: Outer sum via broadcasting
        S = state.points  # (N, D)
        K = self.kernel.points  # (M, D)
        
        # (N, 1, D) + (1, M, D) -> (N, M, D)
        result = S[:, np.newaxis, :] + K[np.newaxis, :, :]
        result_pts = result.reshape(-1, state.n_dims)
        
        return State(np.unique(result_pts, axis=0))


@dataclass
class PartialLatticeOperator(Operator):
    """
    Generative Partial Lattice: Infinite Grid + Mask.
    
    Algebraic Definition:
    S_out = { x | x = x0 + sum(n_i v_i), Mask(x) }
    
    Used for non-bijective "filling" tasks where the output counts
    do not match input counts (e.g. Area Scaling).
    """
    basis: np.ndarray      # (D,) Diagonal basis steps
    origin: np.ndarray     # (D,) Anchor point
    mask_operator: Operator # The "Cookie Cutter" (e.g. InteriorOperator, EnvelopeOperator)
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None # (min, max) Explicit generation scope
    
    def apply(self, state: State) -> State:
        # 1. Determine Scope (Bounding Box of the Target Envelope)
        # We need to know "How much" grid to generate.
        # Usually, the mask_operator implies a scope (e.g. Convex Hull of S_in).
        # But we might need to assume a bounding box derived from S_in.
        
        # 1. Determine Scope (Bounding Box)
        if self.bounds is not None:
            bbox_min, bbox_max = self.bounds
        else:
            bbox_min = state.bbox_min
            bbox_max = state.bbox_max
        
        # Guard: Zero basis step avoid div by zero
        steps = self.basis.copy()
        steps[steps < 1e-6] = 1e-6 # Avoid divzero
        
        # 2. Generate Candidate Lattice (Super-Set)
        # Range of integers for each dim
        n_min = np.floor((bbox_min - self.origin) / steps).astype(int)
        n_max = np.ceil((bbox_max - self.origin) / steps).astype(int)
        
        # Create ranges
        ranges = [np.arange(start, end + 1) for start, end in zip(n_min, n_max)]
        
        # Meshgrid
        grids = np.meshgrid(*ranges, indexing='ij')
        
        if len(grids) == 0: # Empty grid (D=0?) or empty range
             return state.copy() # Should return empty state if empty range
             
        # Check if any range is empty
        if any(len(r) == 0 for r in ranges):
             return State(np.empty((0, state.n_dims)))
        
        stacked = np.stack([g.flatten() for g in grids], axis=-1) # (N_cand, D)
        
        # Convert to Coordinates
        candidates = self.origin + stacked * steps
        
        # 3. Apply Mask (Filtering)
        # We need to check which points satisfy the Mask.
        # For efficiency, we create a temporary State(candidates) and run the filter.
        
        cand_state = State(candidates)
        
        # If mask is an explicit filter (returns state), we use it.
        # e.g. SelectByValue, InteriorOperator
        filtered_state = self.mask_operator.apply(cand_state)
        
        return filtered_state

    def __eq__(self, other):
        if not isinstance(other, PartialLatticeOperator): return False
        
        # Array comparisons
        if not np.array_equal(self.basis, other.basis): return False
        if not np.array_equal(self.origin, other.origin): return False
        if self.mask_operator != other.mask_operator: return False
        
        # Bounds comparison (Tuple of arrays)
        if self.bounds is None and other.bounds is None: return True
        if self.bounds is None or other.bounds is None: return False
        
        # Both not None, compare elements
        if len(self.bounds) != len(other.bounds): return False
        for b1, b2 in zip(self.bounds, other.bounds):
             if not np.array_equal(b1, b2): return False
             
        return True

# Export all
__all__ = [
    'LinearSequenceOperator', 'KroneckerOperator', 'RepeatOperator',
    'ReplicationOperator', 'TilingOperator', 'ReflectTilingOperator',
    'AffineTilingOperator', 'MinkowskiSumOperator', 'PartialLatticeOperator'
]
