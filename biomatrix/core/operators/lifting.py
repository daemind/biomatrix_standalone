# -*- coding: utf-8 -*-
"""
operators/lifting.py - Lifting and Slice Operators

Contains operators for N+1 dimensional lifting and slicing transformations.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from functools import reduce

from .base import Operator, IdentityOperator
from ..state import State


@dataclass
class LiftedSliceOperator(Operator):
    """
    Lift-and-Slice Operator (Universal Porter).
    
    Algebraic Pipeline:
    1. LIFT: Augment R^D -> R^{D+1} with auxiliary 'altitude' label.
    2. TRANSFORM: Apply inner_op to the manifold in R^{D+1}.
    3. CUT: Intersect the manifold with hyperplanes H_l (values or range).
    4. PROJECT: Drop auxiliary dimension, returning to R^D.
    
    AGENT.md: Pure algebraic, N-dimensional agnostic.
    """
    lifter: str = 'mass_rank'
    inner_op: Operator = field(default_factory=IdentityOperator)
    slice_dim: int = -1
    original_dims: int = 0
    slice_values: Optional[List[float]] = None
    slice_range: Optional[Tuple[float, float]] = None
    pre_slice_values: Optional[List[float]] = None
    anchor_point: Optional[np.ndarray] = None
    
    # Lift Parameters
    period: Optional[np.ndarray] = None
    topology_mode: str = 'dist_boundary'
    kernel: Optional[np.ndarray] = None
    scale_matrix: Optional[np.ndarray] = None
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
        
        X = state.points
        n_pts, n_dims = X.shape
        
        # LIFT: Compute labels based on lifter strategy
        labels = self._compute_lift_labels(state, X, n_pts, n_dims)
        
        # Build manifold in R^{D+1}
        X_lifted = np.column_stack([X, labels])
        
        # EXTEND if slice_range specified
        X_base = self._extend_manifold(X_lifted, labels, n_pts, n_dims)
        
        if len(X_base) == 0:
            return State(np.empty((0, self.original_dims if self.original_dims > 0 else n_dims)))
        
        # PRE-SLICE filtering
        X_base = self._apply_pre_slice(X_base)
        
        if len(X_base) == 0:
            return State(np.empty((0, self.original_dims if self.original_dims > 0 else n_dims)))
        
        # TRANSFORM
        s_transformed = self._apply_transform(State(X_base), n_dims)
        X_prime = s_transformed.points
        
        # CUT (post-slice filtering)
        X_cut = self._apply_cut(X_prime)
        
        if len(X_cut) == 0:
            return State(np.empty((0, self.original_dims if self.original_dims > 0 else n_dims)))
        
        # PROJECT back to R^D
        X_out = self._project(X_cut, n_dims)
        
        return State(X_out)
    
    def _compute_lift_labels(self, state, X, n_pts, n_dims):
        """Compute lift labels based on lifter strategy."""
        from ..derive.lifting_kernels import (
            lift_identity, lift_kronecker, lift_topological,
            lift_distance_rank, lift_unique_index, lift_poly2, lift_symmetry,
            lift_connectivity_features
        )
        
        # Dispatch to shared kernels
        # Note: Shared kernels return [X, Features]. We only want Features (Labels).
        # We need to slice the result.
        
        def wrap(kernel_func):
            # Wrapper to extract only the NEW columns (features)
            # Some kernels take 3 args (X, state, params), others 2 (X, params)
            def _wrapped():
                # Try 3 args
                try:
                    res = kernel_func(X, state, self.__dict__)
                except TypeError:
                    # Try 2 args
                    res = kernel_func(X, self.__dict__)
                
                # Exclude original D columns
                return res[:, n_dims:]
            return _wrapped

        lifter_dispatch = {
            'identity': lambda: np.zeros((n_pts, 1)), # Identity adds nothing? Or 0?
            'component_label': wrap(lift_connectivity_features), # Mapped to connectivity
            'connectivity': wrap(lift_connectivity_features),
            'mass_rank': wrap(lift_distance_rank), # Legacy
            'distance_rank': wrap(lift_distance_rank),
            'unique_index': wrap(lift_unique_index),
            'kronecker': wrap(lift_kronecker),
            'topology': wrap(lift_topological),
            'modulo': lambda: (np.floor(X / (self.period[0] + 1e-9)).sum(axis=1)[:,None] if self.period is not None else np.zeros((n_pts,1)))
        }
        
        handler = lifter_dispatch.get(self.lifter, lambda: np.zeros((n_pts, 1)))
        features = handler()
        
        # Feature handling: Expecting 1D labels for visualization/extrusion logic?
        # The existing code expects 'labels' to be likely 1D array for 'X_lifted = np.column_stack([X, labels])'
        # If features has >1 col, X_lifted will have D+L dims.
        # Ensure flattened if L=0? Features (N, L).
        
        if features.ndim == 1:
            return features
        if features.shape[1] == 1:
            return features.ravel()
        
        # If multiple features, returning all?
        # But `_extend_manifold` and `_apply_pre_slice` seem to assume `labels` implies the Slice Dimension.
        # `X_lifted = np.column_stack([X, labels])`.
        # If labels is (N, 2), then D+2.
        # Slice logic typically targets the LAST dimension.
        # So we should probably return the primary feature or all?
        # For 'LiftedSlice', the slice happens on the lifted dims.
        # If we return multiple, we might need to specify which one to slice.
        # Given current architecture `slice_dim=-1`.
        return features[:,-1] # Return last feature as the 'Lift Label'? Or handle multi-dim?
        # For now, let's assume last feature is the primary 'Index/Rank'.

    
    def _extend_manifold(self, X_lifted, labels, n_pts, n_dims):
        """Extend manifold via extrusion if slice_range specified."""
        if self.slice_range is None:
            return X_lifted
        
        z_min, z_max = self.slice_range
        steps = np.arange(np.ceil(z_min - 1e-6), np.floor(z_max + 1e-6) + 1)
        
        if len(steps) == 0:
            return np.empty((0, n_dims + 1))
        
        N_base = len(X_lifted)
        K = len(steps)
        X_ext = np.zeros((N_base * K, n_dims + 1))
        X_ext[:, :-1] = np.repeat(X_lifted[:, :-1], K, axis=0)
        
        # Extrusion strategy depends on lifter
        extrusion_dispatch = {
            'modulo': lambda: np.repeat(labels, K) + np.tile(steps, N_base),
            'unique_index': lambda: np.repeat(labels * K, K) + np.tile(steps, N_base),
            'kronecker': lambda: np.tile(steps, N_base),
        }
        default_extrusion = lambda: np.repeat(labels, K) + np.tile(steps, N_base)
        
        X_ext[:, -1] = extrusion_dispatch.get(self.lifter, default_extrusion)()
        return X_ext
    
    def _apply_pre_slice(self, X_base):
        """Apply pre-transform Z filtering."""
        if self.pre_slice_values is None:
            return X_base
        pre_vals_arr = np.round(self.pre_slice_values, State.DECIMALS)
        labels_pre = np.round(X_base[:, -1], State.DECIMALS)
        return X_base[np.isin(labels_pre, pre_vals_arr)]
    
    def _apply_transform(self, s_base, n_dims):
        """Apply transformation in lifted space."""
        if self.lifter == 'kronecker' and self.kernel is not None and self.scale_matrix is not None:
            pts = s_base.points
            x_in = pts[:, :-1]
            k_indices = pts[:, -1].astype(int)
            k_indices = np.clip(k_indices, 0, len(self.kernel) - 1)
            
            # Dimension guard
            n_dims_in = x_in.shape[1] if x_in.ndim > 1 else 1
            if self.scale_matrix.shape[1] != n_dims_in:
                return s_base
            
            x_scaled = x_in @ self.scale_matrix.T
            offsets = self.kernel[k_indices]
            X_prime = np.column_stack([x_scaled + offsets, k_indices])
            return State(X_prime)
        
        return self.inner_op.apply(s_base)
    
    def _apply_cut(self, X_prime):
        """Apply post-transform hyperplane filtering."""
        if self.slice_values is None:
            return X_prime
        slice_vals_arr = np.round(self.slice_values, State.DECIMALS)
        labels_prime = np.round(X_prime[:, -1], State.DECIMALS)
        return X_prime[np.isin(labels_prime, slice_vals_arr)]
    
    def _project(self, X_cut, n_dims):
        """Project back to original dimension."""
        if self.slice_dim == -1 or self.slice_dim == n_dims:
            return X_cut[:, :-1]
        mask_out = np.ones(X_cut.shape[1], dtype=bool)
        mask_out[self.slice_dim] = False
        return X_cut[:, mask_out]


@dataclass
class FiberProjectionOperator(Operator):
    """
    Fiber Projection Operator: Affine map on fiber dimensions.
    
    Algebraic Structure:
    T = Id_{invariant_dims} ⊗ π_{fiber_dims}
    
    Where π is a translation on the fiber subspace.
    Invariant dimensions remain unchanged.
    
    AGENT.md: Pure tensor algebra, N-dimensional agnostic.
    """
    invariant_dims: List[int]
    fiber_dims: List[int]
    translation: np.ndarray
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
        
        new_points = state.points.copy()
        
        # ALGEBRAIC: Vectorized translation via index arrays
        fiber_arr = np.array(self.fiber_dims)
        trans_arr = np.array(self.translation)
        
        n_fibers = min(len(fiber_arr), len(trans_arr))
        valid_fibers = fiber_arr[:n_fibers]
        valid_trans = trans_arr[:n_fibers]
        
        dim_mask = valid_fibers < state.n_dims
        dims_to_update = valid_fibers[dim_mask]
        trans_to_apply = valid_trans[dim_mask]
        
        new_points[:, dims_to_update] = new_points[:, dims_to_update] + trans_to_apply
        
        return State(new_points)


# Export all
__all__ = ['LiftedSliceOperator', 'FiberProjectionOperator']
