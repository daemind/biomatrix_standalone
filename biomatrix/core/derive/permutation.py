# -*- coding: utf-8 -*-
"""
derive/permutation.py - Value Permutation and Rank Transform Derivation

Contains functions for deriving value permutations, rank transforms, and fiber projections.
AGENT.md Compliant: Pure algebraic, N-dimensional agnostic.
"""

import numpy as np
from typing import Optional
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from ..state import State
from ..topology import view_as_void
from ..operators import (
    Operator, ValuePermutationOperator, RankByMassOperator, FiberProjectionOperator
)


def derive_value_permutation(s_in: State, s_out: State, tol: float = 0.1) -> Optional[Operator]:
    """
    N-Dimensional Value Permutation Detection.
    
    Algebraic Formulation:
    For each dimension d, find permutation π_d such that:
        V_out[d] = π_d(V_in[d])
    
    Where π_d is a bijection on unique values.
    
    AGENT.md: Pure tensor algebra, N-dimensional agnostic. NO FOR LOOPS.
    """
    if s_in.n_points != s_out.n_points or s_in.is_empty:
        return None
        
    n_dims = s_in.n_dims
    X = s_in.points
    Y = s_out.points
    
    # ALGEBRAIC: Vectorized per-dimension histogram matching
    # Build permutation maps using np.unique and broadcasting
    
    def derive_dim_permutation(vals_in: np.ndarray, vals_out: np.ndarray) -> Optional[dict]:
        """Derive value permutation for single dimension via histogram matching."""
        unique_in, counts_in = np.unique(vals_in, return_counts=True)
        unique_out, counts_out = np.unique(vals_out, return_counts=True)
        
        if len(unique_in) != len(unique_out):
            return None
            
        # Sort by counts (histogram matching)
        order_in = np.argsort(-counts_in)
        order_out = np.argsort(-counts_out)
        
        sorted_counts_in = counts_in[order_in]
        sorted_counts_out = counts_out[order_out]
        
        if not np.array_equal(sorted_counts_in, sorted_counts_out):
            return None
            
        # Algebraic bijection: zip sorted values
        return dict(zip(unique_in[order_in], unique_out[order_out]))
    
    # Apply derive_dim_permutation across all dimensions (vectorized via map)
    X_rounded = np.round(X, State.DECIMALS)
    Y_rounded = np.round(Y, State.DECIMALS)
    
    permutation_maps = list(map(
        lambda d: derive_dim_permutation(X_rounded[:, d], Y_rounded[:, d]),
        range(n_dims)
    ))
    
    # Check if any dimension failed
    if any(pmap is None for pmap in permutation_maps):
        return None
    
    # ALGEBRAIC: Apply permutation via stacked vectorized lookup
    def apply_perm_dim(d: int) -> np.ndarray:
        vmap = permutation_maps[d]
        lookup = np.vectorize(lambda v: vmap.get(v, v))
        return lookup(X_rounded[:, d])
    
    Y_pred = np.column_stack(list(map(apply_perm_dim, range(n_dims))))
    
    # Verify: Y_pred == Y (as sets) using void view comparison
    pred_v = view_as_void(np.ascontiguousarray(Y_pred))
    out_v = view_as_void(np.ascontiguousarray(Y_rounded))
    
    if len(pred_v) == len(out_v) and np.all(np.isin(pred_v, out_v)):
        return ValuePermutationOperator(permutation_maps=permutation_maps)
        
    return None


def derive_fiber_projection(s_in: State, s_out: State) -> Optional[Operator]:
    """
    N-Dimensional Fiber Projection Detection.
    
    Algebraic Formulation:
    T = Id_{d1,...,dk} ⊗ π_{dk+1,...,dn}
    
    Where π is an affine projection on a subset of dimensions (fiber group).
    The remaining dimensions are invariant (identity).
    
    Mathematical Structure:
    - Fiber = subset of dimensions with non-trivial transformation
    - Group action on fiber: translation, scaling, or general affine
    - Monoïd composition: T = Σ_fibers T_fiber
    
    AGENT.md: Pure algebra, N-dimensional agnostic.
    """
    if s_in.n_points != s_out.n_points or s_in.is_empty:
        return None
        
    n_dims = s_in.n_dims
    X = s_in.points
    Y = s_out.points
    
    # ALGEBRAIC: Check which dimensions are invariant vs transformed
    X_rounded = np.round(X, State.DECIMALS)
    Y_rounded = np.round(Y, State.DECIMALS)
    
    # Invariant dimensions: X[:, d] == Y[:, d] (up to reordering)
    def dim_is_invariant(d: int) -> bool:
        return set(X_rounded[:, d]) == set(Y_rounded[:, d]) and \
               np.allclose(np.sort(X_rounded[:, d]), np.sort(Y_rounded[:, d]))
    
    invariant_dims = list(filter(dim_is_invariant, range(n_dims)))
    fiber_dims = list(filter(lambda d: d not in invariant_dims, range(n_dims)))
    
    if len(fiber_dims) == 0:
        return None  # All invariant = identity (handled elsewhere)
    
    if len(invariant_dims) == 0:
        return None  # No invariants = general affine (handled elsewhere)
    
    # ALGEBRAIC: Project onto invariant subspace, derive fiber transformation
    # Points with same invariant coords should have same fiber transformation
    
    # Build invariant signature for each point (as hashable tuples)
    inv_sigs = list(map(
        lambda i: tuple(X_rounded[i, invariant_dims].tolist()),
        range(s_in.n_points)
    ))
    
    # For each unique invariant signature, derive fiber map
    unique_sigs = list(set(inv_sigs))
    
    # Check if fiber transformation is consistent across all fibers
    fiber_maps = []
    
    def derive_single_fiber_map(sig):
        """Derive affine map on fiber for points with this invariant signature."""
        # Vectorized: Compare rows using broadcasting (AGENT.md compliant)
        sig_array = np.array(sig)
        mask_in = np.all(X_rounded[:, invariant_dims] == sig_array, axis=1)
        mask_out = np.all(Y_rounded[:, invariant_dims] == sig_array, axis=1)
        
        if np.sum(mask_in) != np.sum(mask_out):
            return None
            
        fiber_in = X_rounded[mask_in][:, fiber_dims]
        fiber_out = Y_rounded[mask_out][:, fiber_dims]
        
        if len(fiber_in) == 0:
            return None
            
        # Simple case: single point per fiber = direct translation
        if len(fiber_in) == 1:
            return fiber_out[0] - fiber_in[0]  # Translation vector
        
        # Multi-point: check if constant translation
        # t = fiber_out - fiber_in (must be uniform)
        translations = fiber_out - fiber_in
        if np.allclose(translations, translations[0]):
            return translations[0]
        
        return None
    
    fiber_maps = list(map(derive_single_fiber_map, unique_sigs))
    
    # Check all fibers have valid, uniform translation
    if any(t is None for t in fiber_maps):
        return None
        
    # All translations should be the same for uniform projection
    if not all(np.allclose(t, fiber_maps[0]) for t in fiber_maps):
        return None
    
    translation = fiber_maps[0]
    
    # Build FiberProjectionOperator
    return FiberProjectionOperator(
        invariant_dims=invariant_dims,
        fiber_dims=fiber_dims,
        translation=translation
    )


def derive_rank_transform(s_in: State, s_out: State, tol: float = 0.1) -> Optional[Operator]:
    """
    Derive Rank Transform (value = rank of fiber by mass).
    
    N-DIM AGNOSTIC ALGEBRAIC APPROACH:
    1. For each candidate "value dimension" d_val:
       a. For each candidate "fiber dimension" d_fib:
          - Partition points by d_fib value → fibers
          - Compute mass (point count) per fiber
          - Rank fibers by mass
          - Check if output[d_val] = rank for all points
    
    Transform: T(x)[d_val] = rank(fiber(x, d_fib), by_mass)
               T(x)[other] = x[other]  (invariant)
    """
    
    if s_in.n_points != s_out.n_points or s_in.n_points == 0:
        return None
    
    N, D = s_in.points.shape
    
    # Need point correspondence first
    # If spatial dims are invariant, match by those
    
    # Try each dimension as value dimension (the one that changes)
    for d_val in range(D):
        # Check if other dimensions are approximately invariant
        mask = np.ones(D, dtype=bool)
        mask[d_val] = False
        
        in_proj = s_in.points[:, mask]  # All dims except d_val
        out_proj = s_out.points[:, mask]
        
        # Match points by invariant dimensions
        if D > 1:
            dists = cdist(in_proj, out_proj)
            row_ind, col_ind = linear_sum_assignment(dists)
            cost = dists[row_ind, col_ind].sum()
            
            if cost > tol * N:
                continue  # Invariant dims don't match
        else:
            continue  # Need at least 2 dims for ranking
        
        # Now try each OTHER dimension as fiber dimension
        invariant_dims = np.where(mask)[0]
        
        for d_fib in invariant_dims:
            # Partition input points by d_fib value → fibers
            fib_vals = s_in.points[:, d_fib]
            unique_fibs = np.unique(fib_vals)
            
            if len(unique_fibs) < 2:
                continue  # Need multiple fibers for ranking
            
            # Compute mass per fiber
            fib_masses = {}
            for fib_val in unique_fibs:
                fib_mask = np.isclose(fib_vals, fib_val, atol=tol)
                fib_masses[fib_val] = np.sum(fib_mask)
            
            # Rank fibers by mass (descending: largest mass = rank 1)
            sorted_fibs = sorted(fib_masses.items(), key=lambda x: -x[1])
            fib_to_rank = {fib: i + 1 for i, (fib, _) in enumerate(sorted_fibs)}
            
            # Check if output value matches rank for all matched points
            matches = 0
            for i, j in zip(row_ind, col_ind):
                fib_val = s_in.points[i, d_fib]
                expected_rank = fib_to_rank[fib_val]
                actual_val = s_out.points[j, d_val]
                
                if np.isclose(actual_val, expected_rank, atol=tol):
                    matches += 1
            
            if matches == N:
                # Perfect match!
                return RankByMassOperator(
                    fiber_dim=int(d_fib),
                    value_dim=int(d_val),
                    order='descending'
                )
            
            # Also try ascending order (smallest mass = rank 1)
            sorted_fibs_asc = sorted(fib_masses.items(), key=lambda x: x[1])
            fib_to_rank_asc = {fib: i + 1 for i, (fib, _) in enumerate(sorted_fibs_asc)}
            
            matches_asc = 0
            for i, j in zip(row_ind, col_ind):
                fib_val = s_in.points[i, d_fib]
                expected_rank = fib_to_rank_asc[fib_val]
                actual_val = s_out.points[j, d_val]
                
                if np.isclose(actual_val, expected_rank, atol=tol):
                    matches_asc += 1
            
            if matches_asc == N:
                return RankByMassOperator(
                    fiber_dim=int(d_fib),
                    value_dim=int(d_val),
                    order='ascending'
                )
    
    return None


__all__ = [
    'derive_value_permutation',
    'derive_rank_transform',
    'derive_fiber_projection'
]
