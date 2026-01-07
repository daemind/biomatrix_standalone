# -*- coding: utf-8 -*-
"""derive/affine.py - Affine transform derivation (Procrustes, scaling, permutation)."""


import numpy as np
from typing import Optional, List
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from functools import reduce

from ..state import State
from ..transform import AffineTransform, RelativeTranslation
from ..topology import partition_by_connectivity, match_objects

from ..operators import (
    Operator, SequentialOperator, ValuePermutationOperator
)

# Import from sibling modules
from .permutation import derive_value_permutation

# Standard Numerical Epsilon for Division stability
NUMERICAL_EPSILON = 1e-9



def derive_matched_affine(s_in: State, s_out: State) -> Optional[Operator]:
    """Hungarian algorithm for optimal point matching + affine."""
    if s_in.is_empty or s_out.is_empty:
        return None
        
        
    comps_in = partition_by_connectivity(s_in)
    comps_out = partition_by_connectivity(s_out)
    
    X = s_in.points
    Y = s_out.points
    N = X.shape[0]
    
    if len(comps_in) <= 1 or len(comps_out) <= 1:
        diff = X[:, None, :] - Y[None, :, :]
        cost_matrix = np.sum(diff ** 2, axis=2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        Y_matched = Y[col_ind]
        
    else:
        matched_pairs = match_objects(comps_in, comps_out)
        
        if len(matched_pairs) != len(comps_in):
             return None
        
        centroids_in = np.array(list(map(lambda p: p[0].centroid, matched_pairs)))
        centroids_out = np.array(list(map(lambda p: p[1].centroid, matched_pairs)))
        
        X = centroids_in
        Y_matched = centroids_out
        pass

    
    # Now check if there's a uniform transform
    # Compute displacement field
    displacements = Y_matched - X  # (N, D)
    
    # Check if all displacements are the same (pure translation)
    disp_mean = displacements.mean(axis=0)
    is_translation = np.allclose(displacements, disp_mean, atol=State.EPSILON)
    
    if is_translation:
        # Pure translation - use RELATIVE form for generalization
        # Store TARGET centroid instead of fixed offset
        target_centroid = Y_matched.mean(axis=0)
        return RelativeTranslation(target_centroid=target_centroid)
    
    # Check if it's a rigid transform (same rotation for all points)
    # Use SVD on the matched pairs
    centroid_src = X.mean(axis=0)
    centroid_tgt = Y_matched.mean(axis=0)
    
    X_centered = X - centroid_src
    Y_centered = Y_matched - centroid_tgt
    
    H = X_centered.T @ Y_centered
    
    # ALGEBRAIC: SVD with rank check guard
    rank = np.linalg.matrix_rank(H)
    if rank < H.shape[0]:
        return None  # Degenerate case
    
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Handle reflection
    det_R = np.linalg.det(R)
    Vt_adj = Vt.copy()
    Vt_adj[-1, :] = np.where(det_R < 0, -Vt[-1, :], Vt[-1, :])
    R = Vt_adj.T @ U.T
    
    # Compute translation
    t = centroid_tgt - R @ centroid_src
    
    # Optional SNAP translation (REMOVED: Pure floats)
    # t_snapped = snap_translation(t)
    # t = t_snapped if t_snapped is not None else t
    
    # Build and verify
    op = AffineTransform(linear=R, translation=t)
    result = op.apply(s_in)
    
    return op if result == s_out else None


def _solve_similarity(p1: np.ndarray, p2: np.ndarray, 
                      q1: np.ndarray, q2: np.ndarray) -> Optional[AffineTransform]:
    """N-dimensional similarity solver: G(p1)=q1, G(p2)=q2."""
    u = p2 - p1

    v = q2 - q1
    
    # 2. Scale derivation
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    
    if norm_u < NUMERICAL_EPSILON: 
        return None  # Singularity (points are identical)
    
    scale = norm_v / norm_u
    
    u_hat = u / norm_u
    v_hat = v / norm_v
    
    H = np.outer(u_hat, v_hat)
    
    rank = np.linalg.matrix_rank(H)
    if rank == 0:
        return None
    
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    det_R = np.linalg.det(R)
    Vt_adj = Vt.copy()
    Vt_adj[-1, :] = np.where(det_R < 0, -Vt[-1, :], Vt[-1, :])
    R = Vt_adj.T @ U.T
    
    linear = scale * R
    translation = q1 - (linear @ p1)
    
    return AffineTransform(linear=linear, translation=translation)


def derive_affine_centered(s_in: State, s_out: State, tol: float = 1e-4) -> Optional[AffineTransform]:
    """Derive affine via SVD (Kabsch/Procrustes)."""
    same_cardinality = (s_in.n_points == s_out.n_points)

    valid_input = not (s_in.is_empty or s_out.is_empty)
    
    if not (same_cardinality and valid_input):
        return None
    
    X = s_in.points
    Y = s_out.points
    N, D = X.shape
    
    centroid_src = np.mean(X, axis=0)
    centroid_tgt = np.mean(Y, axis=0)
    
    src_centered = X - centroid_src
    tgt_centered = Y - centroid_tgt
    
    def sort_by_polar_signature(points: np.ndarray) -> tuple:
        """Sort by N-dimensional polar coordinates."""
        N_pts, D = points.shape
        radii = np.linalg.norm(points, axis=1)
        
        angles = np.zeros((N_pts, max(1, D - 1)))
        eps = 1e-12
        # angle_i = arctan2(x_{i+1}, |x_0:i|)
        cumulative_norm = np.abs(points[:, 0:1])  # Start with |x_0|
        
        # Functional accumulation of angles
        # Use functools.reduce to update cumulative_norm and compute angles
        # state = (cumulative_norm, angles_list)
        
        def compute_angle_step(acc, dim_idx):
            cum_norm, angs = acc
            safe_denom = np.maximum(cum_norm[:, 0], eps)
            new_ang = np.arctan2(points[:, dim_idx + 1], safe_denom)
            new_cum = np.sqrt(cum_norm ** 2 + points[:, dim_idx + 1:dim_idx + 2] ** 2)
            return (new_cum, angs + [new_ang])
        
        # Initial state: cumulative_norm = |x_0|, angles = []
        initial_norm = np.abs(points[:, 0:1])
        _, angle_list = reduce(compute_angle_step, range(D - 1), (initial_norm, []))
        
        # Stack columns
        angles = np.column_stack(angle_list) if angle_list else np.zeros((N_pts, 0))
        
        # Create sort key: (radius, angles...)
        # Lexsort uses last key as primary, so reverse order
        sort_keys = tuple(angles[:, i] for i in range(angles.shape[1] - 1, -1, -1)) + (radii,)
        sort_indices = np.lexsort(sort_keys)
        
        return points[sort_indices], sort_indices
    
    # Try polar ordering first
    src_ordered, src_idx = sort_by_polar_signature(src_centered)
    tgt_ordered, tgt_idx = sort_by_polar_signature(tgt_centered)
    
    # === STEP 3: Scale Derivation ===
    var_src = np.var(src_ordered)
    var_tgt = np.var(tgt_ordered)
    
    if var_src < NUMERICAL_EPSILON:
        scale = 1.0
    else:
        scale = np.sqrt(var_tgt / var_src)
    
    # Helper to solve SVD and build op - ALGEBRAIC VERSION
    def _solve_svd_inner(src, tgt, scale_val, translation_base):
        H = src.T @ tgt
        rank = np.linalg.matrix_rank(H)
        if rank == 0:
            return None
        
        U, Sigma, Vt = np.linalg.svd(H)
        Rotation = Vt.T @ U.T
        
        det_R = np.linalg.det(Rotation)
        Vt_adj = Vt.copy()
        Vt_adj[-1, :] = np.where(det_R < 0, -Vt[-1, :], Vt[-1, :])
        Rotation = Vt_adj.T @ U.T
        
        Linear = scale_val * Rotation
        Translation = centroid_tgt - (Linear @ centroid_src)
        return AffineTransform(linear=Linear, translation=Translation)

    # Try 1: Polar Sort (with Cyclic Shifts for robustness against branch cuts)
    # Rotation changes indices cyclically if points cross the branch cut.
    # We test all N shifts if strictly necessary, or a heuristic subset.
    
    LIMIT_CYCLIC_CHECKS = 30 # Limit for performance on large sets
    
    # Try 1: Direct Match
    op = _solve_svd_inner(src_ordered, tgt_ordered, scale, centroid_tgt)
    if op:
        result = op.apply(s_in)
        if result.n_points == s_out.n_points:
            dists = cdist(result.points, s_out.points)
            if np.max(np.min(dists, axis=1)) < tol and np.max(np.min(dists, axis=0)) < tol:
                return op
    
    # Cyclic Shifts (Standard Polar Sort Weakness Fix)
    # ALGEBRAIC PURITY: Test all cyclic permutations (O(N) hypothesis check)
    # Essential for symmetric objects (e.g., Pure Rigid Body Cube)
    if N > 1:
        tgt_rolled = tgt_ordered
        # Iterate full cycle to guarantee finding the correct correspondence branch
        for i in range(1, N):
            tgt_rolled = np.roll(tgt_rolled, 1, axis=0) # Shift down
            
            # Cyclic shift needed for symmetric objects (same radius points)
            op = _solve_svd_inner(src_ordered, tgt_rolled, scale, centroid_tgt)
            if op:
                result = op.apply(s_in)
                if result.n_points == s_out.n_points:
                    dists = cdist(result.points, s_out.points)
                    if np.max(np.min(dists, axis=1)) < tol:
                         return op

    # Try 2: Distance Spectrum Sort (Fallback for Symmetries)
    # Allows resolving symmetric ambiguities where polar angle is unstable
    def sort_by_distance_spectrum(points: np.ndarray) -> np.ndarray:
        """
        Sort by distance spectrum (lexicographic order of distances to all other points).
        Invariant under isometry.
        """
        D_mat = cdist(points, points)
        D_sorted = np.sort(D_mat, axis=1)[:, 1:]
        keys = tuple(D_sorted[:, i] for i in range(D_sorted.shape[1] - 1, -1, -1))
        radii = np.linalg.norm(points, axis=1)
        keys = keys + (radii,)
        
        idx = np.lexsort(keys)
        return points[idx]

    src_dist = sort_by_distance_spectrum(src_centered)
    tgt_dist = sort_by_distance_spectrum(tgt_centered)
    
    op_dist = _solve_svd_inner(src_dist, tgt_dist, scale, centroid_tgt)
    if op_dist:
        result = op_dist.apply(s_in)
        if result.n_points == s_out.n_points:
             dists = cdist(result.points, s_out.points)
             relaxed_tol = 1e-2 # Higher tolerance for complex symmetries
             if np.max(np.min(dists, axis=1)) < relaxed_tol and np.max(np.min(dists, axis=0)) < relaxed_tol:
                 return op_dist

    return None


def derive_affine_scaled(s_in: State, s_out: State, tol: float = 1e-2) -> Optional[AffineTransform]:
    """Scale-invariant affine derivation using Frobenius norm."""
    if s_in.is_empty or s_out.is_empty:

        return None
    if s_in.n_points != s_out.n_points:
        return None
        
    X = s_in.points
    Y = s_out.points
    
    # Centering
    centroid_src = np.mean(X, axis=0)
    centroid_tgt = np.mean(Y, axis=0)
    
    src_centered = X - centroid_src
    tgt_centered = Y - centroid_tgt
    
    # Frobenius Norm Scale
    norm_src = np.linalg.norm(src_centered, 'fro')
    norm_tgt = np.linalg.norm(tgt_centered, 'fro')
    
    if norm_src < NUMERICAL_EPSILON:
        scale = 1.0
    else:
        scale = norm_tgt / norm_src
    
    # Normalize for rotation extraction
    src_normalized = src_centered / (norm_src + NUMERICAL_EPSILON)
    tgt_normalized = tgt_centered / (norm_tgt + NUMERICAL_EPSILON)
    
    # SVD for Rotation - ALGEBRAIC VERSION
    H = src_normalized.T @ tgt_normalized
    
    rank = np.linalg.matrix_rank(H)
    if rank == 0:
        return None
    
    U, Sigma, Vt = np.linalg.svd(H)
    Rotation = Vt.T @ U.T
    
    # Reflection correction
    det_R = np.linalg.det(Rotation)
    Vt_adj = Vt.copy()
    Vt_adj[-1, :] = np.where(det_R < 0, -Vt[-1, :], Vt[-1, :])
    Rotation = Vt_adj.T @ U.T
    
    # Build Transform
    Linear = scale * Rotation
    Translation = centroid_tgt - (Linear @ centroid_src)
    
    op = AffineTransform(linear=Linear, translation=Translation)
    
    # Validate
    result = op.apply(s_in)
    
    if result.n_points != s_out.n_points:
        return None
        
    dists = cdist(result.points, s_out.points)
    min_dists1 = np.min(dists, axis=1)
    min_dists2 = np.min(dists, axis=0)
    
    max_dist = max(np.max(min_dists1), np.max(min_dists2))
    return op if max_dist < tol else None


def derive_affine_permutation(s_in: State, s_out: State, tol: float = 0.1) -> Optional[Operator]:
    """Derive affine + value permutation: T(x) = (A(spatial), sigma(value))."""
    if s_in.n_points != s_out.n_points or s_in.n_points == 0:

        return None
        
    N, D = s_in.points.shape
    if D < 2:
        return None # Need at least 1 spatial + 1 value dim
        
    
    # Strategy: Iterate over candidate Value Dimension 'd' using map/filter
    # Filter valid dimensions where (D \ {d}) affine fit works
    
    def try_dim_permutation(d: int) -> Optional[Operator]:
        # 1. Project to Spatial Dims
        mask = np.ones(D, dtype=bool)
        mask[d] = False
        
        pts_in_spatial = s_in.points[:, mask]
        pts_out_spatial = s_out.points[:, mask]
        
        # Create temporary states
        s_in_s = State(pts_in_spatial)
        s_out_s = State(pts_out_spatial)
        
        # 2. Derive Affine on Spatial (Recursive call to centered affine)
        op_spatial = derive_affine_centered(s_in_s, s_out_s, tol=tol)
        
        if not op_spatial or not isinstance(op_spatial, AffineTransform):
            return None
            
        # 3. Lift spatial affine to full D dims (Identity on d)
        mat_s = op_spatial.linear
        bias_s = op_spatial.translation
        
        mat_d = np.eye(D)
        bias_d = np.zeros(D)
        
        # Block assignment via fancy indexing
        spatial_indices = np.where(mask)[0]
        # np.ix_ creates mesh for block assignment
        r_idx, c_idx = np.ix_(spatial_indices, spatial_indices)
        mat_d[r_idx, c_idx] = mat_s
        bias_d[spatial_indices] = bias_s
        
        op_lifted = AffineTransform(linear=mat_d, translation=bias_d)
        
        # 4. Apply lifted affine and check permutation on d
        s_pred = op_lifted.apply(s_in)
        op_perm = derive_value_permutation(s_pred, s_out, tol=tol)
        
        if op_perm and isinstance(op_perm, ValuePermutationOperator):
             # Validate: Permutation must affect dimension d!
             has_nontrivial = len(op_perm.permutation_maps) > d and len(op_perm.permutation_maps[d]) > 0
             if has_nontrivial:
                  return SequentialOperator([op_lifted, op_perm])
        return None

    # Apply to all dimensions and pick first valid one
    valid_ops = list(filter(None, map(try_dim_permutation, range(D))))
    
    return valid_ops[0] if valid_ops else None
                          
    return None


def derive_affine_subset(s_in: State, s_out: State, tol: float = 0.1) -> Optional[Operator]:
    """
    Derive Affine Transform for Subsets (A(S_in) \subset S_out).
    Algebraic Similitude Matching:
    1. Normalize Scale (Project to Unit Size).
    2. Solve Correspondence (Hungarian).
    3. Reconstruct Transform.
    """
    if s_in.is_empty or s_out.is_empty: return None
    
    X = s_in.points
    Y = s_out.points
    N_in = X.shape[0]
    
    # Pre-calculate Centroids & Spread
    c_X = np.mean(X, axis=0)
    c_Y = np.mean(Y, axis=0)
    
    X_centered = X - c_X
    Y_centered = Y - c_Y
    
    std_X = np.std(X_centered) # Frobenius norm / sqrt(N*D)
    std_Y = np.std(Y_centered)
    
    # Avoid division by zero
    scale_X = std_X if std_X > NUMERICAL_EPSILON else 1.0
    scale_Y = std_Y if std_Y > NUMERICAL_EPSILON else 1.0
    
    # 1. Project to Normalized Invariant Space (Unit Scale)
    X_norm = X_centered / scale_X
    Y_norm = Y_centered / scale_Y
    
    # 2. Match Structures (Hungarian on Normalized Clouds)
    # This matches shapes regardless of their absolute scale
    cost = cdist(X_norm, Y_norm, metric='sqeuclidean')
    row_ind, col_ind = linear_sum_assignment(cost)
    
    X_matched = X[row_ind]
    Y_matched = Y[col_ind]
    
    # 3. Fit Affine (SVD) on Matched Pairs in Original Scale
    c_X_m = np.mean(X_matched, axis=0)
    c_Y_m = np.mean(Y_matched, axis=0)
    X_c = X_matched - c_X_m
    Y_c = Y_matched - c_Y_m
    
    norm_X = np.linalg.norm(X_c, 'fro')
    norm_Y = np.linalg.norm(Y_c, 'fro')
    
    # Analytic Scale Recovery
    scale_final = (norm_Y / norm_X) if norm_X > NUMERICAL_EPSILON else 1.0
    
    H = (X_c / (norm_X + NUMERICAL_EPSILON)).T @ (Y_c / (norm_Y + NUMERICAL_EPSILON))
    U, S_val, Vt = np.linalg.svd(H)
    Rotation = Vt.T @ U.T
    
    if np.linalg.det(Rotation) < 0:
        Vt[-1, :] *= -1
        Rotation = Vt.T @ U.T
        
    Linear = scale_final * Rotation
    Translation = c_Y_m - (Linear @ c_X_m)
    
    op = AffineTransform(linear=Linear, translation=Translation)
    
    # Verification
    pred = op.apply(State(X))
    dists = np.linalg.norm(pred.points - Y_matched, axis=1)
    
    if np.max(dists) < tol:
        return op
        
    return None



def derive_affine_correspondence(s_in: State, s_out: State, tol: float = 1e-4) -> Optional[AffineTransform]:
    """
    Derive Affine Transform assuming Identity Correspondence (Ordered Points).
    Uses OLS to find Y = X A^T + b.
    Fast and effective if points are already ordered.
    """
    # Sanity check
    if s_in.n_points != s_out.n_points or s_in.n_points < 2: 
        return None
        
    X = s_in.points
    Y = s_out.points
    
    # solve Y = X A^T + b
    # Center data to separate A and b
    centroid_in = np.mean(X, axis=0)
    centroid_out = np.mean(Y, axis=0)
    
    X_c = X - centroid_in
    Y_c = Y - centroid_out
    
    # OLS: A^T = (X_c.T X_c)^-1 X_c.T Y_c
    # A = Y_c.T X_c (X_c.T X_c)^-1
    # Use lstsq for robustness
    
    try:
        # np.linalg.lstsq solves Ax = B. We want X_c @ M = Y_c => M = lstsq(X_c, Y_c)
        # M is A.T
        M, residuals, rank, s = np.linalg.lstsq(X_c, Y_c, rcond=None)
        
        A = M.T
        b = centroid_out - A @ centroid_in
        
        op = AffineTransform(linear=A, translation=b)
        
        # Verify
        result = op.apply(s_in)
        dists = np.linalg.norm(result.points - Y, axis=1)
        if np.max(dists) < tol:
            return op
            
    except np.linalg.LinAlgError:
        return None
        
    return None


def derive_translation_robust(s_in: State, s_out: State) -> Optional[AffineTransform]:
    """
    Derive Robust Translation (Mass-Point Approximation).
    Ignores internal deformation/rotation.
    T(x) = x + (Centroid_out - Centroid_in).
    """
    if s_in.n_points == 0 or s_out.n_points == 0:
        return None
        
    c_in = s_in.centroid
    c_out = s_out.centroid
    
    t = c_out - c_in
    D = s_in.n_dims
    
    return AffineTransform(linear=np.eye(D), translation=t)

__all__ = [
    'derive_matched_affine',
    'derive_affine_centered',
    'derive_affine_scaled', 
    'derive_affine_permutation',
    'derive_affine_subset',
    'derive_affine_correspondence',
    'derive_translation_robust',
    '_solve_similarity'
]

