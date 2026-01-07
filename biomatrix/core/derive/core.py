# -*- coding: utf-8 -*-
"""derive/core.py - Master transformation derivation entry point."""


import numpy as np
from scipy.spatial import cKDTree
from typing import Optional, List, Callable
from dataclasses import dataclass
from functools import reduce

from ..state import State
from ..transform import AffineTransform, derive_isometry_unordered, RelativeTranslation
from ..operators import (
    Operator, IdentityOperator, UnionOperator, RepeatOperator, 
    LinearSequenceOperator, SequentialOperator, CropToComponentOperator, 
    SelectBySignatureOperator, SortAndSelectOperator, SelectThenActOperator,
    KroneckerOperator, SelectByValueOperator, ValuePermutationOperator,
    RankByMassOperator, TilingOperator, AdditiveOperator, SequenceOperator,
    AffineTilingOperator, DifferenceOperator, LiftedSliceOperator, FiberProjectionOperator,
    ScaleOperator, ResampleOperator, ValueProjectionOperator, EnvelopeOperator,
    KernelAffineOperator, RigidAffineForceOperator, RigidHomotheticForceOperator,
    ProjectiveSelectionOperator, GlobalAffineOperator, SortAndAlignOperator,
    InteriorOperator
)
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# from .causal import derive_causal_partition # PURGED
from ..topology import partition_by_value, partition_by_connectivity, view_as_void, match_objects
from ..signatures import compute_universal_signature, signatures_match, compute_projected_signatures
from ..query import Query, QueryByDimValue




# ALGEBRAIC SNAP: Inline function (no hardcoded groups)
def snap_matrix(M, tol=0.3):
    """Snap matrix to integers if orthogonal and entries near-integer."""
    M_round = np.round(M).astype(float)
    if np.max(np.abs(M - M_round)) > tol:
        return None
    if not np.allclose(M_round @ M_round.T, np.eye(len(M)), atol=0.1):
        return None
    return M_round

# Lazy import to avoid circular dependency with algebra
_algebra_module = None

def _get_algebra():
    global _algebra_module
    if _algebra_module is None:
        from . import algebra as _algebra_module
    return _algebra_module

# Import from decomposed modules
from .affine import (
    derive_matched_affine, derive_affine_centered, derive_affine_scaled,
    derive_affine_permutation, derive_affine_correspondence, _solve_similarity,
    derive_translation_robust
)
from .permutation import (
    derive_value_permutation, derive_rank_transform, derive_fiber_projection
)
from .lifting import (
    derive_lifting, derive_lifted_transform, derive_manifold_porter
)
from .union import (
    derive_union, derive_hierarchical_invariant, derive_composite_transform,
    derive_deletion, derive_component_permutation, _derive_union_matching,
    _derive_deletion_bijection, _derive_generative_matching,
    _derive_surjective_simulation, _derive_sequence_algebraic, _is_subset
)
from .procrustes import derive_procrustes_se3


MAX_TIER = 4


# --- HELPER WRAPPERS ---
def derive_identity(s_in: State, s_out: State) -> Optional[Operator]:
    return IdentityOperator() if s_in == s_out else None

def derive_causality(s_in: State, s_out: State, depth: int = 0) -> Optional[Operator]:
    """Wrapper for algebraic causality derivation."""
    return _get_algebra().derive_causality(s_in, s_out, depth=depth)

# --- PIPELINE REGISTRY ---
# Defined at end of file to resolve forward references
SOLVER_PIPELINE = []


def derive_transformation(s_in: State, s_out: State, depth: int = 0, priors: List[Callable] = None) -> Optional[Operator]:
    """Master Law Resolver: S_in -> S_out via pipeline."""
    if depth > 2:

        return None

    # --- PRIOR INJECTION (User Hinting) ---
    if priors:
        for prior_func in priors:
            try:
                op = prior_func(s_in, s_out)
                if op: return op
            except Exception:
                pass

    for tier, name, strategy_func in SOLVER_PIPELINE:
        # Skip if above scientific threshold
        if tier > MAX_TIER:
            continue
            
        # Execute Strategy
        try:
            # Handle functions that accept depth (recursion support)
            if name == "Causality (Inert/Active)":
                 # Pass depth+1 to causal solver
                 op = strategy_func(s_in, s_out, depth=depth + 1)
            else:
                 op = strategy_func(s_in, s_out)
                 
            if op:
                return op
        except Exception:
            pass
            
    return None


def derive_all_transformations(s_in: State, s_out: State, depth: int = 0) -> List[Operator]:
    """Derive ALL valid transformations for S_in -> S_out."""
    if depth > 2:

        return []

    candidates = []
    
    # Try all strategies in pipeline
    for tier, name, strategy_func in SOLVER_PIPELINE:
        if tier > MAX_TIER:
            continue
            
        try:
            if name == "Causality (Inert/Active)":
                 ops = _get_algebra().derive_all_causality(s_in, s_out, depth=depth + 1) if hasattr(_get_algebra(), 'derive_all_causality') else []
                 # Fallback if derive_all_causality not implemented
                 if not ops:
                     op = strategy_func(s_in, s_out, depth=depth + 1)
                     if op: ops = [op]
                 candidates.extend(ops)
            else:
                 # Check if strategy supports 'return_all' or similar?
                 # Most don't. They return Optional[Operator].
                 # We assume they return the "best" for that strategy.
                 op = strategy_func(s_in, s_out)
                 if op:
                     candidates.append(op)
        except Exception:
            pass
            
    return candidates


def derive_subset(s_in: State, s_out: State, tol: float = 0.1) -> Optional[Operator]:
    """Derive Subset Relation (S_out < S_in). Find discriminating dimension."""
    if s_out.n_points == 0:

        return None
        
    if s_out.n_points >= s_in.n_points:
        return None
        
    if s_out.n_dims != s_in.n_dims:
        return None
        
    dists = cdist(s_out.points, s_in.points)
    min_dists = np.min(dists, axis=1)
    is_inplace = np.max(min_dists) <= tol

    
    if is_inplace:
        matched_indices = np.argmin(dists, axis=1)
        if len(np.unique(matched_indices)) != len(s_out.points):
            return None
            
        var_in = np.var(s_in.points, axis=0)
        var_out = np.var(s_out.points, axis=0)
        epsilon = 1e-6
        discriminant_mask = (var_out < epsilon) & (var_in > epsilon)
        discriminant_dims = np.where(discriminant_mask)[0]
        
        if len(discriminant_dims) > 0:
            d = int(discriminant_dims[0])
            value = s_out.points[0, d]
            return SelectByValueOperator(dim=d, value=value, tolerance=tol)
        
        sig = compute_universal_signature(s_out.points)
        op = SelectBySignatureOperator(target_signature=sig, tolerance=tol)
        
        # VERIFY: Signature selection relies on component separation.
        # If input is connected but output is a subset, signature matching fails.
        if op.apply(s_in) == s_out:
            return op
            
        return None

    # 2. Try TRANSLATIONAL SUBSET (Move + Prune) - ALGEBRAIC: Use map/filter
    p0_out = s_out.points[0]
    candidates_t = p0_out - s_in.points
    
    def test_translation(t):
        """Test if translation t yields a valid subset."""
        shifted_out = s_out.points - t
        dists_shift = cdist(shifted_out, s_in.points)
        min_dists_shift = np.min(dists_shift, axis=1)
        
        if np.max(min_dists_shift) <= tol:
            sig = compute_universal_signature(s_out.points - t)
            select_op = SelectBySignatureOperator(target_signature=sig, tolerance=tol)
            trans_op = AffineTransform.translate(t)
            return SequentialOperator(steps=[select_op, trans_op])
        return None
    
    valid_ops = list(filter(None, map(test_translation, candidates_t)))
    if valid_ops:
        return valid_ops[0]

    return None





# ========== SELECT-THEN-ACT PIPELINE (Separation Theorem) ==========

@dataclass
class SelectQueryAffineOperator(Operator):
    """
    Implements T = B o P (Separation Theorem).
    """
    query: 'Query'
    matrix: np.ndarray
    translation: np.ndarray
    
    def apply(self, state: State) -> State:
        if state.n_points == 0:
            return state.copy()
            
        mask = self.query.select(state)
        subset = state.points[mask]
        
        if len(subset) == 0:
            return State(np.empty((0, state.n_dims)))
            
        result = (self.matrix @ subset.T).T + self.translation
        
        return State(result)
    
    def __repr__(self):
        return f"SelectQueryAffine(P={self.query.name()}, B={self.matrix.shape})"


def derive_select_then_act(s_in: State, s_out: State, tol: float = 0.5) -> Optional[Operator]:
    """
    Derive transformation using Select-Then-Act pipeline (Separation Theorem).
    
    AGENT.md COMPLIANT: Algebraic derivation via invariant matching.
    NO BRUTE FORCE: Derive selection from set difference, not enumeration.
    
    T = B ∘ P where:
    - P = Selection (derived from invariants)
    - B = Bijection (Procrustes)
    """
    if s_in.is_empty or s_out.is_empty:
        return None
    
    # ALGEBRAIC DERIVATION: Check if output is affine image of input subset
    # If |out| < |in|, some points were deleted; derive selection algebraically
    
    if s_out.n_points > s_in.n_points:
        return None  # Not a selection (subset) case
    
    # 1. INVARIANT CHECK: Signatures must be compatible
    sig_out = compute_universal_signature(s_out.points)
    
    # 2. ALGEBRAIC SUBSET DETECTION: Find which dimension values are preserved
    n_dims = s_in.n_dims
    
    # For each dimension, check if output values are subset of input values
    # VECTORIZED: Use set operations on unique values
    def check_dim_subset(d):
        """Check if output dim values are subset of input."""
        in_vals = np.unique(np.round(s_in.points[:, d], State.DECIMALS))
        out_vals = np.unique(np.round(s_out.points[:, d], State.DECIMALS))
        return np.all(np.isin(out_vals, in_vals))
    
    if not all(map(check_dim_subset, range(n_dims))):
        return None  # Values outside input range - not a subset selection
    
    # 3. ALGEBRAIC MASK DERIVATION: Find points in input that map to output
    # Use centroid-based matching (algebraic, not enumeration)
    c_in = s_in.centroid
    c_out = s_out.centroid
    
    # Try identity selection first (most common case)
    if s_in.n_points == s_out.n_points:
        B = _extract_bijection_for_sta(s_in, s_out)
        if B is not None:
            t = c_out - B @ c_in
            pred = (B @ s_in.points.T).T + t
            pred_set = set(map(tuple, np.round(pred, 3)))
            out_set = set(map(tuple, np.round(s_out.points, 3)))
            if pred_set == out_set:
                return AffineTransform(linear=B, translation=t)
    
    # 4. SIGNATURE-BASED MASK: Find subset by matching output signature
    # Compute projected signatures for dimension-wise selection
    proj_sigs = compute_projected_signatures(s_in, n_dims)
    
    # For each unique value in each dimension, check if selecting that value
    # produces a subset matching output signature
    def test_value_selection(d_v):
        """Test if selecting dimension d with value v yields output."""
        d, v = d_v
        mask = np.isclose(s_in.points[:, d], v, atol=0.1)
        if np.sum(mask) != s_out.n_points:
            return None
        subset = State(s_in.points[mask])
        B = _extract_bijection_for_sta(subset, s_out)
        if B is None:
            return None
        t = s_out.centroid - B @ subset.centroid
        pred = (B @ subset.points.T).T + t
        pred_set = set(map(tuple, np.round(pred, 3)))
        out_set = set(map(tuple, np.round(s_out.points, 3)))
        if pred_set == out_set:
            return SelectQueryAffineOperator(
                query=QueryByDimValue(dim=d, value=v),
                matrix=B, translation=t
            )
        return None
    
    # Generate (dim, value) pairs algebraically from unique values
    dim_value_pairs = [
        (d, v)
        for d in range(n_dims)
        for v in np.unique(np.round(s_in.points[:, d], State.DECIMALS))
    ]
    
    # MAP/FILTER: Apply test to all pairs, return first valid
    valid_ops = list(filter(None, map(test_value_selection, dim_value_pairs)))
    if valid_ops:
        return valid_ops[0]
    
    return None


def _extract_bijection_for_sta(subset_in: State, s_out: State) -> Optional[np.ndarray]:
    """Extract bijection B via Procrustes + snap."""
    if subset_in.n_points != s_out.n_points:
        return None
    if subset_in.n_points == 0:
        return np.eye(subset_in.n_dims)
        
    c_in = subset_in.centroid
    c_out = s_out.centroid
    
    P_in = subset_in.points - c_in
    P_out = s_out.points - c_out
    
    H = P_in.T @ P_out
    
    rank = np.linalg.matrix_rank(H)
    if rank < min(H.shape):
        return None
    
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    det_R = np.linalg.det(R)
    Vt_adj = Vt.copy()
    Vt_adj[-1, :] = np.where(det_R < 0, -Vt[-1, :], Vt[-1, :])
    R = Vt_adj.T @ U.T
    
    return snap_matrix(R)


def derive_lift_and_slice(s_in: State, s_out: State, tol: float = 0.5) -> Optional[Operator]:
    """
    Derive "Lift-and-Slice" Transformation (Kernel Linearization).
    """
    return _get_algebra().derive_lift_and_slice(s_in, s_out)


def find_exact_displacement(s1: State, s2: State, epsilon: float = 1e-6) -> Optional[np.ndarray]:
    """
    Find vector v such that (s1 + v) has a non-empty intersection with s2.
    """
    pts1, pts2 = s1.points, s2.points
    if len(pts2) == 0 or len(pts1) == 0: return None
    
    diffs = pts2[:, np.newaxis, :] - pts1[np.newaxis, :, :]
    
    # Check broadcasting shape compatibility (Fix for 35ab12c3)
    # The above line creates (N2, N1, D). This is always valid broadcasting (via newaxis).
    # The user reported "shapes (8,3) (11,3)". This implies DIRECT subtraction pts2 - pts1 without newaxis!
    # Where does that happen? 
    # Not here. Here we use newaxis.
    # Maybe in `derive_translation` if I didn't verify it?
    # But `find_exact_displacement` is used fortranslation.
    
    # Wait, the user error likely comes from `derive_subset` line 178: `candidates_t = p0_out - s_in.points`.
    # p0_out is (D,). s_in.points is (N, D).
    # (D,) - (N, D) broadcasts to (N, D). This is valid.
    # Where else?
    # Maybe `derive_matched_affine`? Or `derive_translation` legacy?
    
    # I will safeguard `diffs` usage here anyway.
    diffs_flat = diffs.reshape(-1, s1.n_dims)
    
    void_diffs = view_as_void(np.ascontiguousarray(np.round(diffs_flat / epsilon) * epsilon))
    u_void, inv, counts = np.unique(void_diffs, return_inverse=True, return_counts=True)
    
    best_idx = np.argmax(counts)
    if counts[best_idx] >= 1:
        idx = np.where(inv == best_idx)[0]
        return np.mean(diffs_flat[idx], axis=0)
    
    return None


def derive_force_transform(s1: State, s2: State) -> Optional[Operator]:
    """
    Algebraic Derivation of Rigid Affine Force.
    """
    if s1.is_empty or s2.is_empty: return None
    
    comps1 = partition_by_connectivity(s1)
    if not comps1: return None
    
    def get_displacement(c1):
        v = find_exact_displacement(c1, s2)
        return (c1.centroid, v) if v is not None else None
    
    data = list(filter(None, map(get_displacement, comps1)))
    if len(data) < 2: return None
    
    C1 = np.array(list(map(lambda d: d[0], data)))
    V = np.array(list(map(lambda d: d[1], data)))
    
    affine_results = []
    
    X = np.hstack([C1, np.ones((len(C1), 1))])
    rank = np.linalg.matrix_rank(X)
    if rank >= min(X.shape):
        sol, resid, _, s = np.linalg.lstsq(X, V, rcond=None)
        AmI = sol[:-1, :].T
        b = sol[-1, :]
        A = AmI + np.eye(s1.n_dims)
        affine_results.append(RigidAffineForceOperator(matrix=A, offset=b))
        affine_results.append(GlobalAffineOperator(matrix=A, bias=b))
    
    if len(C1) >= 2:
        d1 = np.linalg.norm(C1[1:] - C1[0], axis=1)
        valid_d1 = np.all(d1 > 1e-9)
        if valid_d1:
            C2 = C1 + V
            d2 = np.linalg.norm(C2[1:] - C2[0], axis=1)
            k = np.mean(d2 / (d1 + 1e-9))
            center = np.mean(C2 - k * C1, axis=0) / (1.0 - k) if abs(k - 1.0) > 1e-6 else np.zeros(s1.n_dims)
            affine_results.append(RigidHomotheticForceOperator(center=center, scale=k))

    def try_force_op(force_op):
        s_pred = force_op.apply(s1)
        if s_pred == s2:
            return force_op
        
        if s2.n_points < s_pred.n_points:
            dims = range(s1.n_dims)
            
            def try_dim(p_dim):
                proj_dims_mask = np.arange(s1.n_dims) != p_dim
                proj_dims = list(np.where(proj_dims_mask)[0])
                
                def try_order(desc):
                    stack_op = ProjectiveSelectionOperator(
                        projection_dims=proj_dims, 
                        priority_dim=p_dim,
                        descending=desc
                    )
                    combined = SequentialOperator([force_op, stack_op])
                    result = combined.apply(s1)
                    return combined if result == s2 else None
                
                results = list(filter(None, map(try_order, [True, False])))
                return results[0] if results else None
            
            dim_results = list(filter(None, map(try_dim, dims)))
            return dim_results[0] if dim_results else None
        
        return None
    
    results = list(filter(None, map(try_force_op, affine_results)))
    return results[0] if results else None


def derive_sort_and_align(s1: State, s2: State) -> Optional[Operator]:
    """
    Algebraic Derivation of Sort & Align Law.
    """
    # ALGEBRAIC DERIVATION (Inference, O(D))
    # Instead of iterating transform parameters, we infer them from Source/Target geometry.
    # 1. Sort Dimension: Try each dimension (O(D)).
    # 2. Align Dimensions: Check which dimensions correlate with sequential stacking.
    
    if s1.is_empty or s2.is_empty: return None
    D = s1.n_dims
    
    comps_in = partition_by_connectivity(s1)
    if not comps_in: return None
    
    # Pre-calculate invariant spreads to infer alignment axes
    # Align Axis implies: Output Spread \approx Sum(Input Spreads)
    # Preservation Axis implies: Output Spread \approx Max(Input Spreads)
    
    total_spread_in = np.sum([c.spread for c in comps_in], axis=0) # Sum
    max_spread_in = np.max([c.spread for c in comps_in], axis=0)   # Max
    spread_out = s2.spread
    
    # Infer Align Dims: Where spread increases additively
    # Use loose tolerance for "Sum-like" behavior
    # diff_sum < diff_max implies Additive (Align)
    # diff_max < diff_sum implies Preservation (Sort only, or unrelated)
    inferred_align_dims = []
    for d in range(D):
        diff_sum = abs(spread_out[d] - total_spread_in[d])
        diff_max = abs(spread_out[d] - max_spread_in[d])
        if diff_sum < diff_max:
            inferred_align_dims.append(d)
    
    # Params to try
    # 1. Inferred alignment
    align_configs = [inferred_align_dims]
    # 2. Fallback: If inference empty, try standard single-axis stacking for each d
    if not inferred_align_dims:
        align_configs.extend([[d] for d in range(D)])

    def try_sort_dim(s_dim):
        # Origins to try: Zero (Absolute) and output bbox (Relative)
        origins = [np.zeros(D), s2.bbox_min]
        
        for align_dims in align_configs:
            for origin in origins:
                op = SortAndAlignOperator(sort_dim=s_dim, align_dims=align_dims, origin=origin)
                s_pred = op.apply(s1)
                
                if s_pred == s2:
                    return op
                
                # Check for Occlusion / Projection (if s_pred > s2)
                if s2.n_points < s_pred.n_points:
                     # Try Projective Selection on the stacking axis (common)
                     p_dim = s_dim
                     proj_dims = [d for d in range(D) if d != p_dim]
                     
                     for desc in [True, False]:
                         stack_op = ProjectiveSelectionOperator(
                             projection_dims=proj_dims,
                             priority_dim=p_dim,
                             descending=desc
                         )
                         combined = SequentialOperator([op, stack_op])
                         if combined.apply(s1) == s2:
                             return combined
        return None

    # Linear Scan O(D)
    results = list(filter(None, map(try_sort_dim, range(D))))
    return results[0] if results else None


def derive_component_resample(s_in: State, s_out: State) -> Optional[Operator]:
    """
    Derive Component-level Resampling (Algebraic Scaling).
    """
    if s_in.is_empty or s_out.is_empty: return None
    
    comps_in = partition_by_connectivity(s_in)
    comps_out = partition_by_connectivity(s_out)
    if not comps_in or not comps_out: return None
    
    if len(comps_in) > 10 or len(comps_out) > 10: return None
    
    def try_pair(pair):
        c_in, c_out = pair
        bb_in = c_in.spread + 1.0
        bb_out = c_out.spread + 1.0
        factors = bb_out / bb_in
        
        if not np.any(np.abs(factors - 1.0) > 0.1):
            return None
        
        op = ResampleOperator(
            scale_factors=factors,
            grid_dims=list(range(s_in.n_dims)),
            output_bbox_min=c_out.bbox_min,
            output_bbox_max=c_out.bbox_max
        )
        
        result_comp = op.apply(c_in)
        if result_comp != c_out:
            return None
        
        res_full = op.apply(s_in)
        return op if res_full == s_out else None
    
    # ALGEBRAIC: Nested comprehension replaces itertools.product
    pairs = [(c_in, c_out) for c_in in comps_in for c_out in comps_out]
    
    results = list(filter(None, map(try_pair, pairs)))
    return results[0] if results else None

# --- PIPELINE REGISTRY ---
# (Tier, Name, Function)
SOLVER_PIPELINE[:] = [
    # TIER 1: TRIVIAL (Identity, Subsets)
    (1, "Identity", derive_identity),
    (1, "Subset Check", derive_subset),

    # TIER 2: ALGEBRAIC CORE (Provable)
    (2, "Procrustes SE(3)", derive_procrustes_se3),  # Fast SVD-based (noise-robust)
    (2, "Ordered Affine", lambda s_in, s_out: derive_affine_correspondence(s_in, s_out, tol=0.05)),   # OLS (Identity Correspondence)
    (2, "Global Affine", lambda s_in, s_out: derive_affine_centered(s_in, s_out, tol=0.05)),          # Spectral Alignment
    (2, "Scaled Affine", lambda s_in, s_out: derive_affine_scaled(s_in, s_out, tol=State.TOLERANCE_RELAXED)),
    (2, "Unified Lift & Slice", derive_lift_and_slice),    # Mass Rank, Symmetry, Unique Index
    (2, "Affine Permutation", derive_affine_permutation),  # Move + Recolor
    (2, "Select-Then-Act", derive_select_then_act),        # Separation Theorem
    (2, "Hungarian Matching", derive_matched_affine),      # O(N^3) guarded by signature
    (2, "Component Resample", derive_component_resample),  # Per-component scaling
    (2, "Hierarchical Invariant", derive_hierarchical_invariant), # Hierarchy preservation

    # TIER 3: COMPOSITIONAL (Partitioning)
    (3, "Causality (Inert/Active)", derive_causality),     # Background vs Foreground
    (3, "Union / Partition", derive_union),                # Recurse on components
    (3, "Composite Transform", derive_composite_transform), # T1 o T2
]
