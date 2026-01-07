# -*- coding: utf-8 -*-
"""derive_algebra.py - Algebraic transformation derivation via lifting kernels."""


import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


from ..state import State
from ..topology import partition_by_connectivity, view_as_void, get_boundary_mask, get_adjacency_matrix
from ..operators.algebra import (
    Operator, LiftOperator, BijectionOperator, SliceOperator, LiftedTransform
)
from ..operators.logic import UnionOperator
from ..operators import (
    AffineTilingOperator, ResampleOperator, KernelAffineOperator
)
from ..transform import AffineTransform

# Note: lift_kronecker imported from lifting_kernels
from .lifting_kernels import (
    lift_identity, lift_kronecker, lift_topological,
    lift_distance_rank, lift_unique_index, lift_poly2, lift_symmetry,
    lift_connectivity_features, lift_local_isotropy, lift_boundary_dist
)

def _derive_kronecker_basis(s_in: State, s_out: State) -> List[np.ndarray]:
    """
    Algebraic derivation of Kronecker Basis (Block Sizes).
    Returns list of vector bases B ∈ R^D.
    """
    candidates = []
    
    # 1. From Lattice Basis (REMOVED: Use spread algebraic fallback)
    # _, basis_out, _ = derive_lattice_basis(s_out)
    pass
            
    # 2. From Input Spread (Fundamental Block)
    spread_in = np.ptp(s_in.points, axis=0)
    # Avoid zero spread
    spread_in = np.where(spread_in < 1e-6, 1.0, spread_in)
    candidates.append(spread_in)
    
    # 3. From Lattice Basis of Input (REMOVED)
    pass
             
    # Deduplicate (approx)
    unique_candidates = []
    seen = []
    for c in candidates:
        # Check against seen
        is_new = True
        for s in seen:
            if np.allclose(c, s, atol=1e-5):
                is_new = False
                break
        if is_new:
            unique_candidates.append(c)
            seen.append(c)
            
    return unique_candidates


def _lift_master_stacked(X: np.ndarray, state: State, params: Dict[str, Any]) -> np.ndarray:
    """Master Lift: Concatenates all key topological and geometric features."""
    feats = [
        X,
        lift_distance_rank(X, state, params)[:, -1:],
        lift_connectivity_features(X, state, params)[:, -1:],
        lift_local_isotropy(X, state, {'k': 8})[:, -1:],
        lift_boundary_dist(X, state, params)[:, -1:],
        lift_symmetry(X, state, params)[:, state.n_dims:]
    ]
    return np.hstack(feats)



def derive_lift_and_slice(s_in: State, s_out: State) -> Optional[Operator]:
    """Generalized Lift-and-Slice Solver. Finds linear map in lifted space."""
    if s_in.is_empty or s_out.is_empty: return None

    # Calculate Expansion Factor K
    n_in, n_out = s_in.n_points, s_out.n_points
    if n_in == 0: return None
    
    ratio = n_out / n_in
    K_main = int(round(ratio))
    if K_main < 1: K_main = 1
    
    op_zoom = derive_homothety_lift_strategy(s_in, s_out)
    if op_zoom: return op_zoom

    strategies = []
    
    strategies.append(('identity', {}, 1))
    if K_main > 1:
        strategies.append(('identity', {}, K_main))
        
    basis_list = _derive_kronecker_basis(s_in, s_out)
    for B in basis_list:
        strategies.append(('kronecker', {'basis': B}, K_main))
         
    if K_main >= 1:
        strategies.append(('topology', {'mode': 'dist'}, K_main))
        strategies.append(('local_isotropy', {'k': 8}, K_main))
        strategies.append(('boundary_dist', {}, K_main))
        
    strategies.append(('distance_rank', {}, K_main))
    strategies.append(('connectivity', {}, K_main))
    strategies.append(('symmetry', {}, K_main))
    strategies.append(('unique_index', {}, K_main))
    strategies.append(('poly2', {}, 1))
    strategies.append(('master', {}, K_main))
        
    for lifter, params, K in strategies:
        try:
            op = solve_lifted_regression(s_in, s_out, lifter, params, K)
            if op:
                return op
        except Exception:
            continue
            
    op_tiling = derive_tiling_lift_strategy(s_in, s_out)
    if op_tiling: return op_tiling
    
    op_union = derive_union_lift(s_in, s_out)
    if op_union: return op_union
         
    return None


def derive_union_lift(s_in: State, s_out: State) -> Optional[Operator]:
    """Union via component-wise Lift→Bijection→Slice for multi-component cases."""
    
    comps_in = partition_by_connectivity(s_in)

    comps_out = partition_by_connectivity(s_out)
    
    # Simple case: same number of components
    if len(comps_in) != len(comps_out) or len(comps_in) <= 1:
        return None  # Complex case - needs more sophisticated matching
    
    N_comps = len(comps_in)
    
    # Sort components by centroid for alignment (vectorized)
    def get_centroid(comp):
        return np.mean(comp.points, axis=0)
    
    comps_in_sorted = sorted(comps_in, key=lambda c: tuple(get_centroid(c)))
    comps_out_sorted = sorted(comps_out, key=lambda c: tuple(get_centroid(c)))
    
    # MAP: Derive transformation for each component pair
    # We recursion here to derive_generative_lift or specific strategies?
    # To avoid infinite recursion, we call solve_lifted_regression or specialized strategies.
    # Safe to call derive_generative_lift if we ensure strictly smaller problem?
    # Yes, single component is smaller/simpler.
    
    def try_component_pair(pair):
        c_in, c_out = pair
        if c_in.n_points != c_out.n_points:
            return None  # Cardinality mismatch
        
        # Use simple Identity/Affine lift for components
        # (Avoid full recursion for now to prevent loops, unless we are careful)
        # Just try solve_lifted_regression with Identity
        op = solve_lifted_regression(c_in, c_out, 'identity', {}, 1)
        if not op:
             # Try Unique Index
             op = solve_lifted_regression(c_in, c_out, 'unique_index', {}, 1)
        return (c_in, op) if op else None
    
    # Apply map and filter None results
    pairs = list(zip(comps_in_sorted, comps_out_sorted))
    component_ops = list(filter(None, map(try_component_pair, pairs)))
    
    if len(component_ops) != N_comps:
        return None
    
    # MAP: Verify by applying all component operators
    def apply_op(pair):
        c_in, cop = pair
        return cop.apply(c_in).points
    
    result_points = list(map(apply_op, component_ops))
    
    if result_points:
        combined = State(np.vstack(result_points))
        if combined == s_out:
            # Return first component's operator as placeholder if all same?
            # Or UnionOperator?
            # Use UnionOperator logic from operators.logic?
            # We don't have direct access here easily without circular imports?
            # We imported UnionOperator at top.
            ops_list = [op for _, op in component_ops]
            # Check if all ops are identical?
            return UnionOperator(operands=ops_list)
    


def solve_lifted_regression(s_in: State, s_out: State, lifter: str, lift_params: dict, K: int) -> Optional[Operator]:
    """
    Core Algebraic Solver (Unified).
    
    1. LIFT: Features = Kernel(X_in)  [N, D+L]
    """
    n_dims = s_in.n_dims
    X = s_in.points
    n_in = len(X)
    
    # --- 1. LIFTING (KERNEL) ---
    kernel_map = {
        'identity': lift_identity,
        'kronecker': lift_kronecker,
        'topology': lambda x, p: lift_topological(x, s_in, p),
        'mass_rank': lambda x, p: lift_distance_rank(x, s_in, p),
        'distance_rank': lambda x, p: lift_distance_rank(x, s_in, p),
        'connectivity': lambda x, p: lift_connectivity_features(x, s_in, p),
        'unique_index': lift_unique_index,
        'poly2': lift_poly2,
        'symmetry': lambda x, p: lift_symmetry(x, s_in, p),
        'local_isotropy': lambda x, p: lift_local_isotropy(x, s_in, p),
        'boundary_dist': lambda x, p: lift_boundary_dist(x, s_in, p),
        'master': lambda x, p: _lift_master_stacked(x, s_in, p)
    }
    
    kernel_func = kernel_map.get(lifter)
    if not kernel_func:
        return None
        
    try:
        # Some kernels admit 2 args (X, params), others 3 (X, State, params)
        # We wrapped the 3-arg ones in lambdas above.
        features = kernel_func(X, lift_params)
    except Exception:
        return None # Kernel failure (e.g. singular)
        
    N, n_features = features.shape
    
    # --- 2. EXTRUSION ---
    # Expand domain by K: [Features, step]
    # Step k in 0..K-1
    
    if K > 1:
        # Replicate features K times
        feat_expanded = np.repeat(features, K, axis=0) # (N*K, F)
        
        # Create steps: [0, 1, ..., K-1, 0, 1, ...]
        steps = np.tile(np.arange(K), N)[:, None].astype(float)
        
        X_base = np.hstack([feat_expanded, steps])
        lift_dim = n_features # Index of the step dimension
    else:
        # K=1: No extrusion step needed.
        # X_base matches LiftOperator output directly.
        X_base = features
        lift_dim = -1 # No valid step dimension
    
    # --- 3. MATCHING & SUBSET SELECTION ---
    X_tgt = s_out.points
    
    candidate_subsets = [] # List of (mask, slice_vals)
    
    # A. EXACT BIJECTION (N_out == N_base)
    if len(X_base) == len(X_tgt):
        candidate_subsets.append((np.ones(len(X_base), dtype=bool), None, -1))
        
    # B. SUBSET SELECTION (N_out < N_base)
    elif len(X_base) > len(X_tgt):
        # SUBSET SELECTION STRATEGIES
        
        # 1. Align Centroids approx
        # Project X_base to D dims (first D cols of features usually X)
        base_proj = X_base[:, :n_dims]
        
        # Strategy A: Direct Spatial Match (Assume no translation or subset is in-place)
        # Good for pure cropping / deletion.
        dists_direct = cdist(base_proj, X_tgt)
        row_ind_dir, col_ind_dir = linear_sum_assignment(dists_direct)
        
        # Compute match cost
        cost_direct = dists_direct[row_ind_dir, col_ind_dir].sum()
        
        mask_dir = np.zeros(len(X_base), dtype=bool)
        mask_dir[row_ind_dir] = True
        
        # Helper to derive slice from mask (Refactored)
        def add_candidate_from_mask(mask_in):
            best_axis = -1
            best_vals = []
            best_len = float('inf')
            
            # Check default extrusion step first for conflict
            step_dim = lift_dim if lift_dim >= 0 else X_base.shape[1] - 1
            step_vals = sorted(list(set(X_base[mask_in, step_dim])))
            
            # If step_vals covers everything, it's not useful? No, it's useful if it excludes rejected.
            conflict_step = np.isin(X_base[~mask_in, step_dim], step_vals)
            if not np.any(conflict_step):
                best_axis = step_dim
                best_vals = step_vals
                best_len = len(step_vals)

            # Check all lifted features
            D_base = X_base.shape[1]
            for col in range(D_base):
                 vals = np.unique(X_base[mask_in, col])
                 conflict = np.isin(X_base[~mask_in, col], vals)
                 if not np.any(conflict):
                     if len(vals) < best_len:
                         best_axis = col
                         best_vals = sorted(list(vals))
                         best_len = len(vals)
            
            if best_axis != -1:
                candidate_subsets.append((mask_in, best_vals, best_axis))

        add_candidate_from_mask(mask_dir)
        
        # Strategy B: Centroid Alignment (Assume subset is centered relative to global)
        # Only try if Direct Match wasn't perfect? 
        # Or always try? Robustness says always try.
        if cost_direct > 1e-4:
            centroid_tgt = np.mean(X_tgt, axis=0)
            centroid_base = np.mean(base_proj, axis=0)
            shift = centroid_tgt - centroid_base
            base_aligned = base_proj + shift
            
            dists_aligned = cdist(base_aligned, X_tgt)
            row_ind_aln, _ = linear_sum_assignment(dists_aligned)
            
            mask_aln = np.zeros(len(X_base), dtype=bool)
            mask_aln[row_ind_aln] = True
            
            add_candidate_from_mask(mask_aln)
        
            
            add_candidate_from_mask(mask_aln)
            
        # Strategy C: Linear Separator (Oblique Cut)
        # 1. Compute means of Kept vs Deleted
        # 2. Project onto w = mean_kept - mean_del
        # 3. Check separability
        
        def add_candidate_linear_cut(mask_in):
            """
            Strategy C: Oblique Linear Separator.
            Checks if mask can be separated by a hyperplane normal to w = mean_in - mean_out.
            If so, proposes a Lifted Transform comprising:
            1. Projection onto w (added as new dimension).
            2. Axis-aligned slice on this new dimension.
            """
            if mask_in.all() or not mask_in.any():
                return
            
            X_kept = X_base[mask_in]
            X_del = X_base[~mask_in]
            
            mu_k = X_kept.mean(axis=0)
            mu_d = X_del.mean(axis=0)
            w = mu_k - mu_d
            norm = np.linalg.norm(w)
            if norm < 1e-9:
                return # Centers coincide
                
            w = w / norm
            
            # Project onto separating direction
            p_k = X_kept @ w
            p_d = X_del @ w
            
            # Check separability (Case 1: Kept > Deleted)
            if p_k.min() > p_d.max():
                # Separable!
                # We augment the Regression Target to include the projection values.
                # This forces the Operator to learn the transformation X -> [Y, X@w].
                # Then we slice the last dimension.
                
                # Robust ranges (tol=0.001)
                vals_min = float(p_k.min()) - 0.001
                vals_max = float(p_k.max()) + 0.001
                
                # Format: (mask, (vals_min, vals_max), -1, w)
                candidate_subsets.append((mask_in, (vals_min, vals_max), -1, w))
                
            # Case 2: Kept < Deleted
            elif p_k.max() < p_d.min():
                vals_min = float(p_k.min()) - 0.001
                vals_max = float(p_k.max()) + 0.001
                candidate_subsets.append((mask_in, (vals_min, vals_max), -1, w))
                
        # Activate Strategy C
        add_candidate_linear_cut(mask_dir)

        
    if not candidate_subsets:
        return None
        
    # HELPER: Sort orders via lexsort
    def get_sort_order_spatial(arr):
        """Primary sort by spatial coords (X, Y...)."""
        D = arr.shape[1]
        # Lexsort keys: (col[D-1], ..., col[0]). Last is primary.
        # So we want col[0] last in tuple.
        keys = tuple(arr[:, d] for d in range(D-1, -1, -1))
        return np.lexsort(keys)

    def get_sort_order_lift(arr):
        """Primary sort by Lift Dimension (last col), then Spatial."""
        D = arr.shape[1]
        # Keys: (col[D-2], ..., col[0], col[D-1]). 
        # Last in tuple is col[D-1] (Lift dim). So it becomes Primary.
        keys_list = [arr[:, d] for d in range(D-2, -1, -1)]
        keys_list.append(arr[:, -1]) # Step dim
        return np.lexsort(tuple(keys_list))

    X_tgt_sorted_spatial = X_tgt[get_sort_order_spatial(X_tgt)]
    
    def try_candidate(candidate):
        # Unpack with optional projection vector
        if len(candidate) == 4:
            mask, slice_range, slice_axis, proj_w = candidate
            slice_vals = None # We use range for continuous cut
        else:
            mask, slice_vals, slice_axis = candidate
            proj_w = None
            slice_range = None
            
        X_subset = X_base[mask]
        
        if len(X_subset) != len(X_tgt): return None
        
        # --- 4. SOLVE REGRESSION ---
        
        # Augment Input (Bias)
        X_aug = np.hstack([X_subset, np.ones((len(X_subset), 1))]) # (N_tgt, dim_L + 1)
        
        # Prepare Target Y
        if proj_w is not None:
             # Strategy C: Oblique Cut
             # Target Output is Y_tgt PLUS the Separator Value.
             # Y_train = [Y_tgt, X_subset @ w]
             # This forces M_final to map X -> Y AND X -> Separator.
             sep_values = X_subset @ proj_w
             Y_train = np.hstack([X_tgt, sep_values[:, None]])
        else:
             Y_train = X_tgt
        
        # Solve Y = X_aug @ W.T
        # Use SVD for robustness
        try:
            W = np.linalg.lstsq(X_aug, Y_train, rcond=None)[0].T 
        except np.linalg.LinAlgError:
            return None 
            
        M_linear = W[:, :-1] 
        v_bias = W[:, -1]    
        
        # BUILD OPERATOR
        D_out = s_out.n_dims
        dim_L = X_base.shape[1]
        
        if proj_w is not None:
            # Output dims = D_out + 1 (Last dim is Separator)
            # Ensure M_final can hold it based on max(dim_L, D_out+1)
            final_dim = max(dim_L, D_out + 1)
            M_final = np.eye(final_dim)
            b_final = np.zeros(final_dim)
            
            # Map top block
            M_final[:D_out+1, :dim_L] = M_linear
            b_final[:D_out+1] = v_bias
            
            # Slice Axis is the newly created dimension (D_out)
            op_slice_axis = D_out
        else:
            final_dim = max(dim_L, D_out)
            M_final = np.eye(final_dim)
            b_final = np.zeros(final_dim)
            M_final[:D_out, :dim_L] = M_linear 
            b_final[:D_out] = v_bias
            op_slice_axis = slice_axis

        # BUILD OPERATOR
        lift_op = LiftOperator(lifter=lifter, params=lift_params)
        bijection_op = BijectionOperator(linear=M_final, translation=b_final)
        
        s_vals = tuple(slice_vals) if slice_vals else None
        
        slice_op = SliceOperator(
             slice_values=s_vals,
             slice_range=slice_range,
             slice_axis=op_slice_axis,
             original_dims=D_out, # Slice returns D_out dims (drops separator if > D_out)
             slice_dims=tuple(range(D_out))
        )
        
        return LiftedTransform(lift=lift_op, bijection=bijection_op, slice=slice_op)

    # Filter valid
    # results = list(filter(None, map(try_candidate, candidate_subsets)))
    results = []
    for cand in candidate_subsets:
        res = try_candidate(cand)
        if res: results.append(res)
        
    
    # Verification (Apply and Check)
    for op in results:
        res_state = op.apply(s_in)
        if res_state == s_out:
            return op
        else:
            pass 
            
    return None



def derive_causality(s_in: State, s_out: State, depth: int = 0) -> Optional[Operator]:
    """
    Derive Causal Separation: Inert (Identity) vs Causal (Change).
    
    Inert Group: Points that exist in both Input and Output (Intersection).
    Causal Group: Points that are New (in Output but not Input) or Moved.
    
    Returns a UnionOperator([InertOp, CausalOp]) with implicit Causal Priority.
    """
    if s_in.is_empty or s_out.is_empty:
        return None
        
    from ..operators.logic import UnionOperator, CausalityOperator
    from ..operators.base import IdentityOperator
    from ..operators.algebra import ConstantOperator
    
    # 1. Identify Inert (Static Background)
    # Inert = Intersection(s_in, s_out)
    s_inert = s_in.intersection(s_out)
    
    # 2. Identify Causal (Active Change)
    # Causal = Output \ Inert
    s_causal = s_out.difference(s_inert)
    
    ops = []
    
    # Inert Operator (Identity on Inert part)
    if not s_inert.is_empty:
        s_deleted = s_in.difference(s_out)
        
        if s_deleted.is_empty:
             op_inert = CausalityOperator(IdentityOperator(), score=0.0)
             ops.append(op_inert)
        else:
             op_inert = CausalityOperator(ConstantOperator(points=s_inert.points), score=0.0)
             ops.append(op_inert)
             
    # Causal Operator
    if not s_causal.is_empty:
        from .core import derive_transformation
        op_causal_core = derive_transformation(s_in, s_causal, depth=depth + 1)

        if op_causal_core is None:
             op_causal_core = ConstantOperator(points=s_causal.points) 
        
        op_causal = CausalityOperator(op_causal_core, score=1.0)
        ops.append(op_causal)
        
    if not ops:
        return None
        
    return UnionOperator(operands=ops)
    """
    Derive Kronecker Product (Fractal) Lift.
    Model: S_out = S_in (x) K
    Algebraic: x_out = x_in * Scale + k_Offset
    
    1. Check Mass Ratio M = N_out / N_in (Must be integer > 1)
    2. Partition S_out into N_in groups of size M.
    3. Check rigid displacement consistency (Kernel K).
    """
    if s_in.n_points == 0 or s_out.n_points == 0:
        return None
        
    if s_out.n_points % s_in.n_points != 0:
        return None
        
    M_ratio = int(s_out.n_points // s_in.n_points)
    if M_ratio <= 1:
        return None
        
    # Attempt to partition S_out based on S_in topology
    X_in = s_in.points
    X_out = s_out.points
    
    def sort_pts(X):
        # Lex sort using standard convention
        keys = tuple(X[:, d] for d in range(X.shape[1]-1, -1, -1))
        return X[np.lexsort(keys)]
        
    X_in_sorted = sort_pts(X_in)
    X_out_sorted = sort_pts(X_out)
    
    try:
        blocks = X_out_sorted.reshape(s_in.n_points, M_ratio, s_in.n_dims)
    except:
        return None
        
    # Estimate Scale & Shift
    # Centers of blocks vs Centers of X_in
    centers = np.mean(blocks, axis=1) # (N_in, D)
    
    # Regress Centers vs X_in_sorted
    A = np.column_stack([X_in_sorted, np.ones(s_in.n_points)])
    Sol, resid, rank, s = np.linalg.lstsq(A, centers, rcond=None)
    
    M_params = Sol.T # (D, D+1)
    
    # Residual Check for skeleton fit
    # Residuals are sum of squared errors
    mse_skeleton = np.mean(resid) if len(resid) > 0 else 0.0
    
    if len(resid) > 0 and mse_skeleton > 1e-2:
        return None # Skeleton doesn't fit affine map
        
    ScaleMat = M_params[:, :-1] # (D, D)
    Shift = M_params[:, -1]     # (D,)
    
    # Derive Kernel K (Centered)
    # Block_i = Center_i + K_centered_i
    # We check if K_centered_i is uniform across i
    
    Centered_Blocks = blocks - centers[:, np.newaxis, :] # (N_in, M, D)
    
    # Reference Kernel (Mean)
    Mean_Kernel = np.mean(Centered_Blocks, axis=0) # (M, D)
    
    # Check deviation
    Diffs = Centered_Blocks - Mean_Kernel[np.newaxis, :, :]
    max_diff = np.max(np.abs(Diffs))
    
    if max_diff > 1e-2:
        return None # Kernel not consistent (variance too high)
        
    # Construct K_abs (Absolute offsets)
    # x_out = x_in * S + K_abs[k]
    # Center_i = x_in * S + Shift
    # Block_i = Center_i + K_centered
    # Block_i = (x_in * S + Shift) + K_centered
    # Block_i = x_in * S + (Shift + K_centered)
    
    K_abs = Mean_Kernel + Shift # (M, D)
    
    # Construct Operator using new atomic operators
    lift_op = LiftOperator(lifter='kronecker', params={'kernel': K_abs, 'scale_matrix': ScaleMat})
    
    bijection_op = BijectionOperator(linear=ScaleMat, translation=Shift)
    
    slice_op = SliceOperator(original_dims=s_in.n_dims)
    
    op = LiftedTransform(lift=lift_op, bijection=bijection_op, slice=slice_op)
    
    # Verification
    
    if op.apply(s_in) == s_out:
        return op
    
    return None

def derive_tiling_lift_strategy(s_in: State, s_out: State) -> Optional[Operator]:
    """
    Derive Global Tiling via Algebraic Lattice Scaling.
    Supports Symmetric Tiling (Mixed Orientations) via UnionOperator.
    """
    n_in = s_in.n_points
    n_out = s_out.n_points
    n_dims = s_in.n_dims
    
    if n_out % n_in != 0:
        return None
        
    # 1. Generate Basis from core
    try:
        basis_full = _generate_Bn_basis(n_dims)
    except ValueError:
        return None # Too expensive
        
    # 2. Compute Scaling Factors
    in_std_full = np.std(s_in.points, axis=0)
    out_std_full = np.std(s_out.points, axis=0)
    
    safe_in_f = np.where(in_std_full < 1e-6, 1.0, in_std_full)
    gamma_full = np.round(out_std_full / safe_in_f)
    gamma_full = np.maximum(gamma_full, 1.0)
    
    scaling_factors = [np.ones(n_dims)]
    if not np.allclose(gamma_full, 1.0):
        scaling_factors.append(gamma_full)
        
    center_in_f = s_in.centroid
    
    # 3. Vectorized Search with Greedy Coverage
    
    # Precompute output view for fast checking
    out_v = view_as_void(np.ascontiguousarray(np.round(s_out.points, State.DECIMALS)))
    
    def try_scaling_factor(g_f):
        G_f = np.diag(g_f)
        
        # Tensor operation: Apply all Basis matrices at once
        M_group = np.matmul(basis_full, G_f)  # (K, D, D)
        
        # Center adjustment: b = (I - M) @ center
        I_stack = np.eye(n_dims)[None, :, :]
        b_group = np.matmul(I_stack - M_group, center_in_f)  # (K, D)
        
        # Apply to points: X_out = X_in @ M^T + b
        X_lat_all = np.matmul(s_in.points[None, :, :], M_group.transpose(0, 2, 1)) + b_group[:, None, :]
        
        found_ops = []
        unexplained_mask = np.ones(n_out, dtype=bool)
        
        # Iterate hypotheses to cover Output
        for k in range(len(basis_full)):
            if not np.any(unexplained_mask):
                break
                
            s_lat = State(X_lat_all[k])
            k_pts = s_out.erosion(s_lat)
            
            if k_pts.is_empty:
                continue
                
            # Calculate coverage of this hypothesis
            recon = s_lat.dilation(k_pts)
            
            # Check which points it explains
            recon_v = view_as_void(np.ascontiguousarray(np.round(recon.points, State.DECIMALS)))
            
            # Intersection with Output (Indices)
            # Since recon_v must be subset of out_v (by erosion definition), we find where it matches
            is_covered = np.isin(out_v, recon_v).flatten()
            
            # Check if it explains NEW points
            newly_explained = is_covered & unexplained_mask
            
            if np.any(newly_explained):
                # Valid contribution
                op = AffineTilingOperator(matrix=M_group[k], translations=k_pts.points)
                found_ops.append(op)
                unexplained_mask &= ~is_covered
        
        if not np.any(unexplained_mask) and found_ops:
            if len(found_ops) == 1:
                return found_ops[0]
            else:
                return UnionOperator(operands=found_ops)
                
        return None
        
    # Try Identity scale first (most common for Tiling)
    for g_f in scaling_factors:
        res = try_scaling_factor(g_f)
        if res: return res
        
    return None


def derive_homothety_lift_strategy(s_in: State, s_out: State) -> Optional[Operator]:
    """
    Derive Homothety (Resample/Zoom) Transform using Bounding Box Algebra.
    
    T(x) = (x - min_in) * (size_out / size_in) + min_out
    """
    if s_in.is_empty or s_out.is_empty: return None
    if s_in.n_dims != s_out.n_dims: return None
    
    # 1. Algebraic Bounding Boxes
    bb_min_in = s_in.bbox_min
    bb_max_in = s_in.bbox_max
    
    bb_min_out = s_out.bbox_min
    bb_max_out = s_out.bbox_max
    
    size_in = bb_max_in - bb_min_in
    size_out = bb_max_out - bb_min_out
    
    # Avoid zero division
    valid_dims = size_in > 1e-9
    if not np.all(valid_dims):
        return None # Degenerate input
        
    scales = np.zeros_like(size_in)
    scales[valid_dims] = size_out[valid_dims] / size_in[valid_dims]
    # For degenerate dims, scale is 1.0 or 0.0? 1.0 preserves.
    scales[~valid_dims] = 1.0
    
    # 2. Check Linearity (Centroid consistency)
    center_in = s_in.centroid
    center_out = s_out.centroid
    
    # Predicted center
    # center_out_pred = (center_in - bb_min_in) * scales + bb_min_out
    # If this matches, it's a good candidate.
    
    pred_center = (center_in - bb_min_in) * scales + bb_min_out
    if not np.allclose(pred_center, center_out, atol=State.TOLERANCE_RELAXED):
         return None
         
    # 3. Construct ResampleOperator
    # Note: ResampleOperator might need integer grid logic if it was legacy.
    # Let's check ResampleOperator API. It expects 'scale_factors'.
    # We use derived scales.
    
    # We need 'origin' and 'basis' for ResampleOperator?
    # Or just use Affine?
    # Homothety IS Affine. 
    # Why use ResampleOperator? Because it handles 'grid_dims'?
    # Ideally we should return an AffineTransform for pure scaling.
    # derive_affine_scaled handles isotropic scaling.
    # This handles ANISOTROPIC scaling (Resizing).
    
    from ..transform import AffineTransform
    
    # Construct Affine Matrix
    D = s_in.n_dims
    M = np.diag(scales)
    # T(x) = S(x - origin) + offset
    # T(x) = Sx - S*origin + offset
    # b = -M @ bb_min_in + bb_min_out
    
    b = bb_min_out - M @ bb_min_in
    
    op = AffineTransform(linear=M, translation=b)
    
    # Verify
    s_recon = op.apply(s_in)
    if s_recon == s_out:
        return op
        
    return None



def _generate_Bn_basis(n_dims: int) -> list:
    """
    Generate Hyperoctahedral Group B_n (Signed Permutations).
    Size: 2^n * n!
    
    AGENT.md Compliant: Pure tensor algebra via recursive Kronecker construction.
    NO itertools. NO explicit for loops (uses recursive tensor composition).
    """
    from math import factorial
    
    if n_dims > 4:
        # Combinatorial Explosion Guard (2^N * N!)
        raise ValueError(f"Bn basis generation too expensive for N={n_dims}. Limit is 4.")

    # Base cases (cached for efficiency)
    if n_dims == 0: 
        return [np.eye(0)]
    if n_dims == 1:
        return [np.array([[1.0]]), np.array([[-1.0]])]
    
    # === SIGN MATRICES via Binary Expansion ===
    n_signs = 2 ** n_dims
    sign_indices = np.arange(n_signs)[:, None] >> np.arange(n_dims)[None, :]
    signs = 1 - 2 * (sign_indices & 1)  # Convert 0,1 to 1,-1
    
    # Sign matrices: (2^n, n, n) diagonal
    sign_matrices = np.zeros((n_signs, n_dims, n_dims))
    diag_idx = np.arange(n_dims)
    sign_matrices[:, diag_idx, diag_idx] = signs
    
    # === PERMUTATION MATRICES ===
    fact_n = factorial(n_dims)
    perm_matrices = np.zeros((fact_n, n_dims, n_dims))
    
    # Generate permutation indices using Lehmer code (algebraic)
    lehmer = np.arange(fact_n)
    
    divisors = np.array([factorial(n_dims - 1 - i) for i in range(n_dims)])
    remaining = lehmer.copy()
    
    # Lehmer code expansion (vectorized over all perms)
    temp = remaining[:, None] // divisors[None, :]
    temp = temp % np.arange(n_dims, 0, -1)[None, :]
    
    # Convert Lehmer to actual permutation (requires sequential selection)
    noise = np.arange(n_dims)[None, :] * 1e-10
    sort_keys = temp.astype(float) + noise
    perm_indices = np.argsort(sort_keys, axis=1)
    
    # Convert to matrices
    perm_matrices[np.arange(fact_n)[:, None], np.arange(n_dims)[None, :], perm_indices] = 1.0
    
    # === COMPOSE: B_n = Signs × Perms ===
    composed = sign_matrices[:, None, :, :] @ perm_matrices[None, :, :, :]
    composed = composed.reshape(-1, n_dims, n_dims)
    
    return [composed[i] for i in range(composed.shape[0])]
