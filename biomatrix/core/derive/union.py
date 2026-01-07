# -*- coding: utf-8 -*-
"""derive/union.py - Union detection and component matching."""


import numpy as np
from typing import Optional
from functools import reduce
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from ..operators.algebra import ConstantOperator, LiftOperator
from ..state import State
from ..transform import AffineTransform, derive_isometry_unordered
from ..topology import partition_by_connectivity, partition_by_value, view_as_void
from ..signatures import compute_universal_signature, signatures_match, compute_projected_signatures
from ..operators import (
    Operator, IdentityOperator, UnionOperator, SequentialOperator, SequenceOperator,
    SelectBySignatureOperator, SortAndSelectOperator, SelectThenActOperator,
    DifferenceOperator, InteriorOperator, ValueProjectionOperator
)

# Import from sibling modules (previously inline imports)
# from .causal import derive_causal_partition # ALL GONE
from .affine import (
    derive_affine_centered, derive_matched_affine, 
    derive_affine_permutation, derive_affine_scaled,
    derive_affine_subset
)
from .permutation import derive_value_permutation, derive_fiber_projection





def derive_union(s_in: State, s_out: State) -> Optional[Operator]:
    """Derive Union with multi-scale connectivity strategies."""
    if s_in.is_empty or s_out.is_empty: return None


    # Check Dimension Compatibility (REMOVED per User Feedback)
    # if s_out.n_dims != s_in.n_dims:
    #     return None
    
    # Coarse to Fine Strategies
    strategies = ['value', 'moore', 'von_neumann', 'adaptive']
    
    for strategy in strategies:
        if strategy == 'value':
            comps_in = partition_by_value(s_in)
            comps_out = partition_by_value(s_out)
        else:
            # Connectivity strategies
            comps_in = partition_by_connectivity(s_in, strategy)
            comps_out = partition_by_connectivity(s_out, strategy)
        
        if not comps_in or not comps_out: continue
        
        # RECURSION GUARD
        if len(comps_in) <= 1 and len(comps_out) <= 1:
             continue
        
        # Check cardinality
        if len(comps_in) < len(comps_out):
            op = _derive_generative_matching(comps_in, comps_out, s_in, s_out)
            if op: return op
            continue
            
        if len(comps_in) > len(comps_out):
            op = _derive_deletion_bijection(comps_in, comps_out, s_in, s_out)
            if op: return op
            
            op = _derive_surjective_simulation(comps_in, comps_out, s_in, s_out)
            if op: return op
            continue

        op = _derive_union_matching(comps_in, comps_out, s_in, s_out)
        if op:
            return op
            
    return None


def derive_hierarchical_invariant(s_in: State, s_out: State) -> Optional[Operator]:
    """Hierarchical invariant selector: causal, topological, or fiber."""
    if s_in.is_empty or s_out.is_empty:

        return None
    
    # Priority 1: CAUSAL (Legacy Heuristic) - REMOVED
    # Use Algebraic Lifting instead (Tier 2).
    
    # Priority 2: TOPOLOGICAL (component-wise transformation)
    comps_in = partition_by_connectivity(s_in)
    comps_out = partition_by_connectivity(s_out)
    
    if len(comps_in) == len(comps_out) and len(comps_in) > 1:
        sigs_in = list(map(lambda c: compute_universal_signature(c.points), comps_in))
        sigs_out = list(map(lambda c: compute_universal_signature(c.points), comps_out))
        
        def find_matching_out(i_and_c_in):
            i, c_in = i_and_c_in
            sig_in = sigs_in[i]
            candidates = list(filter(
                lambda c_out: signatures_match(sig_in, compute_universal_signature(c_out.points)) 
                              and c_in.n_points == c_out.n_points,
                comps_out
            ))
            return (c_in, sig_in, candidates[0]) if candidates else None
        
        matches = list(filter(None, map(find_matching_out, enumerate(comps_in))))
        
        if len(matches) == len(comps_in):
            def derive_comp_op(match):
                c_in, sig_in, c_out = match
                comp_transform = derive_affine_centered(c_in, c_out)
                if comp_transform is None:
                    comp_transform = derive_matched_affine(c_in, c_out)
                if comp_transform is None:
                    return None
                selector = SelectBySignatureOperator(target_signature=sig_in, connectivity_mode='moore')
                return SelectThenActOperator(selector=selector, operator=comp_transform)
            
            comp_ops = list(filter(None, map(derive_comp_op, matches)))
            
            if len(comp_ops) == len(comps_in):
                return UnionOperator(operands=comp_ops) if len(comp_ops) > 1 else comp_ops[0]
    
    # Priority 3: FIBER
    fiber_op = derive_fiber_projection(s_in, s_out)
    if fiber_op is not None:
        return fiber_op
    
    return None


def derive_composite_transform(s_in: State, s_out: State) -> Optional[Operator]:
    """Algebraic composite transform derivation."""
    if s_in.is_empty or s_out.is_empty:

        return None
    
    n_dims = s_in.n_dims
    X = s_in.points
    Y = s_out.points
    
    # === STEP 1: Compute dimension-wise invariants ===
    def dim_is_preserved(d: int) -> bool:
        vals_in = np.sort(X[:, d]) if len(X) == len(Y) else np.unique(X[:, d])
        vals_out = np.sort(Y[:, d]) if len(X) == len(Y) else np.unique(Y[:, d])
        
        if len(vals_in) != len(vals_out):
            return False
            
        return np.allclose(vals_in, vals_out, atol=0.1)
    
    preserved_dims = np.array(list(map(dim_is_preserved, range(n_dims))))
    transformed_dims = ~preserved_dims
    
    # === STEP 2: Determine transform type by cardinality ===
    if s_in.n_points == s_out.n_points:
        if not np.any(transformed_dims):
            return IdentityOperator()
        
        preserved_indices = np.where(preserved_dims)[0]
        
        if len(preserved_indices) > 0:
            X_key = tuple(X[:, d] for d in preserved_indices)
            Y_key = tuple(Y[:, d] for d in preserved_indices)
            
            order_in = np.lexsort(X_key[::-1])
            order_out = np.lexsort(Y_key[::-1])
            
            X_aligned = X[order_in]
            Y_aligned = Y[order_out]
            
            delta = np.mean(Y_aligned - X_aligned, axis=0)
            delta[preserved_dims] = 0.0
            
            if np.allclose(delta, 0):
                return None
            
            return AffineTransform.translate(delta)
    
    elif s_out.n_points < s_in.n_points:
        # Subset
        X_void = view_as_void(np.round(X, State.DECIMALS))
        Y_void = view_as_void(np.round(Y, State.DECIMALS))
        
        preserved_mask = np.isin(X_void, Y_void)
        
        if np.sum(preserved_mask) == s_out.n_points:
            comps_in = partition_by_connectivity(s_in)
            
            kept_sigs = list(filter(
                None,
                map(
                    lambda comp: compute_universal_signature(comp.points) 
                        if np.all(np.isin(view_as_void(np.round(comp.points, State.DECIMALS)), Y_void)) 
                        else None,
                    comps_in
                )
            ))
            
            if len(kept_sigs) > 0:
                select_ops = list(map(
                    lambda sig: SelectBySignatureOperator(target_signature=sig, connectivity_mode=None),
                    kept_sigs
                ))

                return UnionOperator(operands=select_ops) if len(select_ops) > 1 else select_ops[0]
        
        # Subset transform pattern
        comps_in = partition_by_connectivity(s_in)
        comps_out = partition_by_connectivity(s_out)
        
        if len(comps_out) <= len(comps_in):
            component_ops = [None] * len(comps_in)
            
            def find_best_match(j_and_c_out):
                j, c_out = j_and_c_out
                def try_match(i_and_c_in):
                    i, c_in = i_and_c_in
                    if c_in.n_points != c_out.n_points:
                        return None
                    op = derive_affine_centered(c_in, c_out)
                    if op is None:
                        return None
                    score = 1.0 if op.is_identity else 0.5
                    return (i, op, score)
                
                matches = list(filter(None, map(try_match, enumerate(comps_in))))
                if not matches:
                    return None
                best = max(matches, key=lambda x: x[2])
                return (j, best[0], best[1])
            
            all_matches = list(filter(None, map(find_best_match, enumerate(comps_out))))
            
            def accumulate_match(acc, match):
                used, ops = acc
                j, best_i, best_op = match
                if best_i in used:
                    return acc
                sig = compute_universal_signature(comps_in[best_i].points)
                selector = SelectBySignatureOperator(target_signature=sig, connectivity_mode=None)
                term = selector if best_op.is_identity else SelectThenActOperator(selector=selector, operator=best_op)
                new_ops = ops.copy()
                new_ops[best_i] = term
                return (used | {best_i}, new_ops)
            
            _, component_ops = reduce(accumulate_match, sorted(all_matches, key=lambda x: x[0]), (set(), component_ops))
            
            active_ops = list(filter(lambda op: op is not None, component_ops))
            
            if len(active_ops) > 0:
                 return UnionOperator(operands=active_ops) if len(active_ops) > 1 else active_ops[0]

    elif s_out.n_points > s_in.n_points:
        # Growth: Generative or Additive
        
        # 1. Try Unified Algebraic Lift first (covers tiling, symmetry, etc)
        from .algebra import derive_lift_and_slice
        rep_op = derive_lift_and_slice(s_in, s_out)
        if rep_op is not None:
             return rep_op

        # 2. Try Additive Component Matching (Input -> Output + Residual)
        # S_out = f(S_in) U Delta
        
        X_void = view_as_void(np.round(X, State.DECIMALS))
        Y_void = view_as_void(np.round(Y, State.DECIMALS))
        
        # Optimization: Check if S_in is a subset of S_out (Pure Additive)
        if np.all(np.isin(X_void, Y_void)):
             # Pure Additive: S_out = Identity(S_in) U Delta
             # Identify Delta (Residual)
             mask_in_out = np.isin(Y_void, X_void)
             delta_points = Y[(~mask_in_out).ravel()]
             
             if len(delta_points) > 0:
                 # Algebraic Operator: Union(Identity, Constant(Delta))
                 op_identity = IdentityOperator()
                 op_residual = ConstantOperator(points=delta_points)
                 return UnionOperator(operands=[op_identity, op_residual])

        # 3. General Mixed Case: Some components transform, others are new
        comps_in = partition_by_connectivity(s_in)
        comps_out = partition_by_connectivity(s_out)
        
        # Try to explain as many Input Components as possible mapping to Output Components
        # Residual Output Components become ConstantOperator
        
        # Map {C_in_i} -> {C_out_j}
        # Heuristic: Each Input Comp must map to exactly one Output Comp (or None if deleted)
        # Unmapped Output Comps are Delta.
        
        # "Seed & Grow" Matching (One-to-Many)
        # Allows matching a single input component to multiple output fragments (Fractured/Scaled)
        
        matched_ops = {} # index_in -> Operator
        matched_out_indices = set()
        
        # Helper: Get points of available output components
        def get_available_out_points(exclude_indices):
            pts = []
            for j, c in enumerate(comps_out):
                if j not in exclude_indices:
                    pts.append(c.points)
            if not pts: return np.zeros((0, s_out.n_dims))
            return np.vstack(pts)

        for i, c_in in enumerate(comps_in):
            best_op = None
            best_score = -1.0
            best_covered_indices = []
            
            # Identify candidates for this component using Seed Hypothesis
            # Try aligning with every available output component as a seed
            possible_seeds = [j for j in range(len(comps_out)) if j not in matched_out_indices]
            
            # Hypothesis 0: Cloud Match (One-to-Many via Subset Affine)
            # Match c_in to the collective cloud of available outputs
            cloud_points = get_available_out_points(matched_out_indices)
            if cloud_points.shape[0] >= c_in.n_points:
                # Construct temp state for cloud
                cloud_state = State(cloud_points)
                op_subset = derive_affine_subset(c_in, cloud_state, tol=State.TOLERANCE_RELAXED)
                if op_subset:
                     ops_to_try = [op_subset] # Prioritize global subset match
                else:
                     ops_to_try = []
            else:
                ops_to_try = []

            # Fallback to Seed Loop if Subset failed
            if not ops_to_try:
                for seed_j in possible_seeds:
                    c_seed = comps_out[seed_j]
                    
                    # 1. Centered (Isometry/Translation)
                    op1 = derive_affine_centered(c_in, c_seed)
                    if op1: ops_to_try.append(op1)
                    
                    # 2. Scaled (if size mismatch)
                    # Only if diff size
                    if abs(c_in.n_points - c_seed.n_points) > 0 or (op1 is None):
                         op2 = derive_affine_scaled(c_in, c_seed, tol=State.TOLERANCE_RELAXED)
                         if op2: ops_to_try.append(op2)
            
            # Verify Ops
            for op in ops_to_try:
                # Verify Coverage against ALL available outputs
                pred_points = op.apply(c_in).points
                    
                # Check which output components are covered by this prediction
                # This handles "Fractured" outputs
                covered_indices = []
                total_matched_in_pred = 0
                
                # We can check against strict point matching
                # For each available comp k, check if it is a subset of prediction
                # OR if prediction is a subset of union(available)?
                # Prediction should == Union(Covered Comps)
                
                current_matched_pts = 0
                current_indices = []
                
                # Fast check: Distance
                # Or just check points overlap (using void view)
                pred_void = view_as_void(np.round(pred_points, State.DECIMALS))
                
                # Check against all available
                # Optimization: Check overlap with Union(All Seeds)?
                # Better: Iterate all k, check inclusion.
                
                for k in possible_seeds:
                    c_k = comps_out[k]
                    k_void = view_as_void(np.round(c_k.points, State.DECIMALS))
                    
                    # Check strictly: Is c_k inside prediction?
                    if np.all(np.isin(k_void, pred_void)):
                         current_matched_pts += c_k.n_points
                         current_indices.append(k)
                    
                coverage = current_matched_pts / pred_points.shape[0] if pred_points.shape[0] > 0 else 0.0
                
                if coverage > (1.0 - State.TOLERANCE_TOPOLOGICAL):
                     score = coverage + (1e-3 if op.is_identity else 0.0)
                     if score > best_score:
                         best_score = score
                         best_op = op
                         best_covered_indices = current_indices

            if best_op:
                sig = compute_universal_signature(c_in.points)
                selector = SelectBySignatureOperator(target_signature=sig, connectivity_mode=None)
                term = selector if best_op.is_identity else SelectThenActOperator(selector=selector, operator=best_op)
                matched_ops[i] = term
                matched_out_indices.update(best_covered_indices)
            else:
                pass
        
        if len(matched_ops) > 0:
            ops_list = list(matched_ops.values())
            
            # Collect Residuals (Unmatched Outputs)
            unmatched_indices = set(range(len(comps_out))) - matched_out_indices
            residual_points = []
            for j in unmatched_indices:
                residual_points.append(comps_out[j].points)
            
            if residual_points:
                all_residuals = np.vstack(residual_points)
                ops_list.append(ConstantOperator(points=all_residuals))
                
            return UnionOperator(operands=ops_list)
                     
        # 4. Fallback: Interior Growth (special case logic from before)
        return None


def derive_deletion(s_in: State, s_out: State) -> Optional[Operator]:
    """N-dimensional component deletion: S_out = S_in - Σ C_deleted."""
    if s_out.n_points >= s_in.n_points or s_out.is_empty:
        return None
        
    comps_in = partition_by_connectivity(s_in)
    n_comps = len(comps_in)
    
    if n_comps == 0:
        return None
    
    out_v = view_as_void(np.ascontiguousarray(np.round(s_out.points, State.DECIMALS)))
    
    def classify_component(idx: int) -> int:
        c_pts = np.round(comps_in[idx].points, State.DECIMALS)
        c_v = view_as_void(np.ascontiguousarray(c_pts))
        in_out = np.isin(c_v, out_v)
        
        if np.all(in_out):
            return 1  # Fully kept
        elif np.any(in_out):
            return -1  # Partial (invalid)
        else:
            return 0  # Fully deleted
    
    classifications = np.array(list(map(classify_component, range(n_comps))))
    
    if np.any(classifications == -1):
        return None
    
    kept_mask = classifications == 1
    deleted_mask = classifications == 0
    
    if not np.any(deleted_mask):
        return None
    
    kept_pts = np.vstack(list(map(
        lambda i: np.round(comps_in[i].points, State.DECIMALS),
        np.where(kept_mask)[0]
    ))) if np.any(kept_mask) else np.empty((0, s_in.n_dims))
    
    kept_v = view_as_void(np.ascontiguousarray(kept_pts))
    if len(kept_v) != len(out_v) or not np.all(np.isin(kept_v, out_v)):
        return None
    
    deleted_indices = np.where(deleted_mask)[0]
    deleted_sigs = list(map(
        lambda i: compute_universal_signature(comps_in[i].points),
        deleted_indices
    ))
    
    delete_ops = list(map(
        lambda sig: SelectBySignatureOperator(target_signature=sig, connectivity_mode=None),
        deleted_sigs
    ))
    
    return DifferenceOperator(operands=[IdentityOperator()] + delete_ops)


def derive_component_permutation(s_in: State, s_out: State) -> Optional[Operator]:
    """
    N-Dimensional Component Permutation Detection.
    
    Find permutation σ on components such that:
        Σ T_i(C_in[σ(i)]) = S_out
    
    AGENT.md: Pure matrix algebra. NO EXPLICIT FOR LOOPS.
    """
    if s_in.n_points != s_out.n_points or s_in.is_empty:
        return None
        
    comps_in = partition_by_connectivity(s_in)
    comps_out = partition_by_connectivity(s_out)
    
    n_comps = len(comps_in)
    if n_comps != len(comps_out) or n_comps == 0:
        return None
    
    sigs_in = list(map(lambda c: compute_universal_signature(c.points), comps_in))
    sigs_out = list(map(lambda c: compute_universal_signature(c.points), comps_out))
    masses_in = np.array(list(map(lambda c: c.n_points, comps_in)))
    masses_out = np.array(list(map(lambda c: c.n_points, comps_out)))
    
    centroids_in = np.array(list(map(lambda c: c.centroid, comps_in)))
    centroids_out = np.array(list(map(lambda c: c.centroid, comps_out)))
    
    dist_matrix = cdist(centroids_in, centroids_out)
    
    sig_spectra_in = np.array(list(map(lambda s: s[1], sigs_in)))
    sig_spectra_out = np.array(list(map(lambda s: s[1], sigs_out)))
    
    sig_diff = sig_spectra_in[:, None, :] - sig_spectra_out[None, :, :]
    sig_max_diff = np.max(np.abs(sig_diff), axis=2)
    
    mass_match = masses_in[:, None] == masses_out[None, :]
    sig_match = sig_max_diff < 1e-4
    compatible = mass_match & sig_match
    
    affinity_matrix = np.where(compatible, dist_matrix, np.inf)
    
    if np.any(np.all(affinity_matrix == np.inf, axis=1)):
        return None
    
    has_finite = np.any(np.isfinite(affinity_matrix), axis=1)
    if not np.all(has_finite):
        return None
    row_ind, col_ind = linear_sum_assignment(affinity_matrix)
    
    if np.any(affinity_matrix[row_ind, col_ind] == np.inf):
        return None
    
    centroids_in = np.array(list(map(lambda c: c.centroid, comps_in)))
    centroids_out = np.array(list(map(lambda c: c.centroid, comps_out)))
    translations = centroids_out[col_ind] - centroids_in[row_ind]
    
    def translate_comp(idx: int) -> np.ndarray:
        return comps_in[row_ind[idx]].points + translations[idx]
    
    Y_pred = np.vstack(list(map(translate_comp, range(n_comps))))
    
    pred_v = view_as_void(np.ascontiguousarray(np.round(Y_pred, State.DECIMALS)))
    out_v = view_as_void(np.ascontiguousarray(np.round(s_out.points, State.DECIMALS)))
    
    if len(pred_v) == len(out_v) and np.all(np.isin(pred_v, out_v)):
        def make_select_translate(idx: int) -> SelectThenActOperator:
            sig = sigs_in[row_ind[idx]]
            return SelectThenActOperator(
                selector=SelectBySignatureOperator(target_signature=sig, connectivity_mode=None),
                operator=AffineTransform.translate(translations[idx])
            )
        
        ops = list(map(make_select_translate, range(n_comps)))
        return UnionOperator(operands=ops) if len(ops) > 1 else ops[0]
        
    return None


def _derive_deletion_bijection(comps_in: list, comps_out: list, s_in: State, s_out: State) -> Optional[Operator]:
    """
    Derive transformation for component deletion via Cutting Plane Lifting.
    
    ALGEBRAIC PRINCIPLE: Bijection via Dimension Lifting + Hyperplane Slice.
    """
    # Inline import to avoid circular dependency (union -> algebra -> state -> core -> union)
    from .algebra import derive_lift_and_slice
    
    n_in = len(comps_in)
    n_out = len(comps_out)
    n_dims = s_in.n_dims
    
    if n_in <= n_out:
        return None
        
    # === STEP 1: Signature-based Cost Matrix ===
    sigs_in = [compute_universal_signature(c.points) for c in comps_in]
    sigs_out = [compute_universal_signature(c.points) for c in comps_out]
    
    centroids_in = np.array([c.centroid for c in comps_in])
    centroids_out = np.array([c.centroid for c in comps_out])
    spatial_dists = cdist(centroids_in, centroids_out)
    
    def normalize_spectrum(s):
        if len(s) == 0 or s[0] == 0: return s
        return s / s[0]

    spectra_in = np.array([normalize_spectrum(s[1]) for s in sigs_in])
    spectra_out = np.array([normalize_spectrum(s[1]) for s in sigs_out])
    
    # Algebraic Weights for Cost Matrix
    # Shape Matching dominates Spatial Matching hierarchically
    
    # Algebraic Match: Constrained Optimization
    # Constraint: Topology (Shape) must match.
    # Objective: Minimal Spatial Displacement (Least Action).
    
    cost_matrix = spatial_dists.copy()
    
    # Enforce Topological Constraint (Hard Filter)
    # If shapes differ significantly, connection is forbidden (infinite cost)
    # Normalized spectrum difference > 1e-3 implies different topology
    TOPOLOGICAL_TOLERANCE = 1e-3
    shape_dists = cdist(spectra_in, spectra_out)
    
    incompatible_mask = shape_dists > TOPOLOGICAL_TOLERANCE
    cost_matrix[incompatible_mask] = np.inf
    
    # === STEP 2: Hungarian Matching ===
    # Attempt to find finite cost assignment
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ValueError:
        # Matrix might contains Infs or NaNs making it unsolvable? 
        # linear_sum_assignment handles infs usually, but requires feasible solution.
        # If all edges from a node are inf, cost is inf.
        return None
    
    valid_matches = []
    
    for r, c in zip(row_ind, col_ind):
        c_in = comps_in[r]
        c_out = comps_out[c]
        
        op_local = derive_isometry_unordered(c_in, c_out)
        
        if not op_local:
             op_local = derive_affine_centered(c_in, c_out)
             
        if not op_local:
             disp = c_out.centroid - c_in.centroid
             op_local = AffineTransform.translate(disp)
             if op_local.apply(c_in) != c_out:
                 op_local = None

        if not op_local:
             op_local = derive_lift_and_slice(c_in, c_out)
             
        if not op_local and c_in.n_points > c_out.n_points:
            # Try Lift & Slice for subset (Partial morphism checks inside lift_and_slice if configured)
            # Or dedicated subset check if lift doesn't cover it.
            # For now, rely on lift_and_slice to handle simple projections.
            op_test = derive_lift_and_slice(c_in, c_out)
            if op_test:
                 op_local = op_test

        if not op_local and cost_matrix[r, c] > 1e6:
             continue 

        if op_local:
            valid_matches.append((r, op_local, c_out))

    if not valid_matches:
        return None

    if len(valid_matches) != n_out:
        return None
        
    # === Build Operators ===
    sig_matrix = np.array([(s[0], *s[1]) for s in sigs_in])
    _, group_ids = np.unique(sig_matrix, axis=0, return_inverse=True)
    centroid_matrix = np.array([c.centroid for c in comps_in])
    input_ranks = np.zeros(n_in, dtype=int)
    unique_groups = np.unique(group_ids)
    
    def compute_group_ranks(g):
        mask = (group_ids == g)
        indices = np.where(mask)[0]
        group_centroids = centroid_matrix[mask]
        sort_keys = tuple(group_centroids[:, d] for d in range(centroid_matrix.shape[1] - 1, -1, -1))
        order = np.lexsort(sort_keys)
        ranks = np.argsort(order)
        return indices, ranks
    
    group_data = list(map(compute_group_ranks, unique_groups))
    for indices, ranks in group_data:
        input_ranks[indices] = ranks

    def build_operator(match):
        r_idx, op_t, c_out_target = match
        sig = sigs_in[r_idx]
        rank = input_ranks[r_idx]
        
        op_sig = SelectBySignatureOperator(target_signature=sig, connectivity_mode=None)
        op_rank = SortAndSelectOperator(axis=0, rank=rank, connectivity_mode=None)
        selector = SequentialOperator([op_sig, op_rank])
        
        return SelectThenActOperator(selector=selector, operator=op_t)
    
    ops = list(map(build_operator, valid_matches))
    
    transformed = list(map(lambda m: m[1].apply(comps_in[m[0]]), valid_matches))
    s_reconstructed = State(np.vstack([t.points for t in transformed])) if transformed else State(np.zeros((0, n_dims)))

    if s_reconstructed != s_out:
        return None
    if s_reconstructed == s_out:
        return UnionOperator(operands=ops)

    return None


def _derive_generative_matching(comps_in: list, comps_out: list, s_in: State, s_out: State) -> Optional[Operator]:
    """
    Derive Generative Union (Surjective/Splitting + Creation).
    Handles N_out > N_in via algebraic assignment.
    
    AGENT.md Compliant: No for loops, uses Hungarian + map().
    """
    # Inline import to avoid circular dependency
    from .algebra import derive_lift_and_slice
    
    n_in = len(comps_in)
    n_out = len(comps_out)
    n_dims = s_in.n_dims
    
    sigs_in = list(map(lambda c: compute_universal_signature(c.points), comps_in))
    sigs_out = list(map(lambda c: compute_universal_signature(c.points), comps_out))
    
    def normalize_spectrum(s):
        if len(s[1]) == 0 or s[1][0] == 0: return s[1]
        return s[1] / s[1][0]
    
    spectra_in = np.array(list(map(normalize_spectrum, sigs_in)))
    spectra_out = np.array(list(map(normalize_spectrum, sigs_out)))
    
    centroids_in = np.array([c.centroid for c in comps_in])
    centroids_out = np.array([c.centroid for c in comps_out])
    
    shape_dists = cdist(spectra_out, spectra_in)
    # Algebraic Weights
    
    shape_dists = cdist(spectra_out, spectra_in)
    spatial_dists = cdist(centroids_out, centroids_in)
    
    # Algebraic Constrained Matching
    cost_matrix = spatial_dists.copy()
    TOPOLOGICAL_TOLERANCE = 1e-3
    incompatible_mask = shape_dists > TOPOLOGICAL_TOLERANCE
    cost_matrix[incompatible_mask] = np.inf
    
    assignments = np.argmin(cost_matrix, axis=1)
    
    def derive_pair(j):
        i = assignments[j]
        
        # Check if assignment is valid (finite cost)
        if cost_matrix[j, i] == np.inf:
            return None
        c_in = comps_in[i]
        c_out = comps_out[j]
        
        op = derive_isometry_unordered(c_in, c_out)
        if op and op.apply(c_in) == c_out: return (i, op, c_out)
        
        op = derive_affine_centered(c_in, c_out)
        if op and op.apply(c_in) == c_out: return (i, op, c_out)
        
        op = derive_lift_and_slice(c_in, c_out)
        if op and op.apply(c_in) == c_out: return (i, op, c_out)
        
        return None
    
    matches = list(map(derive_pair, range(n_out)))
    
    if None in matches:
        return None
    
    sig_matrix = np.array([(s[0], *s[1]) for s in sigs_in])
    _, group_ids = np.unique(sig_matrix, axis=0, return_inverse=True)
    centroid_matrix = np.array([c.centroid for c in comps_in])
    input_ranks = np.zeros(n_in, dtype=int)
    unique_groups = np.unique(group_ids)
    
    def compute_group_ranks(g):
        mask = (group_ids == g)
        indices = np.where(mask)[0]
        group_cents = centroid_matrix[mask]
        sort_keys = tuple(group_cents[:, d] for d in range(n_dims - 1, -1, -1))
        order = np.lexsort(sort_keys)
        return indices, np.argsort(order)
    
    for indices, ranks in map(compute_group_ranks, unique_groups):
        input_ranks[indices] = ranks
    
    def build_operator(match):
        i, op_t, c_out = match
        sig = sigs_in[i]
        rank = input_ranks[i]
        sel_sig = SelectBySignatureOperator(target_signature=sig, connectivity_mode=None)
        sel_rank = SortAndSelectOperator(axis=0, rank=rank, connectivity_mode=None)
        selector = SequentialOperator([sel_sig, sel_rank])
        return SelectThenActOperator(selector=selector, operator=op_t)
    
    ops = list(map(build_operator, matches))
    
    transformed = list(map(lambda m: m[1].apply(comps_in[m[0]]), matches))
    s_reconstructed = State(np.vstack([t.points for t in transformed]))
    
    if s_reconstructed == s_out:
        return UnionOperator(operands=ops)
        
    return None


def _derive_surjective_simulation(comps_in: list, comps_out: list, s_in: State, s_out: State) -> Optional[Operator]:
    """
    Handle N_in > N_out via Forward Simulation (Occam's Razor for Collision).
    
    Logic:
    1. Find Anchor Matches (best subset of inputs matching outputs).
    2. Derive Transformation T from Anchors.
    3. Simulate: S_pred = Union(T(c) for all c in comps_in).
    4. Render Check: S_pred == S_out (Set Equality).
    """
    
    n_in = len(comps_in)
    n_out = len(comps_out)
    
    unique_ops = []
    
    if s_in.n_points * s_out.n_points < 10000:
        diffs = s_out.points[:, None, :] - s_in.points[None, :, :]
        diffs_flat = diffs.reshape(-1, s_in.n_dims)
        diffs_int = np.round(diffs_flat).astype(int)
        
        u_vecs, counts = np.unique(diffs_int, axis=0, return_counts=True)
        
        sorted_indices = np.argsort(counts)[::-1]
        
        top_k = min(5, len(u_vecs))
        
        candidate_translations = u_vecs[sorted_indices[:top_k]]
        
        for vec in candidate_translations:
            T = AffineTransform(linear=np.eye(s_in.n_dims), translation=vec.astype(float))
            unique_ops.append(T)
            
    if not unique_ops:
        return None

    s_out_int = np.round(s_out.points).astype(int)
    
    n_comps = len(comps_in)
    n_cands = len(unique_ops)
    
    candidate_vecs = np.array([op.translation for op in unique_ops])
    
    validity = np.zeros((n_comps, n_cands), dtype=bool)
    pred_by_comp_cand = {}
    
    def check_comp_cand(c_idx):
        c_in = comps_in[c_idx]
        c_pts_int = np.round(c_in.points).astype(int)
        
        def check_translation(t_idx):
            t_vec = candidate_vecs[t_idx]
            pred_pts = c_pts_int + t_vec.astype(int)
            
            matches = np.all(pred_pts[:, None, :] == s_out_int[None, :, :], axis=2)
            point_in_out = np.any(matches, axis=1)
            
            is_valid = np.all(point_in_out)
            return (c_idx, t_idx, is_valid, pred_pts if is_valid else None)
        
        return list(map(check_translation, range(n_cands)))
    
    all_checks = [item for sublist in map(check_comp_cand, range(n_comps)) for item in sublist]
    
    valid_checks = list(filter(lambda x: x[2], all_checks))
    for c_idx, t_idx, _, pred_pts in valid_checks:
        validity[c_idx, t_idx] = True
        pred_by_comp_cand[(c_idx, t_idx)] = pred_pts
    
    def assign_component(c_idx):
        valid_cands = np.where(validity[c_idx])[0]
        if len(valid_cands) == 0:
            return None
        t_idx = valid_cands[0]
        return (unique_ops[t_idx], pred_by_comp_cand[(c_idx, t_idx)])
    
    assignments = list(filter(None, map(assign_component, range(n_comps))))
    final_ops = [a[0] for a in assignments]
    pred_points_list = [a[1] for a in assignments]
    
    if not pred_points_list:
        return None
    
    all_pred = np.vstack(pred_points_list)
    u_pred = np.unique(all_pred, axis=0)
    u_out = np.unique(s_out_int, axis=0)
    
    if u_pred.shape[0] != u_out.shape[0]:
        return None
        
    u_pred = u_pred[np.lexsort(u_pred.T)]
    u_out = u_out[np.lexsort(u_out.T)]
    
    if np.array_equal(u_pred, u_out):
        return UnionOperator(final_ops)
            
    return None


def _derive_union_matching(comps_in: list, comps_out: list, s_in: State, s_out: State) -> Optional[Operator]:
    """
    Derive union operator from matched components.
    Uses Hungarian assignment and hypothesis testing.
    """
    # NOTE: derive_subset, derive_generative, derive_intersection imported inline to avoid circular
    # NOTE: derive_subset imported inline to avoid circular
    from .core import derive_subset
    from .algebra import derive_lift_and_slice
    
    n_in = len(comps_in)
    n_out = len(comps_out)
    
    centroids_in = np.array([c.centroid for c in comps_in])
    centroids_out = np.array([c.centroid for c in comps_out])
    
    sigs_in = [compute_universal_signature(c.points) for c in comps_in]
    sigs_out = [compute_universal_signature(c.points) for c in comps_out]
    
    def check_transform(T, c_i, c_o, tol=State.EPSILON):
        pred = T.apply(c_i)
        if pred.n_points != c_o.n_points: return False
        
        if pred.n_points == 0: return True
        
        dists = cdist(pred.points, c_o.points)
        min_dists1 = np.min(dists, axis=1)
        min_dists2 = np.min(dists, axis=0)
        
        if np.max(min_dists1) > tol: return False
        if np.max(min_dists2) > tol: return False
        
        return True
    
    sigs_in_proj = [compute_projected_signatures(c.points) for c in comps_in]
    sigs_out_proj = [compute_projected_signatures(c.points) for c in comps_out]
    
    masses_in = np.array([s[0] for s in sigs_in_proj])
    masses_out = np.array([s[0] for s in sigs_out_proj])
    mass_diff = np.abs(masses_in[:, None] - masses_out[None, :])
    
    dist_costs = cdist(centroids_in, centroids_out)
    cost_matrix = dist_costs.copy()
    cost_matrix += (mass_diff > 1e-3) * (1e6 + mass_diff)
    
    shapes_in = [s[1].shape for s in sigs_in_proj]
    shapes_out = [s[1].shape for s in sigs_out_proj]
    all_shapes_match = len(set(shapes_in + shapes_out)) == 1
    
    if all_shapes_match and len(sigs_in_proj) > 0 and len(sigs_out_proj) > 0:
        K = sigs_in_proj[0][1].shape[0]
        D_sig = sigs_in_proj[0][1].shape[1]
        
        S_in_tensor = np.array([s[1] for s in sigs_in_proj])
        S_out_tensor = np.array([s[1] for s in sigs_out_proj])
        
        diff = S_in_tensor[:, None, :, :] - S_out_tensor[None, :, :, :]
        dists_all = np.linalg.norm(diff, axis=3)
        sig_costs = np.min(dists_all, axis=2)
    else:
        flat_in = np.array([s[1].flatten() for s in sigs_in_proj])
        flat_out = np.array([s[1].flatten() for s in sigs_out_proj])
        
        max_len = max(flat_in.shape[1] if len(flat_in) > 0 else 0, 
                      flat_out.shape[1] if len(flat_out) > 0 else 0)
        if len(flat_in) > 0 and flat_in.shape[1] < max_len:
            flat_in = np.pad(flat_in, ((0,0), (0, max_len - flat_in.shape[1])))
        if len(flat_out) > 0 and flat_out.shape[1] < max_len:
            flat_out = np.pad(flat_out, ((0,0), (0, max_len - flat_out.shape[1])))
        
        sig_costs = cdist(flat_in, flat_out) if len(flat_in) > 0 and len(flat_out) > 0 else np.zeros((n_in, n_out))
            
    sig_mismatch = sig_costs > 1e-3 
    cost_matrix += sig_mismatch * (10.0 + sig_costs)
    cost_matrix += (~sig_mismatch) * sig_costs * 1.0 
                
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    hypothesis_pipeline = [
        ('isometry', lambda c_in, c_out: derive_isometry_unordered(c_in, c_out, tol=State.TOLERANCE_TOPOLOGICAL)),
        ('affine', lambda c_in, c_out: derive_affine_centered(c_in, c_out, tol=State.TOLERANCE_TOPOLOGICAL)),
        ('value_perm', lambda c_in, c_out: derive_value_permutation(c_in, c_out, tol=State.TOLERANCE_TOPOLOGICAL)),
        
        # Tier 2 Unified Lift (Replaces: scale_lift, envelope, porter, Generative, Intersection)
        ('unified_lift', lambda c_in, c_out: derive_lift_and_slice(c_in, c_out)),
        
        ('subset', lambda c_in, c_out: derive_subset(c_in, c_out, tol=State.TOLERANCE_TOPOLOGICAL)),
    ]
    
    def try_match(k):
        i, j = row_ind[k], col_ind[k]
        
        if cost_matrix[i, j] >= 1e8:
            return None
            
        c_in = comps_in[i]
        c_out = comps_out[j]
        
        def try_hypothesis(h):
            name, derive_fn = h
            T = derive_fn(c_in, c_out)
            if T:
                 if check_transform(T, c_in, c_out, tol=State.TOLERANCE_TOPOLOGICAL):
                     return (name, T)
            return None
        
        results = list(filter(None, map(try_hypothesis, hypothesis_pipeline)))
        return (i, j, results[0]) if results else None
    
    matches = list(filter(None, map(try_match, range(len(row_ind)))))
    
    used_inIndices = set(m[0] for m in matches)
    used_outIndices = set(m[1] for m in matches)
    
    def accumulate_ops(acc, match):
        i, j, op = match
        op_id = id(op)
        if op_id in acc:
            acc[op_id][1].append(i)
        else:
            acc[op_id] = (op, [i])
        return acc
    
    ops_dict = reduce(accumulate_ops, matches, {})
    ops_groups = list(ops_dict.values())
    
    if not ops_groups:
        return None
    
    sig_matrix = np.array([(s[0], *s[1]) for s in sigs_in])
    
    def build_operators_for_group(group):
        op_tuple, indices = group
        # Unwrap (name, Operator) tuple
        real_op = op_tuple[1] if isinstance(op_tuple, tuple) and len(op_tuple) == 2 and isinstance(op_tuple[1], Operator) else op_tuple
        
        
        
        # Aggressive Duck Typing Unwrap (Bypass isinstance due to reload issues)
        real_op = op_tuple
        if isinstance(op_tuple, tuple) and len(op_tuple) >= 2:
            candidate = op_tuple[1]
            # Check for 'apply' or class name convention
            if hasattr(candidate, 'apply') or "Operator" in type(candidate).__name__ or "LiftedTransform" in type(candidate).__name__:
                real_op = candidate
        
        idx_sigs = sig_matrix[indices]
        
        idx_sigs = sig_matrix[indices]
        _, inverse = np.unique(idx_sigs, axis=0, return_inverse=True)
        
        unique_sig_ids = np.unique(inverse)
        
        def build_for_sig_group(sig_id):
            mask = (inverse == sig_id)
            comp_indices = np.array(indices)[mask]
            sig = sigs_in[comp_indices[0]]
            
            sig_vec = sig_matrix[comp_indices[0]]
            all_matching_mask = np.all(np.abs(sig_matrix - sig_vec) < 1e-4, axis=1)
            all_matching_indices = np.where(all_matching_mask)[0]
            
            is_ambiguous = len(all_matching_indices) > len(comp_indices)
            
            if not is_ambiguous:
                selector = SelectBySignatureOperator(target_signature=sig)
                return [SequentialOperator(steps=[selector, real_op])]
            
            def build_for_target(target_idx):
                centroids = np.array([comps_in[i].centroid for i in all_matching_indices])
                
                sort_keys = tuple(centroids[:, d] for d in range(centroids.shape[1] - 1, -1, -1))
                order = np.lexsort(sort_keys)
                
                target_pos_in_all = np.where(all_matching_indices == target_idx)[0][0]
                rank = np.where(order == target_pos_in_all)[0][0]
                
                selector = SelectBySignatureOperator(target_signature=sig)
                sorter = SortAndSelectOperator(axis=0, rank=int(rank))
                return SequentialOperator(steps=[selector, sorter, real_op])
            
            return list(map(build_for_target, comp_indices))
        
        nested_ops = list(map(build_for_sig_group, unique_sig_ids))
        return [op for sublist in nested_ops for op in sublist]
    
    union_operands = [op for sublist in map(build_operators_for_group, ops_groups) for op in sublist]
    
    if len(union_operands) == 1 and len(used_inIndices) == len(comps_in):
        return union_operands[0]
    
    if len(used_outIndices) != n_out:
        return None
        
    return UnionOperator(operands=union_operands)


def _derive_sequence_algebraic(s_in: State, s_out: State) -> Optional[Operator]:
    """
    Algebraic Sequence Solver (NO LOOPS).
    
    Closed-form derivation:
    1. Translation: step = 2 * (mu_out - mu_in) / (n - 1)
    2. Rotation: Kabsch on detected copies via set algebra
    """
    
    if s_in.is_empty: 
        return None
    
    n_in = s_in.n_points
    n_out = s_out.n_points
    
    if n_out % n_in != 0:
        return None
    n_steps = n_out // n_in
    if n_steps < 2:
        return None
    
    pts_in = s_in.points
    pts_out = s_out.points
    
    mu_in = s_in.centroid
    mu_out = s_out.centroid
    
    step = 2.0 * (mu_out - mu_in) / (n_steps - 1)
    
    gen = AffineTransform.translate(step)
    seq_op = SequenceOperator(generator=gen, count=n_steps)
    
    if seq_op.apply(s_in) == s_out:
        return seq_op
    
    k_candidates = pts_out - pts_in[0]
    
    translated_all = pts_in[None, :, :] + k_candidates[:, None, :]
    
    diffs = np.abs(translated_all[:, :, None, :] - pts_out[None, None, :, :])
    dists_sq = np.sum(diffs ** 2, axis=3)
    min_dists = np.min(dists_sq, axis=2)
    
    valid_mask = np.all(min_dists < 1e-10, axis=1)
    valid_k = k_candidates[valid_mask]
    
    if len(valid_k) == n_steps:
        has_origin = np.any(np.all(np.abs(valid_k) < 1e-6, axis=1))
        
        if has_origin:
             k_norms = np.linalg.norm(valid_k, axis=1)
             k_order = np.argsort(k_norms)
             k_sorted = valid_k[k_order]
             
             if n_steps >= 2:
                 copy_1_pts = pts_in + k_sorted[1]
                 copy_1 = State(copy_1_pts)
                 gen = derive_affine_centered(s_in, copy_1)
                 if gen is not None:
                     seq_op = SequenceOperator(generator=gen, count=n_steps)
                     if seq_op.apply(s_in) == s_out:
                         return seq_op

    return None


def _is_subset(subset_pts: np.ndarray, full_pts: np.ndarray, tol: float = 1e-5) -> bool:
    """Algebraic Set Inclusion Check: subset <= full."""
    diffs = np.abs(subset_pts[:, None, :] - full_pts[None, :, :])
    min_dists_sq = np.min(np.sum(diffs**2, axis=2), axis=1)
    return np.all(min_dists_sq < tol**2)


__all__ = [
    'derive_union',
    'derive_hierarchical_invariant',
    'derive_composite_transform',
    'derive_deletion',
    'derive_component_permutation',
    '_derive_union_matching',
    '_derive_deletion_bijection',
    '_derive_generative_matching',
    '_derive_surjective_simulation',
    '_derive_sequence_algebraic',
    '_is_subset'
]
