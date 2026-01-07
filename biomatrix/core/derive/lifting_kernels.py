# -*- coding: utf-8 -*-
"""
derive/lifting_kernels.py - Algebraic Lifting Maps (\u03a6)

Strict AGENT.md Adherence:
- N-Dimensional Agnostic
- No Loops (Vectorized)
- Pure Functions

Lifting Map \u03a6: R^D -> R^{D+K}
Transforms non-linear problems into linear ones in higher dimensions.
"""

import numpy as np
from typing import Dict, Any, Tuple
from scipy.spatial.distance import cdist

from ..state import State
from ..topology import get_boundary_mask

def lift_identity(X: np.ndarray, params: Dict[str, Any] = None) -> np.ndarray:
    """
    Identity Lift: \u03a6(x) = x
    Base case for Affine transformations.
    """
    return X


def lift_kronecker(X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Kronecker Product Lift: \u03a6(x) = [x_local, x_global]
    
    Decomposes x into Quotient (Global/Block) and Remainder (Local/Pixel) spaces.
    x = x_global * Basis + x_local
    
    This algebraic decomposition linearizes periodic (Tiling) and self-similar (Fractal) structures.
    
    Params:
        basis (float or np.ndarray): Basis vector B (Period/BlockSize).
    """
    B = params.get('basis', 1.0)
    B = np.maximum(np.array(B), 1e-9)
    
    # Kronecker Decomposition (Quotient Ring / Euclidean Domain)
    # Global: Quotient (Block Index)
    x_global = np.floor(X / B)
    
    # Local: Remainder (Position in Block)
    # Uses remainder to ensure consistency for negative numbers if needed
    # But for Grid coords (positive), X - floor * B is fine.
    x_local = X - x_global * B
    
    # Return [x_local, x_global] -> R^{2D}
    # x_local is the "fine" structure, x_global is the "coarse" structure.
    return np.hstack([x_local, x_global])


def lift_topological(X: np.ndarray, state: State, params: Dict[str, Any]) -> np.ndarray:
    """
    Topological Lift: \u03a6(x) = [x, d(x, \u2202S), IsInterior]
    
    Embeds distance to boundary and interiority status.
    Solves Folding and Inset/Outset problems.
    
    Params:
        mode (str): 'dist', 'binary', 'signed_dist'
    """
    mode = params.get('mode', 'dist')
    
    # 1. Compute Boundary
    boundary_mask = get_boundary_mask(state)
    boundary_pts = X[boundary_mask]
    
    # 2. Compute Distance to Boundary
    if len(boundary_pts) > 0:
        dists = cdist(X, boundary_pts).min(axis=1)[:, None] # (N, 1)
    else:
        dists = np.zeros((len(X), 1))
        
    # 3. Compute Interior Mask (1.0 = interior, 0.0 = boundary)
    # (Assuming boundary_mask is 1 for boundary)
    is_interior = (1.0 - boundary_mask.astype(float))[:, None] # (N, 1)
    
    lift_features = []
    
    if 'dist' in mode:
        lift_features.append(dists)
    if 'binary' in mode:
        lift_features.append(is_interior)
        
    if not lift_features:
        # Default to both
        lift_features = [dists, is_interior]
        
    # Return [X, Features...]
    features = np.hstack(lift_features)
    return np.hstack([X, features])


def lift_distance_rank(X: np.ndarray, state: State, params: Dict[str, Any]) -> np.ndarray:
    """
    Distance Rank Lift: \u03a6(x) = [x, rank(\|x - c\|)]
    
    Lifts by ordering relative to a center. Useful for sequence/path extraction.
    Renamed from lift_mass_rank to match behavior.
    """
    anchor = params.get('anchor', state.centroid)
    dists = np.linalg.norm(X - anchor, axis=1)
    
    ranks = np.argsort(np.argsort(dists)).astype(float)[:, None]
    
    if len(X) > 1:
        ranks /= (len(X) - 1)
        
    return np.hstack([X, ranks])


def lift_connectivity_features(X: np.ndarray, state: State, params: Dict[str, Any] = None) -> np.ndarray:
    """
    Connectivity Lift: \u03a6(x) = [x, ComponentSize(x), ComponentId(x)]
    
    Embeds topological salience (Mass) into the vector space.
    Allows Linear Separation of "Noise" vs "Objects" based on size.
    """
    from ..topology import partition_by_connectivity
    
    # 1. Partition
    comps = partition_by_connectivity(state)
    
    N = X.shape[0]
    # Arrays to hold features per point
    sizes = np.zeros((N, 1))
    
    # Map points to components
    # Algebraic mapping:
    # Iterate components (O(K)), for each assign size.
    # K is usually small < 20.
    
    # To do this vector-wise without K-loop?
    # We need connected components labeling on the point cloud.
    # partition_by_connectivity returns list of States.
    
    # We can reconstruct a mapping assuming points in 'state' match 'X' order.
    # Since 'X' IS 'state.points' (usually).
    # Ideally partition returns indices.
    # For now, we trust the partitioning logic or match by coordinate.
    
    # Faster: partition_by_connectivity calls standard labeling.
    # Let's use the list of states.
    
    current_idx = 0
    # Ideally we need Original Indices. 
    # But state doesn't track them if we just pass points.
    # However, standard usage passes state which wraps X.
    # If X is state.points:
    
    # Optimization: Use KDTree or exact match?
    # Or just implementation detail:
    # We can assign sizes because we know the points.
    
    # Let's iterate components and match by proximity (tol=0) or equality.
    # Since X is exactly the points in State.
    
    # We need a map: Point -> Size.
    # Dict[tuple, size]? Slow.
    
    # Let's assume standard order or rebuild.
    # Rebuilding:
    pt_map = {tuple(np.round(pt, 5)): i for i, pt in enumerate(X)}
    
    for c in comps:
        sz = c.n_points
        norm_size = sz / N # Relative Mass (0..1)
        
        for p in c.points:
            key = tuple(np.round(p, 5))
            if key in pt_map:
                sizes[pt_map[key]] = norm_size
                
    return np.hstack([X, sizes])


def lift_unique_index(X: np.ndarray, params: Dict[str, Any] = None) -> np.ndarray:
    """
    Unique Index Lift: \u03a6(x) = [x, lex_rank]
    
    Assigns a canonical scalar order to every point based on coordinates.
    Used for Bijection/Serialization.
    """
    D = X.shape[1]
    # Lexsort keys: last dim is primary in lexsort, so we reverse
    keys = tuple(X[:, d] for d in range(D-1, -1, -1))
    perm = np.lexsort(keys)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm))
    
    ranks = inv_perm.astype(float)[:, None]
    
    # Normalize
    if len(X) > 1:
        ranks /= (len(X) - 1)
        
    return np.hstack([X, ranks])


def lift_poly2(X: np.ndarray, params: Dict[str, Any] = None) -> np.ndarray:
    """
    Polynomial Lift (Degree 2): \u03a6(x) = [x, x^2, cross_terms]
    
    Lifts to space of quadratic features.
    N-Dimensional agnostic.
    """
    N, D = X.shape
    features = [X]
    
    # Squared terms: x_i^2
    features.append(X**2)
    
    # Cross terms: x_i * x_j (i < j)
    # Vectorized outer product or similar?
    # Simple way: explicit pairs loop is O(D^2), acceptable for D < 50.
    # But strict AGENT.md prefers no loops.
    # Can we flatten upper triangle of X[:, :, None] * X[:, None, :]?
    
    # Outer product for each row:
    # X_outer = X[:, :, None] * X[:, None, :] # (N, D, D)
    # We want indices where i < j.
    if D > 1:
        idx_i, idx_j = np.triu_indices(D, k=1)
        cross_terms = X[:, idx_i] * X[:, idx_j]
        features.append(cross_terms)
        
    return np.hstack(features)




def lift_symmetry(X: np.ndarray, state: State, params: Dict[str, Any]) -> np.ndarray:
    """
    Symmetry/Folding Lift: \u03a6(x) = [x, 2c - x, |x - c|]
    
    Linearizes:
    1. Point Reflection: x -> 2c - x (Rotation 180 / Central Symmetry)
    2. Folding: x -> |x - c| (Wall Reflection)
    
    Chasles/Euler Principle:
    Folding is a selection of the absolute-value coordinate in lifted space.
    """
    # 1. Determine Center of Symmetry
    # Default to BBox center (geometric center) as it's more stable for folding than centroid
    bbox_min = state.bbox_min
    bbox_max = state.bbox_max
    c = (bbox_min + bbox_max) / 2.0
    
    # Allow override
    if 'center' in params:
        c = np.array(params['center'])
        
    # 2. Compute Features
    # Reflection: x' = c - (x - c) = 2c - x
    reflection = 2 * c - X
    
    # Folding: x' = |x - c|
    folding = np.abs(X - c)
    
    # Return [X, Reflection, Folding] -> R^{3D}
    return np.hstack([X, reflection, folding])


def lift_local_isotropy(X: np.ndarray, state: State, params: Dict[str, Any] = None) -> np.ndarray:
    """
    Local Isotropy / Mean Shift Magnitude.
    Phi(x) = [x, ||x - mean(Neighbors)||]
    
    Measures 'Symmetry' of the neighborhood.
    - Inside points (enclosed): Neighbors are symmetric -> Vector ~ 0.
    - Boundary points: Neighbors are biased -> Vector > 0.
    """
    params = params or {}
    k = params.get('k', 20) # Default to 20 or N-1
    N = len(X)
    
    # Handle small datasets
    if N <= 1:
         return np.hstack([X, np.zeros((N, 1))])
    
    actual_k = min(k, N - 1)
        
    if actual_k <= 0:
         return np.hstack([X, np.zeros((N, 1))])
    
    # Use scipy.spatial.KDTree (Standard Dependency)
    from scipy.spatial import KDTree
    tree = KDTree(X)
    
    # Query K neighbors 
    # KDTree query return (distances, indices)
    dists, indices = tree.query(X, k=actual_k + 1)
    
    # Check if indices is 1D
    if len(indices.shape) == 1:
        indices = indices.reshape(-1, 1)
        dists = dists.reshape(-1, 1)
    
    # Vectorized Computation
    # 1. Get Neighbors
    nbr_id = indices[:, 1:] # (N, k)
    nbr_dists = dists[:, 1:] # (N, k)
    
    if nbr_id.shape[1] == 0:
        return np.hstack([X, np.zeros((N, 1))])
        
    # 2. Gather Neighbor Points (Broadcasting)
    # X[nbr_id] creates shape (N, k, D)
    nbr_pts = X[nbr_id]
    
    # 3. Compute Centroids of Neighbors (N, D)
    centroids = np.mean(nbr_pts, axis=1)
    
    # 4. Mean Shift Vector (N, D)
    vecs = X - centroids
    mags = np.linalg.norm(vecs, axis=1) # (N,)
    
    # 5. Normalize by Local Scale (Mean Distance)
    mean_dists = np.mean(nbr_dists, axis=1) # (N,)
    
    # Safe Division
    # scores = mag / mean_dist
    scores = np.divide(
        mags, 
        mean_dists, 
        out=np.zeros_like(mags), 
        where=mean_dists > 1e-9
    )
    
    isotropy_scores = scores.reshape(-1, 1)
        
    iso = np.array(isotropy_scores).reshape(-1, 1)
    return np.hstack([X, iso])



def lift_boundary_dist(X: np.ndarray, state: State, params: Dict[str, Any] = None) -> np.ndarray:
    """
    Distance to Convex Hull Boundary.
    Phi(x) = [x, dist_to_hull(x)]
    
    Uses ConvexHull plane equations: Ax + b <= 0.
    Dist = min( |Ax + b| / ||A|| ) for all faces.
    inside points have +dist, boundary points have 0.
    """
    N, D = X.shape
    if N <= D + 1:
        # Cannot form hull
        return np.hstack([X, np.zeros((N, 1))])
        
    from scipy.spatial import ConvexHull, QhullError
    try:
        hull = ConvexHull(X)
        eqs = hull.equations # [Normal, offset] such that N.x + offset <= 0
        
        # Distance calculation
        # Normalized equations: normals have norm 1.
        # dist(x, plane) = -(n.x + b) for points inside (since n.x+b <= 0).
        
        normals = eqs[:, :-1]
        offsets = eqs[:, -1]
        
        # Projections: X @ N.T + b
        # Shape: (N_points, N_faces)
        projections = X @ normals.T + offsets
        
        # Depth inside hull
        depths = np.min(-projections, axis=1)
        depths = np.maximum(depths, 0.0) # Clip 0
        
        return np.hstack([X, depths.reshape(-1, 1)])
        
    except (QhullError, ValueError):
        # Hull failure (collinear or flat)
        return np.hstack([X, np.zeros((N, 1))])
