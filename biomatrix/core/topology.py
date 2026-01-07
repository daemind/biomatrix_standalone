#!/usr/bin/env python3
"""Topological object detection and connectivity analysis."""

import numpy as np
from typing import List, Tuple
from collections import deque

from .state import State


from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from scipy.ndimage import label
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import cdist


def partition_by_value(state: State, projection_dims: List[int] = None) -> List[State]:
    """Partition state into groups by equal value on projection dimension(s)."""
    if state.is_empty:
        return []
    
    points = state.points
    n_points = len(points)
    
    if n_points == 1:
        return [state]
    
    # Auto-detect: use dimension with minimum variance
    if projection_dims is None:
        variances = np.var(points, axis=0)
        min_var_dim = int(np.argmin(variances))
        projection_dims = [min_var_dim]
    
    projection_dims = np.atleast_1d(projection_dims)
    
    # Project onto grouping dimensions
    projected = points[:, projection_dims]
    
    # Group by unique values
    projected_rounded = np.round(projected, decimals=6)
    unique_values, inverse_indices = np.unique(projected_rounded, axis=0, return_inverse=True)
    
    return [State(points=points[inverse_indices == i]) for i in range(len(unique_values))]


def get_component_labels(state: State, mode: str = 'moore', **kwargs) -> np.ndarray:
    """Compute connected component labels for each point."""
    if state.is_empty:
        return np.array([], dtype=int)
    if state.n_points == 1:
        return np.array([0], dtype=int)
        
    A = get_adjacency_matrix(state, mode, **kwargs)
    n_components, labels = connected_components(csr_matrix(A), directed=False)
    return labels

def partition_by_connectivity(state: State, mode: str = None, **kwargs) -> List[State]:
    """Partition state into connected components."""
    if state.is_empty:
        return []
        
    labels = get_component_labels(state, mode, **kwargs)
    n_components = labels.max() + 1 if len(labels) > 0 else 0
    points = state.points
    
    return list(map(lambda i: State(points=points[labels == i]), range(n_components)))


# Alias for backward compatibility and semantic alignment (Shape = Point/Object)
detect_objects = partition_by_connectivity # Alias for topological detection
detect_groups_by_value = partition_by_value # Alias for value grouping

def get_extremities(state: State, mode: str = 'moore') -> List[np.ndarray]:
    """Identify extremities (points with exactly 1 neighbor)."""
    if state.n_points < 2:
        return [p for p in state.points]
    
    A = get_adjacency_matrix(state, mode)
    degree = np.array(A.sum(axis=1)).flatten()
    extremity_mask = degree == 1
    return [state.points[i] for i in np.where(extremity_mask)[0]]




# Optional dependencies - handle gracefully if possible, or assume present for scientific stack


def get_euler_number(state: State) -> int:
    """
    Calculate N-Dimensional Euler Characteristic.
    
    Low-Dim (D <= 3): Voxel Grid method (Objects - Holes).
    High-Dim (D > 3): Graph method (\u03c7 = V - E).
    """
    if state.is_empty:
        return 0
        
    # High-Dimensional Strategy: Graph Topology
    # Avoids grid explosion.
    # \u03c7(G) = |V| - |E|
    if state.n_dims > 3:
        # Build Minimum Spanning Tree or Proximity Graph?
        # Standard adjacency (Moore/Adaptive) represents the 1-skeleton.
        # \u03c7 \approx b0 - b1 (Connected Components - Cycles)
        # Using Adjacency Matrix A.
        # V = N
        # E = Number of edges / 2 (undirected)
        
        # Use simple Moore connectivity to define edges
        # Ideally we use Alpha Complex, but Adjacency is a proxy.
        A = get_adjacency_matrix(state, mode='adaptive')
        n_edges = A.nnz // 2
        n_vertices = state.n_points
        return n_vertices - n_edges
        
    # Low-Dim Voxel Strategy
    if label is None:
        return 1 if not state.is_empty else 0
        
    bbox_min = state.bbox_min.astype(int)
    bbox_max = state.bbox_max.astype(int)
    
    shape = tuple(bbox_max - bbox_min + 1)
    if np.prod(shape) > 1e7:  # 10M voxels limit
         A = get_adjacency_matrix(state, mode='moore')
         return state.n_points - (A.nnz // 2)

    grid = np.zeros(shape, dtype=int)
    indices = (state.points - bbox_min).astype(int)
    valid_mask = np.all((indices >= 0) & (indices < np.array(shape)), axis=1)
    valid_indices = indices[valid_mask]
    
    if len(valid_indices) == 0:
        return 1
        
    raveled_indices = np.ravel_multi_index(valid_indices.T, shape)
    grid.ravel()[raveled_indices] = 1
    binary = grid
    
    # N-dim structure (connectivity)
    ndim = binary.ndim
    structure = np.ones([3] * ndim)
    
    # Count objects
    _, n_objects = label(binary, structure=structure)
    
    # Count holes (background components - 1 for outer space)
    padded = np.pad(binary, pad_width=1, mode='constant', constant_values=0)
    inv_padded = 1 - padded
    _, n_bk = label(inv_padded, structure=structure)
    
    n_holes = n_bk - 1
    return n_objects - n_holes

def is_hollow(state: State) -> bool:
    """Returns True if the object has at least one internal hole (Euler < 1)."""
    return get_euler_number(state) <= 0


def get_generators(n_dims: int, connectivity: int = 8) -> np.ndarray:
    """Generate neighborhood basis vectors. 4=Von Neumann, 8=Moore."""
    
    if connectivity == 4:
        # Von Neumann: Permutations of (±1, 0, ..., 0)
        shp = (2 * n_dims, n_dims)
        gens = np.zeros(shp)
        idx = 0
        for d in range(n_dims):
            gens[idx, d] = 1; idx += 1
            gens[idx, d] = -1; idx += 1
        return gens
        
    else: # connectivity == 8 (Moore)
        # All combinations of {-1, 0, 1} excluding origin
        # Use np.mgrid for algebraic N-D grid generation
        slices = tuple([slice(-1, 2) for _ in range(n_dims)])
        grid = np.mgrid[slices]
        # Reshape to (3^n, n_dims)
        gens_list = np.stack([g.ravel() for g in grid], axis=-1)
        # Remove origin (all zeros row)
        origin_mask = np.all(gens_list == 0, axis=1)
        gens = gens_list[~origin_mask]
        return gens


def detect_sparsity(state: State) -> float:
    """
    Detect the characteristic grid step (sigma) of the state.
    
    Uses cdist (Deterministic O(N^2)) to find the median nearest-neighbor distance.
    If the state is a grid, this is the cell size.
    If sparse, this is the typical separation.
    """
    if state.n_points < 2:
        return 1.0
    
    # Deterministic Algebra: Full Distance Matrix
    # O(N^2) but safe for N=2000 (4M float64 = 32MB)
    dists = cdist(state.points, state.points)
    
    # Mask diagonal (set to infinity)
    np.fill_diagonal(dists, np.inf)
    
    # Find nearest neighbor for each point
    nn_dists = np.min(dists, axis=1)
    
    # Filter out potential duplicates (0 distance)
    valid_dists = nn_dists[nn_dists > 1e-6]
    
    if len(valid_dists) == 0:
        return 1.0
        
    # Use median to be robust against outliers/clusters
    sigma = float(np.median(valid_dists))
    return sigma


def get_adjacency_matrix(state: State, mode: str = None, **kwargs) -> csr_matrix:
    """
    Build sparse adjacency matrix using Deterministic Matrix Algebra (cdist).
    
    Modes:
    - 'moore': Chebyshev Distance (L_inf) <= 1.001. Connects diagonals.
    - 'von_neumann': Manhattan Distance (L_1) <= 1.001. Connects only faces.
    - 'adaptive': Scaled by local sparsity (L2).
    - 'knn': K-Nearest Neighbors (L2) - Graph Topology.
    
    Returns:
        scipy.sparse.csr_matrix: Sparse adjacency matrix (N x N)
    """
    if state.n_points == 0:
        return csr_matrix((0, 0))
    if state.n_points == 1:
        return csr_matrix((1, 1))

    # Auto-detect mode if None
    if mode is None:
        if state.n_dims > 3:
            mode = 'knn'
        else:
            mode = 'moore'
            
    points = state.points
    N = state.n_points
    
    if mode == 'moore':
        # Chebyshev (L_inf) <= 1.001
        dists = cdist(points, points, metric='chebyshev')
        threshold = 1.001
        
        # Boolean Mask (Algebraic)
        mask = (dists <= threshold)
        # Remove self-loops
        np.fill_diagonal(mask, False)
        
    elif mode == 'von_neumann':
        # Manhattan (L_1) <= 1.001
        dists = cdist(points, points, metric='cityblock')
        threshold = 1.001
        
        mask = (dists <= threshold)
        np.fill_diagonal(mask, False)
        
    elif mode == 'adaptive':
        sigma = detect_sparsity(state)
        # L2 Euclidean
        dists = cdist(points, points, metric='euclidean')
        threshold = sigma * 1.5 
        
        mask = (dists <= threshold)
        np.fill_diagonal(mask, False)
        
    elif mode == 'knn':
        k = kwargs.get('k', 5)
        k = min(k, N - 1)
        if k < 1: 
             return csr_matrix((N, N))
             
        dists = cdist(points, points, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        
        # Partition to find k smallest elements (O(N*k) approx)
        # argpartition is faster than full sort
        knn_indices = np.argpartition(dists, k, axis=1)[:, :k]
        
        # Build Adjacency from indices
        sources = np.repeat(np.arange(N), k)
        targets = knn_indices.ravel()
        
        # Symmetrize (Graph Consistency)
        # A_ij = 1 OR A_ji = 1
        rows = np.concatenate([sources, targets])
        cols = np.concatenate([targets, sources])
        data = np.ones(len(rows), dtype=int)
        
        # CSR Constructor sums duplicates
        A = csr_matrix((data, (rows, cols)), shape=(N, N))
        A.data[:] = 1 # Binarize
        return A
        
    else:
        raise ValueError(f"Unknown connectivity mode: {mode}")
        
    # Standard Mode Construction (from mask)
    return csr_matrix(mask.astype(int))


def get_boundary_mask(state: State) -> np.ndarray:
    """
    Detect boundary points (points on the edge of the bounding box).
    
    Algebraic: p is boundary if any coord == bbox_min or bbox_max.
    
    Dimension-agnostic.
    """
    if state.n_points == 0:
        return np.array([], dtype=bool)
    
    # Vectorized: check if any dimension is at boundary
    at_min = np.any(np.isclose(state.points, state.bbox_min), axis=1)
    at_max = np.any(np.isclose(state.points, state.bbox_max), axis=1)
    
    return at_min | at_max


def propagate_from_boundary(adjacency: csr_matrix, boundary_mask: np.ndarray, 
                            max_steps: int = 100) -> np.ndarray:
    """
    Propagate "reachability" from boundary via iterative sparse multiplication.
    
    Algebraic: v_{k+1} = (A @ v_k) | v_k
    Avoids computing A^k matrix powers (which fill in to density).
    
    Returns: Boolean mask of reachable points.
    """
    n = len(boundary_mask)
    if n == 0:
        return np.array([], dtype=bool)
    
    # Current reachable set (frontier + interior)
    reachable = boundary_mask.astype(float)
    
    # Explicit loop with vector multiplication (O(E) per step)
    # instead of Matrix power (O(N*E) per step)
    
    for step in range(max_steps):
        # Propagate one step further
        # A is sparse, reachable is vector. Result is vector.
        current_step = adjacency.dot(reachable)
        
        new_reachable = reachable + current_step
        new_reachable = (new_reachable > 0).astype(float)
        
        # Convergence check
        if np.array_equal(new_reachable, reachable):
            break
            
        reachable = new_reachable
    
    return reachable.astype(bool)


def get_interior(state: State) -> State:
    """
    Interior(Envelope(S)) via Algebraic Connectivity (Matrix Flow).
    
    Definition:
    Hole = Points in Lattice not reachable from Boundary via Manhattan paths in S^c.
    
    Algebraic Formulation:
    1. L = Lattice(BBox(S))
    2. A = Adjacency(L), A_ij = 1 iff ||x_i - x_j||_1 = 1
    3. Boundary = ∂L
    4. Flow = Propagate(Boundary, A) restricted to S^c
    5. Interior = S^c ∖ Flow
    
    Dimension-agnostic (works in N dimensions).
    Uses strict Manhattan distance (no magic thresholds).
    
    Returns: State containing interior points.
    """
    if state.n_points < 3:
        return State(np.empty((0, state.n_dims)))
    
    # Auto-detect variable dimensions
    variances = np.var(state.points, axis=0)
    variable_dims = np.where(variances > 1e-6)[0]
    
    if len(variable_dims) < 2:
        return State(np.empty((0, state.n_dims)))
    
    # 1. Generate Lattice L (BBox)
    points_proj = state.points[:, variable_dims]
    bbox_min = np.floor(points_proj.min(axis=0)).astype(int)
    bbox_max = np.ceil(points_proj.max(axis=0)).astype(int)
    
    # Expand bbox by 1 to ensure boundary connectivity around the shape
    # This prevents edge-touching holes from being misclassified
    # Actually, "Interior" means enclosed by S.
    # If a hole touches the bbox edge, it is NOT enclosed presumably?
    # Unless the user's "Envelope" implies the bbox itself closes the shape?
    # Grid convention: object is usually floating. Background is 0.
    # We treat "reachable from infinity" as exterior.
    
    # Expand bbox by 1
    slices = [slice(bbox_min[d] - 1, bbox_max[d] + 2) for d in range(len(variable_dims))]
    
    # Safeguard: volume of lattice
    vol = np.prod([s.stop - s.start for s in slices])
    if vol > 100000: # Max 100k points for interior check
        return State(np.empty((0, state.n_dims)))
        
    grid = np.mgrid[tuple(slices)]
    lattice = np.stack([g.ravel() for g in grid], axis=-1).astype(float)
    
    # 2. Identify S and S^c in Lattice
    occupied_set = set(map(tuple, np.round(points_proj, 0)))
    
    # Mask for S^c (Empty space)
    is_empty = np.array([
        tuple(np.round(p, 0)) not in occupied_set
        for p in lattice
    ])
    
    if not np.any(is_empty):
        return State(np.empty((0, state.n_dims)))
        
    empty_indices = np.where(is_empty)[0]
    empty_points = lattice[empty_indices]
    
    # 3. Build Adjacency A for S^c (Empty Graph)
    # Only connect empty points to empty points
    n_empty = len(empty_points)
    
    # Efficient Adjacency Construction:
    # Use cdist with L1 metric
    # But for N~1000, 1000x1000 is manageable.
    dists = cdist(empty_points, empty_points, metric='cityblock')
    
    # Adjacency: Distance == 1.0 (Strict unit step)
    # Using isclose to handle float precision
    A = np.isclose(dists, 1.0).astype(float)
    
    # 4. Define Boundary Seeds
    # Points on the edge of the EXPANDED bbox are the source of "Exterior"
    # Since we expanded by 1, any point on the limits is exterior.
    # In the expanded lattice, min is bbox_min - 1, max is bbox_max + 1
    
    # Coordinates in variable dims
    # Check if any coordinate is at min or max of the ranges
    min_vals = np.min(lattice, axis=0)
    max_vals = np.max(lattice, axis=0)
    
    # For each empty point, check if it touches the outer lattice boundary
    is_boundary = np.any(
        (np.isclose(empty_points, min_vals)) | (np.isclose(empty_points, max_vals)),
        axis=1
    )
    
    # 5. Propagate Flow (Reachability)
    # R_{k+1} = R_k + A @ R_k
    # Converges when connected component is filled
    
    reachable = is_boundary.astype(float)
    prev_reachable = np.zeros_like(reachable)
    
    # Power iteration (BFS)
    # Max iterations = number of empty points
    for _ in range(n_empty):
        new_reachable = (A @ reachable) + reachable
        new_reachable = (new_reachable > 0).astype(float)
        
        if np.array_equal(new_reachable, reachable):
            break
        reachable = new_reachable
        
    # 6. Interior = Empty points NOT reachable
    interior_mask = (reachable == 0)
    interior_var = empty_points[interior_mask]
    
    if len(interior_var) == 0:
        return State(np.empty((0, state.n_dims)))
        
    # Reconstruct
    template = state.points[0, :].copy()
    interior_full = np.tile(template, (len(interior_var), 1))
    interior_full[:, variable_dims] = interior_var
    
    return State(interior_full)

def is_convex(state: State) -> bool:
    """
    Returns True if object == ConvexHull(object).
    Uses scipy.spatial.ConvexHull.
    """
    if state.n_points < state.n_dims + 1 or ConvexHull is None:
        return True
    
    # High-Dimensional Safety: QHull is exponential O(N^(D/2))
    # Disable exact convexity check for D > 8
    if state.n_dims > 8:
         return True # Assume convex or skip check
         
    points = state.points # N-dim
    
    try:
        hull = ConvexHull(points)
        
        # Grid based approach for hole checking involves N-dim grid iteration.
        # Can be expensive for high dims.
        # Heuristic: If volume of hull == volume of points (approx)?
        # Better: Sample random points in bbox and check?
        # For N-dim grid check, we need itertools.product over all dims.
        
        bbox_min = state.bbox_min
        bbox_max = state.bbox_max
        
        # Check if volume is tractable (e.g. < 1M points)
        ranges = [range(int(mn), int(mx) + 1) for mn, mx in zip(bbox_min, bbox_max)]
        
        # Create set of present points
        state_pt_set = set(map(tuple, np.round(points).astype(int)))
        
        # N-dim iterator
        # Check volume first? product of ranges.
        # If too large, we skip interior check and return True based on Hull existence.
        
        # Check up to 100k points?
        
        # Helper for internal point check
        # ALGEBRAIC: Point is hole if inside Hull but not in State
        
        # Generate lattice using np.mgrid (AGENT.md compliant)
        slices = [slice(int(mn), int(mx) + 1) for mn, mx in zip(bbox_min, bbox_max)]
        grid = np.mgrid[tuple(slices)]
        lattice = np.stack([g.ravel() for g in grid], axis=-1)
        
        # Vectorized hull membership check
        # val[i] = hull.equations @ lattice[i] = A @ x + b
        # Point is inside if all val <= 0
        vals = lattice @ hull.equations[:, :-1].T + hull.equations[:, -1]
        inside_hull = np.all(vals <= 1e-7, axis=1)
        
        # Check which lattice points are NOT in state
        # Vectorized Set Membership using Void View (Tensor Operation)
        lattice_int = lattice.astype(int)
        lattice_void = view_as_void(np.ascontiguousarray(lattice_int))
        
        # Convert state points to void view for fast lookup
        state_pts = np.round(points).astype(int)
        state_void = view_as_void(np.ascontiguousarray(state_pts))
        
        # np.isin is vectorized set membership
        # in_state[i] = True if lattice[i] in state
        in_state = np.isin(lattice_void, state_void)
        
        # Hole = inside hull AND not in state
        holes = inside_hull & ~in_state
        
        if np.any(holes):
            return False
            
        return True

        
    except Exception:
        # Return True on error (e.g. Qhull error on collinear points)
        # Collinear points (line) are convex.
        return True

def get_topology_vector(state: State) -> np.ndarray:
    """
    Returns extended invariant vector for N-dimensions.
    [mass] + spread (N-dims) + [hollow, convex, euler] + [boundary_metrics]
    """
    mass = float(state.n_points)
    spread = state.spread
    hollow = 1.0 if is_hollow(state) else 0.0
    convex = 1.0 if is_convex(state) else 0.0
    euler = float(get_euler_number(state))
    
    # Boundary Signature to resolve Symmetry
    # Average distance of extremities to centroid
    exts = get_extremities(state)
    n_exts = len(exts)
    
    if n_exts > 0:
        exts_pts = np.array(exts)
        centroid = state.centroid
        ext_dists = np.linalg.norm(exts_pts - centroid, axis=1)
        mean_ext_dist = np.mean(ext_dists)
        std_ext_dist = np.std(ext_dists)
    else:
        mean_ext_dist = 0.0
        std_ext_dist = 0.0
    
    return np.concatenate([
        [mass],
        spread,
        [hollow, convex, euler],
        [float(n_exts), mean_ext_dist, std_ext_dist]
    ])

def match_objects(objects_in: List[State], objects_out: List[State]) -> List[Tuple[State, State]]:
    """
    Match objects between input and output using Spectral Similarity (Identity Matrix pattern).
    Purely algebraic: S = V_in @ V_out.T
    """
    if not objects_in or not objects_out:
        return []
    
    # 1. Build Invariant Feature Matrices (n_obj x n_features)
    V_in = np.array([get_topology_vector(o) for o in objects_in])
    V_out = np.array([get_topology_vector(o) for o in objects_out])
    
    # 2. Vectorized Normalization (Cosine Similarity)
    # Add small epsilon to avoid division by zero for point states
    norm_in = np.linalg.norm(V_in, axis=1, keepdims=True) + 1e-9
    norm_out = np.linalg.norm(V_out, axis=1, keepdims=True) + 1e-9
    
    V_in_n = V_in / norm_in
    V_out_n = V_out / norm_out
    
    # 3. Spectral Similarity Matrix (One-Shot)
    # S[i, j] = cos_sim(V_in[i], V_out[j])
    # For perfect morphisms, S should be a permutation matrix (subset of Identity).
    S = V_in_n @ V_out_n.T
    
    # ALGEBRAIC: greedy matching logic using list comprehension (no for-loop)
    used_out = set()
    def greedy_match(i):
        j = np.argmax(S[i])
        if j not in used_out and S[i, j] > 0.4:
            used_out.add(j)
            return (objects_in[i], objects_out[j])
        return None
        
    return [m for i in range(len(objects_in)) if (m := greedy_match(i)) is not None]

def view_as_void(arr: np.ndarray) -> np.ndarray:
    """
    View a row as a single void element (structured array).
    Used for vectorized set operations (isin, unique) on rows.
    """
    arr = np.ascontiguousarray(arr)
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))

