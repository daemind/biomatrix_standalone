import numpy as np
from dataclasses import dataclass

@dataclass
class State:
    """
    Pure Manifold P ⊂ ℝⁿ.
    NO Semantic Labels (x, y, c).
    NO Grid Topology (width, height).
    Just Points.
    """
    # ALGEBRAIC PRECISION CONSTANTS (N-Dim Agnostic)
    EPSILON = 1e-5              # Tolerance for numerical equality (machine precision)
    TOLERANCE_RELAXED = 1e-2    # Tolerance for complex transforms (symmetric shapes, rotations)
    TOLERANCE_TOPOLOGICAL = 0.1 # Tolerance for topological operations (subset, union, generative)
    TOLERANCE_COARSE = 0.5      # Coarse tolerance for degraded/lattice cases
    DECIMALS = 5                # Quantization for set operations (deterministic hashing)
    
    points: np.ndarray # (N, D) float
    causality_score: float = 0.0 # Causal Priority: 1.0 (Actor) > 0.0 (Inert)

    def __post_init__(self):
        self.points = np.atleast_2d(self.points).astype(float)
        # Note: Do NOT sort here. Sorting is done in __eq__ for set comparison.
        # Preserving order allows correspondence for transform derivation.

    @property
    def n_points(self) -> int: return self.points.shape[0]

    @property
    def n_dims(self) -> int: 
        # Always trust the shape, even if empty.
        # np.atleast_2d ensures len(shape) == 2
        return self.points.shape[1] if len(self.points.shape) > 1 else 0

    @property
    def is_empty(self) -> bool: return self.n_points == 0

    @property
    def centroid(self) -> np.ndarray: 
        return np.mean(self.points, axis=0) if not self.is_empty else np.zeros(self.n_dims)

    @property
    def spread(self) -> np.ndarray: 
        if self.is_empty: return np.zeros(self.n_dims)
        return np.max(self.points, axis=0) - np.min(self.points, axis=0)

    @property
    def bbox_min(self) -> np.ndarray:
        return np.min(self.points, axis=0) if not self.is_empty else np.zeros(self.n_dims)
    
    @property
    def bbox_max(self) -> np.ndarray:
        return np.max(self.points, axis=0) if not self.is_empty else np.zeros(self.n_dims)

    def copy(self) -> 'State':
        return State(self.points.copy(), causality_score=self.causality_score)

    def mask(self, m: np.ndarray) -> 'State':
        """Return a new State with points selected by boolean mask m."""
        if self.is_empty: return self.copy() # Or empty state
        # Ensure mask length matches
        if len(m) != self.n_points:
             # Handle broadcast or error?
             # For now, safe slice
             m = m[:self.n_points]
        return State(self.points[m], causality_score=self.causality_score)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State): return False
        if self.is_empty and other.is_empty: return True
        
        # Algebraic Set Equality: A = B <=> A ⊆ B and B ⊆ A
        # Implementation: Sort Unique points and compare
        # Rounding needed for floats
        
        s1 = np.unique(np.round(self.points, State.DECIMALS), axis=0)
        s2 = np.unique(np.round(other.points, State.DECIMALS), axis=0)
        
        if len(s1) != len(s2): return False
        
        # Lexsort for canonical comparison of sets
        s1 = s1[np.lexsort(s1.T[::-1])]
        s2 = s2[np.lexsort(s2.T[::-1])]
        
        return np.allclose(s1, s2, atol=State.EPSILON)

    @classmethod
    def from_grid(cls, grid: np.ndarray) -> 'State':
        """
        Embed an N-D array (grid) into an N+1 D point cloud.
        Mapping:
            Index (d0, d1, ... dN) -> Dims (0, 1, ..., N)
            Value v                -> Dim N+1
        Result is points in ℝ^(N+1).
        """
        grid = np.array(grid)
        # N-Dimensional Agnostic extraction check
        if grid.size == 0:
            return cls(np.zeros((0, grid.ndim + 1)))
            
        # Find non-zero indices (ALGEBRAIC: vectorized find)
        # indices shape: (K, D_grid)
        indices = np.argwhere(grid != 0)
        
        if len(indices) == 0:
            return cls(np.zeros((0, grid.ndim + 1)))
            
        # Extract values
        values = grid[tuple(indices.T)]
        
        # Stack: [d0, ..., dN, v]
        pts = np.column_stack([indices, values])
        
        return cls(pts)
    
    def to_grid(self, shape: tuple = None, spatial_axes: tuple = None, value_axis: int = -1) -> np.ndarray:
        """
        Project State to a dense array (grid).
        defaults:
          - value_axis = -1 (last dimension)
          - spatial_axes = all other dimensions
        """
        if self.is_empty:
            return np.zeros(shape if shape else (1,) * max(1, self.n_dims - 1), dtype=int)
            
        n = self.n_dims
        
        # Handle negative index
        val_idx = value_axis if value_axis >= 0 else n + value_axis
        
        # Default spatial axes: all except value axis
        if spatial_axes is None:
            spatial_axes = tuple(i for i in range(n) if i != val_idx)
            
        # Determine grid shape
        points_spatial = self.points[:, spatial_axes]
        if shape is None:
            max_coords = np.max(points_spatial, axis=0)
            shape = tuple(int(m) + 1 for m in max_coords)
            
        grid = np.zeros(shape, dtype=int)
        
        # Vectorized Grid Assignment?
        # Difficult with arbitrary dimensions and duplicate handling (last write wins?).
        # For now, fast loop is acceptable for typical Grid sizes.
        # Check bounds in vectorized way?
        
        # Integer coordinates
        coords = points_spatial.astype(int)
        values = self.points[:, val_idx]
        
        # Filter bounds
        in_bounds = np.ones(len(coords), dtype=bool)
        for i, s in enumerate(shape):
            in_bounds &= (coords[:, i] >= 0) & (coords[:, i] < s)
            
        valid_coords = coords[in_bounds]
        valid_values = values[in_bounds]
        
        # Assign
        # Use tuple of columns for N-dim indexing
        if len(valid_coords) > 0:
            grid[tuple(valid_coords.T)] = valid_values.astype(int)
            
        return grid

    @property
    def center(self) -> np.ndarray:
        """Alias for centroid."""
        return self.centroid

    # === Set Theory Operations ===
    
    def difference(self, other: 'State') -> 'State':
        """Set Difference: S \ Other."""
        if self.is_empty: return self
        if other.is_empty: return self
        
        s1 = set(map(tuple, np.round(self.points, State.DECIMALS)))
        s2 = set(map(tuple, np.round(other.points, State.DECIMALS)))
        
        diff = s1 - s2
        if not diff:
            return State(np.empty((0, self.n_dims)))
        return State(np.array(list(diff)))

    def union(self, other: 'State') -> 'State':
        """Set Union: S U Other."""
        if self.is_empty: return other.copy()
        if other.is_empty: return self.copy()
        
        pts = np.vstack([self.points, other.points])
        unique = np.unique(np.round(pts, State.DECIMALS), axis=0)
        return State(unique)

    def intersection(self, other: 'State') -> 'State':
        """Set Intersection: S ∩ Other."""
        if self.is_empty or other.is_empty: 
            return State(np.empty((0, self.n_dims)))
            
        s1 = set(map(tuple, np.round(self.points, State.DECIMALS)))
        s2 = set(map(tuple, np.round(other.points, State.DECIMALS)))
        
        inter = s1 & s2
        if not inter:
            return State(np.empty((0, self.n_dims)))
        return State(np.array(list(inter)))
        
    def is_subset_of(self, other: 'State') -> bool:
        """Set Inclusion: S ⊆ Other?"""
        if self.is_empty: return True
        if other.is_empty: return False
        
        s1 = set(map(tuple, np.round(self.points, State.DECIMALS)))
        s2 = set(map(tuple, np.round(other.points, State.DECIMALS)))
        
        return s1.issubset(s2)

    # === Mathematical Morphology (Lattice Algebra) ===
    
    def erosion(self, structuring_element: 'State') -> 'State':
        """
        Minkowski Erosion (A ⊖ B).
        Finds locations z where B_z ⊆ A.
        Algebraic definition: ∩_{b in B} (A - b).
        """
        if self.is_empty or structuring_element.is_empty:
            return State(np.empty((0, self.n_dims)))
        
        b_points = structuring_element.points
        n_b = len(b_points)
        
        # Optimization: Use the smallest set for intersection initialization?
        # No, we intersect (A - b_i). All have size |A|.
        
        candidates = None
        
        for i in range(n_b):
            bi = b_points[i]
            # S_z = {a - bi | a in A}
            z_set = set(map(tuple, np.round(self.points - bi, State.DECIMALS)))
            
            if candidates is None:
                candidates = z_set
            else:
                candidates &= z_set
            
            if not candidates:
                return State(np.empty((0, self.n_dims)))
                
        return State(np.array(list(candidates)))

    def dilation(self, structuring_element: 'State') -> 'State':
        """
        Minkowski Dilation (A ⊕ B).
        Union of copies of A translated by B.
        A ⊕ B = {a + b | a in A, b in B}.
        """
        if self.is_empty or structuring_element.is_empty:
             return State(np.empty((0, self.n_dims)))
             
        pts_a = self.points
        pts_b = structuring_element.points
        
        # Vectorized outer sum
        sum_pts = pts_a[:, None, :] + pts_b[None, :, :]
        flat_pts = sum_pts.reshape(-1, self.n_dims)
        
        unique = np.unique(np.round(flat_pts, State.DECIMALS), axis=0)
        return State(unique)

    def normalize(self) -> 'State':
        """
        Returns a normalized state centered at origin (Centroid).
        """
        if self.is_empty:
            return self.copy()
        return self.translate(-self.centroid)

    def translate(self, vector: np.ndarray) -> 'State':
        """
        Translate the state by vector.
        """
        if self.is_empty: return self.copy()
        new_pts = self.points + vector
        return State(new_pts, causality_score=self.causality_score)

    def __repr__(self) -> str:
        return f"State(|P|={self.n_points}, D={self.n_dims})"

    def derive_transformation_to(self, other: 'State') -> 'Operator':
        """
        Derive the transformation T such that T(self) = other.
        
        This is the main entry point for transformation derivation.
        
        Returns:
            Operator: The transformation, or None if not derivable.
        
        Example:
            >>> s_in = State.from_grid(input_grid)
            >>> s_out = State.from_grid(output_grid)
            >>> T = s_in.derive_transformation_to(s_out)
            >>> assert T.apply(s_in) == s_out
        """
        from .derive import derive_transformation
        return derive_transformation(self, other)

    def analyze_detection_to(self, other: 'State') -> dict:
        """
        Analyze transformation detection from self to other.
        
        Returns a dictionary with detection results and failure analysis.
        Pure algebraic: delegates to derive functions, no script logic.
        
        Returns:
            dict with keys:
                - n_in, n_out, n_dims: sizes
                - identity, subset, affine, hungarian, lifting, additive, union: detection flags
                - detected: True if any detection succeeded
                - mass_ratio: output/input point ratio
                - failure_reason: string describing why detection failed (if applicable)
                - n_common, n_new, n_deleted: set comparison metrics
                - is_inplace, is_additive: subset pattern flags
        """
        from .derive import (
            derive_transformation, derive_subset, derive_deletion,
            derive_affine_centered, derive_matched_affine, derive_lifting,
            derive_union,
            derive_value_permutation, derive_component_permutation,
            derive_value_permutation, derive_component_permutation,
            derive_hierarchical_invariant,
            derive_composite_transform,
            derive_causality
        )
        from .topology import partition_by_connectivity
        
        s_in, s_out = self, other
        
        # N-DIM AGNOSTIC METRICS via set algebra
        in_pts = set(map(tuple, np.round(s_in.points, 3)))
        out_pts = set(map(tuple, np.round(s_out.points, 3)))
        
        common_pts = in_pts & out_pts
        new_pts = out_pts - in_pts
        del_pts = in_pts - out_pts
        
        result = {
            'n_in': s_in.n_points,
            'n_out': s_out.n_points,
            'n_dims': s_in.n_dims,
            'identity': False,
            'subset': False,
            'affine': False,
            'hungarian': False,
            'lifting': False,
            'additive': False,
            'union': False,
            'composite': False,
            'detected': False,
            'mass_ratio': s_out.n_points / s_in.n_points if s_in.n_points > 0 else 0,
            'failure_reason': None,
            'n_common': len(common_pts),
            'n_new': len(new_pts),
            'n_deleted': len(del_pts),
            'is_inplace': len(new_pts) == 0,
            'is_additive': len(del_pts) == 0,
        }
        
        # 0. Edge Cases (Empty States)
        if s_out.n_points == 0:
            result['subset'] = True # Empty set is subset of any set
            result['detected'] = True
            return result
            
        if s_in.n_points == 0:
            result['union'] = True # Any set is union of empty set and itself
            result['detected'] = True
            return result

        # 1. IDENTITY
        if s_in == s_out:
            result['identity'] = True
            result['detected'] = True
            return result
        
        # 2. SUBSET (S_out < S_in)
        if s_out.n_points < s_in.n_points and s_out.n_points > 0:
            if derive_subset(s_in, s_out):
                result['subset'] = True
                result['detected'] = True
                return result
            if derive_deletion(s_in, s_out):
                result['subset'] = True
                result['detected'] = True
                return result
        
        # 3. SAME CARDINALITY
        if s_in.n_points == s_out.n_points:
            if derive_hierarchical_invariant(s_in, s_out):
                result['hungarian'] = True
                result['detected'] = True
                return result
            if derive_affine_centered(s_in, s_out):
                result['affine'] = True
                result['detected'] = True
                return result
            if derive_matched_affine(s_in, s_out):
                result['hungarian'] = True
                result['detected'] = True
                return result
            if derive_value_permutation(s_in, s_out):
                result['hungarian'] = True
                result['detected'] = True
                return result
            if derive_component_permutation(s_in, s_out):
                result['hungarian'] = True
                result['detected'] = True
                return result
            if derive_composite_transform(s_in, s_out):
                result['hungarian'] = True
                result['detected'] = True
                return result
            
            # Failure classification for same cardinality
            if result['is_inplace']:
                result['failure_reason'] = 'same_card_inplace_permutation'
            elif result['n_common'] > 0:
                result['failure_reason'] = f'same_card_partial_overlap_{result["n_common"]}'
            else:
                result['failure_reason'] = 'same_card_no_overlap'
        
        # 4. MASS INCREASE
        if s_out.n_points > s_in.n_points and s_in.n_points > 0:
            # Allow Lifting (Homothety/Extrapolation) for ANY ratio (e.g. 1.5x)
            if derive_lifting(s_in, s_out):
                result['lifting'] = True
                result['detected'] = True
                return result

            # Fallback to Union / Additive logic
            if derive_union(s_in, s_out):
                result['union'] = True
                result['detected'] = True
                return result
            
            # Failure Classification
            ratio = s_out.n_points / s_in.n_points
            
            if s_out.n_points % s_in.n_points == 0:
                 if result['is_additive']:
                     result['failure_reason'] = f'lifting_failed_mult{ratio:.2f}_is_additive'
                 else:
                     result['failure_reason'] = f'lifting_failed_mult{ratio:.2f}_mixed'
            else:
                 if result['is_additive']:
                     result['failure_reason'] = f'non_int_ratio_{ratio:.2f}_pure_additive'
                 else:
                     result['failure_reason'] = f'non_int_ratio_{ratio:.2f}_mixed'
        
        # 5. UNION (Multi-component)
        if derive_union(s_in, s_out):
            result['union'] = True
            result['detected'] = True
            return result
            
        # 6. CAUSALITY (Inert vs Causal)
        if derive_causality(s_in, s_out):
             result['union'] = True # Causality returns Union
             result['detected'] = True 
             return result
        
        if derive_composite_transform(s_in, s_out):
            result['composite'] = True
            result['detected'] = True
            return result
        
        # 6. Failure classification by component structure
        if not result['failure_reason']:
            comps_in = partition_by_connectivity(s_in)
            comps_out = partition_by_connectivity(s_out)
            if len(comps_in) == len(comps_out):
                result['failure_reason'] = f'union_same_comps_{len(comps_in)}'
            elif len(comps_in) > len(comps_out):
                result['failure_reason'] = f'union_deletion_{len(comps_in)}to{len(comps_out)}'
            else:
                result['failure_reason'] = f'union_creation_{len(comps_in)}to{len(comps_out)}'
        
        return result

