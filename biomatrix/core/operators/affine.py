# -*- coding: utf-8 -*-
"""
operators/affine.py - Affine and Value Transformation Operators

Contains operators for affine transformations, value projections, and permutations.
"""

import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from functools import reduce

from .base import Operator
from ..state import State
from ..topology import partition_by_connectivity, get_interior, view_as_void

# NOTE: generate_lattice imported inline due to circular dependency
# from ..derive.lattice import generate_lattice


@dataclass
class ValueProjectionOperator(Operator):
    """
    Value Projection: Project a dimension to a constant value.
    P_d_c(S) = { (x0, ..., c, ... xn) for x in S }
    
    Mathematically: Collapses the state onto the hyperplane x[dim] = value.
    Used for 'Recolor' (projecting color dimension).
    """
    dim: int
    value: float
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
            
        new_points = state.points.copy()
        new_points[:, self.dim] = self.value
        
        return State(points=new_points)
    
    def to_symbolic(self) -> str:
        return f"π_d{self.dim}={self.value:.0f}"


@dataclass
class ValuePermutationOperator(Operator):
    """
    N-Dimensional Value Permutation Operator.
    
    Applies per-dimension value mappings:
    π: V_d -> V_d' for each dimension d.
    
    AGENT.md: Pure algebraic, N-dim agnostic.
    """
    permutation_maps: List[dict]  # List of {old_val: new_val} per dimension
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
            
        new_points = state.points.copy()
        n_dims = min(len(self.permutation_maps), state.n_dims)
        
        # ALGEBRAIC: Apply permutation per dimension using reduce
        def apply_dim_perm(pts, d):
            if d >= len(self.permutation_maps):
                return pts
            vmap = self.permutation_maps[d]
            lookup = np.vectorize(lambda v: vmap.get(np.round(v, State.DECIMALS), v))
            result = pts.copy()
            result[:, d] = lookup(pts[:, d])
            return result
        
        new_points = reduce(apply_dim_perm, range(n_dims), new_points)
        
        return State(points=new_points)


@dataclass
class ScaleOperator(Operator):
    """
    Scale Operator: N-Dimensional coordinate scaling (Preserves N).
    """
    scale_factors: np.ndarray
    centroid_in: np.ndarray
    centroid_out: np.ndarray
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
            
        centered = state.points - self.centroid_in
        scaled = centered * self.scale_factors
        new_points = scaled + self.centroid_out
        
        return State(new_points)
    
    def to_symbolic(self) -> str:
        s = ",".join(f"{v:.1f}" for v in self.scale_factors[:3])
        return f"S({s})"


@dataclass
class ResampleOperator(Operator):
    """
    Resample Operator: Algebraic Lattice (Basis) Homothety.
    
    T(L) = L', where L = { O + sum n_i v_i }.
    The operator scales the basis vectors: v'_i = scale_i * v_i.
    """
    origin: np.ndarray
    basis: np.ndarray
    counts: np.ndarray
    scale_factors: np.ndarray
    
    def apply(self, state: State) -> State:
        from ..derive.lattice import generate_lattice  # Inline to avoid circular import
        
        new_basis = self.basis * self.scale_factors
        new_counts = np.ceil(self.counts * self.scale_factors).astype(int)
        
        return generate_lattice(self.origin, new_basis, new_counts)
    
    def __repr__(self):
        return f"Resample(scale={self.scale_factors})"


@dataclass
class PermutationOperator(Operator):
    """
    Axis Permutation Operator (Transpose).
    
    Rearranges the dimensions of the state vector.
    P(x_0, x_1, ..., x_n) = (x_{p[0]}, x_{p[1]}, ..., x_{p[n]})
    """
    perm: List[int]
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
        
        # Validate permutation
        n_dims = state.n_dims
        if len(self.perm) != n_dims:
            return state.copy()
        
        new_points = state.points[:, self.perm]
        return State(new_points)
    
    def to_symbolic(self) -> str:
        p = ",".join(str(i) for i in self.perm)
        return f"P({p})"


@dataclass
class ClampOperator(Operator):
    """
    Clamp/Saturate coordinates to a bounding range.
    
    Unlike CropOperator (which removes points outside), this MOVES points to the boundary.
    """
    axis: int = 1
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
        
        new_points = state.points.copy()
        vals = new_points[:, self.axis]
        
        if self.min_val is not None:
            vals = np.maximum(vals, self.min_val)
        if self.max_val is not None:
            vals = np.minimum(vals, self.max_val)
        
        new_points[:, self.axis] = vals
        return State(new_points)


@dataclass
class NormalizeOriginOperator(Operator):
    """
    Normalize origin: shift points so bbox_min = origin.
    
    Dimension-agnostic: normalizes specified dimensions.
    """
    dims_to_normalize: List[int] = None
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
        
        new_points = state.points.copy()
        dims = self.dims_to_normalize or list(range(state.n_dims))
        
        for d in dims:
            if d < state.n_dims:
                new_points[:, d] -= new_points[:, d].min()
        
        return State(new_points)


@dataclass
class GlobalAffineOperator(Operator):
    """Global Affine Operator: T(p) = Ap + b applied to ALL points."""
    matrix: np.ndarray
    bias: np.ndarray
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
        
        if self.matrix.shape[0] != state.n_dims:
            return state.copy()
        
        new_points = state.points @ self.matrix.T + self.bias
        return State(new_points)
    
    # === Algebraic Methods ===
    
    def inverse(self) -> 'GlobalAffineOperator':
        """T⁻¹(x) = A⁻¹x - A⁻¹b."""
        from ..base import NotInvertibleError
        
        if not self.is_invertible:
            raise NotInvertibleError(f"GlobalAffineOperator is singular")
        
        M_inv = np.linalg.inv(self.matrix)
        b_inv = -M_inv @ self.bias
        return GlobalAffineOperator(matrix=M_inv, bias=b_inv)
    
    def to_symbolic(self) -> str:
        """Symbolic representation."""
        det = np.linalg.det(self.matrix)
        b_norm = np.linalg.norm(self.bias)
        
        parts = []
        if b_norm > 1e-6:
            parts.append(f"T({b_norm:.2f})")
        if not np.allclose(self.matrix, np.eye(self.matrix.shape[0])):
            parts.append(f"A(det={det:.2f})")
        
        return " ∘ ".join(parts) if parts else "Id"
    
    # === Algebraic Properties ===
    
    @property
    def is_invertible(self) -> bool:
        return abs(np.linalg.det(self.matrix)) > 1e-10
    
    @property
    def is_linear(self) -> bool:
        return np.allclose(self.bias, 0)
    
    @property
    def preserves_mass(self) -> bool:
        return self.is_invertible
    
    def __repr__(self):
        return f"GlobalAffine(M={self.matrix.shape})"


@dataclass
class RigidAffineForceOperator(Operator):
    """
    Rigid Affine Force Operator (Algebraic Monoïd).
    Applies T_i(p) = p + (A - I)c_i + b
    where c_i is the centroid of the component O_i containing point p.
    """
    matrix: np.ndarray
    offset: np.ndarray
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
        
        comps = partition_by_connectivity(state)
        if not comps:
            return state.copy()
        
        result_pts = []
        A_minus_I = self.matrix - np.eye(self.matrix.shape[0])
        
        for comp in comps:
            c = comp.centroid
            pts = comp.points + A_minus_I @ c + self.offset
            result_pts.append(pts)
        
        return State(np.vstack(result_pts))
    
    def __repr__(self):
        return f"RigidAffineForce(M={self.matrix.shape}, b={self.offset})"


@dataclass
class AffineTiling(Operator):
    """
    Affine Tiling Operator.
    Applies T(p) = Ap + b_j to points, where b_j is one of the translations.
    """
    matrix: np.ndarray
    translations: np.ndarray
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
        
        if self.matrix.shape[0] != state.n_dims:
            return state.copy()
        
        result_pts = []
        transformed_base = state.points @ self.matrix.T
        
        for t in self.translations:
            pts = transformed_base + t
            result_pts.append(pts)
        
        return State(np.vstack(result_pts))
    
    def to_symbolic(self) -> str:
        n = len(self.translations)
        return f"ATile({n})"
    
    def __repr__(self):
        return f"AffineTiling(M={self.matrix.shape}, |T|={len(self.translations)})"


@dataclass
class KernelAffineOperator(Operator):
    """Applies Affine Transformation in a Kernel-Lifted Feature Space."""
    kernel_type: str
    matrix: np.ndarray
    bias: np.ndarray
    
    def apply(self, state: State) -> State:
        X = state.points
        if len(X) == 0: return state.copy()
        
        Phi = self._compute_features(X)
        if Phi.shape[1] != self.matrix.shape[1]:
            return state.copy()
        
        Y = Phi @ self.matrix.T + self.bias
        return State(Y)
        
    def _compute_features(self, X: np.ndarray) -> np.ndarray:
        N, D = X.shape
        
        if self.kernel_type == 'poly2':
            linear = X
            squares = X ** 2
            cross_tensor = X[:, :, None] * X[:, None, :]
            triu_indices = np.triu_indices(D, k=1)
            cross_terms = cross_tensor[:, triu_indices[0], triu_indices[1]]
            return np.hstack([linear, squares, cross_terms])
        elif self.kernel_type == 'radial':
            r = np.linalg.norm(X, axis=1, keepdims=True)
            return np.hstack([X, r])
        elif self.kernel_type == 'rank':
            ranks = np.argsort(np.argsort(X, axis=0), axis=0).astype(float)
            return np.hstack([X, ranks])
        return X


@dataclass
class RigidHomotheticForceOperator(Operator):
    """
    Rigid Homothetic Force Operator (Algebraic Monoïd).
    Applies P' = p + (k - 1)(c_i - C) to component centroids.
    
    Represents attraction/repulsion from a center C in R^D.
    """
    center: np.ndarray
    scale: float
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
        
        comps = partition_by_connectivity(state)
        if not comps:
            return state.copy()
        
        result_pts = []
        k_1 = self.scale - 1.0
        
        for comp in comps:
            c = comp.centroid
            disp = k_1 * (c - self.center)
            pts = comp.points + disp
            result_pts.append(pts)
        
        return State(np.vstack(result_pts))
    
    def __repr__(self):
        return f"RigidHomothetic(scale={self.scale})"


@dataclass
class RankByMassOperator(Operator):
    """
    Rank Transform: T(x)[value_dim] = rank(fiber(x), by_mass).
    
    N-DIM AGNOSTIC pure matrix chain.
    """
    fiber_dim: int
    value_dim: int
    order: str = 'descending'
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
        
        pts = state.points.copy()
        fiber_vals = pts[:, self.fiber_dim]
        
        # Compute mass per fiber
        unique_fibers, inverse, counts = np.unique(fiber_vals, return_inverse=True, return_counts=True)
        
        # Rank fibers by mass
        mass_order = np.argsort(counts)
        if self.order == 'descending':
            mass_order = mass_order[::-1]
        
        # Create rank mapping
        ranks = np.empty_like(mass_order)
        ranks[mass_order] = np.arange(len(mass_order)) + 1
        
        # Apply rank to each point
        pts[:, self.value_dim] = ranks[inverse]
        
        return State(pts)
    
    def __repr__(self):
        return f"RankByMass(fiber={self.fiber_dim}, value={self.value_dim})"


@dataclass
class AdditiveOperator(Operator):
    """
    Additive Operator: S_out = S_in (+) Delta.
    
    Represents mass increase where Output = Input UNION some derived Delta.
    """
    delta_points: Optional[np.ndarray] = None
    delta_transform: Optional[Operator] = None
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
            
        result_points = [state.points]
        
        if self.delta_transform is not None:
            delta_state = self.delta_transform.apply(state)
            if not delta_state.is_empty:
                result_points.append(delta_state.points)
        elif self.delta_points is not None:
            result_points.append(self.delta_points)
            
        all_points = np.vstack(result_points)
        unique_points = np.unique(all_points, axis=0)
        return State(unique_points)
        
    def __repr__(self):
        if self.delta_transform:
            return f"Additive(T={self.delta_transform})"
        return f"Additive(|Delta|={len(self.delta_points) if self.delta_points is not None else 0})"


@dataclass
class InteriorOperator(Operator):
    """
    Topological Interior Operator.
    Returns the interior points of the state components.
    """
    def apply(self, state: State) -> State:
        return get_interior(state)


@dataclass
class SortAndAlignOperator(Operator):
    """
    Sort & Align Operator (Algebraic Chain).
    Organizes components into a sequential chain.
    """
    sort_dim: int
    align_dims: List[int]
    origin: Optional[np.ndarray] = None
    padding: float = 0.0
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
        
        comps = partition_by_connectivity(state)
        if not comps:
            return state.copy()
        
        # Sort by centroid along sort_dim
        comps.sort(key=lambda c: c.centroid[self.sort_dim])
        
        # Compute cumulative offset
        result_pts = []
        current_offset = np.zeros(state.n_dims)
        
        if self.origin is not None:
            current_offset = self.origin - comps[0].bbox_min
        
        for comp in comps:
            pts = comp.points + current_offset
            result_pts.append(pts)
            
            # Update offset for next component
            for d in self.align_dims:
                if d < state.n_dims:
                    spread = comp.bbox_max[d] - comp.bbox_min[d]
                    current_offset[d] += spread + self.padding
        
        return State(np.vstack(result_pts))
    
    def __repr__(self):
        return f"SortAndAlign(sort={self.sort_dim}, align={self.align_dims})"


@dataclass
class ProjectiveSelectionOperator(Operator):
    """
    Algebraic Stacking / Occlusion resolution.
    Collisions occur when points coincide in the 'projection' subspace.
    Priority is determined by the value in the 'priority' dimension.
    """
    projection_dims: List[int]
    priority_dim: int
    descending: bool = True
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
        
        pts = state.points
        proj_pts = pts[:, self.projection_dims]
        priorities = pts[:, self.priority_dim]
        
        # Group by projection using void view
        proj_void = view_as_void(np.round(proj_pts, State.DECIMALS))
        
        unique_projs, inverse = np.unique(proj_void, return_inverse=True)
        
        # For each group, select point with highest/lowest priority
        result_indices = []
        for i in range(len(unique_projs)):
            group_mask = inverse == i
            group_priorities = priorities[group_mask]
            group_indices = np.where(group_mask)[0]
            
            if self.descending:
                winner = group_indices[np.argmax(group_priorities)]
            else:
                winner = group_indices[np.argmin(group_priorities)]
            
            result_indices.append(winner)
        
        return State(pts[result_indices])


# Export all
__all__ = [
    'ValueProjectionOperator', 'ValuePermutationOperator', 'ScaleOperator',
    'ResampleOperator', 'PermutationOperator', 'ClampOperator',
    'NormalizeOriginOperator', 'GlobalAffineOperator', 'RigidAffineForceOperator',
    'RigidHomotheticForceOperator', 'RankByMassOperator', 'AdditiveOperator',
    'InteriorOperator', 'SortAndAlignOperator', 'ProjectiveSelectionOperator',
    'KernelAffineOperator'
]
