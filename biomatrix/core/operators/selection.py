# -*- coding: utf-8 -*-
"""
operators/selection.py - Selection Operators

Contains operators for selecting subsets of states based on various criteria.
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

from .base import Operator
from ..state import State
from ..topology import partition_by_connectivity
from ..signatures import compute_universal_signature, signatures_match


@dataclass
class CropToComponentOperator(Operator):
    """
    Projects the state onto the subspace defined by a connected component.
    (i.e., 'Attention' or 'Focus' on a specific object).
    Points not belonging to the component are discarded.
    """
    component_index: int = -1
    indices: Optional[np.ndarray] = None
    
    def apply(self, state: State) -> State:
        # Case 1: Explicit indices (Algebraic Subset)
        if self.indices is not None:
             if len(self.indices) == 0:
                 return State(np.empty((0, state.n_dims)))
             mask = np.zeros(state.n_points, dtype=bool)
             # Handle indices out of bounds
             valid_indices = self.indices[self.indices < state.n_points]
             mask[valid_indices] = True
             return state.mask(mask)

        # Ensure consistent sorting
        components = partition_by_connectivity(state)
        if components:
            components.sort(key=lambda c: tuple(c.bbox_min))
            
        if not components or self.component_index < 0 or self.component_index >= len(components):
            return State(np.empty((0, state.n_dims)))
            
        return components[self.component_index]

    def __eq__(self, other):
        if not isinstance(other, CropToComponentOperator): return False
        if self.component_index != other.component_index: return False
        
        if self.indices is None and other.indices is None: return True
        if self.indices is None or other.indices is None: return False
        return np.array_equal(self.indices, other.indices)


@dataclass
class SelectBySignatureOperator(Operator):
    """
    Selects components matching a specific Algebraic Invariant Signature.
    (Causal Abstraction: "Select the Red/Large/Square Object").
    
    Attributes:
        target_signature: (mass, spectrum) tuple from compute_universal_signature
        tolerance: float tolerance for spectral matching
    """
    target_signature: Tuple[int, np.ndarray] = field(default_factory=lambda: (0, np.array([])))
    connectivity_mode: str = None # Auto-detect (High-D adaptive)
    tolerance: float = 1e-4

    def apply(self, state: State) -> State:
        # 1. Decompose State into Algebra of Components
        components = partition_by_connectivity(state, mode=self.connectivity_mode)
        if not components:
            return State(np.zeros((0, state.n_dims)))
            
        # 2. Compute Signatures & Match - VECTORIZED (no explicit loops)
        state_rounded = np.ascontiguousarray(np.round(state.points, 5))
        state_void = state_rounded.view(dtype=[('', state_rounded.dtype)] * state_rounded.shape[1]).ravel()
        
        # MAP/FILTER: Find matching components and collect their points
        def is_matching_component(comp):
            sig = compute_universal_signature(comp.points)
            return signatures_match(sig, self.target_signature, self.tolerance)
        
        matching_comps = list(filter(is_matching_component, components))
        
        # FALLBACK: High-D Manifold Repair (Gaps)
        # If no components match (likely due to fragmentation), try k-NN connectivity
        # to bridge small gaps (Distance < k-NN).
        if not matching_comps and self.connectivity_mode is None:
            # Heuristic: Try repairing with k=5 nearest neighbors
            components_knn = partition_by_connectivity(state, mode='knn', k=5)
            if components_knn: # Only if k-NN succeeds
                matching_comps = list(filter(is_matching_component, components_knn))

        # Build mask vectorized via void view membership
        if not matching_comps:
            return State(np.zeros((0, state.n_dims)))
        
        # Concatenate all matching component points
        all_match_pts = np.vstack([np.round(c.points, 5) for c in matching_comps])
        match_void = all_match_pts.view(dtype=[('', all_match_pts.dtype)] * all_match_pts.shape[1]).ravel()
        
        # Vectorized membership check
        full_mask = np.isin(state_void, match_void)
                        
        return state.mask(full_mask)

    def __eq__(self, other):
        if not isinstance(other, SelectBySignatureOperator): return False
        if abs(self.tolerance - other.tolerance) > 1e-9: return False
        
        # Compare target signatures (int, array)
        s1 = self.target_signature
        s2 = other.target_signature
        
        if s1[0] != s2[0]: return False
        if len(s1) > 1 and len(s2) > 1:
            # Check array
            if not np.array_equal(s1[1], s2[1]): return False
            
        return True


@dataclass
class SelectThenActOperator(Operator):
    """
    Selection + Action Composition.
    
    1. Selects a subset of the state using 'selector'.
    2. Applies 'operator' to the selected subset.
    3. Returns the result of the action.
    
    This is the building block for Piecewise transformations where different
    atoms of the input transform differently.
    """
    selector: Operator
    operator: Operator
    
    def apply(self, state: State) -> State:
        # 1. Select
        selected = self.selector.apply(state)
        if selected.is_empty:
            return selected
            
        # 2. Act
        return self.operator.apply(selected)
    
    def __repr__(self):
        return f"Select({self.selector}) -> {self.operator}"

    def __eq__(self, other):
        if not isinstance(other, SelectThenActOperator): return False
        return (self.selector == other.selector and self.operator == other.operator)


@dataclass
class SelectByValueOperator(Operator):
    """CAUSAL Selector: Select points where dim == value."""
    dim: int = 0
    value: float = 0.0
    tolerance: float = 1e-6
    
    def apply(self, state: State) -> State:
        if state.n_points == 0:
            return state.copy()
            
        vals = state.points[:, self.dim]
        mask = np.isclose(vals, self.value, atol=self.tolerance)
        
        return state.mask(mask)
    
    # === Algebraic Methods ===
    
    def to_symbolic(self) -> str:
        return f"σ(d{self.dim}={self.value:.1f})"
    
    # === Algebraic Properties ===
    
    @property
    def is_idempotent(self) -> bool:
        return True
    
    @property
    def category(self):
        from ..base import OperatorCategory
        return OperatorCategory.PROJECTION
    
    @property
    def preserves_mass(self) -> bool:
        return False  # Reduces mass

    def __eq__(self, other):
        if not isinstance(other, SelectByValueOperator): return False
        return (self.dim == other.dim and 
                abs(self.value - other.value) < 1e-9 and
                abs(self.tolerance - other.tolerance) < 1e-9)


@dataclass
class SelectByRangeOperator(Operator):
    """
    Select points where DIMENSION is within [min_val, max_val].
    
    Algebraic N-Dim Agnostic (AGENT.md):
    - Dimension is an explicit index
    - Range is derived algebraically from transformation analysis
    """
    dim: int = 0
    min_val: float = 0.0
    max_val: float = 1.0
    inclusive: bool = True
    
    def apply(self, state: State) -> State:
        if state.n_points == 0:
            return state.copy()
            
        vals = state.points[:, self.dim]
        if self.inclusive:
            mask = (vals >= self.min_val) & (vals <= self.max_val)
        else:
            mask = (vals > self.min_val) & (vals < self.max_val)
        
        return state.mask(mask)


@dataclass
class SortAndSelectOperator(Operator):
    """
    Selects a component based on its Spatial Rank along an axis.
    (Spatial Refinement: "Select the Top-Most/Left-Most Object").
    
    Attributes:
        axis: int (Dimension index 0..D-1)
        rank: int (0 for min/first, -1 for max/last, etc.)
    """
    axis: int = 0
    rank: int = 0
    connectivity_mode: str = None

    def apply(self, state: State) -> State:
        components = partition_by_connectivity(state, mode=self.connectivity_mode)
        if not components:
            return State(np.zeros((0, state.n_dims)))
            
        # Ensure axis is valid relative to data
        if self.axis >= state.n_dims:
            return State(np.zeros((0, state.n_dims)))

        # Sort Algebraically by Centroid projection
        components.sort(key=lambda c: c.centroid[self.axis])
        
        # Select Rank
        try:
            selected_c = components[self.rank]
            c_set = set(map(tuple, selected_c.points))
            mask = np.array([tuple(p) in c_set for p in state.points])
            return state.mask(mask)
        except IndexError:
            return State(np.zeros((0, state.n_dims)))

    def __eq__(self, other):
        if not isinstance(other, SortAndSelectOperator): return False
        return (self.axis == other.axis and 
                self.rank == other.rank and 
                self.connectivity_mode == other.connectivity_mode)


@dataclass
class DeleteOperator(Operator):
    """
    Delete Transformation: Remove points based on condition.
    
    MONOIDE: Transformation that reduces mass.
    T_delete: U → U' where |U'| < |U|
    
    Algebraic: Points are filtered using vectorized boolean mask.
    No explicit loop.
    """
    dim: int  # Dimension to check
    value: float  # Value to match
    keep: bool = True  # If True, KEEP matching points. If False, DELETE them.
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
        
        # Vectorized condition check (algebraic, no loop)
        mask = np.isclose(state.points[:, self.dim], self.value)
        
        # Invert mask if we want to DELETE matches (keep non-matches)
        final_mask = mask if self.keep else ~mask
        
        filtered_points = state.points[final_mask]
        
        if len(filtered_points) == 0:
            return State(np.empty((0, state.n_dims)))
        
        return State(points=filtered_points)
    
    # === Algebraic Methods ===
    
    def to_symbolic(self) -> str:
        mode = "keep" if self.keep else "del"
        return f"δ({mode} d{self.dim}={self.value:.1f})"
    
    @property
    def category(self):
        from ..base import OperatorCategory
        return OperatorCategory.INJECTION
    
    @property
    def preserves_mass(self) -> bool:
        return False


@dataclass
class FilterOperator(Operator):
    """
    Filter operator: keep points matching a predicate.
    
    Mathematical property: P^2 = P (idempotent).
    Applying the same projection twice gives the same result as once.
    
    For dimension-agnostic projection, predicate is defined as:
        - dimension index
        - comparison ('==', '<', '>', '<=', '>=')
        - threshold value
    """
    dim: int
    comparison: str
    threshold: float
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
        
        values = state.points[:, self.dim]
        
        # ALGEBRAIC: Dictionary-based comparison dispatch (no if/elif chain)
        eq = np.isclose(values, self.threshold)
        gt = values > self.threshold
        lt = values < self.threshold
        
        comparators = {
            '==': eq,
            '>': gt,
            '<': lt,
            '>=': gt | eq,
            '<=': lt | eq,
            '!=': ~eq,
        }
        mask = comparators.get(self.comparison, np.ones(len(values), dtype=bool))
        
        projected_points = state.points[mask]
        
        return State(points=projected_points)
    
    # === Algebraic Methods ===
    
    def to_symbolic(self) -> str:
        return f"π(d{self.dim}{self.comparison}{self.threshold:.1f})"
    
    @property
    def category(self):
        from ..base import OperatorCategory
        return OperatorCategory.PROJECTION
    
    @property
    def is_idempotent(self) -> bool:
        return True
    
    @property
    def preserves_mass(self) -> bool:
        return False


# Alias for backward compatibility
ProjectionOperator = FilterOperator


@dataclass
class ModularProjectionOperator(Operator):
    """
    Selects points satisfying a modular arithmetic condition on a projection.
    Generalizes Parity/Checkerboard.
    Condition: (weights . (x - origin)) % mod in remainders
    """
    weights: np.ndarray  # (D,) integer weights
    origin: np.ndarray   # (D,) origin
    mod: int = 2
    remainders: List[int] = field(default_factory=lambda: [0])
    
    def apply(self, state: State) -> State:
        if state.is_empty: return state.copy()
        
        # Relative coordinates (rounded)
        rel = np.round(state.points - self.origin, State.DECIMALS).astype(int)
        
        # Projection (Dot Product)
        # Handle weights broadcasting if needed, but usually (D,)
        proj = np.dot(rel, self.weights)
        
        # Modulo
        m = proj % self.mod
        
        # Mask
        # Vectorized check for multiple remainders
        # m[:, None] == remainders[None, :]
        mask = np.isin(m, self.remainders)
        
        return state.mask(mask)



@dataclass
class ExplicitFilterOperator(Operator):
    """
    Selects points matching an explicit set of coordinates (Relative to Origin).
    Used as a fallback for complex pattern selection in Lattices.
    """
    selected_points: State
    origin: np.ndarray = None
    
    def apply(self, state: State) -> State:
        if state.is_empty: return state.copy()
        
        # If origin provided, match relative coords (Shift Invariant)
        # If not, match absolute (Spatial Invariant)
        
        if self.origin is not None:
             # Normalize state and selection to origin
             s_rel = np.round(state.points - self.origin, State.DECIMALS)
             sel_rel = np.round(self.selected_points.points - self.origin, State.DECIMALS)
             
             # Set Intersection
             # Vectorized isin
             # View as void for N-dim
             # (Assuming s_rel and sel_rel are float)
             
             # Primitives
             set_sel = set(map(tuple, sel_rel))
             mask = np.array([tuple(p) in set_sel for p in s_rel])
             return state.mask(mask)
             
        # Absolute match
        return state.intersection(self.selected_points)

@dataclass
class ConstantOperator(Operator):
    """
    Returns a constant state, regardless of input.
    Used for Ex Nihilo generation (0 -> N) or Replace All.
    """
    constant: State
    
    def apply(self, state: State) -> State:
        return self.constant.copy()

__all__ = [
    'CropToComponentOperator', 'SelectBySignatureOperator', 'SelectThenActOperator',
    'SelectByValueOperator', 'SelectByRangeOperator', 'SortAndSelectOperator',
    'DeleteOperator', 'FilterOperator', 'ProjectionOperator'
]
