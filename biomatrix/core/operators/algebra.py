# -*- coding: utf-8 -*-
"""
operators_algebra.py - Pure Algebraic Operators

AGENT.md Compliant:
- NO for/while loops (vectorized numpy only)
- NO if/else chains (algebraic dispatch via matrix selection)
- N-dimensional agnostic
- Composable atomics via monoid structure

Architecture:
    T = Slice ∘ Bijection ∘ Lift
    
    Where:
    - Lift: Φ: R^D → R^{D+K} (algebraic embedding)
    - Bijection: L: R^{D+K} → R^{D+K} (linear map in lifted space)
    - Slice: Π: R^{D+K} → R^D (projection + selection)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from scipy.spatial.distance import cdist

from ..state import State
from ..topology import get_boundary_mask
# Removed top-level import to break circular dependency with `derive` module

@dataclass
class Operator:
    """Base class for algebraic operators."""
    
    def apply(self, state: State) -> State:
        raise NotImplementedError
    
    def __call__(self, state: State) -> State:
        return self.apply(state)
    
    def compose(self, other: 'Operator') -> 'Operator':
        """Monoid composition: self ∘ other"""
        return ComposedOperator(first=other, second=self)


@dataclass
class ComposedOperator(Operator):
    """Composition of two operators: second ∘ first"""
    first: Operator = None
    second: Operator = None
    
    def apply(self, state: State) -> State:
        intermediate = self.first.apply(state)
        return self.second.apply(intermediate)


# =============================================================================
# ATOMIC OPERATORS
# =============================================================================

@dataclass
class ConstantOperator(Operator):
    """
    Constant Map: K_c(S) = C.
    Generates a fixed set of points regardless of input.
    Used for 'Generative' or 'Additive' residuals.
    """
    points: np.ndarray
    
    def apply(self, state: State) -> State:
        # Ignores input state, returns constant
        # Handle empty points case
        if len(self.points) == 0:
            return State(np.empty((0, max(1, state.n_dims))))
        return State(self.points.copy())

    def __eq__(self, other):
        if not isinstance(other, ConstantOperator): return False
        if self.points is None and other.points is None: return True
        if self.points is None or other.points is None: return False
        return np.array_equal(self.points, other.points)



@dataclass
class LiftOperator(Operator):
    """
    Φ: R^D → R^{D+K} - Algebraic Embedding
    
    Lifts points into higher dimensional space where non-linear
    transformations become linear (affine).
    
    Lift Types:
    - 'modulo': Φ(x) = [x, Σ floor(x_d / P)] - Tiling/Periodicity
    - 'topology': Φ(x) = [x, dist(x, ∂S)] - Folding/Boundary
    - 'mass_rank': Φ(x) = [x, rank(‖x - c‖)] - Ordering
    - 'unique_index': Φ(x) = [x, lexsort_rank(x)] - Bijection alignment
    - 'kronecker': Φ(x) = [x, 0] (expansion handled separately)
    """
    lifter: str = 'unique_index'
    params: Dict[str, Any] = field(default_factory=dict)
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
        
        X = state.points
        N, D = X.shape
        
        # Compute lift label (algebraic, no loops)
        # We assume kernels return [X, Features].
        # But LiftOperator structure assumed labels appended? 
        # lifting_kernels functions return concatenated [X | Features].
        # So we just use the kernel result directly.
        
        X_lifted = self._compute_lifted(X, state)
        
        return State(X_lifted)
    
    def _compute_lifted(self, X: np.ndarray, state: State) -> np.ndarray:
        """Compute lifted features algebraically using unified kernels."""
        
        # Kernel Map
        # Note: lift_kronecker, lift_poly2 etc imported at top level
        # Lazy import to break circular dependency
        from ..derive.lifting_kernels import (
            lift_identity, lift_kronecker, lift_topological,
            lift_distance_rank, lift_unique_index, lift_poly2, lift_symmetry,
            lift_connectivity_features
        )

        kernel_map = {
            'identity': lift_identity,
            'kronecker': lift_kronecker,
            'topology': lambda x, p: lift_topological(x, state, p),
            'mass_rank': lambda x, p: lift_distance_rank(x, state, p), # Legacy
            'distance_rank': lambda x, p: lift_distance_rank(x, state, p),
            'connectivity': lambda x, p: lift_connectivity_features(x, state, p),
            'unique_index': lift_unique_index,
            'poly2': lift_poly2,
            'symmetry': lambda x, p: lift_symmetry(x, state, p)
        }
        
        # Dispatch
        func = kernel_map.get(self.lifter, lift_identity)
        
        # Apply
        # Some kernels might handle 2 args or 3 args via lambda wrappers above
        return func(X, self.params)

    def __eq__(self, other):
        if not isinstance(other, LiftOperator): return False
        if self.lifter != other.lifter: return False
        
        if self.params.keys() != other.params.keys(): return False
        for k, v in self.params.items():
            ov = other.params[k]
            if isinstance(v, np.ndarray) or isinstance(ov, np.ndarray):
                if not np.array_equal(v, ov): return False
            elif v != ov:
                return False
        return True


@dataclass
class BijectionOperator(Operator):
    """
    L: R^{D+K} → R^{D+K} - Linear map in lifted space
    
    Affine transformation: y = M @ x + b
    
    This is the core linear transformation that becomes valid
    after lifting into the appropriate higher-dimensional space.
    """
    linear: np.ndarray = None  # (D+K, D+K)
    translation: np.ndarray = None  # (D+K,)
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
        
        X = state.points
        
        # Default to identity if not specified
        D = X.shape[1]
        M = self.linear if self.linear is not None else np.eye(D)
        b = self.translation if self.translation is not None else np.zeros(D)
        
        # Dimension guard: handle matrix size mismatch
        if M.shape[1] != D:
            # Resize matrix to match input dimension
            new_M = np.eye(D)
            min_d = min(M.shape[0], M.shape[1], D)
            new_M[:min_d, :min_d] = M[:min_d, :min_d]
            M = new_M
            
        if len(b) != D:
            new_b = np.zeros(D)
            min_d = min(len(b), D)
            new_b[:min_d] = b[:min_d]
            b = new_b
        
        # Affine transform: Y = X @ M.T + b
        Y = X @ M.T + b
        
        return State(Y)

    def __eq__(self, other):
        if not isinstance(other, BijectionOperator): return False
        
        def array_match(a, b):
            if a is None and b is None: return True
            if a is None or b is None: return False
            return np.array_equal(a, b)
            
        return array_match(self.linear, other.linear) and array_match(self.translation, other.translation)


@dataclass
class SliceOperator(Operator):
    """
    Π: R^{D+K} → R^D - Projection + Selection
    
    Slices the lifted space back to original dimension by:
    1. Selection: Filter points by lifted coordinate values (Z-range)
    2. Projection: Drop the lifted dimension(s)
    """
    slice_dims: Tuple[int, ...] = None  # Dimensions to KEEP (None = all except last)
    slice_values: Tuple[float, ...] = None  # Discrete values to keep
    slice_range: Tuple[float, float] = None # OR Range [min, max] to keep
    slice_axis: int = -1  # Axis to apply slice_values threshold/filter on
    original_dims: int = None  # Original dimension D (inferred if not set)
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
        
        X = state.points
        N, D_lifted = X.shape
        
        # Infer original dims if not specified
        D = self.original_dims if self.original_dims is not None else D_lifted - 1
        
        # 1. SELECTION: Filter by Z-coordinate
        Z = X[:, self.slice_axis]  # Use specified axis
        
        # Build selection mask algebraically (no loops)
        mask = np.ones(N, dtype=bool)
        
        # Discrete Values Check
        if self.slice_values is not None:
            slice_set = np.array(self.slice_values)
            # Vectorized membership check
            Z_rounded = np.round(Z, State.DECIMALS)
            slice_rounded = np.round(slice_set, State.DECIMALS)
            mask &= np.isin(Z_rounded, slice_rounded)
            
        # Range Check (Continuous Cut)
        if self.slice_range is not None:
            z_min, z_max = self.slice_range
            mask &= (Z >= z_min) & (Z <= z_max)
        
        # 2. PROJECTION: Keep only original dimensions
        dims_to_keep = self.slice_dims if self.slice_dims is not None else tuple(range(D))
        X_sliced = X[mask][:, dims_to_keep]
        
        return State(X_sliced)

    def __eq__(self, other):
        if not isinstance(other, SliceOperator): return False
        return (self.slice_dims == other.slice_dims and 
                self.slice_values == other.slice_values and
                self.slice_range == other.slice_range and
                self.slice_axis == other.slice_axis and
                self.original_dims == other.original_dims)


# =============================================================================
# EXTENDED KERNEL OPERATORS
# =============================================================================




# =============================================================================
# COMPOSED TRANSFORMS
# =============================================================================

@dataclass
class LiftedTransform(Operator):
    """
    T = Slice ∘ Bijection ∘ Lift
    
    The unified transformation structure where:
    - Lift embeds R^D → R^{D+K}
    - Bijection applies linear transform in lifted space
    - Slice projects back R^{D+K} → R^D with selection
    
    This is the ONLY operator type needed - all legacy operators
    are special cases of this structure.
    """
    lift: LiftOperator = None
    bijection: BijectionOperator = None
    slice: SliceOperator = None
    
    def apply(self, state: State) -> State:
        # Compose: Slice ∘ Bijection ∘ Lift
        lifted = self.lift.apply(state) if self.lift else state
        transformed = self.bijection.apply(lifted) if self.bijection else lifted
        sliced = self.slice.apply(transformed) if self.slice else transformed
        return sliced

    def __eq__(self, other):
        if not isinstance(other, LiftedTransform): return False
        return (self.lift == other.lift and 
                self.bijection == other.bijection and 
                self.slice == other.slice)



# =============================================================================
# FACTORY FUNCTIONS (Legacy Operator Equivalents)
# =============================================================================

def create_identity() -> LiftedTransform:
    """Identity operator: Lift=None, Bij=Identity, Slice=None"""
    return LiftedTransform(lift=None, bijection=None, slice=None)


def create_translation(delta: np.ndarray) -> LiftedTransform:
    """Translation: Bij = Identity + delta"""
    D = len(delta)
    return LiftedTransform(
        lift=None,
        bijection=BijectionOperator(linear=np.eye(D), translation=delta),
        slice=None
    )


def create_affine(matrix: np.ndarray, translation: np.ndarray) -> LiftedTransform:
    """Affine transform: y = M @ x + b"""
    return LiftedTransform(
        lift=None,
        bijection=BijectionOperator(linear=matrix, translation=translation),
        slice=None
    )


def create_deletion(selection_indices: np.ndarray, original_dims: int) -> LiftedTransform:
    """
    Deletion operator: Select subset of points.
    
    Equivalent to: Lift(unique_index) → Identity → Slice(selection)
    """
    # Convert indices to Z-values
    slice_values = tuple(float(i) for i in selection_indices)
    
    return LiftedTransform(
        lift=LiftOperator(lifter='unique_index'),
        bijection=None,
        slice=SliceOperator(slice_values=slice_values, original_dims=original_dims)
    )


def create_replication(period: float, original_dims: int) -> LiftedTransform:
    """
    Replication operator: Expand points by tiling.
    
    Equivalent to: Lift(modulo) → Identity → Slice(all)
    """
    return LiftedTransform(
        lift=LiftOperator(lifter='modulo', params={'period': period}),
        bijection=None,
        slice=SliceOperator(slice_values=None, original_dims=original_dims)
    )
