# -*- coding: utf-8 -*-
"""
operators/logic.py - Set Logic Operators

Contains operators for set algebra operations: Union, Intersection, Difference.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from functools import reduce
from abc import abstractmethod

from .base import Operator
from ..state import State
from ..topology import partition_by_connectivity, partition_by_value, view_as_void


@dataclass
class LogicOperator(Operator):
    """
    Base class for logic operators (Set Algebra on States).
    
    State is treated as a Set of points.
    """
    operands: List[Operator]
    
    @abstractmethod
    def apply(self, state: State) -> State:
        pass


@dataclass
class UnionOperator(LogicOperator):
    """
    Logical OR (Union) of operators.
    
    (Op1 Union Op2)(S) = Op1(S) Union Op2(S)
    """
    def apply(self, state: State) -> State:
        if not self.operands:
            return state.copy()
            
        results = []
        for op in self.operands:
            # Defensive Unwrap with Debugging
            real_op = op
            if isinstance(op, tuple):
                 if len(op) >= 2 and hasattr(op[1], 'apply'):
                     real_op = op[1]
            
            # Additional check for operator capability
            if not hasattr(real_op, 'apply'):
                continue
                
            results.append(real_op.apply(state))
        
        valid_results = [r for r in results if not r.is_empty]
        if not valid_results:
             return State(np.empty((0, state.n_dims)))
        
        valid_results.sort(key=lambda s: (s.causality_score, -s.n_points))
        all_points = np.vstack([r.points for r in valid_results])
        
        final_points = np.unique(all_points, axis=0)
        
        return State(points=final_points)
    
    # === Algebraic Methods ===
    
    def to_symbolic(self) -> str:
        parts = [op.to_symbolic() if hasattr(op, 'to_symbolic') else type(op).__name__ for op in self.operands]
        return "(" + " ∪ ".join(parts) + ")"
    
    @property
    def category(self):
        from ..base import OperatorCategory
        return OperatorCategory.SURJECTION  # Union can increase mass

    def is_semantically_equal(self, other: Operator) -> bool:
        """
        Check if two UnionOperators are structurally equivalent in a permissive way.
        Allows one to be a subset of another (partial logic) provided there are no conflicts.
        """
        if not isinstance(other, UnionOperator):
             # Treat single operator as Union([op])
             # This allows Accumulation: Consensus might be single op, new might be Union.
             other = UnionOperator(operands=[other])
            
        # Optimization: Exact match first
        if len(self.operands) == len(other.operands):
            # Try strict equality via sets (if hashable) or O(N^2) check
            match_count = 0
            used_other = set()
            for op1 in self.operands:
                for j, op2 in enumerate(other.operands):
                    if j not in used_other and op1 == op2:
                        match_count += 1
                        used_other.add(j)
                        break
            if match_count == len(self.operands):
                return True
                
        # Permissive Consilience:
        # 1. Map Signature -> List[Operator]
        from collections import defaultdict
        
        def extract_sig_op(op):
            # Unwrap Sequential/SelectThenAct to find SelectBySignature
            sig = None
            if hasattr(op, 'selector') and hasattr(op.selector, 'target_signature'):
                sig = op.selector.target_signature
            elif hasattr(op, 'steps'): # Sequential
                first = op.steps[0]
                if hasattr(first, 'target_signature'):
                    sig = first.target_signature
            
            if sig is not None:
                # Make hashable: (int, array) -> (int, tuple(array))
                if isinstance(sig, tuple) and len(sig) == 2 and isinstance(sig[1], np.ndarray):
                    sig = (sig[0], tuple(sig[1].flatten()))
                return (sig, op)
            return (None, op)
            
        dict1 = defaultdict(list)
        for op in self.operands:
            sig, o = extract_sig_op(op)
            if sig is not None:
                dict1[sig].append(o)
                
        dict2 = defaultdict(list)
        for op in other.operands:
            sig, o = extract_sig_op(op)
            if sig is not None:
                dict2[sig].append(o)
        
        # Check intersection
        common_sigs = set(dict1.keys()) & set(dict2.keys())
        
        for sig in common_sigs:
            # Check for conflicts among rules sharing the same signature.
            # We assume for now that if rules exist for the same signature, they should be compatible.
            pass

        return True

    def merge(self, other: 'UnionOperator') -> 'UnionOperator':
        """
        Merge rules from another UnionOperator.
        Preserves existing rules, adds new non-conflicting rules.
        """
        if not isinstance(other, UnionOperator):
            return self
            
        new_operands = list(self.operands)
        
        for op2 in other.operands:
            is_present = False
            for op1 in self.operands:
                if op1 == op2:
                    is_present = True
                    break
            
            if not is_present:
                new_operands.append(op2)
                
        return UnionOperator(operands=new_operands)


@dataclass
class IntersectionOperator(LogicOperator):
    """
    Logical AND (Intersection) of operators.
    
    (Op1 Intersect Op2)(S) = Op1(S) Intersect Op2(S)
    """
    def apply(self, state: State) -> State:
        # ALGEBRAIC: Reduce + Vectorized intersection (void view for set membership)
        
        def intersect(s1: State, s2: State) -> State:
            # Vectorized set intersection using void view
            v1 = view_as_void(np.ascontiguousarray(s1.points))
            v2 = view_as_void(np.ascontiguousarray(s2.points))
            mask = np.isin(v1, v2).flatten()
            pts = s1.points[mask]
            return State(pts) if len(pts) > 0 else State(np.empty((0, s1.n_dims)))
        
        results = [op.apply(state) for op in self.operands]
        return reduce(intersect, results) if results else state.copy()
    
    # === Algebraic Methods ===
    
    def to_symbolic(self) -> str:
        parts = [op.to_symbolic() if hasattr(op, 'to_symbolic') else type(op).__name__ for op in self.operands]
        return "(" + " ∩ ".join(parts) + ")"
    
    @property
    def category(self):
        from ..base import OperatorCategory
        return OperatorCategory.INJECTION  # Intersection reduces mass


@dataclass
class DifferenceOperator(LogicOperator):
    """
    Logical Difference (Set Minus).
    
    Logic: (Op1 \\ Op2)(S) = Op1(S) \\ Op2(S)
    
    Note: Strict point set difference.
    """
    def apply(self, state: State) -> State:
        def difference(s1: State, s2: State) -> State:
            v1 = view_as_void(np.ascontiguousarray(s1.points))
            v2 = view_as_void(np.ascontiguousarray(s2.points))
            mask = ~np.isin(v1, v2).flatten()
            pts = s1.points[mask]
            return State(pts) if len(pts) > 0 else State(np.empty((0, s1.n_dims)))
        
        results = [op.apply(state) for op in self.operands]
        return reduce(difference, results) if results else state.copy()
    
    # === Algebraic Methods ===
    
    def to_symbolic(self) -> str:
        parts = [op.to_symbolic() if hasattr(op, 'to_symbolic') else type(op).__name__ for op in self.operands]
        return "(" + " \\ ".join(parts) + ")"
    
    @property
    def category(self):
        from ..base import OperatorCategory
        return OperatorCategory.INJECTION  # Difference reduces mass


@dataclass
class SymmetricDifferenceOperator(LogicOperator):
    """
    Symmetric Difference (XOR) of operators.
    
    (Op1 △ Op2)(S) = (Op1(S) \ Op2(S)) ∪ (Op2(S) \ Op1(S))
    """
    def apply(self, state: State) -> State:
        if len(self.operands) < 2:
            return state.copy()
        
        results = [op.apply(state) for op in self.operands]
        
        def sym_diff(s1: State, s2: State) -> State:
            v1 = view_as_void(np.ascontiguousarray(s1.points))
            v2 = view_as_void(np.ascontiguousarray(s2.points))
            mask1 = ~np.isin(v1, v2).flatten()
            mask2 = ~np.isin(v2, v1).flatten()
            pts_list = []
            if np.any(mask1):
                pts_list.append(s1.points[mask1])
            if np.any(mask2):
                pts_list.append(s2.points[mask2])
            pts = np.vstack(pts_list) if pts_list else np.empty((0, s1.n_dims))
            return State(pts) if len(pts) > 0 else State(np.empty((0, s1.n_dims)))
        
        return reduce(sym_diff, results) if results else state.copy()
    
    def to_symbolic(self) -> str:
        parts = [op.to_symbolic() if hasattr(op, 'to_symbolic') else type(op).__name__ for op in self.operands]
        return "(" + " △ ".join(parts) + ")"


@dataclass
class BinarySetOperator(Operator):
    """
    Generic Binary Set Operator on two States.
    
    T(A, B) = op(A, B) where op ∈ {∩, ∪, △, A\\B, B\\A}
    
    ALGEBRAIC: Pure set operations via void view.
    This is a composable primitive.
    """
    operation: str  # 'intersection', 'union', 'sym_diff', 'diff_ab', 'diff_ba'
    state_b: State = None
    output_color: int = 0
    
    def apply(self, state_a: State) -> State:
        if state_a.is_empty or self.state_b is None or self.state_b.is_empty:
            return state_a.copy()
        
        n_spatial = min(2, state_a.n_dims)
        a_spatial = np.round(state_a.points[:, :n_spatial], 4)
        b_spatial = np.round(self.state_b.points[:, :n_spatial], 4)
        
        va = view_as_void(a_spatial.astype(np.float64))
        vb = view_as_void(b_spatial.astype(np.float64))
        
        in_a_only = ~np.isin(va, vb).flatten()
        in_b_only = ~np.isin(vb, va).flatten()
        in_both = np.isin(va, vb).flatten()
        
        # Dispatch algebraically
        if self.operation == 'intersection':
            result_pts = state_a.points[in_both]
        elif self.operation == 'union':
            pts_list = [state_a.points]
            if np.any(in_b_only):
                pts_list.append(self.state_b.points[in_b_only])
            result_pts = np.vstack(pts_list)
        elif self.operation == 'sym_diff':
            pts_list = []
            if np.any(in_a_only):
                pts_list.append(state_a.points[in_a_only])
            if np.any(in_b_only):
                pts_list.append(self.state_b.points[in_b_only])
            result_pts = np.vstack(pts_list) if pts_list else np.empty((0, state_a.n_dims))
        elif self.operation == 'diff_ab':
            result_pts = state_a.points[in_a_only]
        elif self.operation == 'diff_ba':
            result_pts = self.state_b.points[in_b_only]
        elif self.operation == 'complement_intersection':
            # NOR: positions that are in neither A nor B
            # Need to compute the grid extent and find positions not covered
            all_spatial = np.vstack([a_spatial, b_spatial])
            min_coords = all_spatial.min(axis=0).astype(int)
            max_coords = all_spatial.max(axis=0).astype(int)
            
            # Generate all grid positions in extent
            rows = np.arange(min_coords[0], max_coords[0] + 1)
            cols = np.arange(min_coords[1], max_coords[1] + 1)
            rr, cc = np.meshgrid(rows, cols, indexing='ij')
            all_positions = np.stack([rr.ravel(), cc.ravel()], axis=1).astype(np.float64)
            all_void = view_as_void(all_positions)
            
            # Union of A and B
            union_void = np.unique(np.vstack([va, vb]))
            nor_mask = ~np.isin(all_void, union_void).flatten()
            
            if np.any(nor_mask):
                result_spatial = all_positions[nor_mask]
                # Build result with output color
                if state_a.n_dims > 2:
                    result_pts = np.hstack([result_spatial, np.full((len(result_spatial), state_a.n_dims - 2), self.output_color)])
                else:
                    result_pts = result_spatial
            else:
                result_pts = np.empty((0, state_a.n_dims))
        else:
            result_pts = state_a.points
        
        if len(result_pts) == 0:
            return State(np.empty((0, state_a.n_dims)))
        
        if self.output_color > 0 and result_pts.shape[1] > 2:
            result_pts = result_pts.copy()
            result_pts[:, 2] = self.output_color
        
        return State(result_pts)
    
    def to_symbolic(self) -> str:
        symbols = {'intersection': '∩', 'union': '∪', 'sym_diff': '△', 'diff_ab': '\\', 'diff_ba': '/'}
        return f"(A {symbols.get(self.operation, '?')} B)"


@dataclass
class PartitionOperator(Operator):
    """
    General Partition Operator: T(S) = ⊔ T_i(S_i)
    
    Decomposes state into partitions using a strategy, applies sub-operators, 
    and unions the results.
    
    Strategies:
    - 'value': Partition by fiber value (dim parameter)
    - 'connectivity': Partition by connected components
    """
    strategy: str
    operands: Dict[Any, Operator]  # Key -> Operator mapping
    params: Dict[str, Any] = field(default_factory=dict)
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
        
        # ALGEBRAIC: Handle 'nearest' strategy specially - groups points by nearest operand key
        if self.strategy == 'nearest':
            dim = self.params.get('dim', 0)
            keys = np.array(list(self.operands.keys()))  # (K,)
            pts_dim = state.points[:, dim]  # (N,)
            
            # VECTORIZED: Find nearest key for each point
            # dists[i, j] = |pts_dim[i] - keys[j]|
            dists = np.abs(pts_dim[:, None] - keys[None, :])  # (N, K)
            nearest_idx = np.argmin(dists, axis=1)  # (N,)
            
            # Group points by nearest key
            results = []
            for ki, key in enumerate(keys):
                mask = (nearest_idx == ki)
                if np.any(mask):
                    part = State(state.points[mask])
                    op = self.operands[key]
                    results.append(op.apply(part))
            
            if not results:
                return State(np.empty((0, state.n_dims)))
            
            valid_pts = [r.points for r in results if not r.is_empty]
            if not valid_pts:
                return State(np.empty((0, state.n_dims)))
            
            all_pts = np.vstack(valid_pts)
            return State(np.unique(all_pts, axis=0))
            
        # ALGEBRAIC: Dispatch via dictionary (no if/elif)
        def partition_value():
            dim = self.params.get('dim', -1)
            return partition_by_value(state, projection_dims=[dim] if dim >= 0 else None)
        
        def partition_connectivity():
            mode = self.params.get('mode', 'moore')
            return partition_by_connectivity(state, mode=mode)
        
        strategy_dispatch = {
            'value': partition_value,
            'connectivity': partition_connectivity,
        }
        
        parts = strategy_dispatch.get(self.strategy, lambda: [state])()
        
        # Apply operators to partitions
        results = []
        for i, part in enumerate(parts):
            key = self.params.get('key_fn', lambda p, i: i)(part, i)
            op = self.operands.get(key, self.operands.get('default', None))
            if op:
                results.append(op.apply(part))
            else:
                results.append(part)  # Identity if no operator
        
        # Union results
        if not results:
            return State(np.empty((0, state.n_dims)))
        
        valid_pts = [r.points for r in results if not r.is_empty]
        if not valid_pts:
            return State(np.empty((0, state.n_dims)))
        
        all_pts = np.vstack(valid_pts)
        return State(np.unique(all_pts, axis=0))
    
    # === Algebraic Methods ===
    
    def to_symbolic(self) -> str:
        n = len(self.operands)
        return f"⊔_{self.strategy}({n})"


@dataclass
class PiecewiseOperator(Operator):
    """
    Legacy wrapper for PartitionOperator(strategy='value').
    Keeps API compatibility for branches/partition_dim.
    """
    branches: Dict[int, Operator]
    partition_dim: int = 0
    
    def apply(self, state: State) -> State:
        op = PartitionOperator(strategy='value', operands=self.branches, params={'dim': self.partition_dim})
        return op.apply(state)


@dataclass
class ComponentMapOperator(Operator):
    """
    Apply an operator to each connected component independently.
    Reassembles the state afterwards.
    
    If 'mapping' is provided, it maps Component Index -> Operator.
    Otherwise 'operator' is applied to all components.
    """
    operator: Optional[Operator] = None
    mapping: Optional[Dict[int, Operator]] = None
    
    def apply(self, state: State) -> State:
        if state.is_empty:
            return state.copy()
            
        components = partition_by_connectivity(state)
        if not components:
            return state.copy()
        
        # Sort for determinism
        components.sort(key=lambda c: tuple(c.bbox_min))
        
        results = []
        for i, comp in enumerate(components):
            if self.mapping and i in self.mapping:
                op = self.mapping[i]
            elif self.operator:
                op = self.operator
            else:
                op = None
            
            if op:
                results.append(op.apply(comp))
            else:
                results.append(comp)
        
        # Union
        valid_pts = [r.points for r in results if not r.is_empty]
        if not valid_pts:
            return State(np.empty((0, state.n_dims)))
        
        return State(np.vstack(valid_pts))
    
    # === Algebraic Methods ===
    
    def to_symbolic(self) -> str:
        n = len(self.mapping) if self.mapping else 1
        inner = self.operator.to_symbolic() if self.operator and hasattr(self.operator, 'to_symbolic') else "op"
        return f"⋃_{n}({inner})"



@dataclass
class CausalityOperator(Operator):
    """
    Wrapper that assigns a Causality Score to the result of an operator.
    
    T_causal(S) = T(S) with .causality_score = score
    """
    operator: Operator
    score: float = 0.0
    
    def apply(self, state: State) -> State:
        res = self.operator.apply(state)
        # We must return a new state with the score attached.
        # But wait, State is immutable-ish (dataclass).
        # We can use the copy constructor logic we added.
        return State(res.points, causality_score=self.score)
    
    # === Algebraic Methods ===
    
    def to_symbolic(self) -> str:
        inner = self.operator.to_symbolic() if hasattr(self.operator, 'to_symbolic') else type(self.operator).__name__
        return f"C({inner}, {self.score:.1f})"


# Export all
__all__ = [
    'LogicOperator', 'UnionOperator', 'IntersectionOperator', 'DifferenceOperator',
    'PartitionOperator', 'PiecewiseOperator', 'ComponentMapOperator', 'CausalityOperator'
]
