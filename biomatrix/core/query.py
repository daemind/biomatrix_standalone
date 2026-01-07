#!/usr/bin/env python3
"""
query.py - Query Classes for Select-Then-Act Pipeline.

Implements the Separation Theorem: T = B ∘ P where:
- P = Query (Attention) → Generalized as RULE
- B = Matrix (Action) → Generalized as COEFFICIENTS

Each Query has:
- select(state) → boolean mask
- name() → human-readable introspection string
- cost() → MDL complexity for ranking
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from abc import ABC, abstractmethod

from .state import State
from .topology import partition_by_connectivity, get_interior


class Query(ABC):
    """Base class for selection queries (P in T = B ∘ P)."""
    
    @abstractmethod
    def select(self, state: State) -> np.ndarray:
        """Return boolean mask of selected points."""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Human-readable introspection string."""
        pass
    
    def cost(self) -> int:
        """MDL complexity for ranking (lower = simpler)."""
        return 1


class QueryAll(Query):
    """Select all points (identity query)."""
    
    def select(self, state: State) -> np.ndarray:
        return np.ones(state.n_points, dtype=bool)
    
    def name(self) -> str:
        return "ALL"
    
    def cost(self) -> int:
        return 0  # Simplest


class QueryByDimValue(Query):
    """Select points where dimension d equals value v."""
    
    def __init__(self, dim: int, value: float):
        self.dim = dim
        self.value = value
        
    def select(self, state: State) -> np.ndarray:
        return np.isclose(state.points[:, self.dim], self.value, atol=0.1)
    
    def name(self) -> str:
        return f"dim[{self.dim}]=={self.value}"
    
    def cost(self) -> int:
        return 1


class QueryNot(Query):
    """Invert any query (logical NOT)."""
    
    def __init__(self, inner: Query):
        self.inner = inner
        
    def select(self, state: State) -> np.ndarray:
        return ~self.inner.select(state)
    
    def name(self) -> str:
        return f"NOT({self.inner.name()})"
    
    def cost(self) -> int:
        return self.inner.cost() + 1


class QueryLargestComponent(Query):
    """Select the largest connected component."""
    
    def select(self, state: State) -> np.ndarray:
        if state.n_points == 0:
            return np.zeros(0, dtype=bool)
            
        components = partition_by_connectivity(state)
        
        if not components:
            return np.ones(state.n_points, dtype=bool)
            
        largest = max(components, key=lambda c: c.n_points)
        largest_set = set(map(tuple, np.round(largest.points, 3)))
        mask = np.array([tuple(np.round(p, 3)) in largest_set for p in state.points])
        
        return mask
    
    def name(self) -> str:
        return "LARGEST_COMPONENT"
    
    def cost(self) -> int:
        return 2


class QuerySmallestComponent(Query):
    """Select the smallest connected component."""
    
    def select(self, state: State) -> np.ndarray:
        if state.n_points == 0:
            return np.zeros(0, dtype=bool)
            
        components = partition_by_connectivity(state)
        
        if not components:
            return np.ones(state.n_points, dtype=bool)
            
        smallest = min(components, key=lambda c: c.n_points)
        smallest_set = set(map(tuple, np.round(smallest.points, 3)))
        mask = np.array([tuple(np.round(p, 3)) in smallest_set for p in state.points])
        
        return mask
    
    def name(self) -> str:
        return "SMALLEST_COMPONENT"
    
    def cost(self) -> int:
        return 2


class QueryModeValue(Query):
    """Select points where dimension d equals the MODE (most frequent value)."""
    
    def __init__(self, dim: int):
        self.dim = dim
        
    def select(self, state: State) -> np.ndarray:
        if state.n_points == 0:
            return np.zeros(0, dtype=bool)
            
        vals = np.round(state.points[:, self.dim], 1)
        unique, counts = np.unique(vals, return_counts=True)
        mode_val = unique[np.argmax(counts)]
        
        return np.isclose(state.points[:, self.dim], mode_val, atol=0.1)
    
    def name(self) -> str:
        return f"MODE(dim[{self.dim}])"
    
    def cost(self) -> int:
        return 2


class QueryTouchingBorder(Query):
    """Select points touching the grid border."""
    
    def __init__(self, spatial_dims: int = 2):
        self.spatial_dims = spatial_dims
        
    def select(self, state: State) -> np.ndarray:
        if state.n_points == 0:
            return np.zeros(0, dtype=bool)
            
        mins = state.points[:, :self.spatial_dims].min(axis=0)
        maxs = state.points[:, :self.spatial_dims].max(axis=0)
        
        mask = np.zeros(state.n_points, dtype=bool)
        for d in range(self.spatial_dims):
            mask |= np.isclose(state.points[:, d], mins[d], atol=0.1)
            mask |= np.isclose(state.points[:, d], maxs[d], atol=0.1)
        
        return mask
    
    def name(self) -> str:
        return "TOUCHING_BORDER"
    
    def cost(self) -> int:
        return 1


class QueryNotTouchingBorder(Query):
    """Select points NOT touching the border (interior)."""
    
    def __init__(self, spatial_dims: int = 2):
        self.inner = QueryTouchingBorder(spatial_dims)
        
    def select(self, state: State) -> np.ndarray:
        return ~self.inner.select(state)
    
    def name(self) -> str:
        return "INTERIOR"
    
    def cost(self) -> int:
        return 1


class QueryEnclosed(Query):
    """Select points that are enclosed (inside a hole)."""
    
    def select(self, state: State) -> np.ndarray:
        if state.n_points == 0:
            return np.zeros(0, dtype=bool)
            
        interior = get_interior(state)
        
        if interior is None or interior.n_points == 0:
            return np.zeros(state.n_points, dtype=bool)
            
        interior_set = set(map(tuple, np.round(interior.points, 3)))
        mask = np.array([tuple(np.round(p, 3)) in interior_set for p in state.points])
        
        return mask
    
    def name(self) -> str:
        return "ENCLOSED"
    
    def cost(self) -> int:
        return 2


class QueryIsSquare(Query):
    """Select connected components that form a square shape."""
    
    def select(self, state: State) -> np.ndarray:
        if state.n_points == 0:
            return np.zeros(0, dtype=bool)
            
        components = partition_by_connectivity(state)
        
        mask = np.zeros(state.n_points, dtype=bool)
        
        for comp in components:
            if comp.n_points == 0:
                continue
            
            # N-Dimensional Hypercube Check
            # All dimensions with non-zero spread must be roughly equal
            mins = comp.points.min(axis=0)
            maxs = comp.points.max(axis=0)
            dims = maxs - mins + 1
            
            # Filter non-flat dimensions (dims > 1)
            active_dims = dims[dims > 1]
            
            if len(active_dims) >= 2: # Need at least 2 dims to form a "shape"
                mean_dim = np.mean(active_dims)
                # Check deviation from mean (are all sides equal?)
                deviation = np.max(np.abs(active_dims - mean_dim))
                
                # Relative tolerance of 10%
                if deviation / mean_dim < 0.1:
                    comp_set = set(map(tuple, np.round(comp.points, 3)))
                    for i, p in enumerate(state.points):
                        if tuple(np.round(p, 3)) in comp_set:
                            mask[i] = True
        
        return mask
    
    def name(self) -> str:
        return "IS_SQUARE"
    
    def cost(self) -> int:
        return 2



class QueryIntersection(Query):
    """Logical AND of two queries."""
    
    def __init__(self, q1: Query, q2: Query):
        self.q1 = q1
        self.q2 = q2
        
    def select(self, state: State) -> np.ndarray:
        return self.q1.select(state) & self.q2.select(state)
        
    def name(self) -> str:
        return f"({self.q1.name()} AND {self.q2.name()})"
        
    def cost(self) -> int:
        return self.q1.cost() + self.q2.cost() + 1


class QueryUnion(Query):
    """Logical OR of two queries."""
    
    def __init__(self, q1: Query, q2: Query):
        self.q1 = q1
        self.q2 = q2
        
    def select(self, state: State) -> np.ndarray:
        return self.q1.select(state) | self.q2.select(state)
        
    def name(self) -> str:
        return f"({self.q1.name()} OR {self.q2.name()})"
        
    def cost(self) -> int:
        return self.q1.cost() + self.q2.cost() + 1


def generate_candidate_queries(state: State) -> List[Query]:
    """
    Generate all candidate queries for a given state.
    Sorted by MDL cost (simpler first).
    """
    candidates = [QueryAll()]
    
    # Dimension-based queries
    for d in range(state.n_dims):
        unique_vals = np.unique(state.points[:, d])
        for v in unique_vals:
            candidates.append(QueryByDimValue(dim=d, value=v))
        candidates.append(QueryModeValue(dim=d))
    
    # Component queries
    candidates.append(QueryLargestComponent())
    candidates.append(QuerySmallestComponent())
    
    # NOT versions
    for d in range(state.n_dims):
        unique_vals = np.unique(state.points[:, d])
        for v in unique_vals:
            candidates.append(QueryNot(QueryByDimValue(dim=d, value=v)))
    
    # Gestalt queries
    candidates.append(QueryTouchingBorder())
    candidates.append(QueryNotTouchingBorder())
    candidates.append(QueryEnclosed())
    candidates.append(QueryIsSquare())
    
    # Simple Compositional Queries (Depth 1)
    # Combine Dimension queries with Gestalt queries (e.g., Red Square, Blue Border)
    # Limit combinatorial explosion: Only mix cost=1 queries
    
    base_queries = [q for q in candidates if q.cost() <= 1 and isinstance(q, (QueryByDimValue, QueryTouchingBorder))]
    
    # Pairwise Intersection
    for i in range(len(base_queries)):
        for j in range(i + 1, len(base_queries)):
            q1 = base_queries[i]
            q2 = base_queries[j]
            
            # Avoid contradictory combinations (e.g. Color=1 AND Color=2)
            if isinstance(q1, QueryByDimValue) and isinstance(q2, QueryByDimValue) and q1.dim == q2.dim:
                continue
                
            candidates.append(QueryIntersection(q1, q2))
            
    # Pairwise Union (Disjunctive Queries)
    # Useful for "Red OR Blue", "Square OR touching border"
    for i in range(len(base_queries)):
        for j in range(i + 1, len(base_queries)):
            q1 = base_queries[i]
            q2 = base_queries[j]
            
            # Avoid redundant unions (e.g. subset subsumption handled by cost, but explicit check helps)
            if isinstance(q1, QueryByDimValue) and isinstance(q2, QueryByDimValue) and q1.dim == q2.dim:
                # Same dimension: Blue OR Red -> Valid
                pass
            
            candidates.append(QueryUnion(q1, q2))
    
    # Sort by MDL cost
    candidates.sort(key=lambda q: q.cost())
    
    return candidates
