# -*- coding: utf-8 -*-
"""operators/core.py - Core operators (topology, force, fill, sequence)."""


import numpy as np
from typing import Optional, List
from dataclasses import dataclass
from functools import reduce

from ..base import Operator, SequentialOperator
from ..state import State
from ..transform import AffineTransform, derive_isometry_unordered
from ..topology import partition_by_connectivity, get_component_labels, get_interior

from scipy.ndimage import minimum as nd_min, maximum as nd_max, binary_dilation
from scipy.spatial import ConvexHull

# Import from split modules for CropOperator factory
from .selection import FilterOperator as ProjectionOperator
from .logic import IntersectionOperator


def view_as_void(arr):
    """Helper to view array as void for set operations."""
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))


def CropOperator(bbox_min: np.ndarray, bbox_max: np.ndarray, dims: List[int]) -> IntersectionOperator:
    """
    Factory: Crop is just AND (Intersection) of spatial filters.
    Returns an IntersectionOperator of ProjectionOperators.
    """
    filters = [
        f 
        for d_idx, d in enumerate(dims) 
        for f in [ProjectionOperator(dim=d, comparison='>=', threshold=bbox_min[d_idx]),
                  ProjectionOperator(dim=d, comparison='<=', threshold=bbox_max[d_idx])]
    ]
    return IntersectionOperator(operands=filters)


@dataclass
class TopologicalFilterOperator(Operator):
    """
    Topological filter: keep objects matching an invariant predicate.
    Filters at OBJECT level (connected components), not pixel level.
    """
    invariant: str  # 'mass', 'spread', 'centroid', 'value', 'has_neighbor'
    comparison: str  # '==', '<', '>', '<=', '>='
    threshold: float
    dim: int = None
    direction: int = 0
    
    def apply(self, state: State) -> State:
        if state.is_empty: return state.copy()
        
        components = partition_by_connectivity(state)
        grid = state.to_grid()
        
        kept = [c for c in components if self._compare(self._get_invariant(c, state, grid), self.threshold)]
        
        return State(points=np.vstack([c.points for c in kept]) if kept else np.empty((0, state.n_dims)))
    
    def _get_invariant(self, obj: State, full_state: State, grid: np.ndarray = None) -> float:
        dim = self.dim
        features = {
            'mass': lambda: obj.n_points,
            'spread': lambda: obj.spread[dim] if dim is not None else np.sum(obj.spread),
            'centroid': lambda: obj.centroid[dim] if dim is not None else np.linalg.norm(obj.centroid),
            'value': lambda: np.mean(obj.points[:, dim if dim is not None else -1]),
            'is_whole': lambda: 1.0,
            'has_neighbor': lambda: self._detect_neighbor(obj, grid)
        }
        return features.get(self.invariant, lambda: 0.0)()

    def _detect_neighbor(self, obj: State, grid: np.ndarray) -> float:
        if grid is None or obj.is_empty: return 0.0
        mask = obj.to_grid() != 0
        dilated = binary_dilation(mask)
        has_hit = np.any((dilated & ~mask) & (grid != 0))
        return 1.0 if has_hit else 0.0

    def _compare(self, value: float, threshold: float) -> bool:
        eq = np.isclose(value, threshold)
        gt = value > threshold
        lt = value < threshold
        comparators = {
            '==': eq, '>': gt, '<': lt,
            '>=': gt | eq, '<=': lt | eq, '!=': ~eq,
        }
        return comparators.get(self.comparison, False)


@dataclass
class ExtremeFilterOperator(Operator):
    """Selects object(s) with an extreme property value (Top-1)."""
    invariant: str
    extremum: str  # 'min', 'max'
    dim: int = None
    
    def apply(self, state: State) -> State:
        if state.is_empty: return state.copy()
        components = partition_by_connectivity(state)
        if not components: return state.copy()
        
        values = np.array(list(map(self._get_invariant, components)))
        if len(values) == 0: return state.copy()
        
        target_val = np.max(values) if self.extremum == 'max' else np.min(values)
        matches = np.isclose(values, target_val)
        kept_pts = [components[i].points for i in np.where(matches)[0]]
        
        return State(np.vstack(kept_pts) if kept_pts else np.empty((0, state.n_dims)))

    def _get_invariant(self, obj: State) -> float:
        dim = self.dim
        features = {
            'mass': lambda: obj.n_points,
            'spread': lambda: obj.spread[dim] if dim is not None else np.sum(obj.spread),
            'centroid': lambda: obj.centroid[dim] if dim is not None else np.linalg.norm(obj.centroid),
            'value': lambda: np.mean(obj.points[:, dim if dim is not None else -1]),
        }
        return features.get(self.invariant, lambda: 0.0)()


@dataclass
class IsomorphismFilterOperator(Operator):
    """Filters connected components isomorphic to a template."""
    template: State
    
    def apply(self, state: State) -> State:
        if state.is_empty: return state.copy()
        
        objects = partition_by_connectivity(state)
        kept_objects = []
        
        for obj in objects:
            if obj.n_points != self.template.n_points:
                continue
            op = derive_isometry_unordered(obj, self.template)
            if op is not None:
                kept_objects.append(obj)
                
        if not kept_objects:
            return State(np.zeros((0, state.n_dims)))
        return State(np.vstack([o.points for o in kept_objects]))


@dataclass
class ForceOperator(Operator):
    """Force application operator (gravity, attraction)."""
    source_filter: Operator
    target_filter: Operator = None
    force_type: str = 'gravity'
    axis: int = None
    direction: int = 0
    use_local_bounds: bool = False
    max_distance: float = None
    group_by: str = None
    
    def apply(self, state: State) -> State:
        if state.is_empty: return state.copy()
        
        n_pts = state.n_points
        component_labels = get_component_labels(state) if self.group_by == 'connectivity' else np.zeros(n_pts, dtype=int)
        use_components = (self.group_by == 'connectivity')
        
        src_state = self.source_filter.apply(state)
        S_void = view_as_void(np.ascontiguousarray(state.points))
        M_void = view_as_void(np.ascontiguousarray(src_state.points))
        mask_mob = np.isin(S_void, M_void).flatten()
        mask_mob = np.where(use_components, np.ones(n_pts, dtype=bool), mask_mob)
        
        target_axis = self.axis if self.axis is not None else 0
        coords = state.points[:, target_axis]
        
        if self.use_local_bounds:
            limit_mins = nd_min(coords, component_labels, index=component_labels)
            limit_maxs = nd_max(coords, component_labels, index=component_labels)
        else:
            limit_mins = np.zeros_like(coords)
            limit_maxs = np.full_like(coords, state.grid_shape[target_axis] if hasattr(state, 'grid_shape') else 30)
        
        all_points = state.points.copy()
        force_val = (self.max_distance if self.max_distance else 100.0) * self.direction
        targets = coords + mask_mob.astype(float) * force_val
        targets = np.clip(targets, limit_mins, limit_maxs)
        all_points[:, target_axis] = targets
        
        return State(all_points)


@dataclass
class LinearForceOperator(Operator):
    """Apply a Linear Force Vector (Gravity/Wind) to objects."""
    vector: np.ndarray
    group_by: str = 'connectivity'
    
    def apply(self, state: State) -> State:
        if state.is_empty: return state.copy()
        
        v = np.array(self.vector, dtype=float)
        norm = np.linalg.norm(v)
        if norm < 1e-6: return state.copy()
        
        objects = partition_by_connectivity(state) if self.group_by == 'connectivity' else [state]
        if not objects: return state.copy()
        
        direction = v / norm
        depths = [np.dot(obj.centroid, direction) for obj in objects]
        indices = np.argsort(depths)[::-1]
        
        occupied = set(map(tuple, np.round(state.points).astype(int)))
        final_points = []
        steps = max(1, int(np.ceil(norm)))
        step_vec = v / steps
        
        # Vectorized implementation of "Physics" (Collision Detection)
        
        # 1. Pre-compute occupancy map (Set of tuples for O(1) lookup)
        # Note: We use integer grid coordinates for collision
        occupied = set(map(tuple, np.round(state.points).astype(int)))
        
        final_points = []
        
        # Max force magnitude
        max_steps = max(1, int(np.ceil(norm)))
        step_vec = v / max_steps  # Unit step vector (or small step)
        
        # 2. Process objects in causal order (Front-to-Back)
        for idx in indices:
            obj = objects[idx]
            current_int = np.round(obj.points).astype(int)
            
            # Temporarily remove self from occupancy (we are moving)
            obj_tuples = set(map(tuple, current_int))
            occupied -= obj_tuples
            
            # 3. Ray Casting (Vectorized)
            # Generate all potential positions for ALL points in the object at once
            # Shape: (Steps, N_points, D)
            # steps + 1 to include 0
            
            multipliers = np.arange(1, max_steps + 1).reshape(-1, 1, 1)
            trajectories = obj.points[np.newaxis, :, :] + multipliers * step_vec[np.newaxis, np.newaxis, :]
            traj_int = np.round(trajectories).astype(int)
            
            valid_k = 0
            for k in range(max_steps):
                pts_at_k = traj_int[k]
                if np.any(pts_at_k < 0):
                    break
                hits = [tuple(p) in occupied for p in pts_at_k]
                if any(hits):
                    break
                valid_k = k + 1
            
            # Apply valid displacement
            final_obj_pts = obj.points + (valid_k * step_vec)
            final_points.append(final_obj_pts)
            
            # Re-occupy
            occupied.update(set(map(tuple, np.round(final_obj_pts).astype(int))))
            
        return State(np.vstack(final_points))



@dataclass
class HullFillOperator(Operator):
    """Fill operator: Fills the space between points (Hull Projection)."""
    mode: str = 'ortho_hull'
    axis: int = None
    fill_dims: List[int] = None
    projection_value: dict = None
    
    def apply(self, state: State) -> State:
        if state.n_points < 2: return state.copy()
        
        fill_dims = self.fill_dims if self.fill_dims is not None else list(range(state.n_dims))
        return self._apply_axis_fill(state, fill_dims)
        
    def _apply_axis_fill(self, state: State, axes: List[int]) -> State:
        """
        Algebraic Hull Fill (Orthogonal).
        Replaces explicit rasterization loops with accumulative logic.
        """
        # 1. Quantize and normalize
        coords = np.round(state.points).astype(int)
        if len(coords) == 0:
            return state.copy()
            
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        grid_shape = bbox_max - bbox_min + 1
        
        # 2. Map to local grid
        local_coords = coords - bbox_min
        
        # Create boolean grid
        grid = np.zeros(grid_shape, dtype=bool)
        # Use simple indexing for D-dim grid setting is tricky in raw numpy for variable D
        # But we can flattened linear index
        # ravel_multi_index requires tuple of arrays
        
        # Safe N-dim grid setting without explicit tuple unpacking (if D is variable)
        # Actually State enforces known D usually?
        # Use raveled indexing
        raveled_indices = np.ravel_multi_index(tuple(local_coords.T), grid_shape)
        grid.ravel()[raveled_indices] = True
        
        filled_grid = grid
        
        # 3. Apply Fill along axes
        for axis in axes:
             # Algebraic "Between Min and Max" = Forward | AND Backward |
             # Accumulate Max (OR)
             
             # Swap target axis to last for efficient processing
             grid_swapped = np.swapaxes(filled_grid, axis, -1)
             
             # Forward accumulation of "Seen a pixel?"
             forward = np.maximum.accumulate(grid_swapped, axis=-1)
             
             # Backward accumulation
             backward = np.maximum.accumulate(grid_swapped[..., ::-1], axis=-1)[..., ::-1]
             
             # Intersection is the Hull
             filled_1d = (forward & backward)
             
             filled_grid = np.swapaxes(filled_1d, axis, -1)
             
        # 4. Map back to points
        # Get coordinates of True values
        filled_indices = np.where(filled_grid)
        # Stack to (N, D)
        local_points = np.column_stack(filled_indices)
        
        # Recover absolute position
        new_coords = local_points + bbox_min
        
        # Preserve original colors? HullFill implies geometry only usually.
        # But if D includes color? Usually distinct.
        # Original implementation used `ref_pt.copy()` to preserve non-fill dims?
        # If strict geometry fill, colors are 0?
        # Implementation assumption: Fill is spatial.
        # AGENT.md: N-dim agnostic.
        
        return State(new_points.astype(float))


@dataclass
class ConvexifyOperator(Operator):
    """Fill concave regions of a shape."""
    fill_value: float = 1.0
    spatial_dims: Optional[List[int]] = None
    
    def apply(self, state: State) -> State:
        if state.n_points < 3: return state.copy()
        
        dims = self.spatial_dims if self.spatial_dims is not None else list(range(state.n_dims))
        points_spatial = state.points[:, dims]
        
        try:
            hull = ConvexHull(points_spatial)
        except:
            return state.copy()
        
        existing = set(map(tuple, np.round(points_spatial).astype(int)))
        bbox_min = np.floor(points_spatial.min(axis=0)).astype(int)
        bbox_max = np.ceil(points_spatial.max(axis=0)).astype(int)
        
        ranges = [np.arange(bbox_min[d], bbox_max[d] + 1) for d in range(len(dims))]
        grid_coords = np.stack(np.meshgrid(*ranges, indexing='ij'), axis=-1).reshape(-1, len(dims))
        
        A, b = hull.equations[:, :-1], hull.equations[:, -1]
        inside = np.all(np.dot(grid_coords, A.T) + b <= 1e-9, axis=1)
        new_coords = np.array([c for c in grid_coords[inside] if tuple(c) not in existing])
        
        if len(new_coords) == 0: return state.copy()
        
        new_points = np.zeros((len(new_coords), state.n_dims))
        new_points[:, dims] = new_coords
        mask_compl = np.ones(state.n_dims, dtype=bool)
        mask_compl[dims] = False
        new_points[:, mask_compl] = self.fill_value
        
        return State(np.vstack([state.points, new_points]))


@dataclass  
class FillOperator(Operator):
    """T(S) = S ∪ π_d^{-1}(v) ∘ Interior(S). Fill topological interior."""
    dim: int
    value: float
    
    def apply(self, state: State) -> State:
        if state.n_points < 3: return state.copy()
        
        interior = get_interior(state)
        if interior.n_points == 0: return state.copy()
        
        projection = AffineTransform.projection(self.dim, self.value, state.n_dims)
        interior_projected = projection.apply(interior)
        
        all_points = np.vstack([state.points, interior_projected.points])
        return State(np.unique(all_points, axis=0))


@dataclass
class SequenceOperator(Operator):
    """
    Generic Monoid Sequence: State_out = Union(S, G(S), G²(S), ..., G^{n-1}(S))
    """
    generator: Operator
    count: int
    
    def apply(self, state: State) -> State:
        if state.is_empty or self.count <= 0: return state.copy()
        
        if isinstance(self.generator, AffineTransform):
            return self._apply_affine_algebraic(state)
        return self._apply_generic(state)
    
    def _apply_affine_algebraic(self, state: State) -> State:
        A = self.generator.linear
        b = self.generator.translation
        P = state.points
        n, D = self.count, state.n_dims
        
        At = A.T
        powers = np.stack([np.linalg.matrix_power(At, k) for k in range(n)], axis=0)
        powers_cumsum = np.cumsum(powers, axis=0)
        powers_for_b = np.concatenate([np.zeros((1, D, D)), powers_cumsum[:-1]], axis=0)
        b_k = np.einsum('d, kdi -> ki', b, powers_for_b)
        P_transformed = np.einsum('nd, kde -> kne', P, powers)
        result = P_transformed + b_k[:, np.newaxis, :]
        
        return State(result.reshape(-1, D))
    
    def _apply_generic(self, state: State) -> State:
        def fold_step(acc, _):
            states, current = acc
            next_s = self.generator.apply(current)
            return (states + [next_s], next_s)
        
        all_states, _ = reduce(fold_step, range(self.count - 1), ([state], state))
        all_pts = np.vstack([s.points for s in all_states if not s.is_empty])
        return State(np.unique(all_pts, axis=0))


@dataclass
class EnvelopeOperator(Operator):
    """N-Dimensional Envelope Operator (Hull)."""
    rank: int = 1
    mode: str = 'bbox'
    
    def apply(self, state: State) -> State:
        if state.is_empty: return state.copy()
        
        # Simplified: delegate to HullFillOperator for now
        fill_op = HullFillOperator(mode=self.mode, fill_dims=list(range(min(self.rank, state.n_dims))))
        return fill_op.apply(state)





# Export all
__all__ = [
    'view_as_void', 'CropOperator',
    'TopologicalFilterOperator', 'ExtremeFilterOperator', 'IsomorphismFilterOperator',
    'ForceOperator', 'LinearForceOperator',
    'HullFillOperator', 'ConvexifyOperator', 'FillOperator',
    'SequenceOperator', 'EnvelopeOperator'
]
