# -*- coding: utf-8 -*-
"""
graph/semantic.py - Semantic Graph Representation of Transformations

Dual representation: Tensor ≅ Graph
- Tensor: Dense, computational
- Graph: Sparse, interpretable, causal

For ARC generalization, we need rules expressed as:
- "copy pattern to right by its width" (relational)
- NOT "translate by 14" (absolute)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto


class ConceptType(Enum):
    """Types of semantic concepts in transformation graph."""
    SPATIAL = auto()      # x, y, z coordinates
    CHROMATIC = auto()    # color/value dimension
    TOPOLOGICAL = auto()  # connectivity, boundary
    CARDINALITY = auto()  # count, mass
    RELATIONAL = auto()   # relative position, order
    # ARC-specific contextual types
    PATTERN = auto()      # line, rectangle, L-shape, etc.
    CONTEXT = auto()      # boundary, interior, corner
    OBJECT = auto()       # bounded entity in space


class RelationType(Enum):
    """Types of causal/semantic relations."""
    IDENTITY = auto()      # Same value
    TRANSLATE = auto()     # Shift by amount
    SCALE = auto()         # Multiply by factor
    PERMUTE = auto()       # Reorder/swap
    COPY = auto()          # Duplicate
    DEPENDS_ON = auto()    # Causal dependency
    RELATIVE_TO = auto()   # Referenced to another concept
    # ARC-specific contextual relations
    BOUNDED_BY = auto()    # Object is contained by grid/region
    LEFT_OF = auto()       # Spatial relation
    RIGHT_OF = auto()
    ABOVE = auto()
    BELOW = auto()
    INSIDE = auto()        # Containment
    ADJACENT_TO = auto()   # Touching
    SAME_COLOR = auto()    # Chromatic equivalence
    PATTERN_MATCH = auto() # Same pattern type


@dataclass
class Concept:
    """A semantic concept node in the graph."""
    name: str
    ctype: ConceptType
    value: Optional[Any] = None  # Concrete value if known
    reference: Optional[str] = None  # Reference to input property (e.g., "input.width")
    
    def is_relative(self) -> bool:
        """True if value is defined relative to input."""
        return self.reference is not None
    
    def to_symbolic(self) -> str:
        if self.reference:
            return f"{self.name}→{self.reference}"
        elif self.value is not None:
            return f"{self.name}={self.value}"
        return self.name


@dataclass
class Relation:
    """A causal/semantic relation edge in the graph."""
    source: str  # Concept name
    target: str  # Concept name
    rtype: RelationType
    params: Dict[str, Any] = field(default_factory=dict)
    
    def is_parametric(self) -> bool:
        """True if params reference input properties."""
        return any('input.' in str(v) for v in self.params.values())
    
    def to_symbolic(self) -> str:
        params_str = ",".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.source} --{self.rtype.name}({params_str})--> {self.target}"


@dataclass
class SemanticGraph:
    """
    Graph representation of a transformation.
    
    Nodes: Concepts (spatial, chromatic, topological)
    Edges: Relations (translate, scale, permute, copy)
    
    Key insight: Parameters should be REFERENCES (input.width) not VALUES (14).
    """
    concepts: Dict[str, Concept] = field(default_factory=dict)
    relations: List[Relation] = field(default_factory=list)
    
    def add_concept(self, name: str, ctype: ConceptType, value: Any = None, reference: str = None):
        """Add a concept node."""
        self.concepts[name] = Concept(name, ctype, value, reference)
    
    def add_relation(self, source: str, target: str, rtype: RelationType, **params):
        """Add a relation edge."""
        self.relations.append(Relation(source, target, rtype, params))
    
    def is_generalizable(self) -> bool:
        """
        True if all parameters are relative (reference input properties).
        This is the key test for ARC generalization.
        """
        # Check concepts
        for c in self.concepts.values():
            if c.value is not None and c.reference is None:
                # Has absolute value without reference
                return False
        
        # Check relations
        for r in self.relations:
            for k, v in r.params.items():
                if isinstance(v, (int, float)) and 'input.' not in str(v):
                    # Has absolute numeric parameter
                    return False
        
        return True
    
    def to_symbolic(self) -> str:
        """Human-readable representation."""
        lines = ["SemanticGraph:"]
        lines.append("  Concepts:")
        for c in self.concepts.values():
            lines.append(f"    {c.to_symbolic()}")
        lines.append("  Relations:")
        for r in self.relations:
            lines.append(f"    {r.to_symbolic()}")
        lines.append(f"  Generalizable: {self.is_generalizable()}")
        return "\n".join(lines)


def extract_semantic_graph(s_in: 'State', s_out: 'State', operator: 'Operator') -> SemanticGraph:
    """
    Extract semantic graph from derived operator.
    
    Goal: Express the transformation in terms of input properties,
    not absolute values.
    """
    graph = SemanticGraph()
    
    # Add input concepts
    graph.add_concept("input.x", ConceptType.SPATIAL, reference="input.centroid[0]")
    graph.add_concept("input.y", ConceptType.SPATIAL, reference="input.centroid[1]")
    graph.add_concept("input.width", ConceptType.SPATIAL, reference="input.spread[0]")
    graph.add_concept("input.height", ConceptType.SPATIAL, reference="input.spread[1]")
    graph.add_concept("input.colors", ConceptType.CHROMATIC, reference="input.unique_colors")
    
    # Add output concepts
    graph.add_concept("output.x", ConceptType.SPATIAL)
    graph.add_concept("output.y", ConceptType.SPATIAL)
    graph.add_concept("output.colors", ConceptType.CHROMATIC)
    
    # Try to infer relations from operator
    if hasattr(operator, 'translation') and operator.translation is not None:
        t = operator.translation
        # Instead of absolute value, try to express relative to input
        # This is the key insight - we need to find what input property matches
        dx, dy = t[0] if len(t) > 0 else 0, t[1] if len(t) > 1 else 0
        
        # Check if translation matches input properties
        if abs(dx - s_in.spread[0]) < 1e-6:
            graph.add_relation("input.x", "output.x", RelationType.TRANSLATE, 
                             by="input.width")
        elif abs(dx - s_in.spread[1]) < 1e-6:
            graph.add_relation("input.x", "output.x", RelationType.TRANSLATE,
                             by="input.height")
        else:
            # Absolute value - NOT generalizable
            graph.add_relation("input.x", "output.x", RelationType.TRANSLATE,
                             by=dx)
    
    if hasattr(operator, 'bijection') and operator.bijection is not None:
        if hasattr(operator.bijection, 'translation') and operator.bijection.translation is not None:
            t = operator.bijection.translation
            t_norm = np.linalg.norm(t)
            
            # Try to match to input properties
            if abs(t_norm - s_in.spread[0]) < 1e-6:
                graph.add_relation("input", "output", RelationType.TRANSLATE,
                                 by="input.width")
            elif abs(t_norm - s_in.spread[1]) < 1e-6:
                graph.add_relation("input", "output", RelationType.TRANSLATE,
                                 by="input.height")
            else:
                graph.add_relation("input", "output", RelationType.TRANSLATE,
                                 by=f"t={t_norm:.2f}")
    
    return graph


class InputPropertyMatcher:
    """
    Pattern matcher: finds if absolute values match input properties.
    
    This is the 'semantic attention' - like LLM attention over tokens,
    but over semantic concepts from the input.
    """
    
    def __init__(self, state: 'State'):
        """Build vocabulary of input properties."""
        self.vocab = {}
        
        # Spatial properties
        if state.n_dims >= 1:
            self.vocab['input.width'] = state.spread[0]
            self.vocab['input.bbox_max_x'] = state.bbox_max[0]
            self.vocab['input.bbox_min_x'] = state.bbox_min[0]
            self.vocab['input.centroid_x'] = state.centroid[0]
        
        if state.n_dims >= 2:
            self.vocab['input.height'] = state.spread[1]
            self.vocab['input.bbox_max_y'] = state.bbox_max[1]
            self.vocab['input.bbox_min_y'] = state.bbox_min[1]
            self.vocab['input.centroid_y'] = state.centroid[1]
        
        if state.n_dims >= 3:
            self.vocab['input.depth'] = state.spread[2]
        
        # Cardinality properties
        self.vocab['input.n_points'] = state.n_points
        self.vocab['input.n_dims'] = state.n_dims
        
        # Derived properties
        if state.n_dims >= 2:
            self.vocab['input.width+1'] = state.spread[0] + 1
            self.vocab['input.height+1'] = state.spread[1] + 1
            self.vocab['input.2*width'] = 2 * state.spread[0]
            self.vocab['input.2*height'] = 2 * state.spread[1]
    
    def match(self, value: float, tolerance: float = 1e-6) -> Optional[str]:
        """
        Find input property that matches the value.
        Returns property name (e.g., 'input.width') or None.
        """
        for name, prop_val in self.vocab.items():
            if abs(value - prop_val) < tolerance:
                return name
        return None
    
    def match_vector(self, vec: np.ndarray, tolerance: float = 1e-6) -> List[Optional[str]]:
        """Match each component of a vector."""
        return [self.match(v, tolerance) for v in vec]
    
    def match_best(self, value: float) -> Tuple[str, float]:
        """Find closest match and return (name, error)."""
        best_name = None
        best_error = float('inf')
        
        for name, prop_val in self.vocab.items():
            error = abs(value - prop_val)
            if error < best_error:
                best_error = error
                best_name = name
        
        return best_name, best_error


def relativize_operator(operator: 'Operator', s_in: 'State') -> SemanticGraph:
    """
    Convert an operator with absolute values to a semantic graph with relative references.
    
    This is the key function for ARC generalization:
    - Takes: T(14.0)  [absolute, overfits]
    - Returns: T(input.width) [relative, generalizes]
    """
    graph = SemanticGraph()
    matcher = InputPropertyMatcher(s_in)
    
    # Add input concept references
    graph.add_concept("input.spatial", ConceptType.SPATIAL, reference="input.points[:,:2]")
    graph.add_concept("input.values", ConceptType.CHROMATIC, reference="input.points[:,2]")
    graph.add_concept("output.spatial", ConceptType.SPATIAL)
    graph.add_concept("output.values", ConceptType.CHROMATIC)
    
    # Handle AffineTransform
    if hasattr(operator, 'translation') and operator.translation is not None:
        t = operator.translation
        refs = matcher.match_vector(t[:2] if len(t) >= 2 else t)
        
        if refs[0] is not None:
            graph.add_relation("input.spatial", "output.spatial", RelationType.TRANSLATE,
                             dx=refs[0])
        else:
            graph.add_relation("input.spatial", "output.spatial", RelationType.TRANSLATE,
                             dx=float(t[0]))
        
        if len(refs) > 1 and refs[1] is not None:
            graph.relations[-1].params['dy'] = refs[1]
        elif len(t) > 1:
            graph.relations[-1].params['dy'] = float(t[1])
    
    # Handle LiftedTransform
    if hasattr(operator, 'bijection') and operator.bijection is not None:
        bij = operator.bijection
        if hasattr(bij, 'translation') and bij.translation is not None:
            t = bij.translation
            refs = matcher.match_vector(t[:2] if len(t) >= 2 else t)
            
            # Find if the translation matches input properties
            matched_refs = [r for r in refs if r is not None]
            if matched_refs:
                graph.add_relation("input", "output", RelationType.TRANSLATE,
                                 by=matched_refs[0])
            else:
                # Try to match norm
                t_norm = np.linalg.norm(t[:2] if len(t) >= 2 else t)
                norm_ref = matcher.match(t_norm)
                if norm_ref:
                    graph.add_relation("input", "output", RelationType.TRANSLATE,
                                     by=norm_ref)
                else:
                    best_name, error = matcher.match_best(t_norm)
                    graph.add_relation("input", "output", RelationType.TRANSLATE,
                                     by=t_norm, closest=best_name, error=error)
    
    # Handle Lift type
    if hasattr(operator, 'lift') and operator.lift is not None:
        lift = operator.lift
        if hasattr(lift, 'lifter'):
            graph.add_concept("lift.kernel", ConceptType.TOPOLOGICAL, 
                            value=lift.lifter)
            graph.add_relation("input", "lifted", RelationType.DEPENDS_ON,
                             kernel=lift.lifter)
    
    return graph


class ObjectContext:
    """
    Contextual analysis of an object relative to its space.
    
    For ARC: an object is conditioned by its spatial context:
    - Is it on the boundary? Interior? Corner?
    - What pattern does it form? Line, rectangle, L-shape?
    - Where is it relative to other objects?
    """
    
    def __init__(self, state: 'State', grid_shape: Tuple[int, int] = None):
        """Build contextual vocabulary for an object in its space."""
        self.state = state
        self.grid_shape = grid_shape or (int(state.bbox_max[0])+1, int(state.bbox_max[1])+1)
        
        # Compute contextual properties
        self.props = {}
        self._compute_boundary_context()
        self._compute_pattern_type()
        self._compute_symmetry()
    
    def _compute_boundary_context(self):
        """Is the object touching grid boundaries?"""
        if self.state.is_empty:
            return
            
        bbox_min = self.state.bbox_min
        bbox_max = self.state.bbox_max
        h, w = self.grid_shape
        
        # Check boundary contact
        self.props['touches_left'] = bbox_min[1] == 0 if len(bbox_min) > 1 else False
        self.props['touches_right'] = bbox_max[1] >= w - 1 if len(bbox_max) > 1 else False
        self.props['touches_top'] = bbox_min[0] == 0
        self.props['touches_bottom'] = bbox_max[0] >= h - 1
        
        # Derive context type
        n_touches = sum([
            self.props['touches_left'], self.props['touches_right'],
            self.props['touches_top'], self.props['touches_bottom']
        ])
        
        self.props['is_corner'] = n_touches >= 2
        self.props['is_edge'] = n_touches == 1
        self.props['is_interior'] = n_touches == 0
        self.props['is_spanning'] = n_touches >= 3  # spans most of grid
    
    def _compute_pattern_type(self):
        """What geometric pattern does the object form?"""
        if self.state.is_empty:
            return
        
        pts = self.state.points[:, :2]  # spatial dims only
        n = len(pts)
        
        # Bounding box properties
        spread = self.state.spread[:2]
        area = spread[0] * spread[1] if len(spread) >= 2 else spread[0]
        
        # Pattern detection via algebraic properties
        self.props['is_point'] = n == 1
        self.props['is_line_h'] = spread[0] == 0 and spread[1] > 0 if len(spread) >= 2 else False
        self.props['is_line_v'] = spread[1] == 0 and spread[0] > 0 if len(spread) >= 2 else False
        self.props['is_line'] = self.props.get('is_line_h', False) or self.props.get('is_line_v', False)
        
        # Rectangle test: n_points == area coverage
        if len(spread) >= 2:
            expected_rect_pts = (spread[0] + 1) * (spread[1] + 1)
            self.props['is_filled_rect'] = n == expected_rect_pts
            self.props['is_hollow_rect'] = n == 2 * (spread[0] + spread[1]) and n > 4
        
        self.props['is_square'] = len(spread) >= 2 and spread[0] == spread[1] and spread[0] > 0
    
    def _compute_symmetry(self):
        """Check for symmetric patterns."""
        if self.state.is_empty or self.state.n_points < 2:
            return
        
        pts = self.state.points[:, :2]
        centroid = pts.mean(axis=0)
        
        # Check horizontal symmetry: reflect across vertical center line
        reflected_h = pts.copy()
        reflected_h[:, 1] = 2 * centroid[1] - reflected_h[:, 1]
        
        # Check if reflected points match original (within tolerance)
        from ..topology import view_as_void
        orig_void = view_as_void(np.round(pts, 2).astype(float))
        refl_void = view_as_void(np.round(reflected_h, 2).astype(float))
        
        self.props['symmetric_h'] = len(np.intersect1d(orig_void, refl_void)) == len(pts)
        
        # Vertical symmetry
        reflected_v = pts.copy()
        reflected_v[:, 0] = 2 * centroid[0] - reflected_v[:, 0]
        refl_void_v = view_as_void(np.round(reflected_v, 2).astype(float))
        self.props['symmetric_v'] = len(np.intersect1d(orig_void, refl_void_v)) == len(pts)
    
    def get_context_type(self) -> str:
        """Return primary context type."""
        if self.props.get('is_corner'):
            return 'corner'
        elif self.props.get('is_edge'):
            return 'edge'
        elif self.props.get('is_spanning'):
            return 'spanning'
        return 'interior'
    
    def get_pattern_type(self) -> str:
        """Return primary pattern type."""
        if self.props.get('is_point'):
            return 'point'
        elif self.props.get('is_line_h'):
            return 'line_h'
        elif self.props.get('is_line_v'):
            return 'line_v'
        elif self.props.get('is_filled_rect'):
            return 'filled_rect'
        elif self.props.get('is_hollow_rect'):
            return 'hollow_rect'
        elif self.props.get('is_square'):
            return 'square'
        return 'irregular'
    
    def to_concepts(self) -> List[Concept]:
        """Convert context to semantic concepts."""
        concepts = []
        
        concepts.append(Concept(
            name='object.context',
            ctype=ConceptType.CONTEXT,
            value=self.get_context_type()
        ))
        
        concepts.append(Concept(
            name='object.pattern',
            ctype=ConceptType.PATTERN,
            value=self.get_pattern_type()
        ))
        
        if self.props.get('symmetric_h'):
            concepts.append(Concept(
                name='object.symmetric_h',
                ctype=ConceptType.TOPOLOGICAL,
                value=True
            ))
        
        if self.props.get('symmetric_v'):
            concepts.append(Concept(
                name='object.symmetric_v',
                ctype=ConceptType.TOPOLOGICAL,
                value=True
            ))
        
        return concepts


# Export
__all__ = [
    'ConceptType', 'RelationType', 'Concept', 'Relation', 
    'SemanticGraph', 'extract_semantic_graph',
    'InputPropertyMatcher', 'relativize_operator',
    'ObjectContext'
]
