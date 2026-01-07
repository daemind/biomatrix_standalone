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


def compute_object_relations(objects: List['State']) -> List[Relation]:
    """
    Compute spatial relations between multiple objects.
    
    For ARC: understanding how objects relate to each other
    is key to generalization.
    """
    relations = []
    
    for i, obj_a in enumerate(objects):
        for j, obj_b in enumerate(objects):
            if i >= j:
                continue
            
            # Get bboxes
            a_min, a_max = obj_a.bbox_min[:2], obj_a.bbox_max[:2]
            b_min, b_max = obj_b.bbox_min[:2], obj_b.bbox_max[:2]
            a_cent = obj_a.centroid[:2]
            b_cent = obj_b.centroid[:2]
            
            # Relative position (based on centroids)
            if a_cent[1] < b_cent[1] - 1:  # A is left of B
                relations.append(Relation(f"obj_{i}", f"obj_{j}", RelationType.LEFT_OF))
            elif a_cent[1] > b_cent[1] + 1:  # A is right of B
                relations.append(Relation(f"obj_{i}", f"obj_{j}", RelationType.RIGHT_OF))
            
            if a_cent[0] < b_cent[0] - 1:  # A is above B
                relations.append(Relation(f"obj_{i}", f"obj_{j}", RelationType.ABOVE))
            elif a_cent[0] > b_cent[0] + 1:  # A is below B
                relations.append(Relation(f"obj_{i}", f"obj_{j}", RelationType.BELOW))
            
            # Containment: A inside B?
            if (a_min[0] >= b_min[0] and a_max[0] <= b_max[0] and
                a_min[1] >= b_min[1] and a_max[1] <= b_max[1]):
                if obj_a.n_points < obj_b.n_points:
                    relations.append(Relation(f"obj_{i}", f"obj_{j}", RelationType.INSIDE))
            
            # B inside A?
            if (b_min[0] >= a_min[0] and b_max[0] <= a_max[0] and
                b_min[1] >= a_min[1] and b_max[1] <= a_max[1]):
                if obj_b.n_points < obj_a.n_points:
                    relations.append(Relation(f"obj_{j}", f"obj_{i}", RelationType.INSIDE))
            
            # Adjacent: bboxes touch but don't overlap
            h_gap = max(0, max(a_min[1] - b_max[1], b_min[1] - a_max[1]))
            v_gap = max(0, max(a_min[0] - b_max[0], b_min[0] - a_max[0]))
            
            if (h_gap <= 1 and v_gap == 0) or (v_gap <= 1 and h_gap == 0):
                relations.append(Relation(f"obj_{i}", f"obj_{j}", RelationType.ADJACENT_TO))
            
            # Same color?
            if obj_a.n_dims >= 3 and obj_b.n_dims >= 3:
                a_colors = set(np.round(obj_a.points[:, 2], 0))
                b_colors = set(np.round(obj_b.points[:, 2], 0))
                if a_colors == b_colors:
                    relations.append(Relation(f"obj_{i}", f"obj_{j}", RelationType.SAME_COLOR))
    
    return relations


def build_scene_graph(state: 'State', grid_shape: Tuple[int, int] = None) -> SemanticGraph:
    """
    Build complete semantic graph for a scene (state with multiple objects).
    
    This is the main entry point for ARC-style reasoning:
    1. Partition into objects
    2. Analyze each object's context
    3. Compute inter-object relations
    """
    from ..topology import partition_by_connectivity
    
    graph = SemanticGraph()
    
    # Default grid shape from state bounds
    if grid_shape is None:
        grid_shape = (int(state.bbox_max[0]) + 1, int(state.bbox_max[1]) + 1)
    
    # Partition into objects
    objects = partition_by_connectivity(state)
    
    # Add object concepts with context
    for i, obj in enumerate(objects):
        ctx = ObjectContext(obj, grid_shape)
        
        # Add object as concept
        graph.add_concept(
            f"obj_{i}",
            ConceptType.OBJECT,
            value={
                'context': ctx.get_context_type(),
                'pattern': ctx.get_pattern_type(),
                'n_points': obj.n_points,
            }
        )
        
        # Add derived concepts
        for concept in ctx.to_concepts():
            concept.name = f"obj_{i}.{concept.name.split('.')[-1]}"
            graph.concepts[concept.name] = concept
    
    # Compute and add inter-object relations
    relations = compute_object_relations(objects)
    graph.relations.extend(relations)
    
    # Add grid context
    graph.add_concept(
        "grid",
        ConceptType.CONTEXT,
        value={'shape': grid_shape}
    )
    
    return graph


def explain_scene(graph: SemanticGraph) -> str:
    """
    Generate natural language description of a scene.
    """
    lines = []
    
    # Count objects
    objects = [c for c in graph.concepts if c.startswith("obj_") and "." not in c]
    lines.append(f"The scene contains {len(objects)} object(s).")
    
    # Describe each object
    for obj_name in objects:
        obj = graph.concepts.get(obj_name)
        if obj and obj.value:
            ctx = obj.value.get('context', 'unknown')
            pattern = obj.value.get('pattern', 'unknown')
            n_pts = obj.value.get('n_points', 0)
            
            desc = f"  • {obj_name}: A {pattern}"
            if ctx == 'corner':
                desc += " in the corner"
            elif ctx == 'edge':
                desc += " on the edge"
            elif ctx == 'interior':
                desc += " in the interior"
            desc += f" ({n_pts} points)"
            lines.append(desc)
    
    # Describe relations
    if graph.relations:
        lines.append("\nSpatial relations:")
        for r in graph.relations[:10]:
            rel_name = r.rtype.name.lower().replace('_', ' ')
            lines.append(f"  • {r.source} is {rel_name} {r.target}")
    
    return "\n".join(lines)


def explain_transformation(graph_in: SemanticGraph, graph_out: SemanticGraph, 
                          operator: 'Operator' = None) -> str:
    """
    Generate natural language explanation of a transformation.
    
    This is the key for ARC explainability:
    - What changed?
    - What stayed the same?
    - What's the rule?
    """
    lines = ["=== Transformation Explanation ===\n"]
    
    # Object counts
    objs_in = [c for c in graph_in.concepts if c.startswith("obj_") and "." not in c]
    objs_out = [c for c in graph_out.concepts if c.startswith("obj_") and "." not in c]
    
    lines.append(f"INPUT: {len(objs_in)} object(s)")
    lines.append(f"OUTPUT: {len(objs_out)} object(s)")
    
    # Mass change
    if len(objs_out) > len(objs_in):
        lines.append(f"  → Objects ADDED ({len(objs_out) - len(objs_in)} new)")
    elif len(objs_out) < len(objs_in):
        lines.append(f"  → Objects REMOVED ({len(objs_in) - len(objs_out)} deleted)")
    else:
        lines.append("  → Object count PRESERVED")
    
    # Pattern changes
    lines.append("\nPattern Analysis:")
    patterns_in = [graph_in.concepts.get(o).value.get('pattern') for o in objs_in 
                   if graph_in.concepts.get(o) and graph_in.concepts.get(o).value]
    patterns_out = [graph_out.concepts.get(o).value.get('pattern') for o in objs_out 
                    if graph_out.concepts.get(o) and graph_out.concepts.get(o).value]
    
    if patterns_in == patterns_out:
        lines.append("  • Pattern types PRESERVED")
    else:
        lines.append(f"  • Input patterns: {patterns_in}")
        lines.append(f"  • Output patterns: {patterns_out}")
    
    # Context changes
    contexts_in = [graph_in.concepts.get(o).value.get('context') for o in objs_in 
                   if graph_in.concepts.get(o) and graph_in.concepts.get(o).value]
    contexts_out = [graph_out.concepts.get(o).value.get('context') for o in objs_out 
                    if graph_out.concepts.get(o) and graph_out.concepts.get(o).value]
    
    if contexts_in != contexts_out:
        lines.append(f"  • Context changed: {contexts_in} → {contexts_out}")
    
    # Operator description
    if operator:
        lines.append("\nOperator:")
        if hasattr(operator, 'to_symbolic'):
            lines.append(f"  • {operator.to_symbolic()}")
        else:
            lines.append(f"  • {type(operator).__name__}")
    
    # Infer rule type
    lines.append("\nInferred Rule:")
    if len(objs_out) == 2 * len(objs_in):
        lines.append("  → DUPLICATION/TILING: Objects are copied")
    elif len(objs_out) == len(objs_in) and patterns_in == patterns_out:
        lines.append("  → TRANSFORMATION: Objects moved/rotated but preserved")
    elif len(objs_out) < len(objs_in):
        lines.append("  → FILTERING: Some objects removed based on criteria")
    elif patterns_in != patterns_out:
        lines.append("  → RESHAPE: Object patterns modified")
    else:
        lines.append("  → COMPLEX: Multiple operations combined")
    
    return "\n".join(lines)


def explain_arc_solution(s_in: 'State', s_out: 'State', operator: 'Operator' = None,
                         grid_shape: Tuple[int, int] = None) -> str:
    """
    Full explainability pipeline for an ARC solution.
    """
    # Build scene graphs
    graph_in = build_scene_graph(s_in, grid_shape)
    graph_out = build_scene_graph(s_out, grid_shape)
    
    lines = ["=" * 50]
    lines.append("INPUT SCENE:")
    lines.append(explain_scene(graph_in))
    lines.append("")
    lines.append("OUTPUT SCENE:")
    lines.append(explain_scene(graph_out))
    lines.append("")
    lines.append(explain_transformation(graph_in, graph_out, operator))
    lines.append("=" * 50)
    
    return "\n".join(lines)


@dataclass
class SemanticRule:
    """
    A rule inferred from comparing input/output scene graphs.
    
    This is the semantic/algebraic representation of the transformation:
    - source: what to match
    - action: what to do
    - target: where to apply
    - condition: when to apply (optional)
    """
    source: str           # e.g., "object with pattern=filled_rect"
    action: str           # e.g., "color_permute", "translate", "copy"
    target: str           # e.g., "all objects", "object.color"
    condition: str = ""   # e.g., "if context=corner"
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_natural_language(self) -> str:
        """Generate human-readable rule."""
        if self.condition:
            return f"For {self.source} {self.condition}: {self.action} → {self.target}"
        return f"For {self.source}: {self.action} → {self.target}"
    
    def __repr__(self):
        return f"Rule({self.source} → {self.action} → {self.target})"


def derive_object_mapping(objs_in: List['State'], objs_out: List['State']) -> List[Tuple[int, int, dict]]:
    """
    Find correspondence between input and output objects.
    Returns list of (in_idx, out_idx, diff_dict) tuples.
    """
    mappings = []
    
    for i, obj_in in enumerate(objs_in):
        best_match = None
        best_score = -1
        best_diff = {}
        
        for j, obj_out in enumerate(objs_out):
            score = 0
            diff = {}
            
            # Compare n_points
            if obj_in.n_points == obj_out.n_points:
                score += 3
                diff['mass'] = 'preserved'
            else:
                diff['mass'] = ('changed', obj_in.n_points, obj_out.n_points)
            
            # Compare spread (shape)
            spread_in = obj_in.spread[:2] if len(obj_in.spread) >= 2 else obj_in.spread
            spread_out = obj_out.spread[:2] if len(obj_out.spread) >= 2 else obj_out.spread
            if np.allclose(spread_in, spread_out, atol=0.1):
                score += 2
                diff['shape'] = 'preserved'
            else:
                diff['shape'] = ('changed', spread_in.tolist(), spread_out.tolist())
            
            # Compare position (centroid offset)
            cent_in = obj_in.centroid[:2] if len(obj_in.centroid) >= 2 else obj_in.centroid
            cent_out = obj_out.centroid[:2] if len(obj_out.centroid) >= 2 else obj_out.centroid
            translation = cent_out - cent_in
            if np.allclose(translation, 0, atol=0.1):
                score += 1
                diff['position'] = 'same'
            else:
                diff['position'] = ('translated', translation.tolist())
            
            # Compare color distribution
            if obj_in.n_dims >= 3 and obj_out.n_dims >= 3:
                colors_in = set(np.round(obj_in.points[:, 2], 0))
                colors_out = set(np.round(obj_out.points[:, 2], 0))
                if colors_in == colors_out:
                    score += 2
                    diff['color'] = 'preserved'
                else:
                    diff['color'] = ('changed', list(colors_in), list(colors_out))
            
            if score > best_score:
                best_score = score
                best_match = j
                best_diff = diff
        
        if best_match is not None:
            mappings.append((i, best_match, best_diff))
    
    return mappings


def derive_semantic_rules(s_in: 'State', s_out: 'State', 
                          grid_shape: Tuple[int, int] = None) -> List[SemanticRule]:
    """
    Derive semantic transformation rules by comparing input/output.
    
    This is the key function for algebraic ARC solving:
    1. Build scene graphs
    2. Map objects between scenes
    3. Analyze what changed
    4. Express as semantic rules
    """
    from ..topology import partition_by_connectivity
    
    rules = []
    
    # Get objects
    objs_in = partition_by_connectivity(s_in)
    objs_out = partition_by_connectivity(s_out)
    
    # Get mappings
    mappings = derive_object_mapping(objs_in, objs_out)
    
    # Analyze global changes
    if len(objs_out) == 2 * len(objs_in):
        rules.append(SemanticRule(
            source="all objects",
            action="DUPLICATE",
            target="scene",
            params={'factor': 2}
        ))
    elif len(objs_out) < len(objs_in):
        rules.append(SemanticRule(
            source="some objects",
            action="FILTER/DELETE",
            target="scene",
            params={'removed': len(objs_in) - len(objs_out)}
        ))
    
    # Analyze per-object changes
    color_changes = []
    position_changes = []
    
    for in_idx, out_idx, diff in mappings:
        obj_in_ctx = ObjectContext(objs_in[in_idx], grid_shape)
        
        # Color change
        if diff.get('color') and diff['color'] != 'preserved':
            _, colors_in, colors_out = diff['color']
            color_changes.append((colors_in, colors_out))
        
        # Position change  
        if diff.get('position') and diff['position'] != 'same':
            _, translation = diff['position']
            position_changes.append(translation)
    
    # Infer color rule
    if color_changes:
        # Check if all color changes follow same pattern
        if len(set(str(c) for c in color_changes)) == 1:
            colors_in, colors_out = color_changes[0]
            rules.append(SemanticRule(
                source="object.colors",
                action="PERMUTE",
                target="all objects uniformly",
                params={'from': colors_in, 'to': colors_out}
            ))
        else:
            rules.append(SemanticRule(
                source="object.colors",
                action="PERMUTE",
                target="per-object (context-dependent)",
                params={'changes': color_changes}
            ))
    
    # Infer position rule
    if position_changes:
        # Check if all translations are the same
        translations = [np.array(t) for t in position_changes]
        if len(translations) > 1 and all(np.allclose(t, translations[0]) for t in translations):
            rules.append(SemanticRule(
                source="object.position",
                action="TRANSLATE",
                target="all objects uniformly",
                params={'by': translations[0].tolist()}
            ))
        elif position_changes:
            rules.append(SemanticRule(
                source="object.position",
                action="TRANSLATE",
                target="per-object",
                params={'translations': position_changes}
            ))
    
    return rules


def explain_semantic_resolution(s_in: 'State', s_out: 'State',
                                grid_shape: Tuple[int, int] = None) -> str:
    """
    Full semantic resolution: derive rules and explain them.
    """
    rules = derive_semantic_rules(s_in, s_out, grid_shape)
    
    lines = ["=" * 50]
    lines.append("SEMANTIC RESOLUTION")
    lines.append("=" * 50)
    lines.append(f"\nDiscovered {len(rules)} transformation rule(s):\n")
    
    for i, rule in enumerate(rules, 1):
        lines.append(f"{i}. {rule.to_natural_language()}")
        if rule.params:
            for k, v in rule.params.items():
                lines.append(f"     {k}: {v}")
        lines.append("")
    
    # Add interpretation
    lines.append("INTERPRETATION:")
    if any(r.action == "PERMUTE" for r in rules):
        lines.append("  → This is a COLOR TRANSFORMATION task")
        color_rule = next((r for r in rules if r.action == "PERMUTE"), None)
        if color_rule and color_rule.target == "all objects uniformly":
            lines.append("  → The same color mapping applies to all objects")
        else:
            lines.append("  → Color mapping depends on object context")
    
    if any(r.action == "TRANSLATE" for r in rules):
        lines.append("  → This is a SPATIAL TRANSFORMATION task")
        
    if any(r.action == "DUPLICATE" for r in rules):
        lines.append("  → This is a TILING/REPLICATION task")
        
    if any(r.action == "FILTER/DELETE" for r in rules):
        lines.append("  → This is a SELECTION/FILTERING task")
    
    lines.append("=" * 50)
    return "\n".join(lines)


def apply_semantic_rules(rules: List[SemanticRule], s_test: 'State',
                         training_pairs: List[Tuple['State', 'State']] = None) -> 'State':
    """
    Apply derived semantic rules to a test input.
    
    This completes the ARC pipeline:
    1. derive_semantic_rules(train_in, train_out) → rules
    2. apply_semantic_rules(rules, test_in) → predicted_out
    """
    from ..topology import partition_by_connectivity
    from ..state import State
    
    result_points = s_test.points.copy()
    
    for rule in rules:
        if rule.action == "PERMUTE" and rule.source == "object.colors":
            # Apply color permutation
            if 'changes' in rule.params:
                # Build color map from training examples
                color_map = {}
                for colors_in, colors_out in rule.params['changes']:
                    for c_in, c_out in zip(colors_in, colors_out):
                        color_map[float(c_in)] = float(c_out)
                
                # Apply to test
                if s_test.n_dims >= 3:
                    for i in range(len(result_points)):
                        old_color = float(np.round(result_points[i, 2], 0))
                        if old_color in color_map:
                            result_points[i, 2] = color_map[old_color]
                        else:
                            # Color not in training - need to infer pattern
                            # For now, keep original
                            pass
            
            elif 'from' in rule.params and 'to' in rule.params:
                # Uniform color mapping
                color_map = dict(zip(rule.params['from'], rule.params['to']))
                if s_test.n_dims >= 3:
                    for i in range(len(result_points)):
                        old_color = float(np.round(result_points[i, 2], 0))
                        if old_color in color_map:
                            result_points[i, 2] = color_map[old_color]
        
        elif rule.action == "TRANSLATE":
            # Apply translation
            if 'by' in rule.params:
                translation = np.array(rule.params['by'])
                if len(translation) <= s_test.n_dims:
                    result_points[:, :len(translation)] += translation
            
            elif 'translations' in rule.params:
                # Per-object translation - need to match objects
                objs = partition_by_connectivity(s_test)
                if len(objs) == len(rule.params['translations']):
                    offset = 0
                    for obj, trans in zip(objs, rule.params['translations']):
                        trans_arr = np.array(trans)
                        n_pts = obj.n_points
                        # Find which points belong to this object
                        # (simplified: assume order preserved)
                        result_points[offset:offset+n_pts, :len(trans_arr)] += trans_arr
                        offset += n_pts
        
        elif rule.action == "DUPLICATE":
            # Duplicate all points
            factor = rule.params.get('factor', 2)
            duplicated = [result_points]
            for i in range(1, factor):
                duplicated.append(result_points.copy())
            result_points = np.vstack(duplicated)
    
    return State(result_points)


def solve_arc_semantically(training_pairs: List[Tuple['State', 'State']], 
                           test_input: 'State',
                           grid_shape: Tuple[int, int] = None) -> Tuple['State', str]:
    """
    Complete semantic ARC solver.
    
    Returns (predicted_output, explanation)
    """
    # Derive rules from first training pair (could use all)
    train_in, train_out = training_pairs[0]
    rules = derive_semantic_rules(train_in, train_out, grid_shape)
    
    # Apply rules to test
    predicted = apply_semantic_rules(rules, test_input, training_pairs)
    
    # Generate explanation
    explanation = explain_semantic_resolution(train_in, train_out, grid_shape)
    
    return predicted, explanation


def unify_rules_across_pairs(training_pairs: List[Tuple['State', 'State']],
                              grid_shape: Tuple[int, int] = None) -> Dict[str, Any]:
    """
    Find generalizable patterns across ALL training pairs.
    
    The key insight for ARC: rules may have different VALUES per pair,
    but same STRUCTURE. We need to find:
    1. Which rule TYPES are consistent across pairs
    2. What META-PATTERN governs the parameter changes
    """
    all_rules = []
    
    # Derive rules from each pair
    for i, (s_in, s_out) in enumerate(training_pairs):
        rules = derive_semantic_rules(s_in, s_out, grid_shape)
        all_rules.append({'pair': i, 'rules': rules})
    
    # Find structural commonality
    analysis = {
        'n_pairs': len(training_pairs),
        'rule_types': {},
        'structural_consistency': {},
        'meta_pattern': None,
        'explanation': []
    }
    
    # Count rule type occurrences
    for pair_info in all_rules:
        for rule in pair_info['rules']:
            action = rule.action
            if action not in analysis['rule_types']:
                analysis['rule_types'][action] = {'count': 0, 'pairs': [], 'params': []}
            analysis['rule_types'][action]['count'] += 1
            analysis['rule_types'][action]['pairs'].append(pair_info['pair'])
            analysis['rule_types'][action]['params'].append(rule.params)
    
    # Analyze structural consistency
    n_pairs = len(training_pairs)
    for action, info in analysis['rule_types'].items():
        # Rule appears in all pairs?
        appears_in_all = len(set(info['pairs'])) == n_pairs
        analysis['structural_consistency'][action] = {
            'universal': appears_in_all,
            'coverage': len(set(info['pairs'])) / n_pairs
        }
    
    # Try to find meta-pattern for color changes
    if 'PERMUTE' in analysis['rule_types']:
        color_rules = analysis['rule_types']['PERMUTE']
        
        # Analyze all color mappings
        all_color_changes = []
        for params in color_rules['params']:
            if 'changes' in params:
                for colors_in, colors_out in params['changes']:
                    for c_in, c_out in zip(colors_in, colors_out):
                        all_color_changes.append((float(c_in), float(c_out)))
        
        # Look for patterns in the color mappings
        if all_color_changes:
            # Check if there's a consistent function f(c_in) = c_out
            diffs = [c_out - c_in for c_in, c_out in all_color_changes]
            
            # All same diff? (constant offset)
            if len(set(diffs)) == 1:
                analysis['meta_pattern'] = {
                    'type': 'constant_offset',
                    'offset': diffs[0]
                }
                analysis['explanation'].append(
                    f"Color rule: c_out = c_in + {diffs[0]} (constant offset)"
                )
            
            # All same modulo operation?
            elif all(d % 10 == diffs[0] % 10 for d in diffs):
                analysis['meta_pattern'] = {
                    'type': 'modular',
                    'base': 10,
                    'offset_mod': diffs[0] % 10
                }
                analysis['explanation'].append(
                    f"Color rule: c_out = c_in + k (mod 10) pattern"
                )
            
            else:
                # No simple pattern - context-dependent
                analysis['meta_pattern'] = {
                    'type': 'context_dependent',
                    'requires': 'object context analysis'
                }
                analysis['explanation'].append(
                    "Color mapping is context-dependent (no universal formula)"
                )
                
                # Analyze by object position/context
                # This is where the semantic graph really helps!
                analysis['explanation'].append(
                    "Need to analyze: which object property determines the color mapping?"
                )
    
    # Generate summary
    universal_rules = [
        action for action, cons in analysis['structural_consistency'].items()
        if cons['universal']
    ]
    
    analysis['summary'] = {
        'universal_rule_types': universal_rules,
        'generalization_strategy': 'meta_pattern' if analysis['meta_pattern'] else 'structural_only',
        'confidence': 'high' if universal_rules else 'low'
    }
    
    return analysis


def explain_cross_pair_generalization(training_pairs: List[Tuple['State', 'State']],
                                       grid_shape: Tuple[int, int] = None) -> str:
    """
    Explain how the transformation generalizes across training pairs.
    """
    analysis = unify_rules_across_pairs(training_pairs, grid_shape)
    
    lines = ["=" * 60]
    lines.append("CROSS-PAIR GENERALIZATION ANALYSIS")
    lines.append("=" * 60)
    lines.append(f"\nAnalyzed {analysis['n_pairs']} training pairs.\n")
    
    lines.append("Rule Type Frequency:")
    for action, info in analysis['rule_types'].items():
        cons = analysis['structural_consistency'][action]
        status = "✓ Universal" if cons['universal'] else f"Partial ({cons['coverage']:.0%})"
        lines.append(f"  • {action}: {status}")
    
    lines.append("\nMeta-Pattern Analysis:")
    if analysis['meta_pattern']:
        mp = analysis['meta_pattern']
        lines.append(f"  Type: {mp['type']}")
        for k, v in mp.items():
            if k != 'type':
                lines.append(f"  {k}: {v}")
    else:
        lines.append("  No universal meta-pattern found")
    
    lines.append("\nExplanation:")
    for exp in analysis['explanation']:
        lines.append(f"  → {exp}")
    
    lines.append("\nGeneralization Summary:")
    summary = analysis['summary']
    lines.append(f"  Universal rule types: {summary['universal_rule_types']}")
    lines.append(f"  Strategy: {summary['generalization_strategy']}")
    lines.append(f"  Confidence: {summary['confidence']}")
    
    lines.append("=" * 60)
    return "\n".join(lines)


def learn_global_mapping(training_pairs: List[Tuple['State', 'State']]) -> Dict[str, Any]:
    """
    Learn global mapping from UNION of all training pairs.
    
    KEY INSIGHT: Each pair provides PART of the global rule.
    Generalization = UNION, not intersection.
    """
    from ..topology import partition_by_connectivity
    
    global_color_map = {}
    conflicts = []
    
    for pair_idx, (s_in, s_out) in enumerate(training_pairs):
        objs_in = partition_by_connectivity(s_in)
        objs_out = partition_by_connectivity(s_out)
        
        for obj_in, obj_out in zip(objs_in, objs_out):
            # Get unique colors per object
            if obj_in.n_dims >= 3 and obj_out.n_dims >= 3:
                c_in = float(np.round(obj_in.points[0, 2], 0))
                c_out = float(np.round(obj_out.points[0, 2], 0))
                
                if c_in not in global_color_map:
                    global_color_map[c_in] = c_out
                elif global_color_map[c_in] != c_out:
                    conflicts.append({
                        'pair': pair_idx,
                        'color': c_in,
                        'existing': global_color_map[c_in],
                        'new': c_out
                    })
    
    # Complete symmetric bijection: if A→B and B→A, add all inverses
    symmetric_pairs = []
    for c_in, c_out in list(global_color_map.items()):
        if c_out in global_color_map and global_color_map[c_out] == c_in:
            symmetric_pairs.append((c_in, c_out))
    
    # If we have symmetric pairs, the bijection is symmetric: add inverses
    if symmetric_pairs:
        for c_in, c_out in list(global_color_map.items()):
            if c_out not in global_color_map:
                global_color_map[c_out] = c_in  # Add inverse
    
    return {
        'color_map': global_color_map,
        'conflicts': conflicts,
        'is_consistent': len(conflicts) == 0,
        'coverage': len(global_color_map),
        'symmetric': len(symmetric_pairs) > 0
    }


def solve_arc_with_global_mapping(training_pairs: List[Tuple['State', 'State']],
                                   test_input: 'State') -> Tuple['State', str]:
    """
    Solve ARC using global mapping learned from UNION of all pairs.
    
    This is the correct generalization approach:
    - Learn color map from ALL pairs (union, not intersection)
    - Apply to test
    """
    from ..state import State
    
    # Learn global mapping
    learned = learn_global_mapping(training_pairs)
    color_map = learned['color_map']
    
    # Apply to test
    result_points = test_input.points.copy()
    
    if test_input.n_dims >= 3:
        for i in range(len(result_points)):
            old_color = float(np.round(result_points[i, 2], 0))
            if old_color in color_map:
                result_points[i, 2] = color_map[old_color]
    
    predicted = State(result_points)
    
    # Generate explanation
    lines = ["=" * 60]
    lines.append("GLOBAL MAPPING SOLUTION")
    lines.append("=" * 60)
    lines.append(f"\nLearned from {len(training_pairs)} training pairs (UNION approach).")
    lines.append(f"\nGlobal Color Mapping ({learned['coverage']} colors):")
    for c_in in sorted(color_map.keys()):
        lines.append(f"  {int(c_in)} → {int(color_map[c_in])}")
    
    if learned['conflicts']:
        lines.append(f"\n⚠ Conflicts detected: {len(learned['conflicts'])}")
        for c in learned['conflicts']:
            lines.append(f"  Pair {c['pair']}: {c['color']} → {c['new']} vs {c['existing']}")
    else:
        lines.append("\n✓ No conflicts - mapping is consistent!")
    
    lines.append("=" * 60)
    explanation = "\n".join(lines)
    
    return predicted, explanation


def detect_symmetry(s_in: 'State', s_out: 'State', n_spatial: int = 2) -> Optional[Dict[str, Any]]:
    """
    Algebraic Symmetry Detection.
    
    Tests if s_out matches s_in reflected along any combination of spatial axes.
    Reflection = multiplication by diagonal sign matrix: R = diag(signs)
    
    Mathematical: reflected = centroid + (points - centroid) @ diag(signs)
    
    Returns best matching reflection_signs or None if no match.
    """
    if s_in.n_points != s_out.n_points:
        return None
    
    from ..topology import view_as_void
    from itertools import product
    
    n_dims = s_in.n_dims
    centroid = s_in.centroid
    
    # Test all 2^n_spatial sign combinations
    all_sign_combos = list(product([1.0, -1.0], repeat=min(n_spatial, n_dims)))
    
    best_match = None
    best_coverage = 0
    
    for signs_tuple in all_sign_combos:
        # Build full signs array (spatial dims get sign, chromatic dims stay +1)
        signs = np.ones(n_dims)
        signs[:len(signs_tuple)] = np.array(signs_tuple)
        
        # Algebraic reflection via broadcasting
        reflected = centroid + (s_in.points - centroid) * signs
        
        # Compare with s_out using set intersection
        ref_void = view_as_void(np.round(reflected, 4).astype(np.float64))
        out_void = view_as_void(np.round(s_out.points, 4).astype(np.float64))
        
        # Count matching points
        coverage = np.sum(np.isin(ref_void, out_void)) / len(ref_void) if len(ref_void) > 0 else 0
        
        if coverage > best_coverage:
            best_coverage = coverage
            best_match = {
                'signs': signs,
                'coverage': coverage,
                'is_identity': np.all(np.array(signs_tuple) == 1.0)
            }
    
    # Only return if significant match and not identity
    if best_match and best_coverage > 0.9 and not best_match['is_identity']:
        return best_match
    
    return None


def detect_d2_tiling(s_in: 'State', s_out: 'State') -> Optional[Dict[str, Any]]:
    """
    Detect if output is input tiled with D2 (dihedral) group action.
    
    D2 = {identity, flip_h, flip_v, flip_both}
    
    ALGEBRAIC: Checks if output is 2x extent of input AND each quadrant
    matches a D2 transformation of the input.
    
    Returns derived transformation parameters or None.
    """
    from ..topology import view_as_void
    
    if s_in.n_dims < 3 or s_out.n_dims < 3:
        return None
    
    # Get extents in first 2 spatial dims
    in_extent = s_in.points[:, :2].max(axis=0) - s_in.points[:, :2].min(axis=0) + 1
    out_extent = s_out.points[:, :2].max(axis=0) - s_out.points[:, :2].min(axis=0) + 1
    
    # Check if output is exactly 2x input in both dims
    ratio = out_extent / in_extent
    if not (np.allclose(ratio[0], 2.0, atol=0.1) and np.allclose(ratio[1], 2.0, atol=0.1)):
        return None
    
    h, w = int(in_extent[0]), int(in_extent[1])
    
    # Normalize input to origin
    in_min = s_in.points[:, :2].min(axis=0)
    in_pts_norm = s_in.points.copy()
    in_pts_norm[:, :2] -= in_min
    
    # Create grid representation for comparison
    in_grid = np.full((h, w), -1, dtype=int)
    for p in in_pts_norm:
        r, c = int(p[0]), int(p[1])
        if 0 <= r < h and 0 <= c < w:
            in_grid[r, c] = int(p[2])
    
    # Normalize output to origin
    out_min = s_out.points[:, :2].min(axis=0)
    out_pts_norm = s_out.points.copy()
    out_pts_norm[:, :2] -= out_min
    
    out_grid = np.full((2*h, 2*w), -1, dtype=int)
    for p in out_pts_norm:
        r, c = int(p[0]), int(p[1])
        if 0 <= r < 2*h and 0 <= c < 2*w:
            out_grid[r, c] = int(p[2])
    
    # D2 group elements
    identity = in_grid
    flip_h = np.fliplr(in_grid)
    flip_v = np.flipud(in_grid)
    flip_both = np.flipud(np.fliplr(in_grid))
    
    # Check each quadrant
    quadrants = {
        'top_left': out_grid[:h, :w],
        'top_right': out_grid[:h, w:],
        'bottom_left': out_grid[h:, :w],
        'bottom_right': out_grid[h:, w:]
    }
    
    # Expected pattern: one of each D2 element
    expected = [
        ('top_left', flip_both),
        ('top_right', flip_v),
        ('bottom_left', flip_h),
        ('bottom_right', identity)
    ]
    
    matches = sum(np.array_equal(quadrants[name], grid) for name, grid in expected)
    
    if matches == 4:
        return {
            'type': 'd2_tiling',
            'input_extent': np.array([h, w]),
            'group': 'D2',
            'coverage': 1.0
        }
    
    return None


def derive_affine_composition(s_in: 'State', s_out: 'State') -> Optional[Dict[str, Any]]:
    """
    Unified Affine Primitive Decomposition.
    
    Derives T(p) = A @ p + b procedurally from data.
    
    For tiling (output > input): decomposes into Union of T_k
    Each T_k = (A_k, b_k) derived via least squares from correspondences.
    
    The "semantics" EMERGE from A, b values - not hardcoded.
    """
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    
    n_in = s_in.n_points
    n_out = s_out.n_points
    n_dims = min(s_in.n_dims, s_out.n_dims)
    
    if n_in == 0 or n_out == 0:
        return None
    
    # Case 1: Same size → spatial affine + chromatic bijection
    if n_in == n_out:
        n_spatial = min(2, n_dims)  # First 2 dims are spatial
        
        # Use COLOR-weighted Hungarian for robust correspondence
        spatial_dists = cdist(s_in.points[:, :n_spatial], s_out.points[:, :n_spatial])
        if n_dims > n_spatial:
            color_dists = cdist(s_in.points[:, n_spatial:], s_out.points[:, n_spatial:])
            total_dists = spatial_dists + 1000 * color_dists
        else:
            total_dists = spatial_dists
        
        row_ind, col_ind = linear_sum_assignment(total_dists)
        
        # Derive SPATIAL affine only (dims 0-1)
        X_spatial = s_in.points[row_ind, :n_spatial]
        Y_spatial = s_out.points[col_ind, :n_spatial]
        
        X_aug = np.hstack([X_spatial, np.ones((len(X_spatial), 1))])
        solution, _, _, _ = np.linalg.lstsq(X_aug, Y_spatial, rcond=None)
        
        A_spatial = solution[:-1, :].T  # (n_spatial, n_spatial)
        b_spatial = solution[-1, :]      # (n_spatial,)
        
        # Derive CHROMATIC bijection (dim 2+)
        chromatic_bijection = {}
        if n_dims > n_spatial:
            for i, j in zip(row_ind, col_ind):
                c_in = tuple(np.round(s_in.points[i, n_spatial:], 0).tolist())
                c_out = tuple(np.round(s_out.points[j, n_spatial:], 0).tolist())
                if c_in not in chromatic_bijection:
                    chromatic_bijection[c_in] = c_out
        
        # Build full A matrix with identity for chromatic dims
        A = np.eye(n_dims)
        A[:n_spatial, :n_spatial] = A_spatial
        b = np.zeros(n_dims)
        b[:n_spatial] = b_spatial
        
        # Compute coverage
        Y_pred_spatial = X_spatial @ A_spatial.T + b_spatial
        errors = np.linalg.norm(Y_pred_spatial - Y_spatial, axis=1)
        coverage = np.mean(errors < 0.5)
        
        return {
            'type': 'single',
            'A': A,
            'b': b,
            'chromatic_bijection': chromatic_bijection,
            'coverage': coverage,
            'n_components': 1
        }
    
    # Case 2: Tiling (output is multiple of input) → Union of T_k
    ratio = n_out / n_in
    if ratio > 1 and ratio == int(ratio):
        n_tiles = int(ratio)
        
        # Compute input extent for tile assignment
        in_min = s_in.points.min(axis=0)
        in_extent = s_in.points.max(axis=0) - in_min + 1
        out_min = s_out.points.min(axis=0)
        
        # Assign each output point to a tile using modular arithmetic
        # tile_idx = floor((out_pos - out_min) / in_extent)
        out_relative = s_out.points - out_min
        tile_coords = (out_relative[:, :2] / in_extent[:2]).astype(int)  # 2D tile grid
        
        # Create unique tile ID from coords
        tile_ids = tile_coords[:, 0] * 100 + tile_coords[:, 1]  # Simple hash
        unique_tiles = np.unique(tile_ids)
        
        transforms = []
        
        for tile_id in unique_tiles:
            mask = tile_ids == tile_id
            tile_out = s_out.points[mask, :n_dims]
            
            if len(tile_out) != n_in:
                continue  # Skip incomplete tiles
            
            # Match to input via Hungarian with COLOR-weighted distance
            # Color must match exactly, spatial can vary
            spatial_dists = cdist(s_in.points[:, :2], tile_out[:, :2])
            
            # Color distance (high weight = must match)
            if n_dims > 2:
                color_dists = cdist(s_in.points[:, 2:], tile_out[:, 2:])
                # Color mismatch = very high penalty
                total_dists = spatial_dists + 1000 * color_dists
            else:
                total_dists = spatial_dists
            
            row_ind, col_ind = linear_sum_assignment(total_dists)
            
            X = s_in.points[row_ind, :n_dims]
            Y = tile_out[col_ind]
            
            # Derive affine for this tile
            X_aug = np.hstack([X, np.ones((len(X), 1))])
            solution, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)
            
            A_k = solution[:-1, :].T
            b_k = solution[-1, :]
            
            transforms.append({'A': A_k, 'b': b_k})
        
        return {
            'type': 'composition',
            'transforms': transforms,
            'n_components': len(transforms),
            'coverage': 1.0  # TODO: compute actual
        }
    
    return None


def detect_causal_groups(training_pairs: List[Tuple['State', 'State']]) -> Dict[str, Any]:
    """
    Greedy detection of causal groups with GENERALIZED transformation:
    
    Decomposes T = T_spatial ∘ T_chromatic ∘ T_topological
    
    - extent_ratio: shape change (1.0 = same, 2.0 = doubled, etc)
    - spatial_type: STATIC, TRANSLATE, TILE
    - chromatic_type: PRESERVE, BIJECTION
    - topology_type: SAME, GROW, SHRINK
    """
    from ..topology import partition_by_connectivity
    
    observations = []
    chromatic_bijections = {}  # c_in → c_out
    
    # Global extent ratios across all pairs
    global_extent_ratios = []
    
    for pair_idx, (s_in, s_out) in enumerate(training_pairs):
        # Compute extent ratio at STATE level (shape change detection)
        n_spatial = min(s_in.n_dims, s_out.n_dims, 2)  # First 2 dims are spatial
        
        extent_in = s_in.points[:, :n_spatial].max(axis=0) - s_in.points[:, :n_spatial].min(axis=0) + 1
        extent_out = s_out.points[:, :n_spatial].max(axis=0) - s_out.points[:, :n_spatial].min(axis=0) + 1
        extent_ratio = tuple(np.round(extent_out / (extent_in + 1e-10), 1))
        global_extent_ratios.append(extent_ratio)
        
        # Detect periodicity in output (for tiling patterns)
        # Periodicity = autocorrelation peaks at regular intervals
        periodicity_detected = False
        if n_spatial >= 2 and s_out.n_points > 10:
            # Simple periodicity check: unique position count vs total
            unique_colors = len(np.unique(np.round(s_out.points[:, 2] if s_out.n_dims > 2 else s_out.points[:, 0], 0)))
            # If few unique colors but many points, likely periodic
            if unique_colors <= 4 and s_out.n_points >= unique_colors * 3:
                periodicity_detected = True
        
        objs_in = partition_by_connectivity(s_in)
        objs_out = partition_by_connectivity(s_out)
        
        for obj_idx, (obj_in, obj_out) in enumerate(zip(objs_in, objs_out)):
            n_dims = min(obj_in.n_dims, obj_out.n_dims)
            
            # Detect which dimensions are discrete (chromatic)
            # A dimension is discrete if it has few unique values across the object
            discrete_dims = []
            continuous_dims = []
            for d in range(n_dims):
                n_unique = len(np.unique(np.round(obj_in.points[:, d], 0)))
                # Heuristic: if all points have same value, it's chromatic-like
                if n_unique == 1:
                    discrete_dims.append(d)
                else:
                    continuous_dims.append(d)
            
            # Compute SPATIAL transformation (continuous dims only)
            if continuous_dims:
                cont_idx = np.array(continuous_dims)
                centroid_in = np.mean(obj_in.points[:, cont_idx], axis=0)
                centroid_out = np.mean(obj_out.points[:, cont_idx], axis=0)
                spatial_delta = centroid_out - centroid_in
                has_spatial_change = not np.allclose(spatial_delta, 0, atol=0.5)
                spatial_delta_rounded = tuple(np.round(spatial_delta, 1))
            else:
                spatial_delta_rounded = ()
                has_spatial_change = False
            
            # Compute CHROMATIC bijection (discrete dims)
            chromatic_map = {}
            has_chromatic_change = False
            for d in discrete_dims:
                c_in = float(np.round(obj_in.points[0, d], 0))
                c_out = float(np.round(obj_out.points[0, d], 0))
                chromatic_map[d] = (c_in, c_out)
                if c_in != c_out:
                    chromatic_bijections[c_in] = c_out
                    has_chromatic_change = True
            
            # Transformation signature by TYPE (not values!)
            # This groups objects by what KIND of transformation they undergo
            dn_points = obj_out.n_points - obj_in.n_points
            transform_type = (
                'SPATIAL' if has_spatial_change else 'STATIC',
                'CHROMATIC' if has_chromatic_change else 'PRESERVE',
                'GROW' if dn_points > 0 else 'SHRINK' if dn_points < 0 else 'SAME'
            )
            
            # PURE VECTORIAL SIGNATURE (no thresholds, no categories)
            # Feature vector: [n_points_normalized, aspect_ratio, relative_mass, pos_x, pos_y, color]
            
            # Normalized n_points (relative to scene max)
            all_masses = np.array([o.n_points for o in objs_in])
            n_points_norm = obj_in.n_points / (all_masses.max() + 1e-6)
            
            # Aspect ratio (continuous)
            if obj_in.n_dims >= 2 and obj_in.n_points > 1:
                extent = obj_in.points[:, :2].max(axis=0) - obj_in.points[:, :2].min(axis=0) + 1
                aspect_ratio = extent[0] / (extent[1] + 1e-6)
            else:
                aspect_ratio = 1.0
            
            # Relative position (normalized continuous vector)
            centroid_in = obj_in.centroid[:2] if obj_in.n_dims >= 2 else np.zeros(2)
            all_centroids = np.array([o.centroid[:2] for o in objs_in if o.n_dims >= 2])
            if len(all_centroids) > 1:
                global_center = all_centroids.mean(axis=0)
                spread = all_centroids.std(axis=0) + 1e-6
                rel_pos = (centroid_in - global_center) / spread  # Normalized
            else:
                rel_pos = np.zeros(2)
            
            # Color (as continuous value, normalized to [0, 1] range if possible)
            color_value = 0.0
            if n_dims > 2:
                color_value = float(np.mean(obj_in.points[:, 2]))
            
            # Build feature vector (pure continuous, no categories)
            feature_vec = np.array([
                n_points_norm,
                aspect_ratio,
                rel_pos[0],
                rel_pos[1],
                color_value
            ])
            
            features = {
                'feature_vec': feature_vec,  # Pure continuous vector
                'n_points': obj_in.n_points,
                'color': int(np.round(color_value)) if color_value > 0 else None
            }
            
            observations.append({
                'pair': pair_idx,
                'obj': obj_idx,
                'features': features,
                'transform_type': transform_type,
                'spatial_delta': spatial_delta_rounded,
                'chromatic': chromatic_map
            })
    
    # Group by transformation TYPE (not values!)
    groups = {}
    for obs in observations:
        sig = obs['transform_type']
        if sig not in groups:
            groups[sig] = {'obs': [], 'spatial': [], 'chromatic': []}
        groups[sig]['obs'].append(obs['features'])
        groups[sig]['spatial'].append(obs['spatial_delta'])
        groups[sig]['chromatic'].append(obs['chromatic'])
    
    # Profile each group with PURE VECTORIAL CENTROID
    group_profiles = {}
    for sig, data in groups.items():
        if not data['obs']:
            continue
        
        # Stack all feature vectors for this group
        feature_vecs = [f.get('feature_vec') for f in data['obs'] if f.get('feature_vec') is not None]
        
        if feature_vecs:
            # Centroid = mean of all feature vectors (no thresholds, no categories)
            feature_matrix = np.vstack(feature_vecs)
            centroid = feature_matrix.mean(axis=0)
            
            group_profiles[sig] = {
                'count': len(data['obs']),
                'feature_centroid': centroid,  # Pure vectorial centroid
                'transform_type': sig,
                'colors': list(set(f.get('color') for f in data['obs'] if f.get('color') is not None))
            }
    
    # Compute dominant extent_ratio (most common)
    dominant_extent_ratio = max(set(global_extent_ratios), key=global_extent_ratios.count) if global_extent_ratios else (1.0, 1.0)
    is_shape_change = any(r != 1.0 for r in dominant_extent_ratio)
    
    # DERIVE RULES from observations (pure vectorial - no categorical predicates)
    # Each rule = (feature_centroid, transformation)
    rules = []
    for sig, profile in group_profiles.items():
        if 'feature_centroid' in profile:
            rules.append({
                'feature_centroid': profile['feature_centroid'],
                'transformation': sig,
                'colors': profile.get('colors', [])
            })
    
    return {
        'groups': groups,
        'profiles': group_profiles,
        'n_groups': len(groups),
        'observations': observations,
        'chromatic_bijection': chromatic_bijections,
        'extent_ratio': dominant_extent_ratio,
        'is_shape_change': is_shape_change,
        'rules': rules
    }


def explain_rules(rules: List[Dict]) -> str:
    """
    Generate human-readable explanation of derived rules.
    Pure vectorial: shows feature centroid and transformation.
    """
    lines = []
    for i, rule in enumerate(rules, 1):
        centroid = rule.get('feature_centroid', np.zeros(5))
        trans = rule['transformation']
        colors = rule.get('colors', [])
        
        centroid_str = ", ".join(f"{v:.2f}" for v in centroid)
        spatial_t, chromatic_t, size_t = trans
        
        lines.append(f"Rule {i}: centroid=[{centroid_str}] → {spatial_t}/{chromatic_t}/{size_t}")
        if colors:
            lines[-1] += f" (colors: {colors})"
    
    return "\n".join(lines) if lines else "No rules derived"


def match_object_to_group(obj_features: Dict, group_profiles: Dict) -> Optional[Tuple]:
    """
    Match an object to best causal group via PURE EUCLIDEAN DISTANCE.
    
    No thresholds, no weights, no categories.
    Returns the transformation signature for the closest group.
    """
    if not group_profiles:
        return None
    
    obj_vec = obj_features.get('feature_vec')
    if obj_vec is None:
        return None
    
    best_sig = None
    best_dist = float('inf')
    
    # Pure distance-based matching: closest centroid wins
    for sig, profile in group_profiles.items():
        centroid = profile.get('feature_centroid')
        if centroid is None:
            continue
        
        # Euclidean distance in feature space (no thresholds, no weights)
        dist = np.linalg.norm(obj_vec - centroid)
        
        if dist < best_dist:
            best_dist = dist
            best_sig = sig
    
    return best_sig


def solve_arc_with_causal_groups(training_pairs: List[Tuple['State', 'State']],
                                  test_input: 'State') -> Tuple['State', str]:
    """
    Solve ARC using greedy causal group detection.
    
    Applies HYBRID transformations:
    - Spatial (continuous dims): translation
    - Chromatic (discrete dims): bijection
    """
    from ..topology import partition_by_connectivity
    from ..state import State
    from ..operators.affine import ValuePermutationOperator
    
    # Detect groups for transformation types
    detected = detect_causal_groups(training_pairs)
    profiles = detected['profiles']
    
    # Use learn_global_mapping for accurate chromatic bijection
    # (hybrid detection can have dimension detection issues)
    mapping = learn_global_mapping(training_pairs)
    chromatic_bijection = mapping.get('color_map', {})
    
    # Get extent_ratio for shape changes
    extent_ratio = detected.get('extent_ratio', (1.0, 1.0))
    is_shape_change = detected.get('is_shape_change', False)
    
    # If shape change detected, we need to transform input extent
    # Compute input extent
    n_spatial = min(test_input.n_dims, 2)
    input_min = test_input.points[:, :n_spatial].min(axis=0)
    input_max = test_input.points[:, :n_spatial].max(axis=0)
    input_extent = input_max - input_min + 1
    
    # Compute output extent based on ratio
    output_extent = input_extent * np.array(extent_ratio[:n_spatial])
    
    # Partition test
    test_objs = partition_by_connectivity(test_input)
    
    result_points_list = []
    applied = []
    
    # For ARC: first 2 dims are spatial (row, col), rest are chromatic (color)
    # This is a convention, not hardcode - could be inferred from data characteristics
    n_dims = test_input.n_dims
    continuous_dims = [0, 1] if n_dims >= 2 else list(range(n_dims))
    discrete_dims = list(range(2, n_dims))  # Color dims
    
    # Check for D2 group tiling first (mirror tiling pattern)
    s_in_train, s_out_train = training_pairs[0]
    d2_info = detect_d2_tiling(s_in_train, s_out_train)
    
    if d2_info is not None:
        from ..operators.affine import GroupActionTilingOperator
        
        # Get test input extent
        test_extent = test_input.points[:, :2].max(axis=0) - test_input.points[:, :2].min(axis=0) + 1
        
        # Create D2 group tiling operator
        d2_op = GroupActionTilingOperator.d2_from_extent(test_extent)
        predicted = d2_op.apply(test_input)
        
        return predicted, f"D2 GROUP TILING\nInput: {int(test_extent[0])}x{int(test_extent[1])} → Output: {int(2*test_extent[0])}x{int(2*test_extent[1])}\nGroup: D2 = {{identity, flip_h, flip_v, flip_both}}"
    
    # If regular tiling detected (2x, 3x, etc.), use TilingOperator (algebraic)
    if is_shape_change and all(r > 1.0 for r in extent_ratio):
        from ..operators.replication import TilingOperator
        from ..operators.affine import ValuePermutationOperator
        
        tile_factors = [int(r) for r in extent_ratio[:n_spatial]]
        
        # Build translation grid algebraically using meshgrid
        ranges = [np.arange(tile_factors[d]) * input_extent[d] for d in range(n_spatial)]
        grids = np.meshgrid(*ranges, indexing='ij')
        translations = np.stack([g.ravel() for g in grids], axis=-1)
        
        # Pad translations to full n_dims (zeros for non-spatial dims)
        full_translations = np.zeros((translations.shape[0], n_dims))
        full_translations[:, :n_spatial] = translations
        
        # Apply tiling operator
        tiling_op = TilingOperator(translations=full_translations)
        tiled_state = tiling_op.apply(test_input)
        
        # Apply chromatic bijection if any
        if chromatic_bijection:
            perm_maps = [{} for _ in range(n_dims)]
            for d in discrete_dims:
                perm_maps[d] = chromatic_bijection
            bijection_op = ValuePermutationOperator(permutation_maps=perm_maps)
            predicted = bijection_op.apply(tiled_state)
        else:
            predicted = tiled_state
        
        lines = ["TILING via TilingOperator", f"extent_ratio: {extent_ratio}", f"tile: {tile_factors}"]
        return predicted, "\n".join(lines)
    
    for obj in test_objs:
        
        # PURE VECTORIAL FEATURE EXTRACTION (same as training)
        all_masses = np.array([o.n_points for o in test_objs])
        n_points_norm = obj.n_points / (all_masses.max() + 1e-6)
        
        # Aspect ratio (continuous)
        if obj.n_dims >= 2 and obj.n_points > 1:
            extent = obj.points[:, :2].max(axis=0) - obj.points[:, :2].min(axis=0) + 1
            aspect_ratio = extent[0] / (extent[1] + 1e-6)
        else:
            aspect_ratio = 1.0
        
        # Relative position (normalized continuous vector)
        centroid = obj.centroid[:2] if obj.n_dims >= 2 else np.zeros(2)
        all_test_centroids = np.array([o.centroid[:2] for o in test_objs if o.n_dims >= 2])
        if len(all_test_centroids) > 1:
            global_center = all_test_centroids.mean(axis=0)
            spread = all_test_centroids.std(axis=0) + 1e-6
            rel_pos = (centroid - global_center) / spread
        else:
            rel_pos = np.zeros(2)
        
        # Color (continuous)
        color_value = 0.0
        if obj.n_dims > 2:
            color_value = float(np.mean(obj.points[:, 2]))
        
        # Build feature vector (pure continuous, matches training)
        feature_vec = np.array([
            n_points_norm,
            aspect_ratio,
            rel_pos[0],
            rel_pos[1],
            color_value
        ])
        
        obj_features = {
            'feature_vec': feature_vec,
            'n_points': obj.n_points,
            'color': int(np.round(color_value)) if color_value > 0 else None
        }
        
        # Match to group
        matched_sig = match_object_to_group(obj_features, profiles)
        
        if matched_sig:
            # matched_sig is now transform_type = (SPATIAL_TYPE, CHROMATIC_TYPE, SIZE_TYPE)
            spatial_type, chromatic_type, size_type = matched_sig
            
            # Apply GLOBAL chromatic bijection if chromatic transformation
            if chromatic_type == 'CHROMATIC' and chromatic_bijection:
                perm_maps = [{} for _ in range(obj.n_dims)]
                for d in discrete_dims:
                    if d < len(perm_maps):
                        perm_maps[d] = chromatic_bijection
                bijection_op = ValuePermutationOperator(permutation_maps=perm_maps)
                transformed_state = bijection_op.apply(obj)
                transformed = transformed_state.points
            else:
                transformed = obj.points.copy()
            
            # TODO: For spatial transformations, would need to derive actual delta
            # from training data, not from signature
            
            result_points_list.append(transformed)
            applied.append({
                'type': matched_sig,
                'chromatic_applied': chromatic_type == 'CHROMATIC',
                'n_points': obj.n_points
            })
        else:
            # Use global bijection if no group match
            if chromatic_bijection:
                perm_maps = [{} for _ in range(obj.n_dims)]
                for d in discrete_dims:
                    if d < len(perm_maps):
                        perm_maps[d] = chromatic_bijection
                bijection_op = ValuePermutationOperator(permutation_maps=perm_maps)
                transformed_state = bijection_op.apply(obj)
                transformed = transformed_state.points
            else:
                transformed = obj.points.copy()
            result_points_list.append(transformed)
    
    # Combine all points
    all_points = np.vstack(result_points_list) if result_points_list else test_input.points.copy()
    predicted = State(all_points)
    
    # Explanation
    lines = ["=" * 60]
    lines.append("CAUSAL GROUP SOLUTION (TYPE-BASED)")
    lines.append("=" * 60)
    lines.append(f"\nDetected {detected['n_groups']} causal groups")
    lines.append(f"Global chromatic bijection: {chromatic_bijection}")
    
    # Add derived rules explanation
    rules = detected.get('rules', [])
    if rules:
        lines.append("\n--- DERIVED RULES ---")
        lines.append(explain_rules(rules))
    
    lines.append("\nApplied to test:")
    for a in applied[:5]:
        lines.append(f"  type={a['type']}, chromatic_applied={a['chromatic_applied']}")
    lines.append("=" * 60)
    
    return predicted, "\n".join(lines)


# Export
__all__ = [
    'ConceptType', 'RelationType', 'Concept', 'Relation', 
    'SemanticGraph', 'extract_semantic_graph',
    'InputPropertyMatcher', 'relativize_operator',
    'ObjectContext', 'compute_object_relations', 'build_scene_graph',
    'explain_scene', 'explain_transformation', 'explain_arc_solution',
    'SemanticRule', 'derive_semantic_rules', 'explain_semantic_resolution',
    'apply_semantic_rules', 'solve_arc_semantically',
    'unify_rules_across_pairs', 'explain_cross_pair_generalization',
    'learn_global_mapping', 'solve_arc_with_global_mapping',
    'detect_causal_groups', 'solve_arc_with_causal_groups',
    'derive_affine_composition', 'solve_with_affine_composition'
]


def solve_with_affine_composition(training_pairs: List[Tuple['State', 'State']],
                                    test_input: 'State') -> Tuple['State', str]:
    """
    Unified Affine Primitive Solver.
    
    Learns affine composition T = Union(T_k) from training,
    applies to test. No hardcoded patterns - semantics emerge from derived A, b.
    """
    from ..state import State
    from ..operators.affine import GlobalAffineOperator, GroupActionTilingOperator
    
    if not training_pairs:
        return test_input, "No training pairs"
    
    # Check for D2 group tiling first (mirror tiling pattern)
    s_in, s_out = training_pairs[0]
    d2_info = detect_d2_tiling(s_in, s_out)
    
    if d2_info is not None:
        # Get test input extent
        test_extent = test_input.points[:, :2].max(axis=0) - test_input.points[:, :2].min(axis=0) + 1
        
        # Create D2 group tiling operator
        d2_op = GroupActionTilingOperator.d2_from_extent(test_extent)
        predicted = d2_op.apply(test_input)
        
        return predicted, f"D2 GROUP TILING\nInput: {int(test_extent[0])}x{int(test_extent[1])} → Output: {int(2*test_extent[0])}x{int(2*test_extent[1])}\nGroup: D2 = {{identity, flip_h, flip_v, flip_both}}"
    
    # Learn affine composition from first training pair
    composition = derive_affine_composition(s_in, s_out)
    
    # Fallback to existing solver if affine composition fails
    if composition is None:
        return solve_arc_with_causal_groups(training_pairs, test_input)
    
    n_dims = test_input.n_dims
    
    if composition['type'] == 'single':
        # Single affine transform + chromatic bijection
        A = composition['A']
        b = composition['b']
        chromatic_bijection = composition.get('chromatic_bijection', {})
        
        # Pad A and b to match test dimensions
        if A.shape[0] < n_dims:
            A_padded = np.eye(n_dims)
            A_padded[:A.shape[0], :A.shape[1]] = A
            A = A_padded
            b_padded = np.zeros(n_dims)
            b_padded[:len(b)] = b
            b = b_padded
        
        # Apply spatial affine: output = input @ A.T + b
        new_points = test_input.points @ A.T + b
        
        # Apply chromatic bijection on dims 2+
        if chromatic_bijection:
            n_spatial = 2
            for i in range(len(new_points)):
                c_key = tuple(np.round(test_input.points[i, n_spatial:], 0).tolist())
                if c_key in chromatic_bijection:
                    c_new = chromatic_bijection[c_key]
                    new_points[i, n_spatial:] = c_new
        
        predicted = State(new_points)
        
        explanation = f"Single affine + {len(chromatic_bijection)} color mappings"
        
    elif composition['type'] == 'composition':
        # Union of affine transforms
        all_points = []
        
        for t in composition['transforms']:
            A = t['A']
            b = t['b']
            
            # Pad to match dimensions
            if A.shape[0] < n_dims:
                A_padded = np.eye(n_dims)
                A_padded[:A.shape[0], :A.shape[1]] = A
                A = A_padded
                b_padded = np.zeros(n_dims)
                b_padded[:len(b)] = b
                b = b_padded
            
            new_points = test_input.points @ A.T + b
            all_points.append(new_points)
        
        combined = np.vstack(all_points)
        predicted = State(combined)
        
        explanation = f"Composition of {composition['n_components']} affine transforms"
    else:
        return test_input, f"Unknown composition type: {composition['type']}"
    
    return predicted, explanation
