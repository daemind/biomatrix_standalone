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
    
    return {
        'color_map': global_color_map,
        'conflicts': conflicts,
        'is_consistent': len(conflicts) == 0,
        'coverage': len(global_color_map)
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
    'learn_global_mapping', 'solve_arc_with_global_mapping'
]
