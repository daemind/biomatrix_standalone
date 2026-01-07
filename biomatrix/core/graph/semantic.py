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


class RelationType(Enum):
    """Types of causal/semantic relations."""
    IDENTITY = auto()      # Same value
    TRANSLATE = auto()     # Shift by amount
    SCALE = auto()         # Multiply by factor
    PERMUTE = auto()       # Reorder/swap
    COPY = auto()          # Duplicate
    DEPENDS_ON = auto()    # Causal dependency
    RELATIVE_TO = auto()   # Referenced to another concept


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


# Export
__all__ = [
    'ConceptType', 'RelationType', 'Concept', 'Relation', 
    'SemanticGraph', 'extract_semantic_graph',
    'InputPropertyMatcher', 'relativize_operator'
]
