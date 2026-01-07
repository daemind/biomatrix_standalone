#!/usr/bin/env python3
"""
base.py - Abstract Base Classes for Algebraic Operators.
Resolves circular dependencies between operators.py and transform.py.
"""
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Tuple, Optional
from dataclasses import dataclass
from enum import Enum, auto

if TYPE_CHECKING:
    from .state import State


class OperatorCategory(Enum):
    """Categorical classification for composition inference."""
    BIJECTION = auto()      # |T(S)| = |S|, invertible (Affine, Permutation)
    INJECTION = auto()      # |T(S)| <= |S|, (Select, Filter, Delete)
    SURJECTION = auto()     # |T(S)| >= |S|, (Tile, Replicate, Minkowski)
    PROJECTION = auto()     # Idempotent: T² = T (Filter, Select)
    IDENTITY = auto()       # T(S) = S


class NotInvertibleError(Exception):
    """Raised when an operator has no inverse."""
    pass


class Operator(ABC):
    """Abstract base for all operators in the monoïd."""
    
    @abstractmethod
    def apply(self, state: 'State') -> 'State':
        pass
    
    # === Algebraic Methods ===
    
    def inverse(self) -> 'Operator':
        """Return T⁻¹ such that T⁻¹ ∘ T = Identity."""
        raise NotInvertibleError(f"{type(self).__name__} has no explicit inverse")
    
    def simplify(self) -> 'Operator':
        """Algebraic simplification (e.g., T ∘ T⁻¹ = Id)."""
        return self
    
    def to_symbolic(self) -> str:
        """Symbolic representation for human readability."""
        return type(self).__name__
    
    def compose(self, other: 'Operator') -> 'SequentialOperator':
        """Monoïd composition: self ∘ other."""
        other_steps = other.steps if isinstance(other, SequentialOperator) else [other]
        self_steps = self.steps if isinstance(self, SequentialOperator) else [self]
        return SequentialOperator(steps=other_steps + self_steps)
    
    def __matmul__(self, other: 'Operator') -> 'SequentialOperator':
        return self.compose(other)
    
    # === Type Signature for Composition Inference ===
    
    @property
    def category(self) -> OperatorCategory:
        """Categorical type for composition rules."""
        return OperatorCategory.BIJECTION  # Default conservative
    
    @property
    def type_signature(self) -> Tuple[str, str]:
        """(InputType, OutputType) for composition type checking.
        
        Types: 'State', 'Subset', 'Superset', 'Same'
        """
        return ('State', 'State')  # Generic
    
    def is_composable_with(self, other: 'Operator') -> bool:
        """Check if self ∘ other is well-typed."""
        # For now, all operators are composable (State -> State)
        # Future: check dimensional compatibility
        return True
    
    # === Algebraic Properties ===
        
    @property
    def is_identity(self) -> bool:
        return False
    
    @property
    def is_invertible(self) -> bool:
        return self.category == OperatorCategory.BIJECTION
    
    @property
    def is_linear(self) -> bool:
        return False
    
    @property
    def is_idempotent(self) -> bool:
        """T² = T (projections, filters)."""
        return self.category == OperatorCategory.PROJECTION
    
    @property
    def preserves_mass(self) -> bool:
        """True if |T(S)| = |S| for all S."""
        return self.category in (OperatorCategory.BIJECTION, OperatorCategory.IDENTITY)

@dataclass
class SequentialOperator(Operator):
    """Monoïd composition: Sequence of operators applied in order."""
    steps: List[Operator]
    
    def apply(self, state: 'State') -> 'State':
        from functools import reduce
        return reduce(lambda s, op: op.apply(s), self.steps, state)
    
    # === Algebraic Methods ===
    
    def inverse(self) -> 'SequentialOperator':
        """(T1 ∘ T2 ∘ ... ∘ Tn)⁻¹ = Tn⁻¹ ∘ ... ∘ T2⁻¹ ∘ T1⁻¹."""
        inv_steps = [op.inverse() for op in reversed(self.steps)]
        return SequentialOperator(steps=inv_steps)
    
    def simplify(self) -> 'Operator':
        """Algebraic simplification: remove identities, T ∘ T⁻¹ pairs, merge affines."""
        from .operators.base import IdentityOperator
        
        # Step 1: Recursively simplify nested steps
        simplified = [op.simplify() if hasattr(op, 'simplify') else op for op in self.steps]
        
        # Step 2: Filter out identities
        simplified = [op for op in simplified if not op.is_identity]
        
        # Step 3: Merge consecutive AffineTransforms
        merged = []
        for op in simplified:
            if merged and self._can_merge(merged[-1], op):
                merged[-1] = merged[-1].compose(op)
            else:
                merged.append(op)
        
        simplified = merged
        
        if not simplified:
            return IdentityOperator()
        
        if len(simplified) == 1:
            return simplified[0]
        
        return SequentialOperator(steps=simplified)
    
    def _can_merge(self, op1: 'Operator', op2: 'Operator') -> bool:
        """Check if two operators can be algebraically merged."""
        # Import here to avoid circular
        from .transform import AffineTransform
        return isinstance(op1, AffineTransform) and isinstance(op2, AffineTransform)
    
    def to_symbolic(self) -> str:
        """Symbolic representation."""
        symbols = [op.to_symbolic() for op in reversed(self.steps)]
        return " ∘ ".join(symbols)
    
    # === Algebraic Properties ===
    
    @property
    def category(self) -> OperatorCategory:
        """Infer category from composition of steps."""
        categories = [op.category for op in self.steps]
        
        # If all bijections, result is bijection
        if all(c == OperatorCategory.BIJECTION for c in categories):
            return OperatorCategory.BIJECTION
        
        # If any surjection, result grows mass
        if any(c == OperatorCategory.SURJECTION for c in categories):
            return OperatorCategory.SURJECTION
        
        # If any injection without surjection, result shrinks mass
        if any(c == OperatorCategory.INJECTION for c in categories):
            return OperatorCategory.INJECTION
        
        return OperatorCategory.BIJECTION  # Default
    
    @property
    def is_identity(self) -> bool:
        return all(op.is_identity for op in self.steps)
    
    @property
    def is_invertible(self) -> bool:
        return all(op.is_invertible for op in self.steps)
    
    @property
    def is_linear(self) -> bool:
        return all(op.is_linear for op in self.steps)
    
    @property
    def preserves_mass(self) -> bool:
        return all(op.preserves_mass for op in self.steps)
        
    def __repr__(self) -> str:
        return " @ ".join([repr(op) for op in reversed(self.steps)])


# === Operator Analysis Utilities ===

def commutes(op1: Operator, op2: Operator, test_state: 'State' = None) -> bool:
    """Check if op1 ∘ op2 = op2 ∘ op1 (approximately, via test state)."""
    if test_state is None:
        # Generate a small test state
        import numpy as np
        test_state = __import__('biomatrix.core.state', fromlist=['State']).State(
            np.random.rand(10, 3) * 10
        )
    
    result_12 = op1.apply(op2.apply(test_state))
    result_21 = op2.apply(op1.apply(test_state))
    
    if result_12.n_points != result_21.n_points:
        return False
    
    # Compare point sets (order-independent)
    import numpy as np
    pts_12 = np.sort(result_12.points, axis=0)
    pts_21 = np.sort(result_21.points, axis=0)
    
    return np.allclose(pts_12, pts_21, atol=1e-6)


def analyze_composition(ops: List[Operator]) -> dict:
    """Analyze a sequence of operators for composition properties."""
    if not ops:
        return {'empty': True}
    
    categories = [op.category for op in ops]
    
    return {
        'n_ops': len(ops),
        'categories': [c.name for c in categories],
        'is_bijective': all(c == OperatorCategory.BIJECTION for c in categories),
        'is_invertible': all(op.is_invertible for op in ops),
        'preserves_mass': all(op.preserves_mass for op in ops),
        'has_surjection': any(c == OperatorCategory.SURJECTION for c in categories),
        'has_injection': any(c == OperatorCategory.INJECTION for c in categories),
        'symbolic': " ∘ ".join(op.to_symbolic() for op in reversed(ops)),
    }


def infer_result_category(ops: List[Operator]) -> OperatorCategory:
    """Infer the category of a composition of operators."""
    if not ops:
        return OperatorCategory.IDENTITY
    
    categories = [op.category for op in ops]
    
    if all(c == OperatorCategory.IDENTITY for c in categories):
        return OperatorCategory.IDENTITY
    
    if all(c in (OperatorCategory.BIJECTION, OperatorCategory.IDENTITY) for c in categories):
        return OperatorCategory.BIJECTION
    
    if any(c == OperatorCategory.SURJECTION for c in categories):
        return OperatorCategory.SURJECTION
    
    if any(c == OperatorCategory.INJECTION for c in categories):
        return OperatorCategory.INJECTION
    
    if any(c == OperatorCategory.PROJECTION for c in categories):
        return OperatorCategory.PROJECTION
    
    return OperatorCategory.BIJECTION


def extract_operator_type(op: Operator) -> str:
    """
    Extract structural type ignoring parameter values.
    For ARC: detect that T(1,0) and T(5,3) are both 'T' (translation).
    """
    import re
    symbolic = op.to_symbolic()
    # Remove numeric parameters: T(1.00, 2.00) -> T(...)
    pattern = re.sub(r'\([^)]*\)', '(...)', symbolic)
    # Remove angle values: R(45.0°) -> R(...)
    pattern = re.sub(r'\([0-9.]+°\)', '(...)', pattern)
    return pattern


def structural_match(op1: Operator, op2: Operator) -> bool:
    """Check if two operators have same structural type (ignoring parameters)."""
    return extract_operator_type(op1) == extract_operator_type(op2)


def symbolic_match(op1: Operator, op2: Operator) -> bool:
    """Check if two operators have equivalent symbolic structure."""
    return op1.to_symbolic() == op2.to_symbolic()


def extract_symbolic_pattern(ops: List[Operator]) -> Optional[str]:
    """
    Extract common symbolic pattern from multiple operators.
    Returns the shared symbolic form if all operators match, None otherwise.
    
    For ARC: Check if solutions from training pairs share same structure.
    """
    if not ops:
        return None
    
    symbols = [op.to_symbolic() for op in ops]
    
    if len(set(symbols)) == 1:
        return symbols[0]
    
    return None


def unify_solutions(derived_ops: List[Operator], pairs: List[Tuple['State', 'State']] = None) -> dict:
    """
    Analyze derived solutions across multiple pairs to find generalizable pattern.
    
    For ARC:
    - pairs: List of (input, output) training pairs
    - derived_ops: Operators derived for each pair
    
    Returns analysis dict with:
    - 'unified': True if all solutions share same structure
    - 'pattern': Common symbolic pattern if unified
    - 'categories': List of operator categories
    - 'details': Per-pair symbolic forms
    """
    if not derived_ops:
        return {'unified': False, 'pattern': None, 'categories': [], 'details': []}
    
    symbols = [op.to_symbolic() for op in derived_ops]
    structures = [extract_operator_type(op) for op in derived_ops]
    categories = [op.category.name if hasattr(op, 'category') else 'UNKNOWN' for op in derived_ops]
    
    # Exact match (same parameters)
    exact_unified = len(set(symbols)) == 1
    # Structural match (same type, different parameters)
    structural_unified = len(set(structures)) == 1
    
    pattern = symbols[0] if exact_unified else None
    structural_pattern = structures[0] if structural_unified else None
    
    # Verification on pairs
    verified = []
    if pairs:
        for op, (s_in, s_out) in zip(derived_ops, pairs):
            result = op.apply(s_in)
            verified.append(result == s_out)
    
    return {
        'unified': exact_unified,
        'structural_unified': structural_unified,
        'pattern': pattern,
        'structural_pattern': structural_pattern,
        'categories': categories,
        'details': symbols,
        'structures': structures,
        'verified': verified if pairs else None,
        'generalizes': structural_unified and all(verified) if pairs else structural_unified,
    }


def extract_parameters(op: Operator) -> dict:
    """
    Extract numeric parameters from an operator.
    For ARC: analyze what varies between training pairs.
    
    Returns dict of parameter names -> values.
    """
    params = {}
    
    # AffineTransform
    if hasattr(op, 'translation') and op.translation is not None:
        import numpy as np
        params['translation'] = op.translation.tolist() if hasattr(op.translation, 'tolist') else list(op.translation)
    
    if hasattr(op, 'linear') and op.linear is not None:
        import numpy as np
        params['linear_det'] = float(np.linalg.det(op.linear))
    
    # LiftedTransform
    if hasattr(op, 'lift') and op.lift is not None:
        params['lifter'] = op.lift.lifter if hasattr(op.lift, 'lifter') else None
    
    if hasattr(op, 'bijection') and op.bijection is not None:
        if hasattr(op.bijection, 'translation') and op.bijection.translation is not None:
            import numpy as np
            t = op.bijection.translation
            params['bijection_translation_norm'] = float(np.linalg.norm(t))
    
    # Tiling
    if hasattr(op, 'translations') and op.translations is not None:
        params['n_tiles'] = len(op.translations)
    
    if hasattr(op, 'count'):
        params['count'] = op.count
    
    # Selection
    if hasattr(op, 'dim'):
        params['dim'] = op.dim
    if hasattr(op, 'value'):
        params['value'] = op.value
    
    # Union/Composition
    if hasattr(op, 'operands'):
        params['n_operands'] = len(op.operands)
    if hasattr(op, 'steps'):
        params['n_steps'] = len(op.steps)
    
    return params


def compare_parameters(ops: List[Operator]) -> dict:
    """
    Compare parameters across multiple operators.
    Returns which parameters are constant vs varying.
    
    For ARC: identify what's invariant across training pairs.
    """
    if not ops:
        return {'constant': {}, 'varying': {}}
    
    all_params = [extract_parameters(op) for op in ops]
    
    # Find all parameter keys
    all_keys = set()
    for p in all_params:
        all_keys.update(p.keys())
    
    constant = {}
    varying = {}
    
    for key in all_keys:
        values = [p.get(key) for p in all_params]
        # Check if all values are equal
        first = values[0]
        if all(v == first for v in values):
            constant[key] = first
        else:
            varying[key] = values
    
    return {'constant': constant, 'varying': varying}


def analyze_arc_task(training_pairs: List[Tuple['State', 'State']], derive_fn=None) -> dict:
    """
    Full ARC task analysis pipeline.
    
    Given training pairs (input, output), derives solutions and analyzes generalization.
    
    Args:
        training_pairs: List of (State_in, State_out) tuples
        derive_fn: Function to derive operator from pair (default: derive_lift_and_slice)
    
    Returns:
        Complete analysis dict for ARC generalization.
    """
    if derive_fn is None:
        # Lazy import to avoid circular
        from .derive.lifting import derive_lifting
        derive_fn = derive_lifting
    
    # 1. Derive solution for each pair
    derived_ops = []
    for s_in, s_out in training_pairs:
        op = derive_fn(s_in, s_out)
        if op:
            derived_ops.append(op)
    
    if not derived_ops:
        return {'success': False, 'error': 'No solutions derived'}
    
    if len(derived_ops) != len(training_pairs):
        return {'success': False, 'error': f'Only {len(derived_ops)}/{len(training_pairs)} pairs solved'}
    
    # 2. Analyze unification
    unification = unify_solutions(derived_ops, training_pairs)
    
    # 3. Extract parameters
    param_comparison = compare_parameters(derived_ops)
    
    # 4. Build result
    return {
        'success': True,
        'n_pairs': len(training_pairs),
        'n_solved': len(derived_ops),
        'structural_unified': unification['structural_unified'],
        'structural_pattern': unification['structural_pattern'],
        'generalizes': unification['generalizes'],
        'verified': unification['verified'],
        'details': unification['details'],
        'constant_params': param_comparison['constant'],
        'varying_params': param_comparison['varying'],
        'operators': derived_ops,  # Return operators for test inference
    }
