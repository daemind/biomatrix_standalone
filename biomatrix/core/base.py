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
        """Remove consecutive T ∘ T⁻¹ pairs and identity operators."""
        from .operators.base import IdentityOperator
        
        # Filter out identities
        simplified = [op for op in self.steps if not op.is_identity]
        
        if not simplified:
            return IdentityOperator()
        
        if len(simplified) == 1:
            return simplified[0]
        
        return SequentialOperator(steps=simplified)
    
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
