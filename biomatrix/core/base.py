#!/usr/bin/env python3
"""
base.py - Abstract Base Classes for Algebraic Operators.
Resolves circular dependencies between operators.py and transform.py.
"""
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .state import State

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
    
    # === Algebraic Properties ===
        
    @property
    def is_identity(self) -> bool:
        return False
    
    @property
    def is_invertible(self) -> bool:
        return False
    
    @property
    def is_linear(self) -> bool:
        return False
    
    @property
    def preserves_mass(self) -> bool:
        """True if |T(S)| = |S| for all S."""
        return True

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
