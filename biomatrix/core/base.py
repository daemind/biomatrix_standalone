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

class Operator(ABC):
    """
    Abstract base for all operators in the monoïd.
    Algorithm: apply(State) -> State
    """
    
    @abstractmethod
    def apply(self, state: 'State') -> 'State':
        pass
    
    def compose(self, other: 'Operator') -> 'SequentialOperator':
        """
        Monoïd composition: self ∘ other.
        """
        # Determine steps for flattening
        other_steps = other.steps if isinstance(other, SequentialOperator) else [other]
        self_steps = self.steps if isinstance(self, SequentialOperator) else [self]
        
        # New sequence: other applied first, then self
        return SequentialOperator(steps=other_steps + self_steps)
    
    def __matmul__(self, other: 'Operator') -> 'SequentialOperator':
        return self.compose(other)
        
    @property
    def is_identity(self) -> bool:
        return False

@dataclass
class SequentialOperator(Operator):
    """
    Monoïd composition: Sequence of operators applied in order.
    """
    steps: List[Operator]
    
    def apply(self, state: 'State') -> 'State':
        from functools import reduce
        return reduce(lambda s, op: op.apply(s), self.steps, state)
        
    def __repr__(self) -> str:
        return " @ ".join([repr(op) for op in reversed(self.steps)])
