# -*- coding: utf-8 -*-
"""
core/base.py - Core base classes

Fundamental building blocks for BioMatrix.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .state import State


class Operator(ABC):
    """Abstract base class for all operators."""
    
    @abstractmethod
    def apply(self, state: 'State') -> 'State':
        """Apply operator to state."""
        pass
    
    @property
    def is_identity(self) -> bool:
        return False
    
    def __call__(self, state: 'State') -> 'State':
        return self.apply(state)


class SequentialOperator(Operator):
    """Compose multiple operators sequentially."""
    
    def __init__(self, operators: List[Operator]):
        self.operators = operators
    
    def apply(self, state: 'State') -> 'State':
        result = state
        for op in self.operators:
            result = op.apply(result)
        return result


class IdentityOperator(Operator):
    """Identity Operator: I(S) = S."""
    
    def apply(self, state: 'State') -> 'State':
        return state.copy()
    
    @property
    def is_identity(self) -> bool:
        return True


__all__ = ['Operator', 'SequentialOperator', 'IdentityOperator']
