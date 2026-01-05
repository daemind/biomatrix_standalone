# -*- coding: utf-8 -*-
"""
operators/base.py - Base Operator classes

Contains fundamental building blocks for the operator monoïd.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..state import State


class Operator(ABC):
    """Abstract base class for all operators."""
    
    @abstractmethod
    def apply(self, state: 'State') -> 'State':
        """Apply operator to state."""
        pass
    
    @property
    def is_identity(self) -> bool:
        return False


class SequentialOperator(Operator):
    """Compose multiple operators sequentially."""
    
    def __init__(self, operators: list):
        self.operators = operators
    
    def apply(self, state: 'State') -> 'State':
        result = state
        for op in self.operators:
            result = op.apply(result)
        return result


@dataclass
class IdentityOperator(Operator):
    """Identity Operator: I(S) = S."""
    
    def apply(self, state: 'State') -> 'State':
        return state.copy()
    
    @property
    def is_identity(self) -> bool:
        return True


__all__ = ['Operator', 'SequentialOperator', 'IdentityOperator']
