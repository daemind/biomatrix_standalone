# -*- coding: utf-8 -*-
"""
operators/base.py - Base Operator classes

Contains fundamental building blocks for the operator monoïd.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..state import State

# Import base classes from solver/base.py
from ..base import Operator, SequentialOperator


@dataclass
class IdentityOperator(Operator):
    """Identity Operator: I(S) = S."""
    
    def apply(self, state: 'State') -> 'State':
        return state.copy()
    
    def inverse(self) -> 'IdentityOperator':
        return IdentityOperator()
    
    def to_symbolic(self) -> str:
        return "Id"
    
    @property
    def category(self):
        from ..base import OperatorCategory
        return OperatorCategory.IDENTITY
    
    @property
    def is_identity(self) -> bool:
        return True
    
    @property
    def is_invertible(self) -> bool:
        return True


# Export all
__all__ = ['Operator', 'SequentialOperator', 'IdentityOperator']
