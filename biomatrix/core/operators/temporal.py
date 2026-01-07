# -*- coding: utf-8 -*-
"""
operators/temporal.py - Temporal Operators for Dynamic Laws
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from .base import Operator
from ..state import State

@dataclass
class TemporalOperator(Operator):
    """
    Represents a dynamic law F such that S_{t+1} = F(S_t, t)
    
    Attributes:
        velocity: Constant linear velocity vector (Newton 1)
        acceleration: Constant acceleration vector (Newton 2)
        period: Period of cyclic motion (in frames)
        phase_offset: Phase offset for cyclic motion
        sub_operators: List of TemporalOperators for partitioned dynamics
    """
    velocity: Optional[np.ndarray] = None
    acceleration: Optional[np.ndarray] = None
    period: Optional[int] = None
    sub_operators: List['TemporalOperator'] = None
    
    def apply(self, state: State, t: int = 0) -> State:
        """
        Apply dynamics to predict state at t+1 given state at t.
        Note: Simple affine integration for now.
        """
        if state.is_empty:
            return state
            
        points = state.points.copy()
        
        # 1. Apply Partitioned Dynamics
        if self.sub_operators:
             pass
            
        # 2. Apply Global Kinematics
        if self.velocity is not None:
            points += self.velocity
            
        if self.acceleration is not None:
            # v_t = v_0 + a*t. dx = v_t * dt.
            # Simple Euler integration step
            # S_{t+1} = S_t + v + a*t
            points += self.acceleration * t 
            
        return State(points)
        
    def __repr__(self):
        parts = []
        if self.velocity is not None:
            parts.append(f"v={np.round(self.velocity, 3)}")
        if self.acceleration is not None:
            parts.append(f"a={np.round(self.acceleration, 3)}")
        if self.period is not None:
            parts.append(f"T={self.period}")
        return f"TemporalOp({', '.join(parts)})"
