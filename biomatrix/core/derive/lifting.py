# -*- coding: utf-8 -*-
"""
derive/lifting.py - Lifting and Slice Transform Derivation

Wrapper interface for the Unified Algebraic Solver (derive/algebra.py).
Supersedes legacy procedural heuristics (see archive/legacy_heuristics.py).

AGENT.md Compliant: Pure algebraic, N-dimensional agnostic.
"""

from typing import Optional, List
from ..state import State
from ..operators import Operator

# Lazy import to avoid circular dependency
_algebra_module = None

def _get_algebra():
    global _algebra_module
    if _algebra_module is None:
        from . import algebra as _algebra_module
    return _algebra_module


def derive_lifting(s_in: State, s_out: State) -> Optional[Operator]:
    """
    N-Dimensional Generalized Lifting Kernel.
    Delegates to the Unified Algebraic Solver.
    """
    return _get_algebra().derive_lift_and_slice(s_in, s_out)


def derive_lifted_transform(s_in: State, s_out: State) -> Optional[Operator]:
    """
    Derive generic lifted transformation.
    """
    return _get_algebra().derive_lift_and_slice(s_in, s_out)


def derive_manifold_porter(s_in: State, s_out: State, strategies: List[str] = None) -> Optional[Operator]:
    """
    Universal Law Resolver via Porter Architecture (AGENT.md).
    Delegates to Unified Solver.
    """
    if s_in.is_empty or s_out.is_empty: return None
    return _get_algebra().derive_lift_and_slice(s_in, s_out)
