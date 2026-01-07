# -*- coding: utf-8 -*-
"""
derive package - Transformation Derivation Functions

Re-exports all derivation functions from sub-modules for backward compatibility.
"""

# Core entry point
from .core import derive_transformation

# Affine derivation
from .affine import (
    derive_matched_affine,
    derive_affine_centered,
    derive_affine_scaled,
    derive_affine_permutation,
    _solve_similarity
)

# Lifting derivation
from .lifting import derive_lifting


# Union and component matching
from .union import (
    derive_union,
    derive_hierarchical_invariant,
    derive_composite_transform,
    derive_deletion,
    derive_component_permutation,
    _derive_union_matching,
    _derive_deletion_bijection,
    _derive_generative_matching,
    _derive_surjective_simulation,
    _derive_sequence_algebraic,
    _is_subset
)

# Permutation and rank transforms
from .permutation import (
    derive_value_permutation,
    derive_rank_transform,
    derive_fiber_projection
)

# Core functions (active)
from .core import (
    derive_subset,
    derive_select_then_act,
    derive_lift_and_slice,
    derive_force_transform,
    derive_sort_and_align,
    derive_component_resample,
    find_exact_displacement,
    _extract_bijection_for_sta,
    SelectQueryAffineOperator
)

# Algebra (pure algebra implementations)
from .algebra import *

# Causal partition
# from .causal import derive_causal_partition, compute_displacement_field # ALL GONE

# Lattice (Basis only)



# Export all
__all__ = [
    # Main entry
    'derive_transformation',
    
    # Affine
    'derive_matched_affine', 'derive_affine_centered', 'derive_affine_scaled', 
    'derive_affine_permutation', '_solve_similarity',
    
    # Lifting
    'derive_lifting',

    
    # Union
    'derive_union', 'derive_hierarchical_invariant', 'derive_composite_transform',
    'derive_deletion', 'derive_component_permutation', '_derive_union_matching',
    '_derive_deletion_bijection', '_derive_generative_matching', 
    '_derive_surjective_simulation', '_derive_sequence_algebraic', '_is_subset',
    
    # Permutation
    'derive_value_permutation', 'derive_rank_transform', 'derive_fiber_projection',
    
    # Core misc
    'derive_subset', 'derive_select_then_act', 'derive_lift_and_slice', 
    'derive_force_transform', 'derive_sort_and_align', 'derive_component_resample', 
    'find_exact_displacement', '_extract_bijection_for_sta', 'SelectQueryAffineOperator',
    
    # Causal (Purged)
    # 'derive_causal_partition', 'compute_displacement_field',
    
    # Lattice

    'derive_causality'
]
