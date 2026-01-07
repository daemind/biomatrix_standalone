# -*- coding: utf-8 -*-
"""
operators package - Operator implementations

Re-exports all operators from sub-modules for backward compatibility.
"""

# Base operators
from .base import Operator, SequentialOperator, IdentityOperator

# Selection operators
from .selection import (
    CropToComponentOperator, SelectBySignatureOperator, SelectThenActOperator,
    SelectByValueOperator, SelectByRangeOperator, SortAndSelectOperator,
    DeleteOperator, FilterOperator, ProjectionOperator, ModularProjectionOperator,
    ExplicitFilterOperator, ConstantOperator
)

# Logic operators
from .logic import (
    LogicOperator, UnionOperator, IntersectionOperator, DifferenceOperator,
    PartitionOperator, PiecewiseOperator, ComponentMapOperator
)

# Replication operators
from .replication import (
    LinearSequenceOperator, KroneckerOperator, RepeatOperator,
    ReplicationOperator, TilingOperator, ReflectTilingOperator,
    AffineTilingOperator, MinkowskiSumOperator, PartialLatticeOperator
)

# Lifting operators
from .lifting import LiftedSliceOperator, FiberProjectionOperator

# Affine and value operators
from .affine import (
    ValueProjectionOperator, ValuePermutationOperator, ScaleOperator,
    ResampleOperator, PermutationOperator, ClampOperator,
    NormalizeOriginOperator, GlobalAffineOperator, RigidAffineForceOperator,
    RigidHomotheticForceOperator, RankByMassOperator, AdditiveOperator,
    InteriorOperator, SortAndAlignOperator, ProjectiveSelectionOperator,
    KernelAffineOperator
)

# Classes ONLY in core.py (not yet migrated to split modules)
from .core import (
    # Topological filters
    TopologicalFilterOperator, ExtremeFilterOperator, IsomorphismFilterOperator,
    # Fill operators
    HullFillOperator, ConvexifyOperator, FillOperator, EnvelopeOperator,
    # Force operators
    ForceOperator, LinearForceOperator,
    # Other
    SequenceOperator,
    # Helper function
    CropOperator, view_as_void
)

# Algebra (pure algebra implementations)
# from .algebra import * # DISABLED to prevent circular import with derive


# Export all
__all__ = [
    # Base
    'Operator', 'SequentialOperator', 'IdentityOperator',
    
    # Selection
    'CropToComponentOperator', 'SelectBySignatureOperator', 'SelectThenActOperator',
    'SelectByValueOperator', 'SelectByRangeOperator', 'SortAndSelectOperator',
    'DeleteOperator', 'FilterOperator', 'ProjectionOperator', 'ModularProjectionOperator',
    'DeleteOperator', 'FilterOperator', 'ProjectionOperator', 'ModularProjectionOperator',
    'ExplicitFilterOperator', 'ConstantOperator',
    
    # Logic
    'LogicOperator', 'UnionOperator', 'IntersectionOperator', 'DifferenceOperator',
    'PartitionOperator', 'PiecewiseOperator', 'ComponentMapOperator',
    
    # Replication
    'LinearSequenceOperator', 'KroneckerOperator', 'RepeatOperator',
    'ReplicationOperator', 'TilingOperator', 'ReflectTilingOperator',
    'AffineTilingOperator', 'MinkowskiSumOperator', 'PartialLatticeOperator',
    
    # Lifting
    'LiftedSliceOperator', 'FiberProjectionOperator',
    
    # Affine
    'ValueProjectionOperator', 'ValuePermutationOperator', 'ScaleOperator',
    'ResampleOperator', 'PermutationOperator', 'ClampOperator',
    'NormalizeOriginOperator', 'GlobalAffineOperator', 'RigidAffineForceOperator',
    'RigidHomotheticForceOperator', 'RankByMassOperator', 'AdditiveOperator',
    'InteriorOperator', 'SortAndAlignOperator', 'ProjectiveSelectionOperator',
    
    # Core-only (not yet migrated)
    'TopologicalFilterOperator', 'ExtremeFilterOperator', 'IsomorphismFilterOperator',
    'HullFillOperator', 'ConvexifyOperator', 'FillOperator', 'EnvelopeOperator',
    'ForceOperator', 'LinearForceOperator', 'SequenceOperator',
    'CropOperator', 'view_as_void'
]
