#!/usr/bin/env python3
"""
Demo: Automatic Transformation Derivation

This demo shows BioMatrix's core capability: given input and output point sets,
AUTOMATICALLY DERIVE the transformation T such that T(input) = output.

This is NOT Procrustes - it's algebraic law discovery:
- Detects translation, rotation, scaling
- Identifies subset operations (deletion, selection)
- Discovers lifting (tiling, symmetry)
- Handles multi-component unions

No neural network. Pure algebraic inference.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from biomatrix.core.state import State
from biomatrix.core.derive import derive_transformation
from biomatrix.core.transform import AffineTransform

np.random.seed(42)


def demo_translation():
    """Derive pure translation."""
    print("=" * 60)
    print("TEST 1: TRANSLATION DERIVATION")
    print("=" * 60)
    
    # Input: square
    pts_in = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    
    # Output: translated by [3, 2]
    pts_out = pts_in + np.array([3, 2])
    
    s_in = State(pts_in)
    s_out = State(pts_out)
    
    print(f"Input: {s_in.n_points} points at centroid {s_in.centroid}")
    print(f"Output: {s_out.n_points} points at centroid {s_out.centroid}")
    
    # Derive transformation
    T = derive_transformation(s_in, s_out)
    
    if T:
        print(f"Derived: {T}")
        result = T.apply(s_in)
        success = result == s_out
        print(f"Verification: T(input) == output? {success}")
        return success
    else:
        print("Failed to derive transformation")
        return False


def demo_rotation():
    """Derive rotation transformation."""
    print("\n" + "=" * 60)
    print("TEST 2: ROTATION DERIVATION")
    print("=" * 60)
    
    # Input: L-shape
    pts_in = np.array([
        [0, 0], [1, 0], [2, 0],
        [0, 1], [0, 2]
    ], dtype=float)
    
    # Output: rotated 90 degrees
    angle = np.pi / 2
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    centroid = np.mean(pts_in, axis=0)
    pts_out = (pts_in - centroid) @ R.T + centroid
    
    s_in = State(pts_in)
    s_out = State(pts_out)
    
    print(f"Input: L-shape, {s_in.n_points} points")
    print(f"Output: Rotated 90°")
    
    T = derive_transformation(s_in, s_out)
    
    if T:
        print(f"Derived: {T}")
        result = T.apply(s_in)
        success = result == s_out
        print(f"Verification: {success}")
        return success
    else:
        print("Failed")
        return False


def demo_subset():
    """Derive subset/selection operation."""
    print("\n" + "=" * 60)
    print("TEST 3: SUBSET DERIVATION (Selection)")
    print("=" * 60)
    
    # Input: points with different colors (last dim)
    pts_in = np.array([
        [0, 0, 1],  # color 1
        [1, 0, 1],  # color 1
        [2, 0, 2],  # color 2
        [0, 1, 2],  # color 2
        [1, 1, 1],  # color 1
    ], dtype=float)
    
    # Output: only color 1
    pts_out = np.array([
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
    ], dtype=float)
    
    s_in = State(pts_in)
    s_out = State(pts_out)
    
    print(f"Input: {s_in.n_points} points (colors 1 and 2)")
    print(f"Output: {s_out.n_points} points (only color 1)")
    
    T = derive_transformation(s_in, s_out)
    
    if T:
        print(f"Derived: {T}")
        result = T.apply(s_in)
        success = result == s_out
        print(f"Verification: {success}")
        return success
    else:
        print("Failed")
        return False


def demo_scaling():
    """Derive scaling transformation."""
    print("\n" + "=" * 60)
    print("TEST 4: SCALING DERIVATION")
    print("=" * 60)
    
    # Input: unit square
    pts_in = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    
    # Output: scaled 2x
    pts_out = pts_in * 2
    
    s_in = State(pts_in)
    s_out = State(pts_out)
    
    print(f"Input: unit square")
    print(f"Output: 2x scaled square")
    
    T = derive_transformation(s_in, s_out)
    
    if T:
        print(f"Derived: {T}")
        result = T.apply(s_in)
        success = result == s_out
        print(f"Verification: {success}")
        return success
    else:
        print("Failed")
        return False


def demo_reflection():
    """Derive reflection transformation."""
    print("\n" + "=" * 60)
    print("TEST 5: REFLECTION DERIVATION")
    print("=" * 60)
    
    # Input: asymmetric shape
    pts_in = np.array([
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1]
    ], dtype=float)
    
    # Output: reflected about x=1
    pts_out = pts_in.copy()
    pts_out[:, 0] = 2 - pts_out[:, 0]  # Reflect x around x=1
    
    s_in = State(pts_in)
    s_out = State(pts_out)
    
    print(f"Input: asymmetric shape")
    print(f"Output: reflected about x=1")
    
    T = derive_transformation(s_in, s_out)
    
    if T:
        print(f"Derived: {T}")
        result = T.apply(s_in)
        success = result == s_out
        print(f"Verification: {success}")
        return success
    else:
        print("Failed")
        return False


def main():
    print("=" * 60)
    print("BIOMATRIX: AUTOMATIC TRANSFORMATION DERIVATION")
    print("=" * 60)
    print("\nThis demonstrates algebraic law discovery:")
    print("Given (Input, Output) pairs, derive T such that T(Input) = Output\n")
    
    results = {
        'Translation': demo_translation(),
        'Rotation': demo_rotation(),
        'Subset': demo_subset(),
        'Scaling': demo_scaling(),
        'Reflection': demo_reflection(),
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print("""
BioMatrix derives transformations ALGEBRAICALLY:

1. NOT pattern matching - true mathematical inference
2. NOT training - works on first example
3. NOT hardcoded - discovers unknown transformations

The derive_transformation() function:
- Tries identity, then subset, then affine, then lifting
- Composes operators when needed
- Returns a reusable Operator that can be applied to new data

This is the scientific core that differentiates BioMatrix
from simple Procrustes wrappers.
""")


if __name__ == "__main__":
    main()
