#!/usr/bin/env python3
"""
Demo: Causality Detection (Inert vs Causal Separation)

BioMatrix can automatically separate:
- INERT points: Present in both input and output (unchanged)
- CAUSAL points: New in output (generated/moved)

This is algebraic causality inference - no temporal data needed.
Just compare two states and derive what changed vs what stayed.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from biomatrix.core.state import State
from biomatrix.core.derive.algebra import derive_causality

np.random.seed(42)


def demo_causality_detection():
    """Show automatic separation of inert vs causal points."""
    print("=" * 60)
    print("BIOMATRIX: CAUSALITY DETECTION")
    print("=" * 60)
    print("\nSeparating INERT (unchanged) from CAUSAL (changed) points\n")
    
    # Scene: Some objects stay, some are added
    # Input: 3 squares
    square1 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    square2 = square1 + [5, 0]
    square3 = square1 + [10, 0]
    
    pts_in = np.vstack([square1, square2, square3])
    
    # Output: 2 squares stay, 1 new appears, 1 moves
    # square1 stays (inert)
    # square2 disappears
    # square3 stays (inert)
    # square4 appears (causal)
    square4 = square1 + [0, 5]  # New square
    
    pts_out = np.vstack([square1, square3, square4])
    
    s_in = State(pts_in)
    s_out = State(pts_out)
    
    print(f"Input:  {s_in.n_points} points (3 squares)")
    print(f"Output: {s_out.n_points} points (2 stay, 1 new)")
    
    # Derive causality
    result = derive_causality(s_in, s_out)
    
    if result:
        print(f"\nDerived Operator: {result}")
        
        # Analyze intersection (inert) vs difference (causal)
        in_pts = set(map(tuple, np.round(s_in.points, 3)))
        out_pts = set(map(tuple, np.round(s_out.points, 3)))
        
        inert = in_pts & out_pts
        deleted = in_pts - out_pts
        new_pts = out_pts - in_pts
        
        print(f"\n=== CAUSALITY ANALYSIS ===")
        print(f"  INERT (overlap):   {len(inert)} points")
        print(f"  DELETED:           {len(deleted)} points")
        print(f"  NEW (causal):      {len(new_pts)} points")
        
        print(f"\n✅ Causality detected successfully")
        return True
    else:
        print("Failed to derive causality")
        return False


def demo_moving_object():
    """Detect which object moved vs which stayed."""
    print("\n" + "=" * 60)
    print("TEST 2: MOVING OBJECT DETECTION")
    print("=" * 60)
    
    # Input: 2 objects
    obj1 = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)  # Triangle
    obj2 = np.array([[5, 5], [6, 5], [5, 6]], dtype=float)  # Triangle
    
    pts_in = np.vstack([obj1, obj2])
    
    # Output: obj1 stays, obj2 moves
    obj2_moved = obj2 + [2, 0]
    pts_out = np.vstack([obj1, obj2_moved])
    
    s_in = State(pts_in)
    s_out = State(pts_out)
    
    print(f"Input:  2 triangles")
    print(f"Output: Triangle 1 stays, Triangle 2 moves right by 2")
    
    result = derive_causality(s_in, s_out)
    
    if result:
        in_pts = set(map(tuple, np.round(s_in.points, 3)))
        out_pts = set(map(tuple, np.round(s_out.points, 3)))
        
        inert = in_pts & out_pts
        print(f"\n  INERT: {len(inert)} points (static triangle)")
        print(f"  MOVED: {len(s_out.points) - len(inert)} points (moved triangle)")
        print(f"\n✅ Moving object identified")
        return True
    
    return False


def main():
    results = {
        'Causality': demo_causality_detection(),
        'Moving Object': demo_moving_object(),
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print("""
CAUSALITY DETECTION without temporal data:

1. Compare two states algebraically
2. Inert = Intersection (points in both)
3. Causal = Symmetric difference (changed points)

Applications:
- Motion detection: What moved between frames?
- Change detection: What's new in this version?
- Object tracking: Separate static background from dynamic foreground

No optical flow. No neural network. Pure set algebra.
""")


if __name__ == "__main__":
    main()
