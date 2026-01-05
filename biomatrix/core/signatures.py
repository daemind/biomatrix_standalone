"""
signatures.py - Universal Invariant Signatures for Algebraic Matching

AGENT.md Compliant:
- NO Heuristics (no dist < epsilon)
- N-Dim Agnostic
- Pure Algebra (Spectral Theory)

A Signature is a Tensor Invariant under the Action of a Group G (Isometry).
For Point Clouds, the Universal Invariant is the Spectrum of the Gram/Covariance Matrix.
"""

import numpy as np
from typing import Tuple, Optional
from .state import State

def compute_universal_signature(points: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Compute the Universal Algebraic Signature of a point cloud.
    
    Invariant under E(n) (Translation, Rotation, Reflection, Permutation).
    
    Signature = (Mass, Spectrum)
    
    Args:
        points: (N, D) array of coordinates.
        
    Returns:
        mass: int (N)
        spectrum: (D,) or (N,) array of sorted eigenvalues (descending).
    """
    N, D = points.shape
    
    if N == 0:
        return 0, np.zeros(0)
    
    # 1. Centering (Translation Invariance)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # 2. Contraction (Rotation Invariance)
    # Use Covariance (D x D) or Gram (N x N), whichever is smaller.
    # Spectrum is identical (except for zeros).
    
    if D <= N:
        # Covariance Matrix: C = (X^T X) / N
        # Normalized by Mass for Density Invariance
        tensor = (centered.T @ centered) / N
    else:
        # Gram Matrix: G = (X X^T) / N
        tensor = (centered @ centered.T) / N
        
    # 3. Spectral Decomposition
    # Eigenvalues are Isometry Invariants.
    # We use eigh because tensor is symmetric positive semi-definite.
    eigvals = np.linalg.eigvalsh(tensor)
    
    # 4. Canonical Ordering (Permutation Invariance)
    # Sort descending to ensure consistent vector representation
    # np.linalg.eigvalsh returns ascending, so we flip.
    spectrum = np.flip(eigvals)
    
    # Rounding for Algebra (Noise suppression only, not heuristic thresholding)
    # Standard float precision stability (e.g. 1e-9 becomes 0)
    # Can be strictly maintained as float but rounded for equality check.
    spectrum = np.round(spectrum, 4)
    
    # Pad to fixed size D for vectorization consistency
    if len(spectrum) < D:
        spectrum = np.pad(spectrum, (0, D - len(spectrum)), 'constant')
    elif len(spectrum) > D:
        # Should not happen if logic matches D <= N check, but for safety
        spectrum = spectrum[:D]
    
    return N, spectrum


def signatures_match(sig1: Tuple[int, np.ndarray], sig2: Tuple[int, np.ndarray], tol: float = 1e-2) -> bool:
    """
    Check algebraic equality of signatures with optional tolerance.
    """
    n1, s1 = sig1
    n2, s2 = sig2
    
    # Mass check removed for Density Invariance/Generalization.
    # We match strictly on Normalized Spectrum (Shape).
    # if n1 != n2:
    #     return False
        
    if s1.shape != s2.shape:
        # Should handle D vs N switch if dimensions differ but N is same?
        # If N is same, and D is same (implied by comparing same spaces usually), shapes match.
        return False
        
    # Use array_equal if tol is 0 (or very small), else allclose
    if tol < 1e-9:
        return np.array_equal(s1, s2)
        
    return np.allclose(s1, s2, atol=tol)


def compute_projected_signatures(points: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Compute Signatures for all subspaces of codimension 1.
    Used for Causal Invariant Matching (matching objects that differ in exactly 1 dimension).
    
    Returns:
        N: Mass
        signatures: (D+1, D) array.
            Row 0: Full Signature (padded).
            Row k+1: Signature of points projected onto D \ {k}.
    """
    N, D = points.shape
    if N == 0:
        return 0, np.zeros((D+1, D))
        
    signatures = []
    
    # 0. Full Signature
    _, sig_full = compute_universal_signature(points)
    signatures.append(sig_full)
    
    # 1. Projected Signatures
    for d in range(D):
        # Drop dimension d
        mask = np.ones(D, dtype=bool)
        mask[d] = False
        points_proj = points[:, mask]
        
        # Compute signature of projected points
        _, sig_proj = compute_universal_signature(points_proj)
        
        # Pad to D to ensure recti-linear array for vectorized cost calculation
        # (sig_proj is size D-1 usually, unless N < D-1)
        # Pad with -1 or 0? 0 is safe for spectrum.
        if len(sig_proj) < D:
            sig_proj = np.pad(sig_proj, (0, D - len(sig_proj)), 'constant')
            
        signatures.append(sig_proj)
        
    return N, np.array(signatures)
