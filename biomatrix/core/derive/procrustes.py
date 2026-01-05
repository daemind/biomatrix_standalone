"""
Procrustes SE(3) Prior - Fast N-dimensional Orthogonal Procrustes

Derives Y = scale * R @ X + t using SVD.
N-dimensional agnostic, works for any D >= 1.
"""

import numpy as np
from scipy.linalg import svd
from typing import Optional

from ..state import State
from ..operators.base import Operator


class ProcrustesOperator(Operator):
    """
    N-dimensional affine transformation: Y = A @ X + t
    where A = scale * R (orthogonal, possibly with uniform scale)
    """
    
    def __init__(self, A: np.ndarray, t: np.ndarray):
        """
        Args:
            A: DxD matrix (scale * rotation)
            t: D-vector translation
        """
        self.A = np.asarray(A)
        self.t = np.asarray(t)
        self.D = len(t)
    
    def apply(self, state: State) -> State:
        """Apply transformation to state."""
        pts = state.points
        transformed = (self.A @ pts.T).T + self.t
        return State(transformed)
    
    def __repr__(self):
        return f"ProcrustesOperator(D={self.D})"


def derive_procrustes_se3(
    state_in: State,
    state_out: State,
    allow_scale: bool = True,
    tol: float = 1e-6
) -> Optional[ProcrustesOperator]:
    """
    Derive orthogonal Procrustes transformation: Y = scale * R @ X + t
    
    N-dimensional agnostic algorithm:
    1. Center both point sets
    2. Compute scale from Frobenius norms
    3. SVD for optimal rotation
    4. Compute absolute translation
    
    Args:
        state_in: Input state (N points x D dims)
        state_out: Output state (N points x D dims)
        allow_scale: If True, estimate uniform scale
        tol: Numerical tolerance
        
    Returns:
        ProcrustesOperator or None if derivation fails
    """
    X = state_in.points
    Y = state_out.points
    
    # Validate dimensions
    if X.shape != Y.shape:
        return None
    
    N, D = X.shape
    
    if N == 0:
        return None
    
    # Special case: 1 point = pure translation
    if N == 1:
        A = np.eye(D)
        t = Y[0] - X[0]
        return ProcrustesOperator(A, t)
    
    # Center both sets
    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)
    X_c = X - mu_X
    Y_c = Y - mu_Y
    
    # Scale from Frobenius norms
    norm_X = np.linalg.norm(X_c, 'fro')
    norm_Y = np.linalg.norm(Y_c, 'fro')
    
    if norm_X < tol:
        return None
    
    scale = (norm_Y / norm_X) if allow_scale else 1.0
    
    # Cross-covariance matrix
    H = X_c.T @ Y_c  # DxD
    
    # SVD: H = U @ S @ Vt
    U, S, Vt = svd(H)
    
    # Optimal rotation: R = V @ U.T
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det = +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Full transformation matrix
    A = scale * R
    
    # Absolute translation: mu_Y = A @ mu_X + t
    t = mu_Y - A @ mu_X
    
    # RESIDUAL VALIDATION: Reject if Procrustes is a poor fit
    # This prevents overriding non-linear solvers like LiftedTransform
    Y_pred = (A @ X.T).T + t
    residuals = np.linalg.norm(Y_pred - Y, axis=1)
    mean_residual = np.mean(residuals)
    
    # Threshold: reject if mean error > 5% of data spread
    data_spread = np.max([norm_X, norm_Y, 1.0])
    relative_error = mean_residual / data_spread
    
    if relative_error > 0.05:
        return None  # Poor fit - let other solvers try
    
    return ProcrustesOperator(A, t)
