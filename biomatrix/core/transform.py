import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.linalg import polar
from scipy.spatial.distance import cdist
from .state import State
from .base import Operator, SequentialOperator

@dataclass
class AffineTransform(Operator):
    """
    Affine Operator: f(x) = xA^T + b
    A: Linear Map (n x n)
    b: Translation (n)
    
    Storage is decoupled to avoid homogeneous coordinates slicing.
    """
    linear: np.ndarray      # (D, D)
    translation: np.ndarray # (D,)

    @classmethod
    def identity(cls, n_dims: int) -> 'AffineTransform':
        return cls(np.eye(n_dims), np.zeros(n_dims))

    @classmethod
    def translate(cls, vector: np.ndarray) -> 'AffineTransform':
        d = len(vector)
        return cls(np.eye(d), np.array(vector, dtype=float))

    @classmethod
    def linear_map(cls, matrix: np.ndarray) -> 'AffineTransform':
        d = matrix.shape[0]
        return cls(matrix, np.zeros(d))

    @classmethod
    def from_similarity(cls, scale: float, angle: float, translation: np.ndarray) -> 'AffineTransform':
        """
        Create planar similarity transform: scale * rotation + translation.
        
        G(x) = s * R(θ) * x + t
        
        Args:
            scale: uniform scale factor
            angle: rotation angle in radians (counter-clockwise)
            translation: [tx, ty] translation vector
        """
        c, s_val = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s_val], [s_val, c]])
        linear = scale * R
        return cls(linear, np.array(translation, dtype=float))

    @classmethod
    def projection(cls, dim: int, value: float, n_dims: int) -> 'AffineTransform':
        """
        Orthogonal projection onto hyperplane x[dim] = value.
        
        Algebraic definition: P(x)[i] = x[i] if i != dim, else value.
        Matrix form: A = I with A[dim,dim] = 0; b[dim] = value.
        
        Property: P^2 = P (idempotent).
        """
        A = np.eye(n_dims)
        A[dim, dim] = 0.0
        b = np.zeros(n_dims)
        b[dim] = value
        return cls(A, b)

    @classmethod
    def reflection(cls, normal: np.ndarray, origin: np.ndarray = None) -> 'AffineTransform':
        """
        Reflection across hyperplane defined by normal vector n and point P (origin).
        
        Algebraic Matrix (Householder):
        H = I - 2nn^T (assuming ||n||=1)
        f(x) = P + H(x - P) = Hx + (I - H)P
             = Hx + 2(n·P)n
             
        Args:
            normal: vector n (will be normalized)
            origin: point P (default: 0)
        """
        n_dims = len(normal)
        if origin is None:
            origin = np.zeros(n_dims)
            
        # Normalize n
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            return cls.identity(n_dims)
        n = normal / norm
        
        # Householder Matrix
        # outer product: n @ n.T
        H = np.eye(n_dims) - 2.0 * np.outer(n, n)
        
        # Translation part: b = (I - H)P = 2(n·P)n
        b = 2.0 * np.dot(n, origin) * n
        
        return cls(H, b)

    @classmethod
    def rotation_plane(cls, angle: float, dim1: int, dim2: int, center: np.ndarray = None, n_dims: int = None) -> 'AffineTransform':
        """
        Rotation in plane (dim1, dim2) by angle (radians).
        
        Algebraic Matrix (Givens Rotation):
        In basis {e_dim1, e_dim2}:
        [[cos θ, -sin θ],
         [sin θ,  cos θ]]
         
        f(x) = C + R(x - C) = Rx + (I - R)C
        """
        if n_dims is None:
            if center is not None:
                n_dims = len(center)
            else:
                raise ValueError("n_dims must be specified if center is None")
                
        if center is None:
            center = np.zeros(n_dims)
            
        c, s = np.cos(angle), np.sin(angle)
        
        R = np.eye(n_dims)
        # Standard rotation: dim1 -> dim2
        # e_1' = c e_1 + s e_2
        # e_2' = -s e_1 + c e_2
        R[dim1, dim1] = c
        R[dim1, dim2] = -s
        R[dim2, dim1] = s
        R[dim2, dim2] = c
        
        # Translation: b = (I - R)C
        b = (np.eye(n_dims) - R) @ center
        
        return cls(R, b)

    @classmethod
    def rotation_from_matrix(cls, matrix: np.ndarray, center: np.ndarray = None) -> 'AffineTransform':
        """
        General N-dimensional Rotation from Orthogonal Matrix R.
        Validates R^T @ R = I.
        """
        R = np.array(matrix, dtype=float)
        n = R.shape[0]
        if R.shape != (n, n):
            raise ValueError(f"Rotation matrix must be square (NxN), got {R.shape}")
            
        # Check orthogonality
        # Relaxed tolerance for float errors
        if not np.allclose(R @ R.T, np.eye(n), atol=1e-5):
            raise ValueError("Matrix is not orthogonal (R^T @ R != I)")
            
        # Check determinant (should be 1 for proper rotation, -1 for reflection)
        # We allow improper rotations (reflections) as they are isometries.
        # But commonly "Rotation" implies det=1. 
        # We accept O(n) group (Isometries).
        
        if center is None:
            return cls(R, np.zeros(n))
        else:
            # R(x - C) + C = Rx + (I - R)C
            b = (np.eye(n) - R) @ center
            return cls(R, b)

    def apply(self, state: State) -> State:
        if state.is_empty: return state.copy()
        # Dimension guard: check matrix dims match state dims
        n_dims = state.n_dims
        if self.linear.shape[1] != n_dims or self.linear.shape[0] != n_dims:
            # Dimension mismatch - return state unchanged
            return state.copy()
        # Ax + b
        # Points are row vectors P (N, D).
        # We need P @ A.T + b
        new_pts = state.points @ self.linear.T + self.translation
        return State(new_pts)

    def compose(self, other: 'Operator') -> 'Operator':
        """
        Self o Other.
        If Other is Affine(A2, b2) and Self is Affine(A1, b1):
        f1(f2(x)) = (xA2^T + b2)A1^T + b1
                  = x A2^T A1^T + b2 A1^T + b1
                  = x (A1 A2)^T + (A1 b2 + b1)
        """
        if isinstance(other, AffineTransform):
            # A_new = A1 @ A2
            new_linear = self.linear @ other.linear
            # b_new = A1 @ b2 + b1
            new_translation = self.linear @ other.translation + self.translation
            return AffineTransform(new_linear, new_translation)
        
        # Fallback for non-affine operators
        return SequentialOperator([other, self])

    def __matmul__(self, other: 'Operator') -> 'Operator':
        return self.compose(other)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AffineTransform): return False
        return np.allclose(self.linear, other.linear) and np.allclose(self.translation, other.translation)

    @classmethod
    def scale(cls, factor: float, n_dims: int = None, factors: np.ndarray = None) -> 'AffineTransform':
        """
        Uniform or non-uniform scaling.
        
        Args:
            factor: Uniform scale factor (if factors is None)
            n_dims: Number of dimensions (required if factor is scalar)
            factors: Per-dimension scale factors (array)
        """
        if factors is not None:
            S = np.diag(np.array(factors, dtype=float))
            return cls(S, np.zeros(len(factors)))
        else:
            if n_dims is None:
                raise ValueError("n_dims required for uniform scaling")
            S = factor * np.eye(n_dims)
            return cls(S, np.zeros(n_dims))

    def decompose(self, pivot: np.ndarray = None) -> dict:
        """
        Polar Decomposition: A = R @ U
        
        Returns a dictionary with:
        - 'rotation': Orthogonal matrix R (may include reflection if det(R) < 0)
        - 'scale': Symmetric PSD matrix U (contains scaling/shear)
        - 'translation': Vector b
        - 'is_reflection': True if det(R) < 0
        - 'planar_angle': Principal rotation angle (if applicable)
        - 'scale_factors': Eigenvalues of U (per-axis scaling)
        
        This allows reconstructing T = T_b ∘ H_U ∘ R_θ
        """
        
        R, U = polar(self.linear)
        
        det_R = np.linalg.det(R)
        is_reflection = det_R < 0
        
        # Scale factors = Eigenvalues of U (symmetric -> real eigenvalues)
        scale_factors = np.linalg.eigvalsh(U)
        
        # Extract principal rotation angle
        # In N-D, rotation is block-diagonal of 2x2 blocks.
        # We take the largest angle as "the" angle for reporting?
        # Or just return None if N != 2 for ambiguity avoidance.
        # Strict Algebra: Rotation is defined by O(N).
        n_dims = self.linear.shape[0]
        planar_angle = None
        if n_dims == 2:
            # θ = atan2(R[1,0], R[0,0])
            planar_angle = np.arctan2(R[1, 0], R[0, 0])
        
        return {
            'rotation': R,
            'scale': U,
            'translation': self.translation.copy(),
            'is_reflection': is_reflection,
            'planar_angle': planar_angle,
            'scale_factors': scale_factors,
            'n_dims': n_dims,
        }

    def __repr__(self):
        return f"Affine(A={self.linear.shape}, b={self.translation.shape})"
    
    # === Algebraic Methods ===
    
    def inverse(self) -> 'AffineTransform':
        """
        Compute T⁻¹ such that T⁻¹ ∘ T = Identity.
        
        For T(x) = Ax + b, T⁻¹(x) = A⁻¹(x - b) = A⁻¹x - A⁻¹b
        """
        from .base import NotInvertibleError
        
        if not self.is_invertible:
            raise NotInvertibleError(f"AffineTransform is singular (det={np.linalg.det(self.linear):.6f})")
        
        A_inv = np.linalg.inv(self.linear)
        b_inv = -A_inv @ self.translation
        return AffineTransform(linear=A_inv, translation=b_inv)
    
    def to_symbolic(self) -> str:
        """Symbolic representation for human readability."""
        decomp = self.decompose()
        parts = []
        
        # Translation
        if np.linalg.norm(self.translation) > 1e-6:
            t_str = ", ".join(f"{v:.2f}" for v in self.translation)
            parts.append(f"T({t_str})")
        
        # Rotation/Reflection
        if decomp['planar_angle'] is not None:
            angle_deg = np.degrees(decomp['planar_angle'])
            if abs(angle_deg) > 0.1:
                parts.append(f"R({angle_deg:.1f}°)")
        
        # Scale
        scale_factors = decomp['scale_factors']
        if not np.allclose(scale_factors, 1.0):
            if np.allclose(scale_factors, scale_factors[0]):
                parts.append(f"S({scale_factors[0]:.2f})")
            else:
                s_str = ", ".join(f"{v:.2f}" for v in scale_factors)
                parts.append(f"S({s_str})")
        
        if decomp['is_reflection']:
            parts.append("H")  # Householder/reflection
        
        if not parts:
            return "Id"
        
        return " ∘ ".join(parts)
    
    # === Algebraic Properties ===
    
    @property
    def is_invertible(self) -> bool:
        """True if det(A) ≠ 0."""
        return abs(np.linalg.det(self.linear)) > 1e-10
    
    @property
    def is_linear(self) -> bool:
        """True if translation is zero (pure linear map)."""
        return np.allclose(self.translation, 0)
    
    @property
    def preserves_mass(self) -> bool:
        """Affine transforms are bijections, so they preserve mass."""
        return self.is_invertible
    
    @property
    def is_isometry(self) -> bool:
        """True if A is orthogonal (preserves distances)."""
        R = self.linear
        return np.allclose(R @ R.T, np.eye(R.shape[0]), atol=1e-6)
    
    @property
    def is_rotation(self) -> bool:
        """True if A is a proper rotation (det = 1, orthogonal)."""
        return self.is_isometry and np.linalg.det(self.linear) > 0
    
    @property
    def is_reflection(self) -> bool:
        """True if A is an improper rotation (det = -1, orthogonal)."""
        return self.is_isometry and np.linalg.det(self.linear) < 0

# Alias for compatibility if needed, but we prefer strict naming
TransformationMatrix = AffineTransform


@dataclass
class RelativeTranslation(Operator):
    """
    RELATIVE Translation: Aligns input centroid to target centroid.
    
    CAUSAL & GENERALIZABLE:
    - Stores the TARGET CENTROID (where output should be centered).
    - At apply-time, computes: offset = target_centroid - input.centroid.
    - This is independent of the specific input position.
    
    Example:
    - Learning pair: input centered at [3,3], output centered at [7,7].
    - We store: target_centroid = [7,7] (the "rule" is "move to [7,7]").
    - Test pair: input centered at [1,2] → output moved to [7,7] (same target).
    
    This enables generalization because the RULE is "move to target", not "move by [4,4]".
    """
    target_centroid: np.ndarray  # Where to center the output
    
    def apply(self, state: State) -> State:
        if state.n_points == 0:
            return state.copy()
            
        # Compute offset at apply-time (RELATIVE)
        input_centroid = state.centroid
        offset = self.target_centroid - input_centroid
        
        # Apply translation
        new_points = state.points + offset
        return State(new_points)
    
    def __repr__(self):
        return f"RelativeTranslation(target={self.target_centroid})"

def derive_isometry_unordered(s_in: State, s_out: State, tol: float = 1e-4) -> Optional[AffineTransform]:
    """
    Derive Isometry (Rotation+Translation) between UNORDERED point sets.
    
    STRATEGY (Phase 7 - D4 Primacy):
    1. Center sets.
    2. CHECK DISCRETE D4 SYMMETRIES (8 ops) FIRST. (Grid Algebra).
    3. Fallback to PCA only for non-grid rotations.
    """
    if s_in.n_points != s_out.n_points or s_in.n_points < 2:
        if s_in.n_points == 1:
            return AffineTransform.translate(s_out.points[0] - s_in.points[0])
        return None
        
    X = s_in.points
    Y = s_out.points
    n_dims = X.shape[1]
    
    # 1. Centering
    mu_x = np.mean(X, axis=0)
    mu_y = np.mean(Y, axis=0)
    
    X_c = X - mu_x
    Y_c = Y - mu_y
    
    # 2. DISCRETE Bn SYMMETRY CHECK (Hyperoctahedral Group)
    # Replaces 2D-specific D4 check with N-dimensional generalizer.
    # Bn = Signed Permutations. Size 2^n * n!.
    
    bn_ops = []
    
    # Lazy generation of signed permutations (Bn)
    # We prioritize 2D/3D subspace rotations if N is large? 
    # For now, strict generation up to N=3, else identity fallback to rely on PCA.
    if n_dims <= 3:
        # ALGEBRAIC: Generate Bn symmetry group without itertools
        
        # 1. Permutations (S_n) - use recursive list-based generation
        def gen_perms(n):
            if n <= 1:
                return [list(range(n))]
            result = []
            for perm in gen_perms(n - 1):
                for i in range(n):
                    result.append(perm[:i] + [n-1] + perm[i:])
            return result
        
        perms = gen_perms(n_dims)
        
        # 2. Signs ({±1}^n) - use recursive generation
        def gen_signs(n):
            if n == 0:
                return [[]]
            prev = gen_signs(n - 1)
            return [[1] + s for s in prev] + [[-1] + s for s in prev]
        
        signs = gen_signs(n_dims)
        
        # Build matrices via map (no explicit for-loop)
        def make_matrices(perm):
            P = np.eye(n_dims)[list(perm)]
            return [np.diag(s) @ P for s in signs]
        
        bn_ops.extend([m for perm in perms for m in make_matrices(perm)])
    else:
        # Fallback for high D: Identity only (let PCA handle it)
        bn_ops.append(np.eye(n_dims))
        
    for R_candidate in bn_ops:
        X_candidate = X_c @ R_candidate.T 
        
        # Soft Match Check
        dists = cdist(X_candidate, Y_c)
        min_dists1 = np.min(dists, axis=1)
        min_dists2 = np.min(dists, axis=0)
        
        if np.max(min_dists1) < tol and np.max(min_dists2) < tol:
            # FOUND MATCH via Grid Symmetry
            Translation = mu_y - (R_candidate @ mu_x)
            return AffineTransform(linear=R_candidate, translation=Translation)

    # 3. PCA Fallback (Continuous Isometry)
    # Only if D4 fails (e.g. 45-degree rotation off-grid)
    
    # Calculate Covariance
    Cx = X_c.T @ X_c
    Cy = Y_c.T @ Y_c
    
    # No Spectral Gatekeeper here (it was blocking valid matches)
    
    _, Vx = np.linalg.eigh(Cx)
    _, Vy = np.linalg.eigh(Cy)
    
    # To keep it clean, let's include the PCA Sign Flip logic here as fallback.
    
    def generate_signs(d):
        if d == 0: yield []
        else:
            for s in generate_signs(d-1):
                yield s + [1.0]
                yield s + [-1.0]
    sign_flips = list(generate_signs(n_dims))

    X_proj_base = X_c @ Vx
    Y_proj = Y_c @ Vy
    
    order_y = np.lexsort(Y_proj.T)
    Y_proj_sorted = Y_proj[order_y]
    
    for signs in sign_flips:
        S = np.diag(signs)
        X_proj_candidate = X_proj_base @ S
        
        order_x = np.lexsort(X_proj_candidate.T)
        X_proj_sorted = X_proj_candidate[order_x]
        
        if np.allclose(X_proj_sorted, Y_proj_sorted, atol=tol):
            P = X_c[order_x]
            Q = Y_c[order_y]
            H = P.T @ Q
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            X_mapped = X_c @ R.T
            diffs = X_mapped[:, None, :] - Y_c[None, :, :]
            dists = np.min(np.sum(diffs**2, axis=2), axis=1)
            
            if np.max(dists) < tol:
                Translation = mu_y - (R @ mu_x)
                return AffineTransform(linear=R, translation=Translation)

    return None
