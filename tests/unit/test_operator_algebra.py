# -*- coding: utf-8 -*-
"""
tests/unit/test_operator_algebra.py

TDD Tests for Operator Algebra: inverse, simplify, to_symbolic.
"""
import pytest
import numpy as np
from biomatrix.core.transform import AffineTransform
from biomatrix.core.operators.affine import GlobalAffineOperator
from biomatrix.core.operators.base import IdentityOperator
from biomatrix.core.base import SequentialOperator, NotInvertibleError
from biomatrix.core.state import State


class TestAffineInverse:
    """Test inverse() for AffineTransform."""
    
    def test_translation_inverse(self):
        """T⁻¹ ∘ T = Id for translation."""
        T = AffineTransform.translate(np.array([3.0, 5.0]))
        T_inv = T.inverse()
        
        # Compose: T_inv ∘ T should be identity
        composed = T_inv @ T
        
        s = State(np.array([[1.0, 2.0], [4.0, 6.0]]))
        result = composed.apply(s)
        
        np.testing.assert_allclose(result.points, s.points, atol=1e-10)
    
    def test_rotation_inverse(self):
        """R⁻¹ ∘ R = Id for rotation."""
        R = AffineTransform.rotation_plane(
            angle=np.pi/4, dim1=0, dim2=1, 
            center=np.array([0.0, 0.0]), n_dims=2
        )
        R_inv = R.inverse()
        
        composed = R_inv @ R
        
        s = State(np.array([[1.0, 0.0], [0.0, 1.0]]))
        result = composed.apply(s)
        
        np.testing.assert_allclose(result.points, s.points, atol=1e-10)
    
    def test_scale_inverse(self):
        """S⁻¹ ∘ S = Id for scaling."""
        S = AffineTransform.scale(2.0, n_dims=2)
        S_inv = S.inverse()
        
        composed = S_inv @ S
        
        s = State(np.array([[1.0, 2.0], [3.0, 4.0]]))
        result = composed.apply(s)
        
        np.testing.assert_allclose(result.points, s.points, atol=1e-10)
    
    def test_singular_not_invertible(self):
        """Singular matrix raises NotInvertibleError."""
        singular_matrix = np.array([[1.0, 0.0], [0.0, 0.0]])
        T = AffineTransform(linear=singular_matrix, translation=np.zeros(2))
        
        assert not T.is_invertible
        
        with pytest.raises(NotInvertibleError):
            T.inverse()


class TestAffineProperties:
    """Test algebraic properties of AffineTransform."""
    
    def test_identity_is_identity(self):
        """Identity should be identity."""
        I = AffineTransform.identity(2)
        assert I.is_linear
        assert I.is_invertible
        assert I.is_isometry
        assert I.is_rotation
        assert not I.is_reflection
    
    def test_rotation_is_isometry(self):
        """Rotation is isometry with det=1."""
        R = AffineTransform.rotation_plane(np.pi/3, 0, 1, n_dims=2)
        assert R.is_isometry
        assert R.is_rotation
        assert not R.is_reflection
    
    def test_reflection_is_reflection(self):
        """Reflection is isometry with det=-1."""
        H = AffineTransform.reflection(np.array([1.0, 0.0]))
        assert H.is_isometry
        assert H.is_reflection
        assert not H.is_rotation
    
    def test_translation_not_linear(self):
        """Translation is not a linear map."""
        T = AffineTransform.translate(np.array([1.0, 2.0]))
        assert not T.is_linear
        assert T.is_invertible


class TestSymbolicRepresentation:
    """Test to_symbolic() method."""
    
    def test_identity_symbolic(self):
        """Identity returns 'Id'."""
        I = AffineTransform.identity(2)
        assert I.to_symbolic() == "Id"
    
    def test_translation_symbolic(self):
        """Translation shows T(...)."""
        T = AffineTransform.translate(np.array([3.0, 5.0]))
        sym = T.to_symbolic()
        assert "T(" in sym
    
    def test_rotation_symbolic(self):
        """Rotation shows R(angle)."""
        R = AffineTransform.rotation_plane(np.pi/4, 0, 1, n_dims=2)
        sym = R.to_symbolic()
        assert "R(" in sym
        assert "45" in sym  # 45 degrees


class TestSequentialOperatorAlgebra:
    """Test SequentialOperator algebraic methods."""
    
    def test_sequence_inverse(self):
        """(T1 ∘ T2)⁻¹ = T2⁻¹ ∘ T1⁻¹."""
        T1 = AffineTransform.translate(np.array([1.0, 0.0]))
        T2 = AffineTransform.translate(np.array([0.0, 2.0]))
        
        seq = T1 @ T2
        seq_inv = seq.inverse()
        
        composed = seq_inv @ seq
        
        s = State(np.array([[0.0, 0.0], [1.0, 1.0]]))
        result = composed.apply(s)
        
        np.testing.assert_allclose(result.points, s.points, atol=1e-10)
    
    def test_simplify_removes_identity(self):
        """Simplify removes identity operators."""
        T = AffineTransform.translate(np.array([1.0, 2.0]))
        I = IdentityOperator()
        
        seq = SequentialOperator(steps=[T, I])
        simplified = seq.simplify()
        
        # Should reduce to just T
        assert not isinstance(simplified, SequentialOperator)
    
    def test_sequence_to_symbolic(self):
        """Sequential gives composed symbolic representation."""
        T = AffineTransform.translate(np.array([1.0, 2.0]))
        R = AffineTransform.rotation_plane(np.pi/2, 0, 1, n_dims=2)
        
        seq = T @ R
        sym = seq.to_symbolic()
        
        assert "∘" in sym


class TestGlobalAffineOperator:
    """Test GlobalAffineOperator algebraic methods."""
    
    def test_inverse(self):
        """GlobalAffineOperator inverse works."""
        M = np.array([[2.0, 0.0], [0.0, 3.0]])
        b = np.array([1.0, 2.0])
        
        op = GlobalAffineOperator(matrix=M, bias=b)
        op_inv = op.inverse()
        
        s = State(np.array([[1.0, 1.0], [2.0, 2.0]]))
        
        result = op_inv.apply(op.apply(s))
        np.testing.assert_allclose(result.points, s.points, atol=1e-10)
    
    def test_to_symbolic(self):
        """to_symbolic returns readable string."""
        M = np.array([[2.0, 0.0], [0.0, 2.0]])
        b = np.array([1.0, 1.0])
        
        op = GlobalAffineOperator(matrix=M, bias=b)
        sym = op.to_symbolic()
        
        assert "A(" in sym or "T(" in sym
