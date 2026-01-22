"""
Test spline basis functions and parameter mapping.
"""

import numpy as np
import pytest

from scpd.spline import SplineBasis, create_parameter_mapping, unpack_parameters


class TestSplineBasis:
    """Test SplineBasis class."""
    
    def test_basis_shape(self):
        """Basis matrix should have correct shape."""
        df = 6
        basis = SplineBasis(df=df)
        
        s = np.linspace(0, 1, 100)
        B = basis.evaluate(s)
        
        assert B.shape == (100, df), f"Expected shape (100, {df}), got {B.shape}"
    
    def test_basis_non_negative_somewhere(self):
        """Each basis function should be non-zero somewhere."""
        basis = SplineBasis(df=6)
        s = np.linspace(0, 1, 500)
        B = basis.evaluate(s)
        
        for j in range(6):
            assert np.any(np.abs(B[:, j]) > 1e-10), f"Basis function {j} is zero everywhere"
    
    def test_evaluate_function(self):
        """Test function evaluation with coefficients."""
        basis = SplineBasis(df=6)
        s = np.linspace(0, 1, 50)
        
        # Zero coefficients -> zero function
        beta = np.zeros(6)
        f = basis.evaluate_function(s, beta)
        assert np.allclose(f, 0), "Zero coefficients should give zero function"
        
        # Constant-ish via uniform coefficients
        beta = np.ones(6) * 2.0
        f = basis.evaluate_function(s, beta)
        assert np.all(np.isfinite(f)), "Function values should be finite"
    
    def test_log_transform(self):
        """Test exponential transformation for D(s)."""
        basis = SplineBasis(df=6)
        s = np.linspace(0, 1, 50)
        
        beta = np.zeros(6)
        f = basis.evaluate_function(s, beta, log_transform=True)
        assert np.allclose(f, 1.0), "exp(0) should be 1"
        
        beta = np.ones(6)
        f = basis.evaluate_function(s, beta, log_transform=True)
        assert np.all(f > 0), "Exponential should be positive"
    
    def test_second_derivative_matrix(self):
        """Second derivative matrix should have correct shape."""
        df = 6
        n_dense = 500
        basis = SplineBasis(df=df)
        
        B_dd, ds = basis.second_derivative_matrix(n_dense=n_dense)
        
        # Interior points only
        assert B_dd.shape == (n_dense - 2, df), f"Got shape {B_dd.shape}"
        assert ds > 0, "Grid spacing should be positive"
    
    def test_different_df(self):
        """Test various degrees of freedom."""
        for df in [3, 4, 6, 8, 10]:
            basis = SplineBasis(df=df)
            s = np.linspace(0, 1, 100)
            B = basis.evaluate(s)
            assert B.shape[1] == df, f"Expected {df} columns, got {B.shape[1]}"


class TestParameterMapping:
    """Test parameter mapping utilities."""
    
    def test_distribution_only_mode(self):
        """Distribution-only mode should have 2*df parameters."""
        basis = SplineBasis(df=6)
        s_grid = np.linspace(0, 1, 100)
        
        info = create_parameter_mapping(basis, s_grid, mode="distribution_only")
        
        assert info['n_params'] == 12, f"Expected 12 params, got {info['n_params']}"
        assert info['n_D'] == 6
        assert info['n_v'] == 6
        assert info['n_g'] == 0
    
    def test_with_population_mode(self):
        """With-population mode should have 3*df parameters."""
        basis = SplineBasis(df=6)
        s_grid = np.linspace(0, 1, 100)
        
        info = create_parameter_mapping(basis, s_grid, mode="with_population")
        
        assert info['n_params'] == 18, f"Expected 18 params, got {info['n_params']}"
        assert info['n_D'] == 6
        assert info['n_v'] == 6
        assert info['n_g'] == 6
    
    def test_unpack_parameters_distribution_only(self):
        """Test unpacking in distribution-only mode."""
        basis = SplineBasis(df=6)
        s_grid = np.linspace(0, 1, 50)
        
        info = create_parameter_mapping(basis, s_grid, mode="distribution_only")
        B = basis.evaluate(s_grid)
        
        theta = np.zeros(12)
        theta[:6] = 1.0   # log D
        theta[6:12] = 0.5 # v
        
        D, v, g = unpack_parameters(theta, info, B)
        
        assert len(D) == 50
        assert len(v) == 50
        assert len(g) == 50
        assert np.all(D > 0), "D should be positive (exponential)"
        assert np.allclose(g, 0), "g should be zero in distribution_only mode"
    
    def test_unpack_parameters_with_population(self):
        """Test unpacking in with-population mode."""
        basis = SplineBasis(df=6)
        s_grid = np.linspace(0, 1, 50)
        
        info = create_parameter_mapping(basis, s_grid, mode="with_population")
        B = basis.evaluate(s_grid)
        
        theta = np.zeros(18)
        theta[:6] = 0.0   # log D = 0 -> D = 1
        theta[6:12] = 0.3 # v
        theta[12:18] = 0.1 # g
        
        D, v, g = unpack_parameters(theta, info, B)
        
        assert len(D) == 50
        assert len(v) == 50
        assert len(g) == 50
        assert np.all(D > 0)
        assert not np.allclose(g, 0), "g should not be zero in with_population mode"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

